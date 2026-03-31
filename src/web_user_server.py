#!/usr/bin/env python3.9

import os
from flask import Flask, render_template, request, jsonify, session, redirect, url_for, Response, send_from_directory, send_file,make_response
from user_management import UserManager
from story_generator import StoryGenerator
from tts_helper import TTSHelper
from image_generator import ImageGenerator
from flask_cors import CORS
import json
import re
import time
import sys
import random
import uuid
from typing import Optional
import difflib
import subprocess
import subprocess
try:
    import cv2
except Exception:
    cv2 = None
try:
    # Optional LLM client used for ASR intent correction
    from llamaindex_interface import ChatWithRAG
    LLM_AVAILABLE = True
except Exception:
    LLM_AVAILABLE = False

# ROS camera via topic (preferred on robot)
try:
    import rospy
    from sensor_msgs.msg import Image
    from cv_bridge import CvBridge, CvBridgeError
    from queue import Queue
    from threading import Thread
    ROS_AVAILABLE = True
except Exception:
    ROS_AVAILABLE = False

from threading import Lock, Thread
from threading import Event as ThreadEvent

# Human tracking (kinematics + presence detection)
try:
    from human_tracking import HumanTracking
    HUMAN_TRACKING_AVAILABLE = True
except Exception:
    HUMAN_TRACKING_AVAILABLE = False

# # Idle attention (uses human tracker or random gaze)
# try:
#     from idle_attention import IdleAttention
#     IDLE_ATTENTION_AVAILABLE = True
# except Exception:
#     IDLE_ATTENTION_AVAILABLE = False

# Lazy ROS camera subscriber
_ros_cam = None
_ros_cam_lock = Lock()

# HumanTracking singleton
_human_tracker = None
_human_tracker_lock = Lock()

# V4L2 camera handle (non-ROS fallback)
_v4l_cap = None
_v4l_cap_lock = Lock()


#loading env variables
from dotenv import load_dotenv
load_dotenv()

def _ensure_human_tracker():
    global _human_tracker
    if not HUMAN_TRACKING_AVAILABLE:
        return None
    with _human_tracker_lock:
        if _human_tracker is None:
            try:
                _human_tracker = HumanTracking()
            except Exception as e:
                print(f"HumanTracking init failed: {e}")
                return None
    return _human_tracker

def _pick_recent_person(tracker, timeout_sec: float = 0.5):
    """Pick the most recent person with a face in view within timeout."""
    if not tracker:
        return None
    import time as _t
    deadline = _t.time() + max(0.0, timeout_sec)
    picked = None
    while _t.time() < deadline:
        try:
            now = _t.time()
            best = None
            best_ts = 0.0
            tracker.persons_lock.acquire()
            try:
                for _pid, pdata in tracker.persons.items():
                    ts = pdata.get('last_seen') or 0.0
                    if pdata.get('face') and now - ts < tracker.PRESENCE_TIME_THRESHOLD:
                        if ts > best_ts:
                            best_ts = ts
                            best = pdata
            finally:
                tracker.persons_lock.release()
            if best is not None:
                picked = best
                break
        except Exception:
            pass
        _t.sleep(0.05)
    return picked


class CameraCapture:
    def __init__(self, topic="/camera/color/image_raw"):
        self.image_queue = Queue(maxsize=1)
        self.bridge = CvBridge()
        # Initialize ROS node only if not already initialized by other components (e.g., TTSHelper)
        try:
            if ROS_AVAILABLE and not rospy.core.is_initialized():
                rospy.init_node("web_camera_bridge", anonymous=True, disable_signals=True)
        except Exception:
            pass
        self.image_sub = rospy.Subscriber(topic, Image, self._image_callback)

    def _image_callback(self, data):
        try:
            image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            try:
                self.image_queue.get_nowait()
            except:
                pass
            self.image_queue.put_nowait(image)
        except CvBridgeError as e:
            rospy.logerr(str(e))

    def get_latest_image(self):
        if not self.image_queue.empty():
            return self.image_queue.get()
        return None

def _get_ros_frame():
    global _ros_cam
    if not ROS_AVAILABLE:
        return None
    with _ros_cam_lock:
        if _ros_cam is None:
            topic = os.environ.get('CAMERA_ROS_TOPIC', "/camera/color/image_raw")
            _ros_cam = CameraCapture(topic=topic)
    # Fetch latest image with a brief wait for first frame
    wait_seconds = 1.5
    try:
        wait_seconds = float(os.environ.get('CAMERA_FRAME_WAIT_SEC', '1.5'))
    except Exception:
        wait_seconds = 1.5
    deadline = time.time() + max(0.0, wait_seconds)
    frame = _ros_cam.get_latest_image()
    while frame is None and time.time() < deadline:
        time.sleep(0.05)
        frame = _ros_cam.get_latest_image()
    return frame



app = Flask(__name__, template_folder="../templates")
app.secret_key = os.urandom(24)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
USER_DATA_DIR = os.path.join(BASE_DIR, 'user_data')
WORKER_PYTHON = os.getenv(
    "IMAGE_WORKER_PYTHON",
    os.path.join(os.path.dirname(BASE_DIR), ".venv39", "bin", "python"),
)

user_manager = UserManager(
    users_file=os.path.join(BASE_DIR, 'users.json'),
    base_dir=USER_DATA_DIR
)
story_generator = StoryGenerator(llm_model="gemini-2.5-flash")
tts_helper = TTSHelper()
image_generator = ImageGenerator()

# Gemini analysis is delegated to external script (Python 3.9) when requested

# De-duplicate short-interval wait announcements
_last_wait_announce_ts = 0.0
def _announce_wait_once(cooldown_seconds: float = 2.0):
    global _last_wait_announce_ts
    now = time.time()
    if now - _last_wait_announce_ts >= cooldown_seconds:
        try:
            _with_asr_suspended(lambda: tts_helper.speak('I am waiting.'))
        except Exception:
            pass
        _last_wait_announce_ts = now

def _with_asr_suspended(say_callable):
    """Disable ASR audio stream while robot is speaking, then restore."""
    # Ensure human tracking and idle attention are active during TTS
    try:
        tracker = _ensure_human_tracker()
        if tracker and not getattr(tracker, 'should_track', False):
            person = _pick_recent_person(tracker, timeout_sec=0.5)
            tracker.track(person)
    except Exception:
        pass
    # NOTE: Riva ASR disabled (switched to Whisper).
    # try:
    #     from riva_speech_recognition import RivaSpeechRecognition
    #     RivaSpeechRecognition.set_audio_enabled(False)
    # except Exception:
    #     pass
    try:
        return say_callable()
    finally:
        # NOTE: Riva ASR disabled (switched to Whisper).
        # try:
        #     from riva_speech_recognition import RivaSpeechRecognition
        #     RivaSpeechRecognition.set_audio_enabled(True)
        # except Exception:
        #     pass
        # try:
        #     _stop_idle_attention()
        # except Exception:
        #     pass
        pass

# --- Camera REST endpoints ---

# Lightweight LLM-based ASR intent correction
_intent_llm = None
_intent_llm_lock = Lock()
_quiz_llm = None
_quiz_llm_lock = Lock()

def _ensure_intent_llm():
    global _intent_llm
    if not LLM_AVAILABLE:
        return
    with _intent_llm_lock:
        if _intent_llm is None:
            try:
                _intent_llm = ChatWithRAG(
                    model="llama3.1",
                    system_role=(
                        "You correct ASR mishearings for a child's therapy robot. "
                        "Decide if the transcript likely intended the target word(s) given the immediate context. "
                        "Be conservative; only match when highly likely. Respond strictly in compact JSON: "
                        "{\"match\": true|false, \"canonical\": \"<canonical or expected>\"}."
                    ),
                    disable_rag=True,
                    max_tokens=128
                )
            except Exception as e:
                print(f"Warning: failed to initialize intent LLM: {e}")

def _ensure_quiz_llm():
    global _quiz_llm
    if not LLM_AVAILABLE:
        return
    with _quiz_llm_lock:
        if _quiz_llm is None:
            try:
                _quiz_llm = ChatWithRAG(
                    model="phi4:14b",
                    system_role=(
                        "You create short, child-friendly quiz questions. "
                        "Return JSON only. Each item must be {\"question\": \"...\", \"type\": \"yes_no\"|\"wh\"}."
                    ),
                    disable_rag=True,
                    max_tokens=512
                )
            except Exception as e:
                print(f"Warning: failed to initialize quiz LLM: {e}")

def _llm_canonicalize_heard(expected: str, heard: str, context: Optional[str] = None) -> Optional[str]:
    try:
        if not expected or not heard:
            return None
        # Quick fuzzy check: if very close, accept directly
        ratio = difflib.SequenceMatcher(None, expected.lower(), heard.lower()).ratio()
        if ratio >= 0.85:
            return expected
        # Use LLM only if available
        _ensure_intent_llm()
        if _intent_llm is None:
            return None
        ctx = context or ""
        prompt = (
            "Expected: '" + expected + "'\n"
            "Heard: '" + heard + "'\n"
            "Context: '" + ctx + "'\n"
            "Answer in JSON only with keys match (true/false) and canonical."
        )
        resp = _intent_llm.get_response(prompt)
        text = getattr(resp, 'message', None)
        text = getattr(text, 'content', None) if text is not None else str(resp)
        raw = (text or '').strip()
        # Strip code fences if any
        if raw.startswith('```'):
            raw = raw.strip('`')
            # Try to find JSON braces
        # Extract JSON object
        import json as _json
        obj = None
        try:
            obj = _json.loads(raw)
        except Exception:
            # Try to find first {...}
            l = raw.find('{')
            r = raw.rfind('}')
            if l != -1 and r != -1 and r > l:
                try:
                    obj = _json.loads(raw[l:r+1])
                except Exception:
                    obj = None
        if not isinstance(obj, dict):
            return None
        match = obj.get('match') is True
        canonical = obj.get('canonical') or expected
        if match:
            return canonical
        return None
    except Exception as e:
        print(f"LLM correction error: {e}")
        return None

def _edit_distance_limited(a: str, b: str, max_distance: int = 1) -> int:
    """Compute Levenshtein distance with early exit if distance exceeds max_distance."""
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if abs(la - lb) > max_distance:
        return max_distance + 1
    # Initialize previous row
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        curr = [i] + [0] * lb
        min_in_row = curr[0]
        ai = a[i - 1]
        for j in range(1, lb + 1):
            cost = 0 if ai == b[j - 1] else 1
            curr[j] = min(prev[j] + 1,      # deletion
                          curr[j - 1] + 1,  # insertion
                          prev[j - 1] + cost)  # substitution
            if curr[j] < min_in_row:
                min_in_row = curr[j]
        if min_in_row > max_distance:
            return max_distance + 1
        prev = curr
    return prev[-1]

def _fuzzy_canonicalize_heard(expected: str, heard: str) -> Optional[str]:
    """Return expected when a close fuzzy match is detected in heard tokens; else None."""
    try:
        exp = (expected or '').lower().strip()
        hr = (heard or '').lower().strip()
        if not exp or not hr:
            return None
        # Quick path: substring match
        if exp in hr:
            return exp
        import re as _re
        tokens = _re.findall(r"[a-z0-9]+", hr)
        for tok in tokens:
            if tok == exp:
                return exp
            # Accept small edit distances or high similarity ratio
            if _edit_distance_limited(exp, tok, max_distance=1) <= 1:
                return exp
            if difflib.SequenceMatcher(None, exp, tok).ratio() >= 0.83:
                return exp
        return None
    except Exception:
        return None

# Activity runner state
_activity_stop_event = ThreadEvent()
_activity_thread = None
_asr_enabled = True

# Step-by-step confirmation state (for run_saved with therapist confirmation)
_step_confirm_event = ThreadEvent()
_step_current_index = -1       # which step is waiting for confirmation (-1 = not running)
_step_total_count = 0
_step_current_label = ""
_step_next_label = ""
_step_waiting = False           # True when paused waiting for therapist

def _generate_recovery_question(mode, username=''):
    """Capture a camera frame and generate a recovery question via Gemini.
    Returns dict with 'text' and 'object' (or None on failure)."""
    try:
        frame = _get_ros_frame()
        if frame is None:
            print("[Recovery] Camera frame failed")
            return None
        import datetime
        cap_dir = os.path.join(USER_DATA_DIR, username or '_tmp', 'captured_scenes')
        os.makedirs(cap_dir, exist_ok=True)
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        fpath = os.path.join(cap_dir, f'recovery_{mode}_{ts}.jpg')
        cv2.imwrite(fpath, frame)

        script_path = os.path.join(os.path.dirname(BASE_DIR), 'scripts', 'gemini_recovery_question.py')
        if not os.path.exists(script_path):
            print("[Recovery] Script not found")
            return None

        # Resolve child name and age from user profile
        child_name = ''
        child_age = 5
        if username:
            u = user_manager.users.get(username, {})
            child_name = u.get('display_name') or username
            try:
                child_age = int(u.get('age', 5))
            except (ValueError, TypeError):
                child_age = 5

        cmd = [WORKER_PYTHON, script_path, '--image', fpath, '--mode', mode,
               '--child-age', str(child_age)]
        if child_name:
            cmd += ['--child-name', child_name]

        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if proc.returncode != 0:
            print(f"[Recovery] Script error: {proc.stderr}")
            return None
        raw = (proc.stdout or '').strip()
        try:
            result = json.loads(raw)
            return {'text': result.get('text', ''), 'object': result.get('object', None)}
        except Exception:
            return {'text': raw, 'object': None} if raw else None
    except Exception as e:
        print(f"[Recovery] Error: {e}")
        return None

def _wait_until_robot_silent(timeout=15):
    """Block until TTS finishes speaking, plus a short cooldown for mic to clear."""
    start = time.time()
    while getattr(tts_helper, 'is_speaking', lambda: False)() and time.time() - start < timeout:
        time.sleep(0.1)
    # Extra cooldown so the mic doesn't pick up tail-end audio from the speaker
    time.sleep(1.5)

def _enable_face_tracking():
    """Activate face/sound tracking so the robot follows the child during conversation."""
    try:
        tracker = _ensure_human_tracker()
        if tracker:
            person = _pick_recent_person(tracker, timeout_sec=1.0)
            if person:
                tracker.track(person)
                print("[Conversation] Face tracking enabled")
            elif not getattr(tracker, 'should_track', False):
                # No face found yet — track by sound direction
                tracker.track(None)
                print("[Conversation] Tracking enabled (waiting for face/sound)")
    except Exception as e:
        print(f"[Conversation] Tracking error: {e}")

def _signal_child_can_speak():
    """Play show_tablet gesture to let the child know the mic is on and they can talk."""
    if not ROS_AVAILABLE:
        return
    try:
        from qt_gesture_controller.srv import gesture_play
        ges_proxy = rospy.ServiceProxy('/qt_robot/gesture/play', gesture_play)
        rospy.wait_for_service('/qt_robot/gesture/play', timeout=3.0)
        ges_proxy('QT/show_tablet', 1.0)
        print("[Conversation] show_tablet gesture -> child can speak now")
    except Exception as e:
        print(f"[Conversation] gesture error: {e}")

def _detect_red_card(frame):
    """Detect a red card in the camera frame using HSV color thresholding.
    Returns True if a significant red area is found."""
    if frame is None or cv2 is None:
        return False
    try:
        import numpy as np
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Red wraps around hue 0/180, so use two ranges
        lower_red1 = np.array([0, 100, 80])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([165, 100, 80])
        upper_red2 = np.array([180, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = mask1 | mask2
        red_ratio = mask.sum() / 255.0 / (frame.shape[0] * frame.shape[1])
        detected = red_ratio > 0.03  # 3% of frame is red
        if detected:
            print(f"[RedCard] Detected! red_ratio={red_ratio:.4f}")
        return detected
    except Exception as e:
        print(f"[RedCard] Detection error: {e}")
        return False

def _filter_english_only(text):
    """Remove non-English characters, keeping only ASCII letters, digits, basic punctuation, and spaces."""
    import re
    if not text:
        return ''
    filtered = re.sub(r'[^\x20-\x7E]', ' ', text)
    return re.sub(r'\s+', ' ', filtered).strip()

def _generate_conversation_followup(theme, robot_said, child_said, child_name, child_age, followup_number, total_followups, history, is_closing=False):
    """Generate a conversational follow-up using Gemini."""
    # Filter child's speech to English-only before sending to Gemini
    child_said = _filter_english_only(child_said)
    try:
        script_path = os.path.join(os.path.dirname(BASE_DIR), 'scripts', 'gemini_conversation_followup.py')
        if not os.path.exists(script_path):
            print("[Conversation] Follow-up script not found")
            return None
        input_data = json.dumps({
            "theme": theme,
            "robot_said": robot_said,
            "child_said": child_said,
            "child_name": child_name,
            "child_age": child_age,
            "followup_number": followup_number,
            "total_followups": total_followups,
            "history": history,
            "is_closing": is_closing
        })
        proc = subprocess.run(
            [WORKER_PYTHON, script_path],
            input=input_data, capture_output=True, text=True, timeout=60
        )
        if proc.returncode != 0:
            print(f"[Conversation] Follow-up script error: {proc.stderr}")
            return None
        raw = (proc.stdout or '').strip()
        try:
            result = json.loads(raw)
            return result.get('text', '') or None
        except Exception:
            return raw if raw else None
    except Exception as e:
        print(f"[Conversation] Follow-up error: {e}")
        return None

# Streaming TTS queue for partial ASR text
_stream_tts_queue = Queue()
_stream_tts_thread = None
_stream_tts_stop = ThreadEvent()

def _ensure_stream_tts_worker():
    global _stream_tts_thread
    if _stream_tts_thread and _stream_tts_thread.is_alive():
        return
    def worker():
        while not _stream_tts_stop.is_set():
            try:
                text = _stream_tts_queue.get(timeout=0.2)
            except Exception:
                continue
            try:
                if text:
                    tts_helper.speak_story(text, "en-US")
            except Exception as e:
                print(f"[TTS stream] error: {e}")
    _stream_tts_thread = Thread(target=worker, daemon=True)
    _stream_tts_thread.start()

def _enqueue_tts_chunk(text: str):
    if text:
        _ensure_stream_tts_worker()
        _stream_tts_queue.put(text)

def _has_parallel_recognizers(blocks):
    try:
        for b in blocks:
            if b.get('type') == 'logic':
                cond = b.get('cond') or []
                recog_count = 0
                for c in cond:
                    if c.get('type') == 'recognize' and (c.get('target') or 'speech').lower() == 'speech' and (c.get('value') or '').strip():
                        recog_count += 1
                if recog_count >= 2:
                    return True
        return False
    except Exception:
        return False

def _whisper_recognize_once(language=None):
    """Run whisper.py as a subprocess and return recognized text (best-effort)."""
    script_path = os.path.join(BASE_DIR, "whisper.py")
    python_bin = os.getenv("WHISPER_PYTHON", sys.executable)
    try:
        cmd = [python_bin, script_path]
        if language:
            cmd.extend(["--language", language])
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
            env=os.environ.copy(),
        )
        # Always log stderr for debug info (RMS levels, frame counts)
        if proc.stderr and proc.stderr.strip():
            print(f"[Whisper] {proc.stderr.strip()}")
        if proc.returncode != 0:
            print(f"[Whisper] error (code={proc.returncode})")
            return ""
        raw = (proc.stdout or "").strip()
        # Extract the FINAL: line; fall back to last non-empty line
        recognized = ""
        for line in raw.splitlines():
            line = line.strip()
            if line.startswith("FINAL:"):
                recognized = line.replace("FINAL:", "", 1).strip()
        if not recognized:
            # Fallback: use last non-empty line (older whisper format)
            for line in reversed(raw.splitlines()):
                if line.strip():
                    recognized = line.strip()
                    break
        print(f"[Whisper] recognized: {recognized}")
        return recognized
    except Exception as e:
        print(f"[Whisper] exception: {e}")
        return ""

def _whisper_recognize_streaming(language=None):
    """
    Run whisper.py and stream PARTIAL lines to TTS while returning FINAL text.
    """
    if not _asr_enabled:
        return ""
    script_path = os.path.join(BASE_DIR, "whisper.py")
    python_bin = os.getenv("WHISPER_PYTHON", sys.executable)
    try:
        cmd = [python_bin, script_path]
        if language:
            cmd.extend(["--language", language])
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=os.environ.copy(),
        )
        final_text = ""
        if proc.stdout:
            for line in proc.stdout:
                line = line.strip()
                if not line:
                    continue
                if line.startswith("PARTIAL:"):
                    chunk = line.replace("PARTIAL:", "", 1).strip()
                    _enqueue_tts_chunk(chunk)
                elif line.startswith("FINAL:"):
                    final_text = line.replace("FINAL:", "", 1).strip()
        stderr = (proc.stderr.read() if proc.stderr else "").strip()
        rc = proc.wait()
        if rc != 0:
            print(f"[Whisper] error (code={rc}): {stderr}")
        return final_text
    except Exception as e:
        print(f"[Whisper] exception: {e}")
        return ""

def _extract_story_title(text):
    """Extract the title from raw story text.

    Handles multiple formats:
    - ** Title **\\n<title>\\n\\n<story>
    - <title>\\n\\n<story>  (first line before double newline)

    Returns (title, body) where body is the story without the title.
    """
    if not text:
        return '', text

    working = text.replace('**', '').replace('*', '').strip()

    # Format 1: "Title\n<title line>\n..."
    m = re.match(r'^\s*Title\s*\n\s*(.+?)\s*\n', working, flags=re.IGNORECASE)
    if m:
        title = m.group(1).strip()
        body = working[m.end():]
        return title, body

    # Format 2: first line is title, followed by blank line
    m = re.match(r'^(.+?)\n\s*\n', working)
    if m:
        candidate = m.group(1).strip()
        # Only treat as title if it's short (< 80 chars) and doesn't end with
        # sentence-ending punctuation (titles usually don't end with period)
        if len(candidate) < 80 and not re.search(r'[.!?]\s*$', candidate):
            return candidate, working[m.end():]

    return '', text


def clean_story_text(text):
    """
    Clean story text by removing the title, end markers, explanation,
    asterisks, emojis, and other formatting symbols.

    Args:
        text: Raw story text

    Returns:
        str: Cleaned story body suitable for speech and display
    """
    if not text:
        return text

    # Extract and discard the title
    _title, body = _extract_story_title(text)

    # Remove remaining markdown formatting
    cleaned = body.replace('**', '').replace('*', '')

    # Strip everything from "End" onward (includes Explanation section)
    end_match = re.search(r'\n\s*End\s*$|\n\s*End\s*\n', cleaned, flags=re.IGNORECASE)
    if end_match:
        cleaned = cleaned[:end_match.start()]

    # Remove emojis and special symbols, but preserve punctuation and letters
    cleaned = re.sub(r'[^\w\s.,!?;:()"\'-]', '', cleaned)

    # Clean up extra whitespace
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()

    return cleaned

@app.route("/")
def index():
    # If user is logged in, show main dashboard with two options
    if 'username' in session:
        username = session['username']
        user = user_manager.users.get(username)
        return render_template("dashboard.html", logged_in=True, user=user)
    return render_template("index.html", logged_in=False)

@app.route("/api/update_profile", methods=["POST"])
def api_update_profile():
    username = session.get('username')
    if not username:
        return jsonify({"success": False, "error": "Not logged in"}), 401
    data = request.get_json() or {}
    # Allowed editable fields
    allowed = {"display_name", "age", "gender", "disorder", "learning_goals"}
    user = user_manager.users.get(username)
    if not user:
        return jsonify({"success": False, "error": "User not found"}), 404
    try:
        updates = {k: v for k, v in data.items() if k in allowed}
        if 'age' in updates:
            try:
                updates['age'] = int(updates['age'])
                if updates['age'] < 0 or updates['age'] > 150:
                    return jsonify({"success": False, "error": "Invalid age"}), 400
            except Exception:
                return jsonify({"success": False, "error": "Invalid age"}), 400
        user.update(updates)
        # Persist to users.json
        user_manager._save_users()
        # Also persist a profile.json under src/user_data/<username>/
        try:
            import datetime
            user_dir = os.path.join(USER_DATA_DIR, username)
            os.makedirs(user_dir, exist_ok=True)
            profile_path = os.path.join(user_dir, 'profile.json')
            profile_doc = {
                "username": user.get("username"),
                "display_name": user.get("display_name", user.get("username")),
                "age": user.get("age"),
                "gender": user.get("gender"),
                "disorder": user.get("disorder"),
                "learning_goals": user.get("learning_goals", ""),
                "updated_at": datetime.datetime.now().isoformat()
            }
            with open(profile_path, 'w') as pf:
                json.dump(profile_doc, pf, indent=2)
        except Exception as e:
            # Do not fail the request if file write fails; just log
            print(f"Warning: failed to write profile.json: {e}")
        # Return sanitized user
        return jsonify({"success": True, "user": {
            "username": user.get("username"),
            "display_name": user.get("display_name", user.get("username")),
            "age": user.get("age"),
            "gender": user.get("gender"),
            "disorder": user.get("disorder"),
            "learning_goals": user.get("learning_goals", "")
        }})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/register", methods=["POST"])
def api_register():
    data = request.get_json()
    username = data.get("username", "").strip()
    age = data.get("age")
    if not username or age is None:
        return jsonify({"error": "Username and age are required"}), 400
    try:
        age = int(age)
    except Exception:
        return jsonify({"error": "Invalid age"}), 400
    # No password or email
    if user_manager.register_user(username, age):
        return jsonify({"success": True}), 200
    else:
        return jsonify({"error": "Registration failed. Username might already exist or invalid age."}), 400

@app.route("/api/login", methods=["POST"])
def api_login():
    data = request.get_json()
    username = data.get("username", "").strip()
    if not username:
        return jsonify({"error": "Username is required"}), 400
    # No password
    if user_manager.authenticate_user(username):
        session['username'] = username
        try:
            tts_helper.set_current_user(username)
        except Exception:
            pass
        user = user_manager.users[username]
        return jsonify({"success": True, "user": {
            "username": user["username"],
            "age": user["age"],
            "created_at": user["created_at"],
            "last_login": user["last_login"]
        }}), 200
    else:
        return jsonify({"error": "Invalid username"}), 401

@app.route("/api/logout", methods=["POST"])
def api_logout():
    session.pop('username', None)
    user_manager.logout()
    try:
        tts_helper.set_current_user(None)
    except Exception:
        pass
    return jsonify({"success": True})

@app.route("/api/current_user")
def api_current_user():
    username = session.get('username')
    if username and username in user_manager.users:
        user = user_manager.users[username]
        return jsonify({"user": {
            "username": user["username"],
            "age": user["age"],
            "created_at": user["created_at"],
            "last_login": user["last_login"],
            "display_name": user.get("display_name", user.get("username")),
            "gender": user.get("gender"),
            "disorder": user.get("disorder"),
            "learning_goals": user.get("learning_goals", "")
        }})
    return jsonify({"user": None})

@app.route("/api/users")
def api_users():
    users = [
        {
            "username": u["username"],
            "age": u["age"],
            "created_at": u["created_at"],
            "last_login": u["last_login"]
        }
        for u in user_manager.users.values()
    ]
    return jsonify({"users": users})

@app.route("/api/user_stats")
def api_user_stats():
    username = session.get('username')
    if not username:
        return jsonify({"error": "Not logged in"}), 401
    stats = user_manager.get_user_stats(username)
    return jsonify(stats)

@app.route("/api/get_custom_games", methods=["GET"])
def api_get_custom_games():
    """List user's saved DIY activities for dashboard display"""
    username = session.get('username')
    if not username:
        return jsonify({"success": False, "error": "Not logged in"}), 401
    try:
        user_dir = os.path.join(USER_DATA_DIR, username, "activities")
        os.makedirs(user_dir, exist_ok=True)
        games = []
        for fname in os.listdir(user_dir):
            if not fname.endswith('.json'):
                continue
            fpath = os.path.join(user_dir, fname)
            created_at = 'Unknown'
            try:
                if fname.startswith('activity_'):
                    ts = fname.replace('activity_', '').replace('.json', '')
                    created_at = f"{ts[:8]} {ts[8:10]}:{ts[10:12]}:{ts[12:14]}"
            except Exception:
                pass
            try:
                with open(fpath, 'r') as f:
                    data = json.load(f)
                blocks = data.get('blocks', [])
            except Exception:
                blocks = []
            activity_type = 'diy'
            try:
                activity_type = data.get('activity_type', 'diy') if isinstance(data, dict) else 'diy'
            except Exception:
                pass
            games.append({
                "filename": fname,
                "created_at": created_at,
                "blocks_count": len(blocks),
                "activity_type": activity_type
            })
        # newest first by created_at string
        games.sort(key=lambda x: x.get('created_at', ''), reverse=True)
        return jsonify({"success": True, "games": games})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/generate_story", methods=["POST"])
def api_generate_story():
    """Generate a therapeutic story for the logged-in user"""
    username = session.get('username')
    if not username:
        return jsonify({"error": "Not logged in"}), 401
    
    user = user_manager.users.get(username)
    if not user:
        return jsonify({"error": "User not found"}), 404
    
    data = request.get_json() or {}
    child_name = data.get("child_name", username)
    age = data.get("age", user.get("age", 4))
    custom_prompt = data.get("custom_prompt")
    topics = data.get("topics")

    # Load learning goals from profile.json if available
    gender = user.get("gender", "")
    learning_goals = user.get("learning_goals", "")
    
    try:
        profile_path = os.path.join(USER_DATA_DIR, username, "profile.json")
        if os.path.exists(profile_path):
            with open(profile_path, "r") as pf:
                profile = json.load(pf)
            learning_goals = profile.get("learning_goals", learning_goals)
            gender = profile.get("gender", gender)
    except Exception as e:
        print(f"Warning: failed to read profile.json: {e}")
    try:
        result = story_generator.generate_story(
            child_name=child_name,
            age=age,
            gender=gender,
            custom_prompt=custom_prompt,
            topics=topics,
            goals=learning_goals
        )
        if result["success"]:
            return jsonify(result), 200
        else:
            return jsonify({"error": result["error"]}), 500

                
    except Exception as e:
        return jsonify({"error": f"Story generation failed: {str(e)}"}), 500

@app.route("/api/generate_story_stream", methods=["POST"])
def api_generate_story_stream():
    """Generate a therapeutic story with streaming response"""
    username = session.get('username')
    if not username:
        return jsonify({"error": "Not logged in"}), 401
    
    user = user_manager.users.get(username)
    if not user:
        return jsonify({"error": "User not found"}), 404
    
    data = request.get_json() or {}
    child_name = data.get("child_name", username)
    age = data.get("age", user.get("age", 4))
    custom_prompt = data.get("custom_prompt")
    topics = data.get("topics")

    learning_goals = user.get("learning_goals", "")
    gender = user.get("gender", "")
    try:
        profile_path = os.path.join(USER_DATA_DIR, username, "profile.json")
        if os.path.exists(profile_path):
            with open(profile_path, "r") as pf:
                profile = json.load(pf)
            learning_goals = profile.get("learning_goals", learning_goals)
            gender = profile.get("gender", gender)
    except Exception as e:
        print(f"Warning: failed to read profile.json: {e}")
        
    def generate():
        try:
            # Send initial metadata event for streaming clients
            meta = {
                "child_name": child_name,
                "age": age,
                "gender": gender,
                "topics": topics or [],
            }
            yield f"data: {json.dumps({'meta': meta})}\n\n"
            for chunk in story_generator.generate_story_stream(
                child_name=child_name,
                age=age,
                gender=gender,
                custom_prompt=custom_prompt,
                topics=topics,
                goals=learning_goals
            ):
                yield f"data: {json.dumps({'chunk': chunk})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return Response(generate(), mimetype='text/plain')

@app.route("/start_assistant")
def start_assistant():
    # This endpoint can be used to redirect to the main assistant app
    # For now, just show a message
    return "<h2>QTrobot AI Assistant will start here (integration point).</h2>"

@app.route("/quiz_generation")
def quiz_generation_page():
    """Render the quiz generation page."""
    if 'username' not in session:
        return redirect(url_for('index'))
    return render_template("quiz_generation.html")

@app.route("/api/generate_quiz", methods=["POST"])
def api_generate_quiz():
    """Generate quiz questions using Llama."""
    username = session.get('username')
    if not username:
        return jsonify({"success": False, "error": "Not logged in"}), 401
    data = request.get_json() or {}
    topics = data.get("topics") or []
    if isinstance(topics, str):
        topics = [topics]
    topics = [str(t).strip() for t in topics if str(t).strip()]
    difficulty = (data.get("difficulty") or "Med").strip()
    count = int(data.get("count") or 5)
    types = data.get("types") or []

    if not topics:
        return jsonify({"success": False, "error": "Topic is required"}), 400
    if count < 1 or count > 20:
        return jsonify({"success": False, "error": "Count must be 1-20"}), 400
    if not types:
        return jsonify({"success": False, "error": "Select at least one question type"}), 400

    _ensure_quiz_llm()
    if _quiz_llm is None:
        return jsonify({"success": False, "error": "Quiz LLM not available"}), 500

    type_hint = ", ".join(types)
    rule_text = ""
    if types == ["yes_no"] or (len(types) == 1 and types[0] == "yes_no"):
        rule_text = "Rules: yes_no questions must be answerable with a clear yes or no (correct/incorrect)."
    elif types == ["wh"] or (len(types) == 1 and types[0] == "wh"):
        rule_text = "Rules: wh questions must begin with one of: what, when, where, why, who, how."
    else:
        rule_text = (
            "Rules: questions must be answerable with a clear yes or no (correct/incorrect). "
            "questions must begin with one of: what, when, where, why, who, how."
        )

    age_hint = ""
    if difficulty.lower() == "low":
        age_hint = "Target ages 2-3."
    elif difficulty.lower() == "med":
        age_hint = "Target ages 4-5."
    elif difficulty.lower() == "high":
        age_hint = "Target ages 7+."

    topic_text = ", ".join(topics)
    prompt = (
        f"Act as a pediatric educator. Create {count} questions about the topic(s) '{topic_text}'. "
        f"{age_hint} "
        f"Use only these types: {type_hint}. "
        "Goal: Questions must be objectively True or False based on basic object functions or category labels. "
        "Avoid subjective questions like 'Do you like school?' or 'Are there toys?'. "
        "Constraint: Questions must be short (under 8 words). "
        "Return Format: JSON array of objects with keys: 'question', 'type', 'correct_answer', 'accepted_answers'. "
        "For yes_no, correct_answer must be 'yes' or 'no' and accepted_answers should be omitted. "
        "For wh, correct_answer is the primary short answer and accepted_answers must be a list of all "
        "reasonably correct alternative answers (synonyms, related valid answers, plural/singular forms). "
        "Example: if question is 'Where do kids read books in school?', correct_answer is 'classroom' and "
        "accepted_answers could be ['classroom', 'library', 'reading room', 'classrooms']. "
        f"{rule_text}"
    )
    print("education question prompt: ", prompt)

    try:
        resp = _quiz_llm.get_response(prompt)
        text = getattr(resp, 'message', None)
        text = getattr(text, 'content', None) if text is not None else str(resp)
        raw = (text or "").strip()
        if raw.startswith("```"):
            raw = raw.strip("`")
        # Extract JSON
        obj = None
        try:
            obj = json.loads(raw)
        except Exception:
            l = raw.find('[')
            r = raw.rfind(']')
            if l != -1 and r != -1 and r > l:
                obj = json.loads(raw[l:r+1])
        if not isinstance(obj, list):
            return jsonify({"success": False, "error": "LLM returned invalid JSON"}), 500

        questions = []
        for item in obj:
            if not isinstance(item, dict):
                continue
            q = (item.get("question") or "").strip()
            t = (item.get("type") or "").strip().lower()
            correct_answer = (item.get("correct_answer") or "").strip()
            accepted_answers = item.get("accepted_answers") or []
            if not isinstance(accepted_answers, list):
                accepted_answers = []
            accepted_answers = [str(a).strip() for a in accepted_answers if str(a).strip()]
            if not q:
                continue
            if t not in ("yes_no", "wh"):
                # Try to infer
                t = "yes_no" if q.lower().startswith(("is", "are", "do", "does", "can", "did")) else "wh"
            if t == "yes_no":
                correct_answer = correct_answer.lower()
                if correct_answer not in ("yes", "no"):
                    correct_answer = ""
                entry = {"question": q, "type": t, "correct_answer": correct_answer}
            else:
                if not correct_answer:
                    correct_answer = ""
                # Ensure correct_answer is in accepted_answers
                if correct_answer and correct_answer not in accepted_answers:
                    accepted_answers.insert(0, correct_answer)
                entry = {"question": q, "type": t, "correct_answer": correct_answer, "accepted_answers": accepted_answers}
            questions.append(entry)

        # Post-process: generate accepted_answers for WH questions that have empty/insufficient lists
        wh_needing_alts = [q for q in questions if q.get("type") == "wh" and len(q.get("accepted_answers", [])) <= 1]
        if wh_needing_alts and _quiz_llm is not None:
            try:
                alt_input = [{"question": q["question"], "correct_answer": q["correct_answer"]} for q in wh_needing_alts]
                alt_prompt = (
                    "For each question below, generate a list of all reasonably correct alternative answers "
                    "that a child might give. Include the original answer, synonyms, plural/singular forms, "
                    "and semantically valid alternatives. "
                    "Return a JSON array where each element is a list of accepted answer strings, in the same order as the input. "
                    "Example input: [{\"question\": \"Where do kids read books?\", \"correct_answer\": \"classroom\"}] "
                    "Example output: [[\"classroom\", \"classrooms\", \"library\", \"reading room\", \"school\"]] "
                    f"Input: {json.dumps(alt_input)}"
                )
                alt_resp = _quiz_llm.get_response(alt_prompt)
                alt_text = getattr(alt_resp, 'message', None)
                alt_text = getattr(alt_text, 'content', None) if alt_text is not None else str(alt_resp)
                alt_raw = (alt_text or "").strip()
                if alt_raw.startswith("```"):
                    alt_raw = alt_raw.strip("`")
                    if alt_raw.startswith("json"):
                        alt_raw = alt_raw[4:].strip()
                alt_obj = None
                try:
                    alt_obj = json.loads(alt_raw)
                except Exception:
                    l2 = alt_raw.find('[')
                    r2 = alt_raw.rfind(']')
                    if l2 != -1 and r2 != -1 and r2 > l2:
                        alt_obj = json.loads(alt_raw[l2:r2+1])
                if isinstance(alt_obj, list) and len(alt_obj) == len(wh_needing_alts):
                    for q, alts in zip(wh_needing_alts, alt_obj):
                        if isinstance(alts, list):
                            merged = list(alts)
                            if q["correct_answer"] and q["correct_answer"] not in merged:
                                merged.insert(0, q["correct_answer"])
                            q["accepted_answers"] = [str(a).strip() for a in merged if str(a).strip()]
            except Exception as e:
                print(f"Warning: accepted_answers generation failed: {e}")

        return jsonify({
            "success": True,
            "questions": questions,
            "topics": topics,
            "difficulty": difficulty
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/robot_gesture", methods=["POST"])
def api_robot_gesture():
    """Play a gesture and/or emotion on the robot."""
    username = session.get('username')
    if not username:
        return jsonify({"success": False, "error": "Not logged in"}), 401
    if not ROS_AVAILABLE:
        return jsonify({"success": False, "error": "ROS not available"}), 500
    data = request.get_json() or {}
    gesture = (data.get("gesture") or "").strip()
    emotion = (data.get("emotion") or "").strip()
    speed = float(data.get("speed", 1.0))
    try:
        from qt_robot_interface.srv import emotion_show
        from qt_gesture_controller.srv import gesture_play
        if emotion:
            emo_proxy = rospy.ServiceProxy('/qt_robot/emotion/show', emotion_show)
            emo_proxy.wait_for_service(timeout=2.0)
            emo_proxy(emotion)
        if gesture:
            ges_proxy = rospy.ServiceProxy('/qt_robot/gesture/play', gesture_play)
            ges_proxy.wait_for_service(timeout=2.0)
            ges_proxy(gesture, speed)
        return jsonify({"success": True})
    except Exception as e:
        print(f"[Gesture] error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/speech_recognize", methods=["POST"])
def api_speech_recognize():
    """Run whisper ASR once and return the recognized text."""
    username = session.get('username')
    if not username:
        return jsonify({"success": False, "error": "Not logged in"}), 401
    try:
        # Extract ISO-639-1 code from the configured language (e.g. "en-US" -> "en")
        lang_code = None
        config_lang = session.get('lang') or app.config.get('LANG', 'en-US')
        if config_lang and '-' in config_lang:
            lang_code = config_lang.split('-')[0]
        elif config_lang:
            lang_code = config_lang
        text = _whisper_recognize_once(language=lang_code)
        return jsonify({"success": True, "text": text.strip()})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/educational_quiz")
def educational_quiz_page():
    """Render the educational quiz play page."""
    if 'username' not in session:
        return redirect(url_for('index'))
    return render_template("educational_quiz.html")

@app.route("/api/get_saved_quiz")
def api_get_saved_quiz():
    """Return all saved quiz questions of a given type for the current user."""
    username = session.get('username')
    if not username:
        return jsonify({"success": False, "error": "Not logged in"}), 401
    qtype = request.args.get("type", "").strip()
    if qtype not in ("yes_no", "wh"):
        return jsonify({"success": False, "error": "Invalid type. Use yes_no or wh."}), 400

    quiz_dir = os.path.join(USER_DATA_DIR, username, "quizzes", qtype)
    all_questions = []
    if os.path.isdir(quiz_dir):
        for fname in sorted(os.listdir(quiz_dir)):
            if not fname.endswith(".json"):
                continue
            try:
                with open(os.path.join(quiz_dir, fname), "r") as f:
                    qs = json.load(f)
                if isinstance(qs, list):
                    all_questions.extend(qs)
            except Exception:
                continue

    # Merge user-taught answers into accepted_answers
    learned_path = os.path.join(USER_DATA_DIR, username, "quizzes", "learned_answers.json")
    learned = {}
    if os.path.isfile(learned_path):
        try:
            with open(learned_path, "r") as f:
                learned = json.load(f)
        except Exception:
            pass
    if learned:
        for q in all_questions:
            question_key = q.get("question", "").strip()
            extra = learned.get(question_key, [])
            if extra:
                if not q.get("accepted_answers"):
                    q["accepted_answers"] = [q["correct_answer"]] if q.get("correct_answer") else []
                for ans in extra:
                    if ans not in q["accepted_answers"]:
                        q["accepted_answers"].append(ans)

    return jsonify({"success": True, "questions": all_questions})

@app.route("/api/generate_quiz_feedback", methods=["POST"])
def api_generate_quiz_feedback():
    """Pre-generate varied feedback phrases for correct/incorrect answers using the system prompt."""
    username = session.get('username')
    if not username:
        return jsonify({"success": False, "error": "Not logged in"}), 401

    _ensure_quiz_llm()
    if _quiz_llm is None:
        return jsonify({"success": True, "correct": [], "incorrect": []})

    # Load the SAR system prompt for pediatric context
    system_prompt_path = os.path.join(os.path.dirname(BASE_DIR), "documents", "sar_system_prompt.md")
    system_context = ""
    if os.path.isfile(system_prompt_path):
        try:
            with open(system_prompt_path, "r") as f:
                system_context = f.read()
        except Exception:
            pass

    prompt = (
        "You are a socially assistive robot in a pediatric therapeutic setting. "
        "Here is your system prompt for context:\n"
        f"{system_context}\n\n"
        "Generate 10 short, varied, child-friendly phrases for when a child answers a quiz question CORRECTLY, "
        "and 10 short, varied, child-friendly phrases for when a child answers INCORRECTLY. "
        "Follow these rules:\n"
        "- Each phrase must be 2-8 words maximum\n"
        "- Use warm, encouraging, effort-focused language\n"
        "- For incorrect: be gentle, never shaming. Encourage trying again\n"
        "- Vary the style: some excited, some calm, some playful\n"
        "- Do not use emojis\n"
        "Return JSON only: {\"correct\": [\"...\", ...], \"incorrect\": [\"...\", ...]}"
    )

    try:
        resp = _quiz_llm.get_response(prompt)
        text = getattr(resp, 'message', None)
        text = getattr(text, 'content', None) if text is not None else str(resp)
        raw = (text or "").strip()
        if raw.startswith("```"):
            raw = raw.strip("`")
            if raw.startswith("json"):
                raw = raw[4:].strip()
        obj = None
        try:
            obj = json.loads(raw)
        except Exception:
            l = raw.find('{')
            r = raw.rfind('}')
            if l != -1 and r != -1 and r > l:
                obj = json.loads(raw[l:r+1])
        if isinstance(obj, dict):
            correct = [str(s).strip() for s in (obj.get("correct") or []) if str(s).strip()]
            incorrect = [str(s).strip() for s in (obj.get("incorrect") or []) if str(s).strip()]
            return jsonify({"success": True, "correct": correct, "incorrect": incorrect})
    except Exception as e:
        print(f"Warning: feedback generation failed: {e}")

    return jsonify({"success": True, "correct": [], "incorrect": []})

@app.route("/api/teach_quiz_answer", methods=["POST"])
def api_teach_quiz_answer():
    """Save user-taught alternative answers for a specific question."""
    username = session.get('username')
    if not username:
        return jsonify({"success": False, "error": "Not logged in"}), 401
    data = request.get_json() or {}
    question = (data.get("question") or "").strip()
    answers = data.get("answers") or []
    if not question or not answers:
        return jsonify({"success": False, "error": "Question and answers required"}), 400

    answers = [str(a).strip() for a in answers if str(a).strip()]
    if not answers:
        return jsonify({"success": False, "error": "No valid answers provided"}), 400

    learned_path = os.path.join(USER_DATA_DIR, username, "quizzes", "learned_answers.json")
    os.makedirs(os.path.dirname(learned_path), exist_ok=True)

    learned = {}
    if os.path.isfile(learned_path):
        try:
            with open(learned_path, "r") as f:
                learned = json.load(f)
        except Exception:
            learned = {}

    existing = learned.get(question, [])
    for ans in answers:
        if ans not in existing:
            existing.append(ans)
    learned[question] = existing

    with open(learned_path, "w") as f:
        json.dump(learned, f, indent=2)

    return jsonify({"success": True, "total_answers": len(existing)})

@app.route("/api/save_quiz", methods=["POST"])
def api_save_quiz():
    """Save generated quiz questions to the user's folder, split by type."""
    username = session.get('username')
    if not username:
        return jsonify({"success": False, "error": "Not logged in"}), 401
    data = request.get_json() or {}
    questions = data.get("questions") or []
    if not questions:
        return jsonify({"success": False, "error": "No questions provided"}), 400

    import datetime
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    yes_no_qs = [q for q in questions if q.get("type") == "yes_no"]
    wh_qs = [q for q in questions if q.get("type") == "wh"]

    saved = []
    if yes_no_qs:
        folder = os.path.join(USER_DATA_DIR, username, "quizzes", "yes_no")
        os.makedirs(folder, exist_ok=True)
        fpath = os.path.join(folder, f"quiz_{ts}.json")
        with open(fpath, "w") as f:
            json.dump(yes_no_qs, f, indent=2)
        saved.append(fpath)

    if wh_qs:
        folder = os.path.join(USER_DATA_DIR, username, "quizzes", "wh")
        os.makedirs(folder, exist_ok=True)
        fpath = os.path.join(folder, f"quiz_{ts}.json")
        with open(fpath, "w") as f:
            json.dump(wh_qs, f, indent=2)
        saved.append(fpath)

    return jsonify({
        "success": True,
        "yes_no_count": len(yes_no_qs),
        "wh_count": len(wh_qs),
        "files": saved
    })

def _split_story_into_pages(story_text, child_age):
    """Split a story into age-appropriate pages using Gemini.

    Ages 3-4:  1-2 sentences per page
    Ages 5-6:  2-3 sentences per page
    Ages 7+:   3+ sentences per page (context-dependent)

    Returns a list of page strings. Falls back to simple splitting if LLM is unavailable.
    """
    cleaned = clean_story_text(story_text)
    if not cleaned.strip():
        return [cleaned]

    if child_age <= 4:
        sents_per_page = "1-2"
    elif child_age <= 6:
        sents_per_page = "2-3"
    else:
        sents_per_page = "3-5, depending on the narrative flow and context"

    prompt = (
        f"Split the following story into pages for a {child_age}-year-old child.\n"
        f"Each page should have {sents_per_page} sentences.\n"
        f"Rules:\n"
        f"- Keep sentences intact, do NOT rephrase or change any words.\n"
        f"- Group sentences so each page forms a coherent scene or moment.\n"
        f"- Do not split a sentence across pages.\n"
        f"- Return ONLY a JSON array of strings, where each string is one page.\n"
        f"- Example: [\"Page 1 text here.\", \"Page 2 text here.\"]\n\n"
        f"Story:\n{cleaned}"
    )
    raw = _gemini_generate(prompt, system="You split stories into pages. Return JSON only.", max_tokens=4096)
    if raw:
        try:
            print(f"[StoryPages] Gemini raw response: {raw[:500]}")
            if raw.startswith('```'):
                raw = raw.strip('`').strip()
                if raw.startswith('json'):
                    raw = raw[4:].strip()
            start = raw.find('[')
            end = raw.rfind(']')
            if start != -1 and end != -1 and end > start:
                pages = json.loads(raw[start:end+1])
                if isinstance(pages, list) and len(pages) > 0:
                    pages = [str(p).strip() for p in pages if str(p).strip()]
                    if pages:
                        print(f"[StoryPages] Split into {len(pages)} pages (age {child_age})")
                        return pages
        except Exception as e:
            print(f"[StoryPages] LLM page splitting failed: {e}")

    # Fallback: simple sentence-based splitting
    sentences = re.split(r'(?<=[.!?])\s+', cleaned.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return [cleaned]

    if child_age <= 4:
        n = 2
    elif child_age <= 6:
        n = 3
    else:
        n = 4

    pages = []
    for i in range(0, len(sentences), n):
        page = ' '.join(sentences[i:i+n])
        pages.append(page)
    print(f"[StoryPages] Fallback split into {len(pages)} pages (age {child_age}, {n} sents/page)")
    return pages


def _identify_story_scenes(pages):
    """Analyze story pages and identify scene/context changes.

    Uses the quiz LLM to group pages by scene — a scene changes when the
    setting, characters, or action shifts significantly.

    Returns:
        scenes: list of scene description strings (one per unique scene)
        page_to_scene: list of ints mapping each page index to a scene index
    """
    if not pages:
        return [""], [0]

    full_text = "\n\n".join(f"Page {i+1}: {p}" for i, p in enumerate(pages))

    prompt = (
        f"You are analyzing a children's story that has been split into {len(pages)} pages.\n"
        f"Identify where the SCENE or CONTEXT changes — a new scene starts when the "
        f"setting changes, new characters appear, or a significantly different action begins.\n"
        f"Group consecutive pages that share the same scene together.\n\n"
        f"{full_text}\n\n"
        f"Return ONLY a JSON object with:\n"
        f"- \"scenes\": an array of short visual descriptions (1-2 sentences each) "
        f"describing what should be illustrated for each scene. Focus on setting, "
        f"characters, and key action.\n"
        f"- \"page_to_scene\": an array of {len(pages)} integers, where each integer "
        f"is the 0-based scene index for that page.\n\n"
        f"Example for 5 pages with 3 scenes:\n"
        f"{{\"scenes\": [\"A child in a sunny garden planting seeds\", "
        f"\"The child and a rabbit watching rain fall on the garden\", "
        f"\"The child picking colorful flowers from the grown garden\"], "
        f"\"page_to_scene\": [0, 0, 1, 1, 2]}}"
    )
    raw = _gemini_generate(prompt, system="You analyze story structure. Return JSON only.", max_tokens=2048)
    if raw:
        try:
            print(f"[StoryScenes] Gemini raw response: {raw[:500]}")
            obj = _extract_json(raw)
            print(f"[StoryScenes] Parsed JSON: {json.dumps(obj, indent=2) if obj else None}")
            if obj:
                scenes = obj.get('scenes', [])
                mapping = obj.get('page_to_scene', [])
                if (isinstance(scenes, list) and len(scenes) > 0
                        and isinstance(mapping, list) and len(mapping) == len(pages)):
                    if all(isinstance(m, int) and 0 <= m < len(scenes) for m in mapping):
                        return scenes, mapping
        except Exception as e:
            print(f"[StoryScenes] Scene identification failed: {e}")

    # Fallback: treat each page as its own scene
    print("[StoryScenes] Using fallback: one scene per page")
    scenes = [p[:200] for p in pages]  # Use first 200 chars as scene description
    page_to_scene = list(range(len(pages)))
    return scenes, page_to_scene


@app.route("/api/save_story", methods=["POST"])
def api_save_story():
    username = session.get('username')
    if not username:
        return jsonify({"error": "Not logged in"}), 401
    user = user_manager.users.get(username)
    if not user:
        return jsonify({"error": "User not found"}), 404
    data = request.get_json() or {}
    story = data.get("story")
    metadata = data.get("metadata")
    if not story or not metadata:
        return jsonify({"error": "Missing story or metadata"}), 400

    # Extract title from story text and store in metadata
    title, _body = _extract_story_title(story)
    if title:
        metadata['title'] = title
        print(f"[StorySave] Extracted title: {title}")

    # Get child age for page splitting
    child_age = 5
    try:
        child_age = int(metadata.get('age', user.get('age', 5)))
    except (ValueError, TypeError):
        child_age = 5

    # Split story into age-appropriate pages (clean_story_text strips the title)
    pages = _split_story_into_pages(story, child_age)

    # Prepare user stories directory
    user_dir = os.path.join(USER_DATA_DIR, username, "stories")
    os.makedirs(user_dir, exist_ok=True)

    # Use timestamp for unique filename
    import datetime
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"story_{ts}.json"
    fpath = os.path.join(user_dir, fname)

    # Identify scene breaks and map pages to scenes
    scenes, page_to_scene = _identify_story_scenes(pages)
    print(f"[StorySave] {len(scenes)} scenes identified, page_to_scene: {page_to_scene}")

    # Save story, metadata, pages, scenes, and mapping
    with open(fpath, "w") as f:
        json.dump({
            "story": story,
            "metadata": metadata,
            "pages": pages,
            "scenes": scenes,
            "page_to_scene": page_to_scene,
        }, f, indent=2)

    # Generate one image per scene (not per page)
    if image_generator.is_available():
        try:
            user_images_dir = os.path.join(USER_DATA_DIR, username, "story_images", fname.replace(".json", ""))
            os.makedirs(user_images_dir, exist_ok=True)

            for i, scene_desc in enumerate(scenes):
                image_generator.generate_story_scene_image(
                    scene_desc,
                    story_context=f"Story about {metadata.get('child_name', 'a child')}",
                    output_dir=user_images_dir,
                    filename_prefix=f"story_scene_{i:03d}",
                )

            print(f"[StorySave] Generated {len(scenes)} scene images for story {fname}")

        except Exception as e:
            print(f"Error generating scene images for story {fname}: {str(e)}")
    else:
        print("image_generator not available")

    return jsonify({"success": True, "filename": fname})

@app.route("/generate")
def generate_games():
    """Game generation page - shows the original game selection interface"""
    if 'username' not in session:
        return redirect(url_for('index'))
    username = session['username']
    user = user_manager.users.get(username)
    return render_template("index.html", logged_in=True, user=user, show_game_selection=True)

@app.route("/play")
def play_games():
    """Play games page - shows the interactive games interface"""
    if 'username' not in session:
        return redirect(url_for('index'))
    username = session['username']
    user = user_manager.users.get(username)
    # Start continuous human tracking on entering play mode
    try:
        tracker = _ensure_human_tracker()
        if tracker and not tracker.should_track:
            person = _pick_recent_person(tracker, timeout_sec=0.5)
            tracker.track(person)
    except Exception as e:
        print(f"HumanTracking auto-start (/play) error: {e}")
    return render_template("play_games.html", logged_in=True, user=user)

@app.route("/play_scene")
def play_scene_page():
    """Dedicated Scene Detection play page"""
    if 'username' not in session:
        return redirect(url_for('index'))
    username = session['username']
    user = user_manager.users.get(username)
    # Ensure tracking is running on scene play page as well
    try:
        tracker = _ensure_human_tracker()
        if tracker and not tracker.should_track:
            person = _pick_recent_person(tracker, timeout_sec=0.5)
            tracker.track(person)
    except Exception as e:
        print(f"HumanTracking auto-start (/play_scene) error: {e}")
    return render_template("play_scene.html", logged_in=True, user=user)

def _gemini_generate(prompt, system="You are a helpful assistant. Return JSON only when asked.",
                     temperature=0.3, max_tokens=2048):
    """Call Gemini via subprocess for general-purpose text generation.

    Returns the raw response text, or None on failure.
    """
    script_path = os.path.join(os.path.dirname(BASE_DIR), 'scripts', 'gemini_general.py')
    if not os.path.exists(script_path):
        print("[Gemini] gemini_general.py not found")
        return None
    import tempfile
    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
    try:
        tmp.write(prompt)
        tmp.close()
        cmd = [WORKER_PYTHON, script_path,
               '--prompt-file', tmp.name,
               '--system', system,
               '--temperature', str(temperature),
               '--max-tokens', str(max_tokens)]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
        if proc.returncode != 0:
            print(f"[Gemini] Script error: {proc.stderr[:300]}")
            return None
        return (proc.stdout or '').strip()
    except Exception as e:
        print(f"[Gemini] Error: {e}")
        return None
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


def _extract_json(raw):
    """Extract the first JSON object from a string that may contain extra text."""
    raw = raw.strip()
    if raw.startswith('```'):
        raw = raw.strip('`').strip()
        if raw.startswith('json'):
            raw = raw[4:].strip()
    # Try direct parse first
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    # Find first { ... } block
    start = raw.find('{')
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(raw)):
        if raw[i] == '{':
            depth += 1
        elif raw[i] == '}':
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(raw[start:i+1])
                except json.JSONDecodeError:
                    return None
    return None


def _scene_game_generate_question(toy_list, child_age, learning_goals):
    """Use the quiz LLM to generate a scene-game question.

    For ages 2-3: direct request naming one specific object.
    For ages 4-6: criteria-based (e.g. "a red fruit") — multiple toys may match.
    For ages 7+:  complex inference riddle — child must reason about properties.

    Returns dict with keys:
        question  – the text to speak
        target    – exact toy name (ages 2-3) or None (ages 4+)
        criteria  – descriptive criteria string (ages 4+) or None (ages 2-3)
        mode      – "exact" | "criteria"
    """
    # Pick one toy as the primary target (always used as fallback)
    target = random.choice(toy_list)

    # Determine mode based on age
    if child_age <= 3:
        mode = "exact"
    else:
        mode = "criteria"

    goals_clause = ""
    if learning_goals:
        goals_clause = (
            f"The child's therapy session goals are: {learning_goals}. "
            "Weave these goals naturally into the question (e.g. target vocabulary, "
            "sentence structure, or concepts relevant to the goals). "
        )

    if mode == "exact":
        prompt = (
            f"Generate ONE short, direct request that a friendly robot says to a "
            f"{child_age}-year-old child, asking them to find and show \"{target}\".\n"
            f"{goals_clause}"
            f"Use very simple, direct language. Name the object explicitly.\n"
            f"Examples: \"Show me the apple!\", \"Do you have a banana?\", "
            f"\"Can you give me the pear?\"\n"
            f"Return ONLY a JSON object: "
            f"{{\"question\": \"<the sentence>\", \"target\": \"{target}\"}}"
        )
    elif child_age <= 6:
        prompt = (
            f"You are generating a question for an object detection game for a "
            f"{child_age}-year-old child.\n"
            f"Available physical toys: {', '.join(toy_list)}.\n"
            f"{goals_clause}"
            f"Generate ONE request that describes a TARGET object by its observable "
            f"properties (color, category, shape) WITHOUT naming any specific object.\n"
            f"The criteria MUST match at least one toy from the list above.\n"
            f"Use simple, clear language appropriate for ages 4-6.\n"
            f"Examples:\n"
            f"- \"I want a red fruit!\" (matches apple, strawberry, tomato)\n"
            f"- \"Can you find something yellow?\" (matches banana, lemon)\n"
            f"- \"Show me a round vegetable!\" (matches tomato)\n"
            f"Return ONLY a JSON object:\n"
            f"{{\"question\": \"<the sentence>\", "
            f"\"criteria\": \"<short criteria phrase, e.g. red fruit>\"}}"
        )
    else:
        prompt = (
            f"You are generating a question for an object detection game for a "
            f"{child_age}-year-old child.\n"
            f"Available physical toys: {', '.join(toy_list)}.\n"
            f"{goals_clause}"
            f"Generate ONE riddle or multi-step descriptive clue so the child must "
            f"reason about properties (color, shape, texture, category, function) to "
            f"figure out which object to show. Do NOT name any object directly. "
            f"Do NOT use a conversational tone.\n"
            f"The criteria MUST match at least one toy from the list.\n"
            f"Example: \"I am thinking of something round and red that grows on a tree. "
            f"Which one is it?\"\n"
            f"Return ONLY a JSON object:\n"
            f"{{\"question\": \"<the riddle>\", "
            f"\"criteria\": \"<short criteria phrase, e.g. round red tree fruit>\"}}"
        )

    raw = _gemini_generate(prompt, system="You generate game questions for children. Return JSON only.")
    if raw:
        try:
            print(f"[SceneGame] Gemini raw question response: {raw}")
            obj = _extract_json(raw)
            print(f"[SceneGame] Parsed question JSON: {json.dumps(obj, indent=2) if obj else None}")
            if obj and obj.get('question', '').strip():
                q = obj['question'].strip()
                if mode == "exact":
                    return {
                        'question': q,
                        'target': obj.get('target', target),
                        'criteria': None,
                        'mode': 'exact'
                    }
                else:
                    return {
                        'question': q,
                        'target': None,
                        'criteria': obj.get('criteria', ''),
                        'mode': 'criteria'
                    }
        except Exception as e:
            print(f"[SceneGame] Question generation failed: {e}")

    # Fallback
    return {
        'question': f"Can you find the {target}? Show it to me!",
        'target': target,
        'criteria': None,
        'mode': 'exact'
    }


def _run_gemini_detect_and_look(image_path):
    """Run gemini_analyze_image.py to detect the held object and make the robot look at it.

    Returns the detected label (str) or None.
    """
    script_path = os.path.join(os.path.dirname(BASE_DIR), 'scripts', 'gemini_analyze_image.py')
    if not os.path.exists(script_path):
        print("[SceneGame] gemini_analyze_image.py not found")
        return None
    try:
        proc = subprocess.run(
            [WORKER_PYTHON, script_path, '--image', image_path],
            capture_output=True, text=True, timeout=60
        )
        if proc.returncode != 0:
            print(f"[SceneGame] analyze script error: {proc.stderr}")
            return None
        raw = (proc.stdout or '').strip()
        print(f"[SceneGame] gemini_analyze_image raw: {raw}")
        if raw.startswith('```'):
            raw = raw.strip('`').strip()
            if raw.startswith('json'):
                raw = raw[4:].strip()
        obj = json.loads(raw)
        # obj is typically [{"point": [y, x], "label": "..."}]
        item = obj[0] if isinstance(obj, list) and obj else obj
        if not isinstance(item, dict):
            return None
        label = (item.get('label') or '').strip()
        point = item.get('point')
        print(f"[SceneGame] Detected object: {label}, point: {point}")

        # Make the robot look at the detected object
        if point and len(point) >= 2:
            try:
                # point is [y, x] normalised to 0-1000; convert to pixels (640x480)
                norm_y, norm_x = point[0], point[1]
                pixel_u = norm_x * 640.0 / 1000.0
                pixel_v = norm_y * 480.0 / 1000.0
                tracker = _ensure_human_tracker()
                if tracker:
                    kin = tracker.human_detector.kinematics
                    kin.look_at_pixel([pixel_u, pixel_v], depth=1.0, duration=0, sync=False)
                    print(f"[SceneGame] Robot looking at pixel ({pixel_u:.0f}, {pixel_v:.0f})")
            except Exception as e:
                print(f"[SceneGame] look_at_pixel failed: {e}")

        return label or None
    except Exception as e:
        print(f"[SceneGame] analyze script exec failed: {e}")
        return None


def _check_criteria_match(detected_label, criteria):
    """Use Gemini to check if a detected object matches descriptive criteria.

    Returns (matches: bool, reason: str).
    """
    prompt = (
        f"A child showed an object identified as \"{detected_label}\".\n"
        f"The game asked for: \"{criteria}\".\n"
        f"Does \"{detected_label}\" match the criteria \"{criteria}\"?\n"
        f"Consider color, category, shape, and common knowledge about the object.\n"
        f"Return ONLY a JSON object: {{\"match\": true or false, \"reason\": \"<brief explanation>\"}}"
    )
    raw = _gemini_generate(prompt, system="You validate object matches. Return JSON only.")
    if raw:
        try:
            print(f"[SceneGame] Criteria match Gemini raw: {raw}")
            obj = _extract_json(raw)
            print(f"[SceneGame] Criteria match parsed: {json.dumps(obj, indent=2) if obj else None}")
            if obj:
                return bool(obj.get('match', False)), obj.get('reason', '')
        except Exception as e:
            print(f"[SceneGame] Criteria match failed: {e}")
    # Fallback
    match = criteria.lower() in detected_label.lower() or detected_label.lower() in criteria.lower()
    return match, "fallback string match"


def _get_user_age_and_goals(username):
    """Helper to read child age and learning_goals for a user."""
    user = user_manager.users.get(username, {})
    child_age = 5
    try:
        child_age = int(user.get('age', 5))
    except (ValueError, TypeError):
        child_age = 5
    learning_goals = user.get('learning_goals', '')
    try:
        profile_path = os.path.join(USER_DATA_DIR, username, 'profile.json')
        if os.path.exists(profile_path):
            with open(profile_path, 'r') as f:
                profile = json.load(f)
            learning_goals = profile.get('learning_goals', learning_goals)
    except Exception:
        pass
    return child_age, learning_goals


@app.route('/api/scene_game/new_round', methods=['POST'])
def api_scene_game_new_round():
    """Start a new scene detection round using the configured toy list."""
    if 'username' not in session:
        return jsonify({'success': False, 'error': 'Not logged in'}), 401
    try:
        username = session['username']
        child_age, learning_goals = _get_user_age_and_goals(username)

        # Use the configured toy list
        toy_list = _load_scene_toys()
        if not toy_list:
            toy_list = list(SCENE_GAME_DEFAULT_TOYS)

        # Generate age/goal-appropriate question
        result = _scene_game_generate_question(toy_list, child_age, learning_goals)
        question = result['question']

        # Make the robot ask the question
        try:
            _with_asr_suspended(lambda: tts_helper.speak_story(question, 'en-US'))
        except Exception:
            pass

        # Start human tracking during play mode (non-blocking)
        try:
            tracker = _ensure_human_tracker()
            if tracker:
                person = _pick_recent_person(tracker, timeout_sec=0.5)
                tracker.track(person)
        except Exception as e:
            print(f"HumanTracking start error: {e}")

        # Build item cards from the toy list
        items = []
        img_dir = os.path.join(USER_DATA_DIR, 'activity_images')
        os.makedirs(img_dir, exist_ok=True)
        shuffled = list(toy_list)
        random.shuffle(shuffled)
        for label in shuffled:
            img_url = None
            if image_generator.is_available():
                safe = re.sub(r"[^A-Za-z0-9_-]+", "_", label)
                dest = os.path.join(img_dir, f"{safe}.png")
                if not os.path.exists(dest):
                    path = image_generator.generate_image(
                        prompt=f"{label}, single object on simple background, children's book illustration",
                        output_dir=img_dir,
                        filename_prefix=f"scene_{safe}"
                    )
                    if path and not os.path.exists(dest):
                        try:
                            os.replace(path, dest)
                        except Exception:
                            dest = path
                rel = os.path.relpath(dest, USER_DATA_DIR)
                img_url = f"/images/{rel}"
            items.append({
                'label': label,
                'image_path': img_url
            })

        return jsonify({
            'success': True,
            'question': question,
            'items': items,
            'target': result.get('target'),
            'criteria': result.get('criteria'),
            'mode': result['mode']
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/scene_game/hint', methods=['POST'])
def api_scene_game_hint():
    """Generate an age-appropriate hint for the current round's target object."""
    if 'username' not in session:
        return jsonify({'success': False, 'error': 'Not logged in'}), 401
    data = request.get_json() or {}
    mode = (data.get('mode') or 'exact').strip()
    target = (data.get('target') or '').strip()
    criteria = (data.get('criteria') or '').strip()

    username = session['username']
    child_age, learning_goals = _get_user_age_and_goals(username)

    # Build the subject of the hint
    if mode == 'exact' and target:
        subject = f'the object "{target}"'
    elif criteria:
        subject = f'an object matching "{criteria}"'
    else:
        return jsonify({'success': False, 'error': 'No active round'})

    prompt = (
        f"A {child_age}-year-old child is playing an object detection game and needs a hint "
        f"to find {subject} from these toys: {', '.join(_load_scene_toys())}.\n"
        f"Generate ONE short hint that helps the child identify the object.\n"
        f"Age guidelines:\n"
        f"- Ages 2-3: very simple, point out one obvious feature (color or shape). "
        f"Example: \"It is yellow!\"\n"
        f"- Ages 4-6: describe 1-2 properties (color, shape, what it does). "
        f"Example: \"It is red and round, and you can eat it!\"\n"
        f"- Ages 7+: give a clue that requires reasoning but is easier than the original riddle. "
        f"Example: \"You might put this in a salad. It grows on a vine.\"\n"
        f"Return ONLY a JSON object: {{\"hint\": \"<the hint sentence>\"}}"
    )
    raw = _gemini_generate(prompt, system="You generate game hints for children. Return JSON only.")
    if raw:
        try:
            print(f"[SceneGame] Gemini raw hint response: {raw}")
            obj = _extract_json(raw)
            print(f"[SceneGame] Parsed hint JSON: {json.dumps(obj, indent=2) if obj else None}")
            hint = (obj.get('hint', '') if obj else '').strip()
            if hint:
                print(f"[SceneGame] Hint generated: {hint}")
                try:
                    _with_asr_suspended(lambda: tts_helper.speak(hint))
                except Exception:
                    pass
                return jsonify({'success': True, 'hint': hint})
        except Exception as e:
            print(f"[SceneGame] Hint generation failed: {e}")

    hint = f"Look carefully at the toys! Think about {target or criteria}."
    try:
        _with_asr_suspended(lambda: tts_helper.speak(hint))
    except Exception:
        pass
    return jsonify({'success': True, 'hint': hint})

@app.route('/api/scene_game/answer', methods=['POST'])
def api_scene_game_answer():
    """Validate an answer by capturing a camera frame and using gemini_analyze_image.py.

    Step 1: Capture frame, run gemini_analyze_image.py to detect the object + robot looks at it.
    Step 2: Compare detected label against target (exact) or criteria (ages 4+).
    """
    if 'username' not in session:
        return jsonify({'success': False, 'error': 'Not logged in'}), 401
    if cv2 is None:
        return jsonify({'success': False, 'error': 'OpenCV not available'}), 500

    data = request.get_json() or {}
    answer_mode = (data.get('mode') or 'exact').strip()
    target = (data.get('target') or '').strip()
    criteria = (data.get('criteria') or '').strip()

    username = session['username']

    # Capture frame from camera
    frame = _get_ros_frame()
    if frame is None:
        return jsonify({'success': False, 'error': 'Camera read failed'}), 500

    import datetime
    cap_dir = os.path.join(USER_DATA_DIR, username, 'captured_scenes')
    os.makedirs(cap_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    fpath = os.path.join(cap_dir, f'scene_answer_{ts}.jpg')
    cv2.imwrite(fpath, frame)

    # Step 1: detect object + robot looks at it
    detected = _run_gemini_detect_and_look(fpath)
    if not detected:
        try:
            tts_helper.speak("I couldn't see clearly. Can you show me again?")
        except Exception:
            pass
        return jsonify({'success': True, 'correct': None, 'detected': None,
                        'error': 'Vision analysis failed'})

    print(f"[SceneGame] Detected object: {detected}")

    # Step 2: match against target or criteria
    if answer_mode == 'exact':
        # Ages 2-3: simple name comparison
        correct = detected.lower().strip() == target.lower().strip()
        # Also accept if one contains the other (e.g. "red apple" vs "apple")
        if not correct:
            correct = (target.lower() in detected.lower()) or (detected.lower() in target.lower())
        reason = ''
        try:
            if correct:
                tts_helper.speak(f"Great job! That's the {target}!")
            else:
                tts_helper.speak(f"I see a {detected}, but I asked for the {target}. Try again!")
        except Exception:
            pass
    else:
        # Ages 4+: criteria-based — use LLM to check match
        if not criteria:
            return jsonify({'success': False, 'error': 'No criteria provided'}), 400
        correct, reason = _check_criteria_match(detected, criteria)
        try:
            if correct:
                tts_helper.speak(f"Great job! I can see a {detected}. That's right!")
            else:
                tts_helper.speak(f"I see a {detected}, but that doesn't match. Try again!")
        except Exception:
            pass

    return jsonify({
        'success': True,
        'correct': correct,
        'detected': detected,
        'reason': reason
    })

@app.route('/api/human_tracking/untrack', methods=['POST'])
def api_human_tracking_untrack():
    if 'username' not in session:
        return jsonify({'success': False, 'error': 'Not logged in'}), 401
    try:
        tracker = _ensure_human_tracker()
        if tracker:
            tracker.untrack()
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/human_tracking/start', methods=['POST'])
def api_human_tracking_start():
    print("api_human_tracking_start")
    if 'username' not in session:
        return jsonify({'success': False, 'error': 'Not logged in'}), 401
    try:
        print("api_human_tracking_start try")
        tracker = _ensure_human_tracker()
        if not tracker:
            return jsonify({'success': False, 'error': 'HumanTracking unavailable'}), 500
        data = request.get_json() or {}
        pid = data.get('person_id')
        if pid is not None:
            tracker.track_by_id(pid)
        else:
            person = _pick_recent_person(tracker, timeout_sec=0.5)
            tracker.track(person)
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/human_tracking/status', methods=['GET'])
def api_human_tracking_status():
    if 'username' not in session:
        return jsonify({'success': False, 'error': 'Not logged in'}), 401
    try:
        tracker = _ensure_human_tracker()
        running = bool(tracker and getattr(tracker, 'should_track', False))
        current_id = None
        person_present = None
        try:
            if tracker:
                if running:
                    current_id = tracker.get_current_person_id()
                # approximate presence via private helper if available
                if hasattr(tracker, '_presence_now'):
                    person_present = bool(tracker._presence_now())
        except Exception:
            current_id = None
        return jsonify({'success': True, 'running': running, 'person_id': current_id, 'person_present': person_present})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/camera_frame')
def api_camera_frame():
    """Return a single JPEG frame from the robot camera (best effort)."""
    if 'username' not in session:
        return jsonify({'success': False, 'error': 'Not logged in'}), 401
    try:
        if cv2 is None:
            return jsonify({'success': False, 'error': 'OpenCV not available'}), 500
        frame = _get_ros_frame()
        ok = True if frame is not None else False
        if not ok or frame is None:
            return jsonify({'success': False, 'error': 'Camera read failed'}), 500
        # Encode JPEG
        ok, buf = cv2.imencode('.jpg', frame)
        if not ok:
            return jsonify({'success': False, 'error': 'JPEG encode failed'}), 500
        # from flask import make_response
        resp = make_response(buf.tobytes())
        resp.headers['Content-Type'] = 'image/jpeg'
        resp.headers['Cache-Control'] = 'no-store'
        return resp
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/camera_capture', methods=['POST'])
def api_camera_capture():
    """Capture and persist a frame to the user's directory; return served URL."""
    if 'username' not in session:
        return jsonify({'success': False, 'error': 'Not logged in'}), 401
    if cv2 is None:
        return jsonify({'success': False, 'error': 'OpenCV not available'}), 500
    try:
        username = session['username']
        frame = _get_ros_frame()
        if frame is None:
            return jsonify({'success': False, 'error': 'Camera read failed'}), 500
        # Save JPEG under user directory
        import datetime
        user_cap_dir = os.path.join(USER_DATA_DIR, username, 'captured_scenes')
        os.makedirs(user_cap_dir, exist_ok=True)
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        fname = f'scene_{ts}.jpg'
        fpath = os.path.join(user_cap_dir, fname)
        ok = cv2.imwrite(fpath, frame)
        if not ok:
            return jsonify({'success': False, 'error': 'Failed to save image'}), 500
        rel = os.path.relpath(fpath, USER_DATA_DIR)
        # Optional target from request body
        target = None
        try:
            payload = request.get_json(silent=True) or {}
            t = payload.get('target')
            if isinstance(t, str) and t.strip():
                target = t.strip()
        except Exception:
            target = None
        # Call external Gemini script and return its raw JSON/text
        analysis = None
        try:
            script_path = os.path.join(os.path.dirname(BASE_DIR), 'scripts', 'gemini_analyze_image.py')
            if os.path.exists(script_path):
                cmd = [WORKER_PYTHON, script_path, '--image', fpath]

                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                if proc.returncode == 0:
                    analysis = (proc.stdout or '').strip()
                else:
                    print(f"Gemini script error: {proc.stderr}")
        except Exception as _e:
            print(f"Gemini script exec failed: {_e}")
        # Extract only label from returned JSON/text
        detected_label = None
        if isinstance(analysis, str) and analysis:
            try:
                raw = analysis.strip()
                if raw.startswith('```'):
                    raw = raw.strip('`')
                obj = None
                try:
                    obj = json.loads(raw)
                except Exception:
                    l = raw.find('[')
                    r = raw.rfind(']')
                    if l != -1 and r != -1 and r > l:
                        obj = json.loads(raw[l:r+1])
                if isinstance(obj, list) and obj:
                    item = obj[0]
                    if isinstance(item, dict):
                        lbl = item.get('label')
                        if isinstance(lbl, str) and lbl.strip():
                            detected_label = lbl.strip()
                elif isinstance(obj, dict):
                    lbl = obj.get('label')
                    if isinstance(lbl, str) and lbl.strip():
                        detected_label = lbl.strip()
            except Exception:
                detected_label = None
        # Speak feedback based on comparison
        try:
            if target and detected_label:
                if target.lower() in detected_label.lower():
                    _with_asr_suspended(lambda: tts_helper.speak("That's correct!"))
                else:
                    _with_asr_suspended(lambda: tts_helper.speak("No, try again."))
        except Exception:
            pass
        found = None
        if target and detected_label:
            try:
                found = target.lower() in detected_label.lower()
            except Exception:
                found = None
        return jsonify({'success': True, 'image_path': f"/images/{rel}", 'label': detected_label, 'target': target, 'found': found})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/recovery/generate_question', methods=['POST'])
def api_recovery_generate_question():
    """Capture a frame and generate a recovery question using Gemini vision."""
    if 'username' not in session:
        return jsonify({'success': False, 'error': 'Not logged in'}), 401
    if cv2 is None:
        return jsonify({'success': False, 'error': 'OpenCV not available'}), 500
    try:
        username = session['username']
        payload = request.get_json(silent=True) or {}
        mode = payload.get('mode', 'toy')  # 'toy' or 'child'
        child_name = payload.get('child_name', '')

        frame = _get_ros_frame()
        if frame is None:
            return jsonify({'success': False, 'error': 'Camera read failed'}), 500

        # Save frame temporarily
        import datetime
        user_cap_dir = os.path.join(USER_DATA_DIR, username, 'captured_scenes')
        os.makedirs(user_cap_dir, exist_ok=True)
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        fname = f'recovery_{mode}_{ts}.jpg'
        fpath = os.path.join(user_cap_dir, fname)
        cv2.imwrite(fpath, frame)

        # Call Gemini recovery question script
        script_path = os.path.join(os.path.dirname(BASE_DIR), 'scripts', 'gemini_recovery_question.py')
        if not os.path.exists(script_path):
            return jsonify({'success': False, 'error': 'Recovery question script not found'}), 500

        # Resolve child age from user profile
        child_age = 5
        user_data = user_manager.users.get(username, {})
        try:
            child_age = int(user_data.get('age', 5))
        except (ValueError, TypeError):
            child_age = 5

        cmd = [WORKER_PYTHON, script_path, '--image', fpath, '--mode', mode,
               '--child-age', str(child_age)]
        if child_name:
            cmd += ['--child-name', child_name]

        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if proc.returncode != 0:
            print(f"[Recovery question] script error: {proc.stderr}")
            return jsonify({'success': False, 'error': 'Gemini analysis failed'}), 500

        raw = (proc.stdout or '').strip()
        detected_object = None
        try:
            result = json.loads(raw)
            text = result.get('text', '')
            detected_object = result.get('object', None)
        except Exception:
            text = raw

        if not text:
            text = f"Hey {child_name}!" if child_name else "Hey there!"

        return jsonify({'success': True, 'text': text, 'mode': mode, 'object': detected_object})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/conversation/wait_for_turn', methods=['POST'])
def api_conversation_wait_for_turn():
    """Listen for child's speech until red card is shown, then generate follow-up.

    Expects JSON:
      theme, robot_said, child_name, child_age,
      followup_number, total_followups, history
    Returns:
      child_said, followup_text
    """
    if 'username' not in session:
        return jsonify({'success': False, 'error': 'Not logged in'}), 401
    try:
        payload = request.get_json() or {}
        theme = payload.get('theme', 'greeting')
        robot_said = payload.get('robot_said', '')
        child_name = payload.get('child_name', '')
        child_age = int(payload.get('child_age', 5))
        followup_number = int(payload.get('followup_number', 1))
        total_followups = int(payload.get('total_followups', 1))
        history = payload.get('history', [])

        # Wait for robot to finish speaking before opening the mic
        _wait_until_robot_silent()
        # Enable face/sound tracking so robot follows the child
        _enable_face_tracking()
        # Signal child that mic is on — show_tablet gesture
        _signal_child_can_speak()

        # Phase 1: Listen for child's speech while watching for red card
        collected_speech = []
        red_card_seen = False
        _activity_stop_event.clear()

        print(f"[Conversation] Waiting for child's turn (theme={theme}, followup={followup_number}/{total_followups})")

        # Red card watcher runs in parallel with ASR
        red_card_event = ThreadEvent()
        def _watch_red_card():
            while not red_card_event.is_set() and not _activity_stop_event.is_set():
                f = _get_ros_frame()
                if f is not None and _detect_red_card(f):
                    red_card_event.set()
                    return
                time.sleep(0.5)
        red_card_thread = Thread(target=_watch_red_card, daemon=True)
        red_card_thread.start()

        # Collect speech until red card is detected
        max_rounds = 20
        for round_num in range(max_rounds):
            if _activity_stop_event.is_set() or red_card_event.is_set():
                break
            heard = (_whisper_recognize_once() or '').strip()
            if heard:
                collected_speech.append(heard)
                print(f"[Conversation] Heard: {heard}")
            if red_card_event.is_set():
                break

        # Signal watcher to stop and wait for it
        red_card_seen = red_card_event.is_set()
        red_card_event.set()
        red_card_thread.join(timeout=1.0)

        child_said = ' '.join(collected_speech)
        if not child_said:
            child_said = ''
        print(f"[Conversation] Child said: '{child_said}' (red_card={red_card_seen}) -> generating follow-up immediately")

        # Phase 2: Generate follow-up response
        is_closing = payload.get('is_closing', False)
        followup_text = _generate_conversation_followup(
            theme=theme,
            robot_said=robot_said,
            child_said=child_said,
            child_name=child_name,
            child_age=child_age,
            followup_number=followup_number,
            total_followups=total_followups,
            history=history,
            is_closing=is_closing
        )

        if not followup_text:
            if is_closing:
                followup_text = f"I really enjoyed hearing about that, {child_name}!" if child_name else "I really enjoyed hearing about that!"
            elif child_said:
                followup_text = f"That's interesting, {child_name}!" if child_name else "That's interesting!"
            else:
                followup_text = f"It's okay {child_name}, take your time!" if child_name else "It's okay, take your time!"

        return jsonify({
            'success': True,
            'child_said': child_said,
            'followup_text': followup_text,
            'red_card_seen': red_card_seen
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/conversation/check_red_card', methods=['GET'])
def api_check_red_card():
    """Quick check if red card is currently visible in camera."""
    try:
        frame = _get_ros_frame()
        detected = _detect_red_card(frame) if frame is not None else False
        return jsonify({'success': True, 'detected': detected})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route("/builder")
def builder_page():
    """DIY activity builder page"""
    if 'username' not in session:
        return redirect(url_for('index'))
    username = session['username']
    user = user_manager.users.get(username)
    return render_template("diy_builder.html", logged_in=True, user=user)

@app.route("/conversation_builder")
def conversation_builder_page():
    """Conversation flow builder page"""
    if 'username' not in session:
        return redirect(url_for('index'))
    username = session['username']
    user = user_manager.users.get(username)
    return render_template("conversation_builder.html", logged_in=True, user=user)

@app.route("/select_toy")
def select_toy_page():
    """Toy selection page before DIY builder"""
    if 'username' not in session:
        return redirect(url_for('index'))
    username = session['username']
    user = user_manager.users.get(username)
    return render_template("select_toy.html", logged_in=True, user=user)

@app.route('/api/toys', methods=['GET'])
def api_get_toys():
    if 'username' not in session:
        return jsonify({'success': False, 'error': 'Not logged in'}), 401
    username = session['username']
    # defaults
    defaults = [
        {'name': 'Car'}, {'name': 'Doll'}, {'name': 'Puzzle'}, {'name': 'Blocks'},
        {'name': 'Ball'}, {'name': 'Robot'}, {'name': 'Book'}
    ]
    # user toys
    user_dir = os.path.join(USER_DATA_DIR, username)
    os.makedirs(user_dir, exist_ok=True)
    toys_path = os.path.join(user_dir, 'toys.json')
    user_toys = []
    try:
        if os.path.exists(toys_path):
            with open(toys_path, 'r') as f:
                user_toys = json.load(f)
    except Exception:
        user_toys = []
    # merge with uniqueness by name (case-insensitive)
    seen = set()
    toys = []
    for t in defaults + user_toys:
        name = (t.get('name') or '').strip()
        key = name.lower()
        if name and key not in seen:
            seen.add(key)
            toys.append({'name': name, 'image': t.get('image')})
    return jsonify({'success': True, 'toys': toys})

@app.route('/api/toys/add', methods=['POST'])
def api_add_toy():
    if 'username' not in session:
        return jsonify({'success': False, 'error': 'Not logged in'}), 401
    data = request.get_json() or {}
    name = (data.get('name') or '').strip()
    image = (data.get('image') or '').strip()
    if not name:
        return jsonify({'success': False, 'error': 'Toy name is required'}), 400
    username = session['username']
    user_dir = os.path.join(USER_DATA_DIR, username)
    os.makedirs(user_dir, exist_ok=True)
    toys_path = os.path.join(user_dir, 'toys.json')
    toys = []
    try:
        if os.path.exists(toys_path):
            with open(toys_path, 'r') as f:
                toys = json.load(f)
    except Exception:
        toys = []
    # prevent duplicates
    if any((t.get('name') or '').strip().lower() == name.lower() for t in toys):
        return jsonify({'success': True, 'added': False})
    toys.append({'name': name, 'image': image or None})
    try:
        with open(toys_path, 'w') as f:
            json.dump(toys, f, indent=2)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
    return jsonify({'success': True, 'added': True})

@app.route("/api/activity/prepare", methods=["POST"])
def api_activity_prepare():
    """Prepare activity: server-side expansion like generating sentences/images plan"""
    username = session.get('username')
    if not username:
        return jsonify({"success": False, "error": "Not logged in"}), 401
    data = request.get_json() or {}
    blocks = data.get("blocks", [])
    loop_count = int(data.get("loop", 1) or 1)
    # If a Loop block exists, prefer its count
    for blk in blocks:
        if blk.get("type") == "loop":
            try:
                loop_count = max(1, int(blk.get("count", loop_count)))
            except Exception:
                pass
            break

    # Generate images for image blocks if SD pipeline is available
    prepared_blocks = []
    activity_id = None
    images_dir_rel = None
    images_url_dir = None
    # Save DIY images under shared src/user_data/activity_images
    shared_images_dir = os.path.join(USER_DATA_DIR, 'activity_images')
    if image_generator.is_available():
        try:
            os.makedirs(shared_images_dir, exist_ok=True)
            images_dir_rel = os.path.relpath(shared_images_dir, USER_DATA_DIR)
            images_url_dir = f"/images/{images_dir_rel}"
        except Exception:
            images_dir_rel = None
            images_url_dir = None

    # Determine subjects for looped gameplay (e.g., animals)
    subjects = []
    known_animals = [
        'tiger','rabbit','lion','giraffe','elephant','zebra','monkey','panda','bear','fox',
        'hippo','hippopotamus','kangaroo','koala','leopard','cheetah','wolf','deer','camel','rhino',
        'crocodile','alligator','horse','sheep','goat','cow','dog','cat','penguin','dolphin'
    ]
    # Try to infer a seed subject from first image block
    seed_subject = None
    for blk in blocks:
        if blk.get("type") == "image":
            src_text = (blk.get("src") or "").lower()
            # exact or contained match
            for animal in known_animals:
                if src_text == animal or (animal in src_text and len(animal) > 3):
                    seed_subject = animal
                    break
            if not seed_subject and ("animal" in src_text):
                seed_subject = random.choice(known_animals)
            if seed_subject:
                break
    if loop_count > 1 and seed_subject:
        pool = [a for a in known_animals if a != seed_subject]
        random.shuffle(pool)
        needed = loop_count - 1
        subjects = [seed_subject] + pool[:needed]
    elif seed_subject:
        subjects = [seed_subject]
    else:
        subjects = []

    expanded_images = []

    for i, block in enumerate(blocks):
        b = dict(block)
        if b.get("type") == "image":
            src = b.get("src", "")
            # If src looks like a URL, keep it as is
            if src.startswith("http://") or src.startswith("https://"):
                b["image_path"] = src
            else:
                # Treat as prompt/description and generate an image if possible
                if image_generator.is_available():
                    try:
                        # Sanitize filename from prompt (e.g., tiger, rabbit)
                        def generate_named_image(name_hint: str) -> Optional[str]:
                            safe = re.sub(r"[^A-Za-z0-9_-]+", "_", (name_hint or src).strip().lower()) or "image"
                            target_path = os.path.join(shared_images_dir, f"{safe}.png")
                            if os.path.exists(target_path):
                                suffix = 2
                                while os.path.exists(os.path.join(shared_images_dir, f"{safe}_{suffix}.png")):
                                    suffix += 1
                                target_path = os.path.join(shared_images_dir, f"{safe}_{suffix}.png")
                            img_path_fs = image_generator.generate_image(
                                prompt=(name_hint or src or "children illustration"),
                                output_dir=shared_images_dir,
                                filename_prefix=f"activity_scene_{i:03d}"
                            )
                            if not img_path_fs:
                                return None
                            if not os.path.isabs(img_path_fs):
                                img_path_fs = os.path.abspath(img_path_fs)
                            try:
                                os.replace(img_path_fs, target_path)
                            except Exception:
                                target_path = img_path_fs
                            rel = os.path.relpath(target_path, USER_DATA_DIR)
                            return f"/images/{rel}"

                        # Generate first iteration image (seed or src)
                        first_subject = subjects[0] if subjects else src
                        first_img = generate_named_image(first_subject)
                        b["image_path"] = first_img

                        # Generate additional images for further loops
                        if len(subjects) > 1:
                            for idx, subject in enumerate(subjects[1:], start=2):
                                img_url = generate_named_image(subject)
                                if img_url:
                                    expanded_images.append({
                                        "iteration": idx,
                                        "subject": subject,
                                        "image_path": img_url
                                    })
                        # If no subjects detected but loop > 1, just duplicate prompt with suffixes
                        elif loop_count > 1 and not subjects:
                            for idx in range(2, loop_count + 1):
                                img_url = generate_named_image(f"{src}_{idx}")
                                if img_url:
                                    expanded_images.append({
                                        "iteration": idx,
                                        "subject": f"{src}_{idx}",
                                        "image_path": img_url
                                    })
                        
                        # Optional: enrich speech blocks with subject placeholders not implemented in UI yet
                        
                        
                    except Exception as e:
                        b["image_path"] = None
                        print(f"Error generating activity image: {str(e)}")
                else:
                    b["image_path"] = None
        prepared_blocks.append(b)

    return jsonify({
        "success": True,
        "plan": {
            "blocks": prepared_blocks,
            "loop": loop_count,
            "activity_id": None,
            "images_available": image_generator.is_available(),
            "images_dir_rel": images_dir_rel,
            "images_url_dir": images_url_dir,
            "subjects": subjects,
            "expanded_images": expanded_images
        }
    })

@app.route("/api/activity/test", methods=["POST"])
def api_activity_test():
    """Execute the activity once for testing on the robot"""
    username = session.get('username')
    if not username:
        return jsonify({"success": False, "error": "Not logged in"}), 401
    payload = request.get_json() or {}
    blocks = payload.get("blocks", [])
    loop_count = int(payload.get("loop", 1) or 1)
    print(f"[DIY test] blocks={blocks}, loop_count={loop_count}")
    # Reset and run inline (test mode): stop on completion or if stop requested
    _activity_stop_event.clear()
    global _asr_enabled
    _asr_enabled = True
    # Prefer Loop block's count if present
    for blk in blocks:
        if blk.get("type") == "loop":
            try:
                loop_count = max(1, int(blk.get("count", loop_count)))
            except Exception:
                pass
            break
    try:
        _execute_activity(blocks, loop_count, username=username)
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

def _execute_activity(blocks, loop_count, require_confirm=False, username=''):
    """Internal routine to execute a block list with loop count."""
    # Remove loop control blocks from the execution list
    exec_blocks = [b for b in blocks if b.get("type") != "loop"]
    kin = getattr(tts_helper, 'kinematics', None)
    # If there is a logic block with multiple recognize conditions, run continuously until stopped
    continuous_mode = _has_parallel_recognizers(exec_blocks)
    if continuous_mode:
        # Announce the listening state once if any speech recognition is included
        if any(b.get('type') == 'logic' for b in exec_blocks):
            _announce_wait_once()
        while not _activity_stop_event.is_set():
            for block in exec_blocks:
                if _activity_stop_event.is_set():
                    break
                btype = block.get("type")
                if btype == "logic":
                    cond_blocks = (block.get('cond') or [])
                    then_blocks = (block.get('then') or [])

                    def exec_then_block(tblock):
                        ttype = tblock.get('type')
                        if ttype == 'speech':
                            txt = tblock.get('text', '')
                            if txt:
                                _with_asr_suspended(lambda: tts_helper.speak_story(txt, 'en-US'))
                        elif ttype == 'praise':
                            _with_asr_suspended(lambda: tts_helper.speak('Great job!'))
                        elif ttype == 'gesture':
                            g = (tblock.get('name') or tblock.get('gesture') or '').strip()
                            if g and not g.startswith("QT/"):
                                g = "QT/" + g
                            if g and ROS_AVAILABLE:
                                try:
                                    from qt_gesture_controller.srv import gesture_play
                                    ges_proxy = rospy.ServiceProxy('/qt_robot/gesture/play', gesture_play)
                                    ges_proxy.wait_for_service(timeout=2.0)
                                    ges_proxy(g, 1.0)
                                except Exception as e:
                                    print(f"[DIY gesture] error: {e}")
                        elif ttype == 'emotion':
                            emo = (tblock.get('name') or '').strip()
                            if emo and ROS_AVAILABLE:
                                try:
                                    from qt_robot_interface.srv import emotion_show
                                    emo_proxy = rospy.ServiceProxy('/qt_robot/emotion/show', emotion_show)
                                    emo_proxy.wait_for_service(timeout=2.0)
                                    emo_proxy(emo)
                                except Exception as e:
                                    print(f"[DIY emotion] error: {e}")

                    # Launch parallel recognizers that keep firing when matches occur
                    threads = []
                    pair_count = min(len(cond_blocks), len(then_blocks))
                    for i in range(pair_count):
                        c = cond_blocks[i]
                        t = then_blocks[i]
                        if c.get('type') == 'recognize':
                            target = (c.get('target') or 'speech').lower()
                            expected = (c.get('value') or '').strip().lower()
                            if target == 'speech' and expected:
                                def worker(exp=expected, tblock=t):
                                    while not _activity_stop_event.is_set():
                                        try:
                                            # avoid overlap with TTS
                                            guard_start = time.time()
                                            while getattr(tts_helper, 'is_speaking', lambda: False)() and time.time() - guard_start < 10:
                                                if _activity_stop_event.is_set():
                                                    return
                                                time.sleep(0.05)
                                            text = _whisper_recognize_streaming()
                                            heard_raw = (text or '').strip().lower()
                                            import re as _re
                                            heard = _re.sub(r"[^a-z0-9\s]", "", heard_raw)
                                            print(f"[Logic ASR] expected='{exp}' heard='{heard_raw}' -> norm='{heard}'")
                                            if heard and exp not in heard:
                                                fuzzy = _fuzzy_canonicalize_heard(exp, heard)
                                                if fuzzy:
                                                    print(f"[Logic ASR] fuzzy corrected '{heard_raw}' -> '{fuzzy}'")
                                                    heard = fuzzy
                                                else:
                                                    corrected = _llm_canonicalize_heard(exp, heard, context="DIY logic recognize")
                                                    if corrected:
                                                        print(f"[Logic ASR] corrected '{heard_raw}' -> '{corrected}'")
                                                        heard = corrected.lower()
                                            if heard and exp in heard:
                                                exec_then_block(tblock)
                                        except Exception as e:
                                            print(f"ASR error: {e}")
                                            time.sleep(0.2)
                                th = Thread(target=worker, daemon=True)
                                th.start()
                                threads.append(th)
                    # Keep the recognizers alive while not stopped
                    while not _activity_stop_event.is_set():
                        time.sleep(0.1)
                    # Stop requested: threads will exit by checking the event
                    for th in threads:
                        try:
                            th.join(timeout=0.2)
                        except Exception:
                            pass
                    break
            break
        return

    # Execute steps sequentially; within each step, gesture + emotion fire in
    # parallel (non-blocking), then speech plays (blocking, waits for TTS to finish).
    global _step_current_index, _step_total_count, _step_current_label, _step_next_label, _step_waiting
    if require_confirm:
        _step_total_count = len(exec_blocks)
        _step_current_index = 0
        _step_waiting = False
    for _ in range(max(1, loop_count)):
        for block_idx, block in enumerate(exec_blocks):
            btype = block.get("type")

            if btype == "step":
                # --- Step block: gesture, emotion, and speech in one unit ---
                gesture = (block.get("gesture") or "").strip()
                emotion = (block.get("emotion") or "").strip()
                text = (block.get("text") or "").strip()

                # If block uses camera, generate text dynamically via Gemini
                use_camera = (block.get("useCamera") or "").strip()
                toy_detected_in_initial = False
                if use_camera:
                    generated = _generate_recovery_question(use_camera, username=username)
                    if generated and generated.get('text'):
                        text = generated['text']
                        if use_camera == 'toy' and generated.get('object'):
                            toy_detected_in_initial = True
                        print(f"[DIY step] camera-generated text: {text} (object={generated.get('object')})")

                # Gesture: add QT/ prefix if missing
                if gesture and not gesture.startswith("QT/"):
                    gesture = "QT/" + gesture

                # Fire gesture + emotion in parallel (they are non-blocking ROS calls)
                threads = []
                # Speech plays before gesture/emotion have started (blocks until done)
                if text:
                    _with_asr_suspended(lambda: tts_helper.speak_story(text, "en-US"))
                if gesture and ROS_AVAILABLE:
                    def _play_gesture(g=gesture):
                        try:
                            from qt_gesture_controller.srv import gesture_play
                            ges_proxy = rospy.ServiceProxy('/qt_robot/gesture/play', gesture_play)
                            rospy.wait_for_service('/qt_robot/gesture/play', timeout=5.0)
                            result = ges_proxy(g, 1.0)
                            print(f"[DIY step] gesture '{g}' result={result}")
                        except Exception as e:
                            print(f"[DIY step] gesture error: {e}")
                    t = Thread(target=_play_gesture, daemon=True)
                    t.start()
                    threads.append(t)

                if emotion and ROS_AVAILABLE:
                    def _show_emotion(emo=emotion):
                        try:
                            from qt_robot_interface.srv import emotion_show
                            emo_proxy = rospy.ServiceProxy('/qt_robot/emotion/show', emotion_show)
                            rospy.wait_for_service('/qt_robot/emotion/show', timeout=5.0)
                            result = emo_proxy(emo)
                            print(f"[DIY step] emotion '{emo}' result={result}")
                        except Exception as e:
                            print(f"[DIY step] emotion error: {e}")
                    t = Thread(target=_show_emotion, daemon=True)
                    t.start()
                    threads.append(t)

                # Wait for gesture/emotion threads to finish before moving to next step
                for t in threads:
                    t.join(timeout=10.0)

                # Conversation follow-up loop: listen for child + red card, generate follow-up
                num_followups = int(block.get("followups", 0) or 0)
                block_theme = (block.get("theme") or "").strip()
                if block_theme:
                    # Enable face/sound tracking so robot follows the child throughout
                    _enable_face_tracking()
                    # Wait for robot's opening speech to fully finish before starting to listen
                    _wait_until_robot_silent()
                    child_name = ''
                    child_age = 5
                    if username:
                        u = user_manager.users.get(username, {})
                        child_name = u.get('display_name') or username
                        try:
                            child_age = int(u.get('age', 5))
                        except (ValueError, TypeError):
                            child_age = 5
                    conv_history = []
                    last_robot_said = text
                    # Total rounds = follow-up questions + 1 final closing listen
                    total_rounds = num_followups + 1
                    for fu_num in range(1, total_rounds + 1):
                        if _activity_stop_event.is_set():
                            break
                        is_closing = (fu_num == total_rounds)
                        # Wait for robot to finish speaking before opening the mic
                        _wait_until_robot_silent()
                        # Re-enable face tracking for each round
                        _enable_face_tracking()
                        # Signal child that mic is on — show_tablet gesture
                        _signal_child_can_speak()
                        label = "closing" if is_closing else f"follow-up {fu_num}/{num_followups}"
                        print(f"[Conversation] {label}: listening for child (theme={block_theme})")

                        # Red card watcher runs in parallel with ASR
                        red_card_event = ThreadEvent()
                        def _watch_red_card(ev=red_card_event):
                            while not ev.is_set() and not _activity_stop_event.is_set():
                                f = _get_ros_frame()
                                if f is not None and _detect_red_card(f):
                                    ev.set()
                                    return
                                time.sleep(0.5)
                        red_card_thread = Thread(target=_watch_red_card, daemon=True)
                        red_card_thread.start()

                        # Collect speech until red card is detected
                        collected_speech = []
                        for listen_round in range(20):
                            if _activity_stop_event.is_set() or red_card_event.is_set():
                                break
                            heard = (_whisper_recognize_once() or '').strip()
                            if heard:
                                collected_speech.append(heard)
                                print(f"[Conversation] Heard: {heard}")
                            if red_card_event.is_set():
                                break

                        # Signal watcher to stop and wait for it
                        red_card_event.set()
                        red_card_thread.join(timeout=1.0)

                        child_said = ' '.join(collected_speech)
                        print(f"[Conversation] Child said: '{child_said}' -> generating {'closing' if is_closing else 'follow-up'}")

                        # Generate response (follow-up question or closing comment)
                        fu_text = _generate_conversation_followup(
                            theme=block_theme,
                            robot_said=last_robot_said,
                            child_said=child_said,
                            child_name=child_name,
                            child_age=child_age,
                            followup_number=fu_num,
                            total_followups=num_followups,
                            history=conv_history,
                            is_closing=is_closing
                        )
                        if not fu_text:
                            if is_closing:
                                fu_text = f"I really enjoyed hearing about that, {child_name}!" if child_name else "I really enjoyed hearing about that!"
                            else:
                                fu_text = f"That's interesting, {child_name}!" if child_name else "That's interesting!"
                        conv_history.append({'robot': last_robot_said, 'child': child_said})
                        _with_asr_suspended(lambda: tts_helper.speak_story(fu_text, "en-US"))
                        last_robot_said = fu_text
                        print(f"[Conversation] {label}: robot said: {fu_text}")

                # Toy watching loop: if no toy was seen initially, keep watching
                if use_camera == 'toy' and not toy_detected_in_initial:
                    child_name = ''
                    if username:
                        u = user_manager.users.get(username, {})
                        child_name = u.get('display_name') or username
                    print(f"[DIY step] Toy watching: waiting for child to show a toy...")
                    watch_count = 0
                    while not _activity_stop_event.is_set() and watch_count < 15:
                        time.sleep(3)
                        watch_count += 1
                        if _activity_stop_event.is_set():
                            break
                        followup = _generate_recovery_question('toy_followup', username=username)
                        if followup and followup.get('object'):
                            print(f"[DIY step] Toy detected: {followup['object']}")
                            fu_text = followup.get('text', '')
                            if fu_text:
                                _with_asr_suspended(lambda: tts_helper.speak_story(fu_text, "en-US"))
                            break

                # Toy step "well done" complement before moving on
                if use_camera == 'toy':
                    child_name = ''
                    if username:
                        u = user_manager.users.get(username, {})
                        child_name = u.get('display_name') or username
                    wd_text = f"Well done {child_name}!" if child_name else "Well done!"
                    _with_asr_suspended(lambda: tts_helper.speak_story(wd_text, "en-US"))

                # If require_confirm and more steps remain, pause for therapist
                if require_confirm and block_idx < len(exec_blocks) - 1:
                    _step_current_index = block_idx
                    next_blk = exec_blocks[block_idx + 1]
                    _step_current_label = block.get("component", block.get("type", "step"))
                    _step_next_label = next_blk.get("component", next_blk.get("type", "step"))
                    _step_waiting = True
                    _step_confirm_event.clear()
                    print(f"[DIY run] Waiting for therapist confirmation after step {block_idx + 1}/{len(exec_blocks)}")
                    # Block until therapist confirms or activity is stopped
                    while not _step_confirm_event.is_set() and not _activity_stop_event.is_set():
                        _step_confirm_event.wait(timeout=0.5)
                    _step_waiting = False
                    if _activity_stop_event.is_set():
                        print(f"[DIY run] Stopped by therapist after step {block_idx + 1}")
                        _step_current_index = -1
                        return

            # Legacy block types kept for backward compatibility with saved activities
            elif btype == "speech":
                text = block.get("text", "")
                if text:
                    _with_asr_suspended(lambda: tts_helper.speak_story(text, "en-US"))
            elif btype == "gesture":
                name = (block.get("name") or block.get("gesture") or "").strip()
                if name and not name.startswith("QT/"):
                    name = "QT/" + name
                if name and ROS_AVAILABLE:
                    try:
                        from qt_gesture_controller.srv import gesture_play
                        ges_proxy = rospy.ServiceProxy('/qt_robot/gesture/play', gesture_play)
                        rospy.wait_for_service('/qt_robot/gesture/play', timeout=5.0)
                        ges_proxy(name, 1.0)
                    except Exception as e:
                        print(f"[DIY gesture] error: {e}")
            elif btype == "emotion":
                name = (block.get("name") or "").strip()
                if name and ROS_AVAILABLE:
                    try:
                        from qt_robot_interface.srv import emotion_show
                        emo_proxy = rospy.ServiceProxy('/qt_robot/emotion/show', emotion_show)
                        rospy.wait_for_service('/qt_robot/emotion/show', timeout=5.0)
                        emo_proxy(name)
                    except Exception as e:
                        print(f"[DIY emotion] error: {e}")
    # Reset step-by-step state
    if require_confirm:
        _step_current_index = -1
        _step_waiting = False

@app.route("/api/activity/step_status", methods=["GET"])
def api_activity_step_status():
    """Return current step-by-step execution status for therapist UI."""
    return jsonify({
        "waiting": _step_waiting,
        "current_index": _step_current_index,
        "total": _step_total_count,
        "current_label": _step_current_label,
        "next_label": _step_next_label
    })

@app.route("/api/activity/confirm_step", methods=["POST"])
def api_activity_confirm_step():
    """Therapist confirms to proceed to the next step."""
    _step_confirm_event.set()
    return jsonify({"success": True})

@app.route("/api/activity/run_saved", methods=["POST"])
def api_activity_run_saved():
    """Execute a previously saved DIY activity by filename."""
    username = session.get('username')
    if not username:
        return jsonify({"success": False, "error": "Not logged in"}), 401
    data = request.get_json() or {}
    filename = (data.get('filename') or '').strip()
    if not filename:
        return jsonify({"success": False, "error": "Filename required"}), 400
    try:
        user_dir = os.path.join(USER_DATA_DIR, username, "activities")
        fpath = os.path.join(user_dir, filename)
        if not os.path.exists(fpath):
            return jsonify({"success": False, "error": "Activity not found"}), 404
        with open(fpath, 'r') as f:
            saved = json.load(f)
        blocks = saved.get('blocks', [])
        loop_count = int(saved.get('loop', 1) or 1)
        for blk in blocks:
            if blk.get('type') == 'loop':
                try:
                    loop_count = max(1, int(blk.get('count', loop_count)))
                except Exception:
                    pass
                break
        # Determine if therapist confirmation is needed between steps
        require_confirm = len([b for b in blocks if b.get("type") != "loop"]) > 1
        # Run in background (either for parallel recognizers or step-by-step confirm)
        global _activity_thread
        _activity_stop_event.clear()
        global _asr_enabled
        _asr_enabled = True
        if _has_parallel_recognizers(blocks) or require_confirm:
            def runner():
                try:
                    _execute_activity(blocks, loop_count, require_confirm=require_confirm, username=username)
                except Exception as e:
                    print(f"Run activity error: {e}")
            if _activity_thread and _activity_thread.is_alive():
                try:
                    _activity_stop_event.set()
                    _activity_thread.join(timeout=0.5)
                except Exception:
                    pass
                _activity_stop_event.clear()
            _activity_thread = Thread(target=runner, daemon=True)
            _activity_thread.start()
            return jsonify({"success": True, "running": True, "require_confirm": require_confirm})
        else:
            _execute_activity(blocks, loop_count, username=username)
            return jsonify({"success": True, "running": False})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/activity/stop', methods=['POST'])
def api_activity_stop():
    """Signal any running activity to stop continuous listening."""
    username = session.get('username')
    if not username:
        return jsonify({"success": False, "error": "Not logged in"}), 401
    try:
        global _activity_thread
        _activity_stop_event.set()
        global _asr_enabled
        _asr_enabled = False
        if _activity_thread and _activity_thread.is_alive():
            _activity_thread.join(timeout=1.0)
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/my_games")
def my_games_page():
    if 'username' not in session:
        return redirect(url_for('index'))
    username = session['username']
    user = user_manager.users.get(username)
    return render_template("my_games.html", logged_in=True, user=user)

@app.route("/api/activity/save", methods=["POST"])
def api_activity_save():
    """Persist the custom activity for the user"""
    username = session.get('username')
    if not username:
        return jsonify({"success": False, "error": "Not logged in"}), 401
    data = request.get_json() or {}
    try:
        user_dir = os.path.join(USER_DATA_DIR, username, "activities")
        os.makedirs(user_dir, exist_ok=True)
        import datetime
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"activity_{ts}.json"
        fpath = os.path.join(user_dir, fname)
        with open(fpath, "w") as f:
            json.dump(data, f, indent=2)
        return jsonify({"success": True, "filename": fname})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/activity/load_saved", methods=["POST"])
def api_activity_load_saved():
    """Load a previously saved DIY activity by filename for hydration in builder."""
    username = session.get('username')
    if not username:
        return jsonify({"success": False, "error": "Not logged in"}), 401
    data = request.get_json() or {}
    filename = (data.get('filename') or '').strip()
    if not filename:
        return jsonify({"success": False, "error": "Filename required"}), 400
    try:
        user_dir = os.path.join(USER_DATA_DIR, username, "activities")
        fpath = os.path.join(user_dir, filename)
        if not os.path.exists(fpath):
            return jsonify({"success": False, "error": "Activity not found"}), 404
        with open(fpath, 'r') as f:
            saved = json.load(f)
        return jsonify({"success": True, "activity": saved, "filename": filename})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/activity/delete", methods=["POST"])
def api_activity_delete():
    """Delete a previously saved DIY activity by filename."""
    username = session.get('username')
    if not username:
        return jsonify({"success": False, "error": "Not logged in"}), 401
    data = request.get_json() or {}
    filename = (data.get('filename') or '').strip()
    if not filename:
        return jsonify({"success": False, "error": "Filename required"}), 400
    try:
        user_dir = os.path.join(USER_DATA_DIR, username, "activities")
        fpath = os.path.join(user_dir, filename)
        if not os.path.exists(fpath):
            return jsonify({"success": False, "error": "Activity not found"}), 404
        os.remove(fpath)
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/read_user_stories", methods=["POST"])
def api_read_user_stories():
    """Read user's saved stories aloud using robot TTS"""
    username = session.get('username')
    if not username:
        return jsonify({"error": "Not logged in"}), 401
    
    user = user_manager.users.get(username)
    if not user:
        return jsonify({"error": "User not found"}), 404
    
    try:
        # Get user's stories directory
        user_stories_dir = os.path.join(USER_DATA_DIR, username, "stories")
        
        if not os.path.exists(user_stories_dir):
            return jsonify({"error": "No stories found for this user"}), 404
        
        # Get all story files
        story_files = [f for f in os.listdir(user_stories_dir) if f.endswith('.json')]
        
        if not story_files:
            return jsonify({"error": "No stories found for this user"}), 404
        
        # Read the most recent story (or a random one)
        story_files.sort(reverse=True)  # Most recent first
        latest_story_file = story_files[0]
        story_path = os.path.join(user_stories_dir, latest_story_file)
        
        with open(story_path, 'r') as f:
            story_data = json.load(f)
        
        story_text = story_data.get('story', '')
        metadata = story_data.get('metadata', {})
        
        if not story_text:
            return jsonify({"error": "Story content is empty"}), 400
        
        # Clean the story text before speaking
        cleaned_story = clean_story_text(story_text)
        
        # Make the robot speak the story
        if tts_helper.is_available():
            # Determine language from metadata or use default
            language = metadata.get('language', 'en-US')
            # Start human tracking while reading
            tracker = None
            try:
                tracker = _ensure_human_tracker()
                if tracker:
                    person = _pick_recent_person(tracker, timeout_sec=0.5)
                    tracker.track(person)
            except Exception as _e:
                pass
            try:
                success = tts_helper.speak_story(cleaned_story, language)
            finally:
                try:
                    if tracker:
                        tracker.untrack()
                except Exception:
                    pass
            
            if success:
                return jsonify({
                    "success": True,
                    "story": cleaned_story,
                    "metadata": metadata,
                    "filename": latest_story_file,
                    "message": "Story is being read aloud by QTrobot!"
                }), 200
            else:
                return jsonify({
                    "success": False,
                    "error": "Failed to make robot speak the story",
                    "story": cleaned_story,
                    "metadata": metadata,
                    "filename": latest_story_file
                }), 500
        else:
            # TTS not available, return story without speaking
            return jsonify({
                "success": True,
                "story": cleaned_story,
                "metadata": metadata,
                "filename": latest_story_file,
                "message": "TTS not available. Story content provided.",
                "tts_available": False
            }), 200
        
    except Exception as e:
        return jsonify({"error": f"Error reading stories: {str(e)}"}), 500

@app.route("/api/get_specific_story_details", methods=["POST"])
def api_get_specific_story_details():
    """Get details of a specific story without speaking"""
    username = session.get('username')
    if not username:
        return jsonify({"error": "Not logged in"}), 401
    
    data = request.get_json() or {}
    filename = data.get("filename")
    
    if not filename:
        return jsonify({"error": "Filename is required"}), 400
    
    try:
        # Get user's stories directory
        user_stories_dir = os.path.join(USER_DATA_DIR, username, "stories")
        story_path = os.path.join(user_stories_dir, filename)
        
        if not os.path.exists(story_path):
            return jsonify({"error": "Story file not found"}), 404
        
        with open(story_path, 'r') as f:
            story_data = json.load(f)
        
        story_text = story_data.get('story', '')
        metadata = story_data.get('metadata', {})
        
        if not story_text:
            return jsonify({"error": "Story content is empty"}), 400
        
        # Clean the story text for display
        cleaned_story = clean_story_text(story_text)
        
        # Return story details without speaking
        return jsonify({
            "success": True,
            "story": cleaned_story,
            "metadata": metadata,
            "filename": filename
        }), 200
        
    except Exception as e:
        return jsonify({"error": f"Error loading story: {str(e)}"}), 500

@app.route("/api/read_specific_story", methods=["POST"])
def api_read_specific_story():
    """Read a specific story by filename"""
    username = session.get('username')
    if not username:
        return jsonify({"error": "Not logged in"}), 401
    
    data = request.get_json() or {}
    filename = data.get("filename")
    
    if not filename:
        return jsonify({"error": "Filename is required"}), 400
    
    try:
        # Get user's stories directory
        user_stories_dir = os.path.join(USER_DATA_DIR, username, "stories")
        story_path = os.path.join(user_stories_dir, filename)
        
        if not os.path.exists(story_path):
            return jsonify({"error": "Story file not found"}), 404
        
        with open(story_path, 'r') as f:
            story_data = json.load(f)
        
        story_text = story_data.get('story', '')
        metadata = story_data.get('metadata', {})
        
        if not story_text:
            return jsonify({"error": "Story content is empty"}), 400
        
        # Clean the story text before speaking
        cleaned_story = clean_story_text(story_text)
        
        # Make the robot speak the story
        if tts_helper.is_available():
            # Determine language from metadata or use default
            language = metadata.get('language', 'en-US')
            # Start human tracking while reading
            tracker = None
            try:
                tracker = _ensure_human_tracker()
                if tracker:
                    person = _pick_recent_person(tracker, timeout_sec=0.5)
                    tracker.track(person)
            except Exception as _e:
                pass
            try:
                success = tts_helper.speak_story(cleaned_story, language)
            finally:
                try:
                    if tracker:
                        tracker.untrack()
                except Exception:
                    pass
            
            if success:
                return jsonify({
                    "success": True,
                    "story": cleaned_story,
                    "metadata": metadata,
                    "filename": filename,
                    "message": "Story is being read aloud by QTrobot!"
                }), 200
            else:
                return jsonify({
                    "success": False,
                    "error": "Failed to make robot speak the story",
                    "story": cleaned_story,
                    "metadata": metadata,
                    "filename": filename
                }), 500
        else:
            # TTS not available, return story without speaking
            return jsonify({
                "success": True,
                "story": cleaned_story,
                "metadata": metadata,
                "filename": filename,
                "message": "TTS not available. Story content provided.",
                "tts_available": False
            }), 200
        
    except Exception as e:
        return jsonify({"error": f"Error reading story: {str(e)}"}), 500

@app.route("/api/get_user_stories", methods=["GET"])
def api_get_user_stories():
    """Get list of user's saved stories"""
    username = session.get('username')
    if not username:
        return jsonify({"error": "Not logged in"}), 401
    
    try:
        # Get user's stories directory
        user_stories_dir = os.path.join(USER_DATA_DIR, username, "stories")
        
        if not os.path.exists(user_stories_dir):
            return jsonify({"stories": []}), 200
        
        # Debug print statements
        print(f"Looking for stories in: {user_stories_dir}")
        print(f"Files found: {os.listdir(user_stories_dir)}")
        
        # Get all story files with metadata
        stories = []
        for filename in os.listdir(user_stories_dir):
            if filename.endswith('.json'):
                print(f"Processing file: {filename}")
                story_path = os.path.join(user_stories_dir, filename)
                try:
                    with open(story_path, 'r') as f:
                        story_data = json.load(f)
                    
                    print(f"Successfully loaded {filename}")
                    metadata = story_data.get('metadata', {})
                    # Use filename timestamp as fallback if generated_at is null
                    created_at = metadata.get('generated_at')
                    if not created_at and filename.startswith('story_'):
                        # Extract timestamp from filename like 'story_20250702_230233.json'
                        try:
                            timestamp_part = filename.replace('story_', '').replace('.json', '')
                            created_at = f"{timestamp_part[:8]} {timestamp_part[8:10]}:{timestamp_part[10:12]}:{timestamp_part[12:14]}"
                        except:
                            created_at = 'Unknown'
                    
                    # Clean the preview text
                    raw_story = story_data.get('story', '')
                    cleaned_preview = clean_story_text(raw_story)
                    preview = cleaned_preview[:100] + "..." if len(cleaned_preview) > 100 else cleaned_preview
                    
                    stories.append({
                        "filename": filename,
                        "title": f"Story for {metadata.get('child_name', 'Unknown')}",
                        "age": metadata.get('age', 'Unknown'),
                        "word_count": metadata.get('word_count', 0),
                        "created_at": created_at or 'Unknown',
                        "preview": preview
                    })
                    print(f"Added story: {filename}")
                except Exception as e:
                    # Skip corrupted files
                    print(f"Error processing {filename}: {str(e)}")
                    continue
        
        # Sort by creation date (newest first)
        stories.sort(key=lambda x: x.get('created_at', '') or '', reverse=True)
        
        # Debug print: show what we're returning
        print(f"Returning {len(stories)} stories: {stories}")
        
        return jsonify({"stories": stories}), 200
        
    except Exception as e:
        return jsonify({"error": f"Error getting stories: {str(e)}"}), 500

@app.route('/read_story/<filename>')
def read_story_page(filename):
    # Auto-start tracking on read story page load so robot keeps following
    try:
        tracker = _ensure_human_tracker()
        if tracker and not tracker.should_track:
            person = _pick_recent_person(tracker, timeout_sec=0.5)
            tracker.track(person)
    except Exception as e:
        print(f"HumanTracking auto-start (/read_story) error: {e}")
    return render_template('read_story.html')

@app.route('/api/get_story_sentences')
def api_get_story_sentences():
    username = session.get('username')
    filename = request.args.get('filename')
    if not username or not filename:
        return jsonify({'success': False, 'error': 'Missing username or filename'})
    user_stories_dir = os.path.join(USER_DATA_DIR, username, 'stories')
    story_path = os.path.join(user_stories_dir, filename)
    if not os.path.exists(story_path):
        return jsonify({'success': False, 'error': 'Story not found'})
    try:
        with open(story_path, 'r') as f:
            story_data = json.load(f)
        metadata = story_data.get('metadata', {})

        # Use pre-split pages if available (saved at approval time)
        pages = story_data.get('pages')
        if pages and isinstance(pages, list) and len(pages) > 0:
            sentences = pages
        else:
            # Fallback for older stories without pages: split now based on age
            story_text = story_data.get('story', '')
            child_age = 5
            try:
                child_age = int(metadata.get('age', 5))
            except (ValueError, TypeError):
                child_age = 5
            sentences = _split_story_into_pages(story_text, child_age)

        return jsonify({
            'success': True,
            'sentences': sentences,
            'metadata': metadata,
            'images_available': image_generator.is_available()
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/get_sentence_image', methods=['POST'])
def api_get_sentence_image():
    """Get the scene image for a specific page (sentence_index = page index).

    Uses the saved page_to_scene mapping to find the correct scene image.
    The same image is returned for all pages that belong to the same scene.
    """
    username = session.get('username')
    data = request.get_json() or {}
    filename = data.get('filename', '')
    sentence_index = int(data.get('sentence_index', 0))

    if not username or not filename:
        return jsonify({'success': False, 'error': 'Missing username or filename'})

    try:
        user_images_dir = os.path.join(USER_DATA_DIR, username, 'story_images', filename.replace('.json', ''))
        user_stories_dir = os.path.join(USER_DATA_DIR, username, 'stories')
        story_path = os.path.join(user_stories_dir, filename)

        if not os.path.exists(story_path):
            return jsonify({'success': False, 'error': 'Story not found'})
        if not os.path.exists(user_images_dir):
            return jsonify({'success': False, 'error': 'No images directory found'})

        with open(story_path, 'r') as f:
            story_data = json.load(f)

        # Use saved page_to_scene mapping (new format)
        page_to_scene = story_data.get('page_to_scene')
        if page_to_scene and isinstance(page_to_scene, list):
            if sentence_index < 0 or sentence_index >= len(page_to_scene):
                return jsonify({'success': False, 'error': 'Page index out of range'})
            scene_index = page_to_scene[sentence_index]

            # Look for scene image
            pattern = f"story_scene_{int(scene_index):03d}_"
            matches = [f for f in os.listdir(user_images_dir) if f.startswith(pattern) and f.endswith('.png')]
            matches.sort()

            if matches:
                image_path = f"/images/{username}/story_images/{os.path.basename(user_images_dir)}/{matches[0]}"
                return jsonify({'success': True, 'image_path': image_path, 'scene_index': scene_index})

        # Fallback: try legacy story_paragraph_NNN format
        pattern = f"story_paragraph_{int(sentence_index):03d}_"
        matches = [f for f in os.listdir(user_images_dir) if f.startswith(pattern) and f.endswith('.png')]
        matches.sort()

        if matches:
            image_path = f"/images/{username}/story_images/{os.path.basename(user_images_dir)}/{matches[0]}"
            return jsonify({'success': True, 'image_path': image_path, 'scene_index': sentence_index})

        return jsonify({'success': False, 'error': 'No image found for this page'})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/images/<path:filename>')
def serve_image(filename):
    """Serve generated images robustly"""
    full_path = os.path.join(USER_DATA_DIR, filename)
    print(f"Serving image: {filename}")
    print(f"Full path: {full_path}")
    print(f"File exists: {os.path.exists(full_path)}")
    if os.path.exists(full_path):
        return send_file(full_path)
    else:
        return "Image not found", 404

@app.route('/api/speak_sentence', methods=['POST'])
def api_speak_sentence():
    username = session.get('username')
    data = request.get_json() or {}
    sentence = data.get('sentence', '')
    filename = data.get('filename', '')
    if not username or not sentence:
        return jsonify({'success': False, 'error': 'Missing username or sentence'})
    
    # Clean the sentence before speaking
    cleaned_sentence = clean_story_text(sentence)
    
    # Optionally get language from story metadata
    language = 'en-US'
    if filename:
        user_stories_dir = os.path.join(USER_DATA_DIR, username, 'stories')
        story_path = os.path.join(user_stories_dir, filename)
        if os.path.exists(story_path):
            try:
                with open(story_path, 'r') as f:
                    story_data = json.load(f)
                metadata = story_data.get('metadata', {})
                language = metadata.get('language', 'en-US')
            except:
                pass
    
    # Track while speaking this sentence
    tracker = None
    try:
        tracker = _ensure_human_tracker()
        if tracker:
            person = _pick_recent_person(tracker, timeout_sec=0.5)
            tracker.track(person)
    except Exception:
        pass
    try:
        _with_asr_suspended(lambda: tts_helper.speak_story(cleaned_sentence, language))
    finally:
        try:
            if tracker:
                tracker.untrack()
        except Exception:
            pass
    return jsonify({'success': True})

@app.route('/api/movement_settings', methods=['POST'])
def api_movement_settings():
    """Enable or disable movement during speech"""
    username = session.get('username')
    if not username:
        return jsonify({'success': False, 'error': 'Not logged in'}), 401
    
    data = request.get_json() or {}
    enabled = data.get('enabled', True)
    
    try:
        tts_helper.enable_movement(enabled)
        return jsonify({
            'success': True, 
            'movement_enabled': enabled,
            'movement_available': tts_helper.is_movement_available()
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/movement_status', methods=['GET'])
def api_movement_status():
    """Get current movement status"""
    username = session.get('username')
    if not username:
        return jsonify({'success': False, 'error': 'Not logged in'}), 401
    
    try:
        return jsonify({
            'success': True,
            'movement_available': tts_helper.is_movement_available(),
            'movement_enabled': getattr(tts_helper, 'movement_enabled', False)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/volume_settings', methods=['POST'])
def api_volume_settings():
    """Set robot speaker volume (0-100) via ROS service."""
    username = session.get('username')
    if not username:
        return jsonify({'success': False, 'error': 'Not logged in'}), 401

    data = request.get_json() or {}
    try:
        level = int(data.get('level', 50))
        level = max(0, min(100, level))
    except Exception:
        return jsonify({'success': False, 'error': 'Invalid volume level'}), 400

    try:
        # Equivalent to: rosservice call /qt_robot/setting/setVolume "volume: <level>"
        from qt_robot_interface.srv import setting_setVolume
        service = rospy.ServiceProxy('/qt_robot/setting/setVolume', setting_setVolume)
        service(level)

        setattr(tts_helper, 'volume_level', level)
        return jsonify({'success': True, 'volume_level': level, 'applied': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/polly_volume', methods=['POST'])
def api_polly_volume():
    """Set Polly SSML volume in dB (e.g., '+6dB', '-3dB')."""
    username = session.get('username')
    if not username:
        return jsonify({'success': False, 'error': 'Not logged in'}), 401

    data = request.get_json() or {}
    volume_db = str(data.get('volume_db', '')).strip()
    if not volume_db:
        return jsonify({'success': False, 'error': 'Missing volume_db'}), 400

    try:
        if not tts_helper.set_polly_volume(volume_db):
            return jsonify({'success': False, 'error': 'Invalid dB value'}), 400
        return jsonify({'success': True, 'polly_volume': tts_helper.polly_volume})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
@app.route('/api/volume_test', methods=['POST'])
def api_volume_test():
    username = session.get('username')
    if not username:
        return jsonify({'success': False, 'error': 'Not logged in'}), 401
    try:
        _with_asr_suspended(lambda: tts_helper.speak_story("Testing volume.", "en-US"))
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
@app.route('/api/joint_limits', methods=['GET'])
def api_joint_limits():
    """Get joint limits and safe movement ranges"""
    username = session.get('username')
    if not username:
        return jsonify({'success': False, 'error': 'Not logged in'}), 401
    
    try:
        return jsonify({
            'success': True,
            'joint_limits': tts_helper.get_joint_limits(),
            'safe_movement_ranges': tts_helper.get_safe_movement_ranges()
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/head_position', methods=['GET'])
def api_head_position():
    """Get current head position"""
    username = session.get('username')
    if not username:
        return jsonify({'success': False, 'error': 'Not logged in'}), 401
    
    try:
        yaw, pitch = tts_helper.get_current_head_position()
        return jsonify({
            'success': True,
            'head_position': {
                'yaw': yaw,
                'pitch': pitch
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# ===================== SCENE GAME TOY LIST =====================

SCENE_GAME_DEFAULT_TOYS = ['lemon', 'tomato', 'apple', 'banana']
SCENE_GAME_TOYS_FILE = os.path.join(USER_DATA_DIR, 'scene_game_toys.json')

def _load_scene_toys():
    """Load the toy list from disk, falling back to defaults."""
    if os.path.exists(SCENE_GAME_TOYS_FILE):
        try:
            with open(SCENE_GAME_TOYS_FILE, 'r') as f:
                toys = json.load(f)
            if isinstance(toys, list) and len(toys) > 0:
                return toys
        except Exception:
            pass
    return list(SCENE_GAME_DEFAULT_TOYS)

def _save_scene_toys(toys):
    """Persist the toy list to disk."""
    os.makedirs(os.path.dirname(SCENE_GAME_TOYS_FILE), exist_ok=True)
    with open(SCENE_GAME_TOYS_FILE, 'w') as f:
        json.dump(toys, f, indent=2)

@app.route("/object_game_generate")
def object_game_generate_page():
    """Game generation mode – manage the physical toy list for object detection."""
    if 'username' not in session:
        return redirect(url_for('index'))
    user = user_manager.users.get(session['username'])
    return render_template("scene_game_config.html", logged_in=True, user=user)

@app.route('/api/scene_game/toys', methods=['GET'])
def api_scene_game_toys_get():
    """Return the current toy list."""
    return jsonify({'success': True, 'toys': _load_scene_toys()})

@app.route('/api/scene_game/toys', methods=['POST'])
def api_scene_game_toys_post():
    """Add, delete, or reset the toy list."""
    data = request.get_json() or {}
    action = data.get('action', '')
    toys = _load_scene_toys()

    if action == 'add':
        name = (data.get('name') or '').strip()
        if not name:
            return jsonify({'success': False, 'error': 'Name cannot be empty'})
        if name.lower() in [t.lower() for t in toys]:
            return jsonify({'success': False, 'error': f'"{name}" already exists'})
        toys.append(name)
        _save_scene_toys(toys)
        return jsonify({'success': True, 'toys': toys})

    if action == 'delete':
        name = (data.get('name') or '').strip()
        toys = [t for t in toys if t.lower() != name.lower()]
        _save_scene_toys(toys)
        return jsonify({'success': True, 'toys': toys})

    if action == 'reset':
        toys = list(SCENE_GAME_DEFAULT_TOYS)
        _save_scene_toys(toys)
        return jsonify({'success': True, 'toys': toys})

    return jsonify({'success': False, 'error': 'Unknown action'}), 400

@app.route('/api/scene/start', methods=['POST'])
def api_scene_start():
    try:
        toy_list = _load_scene_toys()
        if not toy_list:
            toy_list = list(SCENE_GAME_DEFAULT_TOYS)

        child_age = 5
        learning_goals = ''
        username = session.get('username')
        if username:
            child_age, learning_goals = _get_user_age_and_goals(username)

        result = _scene_game_generate_question(toy_list, child_age, learning_goals)
        question = result['question']
        try:
            _with_asr_suspended(lambda: tts_helper.speak(question))
        except Exception:
            pass
        return jsonify({
            'success': True,
            'question': question,
            'target': result.get('target'),
            'criteria': result.get('criteria'),
            'mode': result['mode']
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# ===================== WH PICTURE SCENE ENDPOINTS =====================

WH_SCENE_DIR_NAME = 'wh_scenes'

def _user_wh_dir(username):
    """Get or create the WH scenes directory for a user."""
    d = os.path.join(USER_DATA_DIR, username, WH_SCENE_DIR_NAME)
    os.makedirs(d, exist_ok=True)
    return d

def _load_scene_index(username):
    """Load the scene index JSON for a user."""
    idx_path = os.path.join(_user_wh_dir(username), 'index.json')
    if os.path.exists(idx_path):
        try:
            with open(idx_path, 'r') as f:
                return json.load(f)
        except Exception:
            return []
    return []

def _save_scene_index(username, index):
    """Save the scene index JSON for a user."""
    idx_path = os.path.join(_user_wh_dir(username), 'index.json')
    with open(idx_path, 'w') as f:
        json.dump(index, f, indent=2)

def _run_gemini_wh_analysis(image_path, child_age, difficulty):
    """Run the Gemini vision worker to analyze a scene and generate WH questions."""
    script_path = os.path.join(os.path.dirname(BASE_DIR), 'scripts', 'gemini_wh_scene.py')
    if not os.path.exists(script_path):
        return None, "Worker script not found"
    payload = json.dumps({
        "image_path": image_path,
        "child_age": child_age,
        "difficulty": difficulty
    })
    try:
        proc = subprocess.run(
            [WORKER_PYTHON, script_path],
            input=payload,
            capture_output=True,
            text=True,
            timeout=90,
            env=os.environ.copy()
        )
        if proc.returncode != 0:
            return None, f"Worker failed: {proc.stderr.strip()}"
        raw = proc.stdout.strip()
        if raw.startswith("```"):
            raw = raw.strip("`")
            if raw.startswith("json"):
                raw = raw[4:].strip()
        result = json.loads(raw)
        return result, None
    except json.JSONDecodeError as e:
        return None, f"JSON parse error: {e}"
    except subprocess.TimeoutExpired:
        return None, "Analysis timed out"
    except Exception as e:
        return None, str(e)


@app.route("/wh_picture_scene")
def wh_picture_scene_page():
    """WH Questions Picture Scene - Prepare (therapist uploads images)."""
    if 'username' not in session:
        return redirect(url_for('index'))
    username = session['username']
    user = user_manager.users.get(username)
    return render_template("wh_picture_scene.html", logged_in=True, user=user)


@app.route("/wh_picture_play")
def wh_picture_play_page():
    """WH Questions Picture Scene - Play session (child plays)."""
    if 'username' not in session:
        return redirect(url_for('index'))
    username = session['username']
    user = user_manager.users.get(username)
    return render_template("wh_picture_play.html", logged_in=True, user=user)


@app.route("/api/wh_scene/upload", methods=["POST"])
def api_wh_scene_upload():
    """Upload a scene image, analyze it with Gemini, generate WH questions."""
    username = session.get('username')
    if not username:
        return jsonify({"success": False, "error": "Not logged in"}), 401

    if 'image' not in request.files:
        return jsonify({"success": False, "error": "No image provided"}), 400

    file = request.files['image']
    if not file.filename:
        return jsonify({"success": False, "error": "Empty filename"}), 400

    # Save image
    wh_dir = _user_wh_dir(username)
    images_dir = os.path.join(wh_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)

    ext = os.path.splitext(file.filename)[1].lower() or '.jpg'
    scene_id = f"scene_{int(time.time())}_{str(uuid.uuid4())[:6]}"
    safe_name = scene_id + ext
    image_path = os.path.join(images_dir, safe_name)
    file.save(image_path)

    # Get user profile for age-appropriate questions
    user = user_manager.users.get(username, {})
    child_age = user.get('age', 5)
    # Default to receptive for initial generation
    difficulty = "receptive"

    # Run Gemini analysis
    result, error = _run_gemini_wh_analysis(image_path, child_age, difficulty)

    index = _load_scene_index(username)

    if result and 'questions' in result:
        # Save questions to a separate JSON file
        questions_path = os.path.join(wh_dir, f"{scene_id}_questions.json")
        with open(questions_path, 'w') as f:
            json.dump(result, f, indent=2)

        rel_path = os.path.relpath(image_path, USER_DATA_DIR)
        entry = {
            "id": scene_id,
            "filename": file.filename,
            "image_path": image_path,
            "image_url": f"/images/{rel_path}",
            "scene_description": result.get("scene_description", ""),
            "question_count": len(result.get("questions", [])),
            "status": "ready",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S")
        }
        index.append(entry)
        _save_scene_index(username, index)

        return jsonify({"success": True, "scene": entry})
    else:
        # Save with error status so therapist can retry
        rel_path = os.path.relpath(image_path, USER_DATA_DIR)
        entry = {
            "id": scene_id,
            "filename": file.filename,
            "image_path": image_path,
            "image_url": f"/images/{rel_path}",
            "scene_description": "",
            "question_count": 0,
            "status": "error",
            "error": error or "Unknown error",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S")
        }
        index.append(entry)
        _save_scene_index(username, index)

        return jsonify({"success": True, "scene": entry, "warning": error})


@app.route("/api/wh_scene/capture", methods=["POST"])
def api_wh_scene_capture():
    """Capture a scene from the robot camera, analyze it, generate WH questions."""
    username = session.get('username')
    if not username:
        return jsonify({"success": False, "error": "Not logged in"}), 401
    if cv2 is None:
        return jsonify({"success": False, "error": "OpenCV not available"}), 500

    frame = _get_ros_frame()
    if frame is None:
        return jsonify({"success": False, "error": "Camera read failed"}), 500

    # Save captured frame
    wh_dir = _user_wh_dir(username)
    images_dir = os.path.join(wh_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)

    scene_id = f"scene_{int(time.time())}_{str(uuid.uuid4())[:6]}"
    safe_name = scene_id + '.jpg'
    image_path = os.path.join(images_dir, safe_name)
    ok = cv2.imwrite(image_path, frame)
    if not ok:
        return jsonify({"success": False, "error": "Failed to save captured image"}), 500

    # Get user profile for age-appropriate questions
    user = user_manager.users.get(username, {})
    child_age = user.get('age', 5)
    difficulty = "receptive"

    # Run Gemini analysis
    result, error = _run_gemini_wh_analysis(image_path, child_age, difficulty)

    index = _load_scene_index(username)
    rel_path = os.path.relpath(image_path, USER_DATA_DIR)
    ts_label = time.strftime("%Y-%m-%d %H:%M:%S")

    if result and 'questions' in result:
        questions_path = os.path.join(wh_dir, f"{scene_id}_questions.json")
        with open(questions_path, 'w') as f:
            json.dump(result, f, indent=2)

        entry = {
            "id": scene_id,
            "filename": f"capture_{ts_label}.jpg",
            "image_path": image_path,
            "image_url": f"/images/{rel_path}",
            "scene_description": result.get("scene_description", ""),
            "question_count": len(result.get("questions", [])),
            "status": "ready",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S")
        }
        index.append(entry)
        _save_scene_index(username, index)
        return jsonify({"success": True, "scene": entry})
    else:
        entry = {
            "id": scene_id,
            "filename": f"capture_{ts_label}.jpg",
            "image_path": image_path,
            "image_url": f"/images/{rel_path}",
            "scene_description": "",
            "question_count": 0,
            "status": "error",
            "error": error or "Unknown error",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S")
        }
        index.append(entry)
        _save_scene_index(username, index)
        return jsonify({"success": True, "scene": entry, "warning": error})


@app.route("/api/wh_scene/list")
def api_wh_scene_list():
    """List all uploaded scenes for the current user."""
    username = session.get('username')
    if not username:
        return jsonify({"success": False, "error": "Not logged in"}), 401
    index = _load_scene_index(username)
    return jsonify({"success": True, "scenes": index})


@app.route("/api/wh_scene/delete", methods=["POST"])
def api_wh_scene_delete():
    """Delete a scene and its questions."""
    username = session.get('username')
    if not username:
        return jsonify({"success": False, "error": "Not logged in"}), 401
    data = request.get_json() or {}
    scene_id = data.get("scene_id")
    if not scene_id:
        return jsonify({"success": False, "error": "Missing scene_id"}), 400

    index = _load_scene_index(username)
    scene = next((s for s in index if s['id'] == scene_id), None)
    if not scene:
        return jsonify({"success": False, "error": "Scene not found"}), 404

    # Delete image file
    try:
        if os.path.exists(scene.get('image_path', '')):
            os.remove(scene['image_path'])
    except Exception:
        pass

    # Delete questions file
    wh_dir = _user_wh_dir(username)
    q_path = os.path.join(wh_dir, f"{scene_id}_questions.json")
    try:
        if os.path.exists(q_path):
            os.remove(q_path)
    except Exception:
        pass

    index = [s for s in index if s['id'] != scene_id]
    _save_scene_index(username, index)
    return jsonify({"success": True})


@app.route("/api/wh_scene/regenerate", methods=["POST"])
def api_wh_scene_regenerate():
    """Re-run Gemini analysis for a scene that failed."""
    username = session.get('username')
    if not username:
        return jsonify({"success": False, "error": "Not logged in"}), 401
    data = request.get_json() or {}
    scene_id = data.get("scene_id")
    if not scene_id:
        return jsonify({"success": False, "error": "Missing scene_id"}), 400

    index = _load_scene_index(username)
    scene = next((s for s in index if s['id'] == scene_id), None)
    if not scene:
        return jsonify({"success": False, "error": "Scene not found"}), 404

    user = user_manager.users.get(username, {})
    child_age = user.get('age', 5)

    result, error = _run_gemini_wh_analysis(scene['image_path'], child_age, "receptive")

    if result and 'questions' in result:
        wh_dir = _user_wh_dir(username)
        q_path = os.path.join(wh_dir, f"{scene_id}_questions.json")
        with open(q_path, 'w') as f:
            json.dump(result, f, indent=2)
        scene['status'] = 'ready'
        scene['scene_description'] = result.get('scene_description', '')
        scene['question_count'] = len(result.get('questions', []))
        scene.pop('error', None)
        _save_scene_index(username, index)
        return jsonify({"success": True})
    else:
        scene['status'] = 'error'
        scene['error'] = error or 'Unknown error'
        _save_scene_index(username, index)
        return jsonify({"success": False, "error": error})


@app.route("/api/wh_scene/get_questions", methods=["POST"])
def api_wh_scene_get_questions():
    """Get the generated WH questions for a scene."""
    username = session.get('username')
    if not username:
        return jsonify({"success": False, "error": "Not logged in"}), 401
    data = request.get_json() or {}
    scene_id = data.get("scene_id")
    if not scene_id:
        return jsonify({"success": False, "error": "Missing scene_id"}), 400

    wh_dir = _user_wh_dir(username)
    q_path = os.path.join(wh_dir, f"{scene_id}_questions.json")
    if not os.path.exists(q_path):
        return jsonify({"success": False, "error": "Questions not found"}), 404

    with open(q_path, 'r') as f:
        result = json.load(f)

    return jsonify({
        "success": True,
        "questions": result.get("questions", []),
        "scene_description": result.get("scene_description", "")
    })


@app.route("/api/wh_scene/save_result", methods=["POST"])
def api_wh_scene_save_result():
    """Save a session result for progress tracking."""
    username = session.get('username')
    if not username:
        return jsonify({"success": False, "error": "Not logged in"}), 401
    data = request.get_json() or {}
    scene_id = data.get("scene_id")
    mode = data.get("mode", "receptive")
    score_val = data.get("score", 0)
    total = data.get("total", 0)

    wh_dir = _user_wh_dir(username)
    results_path = os.path.join(wh_dir, 'results.json')
    results = []
    if os.path.exists(results_path):
        try:
            with open(results_path, 'r') as f:
                results = json.load(f)
        except Exception:
            results = []

    results.append({
        "scene_id": scene_id,
        "mode": mode,
        "score": score_val,
        "total": total,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
    })

    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    return jsonify({"success": True})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)