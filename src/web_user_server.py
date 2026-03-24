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
    # from ultralytics import YOLO  # disabled: not using YOLO
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

        # # Load YOLO model (disabled)
        # self.model = YOLO("yolov8n.pt")  # Load a lightweight pretrained YOLOv8 model
        #
        # # Start processing thread (disabled)
        # self.processing_thread = Thread(target=self.process_images, daemon=True)
        # self.processing_thread.start()

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

    def process_images(self):
        rate = rospy.Rate(10)  # 10 FPS
        while not rospy.is_shutdown():
            image = self.get_latest_image()
            if image is not None:
                # Run YOLO detection on the image
                results = self.model(image)

                for result in results:
                    for box in result.boxes:
                        cls_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        label = self.model.names[cls_id]
                        print(f"Detected {label} with confidence {confidence:.2f}")

                # (Optional) Display image with detections
                annotated_frame = results[0].plot()
                cv2.imshow("YOLOv8 Detection", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            rate.sleep()
        cv2.destroyAllWindows()

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

user_manager = UserManager()
story_generator = StoryGenerator(llm_model="llama3.1:latest")
tts_helper = TTSHelper()
image_generator = ImageGenerator()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
USER_DATA_DIR = os.path.join(BASE_DIR, 'user_data')

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

def clean_story_text(text):
    """
    Clean story text by removing asterisks, emojis, and other formatting symbols
    that should not be spoken or displayed in sentences
    
    Args:
        text: Raw story text
        
    Returns:
        str: Cleaned text suitable for speech and display
    """
    if not text:
        return text
    
    # Remove markdown formatting
    cleaned = text.replace('**', '').replace('*', '')
    
    # Remove emojis and special symbols, but preserve punctuation and letters
    # This regex targets emoji characters and other symbols while keeping text and punctuation
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
            games.append({
                "filename": fname,
                "created_at": created_at,
                "blocks_count": len(blocks)
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
    
    # Prepare user stories directory
    user_dir = os.path.join(USER_DATA_DIR, username, "stories")
    os.makedirs(user_dir, exist_ok=True)
    
    # Use timestamp for unique filename
    import datetime
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"story_{ts}.json"
    fpath = os.path.join(user_dir, fname)
    
    # Save story and metadata
    with open(fpath, "w") as f:
        json.dump({"story": story, "metadata": metadata}, f, indent=2)
    
    # Generate images for all sentences in the story
    if image_generator.is_available():
        try:
            # Split story into paragraphs (blank-line separated)
            paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", story.strip()) if p.strip()]

            # Create user-specific image directory
            user_images_dir = os.path.join(USER_DATA_DIR, username, "story_images", fname.replace(".json", ""))
            os.makedirs(user_images_dir, exist_ok=True)

            # Generate images for each paragraph
            image_paths = []
            for i, paragraph in enumerate(paragraphs):
                image_path = image_generator.generate_story_scene_image(
                    paragraph,
                    story_context=f"Story about {metadata.get('child_name', 'a child')}",
                    output_dir=user_images_dir,
                    filename_prefix=f"story_paragraph_{i:03d}",
                )
                image_paths.append(image_path)

            print(f"Generated {len(image_paths)} images for story {fname}")

        except Exception as e:
            print(f"Error generating images for story {fname}: {str(e)}")
            # Continue even if image generation fails
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

@app.route('/api/scene_game/new_round', methods=['POST'])
def api_scene_game_new_round():
    """Start a new scene detection round: show random objects and ask a question"""
    if 'username' not in session:
        return jsonify({'success': False, 'error': 'Not logged in'}), 401
    try:
        # Define a small vocabulary
        fruits = ['apple', 'banana', 'orange', 'grape', 'strawberry', 'watermelon']
        others = ['book', 'car', 'chair', 'dog', 'cat', 'ball', 'cup', 'pencil']
        import random as _rnd
        _rnd.shuffle(fruits)
        _rnd.shuffle(others)
        # Build candidates: 2 fruits + 3 others
        candidates = fruits[:2] + others[:3]
        _rnd.shuffle(candidates)

        # Question
        question = "I want a fruit, show me a fruit"

        # Make the robot ask the question
        try:
            _with_asr_suspended(lambda: tts_helper.speak_story(question, 'en-US'))
        except Exception:
            pass

        # Start human tracking during play mode (non-blocking)
        try:
            tracker = _ensure_human_tracker()
            if tracker:
                # pick most recent visible person and track; if none, tracker.track(None) will neutral gaze
                person = _pick_recent_person(tracker, timeout_sec=0.5)
                tracker.track(person)
        except Exception as e:
            print(f"HumanTracking start error: {e}")

        # Generate or map images
        items = []
        img_dir = os.path.join(USER_DATA_DIR, 'activity_images')
        os.makedirs(img_dir, exist_ok=True)
        for label in candidates:
            img_url = None
            if image_generator.is_available():
                # save as <label>.png (collision-safe)
                safe = re.sub(r"[^A-Za-z0-9_-]+", "_", label)
                target = os.path.join(img_dir, f"{safe}.png")
                if not os.path.exists(target):
                    path = image_generator.generate_image(
                        prompt=f"{label}, single object on simple background, children's book illustration",
                        output_dir=img_dir,
                        filename_prefix=f"scene_{safe}"
                    )
                    if path and not os.path.exists(target):
                        try:
                            os.replace(path, target)
                        except Exception:
                            target = path
                rel = os.path.relpath(target, USER_DATA_DIR)
                img_url = f"/images/{rel}"
            items.append({
                'label': label,
                'is_fruit': label in fruits,
                'image_path': img_url
            })

        return jsonify({'success': True, 'question': question, 'items': items, 'target': 'fruit'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/scene_game/answer', methods=['POST'])
def api_scene_game_answer():
    if 'username' not in session:
        return jsonify({'success': False, 'error': 'Not logged in'}), 401
    data = request.get_json() or {}
    label = (data.get('label') or '').lower().strip()
    fruits = {'apple','banana','orange','grape','strawberry','watermelon'}
    correct = label in fruits
    try:
        if correct:
            tts_helper.speak("Great job! That's a fruit!")
        else:
            tts_helper.speak("Not quite. Try again and pick a fruit.")
    except Exception:
        pass
    # Continue tracking implicitly; you can stop at end of session via endpoint below
    return jsonify({'success': True, 'correct': correct})

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
                cmd = ['python3.9', script_path, '--image', fpath]

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

@app.route("/builder")
def builder_page():
    """DIY activity builder page"""
    if 'username' not in session:
        return redirect(url_for('index'))
    username = session['username']
    user = user_manager.users.get(username)
    return render_template("diy_builder.html", logged_in=True, user=user)

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
        _execute_activity(blocks, loop_count)
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

def _execute_activity(blocks, loop_count):
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
                        elif ttype == 'gesture' and kin:
                            g = (tblock.get('name') or tblock.get('gesture') or '').lower()
                            if g == 'nod':
                                kin._move_part('head', [0.0, 8.0], sync=False); time.sleep(0.4)
                                kin._move_part('head', [0.0, -4.0], sync=False); time.sleep(0.3)
                                kin._move_part('head', [0.0, 0.0], sync=False)
                            elif g == 'wave':
                                kin._move_part('right_arm', [-70.0, -10.0, -20.0], sync=False)
                                for _i in range(3):
                                    kin._move_part('right_arm', [-70.0, -25.0, -20.0], sync=False); time.sleep(0.25)
                                    kin._move_part('right_arm', [-70.0, 0.0, -20.0], sync=False); time.sleep(0.25)
                                kin._move_part('right_arm', [-55.0, -20.0, -20.0], sync=False)

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

    # Non-continuous mode: Execute in strict sequence A..Z then repeat, as requested
    for _ in range(max(1, loop_count)):
        for block in exec_blocks:
            btype = block.get("type")
            if btype == "speech":
                text = block.get("text", "")
                if text:
                    _with_asr_suspended(lambda: tts_helper.speak_story(text, "en-US"))
            elif btype == "praise":
                _with_asr_suspended(lambda: tts_helper.speak("Great job!"))
            elif btype == "gesture":
                name = (block.get("name") or block.get("gesture") or "").lower()
                if kin:
                    if name == "nod":
                        kin._move_part('head', [0.0, 8.0], sync=False)
                        time.sleep(0.4)
                        kin._move_part('head', [0.0, -4.0], sync=False)
                        time.sleep(0.3)
                        kin._move_part('head', [0.0, 0.0], sync=False)
                    elif name == "wave":
                        kin._move_part('right_arm', [-70.0, -10.0, -20.0], sync=False)
                        for _i in range(3):
                            kin._move_part('right_arm', [-70.0, -25.0, -20.0], sync=False)
                            time.sleep(0.25)
                            kin._move_part('right_arm', [-70.0, 0.0, -20.0], sync=False)
                            time.sleep(0.25)
                        kin._move_part('right_arm', [-55.0, -20.0, -20.0], sync=False)
            elif btype == "recognize":
                target = (block.get("target") or "speech").lower()
                value = (block.get("value") or "").strip()
                _announce_wait_once()
                if target == 'speech' and value:
                    expected = value.strip().lower()
                    start_wait = time.time()
                    max_wait_seconds = 30
                    try:
                        while time.time() - start_wait < max_wait_seconds:
                            guard_start = time.time()
                            while getattr(tts_helper, 'is_speaking', lambda: False)() and time.time() - guard_start < 10:
                                time.sleep(0.05)
                            text = _whisper_recognize_streaming()
                            heard_raw = (text or '').strip().lower()
                            import re as _re
                            heard = _re.sub(r"[^a-z0-9\s]", "", heard_raw)
                            print(f"[Recognize ASR] expected='{expected}' heard='{heard_raw}' -> norm='{heard}'")
                            # Try fuzzy and LLM correction before matching when expected is not contained in heard
                            if heard and expected not in heard:
                                fuzzy = _fuzzy_canonicalize_heard(expected, heard)
                                if fuzzy:
                                    print(f"[Recognize ASR] fuzzy corrected '{heard_raw}' -> '{fuzzy}'")
                                    heard = fuzzy
                                else:
                                    print(f"[Recognize ASR] LLM correcting '{heard_raw}'")
                                    corrected = _llm_canonicalize_heard(expected, heard, context="DIY activity recognize block")
                                    if corrected:
                                        print(f"[Recognize ASR] corrected '{heard_raw}' -> '{corrected}'")
                                        heard = corrected.lower()
                            if heard and expected in heard:
                                break
                    except Exception as e:
                        print(f"ASR error: {e}")
            elif btype == "image":
                pass
            elif btype == "wait":
                pass
            elif btype == "logic":
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
                    elif ttype == 'gesture' and kin:
                        g = (tblock.get('name') or tblock.get('gesture') or '').lower()
                        if g == 'nod':
                            kin._move_part('head', [0.0, 8.0], sync=False); time.sleep(0.4)
                            kin._move_part('head', [0.0, -4.0], sync=False); time.sleep(0.3)
                            kin._move_part('head', [0.0, 0.0], sync=False)
                        elif g == 'wave':
                            kin._move_part('right_arm', [-70.0, -10.0, -20.0], sync=False)
                            for _i in range(3):
                                kin._move_part('right_arm', [-70.0, -25.0, -20.0], sync=False); time.sleep(0.25)
                                kin._move_part('right_arm', [-70.0, 0.0, -20.0], sync=False); time.sleep(0.25)
                            kin._move_part('right_arm', [-55.0, -20.0, -20.0], sync=False)

                if any((c.get('type') == 'recognize' and (c.get('target') or 'speech').lower() == 'speech' and (c.get('value') or '').strip()) for c in cond_blocks):
                    _announce_wait_once()

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
                                try:
                                    guard_start = time.time()
                                    while getattr(tts_helper, 'is_speaking', lambda: False)() and time.time() - guard_start < 10:
                                        time.sleep(0.05)
                                    text = _whisper_recognize_once()
                                    heard = (text or '').strip().lower()
                                    print(f"[Logic ASR] expected='{exp}' heard='{heard}'")
                                    # Try LLM correction before matching
                                    if heard and exp not in heard:
                                        corrected = _llm_canonicalize_heard(exp, heard, context="DIY logic recognize")
                                        if corrected:
                                            print(f"[Logic ASR] corrected '{heard}' -> '{corrected}'")
                                            heard = corrected.lower()
                                    if heard and exp in heard:
                                        exec_then_block(tblock)
                                except Exception as e:
                                    print(f"ASR error: {e}")
                            th = Thread(target=worker, daemon=True)
                            th.start()
                            threads.append(th)
                for th in threads:
                    th.join(timeout=0.1)

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
        # If continuous recognizers exist, run in background until stopped
        global _activity_thread
        _activity_stop_event.clear()
        global _asr_enabled
        _asr_enabled = True
        if _has_parallel_recognizers(blocks):
            def runner():
                try:
                    _execute_activity(blocks, loop_count)
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
            return jsonify({"success": True, "running": True})
        else:
            _execute_activity(blocks, loop_count)
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
        story_text = story_data.get('story', '')
        metadata = story_data.get('metadata', {})
        
        # Clean the story text first
        cleaned_story = clean_story_text(story_text)
        
        # Split into sentences (simple split, can be improved)
        sentences = re.split(r'(?<=[.!?])\s+', cleaned_story.strip())
        sentences = [s for s in sentences if s.strip()]
        
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
    """Get image for a specific sentence (mapped to its paragraph image)"""
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

        # Load story + metadata
        with open(story_path, 'r') as f:
            story_data = json.load(f)
        metadata = story_data.get('metadata', {}) or {}

        # Use saved mapping if available
        mapping = metadata.get('sentence_to_paragraph') or []
        if mapping:
            if sentence_index < 0 or sentence_index >= len(mapping):
                return jsonify({'success': False, 'error': 'Sentence index out of range'})
            paragraph_index = mapping[sentence_index]
        else:
            # Fallback: compute from story text
            story_text = clean_story_text(story_data.get('story', '')).strip()
            paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", story_text) if p.strip()]

            sent_to_para = []
            for pi, p in enumerate(paragraphs):
                sents = re.split(r'(?<=[.!?])\s+', p.strip())
                for s in [x for x in sents if x.strip()]:
                    sent_to_para.append(pi)

            if sentence_index < 0 or sentence_index >= len(sent_to_para):
                return jsonify({'success': False, 'error': 'Sentence index out of range'})
            paragraph_index = sent_to_para[sentence_index]

        # Find the paragraph image
        pattern = f"story_paragraph_{int(paragraph_index):03d}_"
        matches = [f for f in os.listdir(user_images_dir) if f.startswith(pattern) and f.endswith('.png')]
        matches.sort()

        if not matches:
            return jsonify({'success': False, 'error': 'No image found for this paragraph'})

        image_path = f"/images/{username}/story_images/{os.path.basename(user_images_dir)}/{matches[0]}"
        return jsonify({'success': True, 'image_path': image_path, 'paragraph_index': paragraph_index})

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

@app.route('/api/scene/start', methods=['POST'])
def api_scene_start():
    try:
        choices = ['tomato', 'lemon', 'apple', 'banana']
        target = random.choice(choices)
        try:
            _with_asr_suspended(lambda: tts_helper.speak(f"Show me a {target}"))
        except Exception:
            pass
        return jsonify({'success': True, 'target': target})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True) 