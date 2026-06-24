#!/usr/bin/env python3.9

import os
from flask import Flask, render_template, request, jsonify, session, redirect, url_for, Response, send_from_directory, send_file,make_response
from user_management import UserManager
from story_generator import StoryGenerator
from knowledge_base import LanguageInterestKB
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
# When init fails (e.g. the robot motor controllers aren't running, so the
# head-command topic has no subscriber) every request that wants tracking
# would otherwise re-attempt and block ~5s on the ROS timeout, then re-log the
# same scary error. Remember the failure and skip retrying for a cooldown, so
# the app stays responsive but still self-heals once the motors come back.
_human_tracker_failed_at = 0.0
_HUMAN_TRACKER_RETRY_COOLDOWN = 60.0  # seconds

# V4L2 camera handle (non-ROS fallback)
_v4l_cap = None
_v4l_cap_lock = Lock()


#loading env variables
from dotenv import load_dotenv
load_dotenv()

def _ensure_human_tracker():
    global _human_tracker, _human_tracker_failed_at
    if not HUMAN_TRACKING_AVAILABLE:
        return None
    with _human_tracker_lock:
        if _human_tracker is not None:
            return _human_tracker
        # Don't keep re-attempting (each try blocks ~5s on the ROS timeout)
        # while we're inside the cooldown after a recent failure.
        if _human_tracker_failed_at and (
                time.time() - _human_tracker_failed_at < _HUMAN_TRACKER_RETRY_COOLDOWN):
            return None
        try:
            _human_tracker = HumanTracking()
            _human_tracker_failed_at = 0.0
        except Exception as e:
            _human_tracker_failed_at = time.time()
            print(
                "[HumanTracking] Robot motors not available — head tracking "
                "disabled. The '/qt_robot/head_position/command' topic has no "
                "subscriber, so the QTrobot motor controllers are likely not "
                f"running. The app will keep working without head movement and "
                f"retry in {int(_HUMAN_TRACKER_RETRY_COOLDOWN)}s. (detail: {e})"
            )
            return None
    return _human_tracker

def _pause_human_tracking_for_capture():
    """Stop continuous human tracking so the robot's head stays still.

    Used while the therapist is framing a picture-scene shot for the WH-questions
    inference game: head motion would blur the camera preview and the captured
    frame. Safe to call when tracking is already off.
    """
    try:
        tracker = _ensure_human_tracker()
        if tracker and getattr(tracker, 'should_track', False):
            tracker.untrack()
            print("[WH-scene] Human tracking paused for scene capture")
    except Exception as e:
        print(f"[WH-scene] tracking pause error: {e}")


def _resume_human_tracking_after_capture():
    """Re-enable continuous human tracking after a scene capture click.

    Mirrors the auto-start used by /play and /play_scene so the robot resumes
    following the child once the therapist has captured the scene. Safe to call
    when tracking is already running.
    """
    try:
        tracker = _ensure_human_tracker()
        if tracker and not getattr(tracker, 'should_track', False):
            person = _pick_recent_person(tracker, timeout_sec=0.5)
            tracker.track(person)
            print("[WH-scene] Human tracking resumed after scene capture")
    except Exception as e:
        print(f"[WH-scene] tracking resume error: {e}")


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
story_generator = StoryGenerator(llm_model="claude-sonnet-4-6")
knowledge_base = LanguageInterestKB()
tts_helper = TTSHelper()
image_generator = ImageGenerator()


# ===== live step tracing (added for debugging visibility) =====
# Mirrors every print()/stderr line into logs/trace.log with a timestamp and a
# per-request id, and frames each HTTP "work" with begin/done markers + timing.
# Fully additive: if this block is removed the app behaves exactly as before.
import threading as _threading

TRACE_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(TRACE_DIR, exist_ok=True)
_trace_local = _threading.local()

# Daily-rotated trace files: a new logs/trace-YYYY-MM-DD.log is opened whenever
# the calendar date changes (checked at each line write), so each day's logs
# land in their own file.
_trace_rotate_lock = _threading.Lock()
_trace_state = {"date": None, "fh": None}


def _trace_path_for(date_str):
    return os.path.join(TRACE_DIR, f"trace-{date_str}.log")


def _trace_fh_for_today():
    """Append handle for today's dated trace file.

    Opens a new file (closing the previous one) the first time a line is logged
    on a new calendar date — i.e. when the date differs from the currently-open
    log's date.
    """
    today = time.strftime('%Y-%m-%d')
    fh = _trace_state["fh"]
    if _trace_state["date"] == today and fh is not None:
        return fh
    with _trace_rotate_lock:
        if _trace_state["date"] != today or _trace_state["fh"] is None:
            old = _trace_state["fh"]
            _trace_state["fh"] = open(_trace_path_for(today), "a", buffering=1)
            _trace_state["date"] = today
            if old is not None:
                try:
                    old.flush()
                    old.close()
                except Exception:
                    pass
        return _trace_state["fh"]

# High-frequency polling endpoints we don't want to frame (keeps the trace readable).
_TRACE_SKIP = {
    "/api/movement_status", "/api/volume_status", "/api/current_user",
    "/api/camera_frame", "/api/human_tracking/status", "/generate",
    "/play", "/favicon.ico",
}


class _Tee:
    """Write to the original stream and also to the trace file, timestamping at line boundaries."""
    def __init__(self, original):
        self._original = original
        self._buf = ""
        self._lock = _threading.Lock()

    def write(self, data):
        self._original.write(data)
        with self._lock:
            self._buf += data
            while "\n" in self._buf:
                line, self._buf = self._buf.split("\n", 1)
                rid = getattr(_trace_local, "rid", "----")
                fh = _trace_fh_for_today()
                fh.write(f"{time.strftime('%H:%M:%S')} [{rid}] {line}\n")
                fh.flush()

    def flush(self):
        self._original.flush()
        fh = _trace_state["fh"]
        if fh is not None:
            try:
                fh.flush()
            except Exception:
                pass

    def isatty(self):
        return getattr(self._original, "isatty", lambda: False)()

    def fileno(self):
        return self._original.fileno()


if not getattr(sys, "_trace_installed", False):
    _trace_fh_for_today()  # open today's dated file up front
    sys.stdout = _Tee(sys.stdout)
    sys.stderr = _Tee(sys.stderr)
    sys._trace_installed = True
    print(f"[trace] step tracing active -> {_trace_path_for(time.strftime('%Y-%m-%d'))} (daily rotation)")


@app.before_request
def _trace_begin():
    _trace_local.rid = uuid.uuid4().hex[:4]
    _trace_local.t0 = time.time()
    if request.path in _TRACE_SKIP or request.path.startswith("/static"):
        _trace_local.framed = False
        return
    _trace_local.framed = True
    detail = ""
    try:
        if request.method in ("POST", "PUT", "PATCH"):
            j = request.get_json(silent=True)
            if isinstance(j, dict) and j:
                detail = " | fields: " + ", ".join(list(j.keys())[:12])
        elif request.args:
            detail = " | args: " + ", ".join(list(request.args.keys())[:12])
    except Exception:
        pass
    print(f"┌─ WORK  {request.method} {request.path}{detail}")


@app.after_request
def _trace_end(resp):
    if getattr(_trace_local, "framed", False):
        dt = (time.time() - getattr(_trace_local, "t0", time.time())) * 1000.0
        print(f"└─ DONE  {request.method} {request.path} -> {resp.status_code} in {dt:.0f}ms")
    return resp
# ===== end live step tracing =====


def _load_user_profile(username):
    """Merge user_manager entry with any profile.json overrides. Returns a dict."""
    user = user_manager.users.get(username) or {}
    profile = {
        "age": user.get("age"),
        "gender": user.get("gender", ""),
        "disorder": user.get("disorder", ""),
        "learning_goals": user.get("learning_goals", ""),
        "language_age": user.get("language_age"),
    }
    try:
        profile_path = os.path.join(USER_DATA_DIR, username, "profile.json")
        if os.path.exists(profile_path):
            with open(profile_path, "r") as pf:
                pdata = json.load(pf)
            for key in ("age", "gender", "disorder", "learning_goals", "language_age"):
                if pdata.get(key) not in (None, ""):
                    profile[key] = pdata[key]
    except Exception as e:
        print(f"[Profile] Failed to read profile.json for {username}: {e}")
    return profile


def _persona_context_for(username, age, kind="story"):
    """Build a language + interest knowledge-base fragment for `username`.

    Derives developmentally-appropriate language targets (by age) and interest
    themes (by age + gender) from the SLP co-design knowledge base.

    kind: "story" -> narrative-shaped fragment; "question" -> compact fragment.
    Returns an empty string if nothing can be derived.
    """
    try:
        profile = _load_user_profile(username)
        gender = profile.get("gender", "") or ""
        effective_age = age if age is not None else profile.get("age") or 0
        # Optional developmental/language age: when a child's language level
        # differs from their chronological age (e.g. a 9-year-old targeted at
        # MLU 6-8), profile.json may carry "language_age". None -> use age.
        language_age = profile.get("language_age")
        if kind == "question":
            fragment = knowledge_base.build_question_prompt_fragment(
                effective_age, gender, language_age=language_age)
        else:
            fragment = knowledge_base.build_story_prompt_fragment(
                effective_age, gender, language_age=language_age)
        if fragment:
            info = knowledge_base.describe(effective_age, gender, language_age=language_age)
            print(f"[KB] derived level_age={info.get('level_age')} mlu={info.get('mlu_range')} "
                  f"targets={info.get('targets')} "
                  f"speech[{info.get('speech_age_range')}]={info.get('speech_sounds')} "
                  f"interests={info.get('interests')} for "
                  f"user={username} age={effective_age} language_age={language_age} "
                  f"gender='{gender}' kind={kind}")
        return fragment or ''
    except Exception as e:
        print(f"[KB] context build failed for {username}: {e}")
        return ''


def _language_age_for(username, age):
    """Effective developmental/language age for activity complexity decisions.

    Returns profile.language_age when set (a child whose language level differs
    from their chronological age, e.g. a 9-year-old targeted at MLU 6-8), else
    the chronological age. Used so question/scene/image activities pitch their
    complexity at the child's language level — consistent with how the story
    generator and the knowledge base treat language_age.
    """
    try:
        la = _load_user_profile(username).get("language_age")
        if la is not None:
            return int(la)
    except Exception:
        pass
    return age

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
                    model="claude-sonnet-4-6",
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

class _GeminiQuizLLM:
    """Quiz LLM backed by Gemini Flash via the gemini_general.py subprocess worker.

    Exposes the same minimal `.get_response(prompt).message.content` interface
    as the previous ChatWithRAG-based quiz LLM, so existing call sites keep working.
    """
    SYSTEM_ROLE = (
        "You create short, child-friendly quiz questions. "
        "Return JSON only. Each item must be {\"question\": \"...\", \"type\": \"yes_no\"|\"wh\"}."
    )

    def __init__(self, max_tokens=2048, temperature=0.3):
        self.max_tokens = max_tokens
        self.temperature = temperature

    def get_response(self, prompt):
        text = _gemini_generate(
            prompt,
            system=self.SYSTEM_ROLE,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        ) or ""

        class _Msg:
            def __init__(self, content):
                self.content = content

        class _Resp:
            def __init__(self, content):
                self.message = _Msg(content)

        return _Resp(text)


def _ensure_quiz_llm():
    global _quiz_llm
    with _quiz_llm_lock:
        if _quiz_llm is None:
            try:
                _quiz_llm = _GeminiQuizLLM(max_tokens=8192)
            except Exception as e:
                print(f"Warning: failed to initialize quiz LLM: {e}")


def _parse_json_array(raw: str):
    """
    Robust JSON-array extractor for LLM quiz responses.

    Handles the common failure modes that produced "LLM returned invalid JSON"
    in 'both' mode:
      - ```json ... ``` markdown code fences.
      - Top-level object wrapping the array, e.g. {"questions": [...]}.
      - Preamble / postamble text around the JSON.
      - Trailing commas in arrays/objects.
      - Truncated output (best-effort recovery via bracket slicing).

    Returns a Python list on success, or None on failure.
    """
    if not raw:
        return None

    s = raw.strip()

    # 1. Strip ```json ... ``` or ``` ... ``` code fences.
    if s.startswith("```"):
        # Remove first fence line (``` or ```json or ```JSON etc.)
        first_nl = s.find("\n")
        if first_nl != -1:
            s = s[first_nl + 1:]
        # Drop trailing fence.
        if s.rstrip().endswith("```"):
            s = s.rstrip()[:-3]
        s = s.strip()
    # Some models prefix the language right after a single line of backticks.
    if s.lower().startswith("json\n"):
        s = s[5:].lstrip()

    def _try_load(text):
        try:
            return json.loads(text)
        except Exception:
            return None

    def _unwrap(obj):
        """If obj is a dict, look for the array inside it."""
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict):
            for key in ("questions", "data", "items", "result", "results", "list"):
                v = obj.get(key)
                if isinstance(v, list):
                    return v
            # Fallback: first list-valued field.
            for v in obj.values():
                if isinstance(v, list):
                    return v
        return None

    # 2. Try parsing as-is.
    obj = _try_load(s)
    arr = _unwrap(obj)
    if arr is not None:
        return arr

    # 3. Slice between first '[' and last ']'.
    l = s.find("[")
    r = s.rfind("]")
    if l != -1 and r != -1 and r > l:
        sliced = s[l:r + 1]
        obj = _try_load(sliced)
        arr = _unwrap(obj)
        if arr is not None:
            return arr

        # 4. Light repair: drop trailing commas before } or ].
        import re as _re
        repaired = _re.sub(r",(\s*[\]\}])", r"\1", sliced)
        obj = _try_load(repaired)
        arr = _unwrap(obj)
        if arr is not None:
            return arr

    # 5. Truncation recovery. If the array starts with '[' but is cut off
    #    (the LLM hit the token cap mid-object), walk the string with
    #    string-aware bracket/brace tracking, find the last position where a
    #    top-level object closed, and synthesize the closing ']'.
    if l != -1:
        tail = s[l:]
        in_string = False
        escape = False
        bracket_depth = 0
        brace_depth = 0
        last_top_close = -1
        for i, ch in enumerate(tail):
            if escape:
                escape = False
                continue
            if in_string:
                if ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == "[":
                bracket_depth += 1
            elif ch == "]":
                bracket_depth -= 1
            elif ch == "{":
                brace_depth += 1
            elif ch == "}":
                brace_depth -= 1
                if bracket_depth == 1 and brace_depth == 0:
                    last_top_close = i
        if last_top_close > 0:
            candidate = tail[: last_top_close + 1] + "]"
            obj = _try_load(candidate)
            arr = _unwrap(obj)
            if arr is not None:
                return arr

    return None

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


# Inline gesture/emotion tags can sit on the same line as a stray title; the
# title-detection heuristic ignores them when measuring line length and
# punctuation, but the title-stripping path removes the whole line (tag
# included), since a tag bound to a title doesn't apply to the surviving body.
_INLINE_TAG_RE = re.compile(r'\[(?:gesture|emotion):[^\]]+\]\s*', re.IGNORECASE)


def _strip_leading_title(text):
    """Defensive sweep that removes a stray title-like prefix from a story body.

    Even though the story-generation prompt forbids a title in the body, the
    LLM occasionally emits one. `_extract_story_title()` handles the canonical
    "** Title **\\n<title>\\n\\n<body>" header and the loose
    "<title>\\n\\n<body>" form (when the title has no sentence-ending
    punctuation), but several patterns slip past it:
      - "Title: My Adventure\\n<body>"     (inline marker, no newline split)
      - "# My Adventure\\n<body>"          (markdown heading)
      - "**My Adventure**\\n<body>"        (bolded line alone)
      - "My Adventure\\n<body>"            (no blank-line separator)
    Returns the text with the title line(s) removed when detected, or the
    original text unchanged. Idempotent: safe to call multiple times.
    """
    if not text or not text.strip():
        return text

    s = text.lstrip()

    # "** Title **\n<title>\n" header — drop the marker AND the title line.
    m = re.match(r'^\*{1,3}\s*Title\s*\*{1,3}\s*\n+[^\n]+\n+', s, flags=re.IGNORECASE)
    if m:
        return s[m.end():]

    # Markdown heading "# <title>".
    m = re.match(r'^#{1,3}\s+[^\n]+(?:\n+|$)', s)
    if m:
        return s[m.end():]

    # "Title: <title>" inline marker.
    m = re.match(r'^Title\s*:\s*[^\n]+(?:\n+|$)', s, flags=re.IGNORECASE)
    if m:
        return s[m.end():]

    # Bolded/italicized line that contains no other asterisks, alone on a line.
    m = re.match(r'^\*{1,3}[^*\n]+\*{1,3}\s*(?:\n+|$)', s)
    if m:
        return s[m.end():]

    # Loose heuristic: a short first line WITHOUT sentence-ending punctuation,
    # followed by real story content. Mirrors `_extract_story_title`'s Format 2
    # but without requiring a blank-line separator (which is the LLM behaviour
    # that slips past it). Inline gesture/emotion tags don't count toward the
    # title-detection metrics.
    first_nl = s.find('\n')
    if first_nl == -1:
        return text  # single line — nothing to do
    first = s[:first_nl]
    rest = s[first_nl + 1:].lstrip('\n')
    if not rest.strip():
        return text  # no body after the candidate — leave alone

    first_clean = _INLINE_TAG_RE.sub('', first).strip().strip('*').strip()

    if (first_clean
            and len(first_clean) < 80
            and not re.search(r'[.!?][\'"\")\]]?\s*$', first_clean)):
        return rest

    return text


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
    # Optional developmental/language age (decoupled from chronological age).
    language_age = data.get("language_age", user.get("language_age"))

    try:
        profile_path = os.path.join(USER_DATA_DIR, username, "profile.json")
        if os.path.exists(profile_path):
            with open(profile_path, "r") as pf:
                profile = json.load(pf)
            learning_goals = profile.get("learning_goals", learning_goals)
            gender = profile.get("gender", gender)
            if data.get("language_age") is None and profile.get("language_age") is not None:
                language_age = profile.get("language_age")
    except Exception as e:
        print(f"Warning: failed to read profile.json: {e}")
    persona_context = _persona_context_for(username, age, kind="story")

    try:
        result = story_generator.generate_story(
            child_name=child_name,
            age=age,
            gender=gender,
            custom_prompt=custom_prompt,
            topics=topics,
            goals=learning_goals,
            persona_context=persona_context,
            language_age=language_age,
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
    language_age = data.get("language_age", user.get("language_age"))
    try:
        profile_path = os.path.join(USER_DATA_DIR, username, "profile.json")
        if os.path.exists(profile_path):
            with open(profile_path, "r") as pf:
                profile = json.load(pf)
            learning_goals = profile.get("learning_goals", learning_goals)
            gender = profile.get("gender", gender)
            if data.get("language_age") is None and profile.get("language_age") is not None:
                language_age = profile.get("language_age")
    except Exception as e:
        print(f"Warning: failed to read profile.json: {e}")

    def generate():
        try:
            # Send initial metadata event for streaming clients
            meta = {
                "child_name": child_name,
                "age": age,
                "language_age": language_age,
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
                goals=learning_goals,
                persona_context=_persona_context_for(username, age, kind="story"),
                language_age=language_age,
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

    # Detect social-rules topic: triggers a specialised yes/no prompt that asks
    # about kindness, manners, sharing, and respect. Designed for age 7+ but
    # works at any difficulty when explicitly selected.
    SOCIAL_RULE_KEYWORDS = (
        "social rule", "social rules", "social norm", "social norms",
        "etiquette", "manners", "good manners", "kindness", "behavior", "behaviour",
    )
    is_social_rules = any(
        any(kw in t.lower() for kw in SOCIAL_RULE_KEYWORDS) for t in topics
    )
    use_social_rules_branch = is_social_rules and ("yes_no" in types)

    topic_text = ", ".join(topics)

    if use_social_rules_branch:
        # Specialised goal + examples for social-rules yes/no questions.
        # Every example answer must be a clear, widely accepted yes or no.
        # Avoid subjective, vague, culturally specific, or gray-area questions.
        goal_text = (
            "Goal: Generate diverse yes/no questions about social rules, etiquette, kindness, "
            "safety, and basic social norms that a child should learn. Every question MUST have "
            "a clear, widely accepted yes-or-no answer, not an opinion, preference, or gray-area "
            "judgment. The aim is to help children think about expected vs. unexpected behavior, "
            "right vs. wrong actions, and how their behavior affects others. "

            "Cover a diverse mix of the following categories: "
            "1) physical kindness and safety, such as no hitting, kicking, pushing, biting, or throwing objects at people; "
            "2) sharing, turn-taking, and fairness, such as waiting, taking turns, and playing fairly; "
            "3) polite words and manners, such as please, thank you, sorry, excuse me, and greetings; "
            "4) classroom and group behavior, such as listening, raising hands, following rules, and not disrupting others; "
            "5) respecting belongings and personal space, such as asking before touching, borrowing, or entering someone's space; "
            "6) helping and caring for others, such as helping someone who falls or comforting someone who is sad; "
            "7) honesty and responsibility, such as telling the truth, admitting mistakes, and cleaning up after oneself; "
            "8) inclusion and friendship, such as inviting others, not excluding someone, and using kind words; "
            "9) privacy and boundaries, such as not opening private bags, not touching others without permission, and respecting 'no'; "
            "10) community rules and public behavior, such as waiting in line, using an indoor voice, and being careful in shared spaces. "

            "Generate questions from multiple categories rather than repeating the same type of rule. "
            "Use concrete child-friendly situations. Prefer questions about observable actions, not feelings or preferences. "
            "Each question should test one rule only. Do not use complicated moral dilemmas. "

            "Examples of GOOD questions and their answers: "
            "'Is it okay to kick your friend?' → no. "
            "'Should you say thank you when someone helps you?' → yes. "
            "'Is it okay to take a toy without asking?' → no. "
            "'Should you wait your turn in line?' → yes. "
            "'Is it polite to interrupt someone speaking?' → no. "
            "'Should you say sorry when you hurt someone?' → yes. "
            "'Is it okay to laugh at someone who made a mistake?' → no. "
            "'Should you share with a friend who has none?' → yes. "
            "'Is it okay to push someone to go first?' → no. "
            "'Should you raise your hand before speaking in class?' → yes. "
            "'Is it okay to open someone's backpack without asking?' → no. "
            "'Should you help someone who dropped their crayons?' → yes. "
            "'Is it okay to tell a lie to avoid trouble?' → no. "
            "'Should you include a classmate who is left out?' → yes. "
            "'Is it okay to grab a toy from someone?' → no. "
            "'Should you use a quiet voice in the library?' → yes. "
            "'Is it okay to call someone a mean name?' → no. "
            "'Should you clean up after making a mess?' → yes. "
            "'Is it okay to touch someone who says stop?' → no. "
            "'Should you say excuse me when you need to pass?' → yes. "

            "AVOID opinion, vague, absolute, or culturally dependent questions such as: "
            "'Do you like sharing?', 'Is school fun?', 'Should you always be nice?', "
            "'Is it bad to be angry?', 'Is it okay to be sad?', 'Should everyone be your friend?', "
            "'Is it okay to never say no?', or 'Should you give away all your toys?'. "
            "Avoid the word 'always' unless the rule is truly absolute. Avoid questions where the correct "
            "answer depends on context, culture, family rules, or personal preference."
        )

        length_constraint = (
            "Constraint: Questions must be short, under 12 words, and use simple language "
            "a 7- to 8-year-old child can understand."
        )
    else:
        goal_text = (
            "Goal: Questions must be objectively True or False based on basic object functions or category labels. "
            "Avoid subjective questions like 'Do you like school?' or 'Are there toys?'."
        )
        length_constraint = "Constraint: Questions must be short (under 8 words)."

    # Knowledge-base guidance (developmental language targets, articulation /
    # speech-sound targets, MLU sentence-length) derived from the child's profile
    # age / language_age — the same source the story-comprehension and scene-game
    # questions use, so the educational quiz pitches its wording and target speech
    # sounds to the same developmental level. The selected topic still governs the
    # question *content*, so the fragment's interest-theme suggestions are ignored.
    kb_context = _persona_context_for(username, None, kind="question")
    kb_block = ""
    if kb_context:
        kb_block = (
            "Developmental guidance for this child (from the SLP knowledge base) — use it ONLY "
            "to set the question wording level and to favour the target speech sounds; the "
            "questions must still be about the topic(s) above, so ignore its interest-theme "
            f"suggestions for this quiz:\n{kb_context}\n"
        )

    prompt = (
        f"Act as a pediatric educator. Create {count} questions about the topic(s) '{topic_text}'. "
        f"{age_hint} "
        f"Use only these types: {type_hint}. "
        f"{goal_text} "
        f"{length_constraint} "
        f"{kb_block}"
        "Return Format: Respond with ONE JSON array ONLY. The first non-whitespace character of your "
        "response MUST be '[' and the last MUST be ']'. Do NOT wrap the array inside an object "
        "(e.g. do NOT use {\"questions\": [...]}). Do NOT add commentary, markdown, or code fences. "
        "Each array element must be an object with keys: 'question', 'type', 'correct_answer', 'accepted_answers'. "
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
        obj = _parse_json_array(raw)
        if not isinstance(obj, list):
            snippet = raw[:500].replace("\n", " ")
            print(f"[quiz] Failed to parse LLM JSON. Raw start: {snippet!r}")
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
                alt_obj = _parse_json_array(alt_raw)
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

@app.route("/api/generate_wh_options", methods=["POST"])
def api_generate_wh_options():
    """Generate plausible distractor options for WH-quiz questions.

    Input JSON:
        {
            "questions": [{"question": str, "correct_answer": str,
                            "accepted_answers": [str, ...] (optional)}, ...],
            "num_options": 3 | 4
        }
    Output JSON:
        {"success": True,
         "options": [[opt1, opt2, opt3], ...]}  # each list contains the correct_answer
    """
    username = session.get('username')
    if not username:
        return jsonify({"success": False, "error": "Not logged in"}), 401
    data = request.get_json() or {}
    qs = data.get("questions") or []
    try:
        num_options = int(data.get("num_options", 3))
    except (TypeError, ValueError):
        num_options = 3
    num_options = max(2, min(4, num_options))

    if not isinstance(qs, list) or not qs:
        return jsonify({"success": False, "error": "questions required"}), 400

    _ensure_quiz_llm()
    if _quiz_llm is None:
        return jsonify({"success": False, "error": "Quiz LLM not available"}), 500

    distractor_count = num_options - 1
    llm_input = []
    index_map = []  # llm_input position -> original index in `qs` (for alignment)
    for idx, q in enumerate(qs):
        if not isinstance(q, dict):
            continue
        question_text = (q.get("question") or "").strip()
        correct = (q.get("correct_answer") or "").strip()
        accepted = q.get("accepted_answers") or []
        if not question_text or not correct:
            continue
        index_map.append(idx)
        llm_input.append({
            "question": question_text,
            "correct_answer": correct,
            "accepted_answers": [str(a).strip() for a in accepted if str(a).strip()],
        })

    if not llm_input:
        return jsonify({"success": False, "error": "no valid questions"}), 400

    prompt = (
        "You are creating multiple-choice options for a child's quiz. "
        f"For each question below, generate exactly {distractor_count} short, plausible-but-WRONG "
        "answer options that a child might consider. "
        "Rules:\n"
        "- Each option must be 1-3 words, child-friendly, and clearly different from the correct answer "
        "and from any of its accepted_answers.\n"
        "- Options should be in the same category as the correct answer (e.g. if the answer is an animal, "
        "give other animals).\n"
        "- Do NOT include the correct answer or any accepted answer in the distractors.\n"
        "- Do NOT include duplicates.\n"
        "Return JSON only: a list of lists, in the same order as the input. "
        f"Each inner list must contain exactly {distractor_count} distractor strings.\n"
        f"Input: {json.dumps(llm_input)}"
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
            l = raw.find('[')
            r = raw.rfind(']')
            if l != -1 and r != -1 and r > l:
                obj = json.loads(raw[l:r+1])
        if not isinstance(obj, list):
            return jsonify({"success": False, "error": "LLM returned invalid JSON"}), 500

        # Align options to the ORIGINAL input order so the client can map
        # options[i] -> questions[i] positionally. Questions that were skipped
        # (missing question text or correct_answer) keep an empty list, which the
        # client renders as the text-input fallback. Without this, dropping any
        # earlier question shifts every later question onto the wrong options.
        options_out = [[] for _ in qs]
        for i, q in enumerate(llm_input):
            distractors = obj[i] if i < len(obj) and isinstance(obj[i], list) else []
            distractors = [str(d).strip() for d in distractors if str(d).strip()]
            forbidden = {q["correct_answer"].lower()} | {a.lower() for a in q["accepted_answers"]}
            distractors = [d for d in distractors if d.lower() not in forbidden]
            seen = set()
            unique = []
            for d in distractors:
                key = d.lower()
                if key not in seen:
                    seen.add(key)
                    unique.append(d)
            unique = unique[:distractor_count]
            opts = [q["correct_answer"]] + unique
            options_out[index_map[i]] = opts

        return jsonify({"success": True, "options": options_out})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

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

def _generate_story_questions(story_text, child_age, child_name="the child", persona_context="",
                              language_age=None):
    """Generate comprehension questions for a story using Gemini.

    For all ages: questions about main idea and story details.
    For ages > 7: also include inference questions (e.g. character feelings/motivations).

    ``child_age`` is chronological (used for the child's stated identity).
    ``language_age`` (when given) is the developmental/language age that drives
    question complexity — count, wording, and whether deeper inference questions
    are included — so an older child with a language delay gets questions pitched
    at their language level. Falls back to ``child_age`` when None.

    Each question has 3 answer options (1 correct, 2 incorrect).

    Returns a list of dicts:
        [{"question": "...", "type": "...", "correct_answer": "...", "wrong_answers": ["...", "..."]}, ...]
    """
    cleaned = clean_story_text(story_text)
    if not cleaned.strip():
        return []

    complexity_age = language_age if language_age is not None else child_age
    if complexity_age <= 4:
        num_questions = 3
        detail_guidance = (
            "- 1 main idea question (e.g. 'What was the story about?')\n"
            "- 2 detail questions about characters, events, or objects in the story\n"
            "Use very simple language with short sentences (3-6 words per question).\n"
            "Keep answer options very short (1-5 words each)."
        )
    elif complexity_age <= 6:
        num_questions = 4
        detail_guidance = (
            "- 1 main idea question (e.g. 'What was the main thing that happened in the story?')\n"
            "- 3 detail questions about characters, events, settings, or objects\n"
            "Use simple language appropriate for a 5-6 year old.\n"
            "Keep answer options short (1-8 words each)."
        )
    else:
        num_questions = 5
        detail_guidance = (
            "- 1 main idea question (e.g. 'What is the main message of this story?')\n"
            "- 2 detail questions about characters, events, settings, or sequence of events\n"
            "- 2 inference questions that ask the child to think deeper, such as:\n"
            "  * 'Why do you think [character] felt that way?'\n"
            "  * 'What do you think would have happened if...?'\n"
            "  * 'How do you think [character] felt when...?'\n"
            "  * 'Why did [character] decide to...?'\n"
            "Use age-appropriate language for a 7-12 year old.\n"
            "Keep answer options concise (1-12 words each)."
        )

    persona_block = ''
    if persona_context and str(persona_context).strip():
        persona_block = "\n" + str(persona_context).strip() + "\n"

    prompt = (
        f"You are creating comprehension questions for a story read by a {child_age}-year-old child named {child_name}.\n\n"
        f"Story:\n{cleaned}\n"
        f"{persona_block}\n"
        f"Generate exactly {num_questions} questions:\n"
        f"{detail_guidance}\n\n"
        f"For each question, provide:\n"
        f"- 1 correct answer\n"
        f"- 2 plausible but incorrect answers\n"
        f"The wrong answers should be believable but clearly wrong based on the story.\n\n"
        f"Return ONLY a JSON array of objects. Each object has:\n"
        f"- \"question\": the question text\n"
        f"- \"type\": one of \"main_idea\", \"detail\", or \"inference\"\n"
        f"- \"correct_answer\": the correct answer text\n"
        f"- \"wrong_answers\": an array of exactly 2 incorrect answer texts\n\n"
        f"Example: [{{\"question\": \"What was the story about?\", \"type\": \"main_idea\", "
        f"\"correct_answer\": \"A boy who helped his friend\", \"wrong_answers\": [\"A girl who went swimming\", \"A cat who got lost\"]}}]\n"
    )

    raw = _gemini_generate(prompt, system="You generate comprehension questions for children's stories. Return JSON only.", max_tokens=4096)
    if raw:
        try:
            print(f"[StoryQuestions] Gemini raw response: {raw[:500]}")
            if raw.startswith('```'):
                raw = raw.strip('`').strip()
                if raw.startswith('json'):
                    raw = raw[4:].strip()
            start = raw.find('[')
            end = raw.rfind(']')
            if start != -1 and end != -1 and end > start:
                questions = json.loads(raw[start:end+1])
                if isinstance(questions, list) and len(questions) > 0:
                    # Validate structure
                    valid = []
                    for q in questions:
                        if (isinstance(q, dict) and 'question' in q and 'type' in q
                                and 'correct_answer' in q and 'wrong_answers' in q
                                and isinstance(q['wrong_answers'], list) and len(q['wrong_answers']) >= 2):
                            valid.append({
                                "question": q["question"],
                                "type": q["type"],
                                "correct_answer": q["correct_answer"],
                                "wrong_answers": q["wrong_answers"][:2],
                            })
                    if valid:
                        print(f"[StoryQuestions] Generated {len(valid)} questions for age {child_age}")
                        return valid
        except Exception as e:
            print(f"[StoryQuestions] Question generation failed: {e}")

    # Fallback: return basic questions with answer options
    print("[StoryQuestions] Using fallback questions")
    fallback = [
        {"question": "What was the story about?", "type": "main_idea",
         "correct_answer": "Helping a friend", "wrong_answers": ["Going to the store", "Playing alone"]},
        {"question": "Who was the main character in the story?", "type": "detail",
         "correct_answer": child_name, "wrong_answers": ["A teacher", "A stranger"]},
        {"question": "What happened at the end of the story?", "type": "detail",
         "correct_answer": "Everyone was happy", "wrong_answers": ["Everyone was sad", "Nothing happened"]},
    ]
    if complexity_age > 7:
        fallback.append({"question": "Why do you think the main character felt happy at the end?", "type": "inference",
                         "correct_answer": "Because they helped someone", "wrong_answers": ["Because they got a prize", "Because they went home"]})
        fallback.append({"question": "What do you think the story is trying to teach us?", "type": "inference",
                         "correct_answer": "Being kind to others is important", "wrong_answers": ["Being fast is the best", "You should always be alone"]})
    return fallback


def _generate_takeaway_questions(takeaways, story_text, child_age, child_name="the child"):
    """Build one multiple-choice question PER takeaway.

    Each question has:
      - A natural, varied stem (e.g. "What can you learn from their behavior?",
        "What is one lesson from this story?", "Why was it good that they helped?").
      - The takeaway text as the correct answer (verbatim).
      - 2 plausible-but-clearly-wrong distractor "lessons" generated by the LLM.

    A single batched LLM call produces all questions. Returns a list of dicts
    in the same shape as _generate_story_questions entries, or [] on failure.
    """
    if not takeaways:
        return []

    # Normalise takeaways (drop empties, cap to avoid bloating the quiz).
    takeaways = [str(t).strip() for t in takeaways if str(t).strip()]
    takeaways = takeaways[:4]  # safety cap — story_generator emits 2-3 typically
    if not takeaways:
        return []

    cleaned_story = clean_story_text(story_text)

    numbered = "\n".join(f"{i+1}. \"{t}\"" for i, t in enumerate(takeaways))
    prompt = (
        f"You are creating multiple-choice LESSON questions for a {child_age}-year-old "
        f"child named {child_name}. The story below has {len(takeaways)} takeaways. "
        f"Create exactly ONE question per takeaway.\n\n"
        f"Story:\n{cleaned_story}\n\n"
        f"Takeaways (each is the CORRECT answer for one question — use verbatim):\n"
        f"{numbered}\n\n"
        f"For EACH takeaway, produce one question with:\n"
        f"- A natural, kid-friendly QUESTION STEM. Vary the phrasing across questions. Pick whichever "
        f"  fits best:\n"
        f"    * \"What can you learn from their behavior?\"\n"
        f"    * \"What is one lesson from this story?\"\n"
        f"    * \"What did [character name] show us by what they did?\"\n"
        f"    * \"Why was it a good idea to ...?\"\n"
        f"    * \"What can we do like [character]?\"\n"
        f"- The CORRECT answer: the takeaway, exactly as given above (do NOT rephrase, paraphrase, "
        f"  or shorten it).\n"
        f"- 2 WRONG answers: plausible-but-clearly-wrong lessons. They must:\n"
        f"    - Sound like reasonable lessons in general but be wrong for THIS story.\n"
        f"    - Be similar length to the correct answer.\n"
        f"    - NOT be the OTHER takeaways from the list above (those are also correct).\n"
        f"    - NOT be opposites or trivially wrong (\"You should never be kind\").\n\n"
        f"Return ONLY a JSON array of {len(takeaways)} objects in the same order as the takeaways "
        f"above. Each object has keys: \"question\", \"correct_answer\", \"wrong_answers\" "
        f"(array of exactly 2 strings).\n"
        f"Example shape:\n"
        f"[{{\"question\": \"What can you learn from their behavior?\", "
        f"\"correct_answer\": \"<takeaway 1 verbatim>\", "
        f"\"wrong_answers\": [\"<distractor a>\", \"<distractor b>\"]}}, ...]"
    )

    print(f"[TakeawayQuestion] Generating {len(takeaways)} questions for takeaways: {takeaways}")
    try:
        raw = _gemini_generate(
            prompt,
            system="You write children's multiple-choice comprehension questions. Return JSON only.",
            max_tokens=1536,
        )
        if not raw:
            print("[TakeawayQuestion] LLM returned empty response")
            return []
        print(f"[TakeawayQuestion] Raw LLM response (first 500 chars): {raw[:500]!r}")

        arr = _parse_json_array(raw) if "_parse_json_array" in globals() else None
        if not isinstance(arr, list):
            # Minimal inline fallback
            start = raw.find("[")
            end = raw.rfind("]")
            if start != -1 and end != -1 and end > start:
                try:
                    arr = json.loads(raw[start:end + 1])
                except Exception as je:
                    print(f"[TakeawayQuestion] inline JSON parse failed: {je}")
                    arr = None
        if not isinstance(arr, list):
            print(f"[TakeawayQuestion] Could not parse LLM response as JSON array")
            return []
        print(f"[TakeawayQuestion] Parsed {len(arr)} items from LLM")

        results = []
        for i, item in enumerate(arr):
            if not isinstance(item, dict):
                print(f"[TakeawayQuestion] item {i} is not a dict: {item!r}")
                continue
            correct = takeaways[i] if i < len(takeaways) else None
            if not correct:
                print(f"[TakeawayQuestion] item {i}: no corresponding takeaway (index out of range)")
                continue
            qstem = str(item.get("question") or "").strip() \
                    or "What can you learn from this story?"
            wrongs_raw = item.get("wrong_answers") or item.get("distractors") or []
            if not isinstance(wrongs_raw, list):
                print(f"[TakeawayQuestion] item {i}: wrong_answers is not a list: {wrongs_raw!r}")
                continue
            wrongs = [str(w).strip() for w in wrongs_raw
                      if isinstance(w, (str, int, float)) and str(w).strip()]
            # Drop "wrong" answers that are actually other takeaways. Be lenient
            # — only filter EXACT matches; near-matches are still acceptable.
            wrongs_filtered = [w for w in wrongs if w not in takeaways]
            if len(wrongs_filtered) < 2:
                # Top up from un-filtered set if the LLM gave us mostly takeaways
                # as distractors. Better to have a question than to drop it.
                for w in wrongs:
                    if w not in wrongs_filtered:
                        wrongs_filtered.append(w)
                    if len(wrongs_filtered) >= 2:
                        break
            if len(wrongs_filtered) < 2:
                print(f"[TakeawayQuestion] item {i}: not enough distractors (got {wrongs}); skipping")
                continue
            results.append({
                "question": qstem,
                "type": "takeaway",
                "correct_answer": correct,
                "wrong_answers": wrongs_filtered[:2],
            })
        print(f"[TakeawayQuestion] Returning {len(results)} questions")
        return results
    except Exception as e:
        print(f"[TakeawayQuestion] generation failed: {e}")
        return []


_TAG_RE_VALIDATE = re.compile(r"\[(?:gesture|emotion):[^\]]+\]\s*")


def _validate_tag_positions(text):
    """Ensure every [gesture:...] / [emotion:...] tag sits at a valid spot.

    Valid positions:
      - Start of the text (beginning of the first sentence).
      - End of the text (end of the last sentence).
      - Right after a sentence-ending punctuation: '.', '!', '?'.
      - Right after a comma ','.

    Tags found at invalid positions (mid-word, mid-clause) are MOVED to the
    nearest valid position rather than dropped, so the emotional/gesture beat
    is preserved but synchronized with a natural pause. Tag order is preserved
    for tags that snap to the same position.
    """
    if not text or "[" not in text:
        return text

    # Phase 1 — extract tags with their position in the tag-stripped text.
    tags = []  # list of (clean_pos, tag_text_without_trailing_ws)
    pieces = []
    clean_len = 0
    last_end = 0
    for m in _TAG_RE_VALIDATE.finditer(text):
        between = text[last_end:m.start()]
        pieces.append(between)
        clean_len += len(between)
        tags.append((clean_len, m.group(0).rstrip()))
        last_end = m.end()
    pieces.append(text[last_end:])
    if not tags:
        return text
    clean_text = "".join(pieces)

    # Phase 2 — find all valid positions in clean_text.
    valid = {0, len(clean_text)}
    for m in re.finditer(r"[.!?,]", clean_text):
        i = m.end()
        # Walk forward past whitespace so the tag sits flush before the next word.
        while i < len(clean_text) and clean_text[i].isspace():
            i += 1
        valid.add(i)
    valid_sorted = sorted(valid)

    def is_valid(pos):
        if pos == 0 or pos >= len(clean_text):
            return True
        i = pos - 1
        while i >= 0 and clean_text[i].isspace():
            i -= 1
        return i >= 0 and clean_text[i] in ".!?,"

    def snap(pos):
        # Closest valid position by distance; on a tie prefer the EARLIER one
        # (tags describe what's about to happen, so firing before the next
        # sentence is more natural than after it).
        return min(valid_sorted, key=lambda p: (abs(p - pos), p))

    # Phase 3 — fix invalid positions, preserving tag order.
    by_pos = {}
    moved = 0
    for pos, tag in tags:
        new_pos = pos if is_valid(pos) else snap(pos)
        if new_pos != pos:
            moved += 1
        by_pos.setdefault(new_pos, []).append(tag)

    if moved:
        print(f"[TagValidate] moved {moved} mis-placed tag(s) to nearest valid position")

    # Phase 4 — reconstruct the text with tags inserted at the fixed positions.
    out = []
    for i in range(len(clean_text) + 1):
        if i in by_pos:
            for tag in by_pos[i]:
                out.append(tag)
                # Ensure a single space separates the tag from the following
                # non-whitespace character.
                if i < len(clean_text) and not clean_text[i].isspace():
                    out.append(" ")
        if i < len(clean_text):
            out.append(clean_text[i])
    return "".join(out)


def _apply_emotion_tags_with_gemini(story_text):
    """Run a Gemini-Flash pass over a generated story to insert correct
    [gesture:...] and [emotion:...] tags inline.

    Story generation may use a model that under-tags or invents emotion
    names. This pass uses Gemini specifically
    for the tagging step: it preserves the story word-for-word and adds tags
    immediately before the sentence that depicts each emotional beat or
    physical action. Returns the tagged story, or the original on failure.
    """
    if not story_text or not story_text.strip():
        return story_text

    prompt = (
        "You are tagging a children's story for a robot that will read it aloud "
        "while showing matching facial expressions and gestures.\n\n"
        "TASK: Return the SAME story word-for-word, but with [gesture:NAME] and "
        "[emotion:NAME] tags inserted immediately before the sentence that "
        "depicts the emotional beat or physical action.\n\n"
        "ALLOWED EMOTION NAMES (use ONLY these — exact match):\n"
        "  QT/happy, QT/sad, QT/surprised, QT/afraid, QT/angry, QT/calm, QT/shy\n\n"
        "ALLOWED GESTURE NAMES:\n"
        "  hi, bye, nodding-yes, clapping, emotions/hoora, emotions/happy, emotions/calm, emotions/shy,\n"
        "  slight_no, think, sneezing, yawn, breathing_exercise,\n"
        "   kiss, stretching\n\n"
        "RULES:\n"
        "- Tag EVERY clear emotional beat. Whenever a character smiles, laughs,\n"
        "  giggles, or feels happy/proud/excited/relieved/grateful, insert\n"
        "  [emotion:QT/happy]. Whenever they cry, frown, or feel\n"
        "  sad/disappointed/lonely, insert [emotion:QT/sad]. Same rule for\n"
        "  surprised, afraid (scared/nervous/worried), angry (frustrated/mad),\n"
        "  calm (peaceful/content), shy (embarrassed/bashful).\n"
        "- Never invent emotion names. If a feeling is not in the allowlist,\n"
        "  pick the closest one.\n"
        "- PLACEMENT — put each tag at a SENTENCE BOUNDARY (never mid-word),\n"
        "  chosen by WHERE the emotion/action word sits in its sentence:\n"
        "    * EARLY in the sentence (first half) -> put the tag IMMEDIATELY\n"
        "      BEFORE that sentence.\n"
        "    * LATER in the sentence (second half) -> put the tag at the END\n"
        "      of that sentence, immediately AFTER its closing . ! or ?\n"
        "  Anchor tags to the specific sentence (and the half of it) where the\n"
        "  feeling or action occurs, not to the start of the paragraph. The same\n"
        "  emotion may appear multiple times in one paragraph if the character\n"
        "  feels it more than once.\n"
        "- Use gesture tags for physical actions where they fit (waving,\n"
        "  clapping, nodding, hugging, stretching, etc.).\n"
        "- DO NOT change, add, remove, rephrase, or reorder ANY of the\n"
        "  original words. Only insert tags.\n"
        "- If the input already contains tags, KEEP correct ones, FIX invalid\n"
        "  emotion names by remapping to the allowlist, and ADD missing tags\n"
        "  for emotional beats that are currently untagged.\n"
        "- Return ONLY the tagged story text. No JSON, no explanation, no\n"
        "  preamble, no code fences.\n\n"
        "EXAMPLES (tag position follows the emotion word's position):\n"
        "  Early in sentence -> tag BEFORE the sentence:\n"
        "    'Mia felt happy when she saw the puppy.'\n"
        "    => '[emotion:QT/happy] Mia felt happy when she saw the puppy.'\n"
        "  Late in sentence -> tag at the END of the sentence:\n"
        "    'The puppy ran to Mia and she smiled.'\n"
        "    => 'The puppy ran to Mia and she smiled. [emotion:QT/happy]'\n\n"
        f"STORY:\n{story_text}"
    )

    tagged = _gemini_generate(
        prompt,
        system="You add inline gesture/emotion tags to children's stories. Return only the tagged story.",
        temperature=0.2,
        max_tokens=4096,
        label="emotion-tagger",
    )
    if not tagged:
        print("[StoryTagger] Gemini returned nothing — keeping original story untagged")
        return story_text

    # Strip code fences if Gemini wrapped the output despite instructions
    tagged = tagged.strip()
    if tagged.startswith('```'):
        tagged = tagged.strip('`').strip()
        for prefix in ('text\n', 'markdown\n', 'plain\n'):
            if tagged.lower().startswith(prefix):
                tagged = tagged[len(prefix):]
                break

    # Sanity check: if Gemini drastically changed the length, fall back.
    # Tags add length, so the tagged version should be >= original; allow 5%
    # shrinkage as slack for whitespace normalization, but reject anything
    # that lost real text.
    if len(tagged) < int(len(story_text) * 0.95):
        print(f"[StoryTagger] tagged output too short ({len(tagged)} vs {len(story_text)}) — keeping original")
        return story_text

    print(f"[StoryTagger] Gemini tagging pass succeeded ({len(tagged)} chars)")
    return tagged


def _reinject_tags_into_pages(original_story, pages):
    """Re-inject [gesture:...] and [emotion:...] tags from the original story into split pages.

    The page splitter (LLM) may strip proper tags or leave malformed ones.
    Tags are inserted INLINE right before the sentence they originally preceded,
    so the robot's gesture/emotion fires at the right narrative moment — not
    at the start of the whole page.
    """
    import re

    # First, strip ALL tag variants from pages (clean slate)
    bare_tag_re = re.compile(r'\[(?:gesture|emotion):[^\]]+\]|\b(?:gesture|emotion):\S+', re.IGNORECASE)
    result = [re.sub(r'\s{2,}', ' ', bare_tag_re.sub('', p)).strip() for p in pages]

    # Find proper tags in original story and the words that follow each one.
    # Track the ORDER tags appear so identical tags get matched to distinct contexts.
    tag_pattern = re.compile(r'(\[(?:gesture|emotion):[^\]]+\])')
    tags_with_context = []
    for m in tag_pattern.finditer(original_story):
        tag = m.group(1)
        rest = original_story[m.end():].lstrip()
        # Skip any immediately following tags (e.g. [gesture:hi][emotion:QT/happy])
        while rest and rest.startswith('['):
            close = rest.find(']')
            if close != -1:
                rest = rest[close+1:].lstrip()
            else:
                break
        # Grab a longer context window for more reliable matching
        context = rest[:120].strip()
        if context:
            tags_with_context.append((tag, context))

    if not tags_with_context:
        return result

    def _normalize_for_match(s):
        # Loose match: collapse whitespace, drop straight/curly quote variants
        s = re.sub(r'\s+', ' ', s)
        s = s.replace('“', '"').replace('”', '"').replace('‘', "'").replace('’', "'")
        return s

    # Insert each tag inline at its matching sentence within the matching page.
    # Use progressively shorter prefixes so spacing/punctuation tweaks by the
    # page-splitter LLM don't kill the match.
    used_positions = {i: [] for i in range(len(result))}  # page_idx -> [insert_positions]
    for tag, context in tags_with_context:
        norm_context = _normalize_for_match(context)
        inserted = False
        for prefix_len in (60, 40, 25, 15):
            if inserted:
                break
            needle = norm_context[:prefix_len].strip()
            if not needle:
                continue
            for i, page in enumerate(result):
                norm_page = _normalize_for_match(page)
                pos = norm_page.find(needle)
                if pos == -1:
                    continue
                # Map normalized position back to raw page (close enough — both
                # normalizations are length-preserving aside from whitespace
                # collapse, which is rare inside a single sentence).
                raw_pos = page.find(needle) if needle in page else pos
                if raw_pos < 0:
                    raw_pos = pos
                # Skip if we already inserted at (or very near) this spot
                if any(abs(raw_pos - p) < 3 for p in used_positions[i]):
                    continue
                # Insert tag + space at raw_pos, shifting any later positions
                result[i] = page[:raw_pos] + tag + ' ' + page[raw_pos:]
                shift = len(tag) + 1
                used_positions[i] = [p + shift if p >= raw_pos else p for p in used_positions[i]]
                used_positions[i].append(raw_pos)
                inserted = True
                break

    return result


def _split_story_into_pages(story_text, child_age):
    """Split a story into age-appropriate pages using Gemini.

    Sentence count is a SOFT target tied to age:
        Ages 3-4:  ~1-2 sentences per page
        Ages 5-6:  ~2-3 sentences per page
        Ages 7+:   ~3-5 sentences per page

    Narrative flow and scene context come FIRST. Sentences belonging to the
    same scene or context are kept together on a single page even if that
    pushes the count slightly over the target. A page break should land at a
    natural scene/context shift, never in the middle of a continuous moment.

    Returns a list of page strings. Falls back to paragraph-aware splitting
    if the LLM is unavailable.
    """
    # Preserve [gesture:...] / [emotion:...] tags through clean_story_text by
    # swapping them for placeholders, cleaning, then restoring. Same trick for
    # paragraph breaks so the fallback splitter can respect them as scene hints.
    tag_re_preserve = re.compile(r'\[(?:gesture|emotion):[^\]]+\]', re.IGNORECASE)
    saved_tags = tag_re_preserve.findall(story_text)
    placeholder_text = story_text
    for idx, _t in enumerate(saved_tags):
        placeholder_text = placeholder_text.replace(_t, f"TAGPLACEHOLDER{idx}NUM", 1)
    # Mark paragraph breaks before clean_story_text collapses them
    placeholder_text = re.sub(r'\n\s*\n+', ' PARABREAKMARKER ', placeholder_text)
    cleaned = clean_story_text(placeholder_text)
    for idx, t in enumerate(saved_tags):
        cleaned = cleaned.replace(f"TAGPLACEHOLDER{idx}NUM", t, 1)
    # Restore paragraph break marker for the fallback path; for the LLM input
    # we feed plain text with newline-separated paragraphs.
    cleaned_for_llm = cleaned.replace('PARABREAKMARKER', '\n\n')
    cleaned_for_llm = re.sub(r'\n\n\s+', '\n\n', cleaned_for_llm)
    if not cleaned_for_llm.strip():
        return [cleaned_for_llm]

    if child_age <= 4:
        sents_per_page = "about 1 to 2"
    elif child_age <= 6:
        sents_per_page = "about 2 to 3"
    else:
        sents_per_page = "about 3 to 5"

    prompt = (
        f"Split the following story into pages for a {child_age}-year-old child.\n"
        f"\n"
        f"PRIORITIES (in order):\n"
        f"1. Narrative flow and context come FIRST. Keep sentences that belong to\n"
        f"   the same scene, moment, or train of thought together on the SAME page.\n"
        f"   Never split a continuous scene across pages just to hit a sentence count.\n"
        f"2. A page break must fall at a natural scene/context shift — a change of\n"
        f"   setting, time, character focus, or action.\n"
        f"3. As a SOFT target, aim for {sents_per_page} sentences per page. It is\n"
        f"   acceptable to go slightly over (or under) this target when the scene\n"
        f"   demands it. Do NOT force a split mid-scene to satisfy the count.\n"
        f"\n"
        f"HARD RULES:\n"
        f"- Keep every sentence intact and do NOT rephrase or change any words.\n"
        f"- Do not split a single sentence across two pages.\n"
        f"- PRESERVE all [gesture:...] and [emotion:...] tags VERBATIM in their\n"
        f"  exact original positions. Do NOT remove, move, rewrite, or reformat\n"
        f"  any tag. A tag stays attached to the sentence that follows it; if\n"
        f"  that sentence moves to a new page, the tag moves with it.\n"
        f"- Paragraph breaks in the input are strong scene hints — prefer to\n"
        f"  split at paragraph boundaries when possible.\n"
        f"- Return ONLY a JSON array of strings, where each string is one page.\n"
        f"- Example: [\"Page 1 text here.\", \"[emotion:QT/happy] Page 2 text.\"]\n"
        f"\n"
        f"Story:\n{cleaned_for_llm}"
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

    # Fallback: paragraph-aware splitting. A paragraph is treated as a single
    # scene/context block — never split across pages unless its sentence count
    # exceeds the target by more than half (in which case break at the midpoint).
    if child_age <= 4:
        target = 2
    elif child_age <= 6:
        target = 3
    else:
        target = 4

    paragraphs = [p.strip() for p in cleaned_for_llm.split('\n\n') if p.strip()]
    if not paragraphs:
        paragraphs = [cleaned_for_llm.strip()]

    pages = []
    for para in paragraphs:
        sentences = re.split(r'(?<=[.!?])\s+', para.strip())
        sentences = [s.strip() for s in sentences if s.strip()]
        if not sentences:
            continue
        # Keep the whole paragraph as one page when it's at or near target,
        # or only modestly over it (within 1.5x). Only break very long
        # paragraphs, and break them at the most balanced midpoint.
        if len(sentences) <= int(target * 1.5):
            pages.append(' '.join(sentences))
        else:
            # Split into roughly target-sized chunks but never below target/2
            min_chunk = max(1, target // 2)
            i = 0
            while i < len(sentences):
                chunk = sentences[i:i + target]
                # If the leftover after this chunk would be below min_chunk,
                # absorb it into this page instead of leaving an orphan.
                remaining = len(sentences) - (i + len(chunk))
                if 0 < remaining < min_chunk:
                    chunk = sentences[i:]
                    i = len(sentences)
                else:
                    i += len(chunk)
                pages.append(' '.join(chunk))
    print(f"[StoryPages] Fallback split into {len(pages)} pages (age {child_age}, target {target} sents, {len(paragraphs)} paragraphs)")
    return pages


def _identify_story_scenes(chunks, unit_label="paragraph"):
    """Analyze story chunks (paragraphs by default) and identify scenes.

    For each chunk, the LLM picks a scene index. Two chunks share a scene
    ONLY when they depict essentially the same visual moment (same setting,
    same characters, similar action). Default bias: 1 chunk = 1 scene, so the
    image count stays close to the chunk count while still allowing reuse for
    truly identical visuals.

    Returns:
        scenes: list of scene description strings (one per unique scene)
        chunk_to_scene: list of ints mapping each chunk index to a scene index
    """
    if not chunks:
        return [""], [0]

    Unit = unit_label.capitalize()
    full_text = "\n\n".join(f"{Unit} {i+1}: {p}" for i, p in enumerate(chunks))

    prompt = (
        f"You are choosing illustrations for a children's story that has been split "
        f"into {len(chunks)} {unit_label}s.\n\n"
        f"For each {unit_label}, decide which SCENE it depicts. Two {unit_label}s should "
        f"share a scene ONLY IF they show essentially the same visual moment — same setting, "
        f"same characters present, and similar action. If anything important changes (location, "
        f"who is on screen, what they are doing), it is a NEW scene.\n\n"
        f"DEFAULT BIAS: assume each {unit_label} is its own scene. Only merge {unit_label}s when "
        f"the same illustration would clearly work for both. Do not over-merge — we want roughly "
        f"one image per {unit_label} unless they are truly the same visual.\n\n"
        f"{full_text}\n\n"
        f"Return ONLY a JSON object with:\n"
        f"- \"scenes\": an array of short visual descriptions (1-2 sentences each) "
        f"describing what should be illustrated for each scene. Focus on setting, "
        f"characters, and key action. There must be AT MOST {len(chunks)} scenes.\n"
        f"- \"chunk_to_scene\": an array of {len(chunks)} integers, where each integer "
        f"is the 0-based scene index for that {unit_label}.\n\n"
        f"Example for 4 {unit_label}s with 3 scenes ({unit_label}s 0 and 1 share scene 0 "
        f"because they're a single conversation in the same kitchen):\n"
        f"{{\"scenes\": [\"Mom and Lily sitting at a sunny kitchen table eating breakfast\", "
        f"\"Lily walking to school carrying her red backpack along a tree-lined sidewalk\", "
        f"\"Lily showing her drawing to the class at the front of the classroom\"], "
        f"\"chunk_to_scene\": [0, 0, 1, 2]}}"
    )
    raw = _gemini_generate(prompt, system="You analyze story structure. Return JSON only.", max_tokens=2048)
    if raw:
        try:
            print(f"[StoryScenes] Gemini raw response: {raw[:500]}")
            obj = _extract_json(raw)
            print(f"[StoryScenes] Parsed JSON: {json.dumps(obj, indent=2) if obj else None}")
            if obj:
                scenes = obj.get('scenes', [])
                # Accept either the new key or the legacy "page_to_scene" key.
                mapping = obj.get('chunk_to_scene') or obj.get('page_to_scene') or []
                if (isinstance(scenes, list) and len(scenes) > 0
                        and isinstance(mapping, list) and len(mapping) == len(chunks)):
                    if all(isinstance(m, int) and 0 <= m < len(scenes) for m in mapping):
                        return scenes, mapping
        except Exception as e:
            print(f"[StoryScenes] Scene identification failed: {e}")

    # Fallback: treat each chunk as its own scene
    print(f"[StoryScenes] Using fallback: one scene per {unit_label}")
    scenes = [p[:200] for p in chunks]  # Use first 200 chars as scene description
    chunk_to_scene = list(range(len(chunks)))
    return scenes, chunk_to_scene


def _split_into_paragraphs(story_text):
    """Split a story body into paragraphs separated by blank lines."""
    if not story_text:
        return []
    parts = re.split(r"\n\s*\n+", story_text.strip())
    return [p.strip() for p in parts if p.strip()]


def _map_pages_to_paragraphs(pages, paragraphs):
    """For each page, find which paragraph it belongs to.

    Pages are typically a sub-sequence of a paragraph (page splitting may
    break a long paragraph into several pages, but it never splits a sentence
    across paragraphs). We match by substring with a sequential constraint
    (pages can only advance, never go back) so a page from the third
    paragraph never gets matched to the first.
    """
    if not pages:
        return []
    if not paragraphs:
        return [0] * len(pages)

    _tag_re = re.compile(r"\[(gesture|emotion):[^\]]+\]\s*")
    clean_paras = [_tag_re.sub("", p).strip() for p in paragraphs]
    clean_pages = [_tag_re.sub("", p).strip() for p in pages]

    page_to_para = []
    cursor = 0  # Earliest paragraph this page can match into
    for page in clean_pages:
        found = None
        # Try progressively shorter prefixes for robustness against minor diffs.
        for snip_len in (80, 50, 30, 18):
            snippet = page[:snip_len].strip()
            if not snippet:
                continue
            for i in range(cursor, len(clean_paras)):
                if snippet in clean_paras[i]:
                    found = i
                    break
            if found is not None:
                break
        if found is None:
            found = cursor  # Default to current cursor on no match
        page_to_para.append(found)
        cursor = found
    return page_to_para


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

    # Takeaways: the client (templates/index.html) extracts them from the raw
    # LLM output and sends them. Server-side fallback parses them out of the
    # story body in case a non-web caller forwards the full raw response.
    takeaways = data.get("takeaways") or []
    if isinstance(takeaways, str):
        takeaways = [takeaways]
    takeaways = [str(t).strip() for t in takeaways if str(t).strip()]
    if not takeaways:
        tk_match = re.search(
            r"\*\*\s*Takeaways\s*\*\*\s*\n(.+?)(?=\n\s*\*\*\s*(?:Explanation|End)\b|\Z)",
            story, flags=re.IGNORECASE | re.DOTALL,
        )
        if tk_match:
            for line in tk_match.group(1).splitlines():
                line = line.strip()
                line = re.sub(r"^[-*•]\s*", "", line)
                line = re.sub(r"^\d+[.)]\s*", "", line)
                line = line.strip()
                if line:
                    takeaways.append(line)
    if takeaways:
        print(f"[StorySave] Extracted {len(takeaways)} takeaways")

    # Extract title from story text and store in metadata
    title, _body = _extract_story_title(story)
    if title:
        metadata['title'] = title
        print(f"[StorySave] Extracted title: {title}")

    # Defensive: remove a stray title-like prefix that `_extract_story_title`
    # didn't strip out. The prompt forbids a title in the body, but the LLM
    # still emits one occasionally in non-canonical forms ("Title:", "#",
    # bolded line, or no blank-line separator). We do this BEFORE the Gemini
    # tagging pass so the tagger sees the clean body, not the title.
    stripped_story = _strip_leading_title(story)
    if stripped_story != story:
        print(f"[StorySave] Stripped stray title from saved story body")
        story = stripped_story

    # Get child age for page splitting
    child_age = 5
    try:
        child_age = int(metadata.get('age', user.get('age', 5)))
    except (ValueError, TypeError):
        child_age = 5

    # Word-count enforcement: the story_generator prompt caps body length by
    # age tier, but the LLM still sometimes overshoots. If the saved body
    # exceeds the tier's max_words, ask the LLM to rewrite it shorter.
    try:
        min_words, max_words = story_generator.get_word_range_for_age(child_age)
        body_words = len(story.split())
        if body_words > max_words:
            child_name_hint = metadata.get('child_name', '') or user.get('name', '')
            print(f"[StorySave] Body is {body_words} words, cap is {max_words}. Shortening...")
            shortened = story_generator.shorten_story(story, child_age, child_name_hint)
            if shortened and shortened != story:
                new_words = len(shortened.split())
                print(f"[StorySave] Shortened to {new_words} words")
                story = shortened
                metadata['word_count'] = new_words
                metadata['shortened'] = True
    except Exception as e:
        print(f"[StorySave] Word-count enforcement skipped: {e}")

    # Run a Gemini-Flash tagging pass on the story before splitting.
    # Gemini Flash is used here specifically to insert/correct
    # [gesture:...] / [emotion:...] tags inline.
    story = _apply_emotion_tags_with_gemini(story)
    # Validate tag positions: snap any tag that landed mid-word or mid-clause
    # to the nearest valid position (start/end of sentence, after ,/!/?).
    story = _validate_tag_positions(story)

    # Split story into age-appropriate pages (clean_story_text strips the title)
    pages = _split_story_into_pages(story, child_age)

    # Re-inject [gesture:...] and [emotion:...] tags into pages.
    # The page splitter may drop tags, so we match each tag's surrounding text
    # back into the correct sentence.
    pages = _reinject_tags_into_pages(story, pages)

    # Belt-and-suspenders: if a title still leads pages[0] (e.g. the page
    # splitter copied it verbatim), strip it here so the first read page does
    # not start by speaking the title aloud.
    if pages:
        cleaned_first = _strip_leading_title(pages[0])
        if cleaned_first != pages[0]:
            print("[StorySave] Stripped stray title from pages[0]")
            pages[0] = cleaned_first

    # Prepare user stories directory
    user_dir = os.path.join(USER_DATA_DIR, username, "stories")
    os.makedirs(user_dir, exist_ok=True)

    # Use timestamp for unique filename
    import datetime
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"story_{ts}.json"
    fpath = os.path.join(user_dir, fname)

    # Strip tags from pages before scene identification (images don't need gesture/emotion tags)
    _tag_re = re.compile(r'\[(gesture|emotion):[^\]]+\]\s*')
    clean_pages = [_tag_re.sub('', p).strip() for p in pages]

    # Scene identification at PARAGRAPH granularity. Pages are sub-chunks of
    # paragraphs (one paragraph can span several pages), so identifying scenes
    # at the page level caused over-merging and too few images. At paragraph
    # granularity, images ≤ paragraph_count, with reuse only when paragraphs
    # depict the same visual moment.
    paragraphs = _split_into_paragraphs(_tag_re.sub('', story))
    if not paragraphs:
        paragraphs = clean_pages  # Defensive fallback for stories without blank lines
    print(f"[StorySave] {len(paragraphs)} paragraphs detected")

    scenes, paragraph_to_scene = _identify_story_scenes(paragraphs, unit_label="paragraph")
    page_to_paragraph = _map_pages_to_paragraphs(clean_pages, paragraphs)
    # Derive page_to_scene by composing the two mappings.
    page_to_scene = [paragraph_to_scene[p] if 0 <= p < len(paragraph_to_scene) else 0
                     for p in page_to_paragraph]
    print(f"[StorySave] {len(scenes)} scenes identified across {len(paragraphs)} paragraphs")
    print(f"[StorySave] paragraph_to_scene: {paragraph_to_scene}")
    print(f"[StorySave] page_to_paragraph:  {page_to_paragraph}")
    print(f"[StorySave] page_to_scene:      {page_to_scene}")

    # Generate comprehension questions for the story
    q_persona_ctx = _persona_context_for(username, child_age, kind="question")
    questions = _generate_story_questions(
        story, child_age, metadata.get('child_name', 'the child'),
        persona_context=q_persona_ctx,
        language_age=_language_age_for(username, child_age),
    )
    print(f"[StorySave] Generated {len(questions)} comprehension questions")

    # If the story has takeaways (age 7+), append one multiple-choice question
    # PER takeaway at the END, asking what the child can learn. Each takeaway
    # becomes the correct answer; distractors are LLM-generated.
    if takeaways:
        takeaway_qs = _generate_takeaway_questions(
            takeaways, story, child_age, metadata.get('child_name', 'the child'),
        )
        if takeaway_qs:
            questions.extend(takeaway_qs)
            print(f"[StorySave] Appended {len(takeaway_qs)} takeaway questions "
                  f"(now {len(questions)} total)")

    # Save story, metadata, pages, paragraphs, scenes, mappings, questions,
    # and takeaways. paragraph_to_scene + page_to_paragraph are persisted so
    # downstream code (image lookup, debugging) can navigate either way.
    with open(fpath, "w") as f:
        json.dump({
            "story": story,
            "metadata": metadata,
            "pages": pages,
            "paragraphs": paragraphs,
            "scenes": scenes,
            "page_to_scene": page_to_scene,
            "page_to_paragraph": page_to_paragraph,
            "paragraph_to_scene": paragraph_to_scene,
            "questions": questions,
            "takeaways": takeaways,
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

def _log_gemini_io(system, prompt, response=None, label=None):
    """Trace an in-process Gemini call to stdout so its prompt/response land in
    the daily trace log — the same way story generation logs its prompt.
    Covers the emotion tagger and every other _gemini_generate() pass.
    Gated by LOG_LLM_PROMPTS (default on); set LOG_LLM_PROMPTS=0 to silence.
    """
    if os.getenv("LOG_LLM_PROMPTS", "1") == "0":
        return
    tag = f"Gemini:{label}" if label else "Gemini"
    if response is None:
        print(f"[{tag}] >>> PROMPT ({len(prompt)} chars) | system: {system}")
        print(prompt)
    else:
        preview = response if len(response) <= 1500 else (
            response[:1500] + f"\n...(+{len(response) - 1500} more chars)")
        print(f"[{tag}] <<< RESPONSE ({len(response)} chars):")
        print(preview)


def _gemini_generate(prompt, system="You are a helpful assistant. Return JSON only when asked.",
                     temperature=0.3, max_tokens=2048, label=None):
    """Call Gemini via subprocess for general-purpose text generation.

    Returns the raw response text, or None on failure.

    Set the optional ``label`` to name the call site in the trace log
    (e.g. label="emotion-tagger"); otherwise the system prompt identifies it.
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
        _log_gemini_io(system, prompt, label=label)
        cmd = [WORKER_PYTHON, script_path,
               '--prompt-file', tmp.name,
               '--system', system,
               '--temperature', str(temperature),
               '--max-tokens', str(max_tokens)]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
        if proc.returncode != 0:
            print(f"[Gemini] Script error: {proc.stderr[:300]}")
            return None
        out = (proc.stdout or '').strip()
        _log_gemini_io(system, prompt, response=out, label=label)
        return out
    except Exception as e:
        print(f"[Gemini] Error: {e}")
        return None
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


# ── Claude one-shot generation ───────────────────────────────────────────────
# The object-request (scene) game generates its questions with Claude rather
# than Gemini. The Anthropic SDK is called in-process (same venv, same
# ANTHROPIC_API_KEY loaded via load_dotenv) — the pattern the intent/quiz/story
# Claude paths already use. Override the model via the SCENE_GAME_LLM_MODEL env.
SCENE_GAME_LLM_MODEL = os.getenv("SCENE_GAME_LLM_MODEL", "claude-sonnet-4-6")

_anthropic_client = None
_anthropic_client_lock = Lock()


def _get_anthropic_client():
    """Lazily build a shared Anthropic client (reads ANTHROPIC_API_KEY)."""
    global _anthropic_client
    with _anthropic_client_lock:
        if _anthropic_client is None:
            import anthropic
            _anthropic_client = anthropic.Anthropic()
    return _anthropic_client


def _log_claude_io(system, prompt, response=None, label=None):
    """Trace a Claude call to the daily trace log, mirroring _log_gemini_io."""
    if os.getenv("LOG_LLM_PROMPTS", "1") == "0":
        return
    tag = f"Claude:{label}" if label else "Claude"
    if response is None:
        print(f"[{tag}] >>> PROMPT ({len(prompt)} chars) | system: {system}")
        print(prompt)
    else:
        preview = response if len(response) <= 1500 else (
            response[:1500] + f"\n...(+{len(response) - 1500} more chars)")
        print(f"[{tag}] <<< RESPONSE ({len(response)} chars):")
        print(preview)


def _claude_generate(prompt, system="You are a helpful assistant. Return JSON only when asked.",
                     temperature=0.3, max_tokens=2048, label=None, model=None):
    """Call Claude for general-purpose text generation.

    Drop-in replacement for _gemini_generate(): same signature, returns the raw
    response text or None on failure. Runs in-process via the Anthropic SDK.
    """
    model = model or SCENE_GAME_LLM_MODEL
    try:
        _log_claude_io(system, prompt, label=label)
        kwargs = dict(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        # Sonnet/Haiku accept temperature; Opus 4.7+/Fable reject sampling params.
        if temperature is not None and not (
                model.startswith("claude-opus-4-7")
                or model.startswith("claude-opus-4-8")
                or model.startswith("claude-fable")):
            kwargs["temperature"] = temperature
        resp = _get_anthropic_client().messages.create(**kwargs)
        out = "".join(
            getattr(b, "text", "") for b in resp.content
            if getattr(b, "type", None) == "text"
        ).strip()
        _log_claude_io(system, prompt, response=out, label=label)
        return out
    except Exception as e:
        print(f"[Claude] Error: {e}")
        return None


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


def _scene_game_generate_question(toy_list, child_age, learning_goals, persona_context="",
                                  language_age=None):
    """Use Claude to generate a scene-game question.

    The question wording is pitched at the child's target MLU (mean length of
    utterance), which is read directly from the knowledge base.

    For ages 2-3: direct request naming one specific object.
    For ages 4-6: criteria-based (e.g. "a red fruit") — multiple toys may match.
    For ages 7+:  complex inference riddle — child must reason about properties.

    Complexity (the exact/criteria/riddle tier and the level cue given to the
    LLM) follows ``language_age`` when provided — the child's developmental
    language age — so an older child with a language delay is not handed a 7+
    riddle. Falls back to ``child_age`` when None.

    Returns dict with keys:
        question  – the text to speak
        target    – exact toy name (ages 2-3) or None (ages 4+)
        criteria  – descriptive criteria string (ages 4+) or None (ages 2-3)
        mode      – "exact" | "criteria"
    """
    # Pick one toy as the primary target (always used as fallback)
    target = random.choice(toy_list)

    # Complexity tier follows the developmental/language age, not chronological.
    complexity_age = language_age if language_age is not None else child_age

    # Pull the target MLU (mean length of utterance) directly from the knowledge
    # base so the question is pitched at the child's language level — referenced
    # explicitly in the prompt rather than relying only on the persona blob.
    mlu_clause = ""
    try:
        kb_info = knowledge_base.describe(complexity_age)
        mlu_range = kb_info.get('mlu_range') if kb_info else None
        if mlu_range:
            mlu_clause = (
                f"TARGET MLU (knowledge base): the child's mean length of utterance "
                f"is about {mlu_range} words. Keep the question within that length and "
                f"use sentence structures a child at this language level produces.\n"
            )
    except Exception as e:
        print(f"[SceneGame] MLU lookup failed: {e}")

    # Determine mode based on age
    if complexity_age <= 3:
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
    # Knowledge-base guidance is intentionally limited to the MLU only (the
    # mlu_clause built above). The full persona context — language targets
    # (pronouns, -ing forms), speech sounds, and interests — is deliberately
    # NOT woven into the question, so target-driven phrasings like
    # "She is carrying ..." no longer appear. `persona_context` is still
    # accepted for call-site compatibility but is not used here.

    if mode == "exact":
        # For age <=3 we DO NOT call the LLM. Skip it and pick one of the four
        # fixed templates locally — this guarantees a strictly-direct request
        # of the form "<opener> the <target>!" with no extra clauses.
        article = "an" if target[:1].lower() in "aeiou" else "a"
        templates = [
            f"Show me the {target}!",
            f"Where is the {target}?",
            f"Can you find the {target}?",
            f"Let's find the {target}!",
            # Variants with article when "the" feels off (kept rare):
            f"Show me {article} {target}!",
        ]
        question = random.choice(templates)
        return {
            'question': question,
            'target': target,
            'criteria': None,
            'mode': 'exact',
        }
    elif complexity_age <= 6:
        prompt = (
            f"You are generating a question for an object detection game for a "
            f"{complexity_age}-year-old child.\n"
            f"Available physical toys: {', '.join(toy_list)}.\n"
            f"{mlu_clause}"
            f"{goals_clause}"
            f"Generate ONE inference-style request that lets the child figure\n"
            f"out the target object from its observable properties.\n"
            f"\n"
            f"HARD RULE — the QUESTION text must NOT name the target object.\n"
            f"It must NEVER contain any of these noun names from the toy list:\n"
            f"  {', '.join(toy_list)}\n"
            f"Refer to the target only as \"it\", \"something\", \"one\", or by\n"
            f"a generic placeholder like \"a fruit\" or \"a vehicle\". The child\n"
            f"must INFER which object you mean from the description.\n"
            f"\n"
            f"CRITICAL — the criteria must describe ONE simple, concrete object:\n"
            f"- A single noun (the object type) with at most ONE adjective\n"
            f"  describing color, size, or category.\n"
            f"- Good criteria: \"banana\", \"red car\", \"green dinosaur\",\n"
            f"  \"tomato\", \"yellow fruit\", \"round ball\".\n"
            f"- BAD criteria (do NOT produce these):\n"
            f"    * \"red car moving block\" (compound / multi-object)\n"
            f"    * \"big round shiny red fruit on a tree\" (too many properties)\n"
            f"    * \"toy that you can stack\" (function-based, vague)\n"
            f"- Do not chain multiple objects or stack three+ adjectives.\n"
            f"\n"
            f"The criteria MUST match at least one toy from the list above.\n"
            f"Use simple, clear language appropriate for ages 4-6.\n"
            f"Good examples (target NOT named in the question):\n"
            f"- Question: \"I want a red fruit!\" (criteria: red fruit)\n"
            f"- Question: \"Can you find something yellow?\" (criteria: yellow)\n"
            f"- Question: \"Show me something green that goes ROAR!\"\n"
            f"  (criteria: green dinosaur)\n"
            f"BAD example (do NOT do this — names the target):\n"
            f"- Question: \"Show me the red apple!\" — \"apple\" is the target name.\n"
            f"\n"
            f"Return ONLY a JSON object:\n"
            f"{{\"question\": \"<the sentence — must NOT contain any toy name>\", "
            f"\"criteria\": \"<short criteria phrase: one noun + at most one adjective>\"}}"
        )
    else:
        prompt = (
            f"You are generating a question for an object detection game for a "
            f"{complexity_age}-year-old child.\n"
            f"Available physical toys: {', '.join(toy_list)}.\n"
            f"{mlu_clause}"
            f"{goals_clause}"
            f"Generate ONE riddle that requires the child to reason about\n"
            f"properties (color, shape, size, function, where it is found) to\n"
            f"figure out the answer. Do NOT use a conversational tone.\n"
            f"\n"
            f"HARD RULE — the QUESTION (riddle) text must NEVER name the target\n"
            f"object. It must NOT contain any of these noun names from the toy list:\n"
            f"  {', '.join(toy_list)}\n"
            f"Use only pronouns (\"it\", \"I\") and property descriptions. The\n"
            f"child must INFER the target from the clues.\n"
            f"\n"
            f"CRITICAL — the underlying TARGET must be ONE simple, concrete object:\n"
            f"- A single noun (the object type), optionally with ONE color or\n"
            f"  size adjective. Examples of acceptable targets: \"banana\",\n"
            f"  \"red car\", \"green dinosaur\", \"tomato\".\n"
            f"- The riddle text may use 2-3 properties as clues, but the\n"
            f"  \"criteria\" field MUST be the simple target description (one\n"
            f"  noun + at most one adjective).\n"
            f"- Do NOT chain multiple objects or invent compound targets like\n"
            f"  \"red car moving block\" or \"shiny round tree fruit\".\n"
            f"\n"
            f"The target MUST match at least one toy from the list.\n"
            f"Good example: \"I am round and red, and I grow on a tree. What am I?\"\n"
            f"  (criteria: \"red apple\") — note the riddle does NOT say \"apple\".\n"
            f"BAD example (do NOT do this — names the target):\n"
            f"- \"Find the red apple that grows on a tree.\"\n"
            f"\n"
            f"Return ONLY a JSON object:\n"
            f"{{\"question\": \"<the riddle — must NOT contain any toy name>\", "
            f"\"criteria\": \"<simple target: one noun + at most one adjective>\"}}"
        )

    def _question_leaks_target(q_text):
        """Return True if the question text contains any toy name from the list."""
        ql = q_text.lower()
        for toy in toy_list:
            tl = toy.lower().strip()
            if not tl:
                continue
            # Word-boundary check so "carp" doesn't match "carpet"
            if re.search(rf'\b{re.escape(tl)}\b', ql):
                return tl
        return None

    last_obj = None
    for attempt in range(2):
        raw = _claude_generate(prompt, system="You generate game questions for children. Return JSON only.",
                               label="scene-game-question")
        if not raw:
            continue
        try:
            print(f"[SceneGame] Gemini raw question response (attempt {attempt + 1}): {raw}")
            obj = _extract_json(raw)
            print(f"[SceneGame] Parsed question JSON: {json.dumps(obj, indent=2) if obj else None}")
            if obj and obj.get('question', '').strip():
                q = obj['question'].strip()
                last_obj = obj
                leaked = _question_leaks_target(q)
                if leaked:
                    print(f"[SceneGame] Question leaked target name '{leaked}'. Retrying...")
                    # Strengthen the prompt for the retry
                    prompt = (
                        prompt
                        + f"\n\nPREVIOUS ATTEMPT FAILED — your last question contained the\n"
                        + f"forbidden word \"{leaked}\". Rewrite the question so it contains\n"
                        + f"NONE of these words: {', '.join(toy_list)}. Refer to the target\n"
                        + f"only as \"it\" or \"something\"."
                    )
                    continue
                return {
                    'question': q,
                    'target': None,
                    'criteria': obj.get('criteria', ''),
                    'mode': 'criteria'
                }
        except Exception as e:
            print(f"[SceneGame] Question generation failed: {e}")

    # If both attempts leaked the target name, sanitize the last question by
    # replacing the leaked toy name with "it".
    if last_obj and last_obj.get('question'):
        q = last_obj['question'].strip()
        for toy in toy_list:
            tl = toy.lower().strip()
            if tl:
                q = re.sub(rf'\b{re.escape(tl)}\b', 'it', q, flags=re.IGNORECASE)
        q = re.sub(r'\s{2,}', ' ', q).strip()
        print(f"[SceneGame] Returning sanitized question: {q}")
        return {
            'question': q,
            'target': None,
            'criteria': last_obj.get('criteria', ''),
            'mode': 'criteria'
        }

    # Final fallback: a generic inference question keyed to the picked target's
    # color/category words is hard to derive without metadata, so use a safe
    # generic phrasing that doesn't name any specific toy.
    return {
        'question': "Can you find something special in front of you?",
        'target': None,
        'criteria': target,
        'mode': 'criteria'
    }


# ---------- Direction mode (spatial preposition teaching) ----------

# Canonical relation key -> list of natural phrases the spoken instruction
# may use (chosen randomly per round so the child hears varied vocabulary).
# The canonical key is the only thing passed to the worker for grounding.
DIRECTION_RELATION_PHRASES = {
    "next_to":     ["next to", "beside"],
    "above":       ["on top of", "above", "on"],
    "under":       ["under", "below"],
    "behind":      ["behind"],
    "in_front_of": ["in front of"],
    # Containment relations — taught to children aged 3 and under as the
    # first spatial concepts before they move on to the richer 2D/3D set.
    "in":          ["in", "inside"],
    "out":         ["out of", "outside"],
}

# ---------- Toy categorisation ----------
#
# Every toy is sorted into a semantic category. The "container" category is
# special: it supplies the *destination* of a spatial-reasoning round (the
# place an object is put into / onto). Every other category supplies the
# *object* being moved. Matching is whole-word and case-insensitive, so
# "blue dinosaur" -> dinosaur and "red block" -> block.
#
# Order matters: "container" is checked first so e.g. a "lunch box" is treated
# as a container rather than food.
SCENE_TOY_CATEGORIES = {
    "container": {
        "tray", "box", "bowl", "basket", "bucket", "cup", "jar", "pot",
        "tin", "can", "mug", "plate", "dish", "crate", "container",
        "pan", "saucer", "pail", "vase", "carton", "tub", "glass", "bin",
    },
    "fruit": {
        "apple", "banana", "orange", "lemon", "lime", "grape", "grapes",
        "strawberry", "pear", "peach", "cherry", "watermelon", "melon",
        "mango", "tomato", "pineapple", "kiwi", "plum", "blueberry",
        "raspberry", "blackberry", "apricot", "fig", "papaya", "avocado",
        "pomegranate", "coconut",
    },
    "dinosaur": {
        "dinosaur", "dino", "trex", "rex", "raptor", "triceratops",
        "stegosaurus", "brontosaurus", "velociraptor", "pterodactyl",
        "diplodocus", "spinosaurus", "ankylosaurus", "pterosaur",
    },
    "food": {
        "cookie", "cake", "bread", "pizza", "sandwich", "donut", "doughnut",
        "egg", "carrot", "broccoli", "candy", "biscuit", "cheese", "hotdog",
        "burger", "muffin", "pretzel", "cupcake", "lollipop", "icecream",
        "cracker", "popcorn", "fries", "waffle", "pancake", "taco", "noodle",
        "noodles", "corn", "chocolate", "jelly", "sausage", "meatball",
        "pasta", "rice", "sushi", "pie",
    },
    "animal": {
        "dog", "puppy", "cat", "kitten", "bear", "teddy", "lion", "tiger",
        "elephant", "giraffe", "monkey", "rabbit", "bunny", "duck", "cow",
        "horse", "pig", "sheep", "frog", "fox", "panda", "zebra", "penguin",
        "owl", "fish", "shark", "whale", "dolphin", "turtle", "snake", "bird",
        "chicken", "goat", "mouse", "hamster", "koala", "hippo", "rhino",
        "deer", "wolf", "crocodile", "alligator", "lizard", "butterfly",
        "bee", "ladybug", "snail", "crab", "octopus", "unicorn", "dragon",
    },
    "vehicle": {
        "car", "truck", "bus", "train", "plane", "airplane", "helicopter",
        "boat", "ship", "rocket", "bike", "bicycle", "motorcycle", "tractor",
        "van", "taxi", "ambulance", "firetruck", "digger", "excavator",
        "scooter", "jet", "submarine",
    },
    "block": {
        "block", "blocks", "cube", "cubes", "lego", "legos", "brick", "bricks",
        "domino", "dominoes", "duplo", "duplos", "jenga",
    },
    "toy": {
        "doll", "ball", "robot", "book", "puzzle", "balloon", "top", "spinner",
        "kite", "crayon", "marker", "drum", "bell", "whistle", "slinky",
        "figure", "figurine", "yoyo",
    },
}

# Containers that are flat surfaces use the preposition "on"; everything else
# is an enclosing container and uses "in".
SCENE_SURFACE_CONTAINERS = {"tray", "plate", "dish", "mat", "table"}


def _categorize_toy(toy_name):
    """Return the category key for a toy name, or ``'other'`` if unmatched.

    Matching is whole-word and case-insensitive. The first category in
    ``SCENE_TOY_CATEGORIES`` order with a matching word wins (container first),
    so "blue dinosaur" -> 'dinosaur' and "red block" -> 'block'.
    """
    if not toy_name:
        return "other"
    tokens = set(re.findall(r"[a-z]+", toy_name.lower()))
    for category, words in SCENE_TOY_CATEGORIES.items():
        if tokens & words:
            return category
    return "other"


def _is_direction_container(toy_name):
    """True if a toy is a container (the destination of a direction round)."""
    return _categorize_toy(toy_name) == "container"


def _scene_game_generate_direction_question(toy_list, child_age=None):
    """Build a spatial-reasoning round of the form
    "Let's put the <object> <relation> the <reference>".

    A relation is chosen at random from whatever the current toys can support:

      * Containment ("in", "on") needs a container destination plus a different
        object to move into / onto it. Enclosing containers (box, bowl, basket)
        read as "in" (canonical ``in``); flat surfaces (tray, plate) read as
        "on" (canonical ``above``).
      * Positional ("next to" / "beside", "under" / "below", "behind") works
        with any two distinct toys — the reference need not be a container.

    Returns None — so the caller falls back to another mode — when the toy
    list cannot support any relation (fewer than two toys, and no
    container+object pair).

    ``child_age`` is accepted for call-site compatibility but no longer steers
    relation choice; difficulty now comes from the object/relation mix.
    """
    # Group toys by category.
    by_category = {}
    for toy in toy_list:
        by_category.setdefault(_categorize_toy(toy), []).append(toy)

    containers = by_category.get("container", [])
    non_containers = [t for c, toys in by_category.items()
                      if c != "container" for t in toys]
    all_toys = list(toy_list)

    # Split containers into flat surfaces ("on" / canonical 'above') and
    # enclosing containers ("in" / canonical 'in').
    surface_containers, enclosing_containers = [], []
    for c in containers:
        tokens = set(re.findall(r"[a-z]+", c.lower()))
        (surface_containers if tokens & SCENE_SURFACE_CONTAINERS
         else enclosing_containers).append(c)

    # Build the menu of relations the current toys can actually pose. Each key
    # is a canonical relation; containment relations carry their destination
    # pool so we don't re-derive it below.
    candidates = []
    if enclosing_containers and non_containers:
        candidates.append("in")
    if surface_containers and non_containers:
        candidates.append("on")          # canonical 'above', spoken "on"
    if len(all_toys) >= 2:
        candidates.extend(["next_to", "under", "behind"])

    if not candidates:
        print(f"[SceneGame] direction mode needs two distinct toys, or a "
              f"container plus an object; current toys: {toy_list}")
        return None

    choice = random.choice(candidates)

    if choice == "in":
        obj_a = random.choice(non_containers)
        obj_b = random.choice(enclosing_containers)
        relation, phrase = "in", "in"
    elif choice == "on":
        obj_a = random.choice(non_containers)
        obj_b = random.choice(surface_containers)
        relation, phrase = "above", "on"
    else:
        # Positional relation: any two distinct toys; varied spoken vocabulary.
        obj_a, obj_b = random.sample(all_toys, 2)
        relation = choice
        phrase = random.choice(DIRECTION_RELATION_PHRASES[relation])

    question = f"Let's put the {obj_a} {phrase} the {obj_b}."
    return {
        'question': question,
        'mode': 'direction',
        'obj_a': obj_a,
        'obj_b': obj_b,
        'relation': relation,
        'phrase': phrase,
        'category': _categorize_toy(obj_a),
        # `target` / `criteria` kept for response-shape uniformity with
        # the other modes; downstream consumers ignore them in direction mode.
        'target': None,
        'criteria': None,
    }


# Reverse of DIRECTION_RELATION_PHRASES: spoken phrase -> canonical relation.
# Longer phrases listed naturally (lookup is exact on the phrase string).
DIRECTION_PHRASE_TO_RELATION = {
    phrase: relation
    for relation, phrases in DIRECTION_RELATION_PHRASES.items()
    for phrase in phrases
}

# Olivia plays a fixed, curated set of direction-mode rounds instead of the
# randomly generated ones. Each entry is (obj_a, spoken phrase, obj_b); the
# canonical relation is derived from the phrase so the round still validates
# through the normal spatial validator (api_scene_game_answer).
OLIVIA_DIRECTION_ROUNDS = [
    ("grape", "in", "box"),
    ("tray", "in front of", "box"),
    ("lemon", "on", "tray"),
    ("banana", "in", "bowl"),
]


def _scene_game_olivia_direction_question():
    """Return one fixed direction-mode round for Olivia, chosen at random.

    Mirrors the result shape of ``_scene_game_generate_direction_question`` so
    the round behaves like any other (spoken prompt, spatial validation, hints).
    """
    obj_a, phrase, obj_b = random.choice(OLIVIA_DIRECTION_ROUNDS)
    relation = DIRECTION_PHRASE_TO_RELATION.get(phrase, phrase)
    return {
        'question': f"Let's put the {obj_a} {phrase} the {obj_b}.",
        'mode': 'direction',
        'obj_a': obj_a,
        'obj_b': obj_b,
        'relation': relation,
        'phrase': phrase,
        'category': _categorize_toy(obj_a),
        'target': None,
        'criteria': None,
    }


def _run_gemini_validate_spatial(image_path, obj_a, obj_b, relation, toy_list=None):
    """Run gemini_validate_spatial.py and return its parsed JSON, or None on failure."""
    script_path = os.path.join(os.path.dirname(BASE_DIR), 'scripts', 'gemini_validate_spatial.py')
    if not os.path.exists(script_path):
        print("[SceneGame] gemini_validate_spatial.py not found")
        return None
    cmd = [
        WORKER_PYTHON, script_path,
        '--image', image_path,
        '--obj-a', obj_a,
        '--obj-b', obj_b,
        '--relation', relation,
    ]
    if toy_list:
        cmd.extend(['--toy-list', ','.join(toy_list)])
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if proc.returncode != 0:
            print(f"[SceneGame] validate_spatial error: {proc.stderr.strip()}")
            return None
        raw = (proc.stdout or '').strip()
        print(f"[SceneGame] gemini_validate_spatial raw: {raw}")
        if raw.startswith('```'):
            raw = raw.strip('`').strip()
            if raw.startswith('json'):
                raw = raw[4:].strip()
        return json.loads(raw)
    except Exception as e:
        print(f"[SceneGame] validate_spatial exec failed: {e}")
        return None


# Relations where a single 2D frame cannot reliably disambiguate the
# configuration. For these we capture a short MP4 and let Gemini reason
# over the clip using parallax/occlusion/containment across frames.
#   - behind / in_front_of: pure depth, needs parallax cues.
#   - in / out: containment hinges on partial occlusion of the inside
#     object by the container, which is far easier to read across a few
#     frames of motion than from a single static frame.
VIDEO_DEPTH_RELATIONS = {"behind", "in_front_of", "in", "out"}

DIRECTION_VIDEO_DURATION_SEC = 3.0
DIRECTION_VIDEO_FPS = 10


def _capture_scene_video(out_path, duration_sec=DIRECTION_VIDEO_DURATION_SEC,
                         fps=DIRECTION_VIDEO_FPS):
    """Record ``duration_sec`` of frames from the ROS camera into ``out_path``.

    Returns True on success, False on any capture/writer failure so the
    caller can fall back to the single-frame validator instead of erroring
    the whole round.
    """
    if cv2 is None:
        print("[SceneGame] OpenCV not available — cannot record video")
        return False
    first = _get_ros_frame()
    if first is None:
        print("[SceneGame] No initial frame for video capture")
        return False
    h, w = first.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, float(fps), (w, h))
    if not writer.isOpened():
        print(f"[SceneGame] VideoWriter failed to open for {out_path}")
        return False
    try:
        writer.write(first)
        n_frames = max(1, int(round(fps * duration_sec)))
        period = 1.0 / float(fps)
        start = time.time()
        for i in range(1, n_frames):
            target = start + i * period
            now = time.time()
            if target > now:
                time.sleep(target - now)
            frame = _get_ros_frame()
            if frame is None:
                continue
            if frame.shape[:2] != (h, w):
                frame = cv2.resize(frame, (w, h))
            writer.write(frame)
        return True
    finally:
        writer.release()


def _run_gemini_validate_spatial_video(video_path, obj_a, obj_b, relation, toy_list=None):
    """Run gemini_validate_spatial_video.py for a captured MP4 clip."""
    script_path = os.path.join(os.path.dirname(BASE_DIR), 'scripts', 'gemini_validate_spatial_video.py')
    if not os.path.exists(script_path):
        print("[SceneGame] gemini_validate_spatial_video.py not found")
        return None
    cmd = [
        WORKER_PYTHON, script_path,
        '--video', video_path,
        '--obj-a', obj_a,
        '--obj-b', obj_b,
        '--relation', relation,
    ]
    if toy_list:
        cmd.extend(['--toy-list', ','.join(toy_list)])
    try:
        # Generous timeout: video upload + Files API processing + inference
        # can take 20-60s for a 3s clip.
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if proc.returncode != 0:
            print(f"[SceneGame] validate_spatial_video error: {proc.stderr.strip()}")
            return None
        raw = (proc.stdout or '').strip()
        print(f"[SceneGame] gemini_validate_spatial_video raw: {raw}")
        if raw.startswith('```'):
            raw = raw.strip('`').strip()
            if raw.startswith('json'):
                raw = raw[4:].strip()
        return json.loads(raw)
    except Exception as e:
        print(f"[SceneGame] validate_spatial_video exec failed: {e}")
        return None


def _run_gemini_detect_and_look(image_path):
    """Run gemini_analyze_image.py to detect the held object and make the robot look at it.

    Returns a dict with 'label', 'color', 'shape' (and optionally 'point'), or None on failure.
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
        # obj is typically [{"point": [y, x], "label": "...", "color": "...", "shape": "..."}]
        item = obj[0] if isinstance(obj, list) and obj else obj
        if not isinstance(item, dict):
            return None
        label = (item.get('label') or '').strip()
        color = (item.get('color') or '').strip()
        shape = (item.get('shape') or '').strip()
        point = item.get('point')
        print(f"[SceneGame] Detected object: {label}, color: {color}, shape: {shape}, point: {point}")

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

        return {'label': label or None, 'color': color or None, 'shape': shape or None, 'point': point}
    except Exception as e:
        print(f"[SceneGame] analyze script exec failed: {e}")
        return None


def _check_criteria_match(detected_label, criteria, detected_color=None, detected_shape=None):
    """Check whether a detected object matches descriptive criteria, using
    EVERY attribute returned by the robotics-API analyzer (label + color +
    shape), not just the label.

    Example: criteria \"red car\" + detection {label: \"toy car\", color:
    \"red\"} should match — the label alone wouldn't, but color seals it.

    Returns (matches: bool, reason: str).
    """
    crit_lower = (criteria or '').lower().strip()
    label_lower = (detected_label or '').lower().strip()
    color_lower = (detected_color or '').lower().strip()
    shape_lower = (detected_shape or '').lower().strip()

    # Fast path: token-level satisfaction. Split criteria into words; require
    # every word to be supported by SOME attribute of the detection. Stop-words
    # ("a", "the", "an") are skipped.
    stop = {"a", "an", "the", "some", "any", "this", "that"}
    crit_tokens = [t for t in re.split(r'\s+', crit_lower) if t and t not in stop]
    if crit_tokens:
        attribute_blob = ' '.join([label_lower, color_lower, shape_lower]).strip()
        all_satisfied = True
        for tok in crit_tokens:
            tok_clean = re.sub(r'[^\w-]', '', tok)
            if not tok_clean:
                continue
            # Require token (or its singular/plural variant) to appear in any
            # attribute. Loose substring match handles "car" vs "toy car".
            variants = {tok_clean, tok_clean.rstrip('s'), tok_clean + 's'}
            if not any(v and v in attribute_blob for v in variants):
                all_satisfied = False
                break
        if all_satisfied:
            return True, f"matched via attributes (label={detected_label!r}, color={detected_color!r}, shape={detected_shape!r})"

    # LLM path: ask Claude to reason across all attributes
    prompt = (
        f"A child showed a physical object. The vision system detected:\n"
        f"  label: \"{detected_label or 'unknown'}\"\n"
        f"  color: \"{detected_color or 'unknown'}\"\n"
        f"  shape: \"{detected_shape or 'unknown'}\"\n"
        f"The game asked for: \"{criteria}\".\n"
        f"\n"
        f"Decide if the detected object matches the criteria. Use ALL of the\n"
        f"detected attributes (label AND color AND shape), not just the label.\n"
        f"\n"
        f"Examples of correct matches:\n"
        f"- criteria \"red car\" + label \"toy car\" + color \"red\" → MATCH\n"
        f"  (label gives \"car\", color gives \"red\")\n"
        f"- criteria \"yellow fruit\" + label \"banana\" + color \"yellow\" → MATCH\n"
        f"- criteria \"green dinosaur\" + label \"t-rex\" + color \"green\" → MATCH\n"
        f"- criteria \"round ball\" + label \"ball\" + shape \"round\" → MATCH\n"
        f"\n"
        f"Examples of non-matches:\n"
        f"- criteria \"red car\" + label \"truck\" + color \"red\" → NO MATCH\n"
        f"  (\"truck\" isn't a car)\n"
        f"- criteria \"red apple\" + label \"apple\" + color \"green\" → NO MATCH\n"
        f"\n"
        f"Return ONLY a JSON object: {{\"match\": true or false, \"reason\": \"<brief explanation>\"}}"
    )
    raw = _claude_generate(prompt, system="You validate object matches. Return JSON only.",
                           label="scene-game-criteria-match")
    if raw:
        try:
            print(f"[SceneGame] Criteria match Claude raw: {raw}")
            obj = _extract_json(raw)
            print(f"[SceneGame] Criteria match parsed: {json.dumps(obj, indent=2) if obj else None}")
            if obj:
                return bool(obj.get('match', False)), obj.get('reason', '')
        except Exception as e:
            print(f"[SceneGame] Criteria match failed: {e}")

    # Fallback: lenient string contain across all attributes
    blob = ' '.join([label_lower, color_lower, shape_lower])
    match = bool(crit_lower) and (
        crit_lower in blob or any(tok in blob for tok in crit_tokens)
    )
    return match, "fallback string match (all attributes)"


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


@app.route('/api/scene_game/hint', methods=['POST'])
def api_scene_game_hint():
    """Generate an age-appropriate hint for the current round's target object."""
    if 'username' not in session:
        return jsonify({'success': False, 'error': 'Not logged in'}), 401
    data = request.get_json() or {}
    mode = (data.get('mode') or 'exact').strip()
    target = (data.get('target') or '').strip()
    criteria = (data.get('criteria') or '').strip()
    obj_a = (data.get('obj_a') or '').strip()
    obj_b = (data.get('obj_b') or '').strip()
    relation = (data.get('relation') or '').strip()

    username = session['username']
    child_age, learning_goals = _get_user_age_and_goals(username)

    # Direction mode hint: deterministic restatement of the spatial relation,
    # phrased as a coaching cue. No LLM round-trip needed.
    if mode == 'direction':
        if not (obj_a and obj_b and relation):
            return jsonify({'success': False, 'error': 'No active direction round'})
        phrase = DIRECTION_RELATION_PHRASES.get(relation, [relation])[0]
        hint = (f"Look! The {obj_a} should be {phrase} the {obj_b}. "
                f"Try moving the {obj_a}.")
        try:
            _with_asr_suspended(lambda: tts_helper.speak(hint))
        except Exception:
            pass
        return jsonify({'success': True, 'hint': hint})

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
    raw = _claude_generate(prompt, system="You generate game hints for children. Return JSON only.",
                           label="scene-game-hint")
    if raw:
        try:
            print(f"[SceneGame] Claude raw hint response: {raw}")
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
    obj_a = (data.get('obj_a') or '').strip()
    obj_b = (data.get('obj_b') or '').strip()
    relation = (data.get('relation') or '').strip()

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

    # Direction mode: two-object + spatial-relation validator. Different
    # response shape from the single-object modes, so dispatch here and
    # short-circuit the rest of the function.
    if answer_mode == 'direction':
        if not (obj_a and obj_b and relation):
            return jsonify({'success': False,
                            'error': 'Missing obj_a/obj_b/relation for direction mode'}), 400
        toy_list = _load_scene_toys() or list(SCENE_GAME_DEFAULT_TOYS)
        used_video = False
        if relation in VIDEO_DEPTH_RELATIONS:
            # Depth relations need temporal/parallax cues that a single
            # still frame can't provide. Record a short MP4 and ship it
            # to the video worker. Fall back to the still-frame validator
            # if recording fails so a missing codec doesn't break the round.
            video_path = os.path.join(cap_dir, f'scene_answer_{ts}.mp4')
            if _capture_scene_video(video_path,
                                    duration_sec=DIRECTION_VIDEO_DURATION_SEC,
                                    fps=DIRECTION_VIDEO_FPS):
                used_video = True
                result = _run_gemini_validate_spatial_video(
                    video_path, obj_a, obj_b, relation, toy_list=toy_list,
                )
            else:
                print("[SceneGame] video capture failed for depth relation; "
                      "falling back to still-frame validator")
                result = _run_gemini_validate_spatial(fpath, obj_a, obj_b, relation, toy_list=toy_list)
        else:
            result = _run_gemini_validate_spatial(fpath, obj_a, obj_b, relation, toy_list=toy_list)
        if result is None:
            try:
                tts_helper.speak("I couldn't see clearly. Can you show me again?")
            except Exception:
                pass
            return jsonify({'success': True, 'correct': None,
                            'obj_a': obj_a, 'obj_b': obj_b, 'relation': relation,
                            'actual_relation': None,
                            'error': 'Vision analysis failed'})
        correct = bool(result.get('correct'))
        actual_relation = result.get('actual_relation') or 'other'
        reason = (result.get('reason') or '').strip()
        # Served URL of the exact media Gemini analysed, so the UI can show the
        # captured frame/clip next to the prompt and the model's raw response.
        image_url = '/images/' + os.path.relpath(fpath, USER_DATA_DIR)
        video_url = None
        if used_video:
            video_url = '/images/' + os.path.relpath(video_path, USER_DATA_DIR)
        try:
            if correct:
                tts_helper.speak(f"Great job! The {obj_a} is {DIRECTION_RELATION_PHRASES[relation][0]} the {obj_b}!")
            elif not result.get('obj_a_found', True) or not result.get('obj_b_found', True):
                missing = obj_a if not result.get('obj_a_found') else obj_b
                tts_helper.speak(f"I can't see the {missing}. Try again!")
            else:
                tts_helper.speak(f"Not quite — try moving the {obj_a}.")
        except Exception:
            pass
        return jsonify({
            'success': True,
            'correct': correct,
            'mode': 'direction',
            'obj_a': obj_a,
            'obj_b': obj_b,
            'relation': relation,
            'actual_relation': actual_relation,
            'obj_a_found': bool(result.get('obj_a_found', False)),
            'obj_b_found': bool(result.get('obj_b_found', False)),
            'reason': reason,
            'used_video': used_video,
            # Inference transparency: the exact media, prompt and model output.
            'image_url': image_url,
            'video_url': video_url,
            'prompt': result.get('prompt'),
            'raw_response': result.get('raw_response'),
        })

    # Step 1: detect object + robot looks at it
    detection = _run_gemini_detect_and_look(fpath)
    if not detection or not detection.get('label'):
        try:
            tts_helper.speak("I couldn't see clearly. Can you show me again?")
        except Exception:
            pass
        return jsonify({'success': True, 'correct': None, 'detected': None,
                        'color': None, 'shape': None,
                        'error': 'Vision analysis failed'})

    detected = detection['label']
    detected_color = detection.get('color')
    detected_shape = detection.get('shape')
    print(f"[SceneGame] Detected object: {detected}, color: {detected_color}, shape: {detected_shape}")

    # Step 2: match against target or criteria
    if answer_mode == 'exact':
        # Ages 2-3: name comparison, but use ALL detection attributes so
        # e.g. target "red car" matches label "toy car" + color "red".
        target_lower = target.lower().strip()
        detected_lower = detected.lower().strip()
        attr_blob = ' '.join([detected_lower,
                              (detected_color or '').lower().strip(),
                              (detected_shape or '').lower().strip()]).strip()
        target_tokens = [t for t in re.split(r'\s+', target_lower) if t and t not in {"a", "an", "the"}]
        correct = detected_lower == target_lower
        if not correct:
            correct = (target_lower in detected_lower) or (detected_lower in target_lower)
        if not correct and target_tokens:
            # All target tokens must appear across label+color+shape
            correct = all(
                any(v in attr_blob for v in {tok, tok.rstrip('s'), tok + 's'})
                for tok in target_tokens
            )
        reason = ''
        try:
            if correct:
                tts_helper.speak(f"Great job! That's the {target}!")
            else:
                tts_helper.speak(f"I see a {detected}, but I asked for the {target}. Try again!")
        except Exception:
            pass
    else:
        # Ages 4+: criteria-based — use LLM with full detection attributes
        if not criteria:
            return jsonify({'success': False, 'error': 'No criteria provided'}), 400
        correct, reason = _check_criteria_match(detected, criteria,
                                                detected_color=detected_color,
                                                detected_shape=detected_shape)
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
        'color': detected_color,
        'shape': detected_shape,
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
        # Extract label, color, shape from returned JSON/text
        detected_label = None
        detected_color = None
        detected_shape = None
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
                item = None
                if isinstance(obj, list) and obj:
                    item = obj[0]
                elif isinstance(obj, dict):
                    item = obj
                if isinstance(item, dict):
                    lbl = item.get('label')
                    if isinstance(lbl, str) and lbl.strip():
                        detected_label = lbl.strip()
                    clr = item.get('color')
                    if isinstance(clr, str) and clr.strip():
                        detected_color = clr.strip()
                    shp = item.get('shape')
                    if isinstance(shp, str) and shp.strip():
                        detected_shape = shp.strip()
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
        return jsonify({'success': True, 'image_path': f"/images/{rel}", 'label': detected_label,
                        'color': detected_color, 'shape': detected_shape,
                        'target': target, 'found': found})
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

        # Include questions if available; generate on-the-fly for older stories
        # Also regenerate if old format (missing correct_answer/wrong_answers)
        questions = story_data.get('questions', [])
        needs_regen = (not questions
                       or (questions and 'correct_answer' not in questions[0]))
        if needs_regen:
            story_text = story_data.get('story', '')
            child_age = 5
            try:
                child_age = int(metadata.get('age', 5))
            except (ValueError, TypeError):
                child_age = 5
            child_name = metadata.get('child_name', 'the child')
            q_persona_ctx = _persona_context_for(username, child_age, kind="question")
            questions = _generate_story_questions(
                story_text, child_age, child_name,
                persona_context=q_persona_ctx,
                language_age=_language_age_for(username, child_age),
            )
            # Save back to the story file so we don't regenerate next time
            if questions:
                try:
                    story_data['questions'] = questions
                    with open(story_path, 'w') as f:
                        json.dump(story_data, f, indent=2)
                    print(f"[StoryQuestions] Backfilled {len(questions)} questions for {filename}")
                except Exception as e:
                    print(f"[StoryQuestions] Failed to backfill questions: {e}")

        # Takeaways are populated for age 7+ stories. For older stories that
        # predate the takeaways feature, the field will be missing -> default [].
        takeaways = story_data.get('takeaways', []) or []
        if not isinstance(takeaways, list):
            takeaways = []

        return jsonify({
            'success': True,
            'sentences': sentences,
            'metadata': metadata,
            'images_available': image_generator.is_available(),
            'questions': questions,
            'takeaways': takeaways,
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

def _normalize_tag_name(name):
    """Normalize tag names: QThappy -> QT/happy, strip punctuation."""
    name = name.strip().rstrip('.,;!?')
    if name.upper().startswith('QT') and '/' not in name:
        name = 'QT/' + name[2:]
    return name


def _split_page_into_segments(page_text):
    """Split a page into segments at gesture/emotion tag boundaries.

    Returns a list of (text, gestures, emotions) tuples.
    Each segment is the text that follows its tags until the next tag or end of page.

    Example input:
      "Hello world. [gesture:hi] Nice to meet you. [emotion:QT/happy] I am happy."
    Returns:
      [("Hello world.", [], []),
       ("Nice to meet you.", ["hi"], []),
       ("I am happy.", [], ["QT/happy"])]
    """
    import re
    # Match all tag variants
    tag_re = re.compile(
        r'(\[gesture:[^\]]+\]|\[emotion:[^\]]+\]|\bgesture:\S+|\bemotion:\S+)',
        re.IGNORECASE
    )
    gesture_val_re = re.compile(r'\[gesture:([^\]]+)\]|gesture:(\S+)', re.IGNORECASE)
    emotion_val_re = re.compile(r'\[emotion:([^\]]+)\]|emotion:(\S+)', re.IGNORECASE)

    # Split text by tags, keeping the tags as separators
    parts = tag_re.split(page_text)

    segments = []
    pending_gestures = []
    pending_emotions = []

    for part in parts:
        part_stripped = part.strip()
        if not part_stripped:
            continue

        # Check if this part is a tag
        gm = gesture_val_re.fullmatch(part_stripped)
        em = emotion_val_re.fullmatch(part_stripped)
        if gm:
            val = gm.group(1) or gm.group(2)
            pending_gestures.append(_normalize_tag_name(val))
        elif em:
            val = em.group(1) or em.group(2)
            pending_emotions.append(_normalize_tag_name(val))
        else:
            # This is text — attach any pending tags to it
            cleaned = re.sub(r'\s{2,}', ' ', part_stripped).strip()
            if cleaned:
                segments.append((cleaned, pending_gestures, pending_emotions))
                pending_gestures = []
                pending_emotions = []

    # If there are leftover tags with no following text, attach to last segment
    if (pending_gestures or pending_emotions) and segments:
        text, g, e = segments[-1]
        segments[-1] = (text, g + pending_gestures, e + pending_emotions)

    # If no segments were created, return the whole page as one segment
    if not segments:
        cleaned = tag_re.sub('', page_text)
        cleaned = re.sub(r'\s{2,}', ' ', cleaned).strip()
        return [(cleaned, [], [])]

    # Deduplicate: track which gesture/emotion names have been seen
    seen_gestures = set()
    seen_emotions = set()
    deduped = []
    for text, gestures, emotions in segments:
        new_g = [g for g in gestures if g not in seen_gestures]
        new_e = [e for e in emotions if e not in seen_emotions]
        seen_gestures.update(new_g)
        seen_emotions.update(new_e)
        deduped.append((text, new_g, new_e))

    return deduped


# Valid robot emotions (must match the set the QT robot can actually display).
# Anything else is remapped to the closest available expression so hallucinated
# names from the LLM (e.g. "QT/relieved") don't silently fail.
_VALID_EMOTIONS = {"QT/happy", "QT/sad", "QT/surprise", "QT/afraid", "QT/angry", "QT/calm", "QT/shy"}
_EMOTION_REMAP = {
    "relieved": "QT/happy", "joyful": "QT/happy", "excited": "QT/happy",
    "proud": "QT/happy", "grateful": "QT/happy", "delighted": "QT/happy",
    "amused": "QT/happy", "content": "QT/calm", "peaceful": "QT/calm",
    "frustrated": "QT/angry", "annoyed": "QT/angry", "mad": "QT/angry",
    "scared": "QT/afraid", "fearful": "QT/afraid", "nervous": "QT/afraid",
    "worried": "QT/afraid", "anxious": "QT/afraid",
    "upset": "QT/sad", "disappointed": "QT/sad", "lonely": "QT/sad",
    "shocked": "QT/surprise", "amazed": "QT/surprise", "astonished": "QT/surprise",
    "embarrassed": "QT/shy", "bashful": "QT/shy",
}


def _resolve_emotion(name):
    """Normalize an emotion tag to a valid QT/ name, or return None to skip."""
    raw = (name or "").strip()
    if not raw:
        return None
    # Normalize prefix and case
    if raw.upper().startswith("QT/"):
        bare = raw[3:]
    elif raw.upper().startswith("QT"):
        bare = raw[2:]
    else:
        bare = raw
    bare = bare.lower().strip().strip('/')
    candidate = f"QT/{bare}"
    # Match valid set case-insensitively
    for v in _VALID_EMOTIONS:
        if v.lower() == candidate.lower():
            return v
    # Try remap table
    if bare in _EMOTION_REMAP:
        return _EMOTION_REMAP[bare]
    print(f"[StoryTags] unknown emotion '{name}' — skipping (no remap)")
    return None


def _play_tags(gestures, emotions, pre_speech_pause=1.0):
    """Fire gesture/emotion via rostopic pub and optionally wait for the
    motion to start before the caller continues to TTS.

    `pre_speech_pause` is the number of seconds to block after publishing the
    ROS topic(s). Use ~1.0s for sentence-boundary tags so the new face/gesture
    is visibly set up before the next sentence is spoken. Use 0s for
    mid-sentence tags (after a comma/semicolon/colon) so the face change lands
    concurrently with the following clause instead of inserting a long pause
    inside what is grammatically one sentence.
    """
    import time
    import threading

    if not gestures and not emotions:
        return

    def _rostopic_pub(topic, data):
        try:
            subprocess.Popen(
                ['rostopic', 'pub', '--once', topic, 'std_msgs/String', f'data: \'{data}\''],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            print(f"[StoryTags] published {topic}: {data}")
        except Exception as e:
            print(f"[StoryTags] rostopic pub failed: {e}")

    for g in gestures:
        name = g.strip()
        if not name.startswith("QT/"):
            name = "QT/" + name
        threading.Thread(target=_rostopic_pub, args=('/qt_robot/gesture/play', name), daemon=True).start()

    for e in emotions:
        resolved = _resolve_emotion(e)
        if not resolved:
            continue
        threading.Thread(target=_rostopic_pub, args=('/qt_robot/emotion/show', resolved), daemon=True).start()

    if pre_speech_pause > 0:
        time.sleep(pre_speech_pause)


# Punctuation that ends a "clause but not a sentence". A tag that sits right
# after one of these belongs to the next clause of the SAME sentence and
# should fire concurrently with the next TTS call, not after a 1-second pause.
_CLAUSE_INTERNAL_TERMINATORS = (",", ";", ":", "—", "–")


# Common abbreviations whose trailing period is NOT a sentence boundary.
# Stored lowercase, without the trailing period, with internal periods kept
# (so the multi-letter "u.s.a" matches strings ending in "U.S.A.").
_SENTENCE_ABBREVIATIONS = {
    "mr", "mrs", "ms", "mx", "dr", "st", "jr", "sr",
    "prof", "rev", "hon", "sgt", "capt", "lt", "col", "gen",
    "etc", "vs", "inc", "co", "ltd", "corp", "no",
    "i.e", "e.g", "p.s", "p.p.s",
    "u.s", "u.s.a", "u.k", "u.n", "e.u",
}


def _ends_with_abbreviation(chunk):
    """Return True if `chunk` ends with a known abbreviation or a single
    capital-letter + period (e.g., "T." in "T. Rex"). Both indicate that the
    naive sentence-split regex made a false boundary here.
    """
    # Drop trailing closing quotes/parens so the last token is the word itself.
    s = chunk.rstrip(' "\'”’‘“)')
    if not s.endswith("."):
        return False
    # Single capital letter ending: "T.", "A." (used for initials and "T. Rex").
    if re.search(r"(?<![A-Za-z])[A-Z]\.$", s):
        return True
    # Last token (letters with optional internal periods) ending in ".".
    m = re.search(r"([A-Za-z][A-Za-z.]*)\.$", s)
    if not m:
        return False
    word = m.group(1).lower().rstrip(".")
    return word in _SENTENCE_ABBREVIATIONS


def _split_into_sentences(text):
    """Split a paragraph of plain (tag-stripped) text into individual sentences.

    Strategy:
      1. Split aggressively on sentence-ending punctuation followed by whitespace.
      2. Merge adjacent parts back together when the preceding part ends with a
         known abbreviation (Mr., Mrs., Dr., St., U.S.A., etc.) or with a single
         capital letter + period (e.g., "T." in "T. Rex"). Both cases indicate
         the regex made a false sentence boundary.

    Strong enough for children's stories — handles titles, initials, and
    common period-bearing abbreviations without an NLP dependency.
    """
    if not text or not text.strip():
        return []
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    merged = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if merged and _ends_with_abbreviation(merged[-1]):
            merged[-1] = merged[-1] + " " + part
        else:
            merged.append(part)
    return merged


@app.route('/api/speak_sentence', methods=['POST'])
def api_speak_sentence():
    username = session.get('username')
    data = request.get_json() or {}
    sentence = data.get('sentence', '')
    filename = data.get('filename', '')
    if not username or not sentence:
        return jsonify({'success': False, 'error': 'Missing username or sentence'})

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

    # Track while speaking
    tracker = None
    try:
        tracker = _ensure_human_tracker()
        if tracker:
            person = _pick_recent_person(tracker, timeout_sec=0.5)
            tracker.track(person)
    except Exception:
        pass

    try:
        # Split the page into segments at tag boundaries. Each segment may
        # contain multiple sentences; fire the segment's gesture/emotion once
        # before its first sentence, then send each sentence to Qwen TTS
        # individually so the API receives one sentence at a time rather than
        # a whole paragraph.
        #
        # `prev_terminator` tracks the last non-whitespace, non-quote character
        # of the previous segment. If it's clause-internal punctuation (comma,
        # semicolon, colon, em-dash), the upcoming tag belongs to the same
        # sentence as the previous clause — fire it concurrently with the next
        # TTS call (no pre-speech sleep) so we don't insert an unnatural pause
        # in the middle of one sentence.
        segments = _split_page_into_segments(sentence)
        prev_terminator = None
        for text, gestures, emotions in segments:
            cleaned = clean_story_text(text)
            if not cleaned:
                continue
            mid_sentence = prev_terminator in _CLAUSE_INTERNAL_TERMINATORS
            pre_pause = 0.0 if mid_sentence else 1.0
            _play_tags(gestures, emotions, pre_speech_pause=pre_pause)
            for s in _split_into_sentences(cleaned):
                if not s:
                    continue
                _with_asr_suspended(lambda c=s: tts_helper.speak_story(c, language))
            tail = cleaned.rstrip().rstrip('"\'”’')
            prev_terminator = tail[-1] if tail else None
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
    """Set robot speaker volume (0-100) via ALSA hardware mixer over SSH.

    /qt_robot/setting/setVolume only affects the QT TTS engine (talkText) and
    does not change loudness for file-based playback (talkAudio used by the
    qwen/polly engines). The ALSA Headphone mixer on the head computer is the
    actual lever — works for all engines, real-time.
    """
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
        applied = tts_helper.set_hardware_volume(level)
        if not applied:
            return jsonify({'success': False, 'error': 'Hardware volume change failed (check ROBOT_HOST/ROBOT_USER/ROBOT_PASSWORD and sshpass)'}), 500

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

SCENE_GAME_DEFAULT_TOYS = ['lemon', 'tomato', 'apple', 'banana', 'tray', 'box','bowl']
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

        # 'auto' (default) keeps the existing age-based selection.
        # 'direction' forces the new spatial-preposition mode; falls back to
        # auto if the toy list has fewer than 2 entries.
        # 'criteria' forces the criteria/riddle path (and is also picked by
        # auto for ages 4+).
        data = request.get_json(silent=True) or {}
        requested_mode = (data.get('mode') or 'auto').strip().lower()
        if requested_mode not in ('auto', 'criteria', 'direction'):
            requested_mode = 'auto'

        result = None
        if requested_mode == 'direction':
            # Olivia always plays the fixed, curated set of direction rounds.
            if username == 'olivia':
                result = _scene_game_olivia_direction_question()
            else:
                result = _scene_game_generate_direction_question(toy_list, child_age=child_age)
            if result is None:
                # Generator bailed: either toy list <2, or in/out was the only
                # option but the list has no container. Fall through to auto.
                print("[SceneGame] direction mode unavailable for current toy list; falling back to auto")
        if result is None:
            persona_ctx = _persona_context_for(username, child_age, kind="question") if username else ''
            result = _scene_game_generate_question(
                toy_list, child_age, learning_goals, persona_context=persona_ctx,
                language_age=_language_age_for(username, child_age) if username else None,
            )

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
            'obj_a': result.get('obj_a'),
            'obj_b': result.get('obj_b'),
            'relation': result.get('relation'),
            'phrase': result.get('phrase'),
            'mode': result['mode'],
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

def _run_gemini_wh_analysis(image_path, child_age, difficulty, language_age=None):
    """Run the Gemini vision worker to analyze a scene and generate WH questions.

    ``language_age`` (developmental/language age) is forwarded to the worker so
    image questions are pitched at the child's language level rather than their
    chronological age. The worker falls back to ``child_age`` when it is absent.
    """
    script_path = os.path.join(os.path.dirname(BASE_DIR), 'scripts', 'gemini_wh_scene.py')
    if not os.path.exists(script_path):
        return None, "Worker script not found"
    payload = json.dumps({
        "image_path": image_path,
        "child_age": child_age,
        "language_age": language_age,
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


def _questions_path(wh_dir, scene_id, mode):
    """Path on disk for a scene's question set in the given mode."""
    if mode == "expressive":
        return os.path.join(wh_dir, f"{scene_id}_questions_expressive.json")
    return os.path.join(wh_dir, f"{scene_id}_questions.json")


def _generate_and_save_both_modes(image_path, child_age, wh_dir, scene_id, language_age=None):
    """Generate receptive + expressive question sets and persist both.

    ``language_age`` (when given) pitches question complexity at the child's
    developmental language age instead of their chronological age.

    Returns a dict describing what was produced and any per-mode errors so the
    caller can surface partial-success states to the therapist UI.
    """
    summary = {
        "scene_description": "",
        "receptive_count": 0,
        "expressive_count": 0,
        "errors": {},
    }
    for mode in ("receptive", "expressive"):
        result, error = _run_gemini_wh_analysis(image_path, child_age, mode, language_age=language_age)
        if result and "questions" in result:
            with open(_questions_path(wh_dir, scene_id, mode), "w") as f:
                json.dump(result, f, indent=2)
            count = len(result.get("questions", []))
            if mode == "receptive":
                summary["receptive_count"] = count
                summary["scene_description"] = result.get("scene_description", "") or summary["scene_description"]
            else:
                summary["expressive_count"] = count
                if not summary["scene_description"]:
                    summary["scene_description"] = result.get("scene_description", "")
        else:
            summary["errors"][mode] = error or "Unknown error"
    return summary


@app.route("/wh_picture_scene")
def wh_picture_scene_page():
    """WH Questions Picture Scene - Prepare (therapist uploads images)."""
    if 'username' not in session:
        return redirect(url_for('index'))
    username = session['username']
    user = user_manager.users.get(username)
    # Hold the robot's head still while the therapist frames a new scene.
    # Tracking resumes after the Capture Scene button is clicked.
    _pause_human_tracking_for_capture()
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

    # Generate both receptive and expressive question sets so the play page
    # can switch modes without a second Gemini round-trip.
    summary = _generate_and_save_both_modes(image_path, child_age, wh_dir, scene_id,
                                            language_age=_language_age_for(username, child_age))

    index = _load_scene_index(username)
    rel_path = os.path.relpath(image_path, USER_DATA_DIR)
    ready = summary["receptive_count"] > 0 or summary["expressive_count"] > 0
    entry = {
        "id": scene_id,
        "filename": file.filename,
        "image_path": image_path,
        "image_url": f"/images/{rel_path}",
        "scene_description": summary["scene_description"],
        "question_count": summary["receptive_count"],
        "expressive_count": summary["expressive_count"],
        "status": "ready" if ready else "error",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if summary["errors"]:
        entry["errors"] = summary["errors"]
    index.append(entry)
    _save_scene_index(username, index)

    if ready:
        response = {"success": True, "scene": entry}
        if summary["errors"]:
            response["warning"] = "; ".join(f"{m}: {e}" for m, e in summary["errors"].items())
        return jsonify(response)
    return jsonify({"success": True, "scene": entry, "warning": "; ".join(f"{m}: {e}" for m, e in summary["errors"].items()) or "Unknown error"})


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

    # Frame is safely on disk — resume head tracking now so the robot can follow
    # the child again while Gemini runs (the analysis can take several seconds).
    _resume_human_tracking_after_capture()

    # Get user profile for age-appropriate questions
    user = user_manager.users.get(username, {})
    child_age = user.get('age', 5)

    summary = _generate_and_save_both_modes(image_path, child_age, wh_dir, scene_id,
                                            language_age=_language_age_for(username, child_age))

    index = _load_scene_index(username)
    rel_path = os.path.relpath(image_path, USER_DATA_DIR)
    ts_label = time.strftime("%Y-%m-%d %H:%M:%S")
    ready = summary["receptive_count"] > 0 or summary["expressive_count"] > 0
    entry = {
        "id": scene_id,
        "filename": f"capture_{ts_label}.jpg",
        "image_path": image_path,
        "image_url": f"/images/{rel_path}",
        "scene_description": summary["scene_description"],
        "question_count": summary["receptive_count"],
        "expressive_count": summary["expressive_count"],
        "status": "ready" if ready else "error",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if summary["errors"]:
        entry["errors"] = summary["errors"]
    index.append(entry)
    _save_scene_index(username, index)

    if ready:
        response = {"success": True, "scene": entry}
        if summary["errors"]:
            response["warning"] = "; ".join(f"{m}: {e}" for m, e in summary["errors"].items())
        return jsonify(response)
    return jsonify({"success": True, "scene": entry, "warning": "; ".join(f"{m}: {e}" for m, e in summary["errors"].items()) or "Unknown error"})


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

    # Delete questions files (both modes)
    wh_dir = _user_wh_dir(username)
    for mode in ("receptive", "expressive"):
        q_path = _questions_path(wh_dir, scene_id, mode)
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

    wh_dir = _user_wh_dir(username)
    summary = _generate_and_save_both_modes(scene['image_path'], child_age, wh_dir, scene_id,
                                            language_age=_language_age_for(username, child_age))
    ready = summary["receptive_count"] > 0 or summary["expressive_count"] > 0

    if ready:
        scene['status'] = 'ready'
        scene['scene_description'] = summary['scene_description']
        scene['question_count'] = summary['receptive_count']
        scene['expressive_count'] = summary['expressive_count']
        scene.pop('error', None)
        if summary['errors']:
            scene['errors'] = summary['errors']
        else:
            scene.pop('errors', None)
        _save_scene_index(username, index)
        return jsonify({"success": True})
    else:
        scene['status'] = 'error'
        scene['errors'] = summary['errors']
        scene['error'] = "; ".join(f"{m}: {e}" for m, e in summary['errors'].items()) or 'Unknown error'
        _save_scene_index(username, index)
        return jsonify({"success": False, "error": scene['error']})


@app.route("/api/wh_scene/get_questions", methods=["POST"])
def api_wh_scene_get_questions():
    """Get the generated WH questions for a scene in the requested mode."""
    username = session.get('username')
    if not username:
        return jsonify({"success": False, "error": "Not logged in"}), 401
    data = request.get_json() or {}
    scene_id = data.get("scene_id")
    mode = (data.get("mode") or "receptive").lower()
    if mode not in ("receptive", "expressive"):
        mode = "receptive"
    if not scene_id:
        return jsonify({"success": False, "error": "Missing scene_id"}), 400

    wh_dir = _user_wh_dir(username)
    q_path = _questions_path(wh_dir, scene_id, mode)
    if not os.path.exists(q_path):
        msg = ("Expressive questions not generated for this scene yet. "
               "Ask the therapist to regenerate.") if mode == "expressive" else "Questions not found"
        return jsonify({"success": False, "error": msg}), 404

    with open(q_path, 'r') as f:
        result = json.load(f)

    return jsonify({
        "success": True,
        "mode": mode,
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