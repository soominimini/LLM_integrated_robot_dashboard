#!/usr/bin/env python3

import os
from flask import Flask, render_template, request, jsonify, session, redirect, url_for, Response, send_from_directory, send_file
from user_management import UserManager
from story_generator import StoryGenerator
from tts_helper import TTSHelper
from image_generator import ImageGenerator
from flask_cors import CORS
import json
import re
import time
import random
from typing import Optional
import difflib
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
    from ultralytics import YOLO
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

# # IdleAttention singleton and runner
# _idle_attention = None
# _idle_attn_lock = Lock()
# _idle_thread = None
# _idle_stop = ThreadEvent()

# def _ensure_idle_attention():
    # global _idle_attention
    # if not IDLE_ATTENTION_AVAILABLE:
    #     return None
    # with _idle_attn_lock:
    #     if _idle_attention is None:
    #         try:
    #             tracker = _ensure_human_tracker()
    #             attn = IdleAttention()
    #             attn.setup(attention_time=2, human_tracker=tracker)
    #             _idle_attention = attn
    #         except Exception as e:
    #             print(f"IdleAttention init failed: {e}")
    #             return None
    # return _idle_attention

# def _idle_loop():
#     attn = _idle_attention
#     if not attn:
#         return
#     while not _idle_stop.is_set():
#         try:
#             attn.process()
#         except Exception as e:
#             print(f"IdleAttention process error: {e}")
#             time.sleep(0.1)

# def _start_idle_attention():
#     global _idle_thread
#     attn = _ensure_idle_attention()
#     if not attn:
#         return
#     try:
#         attn.start()
#     except Exception:
#         pass
#     if not _idle_thread or not _idle_thread.is_alive():
#         _idle_stop.clear()
#         _idle_thread = Thread(target=_idle_loop, daemon=True)
#         _idle_thread.start()

# def _stop_idle_attention():
#     global _idle_thread
#     try:
#         _idle_stop.set()
#         attn = _idle_attention
#         if attn:
#             attn.stop()
#         if _idle_thread and _idle_thread.is_alive():
#             _idle_thread.join(timeout=0.5)
#     except Exception:
#         pass

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

        # Load YOLO model
        self.model = YOLO("yolov8n.pt")  # Load a lightweight pretrained YOLOv8 model

        # Start processing thread
        self.processing_thread = Thread(target=self.process_images, daemon=True)
        self.processing_thread.start()

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
    # Fetch latest queued image
    frame = _ros_cam.get_latest_image()
    return frame

# Camera helper: try env overrides, common indices, and optional GStreamer
def _open_camera():
    if cv2 is None:
        return None, 'OpenCV not available'
    # 1) explicit device from env (index or path)
    dev = os.environ.get('CAMERA_DEVICE')
    if dev:
        try:
            idx = int(dev)
            cap = cv2.VideoCapture(idx)
            if cap and cap.isOpened():
                return cap, None
        except Exception:
            # try as device path
            cap = cv2.VideoCapture(dev)
            if cap and cap.isOpened():
                return cap, None
    # 2) try common indices
    for idx in [0, 1, 2, 3]:
        try:
            cap = cv2.VideoCapture(idx)
            if cap and cap.isOpened():
                return cap, None
            if cap: cap.release()
        except Exception:
            pass
    # 3) try GStreamer pipeline from env
    gst = os.environ.get('CAMERA_GSTREAMER')
    if gst:
        try:
            cap = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)
            if cap and cap.isOpened():
                return cap, None
        except Exception:
            pass
    return None, 'No camera available (tried indices 0-3, CAMERA_DEVICE, CAMERA_GSTREAMER)'

app = Flask(__name__, template_folder="../templates")
app.secret_key = os.urandom(24)
CORS(app)

user_manager = UserManager()
story_generator = StoryGenerator()
tts_helper = TTSHelper()
image_generator = ImageGenerator()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
USER_DATA_DIR = os.path.join(BASE_DIR, 'user_data')

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
    try:
        from riva_speech_recognition import RivaSpeechRecognition
        RivaSpeechRecognition.set_audio_enabled(False)
    except Exception:
        pass
    try:
        return say_callable()
    finally:
        try:
            from riva_speech_recognition import RivaSpeechRecognition
            RivaSpeechRecognition.set_audio_enabled(True)
        except Exception:
            pass
        # try:
        #     _stop_idle_attention()
        # except Exception:
        #     pass

# Lightweight LLM-based ASR intent correction
_intent_llm = None
_intent_llm_lock = Lock()

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
    child_name = data.get("child_name", username)  # Use username as default child name
    age = data.get("age", user.get("age", 4))  # Use user's age as default
    custom_prompt = data.get("custom_prompt")
    
    try:
        result = story_generator.generate_story(
            child_name=child_name,
            age=age,
            custom_prompt=custom_prompt
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
    child_name = data.get("child_name", username)  # Use username as default child name
    age = data.get("age", user.get("age", 4))  # Use user's age as default
    custom_prompt = data.get("custom_prompt")
    
    def generate():
        try:
            for chunk in story_generator.generate_story_stream(
                child_name=child_name,
                age=age,
                custom_prompt=custom_prompt
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
            # Split story into sentences
            #sentences = re.split(r'(?<=[.!?])\s+', story.strip())
            sentences = [s.strip() for s in re.split(r'\n\s*\n', story) if s.strip()]
            #sentences = [s for s in sentences if s.strip()]
            
            # Create user-specific image directory
            user_images_dir = os.path.join(USER_DATA_DIR, username, 'story_images', fname.replace('.json', ''))
            os.makedirs(user_images_dir, exist_ok=True)
            
            # Generate images for each sentence
            image_paths = []
            for i, sentence in enumerate(sentences):
                image_path = image_generator.generate_story_scene_image(
                    sentence, 
                    story_context=f"Story about {metadata.get('child_name', 'a child')}",
                    output_dir=user_images_dir,
                    filename_prefix=f"story_scene_{i:03d}"
                    #if !image_path.empty():
                        #prev_img = image_path[-1]
                    #else:
                        #prev_img = None
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
        prefer_mode = os.environ.get('CAMERA_MODE', '').lower().strip()  # 'ros_only' | 'v4l_only' | ''
        frame = None if prefer_mode == 'v4l_only' else _get_ros_frame()
        if frame is None and prefer_mode != 'ros_only':
            # Fallback to V4L/GStreamer
            cap, err = _open_camera()
            if not cap:
                return jsonify({'success': False, 'error': err}), 500
            ok, frame = cap.read()
            cap.release()
            if not ok or frame is None:
                return jsonify({'success': False, 'error': 'Camera read failed'}), 500
        if not ok or frame is None:
            return jsonify({'success': False, 'error': 'Camera read failed'}), 500
        # Encode JPEG
        ok, buf = cv2.imencode('.jpg', frame)
        if not ok:
            return jsonify({'success': False, 'error': 'JPEG encode failed'}), 500
        from flask import make_response
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
        prefer_mode = os.environ.get('CAMERA_MODE', '').lower().strip()
        frame = None if prefer_mode == 'v4l_only' else _get_ros_frame()
        if frame is None and prefer_mode != 'ros_only':
            cap, err = _open_camera()
            if not cap:
                return jsonify({'success': False, 'error': err}), 500
            ok, frame = cap.read()
            cap.release()
            if not ok or frame is None:
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
        return jsonify({'success': True, 'image_path': f"/images/{rel}"})
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
                                    from riva_speech_recognition import RivaSpeechRecognition
                                    while not _activity_stop_event.is_set():
                                        try:
                                            # avoid overlap with TTS
                                            guard_start = time.time()
                                            while getattr(tts_helper, 'is_speaking', lambda: False)() and time.time() - guard_start < 10:
                                                if _activity_stop_event.is_set():
                                                    return
                                                time.sleep(0.05)
                                            asr = RivaSpeechRecognition(language='en-US', detection_timeout=5)
                                            text, _lang = asr.recognize_once()
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
                        from riva_speech_recognition import RivaSpeechRecognition
                        while time.time() - start_wait < max_wait_seconds:
                            guard_start = time.time()
                            while getattr(tts_helper, 'is_speaking', lambda: False)() and time.time() - guard_start < 10:
                                time.sleep(0.05)
                            asr = RivaSpeechRecognition(language='en-US', detection_timeout=5)
                            text, _lang = asr.recognize_once()
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
                                    from riva_speech_recognition import RivaSpeechRecognition
                                    guard_start = time.time()
                                    while getattr(tts_helper, 'is_speaking', lambda: False)() and time.time() - guard_start < 10:
                                        time.sleep(0.05)
                                    asr = RivaSpeechRecognition(language='en-US', detection_timeout=5)
                                    text, _lang = asr.recognize_once()
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
    """Get image for a specific sentence"""
    username = session.get('username')
    data = request.get_json() or {}
    filename = data.get('filename', '')
    sentence_index = data.get('sentence_index', 0)
    
    print(f"Getting image for user: {username}, filename: {filename}, sentence_index: {sentence_index}")
    
    if not username or not filename:
        return jsonify({'success': False, 'error': 'Missing username or filename'})
    
    try:
        # Check for existing images
        user_images_dir = os.path.join(USER_DATA_DIR, username, 'story_images', filename.replace('.json', ''))
        print(f"Looking for images in: {user_images_dir}")
        
        if not os.path.exists(user_images_dir):
            print(f"Images directory does not exist: {user_images_dir}")
            return jsonify({'success': False, 'error': 'No images directory found'})
        
        print(f"Images directory exists. Files found: {os.listdir(user_images_dir)}")
        
        # Look for existing image files for this sentence
        # Try both patterns: with sentence index and without
        pattern_with_index = f'story_scene_{sentence_index:03d}_'
        pattern_without_index = 'story_scene_'
        
        existing_images = [f for f in os.listdir(user_images_dir) if f.startswith(pattern_with_index)]
        print(f"Looking for pattern '{pattern_with_index}', found: {existing_images}")
        
        # If no images found with index, try without index (for older images)
        if not existing_images:
            all_story_images = [f for f in os.listdir(user_images_dir) if f.startswith(pattern_without_index) and f.endswith('.png')]
            print(f"Looking for pattern '{pattern_without_index}', found: {all_story_images}")
            # Sort by creation time and take the sentence_index-th image
            all_story_images.sort()
            if sentence_index < len(all_story_images):
                existing_images = [all_story_images[sentence_index]]
                print(f"Using image {sentence_index} from sorted list: {existing_images}")
        
        if existing_images:
            # Use the first existing image
            image_path = f"/images/{username}/story_images/{os.path.basename(user_images_dir)}/{existing_images[0]}"
            print(f"Returning image path: {image_path}")
            return jsonify({
                'success': True, 
                'image_path': image_path
            })
        else:
            print(f"No image found for sentence {sentence_index}")
            return jsonify({'success': False, 'error': 'No image found for this sentence'})
            
    except Exception as e:
        print(f"Error getting image: {str(e)}")
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

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True) 