import time
from pylips.speech import RobotFace

STRONG_EXPRESSIONS = {
    "surprise": {
        "AU1": 4.0,   # inner brow raise
        "AU2": 4.0,   # outer brow raise
        "AU5": 3.0,   # upper lid raise
        "AU15": 2.0,  # lip corner down
        "AU26": 2.5,  # jaw drop
    },
    "happy": {
        "AU1": 1.5,
        "AU2": 1.5,
        "AU7": 1.2,
        "AU12": 3.5,  # smile
    },
    "sad": {
        "AU1": 3.0,
        "AU4": 2.0,
        "AU15": 3.0,
        "AU17": 1.5,
    },
    "angry": {
        "AU4": 4.0,
        "AU7": 2.0,
        "AU23": 2.5,
        "AU24": 1.5,
    },
    "default_happy" : {
    'AU1': 0.5,
    'AU2': 1,
    'AU7': 0.5,
    'AU12': 1
},

"default_sad" :{
    'AU1': 1,
    'AU4': 0.5,
    'AU5': -1,
    'AU15': 1.5,

},

"default_surprise" : {
    'AU1': 1,
    'AU2': 1,
    'AU5': 1,
    'AU15': 0.5,
    'AU26': 1

},

"default_fear" : {
    'AU1': 1,
    'AU2': 1,
    'AU4': 1,
    'AU5': 1,
    'AU7': 1,
    'AU15': 1,
    'AU20': 1,
    'AU26': 1

},

"default_angry" : {
    'AU4': 3,
    'AU7': 1,
    'AU15': 1,
    'AU23': 1

},

"default_disgust" : {
    'AU9': 1,
    'AU15': 1,
    'AU17': 1,}
}

DEFAULT_SEQUENCE = [
    "default_happy",
    "default_sad",
    "default_surprise",
    "default_fear",
    "default_angry",
    "default_disgust",
]

def express_default(face: RobotFace, name: str, ms: int = 1000, hold_sec: float = 1.5):
    key = (name or "").strip().lower()
    if key not in STRONG_EXPRESSIONS:
        raise ValueError(f"Unknown expression '{name}'. Options: {list(STRONG_EXPRESSIONS.keys())}")
    face.express(STRONG_EXPRESSIONS[key], ms)
    time.sleep(hold_sec)

def reset_face(face: RobotFace, ms: int = 800):
    # zero common AUs back to neutral
    neutral = {
        "AU1": 0, "AU2": 0, "AU4": 0, "AU5": 0, "AU7": 0, "AU9": 0,
        "AU10": 0, "AU12": 0, "AU13": 0, "AU14": 0, "AU15": 0, "AU16": 0,
        "AU17": 0, "AU18": 0, "AU20": 0, "AU23": 0, "AU24": 0, "AU25": 0,
        "AU26": 0, "AU27": 0, "AU43": 0
    }
    face.express(neutral, ms)

if __name__ == "__main__":
    face = RobotFace(robot_name="default", server_ip="http://192.168.100.2:8000")

    time.sleep(1)

    for expr_name in DEFAULT_SEQUENCE:
        express_default(face, expr_name, ms=1200, hold_sec=1.6)
        reset_face(face, ms=700)
        time.sleep(0.3)