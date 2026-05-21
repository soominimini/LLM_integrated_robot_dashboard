#!/usr/bin/env python3.9
"""Validate a spatial-direction setup from a short video clip.

Used for the depth relations (behind / in_front_of) where a single 2D
frame cannot reliably disambiguate the configuration. The child holds
the objects (or slowly tilts the camera) during a ~3-second capture so
the model can use parallax across frames to infer relative depth.

The video is uploaded via the Gemini Files API (mirroring the pattern in
``robotics-samples/Getting Started/gemini_robotics_er.ipynb`` cell 62)
and the file handle is passed as a content part to ``generate_content``.

Input (args):
    --video      path to the captured MP4
    --obj-a      moving object the robot named first (e.g. "banana")
    --obj-b      reference object (e.g. "blue block")
    --relation   one of: next_to, above, under, behind, in_front_of
    --toy-list   optional comma-separated list of valid toys

Output (stdout, single JSON line; same schema as the still worker):
    {
      "obj_a_found":     true | false,
      "obj_b_found":     true | false,
      "actual_relation": "next_to" | "above" | "under" | "behind"
                       | "in_front_of" | "other",
      "correct":         true | false,
      "reason":          "<short child-friendly explanation>"
    }
"""

import argparse
import json
import os
import sys
import time

from google import genai
from google.genai import types


RELATION_PHRASE = {
    "next_to": "next to",
    "above": "on top of",
    "under": "under",
    "behind": "behind",
    "in_front_of": "in front of",
    "in": "in",
    "out": "out of",
}


def _state_name(file_obj):
    """Return the upload state as a string regardless of SDK enum shape."""
    state = getattr(file_obj, "state", None)
    return getattr(state, "name", None) or str(state) or ""


def main():
    model_id = os.getenv("GEMINI_VISION_MODEL", "gemini-2.5-flash")
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY or GOOGLE_API_KEY")

    p = argparse.ArgumentParser(description="Validate spatial direction from a video clip")
    p.add_argument('--video', required=True, help='Path to MP4 file')
    p.add_argument('--obj-a', required=True, dest='obj_a')
    p.add_argument('--obj-b', required=True, dest='obj_b')
    p.add_argument('--relation', required=True, choices=list(RELATION_PHRASE.keys()))
    p.add_argument('--toy-list', default='')
    args = p.parse_args()

    if not os.path.exists(args.video):
        print(json.dumps({
            "obj_a_found": False, "obj_b_found": False,
            "actual_relation": "other", "correct": False,
            "reason": f"Video not found: {args.video}",
        }))
        return 1

    client = genai.Client(api_key=api_key)

    # Upload the clip and wait for it to leave PROCESSING before we can
    # use it as a content part. Mirrors cell 62 of the robotics-samples
    # notebook. We cap the wait at 60s so a stuck upload doesn't hang
    # the whole answer flow.
    myfile = client.files.upload(file=args.video)
    deadline = time.time() + 60.0
    while _state_name(myfile) == "PROCESSING" and time.time() < deadline:
        time.sleep(1)
        myfile = client.files.get(name=myfile.name)
    if _state_name(myfile) != "ACTIVE":
        print(json.dumps({
            "obj_a_found": False, "obj_b_found": False,
            "actual_relation": "other", "correct": False,
            "reason": f"Video file state: {_state_name(myfile)}",
        }))
        try:
            client.files.delete(name=myfile.name)
        except Exception:
            pass
        return 1

    rel_phrase = RELATION_PHRASE[args.relation]
    toy_clause = ""
    if args.toy_list:
        toy_clause = (
            f"The valid game objects are: {args.toy_list}. "
            "Only identify objects from this list.\n"
        )

    prompt = (
        "You are judging a children's spatial-direction game.\n"
        f"{toy_clause}"
        f"The child was asked to arrange the scene so that the {args.obj_a} is "
        f"{rel_phrase} the {args.obj_b}.\n"
        "\n"
        "You are watching a short video (camera-facing view) of the child's\n"
        "setup. Use motion ACROSS frames to infer depth and containment more\n"
        "reliably than a single still frame would allow.\n"
        "\n"
        f"1. Is the {args.obj_a} present in the scene?\n"
        f"2. Is the {args.obj_b} present in the scene?\n"
        f"3. What is the actual spatial relation of the {args.obj_a} TO the "
        f"{args.obj_b}? Pick ONE:\n"
        "   - next_to       (side by side, similar height)\n"
        "   - above         (higher than / on top of)\n"
        "   - under         (lower than / underneath)\n"
        "   - behind        (further from camera; partially hidden by the other)\n"
        "   - in_front_of   (closer to camera; may partially block the other)\n"
        "   - in            (inside / contained by the other; partially hidden by its walls or rim)\n"
        "   - out           (outside / not contained by the other; fully visible and separate)\n"
        "   - other         (can't tell confidently)\n"
        f"4. Does that match the requested relation '{args.relation}'?\n"
        "\n"
        "Cues to use across frames:\n"
        "- Parallax: an object CLOSER to the camera shifts MORE in the image\n"
        "  than a FARTHER object for the same scene motion. (depth)\n"
        "- Occlusion: an object that consistently hides part of another is\n"
        "  in front of it. (depth)\n"
        "- Containment ('in'): if the moving object stays inside the rim of\n"
        "  the reference object across the clip — partly hidden by the\n"
        "  container's walls — it is 'in'. If the child visibly LIFTS the\n"
        "  moving object OUT of the container during the clip, classify by\n"
        "  the END state of the video.\n"
        "- Separation ('out'): if there is a clear visible gap between the\n"
        "  two objects throughout the clip (or by the end of it), it is 'out'.\n"
        "If you cannot tell confidently from any of these cues, return 'other'.\n"
        "\n"
        "Return ONLY a JSON object with no markdown fences:\n"
        "{\n"
        "  \"obj_a_found\": true|false,\n"
        "  \"obj_b_found\": true|false,\n"
        "  \"actual_relation\": \"next_to|above|under|behind|in_front_of|in|out|other\",\n"
        "  \"correct\": true|false,\n"
        "  \"reason\": \"<short, child-friendly explanation>\"\n"
        "}\n"
        "If either object is missing, set correct=false."
    )

    response = client.models.generate_content(
        model=model_id,
        contents=[myfile, prompt],
        config=types.GenerateContentConfig(temperature=0.2),
    )

    raw = (response.text or '').strip()
    if raw.startswith('```'):
        raw = raw.strip('`').strip()
        if raw.startswith('json'):
            raw = raw[4:].strip()
    try:
        result = json.loads(raw)
    except Exception:
        result = {
            "obj_a_found": False, "obj_b_found": False,
            "actual_relation": "other", "correct": False,
            "reason": raw or "Could not parse vision response",
        }
    print(json.dumps(result))

    # Best-effort cleanup of the uploaded clip — failures are non-fatal.
    try:
        client.files.delete(name=myfile.name)
    except Exception:
        pass
    return 0


if __name__ == '__main__':
    sys.exit(main())
