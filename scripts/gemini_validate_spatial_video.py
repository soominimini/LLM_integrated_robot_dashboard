#!/usr/bin/env python3.9

import argparse
import json
import os
import sys
import time

from google import genai
from google.genai import types


RELATION_PHRASE = {
    "in": "in",
    "out": "out",
    "behind": "behind",
    "in_front_of": "in front of",
}

RELATION_RULES = {
    "in": "inside, contained by; partially hidden by walls or rim of",
    "out": "outside, not inside, not contained by",
    "behind": "further from camera than, may be partially hidden by",
    "in_front_of": "closer to camera than, may partially block",
}

def _state_name(file_obj):
    """Return the upload state as a string regardless of SDK enum shape."""
    state = getattr(file_obj, "state", None)
    return getattr(state, "name", None) or str(state) or ""


def main():
    # Gemini Robotics-ER by default (same embodied-reasoning model as the
    # still-frame validator and object detector). Override with
    # SPATIAL_ER_VIDEO_MODEL (e.g. set to gemini-2.5-flash to revert).
    model_id = os.getenv("SPATIAL_ER_VIDEO_MODEL", "gemini-robotics-er-1.6-preview")
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

    active_rule = RELATION_RULES[args.relation]

    prompt = (
        "You are validating a therapeutic spatial-direction game for a QT robot interacting with young children. Accuracy is critical."
        f"The child was asked to arrange the scene so that the {args.obj_a} is "
        f"{rel_phrase} the {args.obj_b}.\n"
        "\n"
        "A video is taken from the front of the child (camera-facing view).\n"
        "Criteria:\n"
        f"1. Is the {args.obj_a} present in the scene?\n"
        f"2. Is the {args.obj_b} present in the scene?\n"
        f"3. Verification: Does the spatial arrangement meet the following rule for '{args.relation}'? ({active_rule})\n"
        "Instructions:\n"
        "Return ONLY a JSON object with no markdown fences.\n"
        "{\n"
        "  \"obj_a_found\": true|false,\n"
        "  \"obj_b_found\": true|false,\n"
        "  \"actual_relation\": \"in|out|behind|in_front_of|other\",\n"
        "  \"correct\": true|false,\n"
        "  \"reason\": \"<short, child-friendly explanation>\"\n"
        "}\n"
        f"`actual_relation` is deprecated; just set it to '{args.relation}' regardless of whether the criteria is met.\n"
        "Set `correct` to true if and only if both objects are found and the verification is satisfied.\n"
        "Set `reason` to a short explanation for the child to understand your verdict.\n"
        "\n"
        "Tips:\n"
        "Use motion ACROSS frames to infer depth and containment more reliably than a single frame would allow.\n"
        "An object CLOSER to the camera shifts MORE in the image than a FARTHER object for the same scene motion.\n"
        "An object that consistently hides part of another is in front of it.\n"
        "If the moving object stays inside the rim of the reference object across the clip — partly hidden by the container's walls — it is 'in'.\n"
        "If the child visibly LIFTS the moving object OUT of the container during the clip, classify by the END state (last frame) of the video.\n"
        "If there is a clear visible gap between the two objects by the end of the clip, it is 'out'.\n"
    )

    response = client.models.generate_content(
        model=model_id,
        contents=[myfile, prompt],
        config=types.GenerateContentConfig(
            temperature=0.2,
            system_instruction=(
                "You judge a children's spatial-direction game from a short "
                "video. Return JSON only."),
        ),
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
    # Surface what was sent to / returned by Gemini so the UI can show the
    # therapist exactly how the verdict was reached.
    result["prompt"] = prompt
    result["raw_response"] = raw
    print(json.dumps(result))

    # Best-effort cleanup of the uploaded clip — failures are non-fatal.
    try:
        client.files.delete(name=myfile.name)
    except Exception:
        pass
    return 0


if __name__ == '__main__':
    sys.exit(main())