#!/usr/bin/env python3.9

import os
import argparse
import sys
import json
from google import genai
from google.genai import types


# Canonical relation key -> phrase used by the robot when speaking the
# instruction. Single phrase here keeps prompt grounding unambiguous; the
# question generator on the server side may swap synonyms in the spoken
# sentence (e.g. "beside" instead of "next to") but always passes the
# canonical key to this worker.
RELATION_PHRASE = {
    "on": "on",
    "off": "off",
    "next_to": "next to",
    "under": "under",
    "above": "above",
    "away_from": "away from",
}

RELATION_RULES = {
    "on": "on top of, touching each other, not next to, not under",
    "off": "not on, not touching each other",
    "next_to": "side by side, not under, not above",
    "under": "below, lower than, not next to, not above",
    "above": "above, higher than, not next to, not under",
    "away_from": "There must be a significant, visible empty gap separating the two objects (e.g., a distance wider than the reference object itself)."
}

def main():
    # Gemini Robotics-ER is the embodied-reasoning model (also used by the
    # object detector); it is the default backend for single-frame spatial
    # validation. Override with SPATIAL_ER_MODEL.
    model_id = os.getenv("SPATIAL_ER_MODEL", "gemini-robotics-er-1.6-preview")
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY or GOOGLE_API_KEY")

    p = argparse.ArgumentParser(description="Validate spatial direction setup")
    p.add_argument('--image', required=True, help='Path to image file')
    p.add_argument('--obj-a', required=True, dest='obj_a',
                   help='Moving object named first in the instruction')
    p.add_argument('--obj-b', required=True, dest='obj_b',
                   help='Reference object the relation is measured against')
    p.add_argument('--relation', required=True,
                   choices=list(RELATION_PHRASE.keys()),
                   help='Canonical spatial relation key')
    p.add_argument('--toy-list', default='',
                   help='Comma-separated valid toys (optional)')
    args = p.parse_args()

    with open(args.image, 'rb') as f:
        image_bytes = f.read()

    rel_phrase = RELATION_PHRASE[args.relation]

    active_rule = RELATION_RULES[args.relation]

    prompt = (
        "You are validating a therapeutic spatial-direction game for a QT robot interacting with young children. Accuracy is critical."
        f"The child was asked to arrange the scene so that the {args.obj_a} is "
        f"{rel_phrase} the {args.obj_b}.\n"
        "\n"
        "An image is taken from the front of the child (camera-facing view).\n"
        "Criteria:\n"
        f"1. Is the {args.obj_a} present in the scene?\n"
        f"2. Is the {args.obj_b} present in the scene?\n"
        f"3. Verification: Does the spatial arrangement meet the following rule for '{args.relation}'? ({active_rule})\n"
        "Instructions:\n"
        "Return ONLY a JSON object with no markdown fences.\n"
        "{\n"
        "  \"obj_a_found\": true|false,\n"
        "  \"obj_b_found\": true|false,\n"
        "  \"actual_relation\": \"on|off|next_to|under|above|away_from\",\n"
        "  \"correct\": true|false,\n"
        "  \"reason\": \"<short, child-friendly explanation>\"\n"
        "}\n"
        f"`actual_relation` is deprecated; just set it to '{args.relation}' regardless of whether the criteria is met.\n"
        "Set `correct` to true if and only if both objects are found and the verification is satisfied.\n"
        "Set `reason` to a short explanation for the child to understand your verdict.\n"
    )

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model_id,
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type='image/jpeg'),
            prompt,
        ],
        config=types.GenerateContentConfig(
            temperature=0.2,
            system_instruction=(
                "You judge a children's spatial-direction game from one photo. "
                "Return JSON only."),
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
            "obj_a_found": False,
            "obj_b_found": False,
            "actual_relation": "other",
            "correct": False,
            "reason": raw or "Could not parse vision response",
        }
    # Surface what was sent to / returned by Gemini so the UI can show the
    # therapist exactly how the verdict was reached.
    result["prompt"] = prompt
    result["raw_response"] = raw
    print(json.dumps(result))
    return 0


if __name__ == '__main__':
    sys.exit(main())