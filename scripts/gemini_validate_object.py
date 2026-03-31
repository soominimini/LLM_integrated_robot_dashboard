#!/usr/bin/env python3.9
"""Validate whether an object shown to the camera matches given criteria.

Used by the scene detection game for ages 4-6 where questions describe
properties (e.g. "a red fruit") rather than naming a specific object.
Gemini vision identifies the object AND checks it against the criteria.
"""

import os
import argparse
import sys
import json
from google import genai
from google.genai import types


def main():
    MODEL_ID = os.getenv("GEMINI_VISION_MODEL", "gemini-2.5-flash")
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY or GOOGLE_API_KEY")

    p = argparse.ArgumentParser(description="Validate object against criteria")
    p.add_argument('--image', required=True, help='Path to image file')
    p.add_argument('--criteria', required=True,
                   help='Criteria description, e.g. "a red fruit"')
    p.add_argument('--toy-list', required=False, default='',
                   help='Comma-separated list of valid toys')
    args = p.parse_args()

    with open(args.image, 'rb') as f:
        image_bytes = f.read()

    toy_clause = ""
    if args.toy_list:
        toy_clause = (
            f"The valid objects in this game are: {args.toy_list}. "
            "Only consider these objects when identifying what is shown.\n"
        )

    prompt = (
        "A child is showing an object to the camera as part of a game.\n"
        f"{toy_clause}"
        "Your task:\n"
        "1. Identify the object the child is holding or showing.\n"
        "2. Determine its visual properties: color (the dominant RGB color you see), "
        "category (fruit, vegetable, animal, toy, etc.), shape, size.\n"
        f"3. Check if the object matches this criteria: \"{args.criteria}\"\n"
        "\n"
        "Return ONLY a JSON object with these fields:\n"
        "{\n"
        "  \"object\": \"<name of identified object>\",\n"
        "  \"color\": \"<dominant color>\",\n"
        "  \"category\": \"<category>\",\n"
        "  \"matches_criteria\": true or false,\n"
        "  \"reason\": \"<brief explanation of why it matches or not>\"\n"
        "}\n"
        "If you cannot see any object, set object to null and matches_criteria to false."
    )

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=MODEL_ID,
        contents=[
            types.Part.from_bytes(
                data=image_bytes,
                mime_type='image/jpeg',
            ),
            prompt
        ],
        config=types.GenerateContentConfig(
            temperature=0.3,
        )
    )

    raw = (response.text or '').strip()
    try:
        if raw.startswith('```'):
            raw = raw.strip('`').strip()
            if raw.startswith('json'):
                raw = raw[4:].strip()
        result = json.loads(raw)
        print(json.dumps(result))
    except Exception:
        print(json.dumps({"object": None, "matches_criteria": False,
                          "reason": raw}))

    return 0


if __name__ == '__main__':
    sys.exit(main())
