#!/usr/bin/env python3.9

import os
import argparse
import sys
from google import genai
from google.genai import types


def main():
    MODEL_ID = "gemini-robotics-er-1.6-preview"

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GOOGLE_API_KEY (or GEMINI_API_KEY)")
    client = genai.Client(api_key=api_key)

    p = argparse.ArgumentParser(description="Analyze an image with Gemini and print result text")
    p.add_argument('--image', required=True, help='Path to image file (jpg/png)')
    p.add_argument('--toy-list', default='', help='Comma-separated valid toys (optional)')
    args = p.parse_args()

    with open(args.image, 'rb') as f:
        image_bytes = f.read()

    toy_clause = ""
    if args.toy_list:
        toy_clause = (
            f"The valid game objects are: {args.toy_list}. "
            "You MUST set the 'label' to exactly one of the items from this list. "
        )

    # 3. Define the prompt dynamically
    PROMPT = (
        "Point to no more than 1 item a person is holding in the image.\n"
        f"{toy_clause}\n"
        "Return the object's identifying name, its dominant color, and its shape.\n"
        "The answer should follow the json format:\n"
        "[{\"point\": <point>, \"label\": <label>, \"color\": <color>, \"shape\": <shape>}].\n"
        "The points are in [y, x] format normalized to 0-1000.\n"
        "Tips:\n"
        "Do not use generic terms like 'stress ball', 'ball', 'toy', or 'object'.\n"
        "Be lenient with the dominant color as the lighting may cause the color to look different (i.e. purple may look like dark blue).\n"
        "If the child is holding more than 1 item, choose the item that most closely reflects the desired object.\n"
    )

    # Emit the exact prompt on stderr
    sys.stderr.write("<<<DETECTION_PROMPT_START>>>\n" + PROMPT + "\n<<<DETECTION_PROMPT_END>>>\n")
    sys.stderr.flush()

    image_response = client.models.generate_content(
        model=MODEL_ID,
        contents=[
            types.Part.from_bytes(
                data=image_bytes,
                mime_type='image/jpeg',
            ),
            PROMPT
        ],
        config=types.GenerateContentConfig(
            temperature=0.2,
            thinking_config=types.ThinkingConfig(thinking_budget=0)
        )
    )
    print(image_response.text)
    # Print plain text to stdout for server to capture

    return 0


if __name__ == '__main__':
    sys.exit(main())