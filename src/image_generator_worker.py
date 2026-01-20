#!/usr/bin/env python3

import json
import os
import sys

from PIL import Image
from google import genai


def main() -> int:
    payload_raw = sys.stdin.read()
    if not payload_raw.strip():
        print("Expected JSON payload on stdin", file=sys.stderr)
        return 2

    try:
        payload = json.loads(payload_raw)
    except Exception as e:
        print(f"Invalid JSON payload: {e}", file=sys.stderr)
        return 2

    prompt = str(payload.get("prompt") or "").strip()
    output_path = str(payload.get("output_path") or "").strip()
    reference_image = payload.get("reference_image")

    if not prompt:
        print("Missing prompt", file=sys.stderr)
        return 2
    if not output_path:
        print("Missing output_path", file=sys.stderr)
        return 2

    try:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            print("Missing GOOGLE_API_KEY (or GEMINI_API_KEY)", file=sys.stderr)
            return 3
        client = genai.Client(api_key=api_key)

        contents = [prompt]
        if reference_image and os.path.exists(reference_image):
            # Provide the first generated image as a visual guide
            ref_img = Image.open(reference_image).convert("RGB")
            # The SDK accepts a PIL.Image directly in contents
            contents.append(ref_img)

        response = client.models.generate_content(
            model="gemini-2.5-flash-image",
            contents=contents,
        )

        saved = False
        # NOTE: response.parts shape depends on SDK; this matches what you requested.
        for part in response.parts:
            if getattr(part, "text", None) is not None:
                print(part.text)
            elif getattr(part, "inline_data", None) is not None:
                image = part.as_image()
                image.save(output_path)
                print(output_path)
                saved = True
                break

        if not saved:
            print("No image data returned", file=sys.stderr)
            return 4

        return 0

    except Exception as e:
        print(f"Worker failed: {e}", file=sys.stderr)
        return 5


if __name__ == "__main__":
    raise SystemExit(main())