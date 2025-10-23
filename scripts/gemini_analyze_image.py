#!/usr/bin/env python3.9

import os
import argparse
import sys
from google import genai
from google.genai import types


def main():
    # Initialize the GenAI client and specify the model
    MODEL_ID = "gemini-robotics-er-1.5-preview"
    PROMPT = """
              Point to no more than 1 item a person is holding in the image. The label returned
              should be an identifying name for the object detected.
              The answer should follow the json format: [{"point": <point>,
              "label": <label1>}, ...]. The points are in [y, x] format
              normalized to 0-1000.
            """
    client = genai.Client(api_key="***REMOVED***")


    p = argparse.ArgumentParser(description="Analyze an image with Gemini and print result text")
    p.add_argument('--image', required=True, help='Path to image file (jpg/png)')
    args = p.parse_args()
    with open(args.image, 'rb') as f:
        image_bytes = f.read()

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
            temperature=0.5,
            thinking_config=types.ThinkingConfig(thinking_budget=0)
        )
    )
    print(image_response.text)
    # Print plain text to stdout for server to capture

    return 0


if __name__ == '__main__':
    sys.exit(main())




