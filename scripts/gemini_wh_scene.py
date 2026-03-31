#!/usr/bin/env python3.9
"""
Gemini Vision worker: analyzes an uploaded scene image and generates
WH-questions with answers and visual-choice labels.

Called as a subprocess from the main server (Python 3.8).
Reads JSON config from stdin, prints JSON result to stdout.

Input JSON:
  { "image_path": "/path/to/image.jpg",
    "child_age": 5,
    "difficulty": "receptive" | "expressive" }

Output JSON:
  { "scene_description": "A boy sleeping in bed...",
    "questions": [
      { "wh_type": "who", "question": "Who is in the picture?",
        "answer": "a boy",
        "visual_choices": ["a boy", "a girl", "a man", "a woman"],
        "evidence_hint": "Look at the person in the bed" },
      ...
    ] }
"""

import os
import sys
import json

from google import genai
from google.genai import types


def main():
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        print(json.dumps({"error": "Missing GOOGLE_API_KEY or GEMINI_API_KEY"}))
        return 1

    raw = sys.stdin.read().strip()
    if not raw:
        print(json.dumps({"error": "No input provided"}))
        return 1

    cfg = json.loads(raw)
    image_path = cfg.get("image_path")
    child_age = cfg.get("child_age", 5)
    difficulty = cfg.get("difficulty", "receptive")

    if not image_path or not os.path.exists(image_path):
        print(json.dumps({"error": f"Image not found: {image_path}"}))
        return 1

    with open(image_path, "rb") as f:
        image_bytes = f.read()

    # Determine mime type
    ext = os.path.splitext(image_path)[1].lower()
    mime_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
                ".gif": "image/gif", ".webp": "image/webp", ".bmp": "image/bmp"}
    mime_type = mime_map.get(ext, "image/jpeg")

    model_id = os.getenv("GEMINI_VISION_MODEL", "gemini-2.5-flash")

    prompt = f"""You are a pediatric speech-language pathologist creating therapy materials.

IMPORTANT: The photo may show a printed card, page, or picture being held by someone or placed on a surface.
Focus ONLY on the illustration or scene depicted ON the card/page itself.
Ignore everything outside the card — hands holding it, the person, the table, background, etc.
Treat the illustration on the card as the entire scene for your analysis.

Analyze this image and generate WH-questions for a child aged {child_age}.

For EACH of the 5 WH-question types (who, what, when, where, why), generate:
1. A simple, clear question about the scene
2. The correct answer (short, 1-5 words)
3. Four visual choices (one correct + three plausible distractors), each 1-4 words
4. An evidence hint telling the child where to look in the picture

{"For receptive mode: make questions simple with obvious visual choices." if difficulty == "receptive" else "For expressive mode: make questions slightly more challenging, requiring inference."}

Return ONLY valid JSON in this exact format:
{{
  "scene_description": "Brief description of the scene",
  "questions": [
    {{
      "wh_type": "who",
      "question": "Who is in the picture?",
      "answer": "a boy",
      "visual_choices": ["a boy", "a girl", "a man", "a dog"],
      "evidence_hint": "Look at the person in the picture"
    }},
    {{
      "wh_type": "what",
      "question": "What is he doing?",
      "answer": "sleeping",
      "visual_choices": ["sleeping", "eating", "running", "reading"],
      "evidence_hint": "Look at what the person is doing"
    }},
    {{
      "wh_type": "when",
      "question": "When is it?",
      "answer": "nighttime",
      "visual_choices": ["nighttime", "morning", "afternoon", "lunchtime"],
      "evidence_hint": "Look at the light and setting"
    }},
    {{
      "wh_type": "where",
      "question": "Where is he?",
      "answer": "in bed",
      "visual_choices": ["in bed", "at school", "in the park", "at the store"],
      "evidence_hint": "Look at the place around the person"
    }},
    {{
      "wh_type": "why",
      "question": "Why is he sleeping?",
      "answer": "because he is tired",
      "visual_choices": ["because he is tired", "because he is hungry", "because he is sad", "because it is raining"],
      "evidence_hint": "Think about why someone would do this"
    }}
  ]
}}
"""

    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model=model_id,
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
            prompt
        ],
        config=types.GenerateContentConfig(
            temperature=0.4,
        )
    )

    text = response.text.strip()
    # Strip markdown fences if present
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:].strip()

    # Parse and validate
    result = json.loads(text)
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
