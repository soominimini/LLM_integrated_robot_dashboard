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

Receptive output JSON:
  { "scene_description": "A boy sleeping in bed...",
    "questions": [
      { "wh_type": "who", "question": "Who is in the picture?",
        "answer": "a boy",
        "visual_choices": ["a boy", "a girl", "a man", "a woman"],
        "evidence_hint": "Look at the person in the bed" },
      ...
    ] }

Expressive output JSON (open-ended imagination questions; any answer is acceptable):
  { "scene_description": "A boy playing soccer in a park...",
    "questions": [
      { "wh_type": "what", "question": "What would he do after playing soccer?",
        "answer": "", "visual_choices": [],
        "evidence_hint": "Imagine what comes next." },
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

    card_framing = (
        "IMPORTANT: The photo may show a printed card, page, or picture being held "
        "by someone or placed on a surface. Focus ONLY on the illustration or scene "
        "depicted ON the card/page itself. Ignore everything outside the card — "
        "hands holding it, the person, the table, background, etc. Treat the "
        "illustration on the card as the entire scene for your analysis."
    )

    if difficulty == "expressive":
        prompt = f"""You are a pediatric speech-language pathologist creating therapy materials.

{card_framing}

Analyze this image and generate 5 OPEN-ENDED IMAGINATION questions for a child aged {child_age}.
These questions invite the child to imagine, predict, reflect, or share personal experience.
THEY HAVE NO SINGLE CORRECT ANSWER — any answer the child gives is acceptable.

Cover these 5 different kinds of imagination prompts (one question each, in this order):
1. FUTURE: what might happen next or after the scene
2. PAST: what might have happened just before the scene
3. PERSONAL: connect the scene to the child's own life or preferences (e.g., "Do you like...?", "Have you ever...?")
4. ALTERNATIVE: imagine a different choice, place, or outcome (e.g., "What else could they do?", "Where else could they go?")
5. FEELING: how a character might feel, or how the child would feel in the scene

Each question must:
- Be simple and warm, age-appropriate for {child_age} years old
- Be answerable in 1–2 spoken sentences by a child
- NOT have a right or wrong answer

Use `wh_type` from {{"what", "when", "why"}} only — pick whichever best fits the question
(e.g., "what" for future/past/alternative/personal, "why" for feeling, "when" works for past/future timing).

Leave `answer` as an empty string and `visual_choices` as an empty array — they are not used in expressive mode.
`evidence_hint` should be a short imagination prompt (e.g., "Imagine what comes next.", "Think about how you would feel.").

Return ONLY valid JSON in this exact format:
{{
  "scene_description": "Brief description of the scene",
  "questions": [
    {{
      "wh_type": "what",
      "question": "What do you think he will do after playing soccer?",
      "answer": "",
      "visual_choices": [],
      "evidence_hint": "Imagine what comes next."
    }},
    {{
      "wh_type": "what",
      "question": "What was he doing right before this picture?",
      "answer": "",
      "visual_choices": [],
      "evidence_hint": "Imagine what happened just before."
    }},
    {{
      "wh_type": "what",
      "question": "Do you like playing soccer too?",
      "answer": "",
      "visual_choices": [],
      "evidence_hint": "Think about your own experience."
    }},
    {{
      "wh_type": "what",
      "question": "What else could he play instead of soccer?",
      "answer": "",
      "visual_choices": [],
      "evidence_hint": "Imagine a different game."
    }},
    {{
      "wh_type": "why",
      "question": "How do you think he feels while playing?",
      "answer": "",
      "visual_choices": [],
      "evidence_hint": "Look at his face and body, and imagine the feeling."
    }}
  ]
}}
"""
    else:
        prompt = f"""You are a pediatric speech-language pathologist creating therapy materials.

{card_framing}

Analyze this image and generate WH-questions for a child aged {child_age}.

For EACH of the 5 WH-question types (who, what, when, where, why), generate:
1. A simple, clear question about the scene
2. The correct answer (short, 1-5 words)
3. Four visual choices (one correct + three plausible distractors), each 1-4 words
4. An evidence hint telling the child where to look in the picture

For receptive mode: make questions simple with obvious visual choices.

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
