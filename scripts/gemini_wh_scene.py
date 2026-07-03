#!/usr/bin/env python3.9
"""
Gemini Vision worker: analyzes an uploaded scene image and generates
WH-questions with answers and visual-choice labels.

Called as a subprocess from the main server (Python 3.8).
Reads JSON config from stdin, prints JSON result to stdout.

Input JSON:
  { "image_path": "/path/to/image.jpg",
    "child_age": 5,
    "language_age": 5,   # optional; developmental age driving question complexity
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
    # Pitch question complexity at the child's developmental/language age when
    # provided (e.g. a 9-year-old with an MLU-6-8 target -> language_age 5),
    # otherwise fall back to chronological age. The age below is used only as a
    # complexity cue (never as identity), so a single effective age suffices.
    language_age = cfg.get("language_age")
    if language_age is not None:
        child_age = language_age
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

Analyze the illustration and generate WH-questions for a child aged {child_age}.

Generate exactly 5 questions: one each for who, what, when, where, and why.

Developmental guidance:
- For ages 3: use very short, concrete questions with directly visible answers.
- For ages 4: use simple WH-questions with obvious visual choices.
- For ages 5: simple inference questions are allowed when supported by the picture.
- For receptive mode, keep questions simple and make the choices visually distinguishable.

Grounding rules:
- Do not invent details that are not visible or strongly supported by the scene.
- Do not assume gender unless clearly shown. Use “the child,” “the person,” “the animal,” or “the character” when gender is unclear.
- For “when” questions, use only visually supported time/context cues such as nighttime, morning, winter, bedtime, mealtime, or rainy day.
- For “why” questions, use only simple everyday reasoning supported by visible evidence.
- If a WH type is difficult for the image, generate the safest simple question possible.
- Do not refer to the card, page, photo, or illustration in the questions.

For each question, provide:
1. wh_type
2. question
3. answer: short, 1–5 words
4. visual_choices: exactly 4 choices, each 1–4 words
5. evidence_type: "visible" or "simple_inference"
6. evidence_hint: tell the child where to look in the picture

Choice rules:
- The correct answer must appear exactly once in visual_choices.
- Include one correct answer and three plausible distractors.
- Distractors should be from the same general category when possible.
- Avoid silly, random, or obviously impossible distractors.
- Keep all choices child-friendly and concrete.

Return ONLY valid JSON. Do not include markdown or explanations.

Use this exact format:
{
  "scene_description": "Brief description of the scene",
  "questions": [
    {
      "wh_type": "who",
      "question": "...",
      "answer": "...",
      "visual_choices": ["...", "...", "...", "..."],
      "evidence_type": "visible",
      "evidence_hint": "..."
    },
    {
      "wh_type": "what",
      "question": "...",
      "answer": "...",
      "visual_choices": ["...", "...", "...", "..."],
      "evidence_type": "visible",
      "evidence_hint": "..."
    },
    {
      "wh_type": "when",
      "question": "...",
      "answer": "...",
      "visual_choices": ["...", "...", "...", "..."],
      "evidence_type": "visible",
      "evidence_hint": "..."
    },
    {
      "wh_type": "where",
      "question": "...",
      "answer": "...",
      "visual_choices": ["...", "...", "...", "..."],
      "evidence_type": "visible",
      "evidence_hint": "..."
    },
    {
      "wh_type": "why",
      "question": "...",
      "answer": "...",
      "visual_choices": ["...", "...", "...", "..."],
      "evidence_type": "simple_inference",
      "evidence_hint": "..."
    }
  ]
}
"""

    # Emit the prompt on stderr (stdout must stay pure JSON) so the server can
    # trace it into the daily trace log like every other LLM call. Printed
    # before the API call so failed calls still leave the prompt in the trace.
    print(
        f"[wh-scene-prompt]\nmodel={model_id} difficulty={difficulty} "
        f"effective_age={child_age}\n{prompt}\n[/wh-scene-prompt]",
        file=sys.stderr,
    )

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
