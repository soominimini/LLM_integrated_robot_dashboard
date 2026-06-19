# Activity 6 — WH Picture Scene

- **Entry points:** pages `/wh_picture_scene` (therapist uploads/prepares), `/wh_picture_play` (child plays); APIs `/api/wh_scene/upload|capture|list|delete|regenerate|get_questions|save_result`.
- **Worker:** `scripts/gemini_wh_scene.py` (vision). **Model:** `gemini-2.5-flash` (`GEMINI_VISION_MODEL`). **Context window: 1,048,576.** `temperature=0.4`.
- **Age-varied?** **Yes** — the child's age is interpolated into the prompt as a complexity cue (`Analyze this image and generate ... for a child aged {child_age}`). The server forwards `language_age` when set, and the worker uses it in place of chronological age (`if language_age is not None: child_age = language_age`). Two **difficulty modes** (`receptive`, `expressive`) produce different question sets; the server generates and saves both (`_generate_and_save_both_modes`).
- Inputs (stdin JSON): `image_path`, `child_age`, optional `language_age`, `difficulty`.

## Shared card-framing clause (`{card_framing}`, prepended in both modes)
```
IMPORTANT: The photo may show a printed card, page, or picture being held by someone or placed on a surface. Focus ONLY on the illustration or scene depicted ON the card/page itself. Ignore everything outside the card — hands holding it, the person, the table, background, etc. Treat the illustration on the card as the entire scene for your analysis.
```

## Mode `receptive` (verbatim)
```
You are a pediatric speech-language pathologist creating therapy materials.

{card_framing}

Analyze this image and generate WH-questions for a child aged {child_age}.

For EACH of the 5 WH-question types (who, what, when, where, why), generate:
1. A simple, clear question about the scene
2. The correct answer (short, 1-5 words)
3. Four visual choices (one correct + three plausible distractors), each 1-4 words
4. An evidence hint telling the child where to look in the picture

For receptive mode: make questions simple with obvious visual choices.

Return ONLY valid JSON in this exact format:
{
  "scene_description": "Brief description of the scene",
  "questions": [
    {
      "wh_type": "who",
      "question": "Who is in the picture?",
      "answer": "a boy",
      "visual_choices": ["a boy", "a girl", "a man", "a dog"],
      "evidence_hint": "Look at the person in the picture"
    },
    {
      "wh_type": "what",
      "question": "What is he doing?",
      "answer": "sleeping",
      "visual_choices": ["sleeping", "eating", "running", "reading"],
      "evidence_hint": "Look at what the person is doing"
    },
    {
      "wh_type": "when",
      "question": "When is it?",
      "answer": "nighttime",
      "visual_choices": ["nighttime", "morning", "afternoon", "lunchtime"],
      "evidence_hint": "Look at the light and setting"
    },
    {
      "wh_type": "where",
      "question": "Where is he?",
      "answer": "in bed",
      "visual_choices": ["in bed", "at school", "in the park", "at the store"],
      "evidence_hint": "Look at the place around the person"
    },
    {
      "wh_type": "why",
      "question": "Why is he sleeping?",
      "answer": "because he is tired",
      "visual_choices": ["because he is tired", "because he is hungry", "because he is sad", "because it is raining"],
      "evidence_hint": "Think about why someone would do this"
    }
  ]
}
```

## Mode `expressive` (verbatim)
Open-ended imagination questions; any answer is acceptable.
```
You are a pediatric speech-language pathologist creating therapy materials.

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

Use `wh_type` from {"what", "when", "why"} only — pick whichever best fits the question
(e.g., "what" for future/past/alternative/personal, "why" for feeling, "when" works for past/future timing).

Leave `answer` as an empty string and `visual_choices` as an empty array — they are not used in expressive mode.
`evidence_hint` should be a short imagination prompt (e.g., "Imagine what comes next.", "Think about how you would feel.").

Return ONLY valid JSON in this exact format:
{
  "scene_description": "Brief description of the scene",
  "questions": [
    {
      "wh_type": "what",
      "question": "What do you think he will do after playing soccer?",
      "answer": "",
      "visual_choices": [],
      "evidence_hint": "Imagine what comes next."
    },
    {
      "wh_type": "what",
      "question": "What was he doing right before this picture?",
      "answer": "",
      "visual_choices": [],
      "evidence_hint": "Imagine what happened just before."
    },
    {
      "wh_type": "what",
      "question": "Do you like playing soccer too?",
      "answer": "",
      "visual_choices": [],
      "evidence_hint": "Think about your own experience."
    },
    {
      "wh_type": "what",
      "question": "What else could he play instead of soccer?",
      "answer": "",
      "visual_choices": [],
      "evidence_hint": "Imagine a different game."
    },
    {
      "wh_type": "why",
      "question": "How do you think he feels while playing?",
      "answer": "",
      "visual_choices": [],
      "evidence_hint": "Look at his face and body, and imagine the feeling."
    }
  ]
}
```
