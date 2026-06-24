# Object-Request Game — End-to-End Flow (models & prompts)

> Scope: the **object-request** game (robot asks for an object → child holds it up →
> robot recognizes and judges it). Reflects the current code in
> `src/web_user_server.py` after the Gemini→Claude migration and the MLU-only
> knowledge-base change. Line numbers are approximate.

---

## Entry points

| Route | Handler | Purpose |
|---|---|---|
| `POST /api/scene/start` | `api_scene_start` (~6451) | **The endpoint the live UI uses.** Generates the question only. |
| `POST /api/scene_game/answer` | `api_scene_game_answer` (~4201) | Capture frame → detect held object → judge. |
| `POST /api/scene_game/hint` | `api_scene_game_hint` (~4051) | Optional hint. |
| `GET/POST /api/scene_game/toys` | (~6416) | Configure the physical toy list. |

Frontend page: `templates/play_scene.html` (calls `/api/scene/start`, then
`/api/scene_game/answer`, then optionally `/api/scene_game/hint`).

---

## PHASE 1 — Generate the question · `_scene_game_generate_question` (~3304)

`complexity_age = language_age or child_age` picks one of three tiers:

| Tier | Age | How | Model |
|---|---|---|---|
| A — exact | ≤3 | **No LLM** — random pick from 5 fixed templates (`"Show me the {target}!"`, …) | none |
| B — criteria | 4–6 | inference-request prompt → LLM | **Claude** |
| C — riddle | 7+ | riddle prompt → LLM | **Claude** |

### Knowledge base: MLU only
Immediately after the tier is chosen, the MLU is read directly from the KB and
injected as the only KB-derived guidance:

```python
kb_info = knowledge_base.describe(complexity_age)
mlu_clause = (f"TARGET MLU (knowledge base): the child's mean length of utterance "
              f"is about {mlu_range} words. Keep the question within that length and "
              f"use sentence structures a child at this language level produces.\n")
```

The full persona fragment (pronoun / `-ing` language targets, speech sounds,
interests) is **deliberately not** injected — it previously produced phrasings
like *"She is carrying small fruit."* The therapist's `learning_goals` clause is
still included (it is not KB-derived).

### LLM call — `_claude_generate` (~3240)
- **Model:** `claude-sonnet-4-6` (default; override with `SCENE_GAME_LLM_MODEL` env).
  In-process Anthropic SDK. `temperature=0.3` (auto-dropped for Opus 4.7+/Fable).
- **System:** `"You generate game questions for children. Return JSON only."`
- **Prompt (tier B, abridged):** *"Generate ONE inference-style request… HARD RULE —
  the QUESTION text must NOT name the target… refer to it only as 'it'/'something'…
  criteria = one noun + at most one adjective."* + `mlu_clause`.
- **Tier C** swaps in a riddle prompt (reason about color/shape/size/function).
- A leak-guard retries (still Claude) if a toy name appears, then sanitizes / falls back.
- **Returns:** `{question, target (exact only), criteria (4+), mode}`.

> Sibling: **direction mode** (`_scene_game_generate_direction_question`, ~3647) —
> *"put the {obj} {relation} the {ref}"*, built server-side, **no LLM**.

---

## PHASE 2 — Robot asks · `api_scene_start` (~6451)

- Loads the configured toy list; selects mode (`auto`/`criteria`/`direction`).
- Calls the question generator.
- Speaks the question via `tts_helper` (ASR suspended).
- Returns `question / target / criteria / mode` (+ direction fields) to the UI.
  **No on-screen item cards or generated toy images** — the child uses the real
  physical toys in front of the camera.

---

## PHASE 3 — Recognize the held object · `api_scene_game_answer` (~4201)

**Step 1 — capture:** `_get_ros_frame()` → `cv2.imwrite` a JPG.

**Step 2 — detect (vision):** `_run_gemini_detect_and_look` (~3895) →
`scripts/gemini_analyze_image.py`
- **Model:** `gemini-robotics-er-1.6-preview` · `temperature=0.5`, `thinking_budget=0`.
- **Prompt (verbatim):**
  ```
  Point to no more than 1 item a person is holding in the image.
  Return the object's identifying name, its dominant color, and its shape.
  The answer should follow the json format:
  [{"point": <point>, "label": <label>, "color": <color>, "shape": <shape>}, ...].
  The points are in [y, x] format normalized to 0-1000.
  ```
- Returns `label, color, shape, point`; the `point` drives `look_at_pixel` so the
  robot's head turns to the object. **Kept on Gemini** (specialized pointing model;
  no Claude equivalent for the gaze coordinate).

**Step 3 — judge** (branch on `answer_mode`):

| Mode | Logic | Model |
|---|---|---|
| exact (≤3) | inline (~4322): exact / substring / all-token match of `target` vs label+color+shape | none (local) |
| criteria (4+) | `_check_criteria_match` (~3950): ① local token fast-path → ② LLM fallback → ③ lenient string | **Claude** (in ②) |

`_check_criteria_match` ② — `_claude_generate`, system
`"You validate object matches. Return JSON only."`, prompt: *"Detected
label/color/shape … game asked for '{criteria}'. Decide if it matches using label
AND color AND shape … Return {"match": bool, "reason": ...}"*.

**Step 4 — feedback:** `tts_helper.speak` ("Great job!… / Try again!").

---

## Hint path · `api_scene_game_hint` (~4125)
- direction mode → deterministic phrase (no LLM).
- exact / criteria → `_claude_generate`, system
  `"You generate game hints for children. Return JSON only."` (**Claude**).

---

## Direction mode (sibling spatial game)
Answers for `mode == "direction"` go to `_run_gemini_validate_spatial[_video]`
(~3774 / ~3861) → `scripts/gemini_validate_spatial.py` /
`gemini_validate_spatial_video.py`.
- **Model:** `gemini-2.5-flash` (still frame, or 3-sec MP4 for depth relations).
- **Still on Gemini** — these are vision tasks, and the video path uploads an MP4
  via the Gemini Files API. Claude supports image input but **not** video, so the
  video validator cannot move to Claude as-is.

---

## Model map (current)

| Stage | Function | Model | Provider |
|---|---|---|---|
| Question gen (4–6, 7+) | `_scene_game_generate_question` | `claude-sonnet-4-6` | **Claude** |
| Question gen (≤3) | same | local templates | — |
| Held-object detection | `_run_gemini_detect_and_look` | `gemini-robotics-er-1.6-preview` | Gemini (vision/point) |
| Criteria match (4+) | `_check_criteria_match` | `claude-sonnet-4-6` | **Claude** |
| Exact match (≤3) | inline | local | — |
| Hint (4+) | `api_scene_game_hint` | `claude-sonnet-4-6` | **Claude** |
| Spatial validation (direction) | `_run_gemini_validate_spatial[_video]` | `gemini-2.5-flash` | Gemini (vision/video) |

The Claude calls all go through the in-process `_claude_generate` helper
(`anthropic.Anthropic()`, key from `ANTHROPIC_API_KEY` in `src/.env`).

---

## Change log (vs. original Gemini implementation)
1. **Question generation** → Claude (was Gemini Flash).
2. **Knowledge-base reference** → MLU only (dropped pronoun/`-ing` targets,
   speech sounds, interests).
3. **Criteria-match judge** (4+) → Claude (was Gemini Flash).
4. **Hint** → Claude (was Gemini Flash).
5. Still on Gemini, by design: the robotics-ER **detector** (pointing/gaze) and the
   direction-mode **spatial validators** (vision + video).

## Removed dead code
The orphaned `/api/scene_game/new_round` endpoint (`api_scene_game_new_round`) was
**deleted**. It built one card per toy and generated a missing image per toy via
`image_generator` (`prompt=f"{label}, single object on simple background,
children's book illustration"`), but no template, static JS, or Python ever called
it — the live UI uses `/api/scene/start`, which does not build image cards. The
per-toy images were never shown to the user (the game uses real physical toys), so
the endpoint and its image-generation block were removed.
