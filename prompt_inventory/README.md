# Prompt & Model Inventory — QTrobot `version_1_llm_gemini`

Generated: 2026-06-19. Source: `src/web_user_server.py`, `src/story_generator.py`,
`src/qt_ai_data_assistant.py`, `src/llm_prompts.py`, `src/image_generator.py`,
`src/knowledge_base.py`, `src/persona_rag.py`, `documents/sar_system_prompt.md`,
and `scripts/*.py`.

This directory documents, for **each activity**:

1. **The entire prompt** sent to the LLM (verbatim templates, with the runtime
   placeholders marked), including **per-age variations** where they exist.
2. **The model** the activity calls and **its context window**.

> Verbatim prompts are reproduced in the per-activity files. Python f-string
> placeholders like `{child_age}` / `{age}` / `{toy_list}` are shown exactly as
> they appear in the code; the value substituted at runtime is described next to
> each one.

---

## How the system calls LLMs (architecture in one paragraph)

The Flask app (`src/web_user_server.py`) runs under Python 3.8. It does **not**
call the Gemini SDK directly — it shells out to Python-3.9 worker scripts in
`scripts/` (via `WORKER_PYTHON`, default `./.venv39/bin/python`). Text-only Gemini
calls go through `scripts/gemini_general.py` (helper `_gemini_generate`); vision
calls go through dedicated scripts. Stories go through `StoryGenerator`
(`src/story_generator.py`), which routes to `scripts/claude_story.py` (Claude),
`scripts/gemini_story.py` (Gemini), or a local Ollama server based on the model
name. The spoken conversation assistant (`src/qt_ai_data_assistant.py`) is a ROS
node that talks to Claude through LlamaIndex (`src/llamaindex_interface.py`).

`.env` does **not** override the model env vars, so the script defaults apply
everywhere (see `context_windows.md`).

---

## Master table — activity → model → context window

| # | Activity (user entry point) | LLM call site | Model | Context window | Age-varied prompt? |
|---|------|------|-------|----------------|--------------------|
| 1 | **AI Conversation Assistant** (spoken; `qt_ai_data_assistant.py`, `/start_assistant`) | `llamaindex_interface.ChatWithRAG` → `Anthropic(...)` | `claude-sonnet-4-6` | **1,000,000** | No (age passed as context text) |
| 2 | **Story / Read Story** (`/read_story`, `/api/generate_story[_stream]`) | `StoryGenerator` → `scripts/claude_story.py` | `claude-sonnet-4-6` *(configurable)* | **1,000,000** | **Yes** (4 age tiers) |
| 2a | ↳ Story comprehension questions | `_gemini_generate` → `gemini_general.py` | `gemini-2.5-flash` | **1,048,576** | **Yes** (3 bands) |
| 2b | ↳ Story takeaway MCQs | `_gemini_generate` | `gemini-2.5-flash` | 1,048,576 | Soft (age in text) |
| 2c | ↳ Story gesture/emotion tagging | `_gemini_generate` | `gemini-2.5-flash` | 1,048,576 | No |
| 2d | ↳ Story page splitting | `_gemini_generate` | `gemini-2.5-flash` | 1,048,576 | **Yes** (3 bands) |
| 2e | ↳ Story scene identification (for images) | `_gemini_generate` | `gemini-2.5-flash` | 1,048,576 | No |
| 2f | ↳ Story sentence illustrations | `image_generator.py` worker | `gemini-2.5-flash-image` | **32,768** | No |
| 3 | **Educational Quiz** (`/quiz_generation`, `/educational_quiz`) | `_GeminiQuizLLM` / `_gemini_generate` | `gemini-2.5-flash` | 1,048,576 | **Yes** (difficulty→age) |
| 4 | **Scene / Object Game** (`/play_scene`, `/object_game_generate`, `/api/scene/*`) | mixed (below) | mixed | — | **Yes** (question gen) |
| 4a | ↳ Question / riddle generation | `_gemini_generate` | `gemini-2.5-flash` | 1,048,576 | **Yes** (3 tiers) |
| 4b | ↳ Spatial-relation validation (still + video) | `gemini_validate_spatial[_video].py` | `gemini-2.5-flash` | 1,048,576 | No |
| 4c | ↳ Held-object detection / pointing | `gemini_analyze_image.py` | `gemini-robotics-er-1.6-preview` | **131,072** (see notes) | No |
| 5 | **Toy interaction & recovery** (`/select_toy`, `/api/recovery/generate_question`) | `gemini_recovery_question.py` (vision) | `gemini-2.5-flash` | 1,048,576 | **Yes** (4 bands) |
| 6 | **Conversation follow-up** (`/conversation_builder`, `/api/conversation/wait_for_turn`) | `gemini_conversation_followup.py` | `gemini-2.5-flash` | 1,048,576 | **Yes** (4 bands) |
| 7 | **WH Picture Scene** (`/wh_picture_scene`, `/wh_picture_play`, `/api/wh_scene/*`) | `gemini_wh_scene.py` (vision) | `gemini-2.5-flash` | 1,048,576 | **Yes** (age cue + modes) |
| H | **(Helper) ASR intent correction** | `_ensure_intent_llm` → `ChatWithRAG` | `claude-sonnet-4-6` | 1,000,000 | No |

*Story model is configurable:* `StoryGenerator(llm_model=...)` is constructed with
`"claude-sonnet-4-6"` in `web_user_server.py:241`. If changed to a `gemini-*`
name it routes to `gemini_story.py` (`gemini-2.5-flash`, 1,048,576); any other name
is treated as a local **Ollama** model.

---

## Age-variation summary

| Activity | Age affects prompt? | How |
|----------|---------------------|-----|
| Story (main) | **Yes, strongly** | 4 tiers: **3**, **4–5**, **6–7**, **8–10**. Different templates, word ranges, language guidelines; WH-question short-story format at 4–5; explicit Takeaways at 7+ (`requires_takeaways`); embedded WH-questions at 6–7 (`requires_wh_questions`). Tier chosen by `language_age` if set, else age. Plus a knowledge-base fragment derived from age + gender. |
| Story comprehension Qs | Yes | `complexity_age ≤4` → 3 Qs; `≤6` → 4 Qs; else 5 Qs incl. 2 inference. |
| Story page splitting | Yes | `age ≤4` → ~1–2 sentences/page; `≤6` → 2–3; else 3–5. |
| Story takeaway MCQs | Soft | `{child_age}` interpolated into the prompt text; no hard branch. |
| Story gesture tagging / scene id / image gen | No | — |
| Educational Quiz | **Yes** | `difficulty` → age band: `low`→“Target ages 2-3”, `med`→“4-5”, `high`→“7+”. Social-rules branch targets 7+ with its own goal + length rule. |
| Quiz feedback / WH options | No | — |
| Scene/Object game question | **Yes** | `≤3` → direct request (no LLM, fixed templates); `4–6` → criteria prompt; `7+` → riddle prompt. Tier chosen by `language_age` if set. |
| Spatial validation / object detection | No | — |
| Toy recovery question | **Yes** | Each mode (`toy`, `toy_followup`, `child`) has 4 age bands: **2–3 / 4–5 / 6–7 / 8+**. |
| Conversation follow-up | **Yes** | 4 age bands **2–3 / 4–5 / 6–7 / 8+**, plus “easy for a `{age}`-year-old” question rule. |
| WH Picture Scene | **Yes** | “for a child aged `{child_age}`” complexity cue (uses `language_age` when set); receptive vs expressive question sets. |
| AI Conversation Assistant | No hard branch | Child age is appended to each query as `"<name> (Age: N)"` context; the SAR system prompt defaults to ages 4–10. |
| ASR intent correction | No | — |

**Structured knowledge base (extra age/gender variation).** Beyond the tiers above,
a developmental knowledge base (`knowledge_base.py` + `Simple_version_slp_codesign_knowledge_base.json`)
is resolved by age/gender and **injected into the prompt** of three activities —
**story generation, story comprehension questions, and scene/object-game question
generation** — supplying an MLU target, age-appropriate language targets, and
interest themes. It is **not** added to any other activity. The data and the actual
per-age resolved text are in [`knowledge_base_data.md`](knowledge_base_data.md).

---

## File index

- [`context_windows.md`](context_windows.md) — every model, its context window (with sources), and the output-token caps the code sets.
- [`token_usage.md`](token_usage.md) — **final results: measured prompt token usage per activity vs. the model context window** (opens with **how to calculate tokens**; includes per-age story sizes, image/video/RAG adders, and % utilization).
- [`count_tokens.py`](count_tokens.py) — runnable script that assembles each activity's real prompt and counts it via the provider `count_tokens` APIs (run from the demo root).
- [`activity_01_ai_conversation_assistant.md`](activity_01_ai_conversation_assistant.md)
- [`activity_02_story_reading.md`](activity_02_story_reading.md)
- [`activity_03_educational_quiz.md`](activity_03_educational_quiz.md)
- [`activity_04_scene_object_game.md`](activity_04_scene_object_game.md)
- [`activity_05_toy_recovery_and_followup.md`](activity_05_toy_recovery_and_followup.md)
- [`activity_06_wh_picture_scene.md`](activity_06_wh_picture_scene.md)
- [`activity_07_image_generation.md`](activity_07_image_generation.md)
- [`knowledge_base_data.md`](knowledge_base_data.md) — **the structured knowledge base itself**: its data, where it is/ isn't injected, and the actual per-age resolved fragments (the real content behind `{persona_section}` etc.).
- [`helpers_and_shared.md`](helpers_and_shared.md) — ASR intent correction + the fragment *templates* (knowledge-base & legacy persona).
