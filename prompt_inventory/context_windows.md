# Model context windows (per activity's model)

"Context window" = the maximum **input** tokens the model accepts. The code also sets **output** caps (`max_tokens` / `max_output_tokens`); 
those are far below the context window and are listed separately so the two are not confused.

## Models used in this project

| Model (as written in code) | Context window (input tokens) | Model max output | Output cap set in this code |
|---|---|---|---|
| `claude-sonnet-4-6` | **1,000,000** (1M) | 64,000 (64K) | story `max_tokens=4096`; intent-correction `max_tokens=128`; RAG chat `max_tokens=4096` |
| `gemini-2.5-flash` | **1,048,576** (~1M) | 65,536 (64K) | quiz `max_tokens` up to 8192; story sub-passes 1536–4096; others use model default |
| `gemini-2.5-flash-image` (“Nano Banana”) | **32,768** | 8,192 | image worker requests 1 image |
| `gemini-robotics-er-1.6-preview` | **131,072 effective** (docs list 1,048,576) | 65,536 (64K) | uses default; `thinking_budget=0` |

### Notes / caveats

- **`claude-sonnet-4-6` = 1,000,000-token context window** (max output 64K). This
  is the authoritative figure from the Claude API model catalog. The project caps
  *output* much lower (`max_tokens=4096` for stories, `128` for ASR intent
  correction) — those are output limits, not the context window.
- **`gemini-2.5-flash` ≈ 1,048,576 input tokens** (commonly described as “1M”),
  65,536 max output.
- **`gemini-2.5-flash-image` = 32,768 input / 8,192 output.** This is the image
  generator; it is the smallest context window of any model in the project.
- **`gemini-robotics-er-1.6-preview`** is documented at 1,048,576 input tokens,
  but Google’s API currently returns a **131,072 (128K)** effective input limit in
  practice (reported discrepancy); 64K output. It is built on Gemini 3.0 Flash and
  used only for held-object detection / 2-D pointing.

## Which activity uses which model

| Activity | Model | Context window |
|---|---|---|
| AI Conversation Assistant (spoken) | `claude-sonnet-4-6` | 1,000,000 |
| Story generation (default) | `claude-sonnet-4-6` | 1,000,000 |
| Story sub-passes: comprehension Qs, takeaway MCQs, gesture tagging, page splitting, scene identification | `gemini-2.5-flash` | 1,048,576 |
| Story sentence illustrations | `gemini-2.5-flash-image` | 32,768 |
| Educational Quiz (generation, feedback, WH options) | `gemini-2.5-flash` | 1,048,576 |
| Scene/Object game question & riddle generation | `gemini-2.5-flash` | 1,048,576 |
| Scene/Object game spatial validation (still + video) | `gemini-2.5-flash` | 1,048,576 |
| Scene/Object game held-object detection / pointing | `gemini-robotics-er-1.6-preview` | 131,072 (eff.) |
| Toy interaction & recovery question (vision) | `gemini-2.5-flash` | 1,048,576 |
| Conversation follow-up | `gemini-2.5-flash` | 1,048,576 |
| WH Picture Scene (vision) | `gemini-2.5-flash` | 1,048,576 |
| ASR intent correction (helper) | `claude-sonnet-4-6` | 1,000,000 |

## Where each model name is set in code

- `claude-sonnet-4-6`
  - `config/default.yaml` → `llm` parameter (the conversation assistant).
  - `src/web_user_server.py:241` → `StoryGenerator(llm_model="claude-sonnet-4-6")`.
  - `src/web_user_server.py:383` → intent-correction `ChatWithRAG(model="claude-sonnet-4-6")`.
  - `scripts/claude_story.py` → `--model` default `claude-sonnet-4-6`.
- `gemini-2.5-flash`
  - `scripts/gemini_general.py`, `gemini_story.py` → `--model` default.
  - `gemini_recovery_question.py`, `gemini_conversation_followup.py`,
    `gemini_wh_scene.py`, `gemini_validate_spatial[_video].py` → `GEMINI_VISION_MODEL`
    env default `gemini-2.5-flash`.
- `gemini-2.5-flash-image`
  - `src/image_generator.py` → `GOOGLE_IMAGE_MODEL` env default.
- `gemini-robotics-er-1.6-preview`
  - `scripts/gemini_analyze_image.py` → hardcoded `MODEL_ID`.

## Sources (context-window figures)

- Claude `claude-sonnet-4-6`: Claude API model catalog (1M context, 64K output) — `claude-api` skill reference.
- Gemini 2.5 Flash (~1,048,576 input): [datastudios.org — Gemini 2.5 Flash context window](https://www.datastudios.org/post/google-gemini-2-5-flash-context-window-token-limits), [Gemini API tokens docs](https://ai.google.dev/gemini-api/docs/tokens).
- Gemini 2.5 Flash Image / Nano Banana (32,768 / 8,192): [OpenRouter — gemini-2.5-flash-image](https://openrouter.ai/google/gemini-2.5-flash-image), [Galaxy.ai model specs](https://blog.galaxy.ai/model/gemini-2-5-flash-image).
- Gemini Robotics-ER 1.6 (docs 1,048,576 / effective 131,072): [Google DeepMind model card](https://deepmind.google/models/model-cards/gemini-robotics-er-1-6/), [Google AI dev forum — input limit discrepancy](https://discuss.ai.google.dev/t/gemini-robotics-er-1-6-preview-input-token-limit-doesnt-match-documentation/140573).
