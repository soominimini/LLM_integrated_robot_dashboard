# Model Usage by Activity

Which AI model powers which task, per activity. Last updated **2026-07-03**
(the day the story-pipeline text passes moved from Gemini 2.5 Flash to Claude
Sonnet 4.6, and the knowledge base moved to
`documents/restructured_knowledge_base_v2.json`).

All Claude calls run in-process via the Anthropic SDK
(`_claude_generate` / `_claude_generate_image` in `src/web_user_server.py`);
Gemini calls run through worker scripts in `scripts/` (Python 3.9 venv).

---

## 1. Story activity (generate → enrich → read aloud)

| Task | Model | Where | Trace tag |
|---|---|---|---|
| Story + WH-question generation (KB fragment injected) | **Claude Sonnet 4.6** | `StoryGenerator(llm_model="claude-sonnet-4-6")` → `scripts/claude_story.py` | `[StoryGenerator]` |
| Emotion/gesture tagging (`[emotion:QT/...]`, `[gesture:...]`) | **Claude Sonnet 4.6** | `_apply_emotion_tags_with_llm()` | `[Claude:emotion-tagger]` |
| Page splitting (2–3 sentences/page) | **Claude Sonnet 4.6** | `_split_story_into_pages()` | `[Claude:page-splitter]` |
| Scene analysis (which paragraphs share an illustration) | **Claude Sonnet 4.6** | `_split_paragraphs_into_scenes()` | `[Claude:scene-analyzer]` |
| Story comprehension questions (4 MC, KB fragment injected) | **Claude Sonnet 4.6** | `_generate_story_questions()` | `[Claude:story-questions]` |
| Takeaway quiz questions | **Claude Sonnet 4.6** | `_generate_takeaway_questions()` | `[Claude:takeaway-questions]` |
| **Scene illustration image** | **Gemini 2.5 Flash Image** | `src/image_generator.py` (`GOOGLE_IMAGE_MODEL`) | `Generating image for prompt:` |
| Reading aloud (TTS) | **Qwen3 TTS realtime**, custom cloned voice | `src/tts_helper.py` (`TTS_ENGINE=qwen`, default) | `self.engine: qwen` |

## 2. Educational quiz (yes/no + WH generation)

| Task | Model | Where | Trace tag |
|---|---|---|---|
| KB wording-level calibration (MLU only — `include_targets=False`, so no sound/interest steering; not an LLM call) | — (v2 knowledge base, local lookup) | `build_question_prompt_fragment()` | `[KB] derived ... kind=question` |
| Yes/No question batch (up to 100 per call) | **Claude Sonnet 4.6** | `_ClaudeQuizLLM` (`MODEL = "claude-sonnet-4-6"`, `max_tokens=16384`) | `[Claude]` |
| WH question batch with `accepted_answers` lists (up to 100 per call) | **Claude Sonnet 4.6** | same `_ClaudeQuizLLM` instance | `[Claude]` |
| WH `accepted_answers` backfill (repair pass for questions returned with missing/thin alternative lists) | **Claude Sonnet 4.6** | follow-up call in `api_generate_quiz()` | `[Claude]` |

Timing reference (2026-07-03 session, count=100, user sophie): yes/no batch ≈ 34 s
(8.8k chars); WH batch ≈ 73 s (18.7k chars — `accepted_answers` roughly doubles
the output) + ≈ 2 s backfill for 2/100 questions. `max_tokens` must stay below
~21k: the Anthropic SDK rejects non-streaming calls whose worst case exceeds
10 minutes (this caused a one-off "LLM returned invalid JSON" at 32768).

## 3. Scene game (object-request game)

| Task | Model | Where | Trace tag |
|---|---|---|---|
| Game question generation (KB fragment injected) | **Claude Sonnet 4.6** | `SCENE_GAME_LLM_MODEL` default | `[Claude:scene-game-question]` |
| Object/criteria match validation | **Claude Sonnet 4.6** | `SCENE_GAME_LLM_MODEL` default | `[Claude:scene-game-criteria-match]` |
| Object detection in camera frame (points/labels/colors) | **Gemini Robotics-ER 1.6 preview** | `scripts/gemini_analyze_image.py` |  |

## 4. Movement / spatial direction game

| Task | Model | Where |
|---|---|---|
| Still-frame spatial validation (left/right/on/under…) | **Claude Sonnet 4.6** (vision) | `_claude_generate_image`, `SPATIAL_VALIDATION_MODEL` default |
| Video-clip depth relations (behind / in front / in / out) | **Gemini 2.5 Flash** (video) | `scripts/gemini_validate_spatial_video.py` |
| Object detection in camera frame | **Gemini Robotics-ER 1.6 preview** | `scripts/gemini_analyze_image.py` |

## 5. WH picture-scene activity

| Task | Model | Where | Trace tag |
|---|---|---|---|
| Scene photo capture | — (no AI; RealSense via ROS, published by `qt_nuitrack_app`) | `/api/wh_scene/capture` | |
| Receptive question set (5 WH questions + visual choices, vision) | **Claude Sonnet 4.6** (since 2026-07-03; was Gemini 2.5 Flash) | `_run_wh_scene_analysis()` → `_claude_generate_image` | `[Claude:wh-scene-receptive]` |
| Expressive question set (5 open-ended imagination questions, vision) | **Claude Sonnet 4.6** | same — one call per mode, both at capture/upload/regenerate time | `[Claude:wh-scene-expressive]` |
| Play session (questions read from saved JSON) | — (no LLM) + **Qwen3 TTS** for reading aloud | `/wh_picture_play` | |

The former Gemini worker `scripts/gemini_wh_scene.py` is no longer called.

## 6. Camera-based conversation & recovery

| Task | Model | Where |
|---|---|---|
| Recovery question from camera frame (re-engage child) | **Gemini 2.5 Flash** (vision) | `scripts/gemini_recovery_question.py` |
| Conversational follow-ups | **Gemini 2.5 Flash** | `scripts/gemini_conversation_followup.py` |

## 7. Speech (cross-cutting, all activities)

| Task | Model | Where |
|---|---|---|
| Speech-to-text (child's speech) | **OpenAI gpt-4o-transcribe** | `src/whisper.py` (name is historical — it calls the OpenAI API) |
| ASR intent correction (did the child mean the target word?) | **Claude Sonnet 4.6** | `ChatWithRAG(model="claude-sonnet-4-6")`, `_ensure_intent_llm()` |
| Text-to-speech | **Qwen3 TTS realtime** (`qwen3-tts-vd-realtime-2026-01-15`), custom voice | `src/tts_helper.py`; alternatives: `qt` (robot built-in), `polly` (AWS Polly) |

---

## Environment-variable overrides

| Variable | Default | Controls |
|---|---|---|
| `SCENE_GAME_LLM_MODEL` | `claude-sonnet-4-6` | All `_claude_generate` passes: scene game, emotion tagger, page splitter, scene analyzer, story/takeaway questions |
| `SPATIAL_VALIDATION_MODEL` | `claude-sonnet-4-6` | Still-frame spatial validation |
| `WH_SCENE_LLM_MODEL` | `claude-sonnet-4-6` | WH picture-scene question generation (vision) |
| `GOOGLE_IMAGE_MODEL` | `gemini-2.5-flash-image` | Story scene illustrations |
| `GEMINI_VISION_MODEL` | `gemini-2.5-flash` | Remaining Gemini vision workers (recovery question, spatial video) |
| `TTS_ENGINE` | `qwen` | TTS backend (`qwen` / `qt` / `polly`) |
| `QWEN_MODEL` / `QWEN_VOICE` | `qwen3-tts-vd-realtime-2026-01-15` / custom clone | Qwen TTS model + voice |

Note: the story generator model is hardcoded at `StoryGenerator(llm_model="claude-sonnet-4-6")`
in `src/web_user_server.py` (a `gemini-*` id would route to `scripts/gemini_story.py` instead).
