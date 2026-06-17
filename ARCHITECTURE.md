# QTrobot — System Architecture

> Regenerated 2026-06-17 from a direct read of the source tree. This document
> replaces the previous `ARCHITECTURE.md`. File/line references point at the
> code as it exists in `src/`, `scripts/`, `config/`, `documents/`, and
> `templates/`.

---

## 1. Overview

This folder contains **two independent applications** that happen to share a
small set of library modules. They have separate entry points, separate speech
stacks, and separate LLM wiring, and **neither imports the other**:

| | App A — Robot Assistant | App B — Web Game Platform |
|---|---|---|
| Entry point | `src/qt_ai_data_assistant.py` | `src/web_user_server.py` (6.5k lines) |
| How it runs | ROS node, autostarted (`scripts/autostart/`) | `python3 web_user_server.py`, Flask on `:8080` |
| Audience | End user talking to the robot | Therapist/clinician in a browser + child via robot |
| ASR | NVIDIA Riva + Silero VAD | `whisper.py` subprocess (OpenAI `gpt-4o-transcribe`) |
| Conversational LLM | Claude `claude-sonnet-4-6` via LlamaIndex (+ RAG) | per-feature: Claude/Gemini story + Gemini vision scripts |
| TTS | Robot built-in voice (ROS `behavior/talkText`) | `tts_helper` (Qwen / Polly / QT) |
| Vision | DeepFace (face detect + re-ID) | Gemini vision scripts on camera frames |
| Config | `config/default.yaml` via ParamifyWeb | env vars + on-disk JSON |

**The directory name (`version_1_llm_gemini`) is misleading.** App A's
conversational model defaults to **Anthropic Claude**, its embeddings are
**local Ollama**, its ASR is **NVIDIA Riva**, and its TTS is the **robot's own
voice**. Gemini is used heavily, but mostly by App B's vision/validation/quiz
subprocess scripts.

### The subprocess pattern (important)

Both servers run under a Python (3.8 / 3.9-ROS) environment that does **not**
have `google-genai`, `anthropic`, or `openai` installed. Therefore all heavy AI
work is **shelled out to Python-3.9 subprocesses** (`.venv39`). This is why
`scripts/gemini_*.py`, `scripts/claude_story.py`, `src/whisper.py`, and
`src/image_generator_worker.py` are "used" without ever being `import`ed.

Standardized subprocess contract:
- **Input:** `--prompt-file` / `--image` / `--video` CLI args, or a JSON object on stdin.
- **Output:** plain text, a JSON object on stdout, or `CHUNK:`-prefixed lines for streaming.

---

## 2. Architecture diagram

```
                         version_1_llm_gemini/
   ┌──────────────────────────────────┐   ┌────────────────────────────────────────┐
   │  APP A: ROBOT ASSISTANT (ROS)     │   │  APP B: FLASK GAME PLATFORM (port 8080)  │
   │  qt_ai_data_assistant.py          │   │  web_user_server.py  (browser <-> robot) │
   │                                   │   │                                          │
   │  ASR: Riva + Silero VAD           │   │  ASR: whisper.py subprocess              │
   │   (riva_speech_recognition_vad)   │   │       (OpenAI gpt-4o-transcribe)         │
   │  LLM: Claude via LlamaIndex       │   │  LLM: story_generator -> claude/gemini   │
   │   (+ Ollama embeds, RAG over PDFs)│   │       scripts; gemini_* vision scripts   │
   │  TTS: ROS behavior/talkText       │   │  TTS: tts_helper (Qwen/Polly/QT)         │
   │  Vision: DeepFace faces           │   │  Vision: Gemini scripts on camera frames │
   │  Behaviors: idle_attention,       │   │  Images: image_generator -> worker(.venv39)│
   │   human_tracking, command_iface   │   │  Personas: persona_rag                   │
   │  Config: ParamifyWeb + default.yaml│   │  UI: templates/*.html                    │
   └───────────────┬───────────────────┘   └──────────────────┬───────────────────────┘
                   │             shared library modules        │
                   └───►  user_management • kinematics/* • human_tracking • utils/*  ◄──┘
                                   (+ documents/sar_system_prompt.md, used by BOTH)
```

---

## 3. Entry points & how to run

### App A — robot assistant
Autostarted via `scripts/autostart/start_qt_ai_data_assitant.sh`, which waits for
ROS topics + the Riva docker, then:
```bash
cd ~/robot/code/tutorials/demos/qt_ai_data_assistant/
source venv/bin/activate
python3.9 src/qt_ai_data_assistant.py
```
`QTAIDataAssistant(ParamifyWeb, BaseNode)` reads `config/default.yaml`, serves a
live config web panel (ParamifyWeb), and runs the conversation loop on a
background thread (`utils/base_node.py`).

### App B — web game platform
```bash
python3 src/web_user_server.py      # Flask, host=0.0.0.0, port=8080, debug=True
```
Runs on the robot; a therapist drives it from a browser. ROS / `cv2` / LLM
imports are `try/except`-guarded (`ROS_AVAILABLE`, `HUMAN_TRACKING_AVAILABLE`,
`LLM_AVAILABLE`) so it degrades gracefully on a dev machine.

---

## 4. App A — Robot conversational assistant

**Class:** `QTAIDataAssistant(ParamifyWeb, BaseNode)` in `qt_ai_data_assistant.py`.

### Runtime loop
Single worker thread (`BaseNode._run`):
1. State `IDLE` → `asr.recognize_once()` — Silero VAD gates when Riva starts
   transcribing (`riva_speech_recognition_vad.py:278`).
2. On the first interim Riva result the robot turns its head toward the speaker
   (sound-direction + DeepFace face fusion, `acknowledge_human`).
3. Final transcript → `_asr_callback` → `ChatWithRAG.get_stream_response()`
   streams the Claude reply.
4. The reply is spoken **sentence-by-sentence** (so TTS can start mid-response)
   via ROS `/qt_robot/behavior/talkText` (`command_interface.py:165`).
5. After `IDLE_INTERACTION_TIMEOUT` (60 s) of inactivity, arms home and idle
   attention resumes.

There is **no TTS library in App A** — speech is the robot's built-in voice via
the ROS `behavior_talk_text` service in `command_interface.py`. Volume/voice are
set via `/qt_robot/setting/setVolume` and `/qt_robot/speech/config`.

### LLM / RAG (`llamaindex_interface.py`)
- Provider: `llama_index.llms.anthropic.Anthropic`, model from config `llm`
  (default `claude-sonnet-4-6`), `max_tokens=4096`.
- Embeddings: `OllamaEmbedding("mxbai-embed-large:latest")` — local Ollama, only
  when RAG is enabled.
- Documents: `SimpleDirectoryReader(input_dir=docs, required_exts=formats,
  num_files_limit=max_docs).load_data()` → `VectorStoreIndex.from_documents`
  (rebuilt in memory each start).
- Chat memory: `ChatMemoryBuffer` backed by `SimpleChatStore`, persisted **per
  user** to `<user>/chat_history/chat_memory.json`.
- `disable_rag=True` swaps the context engine for a plain `SimpleChatEngine`;
  toggled live via the `on_disable_rag_set` param callback.

### Command protocol
The system prompt (`llm_prompts.py` / config `role`) instructs the model to emit
JSON commands. `proccess_response` (`qt_ai_data_assistant.py:322`) tries
`json.loads` on each streamed chunk; if it parses, the chunk is dispatched to
`CommandInterface.execute` instead of being spoken:
- `{"command":"pause_interaction"}` → PAUSED + `hold_on`, plays "confused" emotion.
- `{"command":"forget_conversation"}` → `chat.clear_memmory()`.
- `{"command":"set_language","code":"fr-FR"}` → reconfigures robot TTS and
  **rebuilds the Riva ASR** for the new locale.
While paused, utterances are routed through a `WakeupPrompt` classifier
(`llm_prompts.py`) that only resumes on an explicit request.

### Vision & attention
- `human_presence_detection.py` — subscribes to `/camera/color/image_raw`;
  DeepFace `extract_faces` (retinaface) for detection + `represent` (VGG-Face) /
  `verify` for re-ID of up to 5 known faces; computes each face's 3D position via
  `kinematics.pixel_to_base`; fuses `/qt_respeaker_app/sound_direction` to pick
  the active speaker.
- `human_tracking.py` — single-worker thread pool, gazes at a target face at
  ~10 Hz.
- `idle_attention.py` — random gaze / face-tracking when idle.

---

## 5. App B — Flask educational game platform

**Single monolith:** `web_user_server.py`. Singletons instantiated at import:
`UserManager`, `StoryGenerator(claude-sonnet-4-6)`, `PersonaRAG`, `TTSHelper`,
`ImageGenerator`.

### Feature / route map (→ template)

| Feature | Template(s) | Key routes |
|---|---|---|
| Auth / profile | `index.html`, `dashboard.html` | `/api/register`, `/api/login`, `/api/current_user`, `/api/update_profile` |
| Story reading | `read_story.html` | `/api/generate_story[_stream]`, `/api/save_story`, `/read_story/<f>`, `/api/get_story_sentences`, `/api/speak_sentence` |
| Educational quiz | `quiz_generation.html`, `educational_quiz.html` | `/api/generate_quiz`, `/api/save_quiz`, `/api/get_saved_quiz`, `/api/generate_quiz_feedback`, `/api/teach_quiz_answer` |
| Scene / object game | `scene_game_config.html`, `play_scene.html`, `select_toy.html` | `/api/scene/start`, `/api/scene_game/{new_round,hint,answer}`, `/api/scene_game/toys` |
| WH-picture play | `wh_picture_scene.html`, `wh_picture_play.html` | `/api/wh_scene/{upload,capture,list,get_questions,save_result}` |
| DIY "Recovery Strategy" builder | `diy_builder.html` | `/api/activity/{prepare,test,save,run_saved,stop,step_status,confirm_step}` |
| Conversation builder | `conversation_builder.html` | `/api/conversation/{wait_for_turn,check_red_card}` |
| My games / play hub | `my_games.html`, `play_games.html` | `/my_games`, `/api/get_custom_games` |
| Robot / camera control | — | `/api/robot_gesture`, `/api/camera_frame`, `/api/camera_capture`, `/api/human_tracking/*`, `/api/head_position`, `/api/volume_*` |

### Speech I/O
- **ASR:** `whisper.py` (despite the name, OpenAI `gpt-4o-transcribe`) run as a
  subprocess. It subscribes to the ReSpeaker mic topic, does RMS-based VAD, and
  prints `PARTIAL:` lines (~every 2 s) and a final `FINAL:` line.
  `_whisper_recognize_once` parses `FINAL:`; `_whisper_recognize_streaming`
  (`web_user_server.py:885`) pipes `PARTIAL:` chunks straight into TTS.
- **TTS:** `tts_helper.TTSHelper`, engine chosen by `TTS_ENGINE`
  (default `qwen`). See §7.

### Example game flows
- **Scene / object answer** (`/api/scene_game/answer`): grab a ROS camera frame →
  save JPEG under `user_data/<user>/captured_scenes/` → run
  `gemini_analyze_image.py` (label/color/shape + a point) → robot looks at the
  object via `kinematics.look_at_pixel` → match vs target/criteria → TTS feedback.
  "Direction" mode records a 3 s MP4 and uses `gemini_validate_spatial_video.py`;
  flat relations use `gemini_validate_spatial.py`.
- **WH-picture:** therapist uploads/captures an image → `gemini_wh_scene.py` runs
  twice (receptive + expressive) → child answers verbally (Whisper) → score saved.
- **Conversation turn** (`/api/conversation/wait_for_turn`): wait for TTS to
  finish → open mic → run a parallel red-card watcher (HSV red-area threshold) +
  up to 20 Whisper rounds → `gemini_conversation_followup.py` generates the next
  turn.

### Robot / camera control
- Camera: `CameraCapture` lazy singleton subscriber to `/camera/color/image_raw`
  (1-deep queue); `_get_ros_frame()` waits ~1.5 s. A `cv2`/V4L2 fallback exists.
- Emotion/gesture: ROS service proxies (`qt_gesture_controller/gesture_play`,
  `qt_robot_interface/emotion_show`) and `rostopic pub` subprocesses for story
  emotion/gesture tags.
- `human_tracking` singleton with a 60 s failure cooldown; paused/resumed around
  scene capture to avoid motion blur.

---

## 6. Shared library modules

| Module | Used by | Purpose |
|---|---|---|
| `user_management.py` | A + B | User store (`users.json`), per-user dirs, chat-memory path |
| `kinematics/` (`kinematic_interface`, `head_solver`, `arms_solver`) | A + B | Head/arm IK, pixel↔base transforms, look-at |
| `human_tracking.py` | A + B | Head-follow gaze (depends on `human_presence_detection`) |
| `utils/` (`base_node`, `logger`, `utils`) | A | Threaded node base, logging, sentence splitting |

---

## 7. TTS engines (`tts_helper.py`, App B)

Engine selected by `TTS_ENGINE` env (default `qwen`):

- **`qwen`** (default) — DashScope Qwen realtime TTS over websocket
  (`qwen3-tts-vd-realtime-…`, custom voice). Writes a WAV, optionally
  speed-adjusts via `wav_speed.adjust_wav_speed` (ffmpeg `atempo`, `TTS_SPEED`),
  base64-uploads to the robot, plays + lip-syncs through
  `/qt_robot/behavior/talkAudio`.
- **`polly`** — AWS Polly neural (voice `Justin`) via
  `tts/local_polly_generator.generate_polly_audio`; MP3→WAV (ffmpeg); optional
  pylips `RobotFace` lipsync; plays remotely over SSH.
- **`qt`** — robot's built-in voice via ROS `behavior/talkText`.

`movement_enabled` defaults to **False** (head/arm motion during speech is
delegated to HumanTracking). pylips lipsync is optional/experimental.

---

## 8. Subprocess workers (`scripts/`)

All are `#!/usr/bin/env python3.9`, invoked as subprocesses. Gemini scripts read
`GEMINI_API_KEY` / `GOOGLE_API_KEY`; vision default model `gemini-2.5-flash`
(env `GEMINI_VISION_MODEL`).

| Script | Caller | Model | I/O |
|---|---|---|---|
| `claude_story.py` | `story_generator` | `claude-sonnet-4-6` (prompt caching) | `--prompt-file [--stream]` → text / `CHUNK:` |
| `gemini_story.py` | `story_generator` | `gemini-2.5-flash` | `--prompt-file [--stream]` → text / `CHUNK:` |
| `gemini_general.py` | App B (quiz/scene text) | `gemini-2.5-flash` | `--prompt-file --system …` → text |
| `gemini_analyze_image.py` | App B (object detect + gaze) | `gemini-robotics-er-1.6-preview` | `--image` → JSON `[{point,label,color,shape}]` |
| `gemini_validate_spatial.py` | App B (still spatial) | `gemini-2.5-flash` | `--image --obj-a --obj-b --relation` → JSON verdict |
| `gemini_validate_spatial_video.py` | App B (depth relations) | `gemini-2.5-flash` | `--video …` → JSON verdict |
| `gemini_recovery_question.py` | App B | `gemini-2.5-flash` | `--image --mode …` → JSON `{text,object}` |
| `gemini_wh_scene.py` | App B (WH questions) | `gemini-2.5-flash` | stdin JSON → JSON `{scene_description,questions}` |
| `gemini_conversation_followup.py` | App B (conversation) | `gemini-2.5-flash` | stdin JSON → JSON `{text}` |

`story_generator.py` routes by model prefix: `claude*` → `claude_story.py`,
`gemini*` → `gemini_story.py`, else → local Ollama (`localhost:11434`). Streaming
uses the `CHUNK:`-prefixed line protocol (`story_generator.py:909`).
`image_generator.py` → `image_generator_worker.py` (`gemini-2.5-flash-image`,
a.k.a. "nanobanana"); the first existing image in the output dir is passed as a
reference for style consistency.

---

## 9. `documents/` — what is read, and when

| File | Read by | When / how |
|---|---|---|
| `personas_rag.json` | `persona_rag.py:30` (App B) | 4 reference clinical personas. `match(age, disorder)` keyword-scores the best one; `build_story_context` / `build_question_context` inject a `--- PERSONA CONTEXT ---` block into story/quiz prompts. Keyword matching, **not** vector RAG. |
| `sar_system_prompt.md` | **Both apps** — `config/default.yaml` `system_prompt_file` (A) and `web_user_server.py:1676` (B) | "Socially Assistive Robot" framing prompt (child-safety / therapist-authority guidance). App A prepends it to the robot role; App B prepends it in `/api/generate_quiz_feedback` for child-friendly feedback phrasing. |
| `story for 4 to 7 years old/story_corpus.json` | `story_generator.py:380` (App B) | 10 fables + 30 WH-question stories used as **few-shot examples** in story prompts for ages 4–7. Not vector RAG. |
| `QTrobot.pdf` | App A RAG (`llamaindex_interface.py:116`) | Intended retrieval corpus for the robot's conversation **only if** the config `docs` path points here. By default `docs` points at the deployed path `/home/qtrobot/robot/code/.../documents`, so in this repo it is ingested only if `docs` is repointed. RAG loads `.pdf` only. |
| `QTrobot_research_papers.txt` | — | Not ingested (RAG `formats` default = `['.pdf']`). Reference material. |
| `story for kid.{odt,pdf}`, `story with wh questions.pdf` | — | Authoring/source material for `story_corpus.json`. Only picked up by App A RAG if `docs` repointed here (`.pdf` only). |

Only `personas_rag.json`, `sar_system_prompt.md`, and `story_corpus.json` are
actively read by code today. (A stray `documents/.~lock.story for kid.odt#`
LibreOffice lock file is junk — safe to delete.)

---

## 10. Configuration & environment

- **`config/default.yaml`** (App A, via ParamifyWeb) — params: `system_prompt_file`
  (→ `sar_system_prompt.md`), `docs`, `formats` (`['.pdf']`), `max_docs` (5),
  `llm` (`claude-sonnet-4-6`), `mem_store`, `paused`, `lang`, `disable_rag`,
  `enable_scene`, `hold_on`, `volume`, `role` (the full system prompt).
- **`src/.env`** (gitignored) — `GOOGLE_API_KEY`, `GEMINI_API_KEY`,
  `OPENAI_API_KEY`, `QWEN_API_KEY`, `ANTHROPIC_API_KEY`, AWS Polly creds,
  `TTS_ENGINE`/`TTS_SPEED`, robot connection vars.
- **`env.polly`** (gitignored) — Polly-specific overrides.

---

## 11. External AI services & models

| Service | SDK / access | Models | Used by |
|---|---|---|---|
| Anthropic Claude | `anthropic` / LlamaIndex | `claude-sonnet-4-6` | A (conversation), B (story, quiz) |
| Google Gemini | `google-genai` | `gemini-2.5-flash`, `gemini-2.5-flash-image`, `gemini-robotics-er-1.6-preview` | B (vision, validation, text, images) |
| OpenAI | `openai` | `gpt-4o-transcribe` (via `whisper.py`) | B (ASR) |
| Qwen / Alibaba DashScope | `dashscope` | `qwen3-tts-vd-realtime-…` | B (default TTS) |
| AWS Polly | `boto3` | neural, voice `Justin` | B (alt TTS) |
| NVIDIA Riva | `riva.client` (gRPC `localhost:50051`) | streaming ASR | A (ASR) |
| Ollama | LlamaIndex | `mxbai-embed-large` | A (embeddings) |

---

## 12. Data & storage layout

Flat JSON files on disk — no database.
- `src/users.json` — user records (keyed by username: age, gender, disorder,
  learning_goals, display_name; note `password_hash` is empty in practice).
- `src/user_data/<username>/` — `stories/` (`story_*.json` with
  pages/paragraphs/scenes/questions/takeaways), `story_images/<story>/`,
  `quizzes/{yes_no,wh}/` + `learned_answers.json`, `wh_scenes/`,
  `captured_scenes/`, `chat_history/`, `activities/` (DIY + conversation).
- `src/user_data/activity_images/` — shared DIY-generated images.
- Sessions store only `username`; `app.secret_key = os.urandom(24)` (regenerated
  each restart, so cookies don't survive a restart).

---

## 13. ROS interface

Both apps drive the same physical robot through ROS. Service/topic names observed
in the code:

**Services**
| Service | Used by | Purpose |
|---|---|---|
| `/qt_robot/behavior/talkText` | A, B(`qt`) | Speak text (built-in voice + visemes) |
| `/qt_robot/behavior/talkAudio` | B (`qwen`/`polly`) | Play an uploaded audio clip |
| `/qt_robot/speech/config` | A | Set TTS language/pitch/speed |
| `/qt_robot/setting/setVolume` | A, B | Speaker volume |
| `/qt_robot/setting/uploadBase64` | B (`qwen`) | Upload generated WAV to the robot |
| `/qt_robot/emotion/show` | A, B | Facial emotion |
| `qt_gesture_controller/gesture_play` | B | Play arm gesture |
| `/qt_respeaker_app/tuning/{set,get}` | A | Mic AGC tuning (gates VAD) |

**Topics**
| Topic | Direction | Purpose |
|---|---|---|
| `/camera/color/image_raw` | subscribe | Camera frames (A vision, B capture) |
| `/qt_respeaker_app/channel0` | subscribe | Microphone audio (Riva / whisper) |
| `/qt_respeaker_app/sound_direction` | subscribe | Sound DOA → active-speaker fusion (A) |
| `/qt_robot/head_position/command` | publish | Head IK target |
| `/qt_robot/{right,left}_arm_position/command` | publish | Arm IK targets |
| `sensor_msgs/JointState` | subscribe | Joint feedback (kinematics) |

App B also fires story emotion/gesture tags via `rostopic pub` subprocesses to
`/qt_robot/emotion/show` and `/qt_robot/gesture/play`.

---

## 14. Threading & concurrency

- **App A:** main loop thread (`BaseNode._run`) + ParamifyWeb Flask thread + Riva
  ASR event thread + ROS audio subscriber + vision/idle BaseNode threads +
  HumanTracking thread pool + per-utterance `command_interface.execute`
  `ThreadPoolExecutor`. State guarded by `state_lock`.
- **App B:** Flask request threads + lazy ROS subscriber thread + per-feature
  background threads (TTS streaming worker, red-card watcher, step-confirm) +
  subprocess workers.

---

## 15. Known issues & tech debt

1. **App A crashes on startup as committed:** `SceneDetection` is referenced at
   `qt_ai_data_assistant.py:125`, but its import is commented out (line 24) and
   `src/scene_detection.py` does not exist → `NameError` in `setup()`. The class
   must be restored (or the references removed) before App A can run.
2. **Hardcoded secret:** `tts_helper.py:72` contains a fallback DashScope API key
   in a git-tracked file. Revoke and remove it.
3. **`debug=True` + `host=0.0.0.0`** (`web_user_server.py:6535`) exposes the
   Werkzeug debugger on the network — an RCE risk on a deployed robot.
4. **Auth is cosmetic:** `password_hash` values in `users.json` are empty strings
   — it is a profile selector, not real authentication.
5. **Arity bug:** `_function_call_response_callback` calls `proccess_response`
   with 3 args (`qt_ai_data_assistant.py:178`) but the method takes 2 (`:322`) —
   `TypeError` on the `get_datetime` command path.
6. Residual commented-out code inside live files: Riva ASR block and IdleAttention
   in `web_user_server.py`; `/start_assistant` stub; duplicated
   camera-analysis route (`/api/camera_capture`); legacy DIY block handlers.

### Dead files removed in this cleanup (2026-06-17)
These had zero live references and were deleted:
- `src/user_interface.py`, `src/user_web_interface.py` — duplicate Flask login/dashboard UIs, superseded by `web_user_server.py`'s own auth.
- `src/riva_speech_recognition.py` — superseded by `riva_speech_recognition_vad.py` (only referenced in commented-out code).
- `src/version.py` — standalone `torch` version print, never imported.
- `scripts/gemini_validate_object.py` — superseded by `gemini_validate_spatial.py`; never invoked.

---

## 16. File index (live code)

```
src/
  qt_ai_data_assistant.py        App A entry (ROS node)
  command_interface.py           A: robot control + TTS (behavior/talkText)
  riva_speech_recognition_vad.py A: Riva ASR + Silero VAD
  llamaindex_interface.py        A: Claude + Ollama embeds + RAG (also optional in B)
  llm_prompts.py                 A: system prompt + wakeup classifier
  idle_attention.py              A: idle gaze
  human_presence_detection.py    A: DeepFace face detect/re-ID
  human_tracking.py              A+B: head-follow gaze
  kinematics/                    A+B: head/arm IK
  utils/                         A: base node, logger, sentence utils
  user_management.py             A+B: user store
  user_cli_interface.py          A: CLI login

  web_user_server.py             App B entry (Flask :8080)
  story_generator.py             B: story prompts -> subprocess workers
  persona_rag.py                 B: clinical persona matching
  image_generator.py             B: -> image_generator_worker.py
  image_generator_worker.py      B: subprocess (Gemini image gen)
  tts_helper.py                  B: Qwen/Polly/QT TTS
  wav_speed.py                   B: ffmpeg atempo (lazy-imported by tts_helper)
  whisper.py                     B: subprocess ASR (OpenAI gpt-4o-transcribe)

scripts/
  claude_story.py, gemini_story.py            story workers
  gemini_general.py                           general text LLM
  gemini_analyze_image.py                     object detect + point-to-gaze
  gemini_validate_spatial.py, *_video.py      spatial relation validation
  gemini_recovery_question.py                 recovery prompts
  gemini_wh_scene.py                          WH-question generation
  gemini_conversation_followup.py             conversation follow-ups
  autostart/                                  App A launch scripts

config/default.yaml              App A parameters / system prompt
documents/                       personas_rag.json, sar_system_prompt.md,
                                 story_corpus.json (+ RAG PDFs)
templates/                       App B HTML screens
```

> Note: a stale `ARCHITECTURE_diagram.png` still exists in this folder; it
> reflects the previous architecture and is no longer referenced.
