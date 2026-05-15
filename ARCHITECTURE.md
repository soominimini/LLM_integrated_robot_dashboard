# QTrobot AI Data Assistant — System Architecture

## 1. System Overview

A socially assistive robot platform for pediatric speech-language therapy, built on QTrobot hardware. The system has **two cooperating processes**:

- **Flask web server** (`src/web_user_server.py`, Python 3.8) — the active therapist-facing application. Runs the story builder, quiz authoring, scene game, recovery activity builder, and conversation flow builder. Drives the robot through the Robot Operating System service calls. **All large language model calls in this process go to Google Gemini** except for one narrow Ollama use (mishearing correction in quiz answers).
- **Robot Operating System brain** (`src/qt_ai_data_assistant.py`, Python 3.9) — a separate ROS node that runs the open-ended free-conversation mode. Uses Riva for speech recognition, Ollama for the language model, LlamaIndex for retrieval-augmented generation over local documents, and Moondream for camera scene captioning. **This process is not used by the web server.**

Both are governed by a layered ethical system prompt designed for child safety.

```
┌──────────────────────────────────────────────────────────────────┐
│                        THERAPIST / CHILD                         │
│                  (Speech, Gestures, Objects, Browser)             │
└──────────┬──────────────────────────────────────▲────────────────┘
           │ Audio / Visual Input                 │ Speech / Movement Output
           ▼                                      │
┌──────────────────────┐              ┌──────────────────────────┐
│   PERCEPTION LAYER   │              │    EXPRESSION LAYER      │
│  ┌────────────────┐  │              │  ┌────────────────────┐  │
│  │ Whisper        │  │  WEB         │  │ QTrobot Acapela    │  │
│  │ (gpt-4o-       │  │  PATH        │  │ text-to-speech     │  │
│  │  transcribe)   │  │              │  │ (default, mouth    │  │
│  │ Camera frames  │  │              │  │  sync via visemes) │  │
│  │ (Robot OS feed)│  │              │  │ Amazon Polly +     │  │
│  │ Red-card OpenCV│  │              │  │  Pylips lipsync    │  │
│  ├────────────────┤  │              │  │  (optional)        │  │
│  │ Riva speech    │  │  ROBOT OS    │  │ Robot OS gestures  │  │
│  │ Silero voice   │  │  PATH        │  │ Robot OS emotions  │  │
│  │  activity det. │  │              │  │ Head / arm inverse │  │
│  │ DeepFace face  │  │              │  │  kinematics        │  │
│  │ Moondream      │  │              │  │ HumanTracking gaze │  │
│  └────────────────┘  │              │  └────────────────────┘  │
└──────────┬───────────┘              └──────────▲───────────────┘
           │                                     │
           ▼                                     │
┌──────────────────────────────────────────────────────────────────┐
│                       COGNITION LAYER                            │
│                                                                  │
│  WEB PATH (web_user_server.py — almost all Google Gemini)        │
│  ┌──────────────────────┐  ┌────────────────────────────────┐   │
│  │ Gemini 2.5 Flash     │  │ Gemini 2.5 Flash Image         │   │
│  │  • story generation  │  │  • story illustrations         │   │
│  │  • quiz questions    │  │  • scene-game item cards       │   │
│  │  • quiz feedback     │  └────────────────────────────────┘   │
│  │  • emotion re-tagging│  ┌────────────────────────────────┐   │
│  │  • page splitting    │  │ Gemini Robotics ER 1.5 Preview │   │
│  │  • scene grouping    │  │  • object detection            │   │
│  │  • comprehension Qs  │  │    (scene game)                │   │
│  │  • follow-ups, etc.  │  └────────────────────────────────┘   │
│  └──────────────────────┘  ┌────────────────────────────────┐   │
│                            │ Ollama gemma4:e4b              │   │
│                            │  • mishearing correction for   │   │
│                            │    quiz answers (the only      │   │
│                            │    Ollama use in the web path) │   │
│                            └────────────────────────────────┘   │
│                                                                  │
│  ROBOT OS PATH (qt_ai_data_assistant.py)                         │
│  ┌──────────────────────┐  ┌────────────────────────────────┐   │
│  │ Ollama (config       │  │ LlamaIndex retrieval-augmented │   │
│  │  default gemma4:e4b) │  │  generation over documents/    │   │
│  │  conversation +      │  │ Embeddings: mxbai-embed-large  │   │
│  │  function calling    │  │ Per-user ChatMemoryBuffer      │   │
│  └──────────────────────┘  └────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────┘
           │                                     ▲
           ▼                                     │
┌──────────────────────────────────────────────────────────────────┐
│                     ORCHESTRATION LAYER                          │
│  ┌──────────────────────────┐  ┌─────────────────────────────┐  │
│  │ QTAIDataAssistant        │  │ Flask Web Server             │  │
│  │ (Robot OS node)          │  │ (Therapist / child interface)│  │
│  │ State: IDLE → LISTENING  │  │ Routes: /api/*, page renders │  │
│  │  → PROCESSING → RESPOND  │  │ Session-based authentication │  │
│  └──────────────────────────┘  └─────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
           │                                     │
           ▼                                     ▼
┌──────────────────────────────────────────────────────────────────┐
│                        DATA LAYER                                │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────────────┐  │
│  │ User profiles │  │ Chat memory  │  │ Quizzes / stories /   │  │
│  │ (users.json)  │  │ (per-user)   │  │ activities / learned  │  │
│  └──────────────┘  └──────────────┘  └───────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

---

## 2. Entry Points

### 2.1 Robot Main Process — `qt_ai_data_assistant.py`

**Runtime**: Python 3.9 (required for Gemini API)
**Startup**: `scripts/autostart/start_qt_ai_data_assitant.sh`

Startup sequence:
1. Source ROS environment (`/home/qtrobot/robot/autostart/qt_robot.inc`)
2. Wait for ROS services (`/qt_robot/emotion/show`, `/qt_robot/gesture/play`, `/qt_robot/head_position/command`)
3. Wait ~100s for Riva ASR Docker container
4. Activate `.venv39` virtual environment
5. Run `python3.9 src/qt_ai_data_assistant.py`

### 2.2 Web Server — `web_user_server.py`

**Runtime**: Python 3.8 (Flask)
**Port**: 6060 (default)
**Startup**: Manual or via separate autostart script
- Loads `.env` via `python-dotenv`
- Initializes: UserManager, StoryGenerator, TTSHelper, ImageGenerator

### 2.3 Configuration — `config/default.yaml`

```yaml
parameters:
  system_prompt_file: "documents/sar_system_prompt.md"
  docs: "documents/"            # RAG document folder
  formats: [".pdf"]             # Allowed document formats
  max_docs: 5                   # Max documents for RAG
  llm: "llama3.1"              # Default LLM model
  lang: "en-US"                # Default language
  disable_rag: false            # Toggle RAG
  enable_scene: false           # Toggle camera scene processing
  volume: 33                    # Speaker volume (0-100)
  role: |                       # Fallback system prompt (if file not found)
    You are a humanoid social robot assistant named "QTrobot"...
```

---

## 3. Module Reference

### 3.1 Core Orchestration

#### `src/qt_ai_data_assistant.py` — Robot Brain

**Class**: `QTAIDataAssistant(ParamifyWeb, BaseNode)`

**State Machine**:
```
IDLE ──(speech detected)──► LISTENING ──(ASR complete)──► PROCESSING
  ▲                                                           │
  │                                                     (LLM response)
  │                                                           │
  └──(response complete)──── RESPONDING ◄─────────────────────┘
                                │
                          (user says "hold on")
                                │
                              PAUSED ──(user says "start conversation")──► IDLE
```

**Key Methods**:
| Method | Purpose |
|--------|---------|
| `setup()` | Initialize all subsystems, authenticate user, load system prompt |
| `process()` | Main loop: listen → recognize → respond → reset |
| `_asr_callback(text, lang)` | Process recognized speech through LLM |
| `proccess_response(user, response_stream)` | Stream LLM response, execute commands |
| `_function_call_response_callback(function, result)` | Handle LLM tool calls |
| `_reset_chat_engine()` | Rebuild ChatWithRAG with user-specific memory |
| `_set_language(language)` | Change TTS + ASR language |

**System Prompt Loading** (setup, lines 78-88):
```
1. Check self.parameters.role (from config/web UI)
2. If empty → load documents/sar_system_prompt.md
3. Prepend SAR prompt + append ConversationPrompt['system_role']
4. Fallback to ConversationPrompt['system_role'] alone
```

#### `src/command_interface.py` — Command Execution

**Class**: `CommandInterface`

Executes LLM-generated JSON commands via ROS services.

**Command Registry**:
| Command | ROS Service / Action |
|---------|---------------------|
| `talk` | `/qt_robot/behavior/talkText` |
| `look_at_xyz` | IK solver → head position publisher |
| `look_at_pixel` | Pixel→3D transform → IK → head |
| `point_at_pixel` | Pixel→3D → arm IK |
| `point_at_xyz` | 3D → arm IK |
| `pause_interaction` | Set hold_on flag |
| `resume_interaction` | Clear hold_on flag |
| `forget_conversation` | Clear chat memory |
| `get_datetime` | Return current datetime |
| `set_language` | Update TTS + ASR language |

**Execution Model**: `ThreadPoolExecutor` — commands run in parallel (e.g., talk + gesture simultaneously).

---

### 3.2 Language & Cognition

#### `src/llamaindex_interface.py` — Local language model + retrieval engine

**Class**: `ChatWithRAG`

**Where it is used**:
- **Robot Operating System path** (`qt_ai_data_assistant.py`) — primary user; powers free-conversation mode with retrieval-augmented generation over `documents/`.
- **Web server path** (`web_user_server.py`) — used in **one** narrow place only: the mishearing-correction helper `_ensure_intent_llm()` at `src/web_user_server.py:292`, which checks whether a Whisper transcript likely intended an expected quiz answer. This is the only place the web server touches Ollama.

**Components**:
- **Language model**: Ollama (local). Web server intent helper hardcodes `gemma4:e4b`. Robot Operating System path uses whatever `parameters.llm` is set to in `config/default.yaml` (current default: `gemma4:e4b`).
- **Embeddings**: `OllamaEmbedding("mxbai-embed-large:latest")` — only loaded when retrieval-augmented generation is enabled, which the web server intent helper disables (`disable_rag=True`).
- **Document loader**: `SimpleDirectoryReader` (PDF, TXT, MD, DOCX)
- **Index**: `VectorStoreIndex` (in-memory)
- **Memory**: `ChatMemoryBuffer` → `SimpleChatStore` (persisted per user)
- **Chat engine**: `CustomChatEngine` (extends `ContextChatEngine`) with camera context injection

**Key Methods**:
| Method | Purpose |
|--------|---------|
| `get_stream_response(text, user_id)` | Stream LLM tokens, yield complete sentences |
| `get_response(text)` | Non-streaming response |
| `get_raw_chat(system, user)` | Direct LLM call (no memory/RAG) |
| `update_camera_feed(scene_desc)` | Inject scene context into chat engine |
| `clear_memmory()` | Reset conversation history |
| `close()` | Persist memory to user's chat_store.json |

**RAG Flow**:
```
User Query
  → VectorStoreIndex.as_retriever() retrieves relevant document chunks
  → ContextChatEngine combines: system prompt + document context + camera context + query
  → Ollama LLM generates response
  → Response streamed token-by-token
  → split_into_sentences() yields complete sentences for TTS
```

#### `src/llm_prompts.py` — Prompt Definitions

**ConversationPrompt**: Default robot personality and response guidelines
- Keep responses short (1-2 sentences)
- Plain text only, no formatting
- Multi-language support
- Special JSON commands for: pause, forget, set_language

**WakeupPrompt**: Detects if user wants to start conversation (for PAUSED state)

---

### 3.3 Speech I/O

#### `src/whisper.py` — Speech recognition for the web server (active)

**Type**: Subprocess invoked by the Flask server (`web_user_server.py:791,831`)
**Backend**: OpenAI `gpt-4o-transcribe`

This is the speech recognizer for **every** voice interaction in the web server: quiz answers, conversation flows, recovery activities, and any "press to talk" surface in the templates. Riva is **not** imported by `web_user_server.py`.

**Parameters** (environment-configurable):
| Param | Default | Env Var |
|-------|---------|---------|
| Silence threshold | 0.008 RMS | `WHISPER_SILENCE_THRESHOLD` |
| Silence duration | 1.5s | `WHISPER_SILENCE_DURATION` |
| Max record time | 15s | `WHISPER_MAX_RECORD` |
| Stream interval | 2.0s | `WHISPER_STREAM_INTERVAL` |

**Output Protocol**:
```
PARTIAL:<intermediate text>    (emitted every stream_interval)
FINAL:<final transcription>    (emitted once at end)
```

**Language**: Passed via `--language` command-line argument (ISO-639-1 code extracted from config).

#### `src/riva_speech_recognition_vad.py` — Speech recognition for the Robot Operating System path (legacy / not in web server)

**Class**: `RivaSpeechRecognitionSilero`

Imported only by `src/qt_ai_data_assistant.py:23`. The Flask web server does **not** import or call this module.

- **Speech recognizer**: NVIDIA Riva (gRPC, Docker container)
- **Voice activity detection**: Silero (confidence threshold 0.6)
- **Audio**: 16 kHz, mono, from Robot Operating System topic `/qt_respeaker_app/channel0`
- **Languages**: en-US, en-GB, ar-AR, de-DE, es-ES, fr-FR, hi-IN, it-IT, ja-JP, ru-RU, ko-KR, pt-BR, zh-CN

**Event flow** (free-conversation mode only):
```
Audio chunks from Robot Operating System microphone topic
  → Silero voice activity detection
  → Event.RECOGNIZING fired → robot starts tracking the speaker
  → Riva processes the audio stream
  → returns recognized text + language
```

#### `src/tts_helper.py` — Text-to-speech

**Class**: `TTSHelper`

**Engines**:
| Engine | Service | Mouth synchronization |
|--------|---------|------------|
| `qt` (default) | Robot Operating System service `/qt_robot/behavior/talkText` (Acapela) | Built-in viseme support |
| `polly` | Amazon Polly → SSH upload → robot playback | Via Pylips (socketio) |

**Joint Limits** (for movement during speech):
```
Head:      HeadYaw [-90, 90], HeadPitch [-15, 25]
Right Arm: ShoulderPitch [-140, 140], ShoulderRoll [-75, 7], ElbowRoll [-90, -7]
Left Arm:  ShoulderPitch [-140, 140], ShoulderRoll [-75, 7], ElbowRoll [-90, -7]
```

---

### 3.4 Vision & Perception

The visual stack is split between the active web server path and the Robot Operating System path.

#### Active in the web server

##### `scripts/gemini_analyze_image.py` — Object detection (web server: scene game)

**Model**: `gemini-robotics-er-1.5-preview`
**Runtime**: Python 3.9 (subprocess)

- **Input**: image file path (`--image` argument)
- **Output**: JSON array of detected objects with normalized point coordinates
- **Format**: `[{"point": [y, x], "label": "object_name"}]` (coordinates 0–1000)
- **Prompt**: "Point to no more than 1 item a person is holding in the image."
- **Called by**: scene-game answer route (`/api/scene_game/answer`) and `/api/camera_capture` in `web_user_server.py`

##### Red-card detection (`web_user_server.py:678`)

OpenCV thresholding in hue/saturation/value color space, used by the conversation flow builder and the recovery activity follow-up loop.

- Hue ranges: `[0, 10] ∪ [165, 180]` (covers wraparound red)
- Saturation: > 100, value: > 80
- Triggers when red pixel ratio exceeds 3% of the frame

##### Camera frames

The web server pulls frames directly from a Robot Operating System camera topic via `_get_ros_frame()` and serves them on `/api/camera_frame`.

#### Active in the Robot Operating System path only (not used by the web server)

##### `src/human_presence_detection.py` — Face detection

**Class**: `HumanPresenceDetection`

- **Backend**: DeepFace + RetinaFace
- **Input**: Robot Operating System camera topic
- **Output**: per-person face bounding box, 3D position, emotion, embedding
- **Features**: temporal filtering, external voice-activity-detection trigger, callback-based

##### `src/human_tracking.py` — Gaze following

**Class**: `HumanTracking`

- **Input**: `HumanPresenceDetection` callbacks
- **Output**: smooth head movement that follows the active speaker
- **Features**: person identity tracking, absence memory (forgets after 10 minutes)

> The web server *imports* `HumanTracking` lazily inside a `try/except` (`web_user_server.py:48–50`) and uses it during story read-aloud to follow the listener's face. Face detection itself is not initialized by the web server; tracking falls back gracefully when unavailable.

##### `src/idle_attention.py` — Idle gaze

**Class**: `IdleAttention`

- Random gaze at detected persons or random directions
- Prevents staring; creates natural "looking around" behavior
- Active only while the Robot Operating System brain is in the IDLE state

##### `src/scene_detection.py` — Scene captioning (Moondream)

**Class**: `SceneDetection`

Used **only** by `qt_ai_data_assistant.py` (free-conversation mode), and only when the `enable_scene` parameter is true. The Flask web server does not import this module.

- **Model**: Moondream (run locally via Ollama)
- **Input**: camera frames at configurable framerate (default 0.1 frames/second)
- **Output**: scene description text → injected into the conversation language model context
- **Prompt**: "Describe in details what you see. If you see people, also describe how they dressed and what they carry."

---

### 3.5 User & Data Management

#### `src/user_management.py` — Multi-User System

**Class**: `UserManager`

| Method | Purpose |
|--------|---------|
| `register_user(username, age, password)` | Create new user |
| `authenticate_user(username)` | Login user |
| `get_current_user()` | Return active user info |
| `get_user_mem_store_path()` | Path to user's chat memory |

#### `src/kinematics/kinematic_interface.py` — Robot Kinematics

**Class**: `QTrobotKinematicInterface`

**ROS Publishers**:
- `/qt_robot/head_position/command` (Float64MultiArray)
- `/qt_robot/right_arm_position/command` (Float64MultiArray)
- `/qt_robot/left_arm_position/command` (Float64MultiArray)

**Home Positions**: Head [0, 0], Right arm [-90, -55, -35], Left arm [90, -55, -35]

#### `src/utils/`

- `base_node.py` — Abstract threaded component with pause/resume
- `utils.py` — `split_into_sentences()`, `get_utc_timestamp()`

---

### 3.6 Content Generation

#### Story Pipeline — Layered View

The story system is a stack of independent layers. Each layer consumes the previous layer's output and adds one concern (length safety, gesture tagging, page breaks, illustration, comprehension). The robot only ever sees the bottom layer.

```
┌────────────────────────────────────────────────────────────────────────────┐
│ INPUT                                                                      │
│   child_name, age, gender, topics[], goals (clinician), disorder (profile) │
└─────────────────────────────────┬──────────────────────────────────────────┘
                                  ▼
╔════════════════════════════════════════════════════════════════════════════╗
║ LAYER 1 — GENERATION                          src/story_generator.py       ║
║   Compose prompt → call Gemini → receive raw story                         ║
║                                                                            ║
║   _build_prompt():                                                         ║
║     _get_age_tier(age)              → MASTER_TEMPLATE  (3, 7–8, 9–12)     ║
║                                     or WH_MASTER_TEMPLATE (4–6)            ║
║     _get_theme_guidance(topics)     → setting / obstacle / resolution      ║
║                                       / vocabulary_focus  (merged)         ║
║     _format_goals_section(goals)    → clinician goals + 4 default goals    ║
║     PersonaRAG.build_story_prompt_fragment(age, disorder)                  ║
║                                     → matched persona's goals / interests  ║
║                                       / constraints                        ║
║     _load_wh_examples(corpus, n=2)  → few-shot for ages 4–6 only           ║
║                                       (story_corpus.json)                  ║
║                                                                            ║
║   scripts/gemini_story.py  (Python 3.9 subprocess, gemini-2.5-flash)       ║
║       blocking → returns full text                                         ║
║       streaming → emits "CHUNK:<line>" per token group                     ║
║                                                                            ║
║   RAW OUTPUT (still tagged with inline [gesture:NAME]/[emotion:QT/…]):     ║
║     ** Title **                                                            ║
║     <title>                                                                ║
║     <story body with inline tags>                                          ║
║     ** End **                                                              ║
║     ** Takeaways **           (only when tier requires_takeaways)          ║
║     - <lesson 1>                                                           ║
║     - <lesson 2>                                                           ║
║     ** Questions **           (only when tier is wh_question_format)       ║
║     1. <WH-question whose answer is verbatim in the story>                 ║
║     ** Explanation of the output **                                        ║
║     <how story matches topics + goals>                                     ║
╚════════════════════════════════════════════════════════════════════════════╝
                                  │ raw story text
                                  ▼
╔════════════════════════════════════════════════════════════════════════════╗
║ LAYER 2 — STRICT RULE / SAFETY NET            api_save_story (server)      ║
║   Enforce the contract the prompt asked for, in case the LLM ignored it.   ║
║                                                                            ║
║   _extract_story_title()      → pulls "** Title **" block into metadata    ║
║   Takeaways regex             → recovers ** Takeaways ** bullets           ║
║   StoryGenerator.get_word_range_for_age(age) → (min_words, max_words)      ║
║   IF body.split() > max_words:                                             ║
║       StoryGenerator.shorten_story(body, age, child_name)                  ║
║         re-prompts Gemini with SHORTEN_TEMPLATE                            ║
║         preserves plot + names + inline tags                               ║
║         drops adjectives / side dialogue first                             ║
╚════════════════════════════════════════════════════════════════════════════╝
                                  │ length-safe story
                                  ▼
╔════════════════════════════════════════════════════════════════════════════╗
║ LAYER 3 — TAGGING (gesture + emotion)         web_user_server._apply_…     ║
║   Many generators miss or invent emotion tags. A second Gemini pass        ║
║   re-tags the story word-for-word, then positions are validated.           ║
║                                                                            ║
║   _apply_emotion_tags_with_gemini(story)                                   ║
║       prompt: "return SAME story word-for-word with correct tags"          ║
║       allowed gestures: hi, bye, nodding-yes, clapping, hoora, …           ║
║       allowed emotions: QT/happy, QT/sad, QT/surprised, QT/afraid,         ║
║                         QT/angry, QT/calm, QT/shy                          ║
║       guard: reject re-tagged output if <95% of original length            ║
║                                                                            ║
║   _validate_tag_positions(story)                                           ║
║       snap any tag that landed mid-word / mid-clause to the                ║
║       nearest sentence boundary (start, end, or after , ! ?)               ║
╚════════════════════════════════════════════════════════════════════════════╝
                                  │ correctly-tagged story
                                  ▼
╔════════════════════════════════════════════════════════════════════════════╗
║ LAYER 4 — SPLITTING (pages + paragraphs)      web_user_server._split_…     ║
║   Group sentences into "pages" the robot will read one at a time, while    ║
║   keeping paragraph structure for image grouping.                          ║
║                                                                            ║
║   _split_story_into_pages(story, age)         (Gemini, JSON array out)     ║
║       sents/page target:  ≤4 → 1–2 | ≤6 → 2–3 | 7+ → 3–5                  ║
║       priority: scene/context coherence > sentence count                   ║
║       hard rule: preserve [gesture/emotion] tags verbatim                  ║
║       fallback (no LLM): paragraph-aware splitter                          ║
║                                                                            ║
║   _reinject_tags_into_pages(original, pages)                               ║
║       page splitter sometimes drops tags — re-attach each tag inline       ║
║       before the same sentence it preceded in the original                 ║
║                                                                            ║
║   _split_into_paragraphs(story)               (blank-line separator)       ║
║   _map_pages_to_paragraphs(pages, paragraphs)                              ║
║       sequential substring match — a page can only advance forward         ║
╚════════════════════════════════════════════════════════════════════════════╝
                                  │ pages[], paragraphs[], page_to_paragraph[]
                                  ▼
╔════════════════════════════════════════════════════════════════════════════╗
║ LAYER 5 — SCENE & IMAGE GENERATION            web_user_server / image_gen  ║
║   Pick where to draw illustrations (1 image per visual moment, not per     ║
║   page) and generate them with style consistency.                          ║
║                                                                            ║
║   _identify_story_scenes(paragraphs)          (Gemini, JSON out)           ║
║       returns scenes[] + paragraph_to_scene[]                              ║
║       bias: 1 paragraph = 1 scene; merge only on identical visual          ║
║   compose page_to_scene = paragraph_to_scene[ page_to_paragraph[p] ]       ║
║                                                                            ║
║   ImageGenerator.generate_story_scene_image() per scene                    ║
║       Path A: direct google-genai SDK (if Py3.9 in-process)                ║
║       Path B: subprocess → src/image_generator_worker.py (.venv39)         ║
║       style: soft round shapes, pastel palette, thick outlines             ║
║       reference: first generated image fed back as visual guide for the    ║
║                  rest, so all scenes share style                           ║
║                                                                            ║
║   output: user_data/<user>/story_images/<file>/story_scene_NNN_*.png       ║
╚════════════════════════════════════════════════════════════════════════════╝
                                  │ scenes[], page_to_scene[], PNGs on disk
                                  ▼
╔════════════════════════════════════════════════════════════════════════════╗
║ LAYER 6 — COMPREHENSION QUESTIONS             web_user_server._generate_…  ║
║   Build the post-reading quiz that appears at the end of the story.        ║
║                                                                            ║
║   _generate_story_questions(story, age, name)         (Gemini, JSON out)   ║
║       3 MCQs scaled to age:                                                ║
║         young   → 1 main_idea + 2 detail                                   ║
║         middle  → 1 main_idea + 2 detail + 1 inference                     ║
║         older   → main_idea + detail + inference + cause/effect            ║
║       fallback: hard-coded generic questions on Gemini failure             ║
║                                                                            ║
║   _generate_takeaway_questions(takeaways, story, age, name)  (ages 7+)     ║
║       one MCQ PER takeaway: takeaway = correct answer,                     ║
║       distractors generated to be believable-but-wrong-for-this-story      ║
╚════════════════════════════════════════════════════════════════════════════╝
                                  │ questions[], takeaways[]
                                  ▼
╔════════════════════════════════════════════════════════════════════════════╗
║ LAYER 7 — PERSISTENCE                                                      ║
║   user_data/<user>/stories/story_<ts>.json                                 ║
║   {                                                                        ║
║     story, metadata{title, child_name, age, age_tier, target_word_range}, ║
║     pages[], paragraphs[], scenes[],                                       ║
║     page_to_scene[], page_to_paragraph[], paragraph_to_scene[],            ║
║     questions[{question,type,correct_answer,wrong_answers[]}],             ║
║     takeaways[]                                                            ║
║   }                                                                        ║
╚════════════════════════════════════════════════════════════════════════════╝
                                  │
                                  ▼
╔════════════════════════════════════════════════════════════════════════════╗
║ LAYER 8 — READ-ALOUD / OUTPUT TO ROBOT        /api/speak_sentence          ║
║   Triggered once per page on the read_story.html UI.                       ║
║                                                                            ║
║   _split_page_into_segments(page)                                          ║
║       break a page into (text, gestures[], emotions[]) tuples              ║
║       at every [gesture:…] / [emotion:…] boundary                          ║
║                                                                            ║
║   For each segment:                                                        ║
║     _play_tags(gestures, emotions)                                         ║
║       → ROS /qt_robot/gesture/play                                         ║
║       → ROS /qt_robot/emotion/show                                         ║
║     For each sentence in _split_into_sentences(segment.text):              ║
║       _with_asr_suspended( tts_helper.speak_story(sentence, lang) )        ║
║         → ROS /qt_robot/behavior/talkText  (QT) — with viseme mouth sync   ║
║         → OR Polly + SSH upload + talkAudio   (POLLY_*) — Pylips lipsync   ║
║                                                                            ║
║   Concurrently: HumanTracking follows the listener's face during reading.  ║
╚════════════════════════════════════════════════════════════════════════════╝
                                  │
                                  ▼
                       🤖 robot speaks the page,
                       gestures fire at tagged sentences,
                       facial emotion changes per tag,
                       scene illustration is shown in the UI
```

#### `src/story_generator.py` — Therapeutic Story Generation

**Class**: `StoryGenerator`
**Default LLM**: Gemini (`gemini-2.5-flash`) invoked via Python 3.9 subprocess (`scripts/gemini_story.py`). Falls back to local Ollama when `llm_model` does not start with `gemini` (the `_is_ollama_model()` switch hits `http://localhost:11434/api/generate`).

**Age Tiers** (defined in `StoryGenerator.AGE_TIERS`):
| Tier | Ages | Word Range | Format | Takeaways |
|------|------|-----------|--------|-----------|
| `early_preschool` | 3 | 50–100 | 3–5 word sentences, repeated patterns, 2–3 characters max | No |
| `wh_question_format` | 4–6 | 40–90 | 3–4 concrete present-tense sentences + 5–7 WHO/WHAT/WHERE questions whose answers appear verbatim | No |
| `early_school_age` | 7–8 | 250–400 | Three-act, emotional vocabulary, 4–5 characters | Yes (2–3) |
| `school_age` | 9–12 | 400–600 | Subordinate clauses, internal conflict, figurative language | Yes (2–3) |

**Prompt Assembly** (`_build_prompt()`):
```
age tier → theme guidance → therapy goals → persona context → output format
```
- **`MASTER_TEMPLATE`** — generic three-act narrative (tiers 3, 7–8, 9–12)
- **`WH_MASTER_TEMPLATE`** — short vignette + WH-questions (ages 4–6); few-shot examples are loaded from `documents/story for 4 to 6 years old/story_corpus.json` by `_load_wh_examples()` (preferring examples whose `setting` matches the requested topics)
- **`TAKEAWAYS_PROMPT_BLOCK`** — appended when the tier has `requires_takeaways: True`; instructs the model to emit a `** Takeaways **` section after `** End **`
- **Inline robot tags** — every emotional beat is tagged `[gesture:NAME]` / `[emotion:QT/…]` so the reader can fire gestures and facial expressions at the right sentence

**Theme Guidance** (`THEME_GUIDANCE`): per-topic `setting / obstacle / resolution / vocabulary_focus` blocks for `season`, `school`, `family`, `friends`, `animals`, `adventure`. Multiple selected topics are merged; unmatched topics fall back to a generic default.

**Persona Injection**: `_persona_context_for(username, age, kind="story")` calls `PersonaRAG.build_story_prompt_fragment(age, disorder)` — see `src/persona_rag.py`. The matched persona's therapy goals, structured language targets, interests, and constraints are inlined under a `--- PERSONA CONTEXT ---` block.

**Public API**:
| Method | Purpose |
|--------|---------|
| `generate_story(child_name, age, gender, topics, goals, persona_context, custom_prompt)` | Blocking; returns `{success, story, metadata}` |
| `generate_story_stream(...)` | SSE-style chunk generator — Gemini emits `CHUNK:<line>` lines; Ollama yields `response` deltas |
| `get_word_range_for_age(age)` | Returns `(min, max)` for the tier — used by `api_save_story` to gate the shortener |
| `shorten_story(body, age, child_name)` | Safety net: re-prompts the LLM with `SHORTEN_TEMPLATE` to rewrite the body within the cap, preserving inline tags |

**Legacy variant**: `src/story_generator_ashley.py` is an older Ollama-based prototype with hard-coded school/nature templates. It is **not** imported by `web_user_server.py` and is kept only for reference.

#### `scripts/gemini_story.py` — Gemini Story Worker (Python 3.9)

Subprocess invoked by `StoryGenerator._run_gemini()`. Accepts the prompt via `--prompt-file` (temp file, avoids shell escaping) and `--model`. With `--stream`, prints each chunk on its own line prefixed `CHUNK:`. The system instruction frames Gemini as a "clinical storyteller for pediatric speech-language therapy" and forbids preamble.

#### Post-Generation Pipeline (in `web_user_server.py` `api_save_story`)

After `StoryGenerator` returns raw text, `/api/save_story` runs a multi-stage pipeline before persisting:

```
raw story
  → extract takeaways (** Takeaways ** block) + title (** Title **)
  → shorten_story() if body > tier max_words
  → _apply_emotion_tags_with_gemini()       # Gemini Flash re-tags emotions/gestures
  → _validate_tag_positions()                # snap mid-word tags to sentence boundaries
  → _split_story_into_pages(story, age)      # Gemini groups sentences into age-appropriate pages
  → _reinject_tags_into_pages()              # tag positions preserved across split
  → _split_into_paragraphs() + _identify_story_scenes()
                                             # Gemini decides which paragraphs share a visual scene
  → _map_pages_to_paragraphs() → page_to_scene
  → _generate_story_questions() (Gemini)     # 3 comprehension MCQs (main_idea / detail / inference)
  → _generate_takeaway_questions() (Gemini)  # 1 MCQ per takeaway (ages 7+)
  → persist {story, metadata, pages, paragraphs, scenes, page_to_scene,
             page_to_paragraph, paragraph_to_scene, questions, takeaways}
  → ImageGenerator.generate_story_scene_image() per scene
```

Page sentence-count target scales with age (1–2 for ≤4, 2–3 for ≤6, 3–5 for 7+). Scene merging is biased toward "1 paragraph = 1 scene" — paragraphs share an image only when same setting + same characters + similar action.

#### `src/image_generator.py` — Story Illustration

**Class**: `ImageGenerator`
**Model**: `gemini-2.5-flash-image` (env override: `GOOGLE_IMAGE_MODEL`)

- Generates children's book-style illustrations (soft shapes, pastel palette, thick outlines, minimal shading)
- **Dual-path execution**:
  - Path A: direct SDK call when `google-genai` is importable (Python 3.9+)
  - Path B: subprocess fallback when running under Python 3.8 — spawns `.venv39/bin/python src/image_generator_worker.py` with a JSON payload on stdin
- **Style consistency**: when the output dir already contains a PNG, that image is passed back as a reference so subsequent scenes match the established style
- `generate_story_scene_image(sentence, story_context, …)` is the entry point used by `api_save_story`

#### `src/image_generator_worker.py` — Python 3.9 Image Worker

Reads a JSON payload `{prompt, output_path, reference_image}` from stdin, calls `client.models.generate_content(model="gemini-2.5-flash-image", contents=[prompt, optional_reference_image])`, saves the first inline image part to `output_path`, and prints the path on stdout.

---

## 4. Web Server API Reference

### 4.1 Authentication
| Method | Route | Purpose |
|--------|-------|---------|
| POST | `/api/register` | Register new user |
| POST | `/api/login` | Authenticate |
| POST | `/api/logout` | Clear session |
| GET | `/api/current_user` | Get logged-in user |

### 4.2 User Profile
| Method | Route | Purpose |
|--------|-------|---------|
| POST | `/api/update_profile` | Update age, gender, learning goals |
| GET | `/api/users` | List all users |
| GET | `/api/user_stats` | User statistics |

### 4.3 Story Generation & Reading
| Method | Route | Purpose |
|--------|-------|---------|
| POST | `/api/generate_story` | Generate story (blocking) |
| POST | `/api/generate_story_stream` | Generate story (SSE streaming) |
| POST | `/api/save_story` | Save story + generate images |
| GET | `/api/get_user_stories` | List user's stories |
| POST | `/api/get_specific_story_details` | Fetch story details |
| GET | `/api/get_story_sentences` | Get story sentences |
| GET | `/api/get_sentence_image` | Get paragraph image |
| POST | `/api/speak_sentence` | Robot reads sentence aloud |
| GET | `/read_story/<filename>` | Story reading page |

### 4.4 Quiz System
| Method | Route | Purpose |
|--------|-------|---------|
| POST | `/api/generate_quiz` | LLM generates quiz questions |
| POST | `/api/save_quiz` | Save quiz (split yes_no/wh folders) |
| GET | `/api/get_saved_quiz?type=` | Load saved quizzes by type |
| POST | `/api/teach_quiz_answer` | Save user-taught alternative answers |
| POST | `/api/generate_quiz_feedback` | Pre-generate varied feedback phrases |
| GET | `/quiz_generation` | Quiz builder page |
| GET | `/educational_quiz` | Quiz playing page |

### 4.5 DIY Activity Builder (Recovery Strategy Builder)
| Method | Route | Purpose |
|--------|-------|---------|
| GET | `/api/get_custom_games` | List saved activities (includes `activity_type`) |
| POST | `/api/activity/save` | Save activity (JSON blocks) |
| POST | `/api/activity/load_saved` | Load saved activity |
| POST | `/api/activity/prepare` | Prepare for execution |
| POST | `/api/activity/run_saved` | Execute activity (with therapist step-by-step confirmation) |
| POST | `/api/activity/stop` | Stop running activity |
| POST | `/api/activity/test` | Test blocks on robot |
| POST | `/api/activity/delete` | Delete a saved activity |
| GET | `/api/activity/step_status` | Poll step-by-step execution state |
| POST | `/api/activity/confirm_step` | Therapist confirms to proceed to next step |

### 4.5.1 Recovery Camera & Question Generation
| Method | Route | Purpose |
|--------|-------|---------|
| POST | `/api/recovery/generate_question` | Capture camera + Gemini generates toy/child question |

### 4.5.2 Conversation Flow
| Method | Route | Purpose |
|--------|-------|---------|
| POST | `/api/conversation/wait_for_turn` | Listen for child speech + red card detection + generate follow-up |
| GET | `/api/conversation/check_red_card` | Quick red card visibility check |

### 4.6 Camera & Vision
| Method | Route | Purpose |
|--------|-------|---------|
| GET | `/api/camera_frame` | Get latest camera JPEG |
| POST | `/api/camera_capture` | Capture + Gemini ER object detection |
| GET | `/api/scene_start` | Start scene game |
| POST | `/api/scene_game_new_round` | Generate scene question |
| POST | `/api/scene_game_answer` | Check answer |

### 4.7 Robot Control
| Method | Route | Purpose |
|--------|-------|---------|
| POST | `/api/robot_gesture` | Play gesture + emotion |
| POST | `/api/speech_recognize` | Whisper ASR (blocking) |
| POST | `/api/speak_sentence` | TTS a sentence |
| POST | `/api/head_position` | Move robot head |
| POST | `/api/volume_settings` | Set speaker volume |
| POST | `/api/human_tracking_start` | Start person tracking |
| POST | `/api/human_tracking_untrack` | Stop tracking |

### 4.8 Pages
| Route | Template | Purpose |
|-------|----------|---------|
| `/` | `index.html` | Login / registration / game selection |
| `/dashboard` | `dashboard.html` | Main dashboard |
| `/play` | `play_games.html` | Game selection |
| `/educational_quiz` | `educational_quiz.html` | Play quizzes |
| `/quiz_generation` | `quiz_generation.html` | Build quizzes |
| `/read_story/<file>` | `read_story.html` | Story reading |
| `/play_scene` | `play_scene.html` | Object detection game |
| `/builder` | `diy_builder.html` | Recovery strategy builder |
| `/conversation_builder` | `conversation_builder.html` | Conversation flow builder |
| `/my_games` | `my_games.html` | Saved activities & conversations |
| `/select_toy` | `select_toy.html` | Toy selection |

---

## 5. External Services & Models

### 5.1 Language models

The web server is almost entirely a Gemini application; the Robot Operating System brain is a local Ollama application.

#### Web server (`web_user_server.py`) — active

| Model | Provider | Used for | Called from |
|-------|----------|----------|-------------|
| `gemini-2.5-flash` | Google Gemini | Story generation (age-tiered, themed, with-question format) | `scripts/gemini_story.py` → `StoryGenerator` → `/api/generate_story[_stream]` |
| `gemini-2.5-flash` | Google Gemini | Story post-processing: emotion/gesture re-tagging, page splitting, scene grouping, comprehension and takeaway questions | `scripts/gemini_general.py` (via `_gemini_generate()`) → `/api/save_story`, `/api/get_story_sentences` |
| `gemini-2.5-flash` | Google Gemini | Quiz generation and pre-generated quiz feedback phrases | `_GeminiQuizLLM` (`web_user_server.py:313`) → `scripts/gemini_general.py` |
| `gemini-2.5-flash` | Google Gemini | Scene-game question generation | `_scene_game_generate_question` → `scripts/gemini_general.py` |
| `gemini-2.5-flash` | Google Gemini | Recovery question generation (toy / child observation) | `scripts/gemini_recovery_question.py` → `/api/recovery/generate_question` |
| `gemini-2.5-flash` | Google Gemini | Conversation follow-up generation (red-card turn-taking) | `scripts/gemini_conversation_followup.py` → `/api/conversation/wait_for_turn` |
| `gemini-2.5-flash` | Google Gemini | With-question scene analysis | `scripts/gemini_wh_scene.py` → `/api/wh_scene/capture` |
| `gemini-2.5-flash-image` | Google Gemini | Story scene illustrations and scene-game item cards | `src/image_generator.py` / `src/image_generator_worker.py` |
| `gemini-robotics-er-1.5-preview` | Google Gemini | Object detection and localization (scene game) | `scripts/gemini_analyze_image.py` → `/api/scene_game/answer`, `/api/camera_capture` |
| `gemma4:e4b` | Ollama (local) | Mishearing correction for quiz answers — checks whether a Whisper transcript likely intended an expected canonical answer. **The only Ollama call in the web server.** | `_ensure_intent_llm()` at `web_user_server.py:292` |

#### Robot Operating System brain (`qt_ai_data_assistant.py`) — separate process

| Model | Provider | Used for |
|-------|----------|----------|
| `gemma4:e4b` (default) | Ollama (local) | Free-conversation language model. The default is set in `config/default.yaml` and can be overridden at the command line. |
| `mxbai-embed-large:latest` | Ollama (local) | Document embeddings for retrieval-augmented generation over `documents/` |
| `moondream` | Ollama (local) | Camera scene captioning (only when `enable_scene` is true) |

#### Removed / no longer used

The following models are referenced in old documentation but not in the current source: `llama3.1` appears only in the dead `src/story_generator_ashley.py` (not imported anywhere); `phi4:14b` does not appear in any source file.

### 5.2 Speech services

| Service | Used in | Purpose | Interface |
|---------|---------|---------|-----------|
| OpenAI `gpt-4o-transcribe` | Web server (every voice surface) | Speech recognition | Subprocess via `src/whisper.py`; returns `PARTIAL:` / `FINAL:` lines |
| NVIDIA Riva | Robot Operating System brain only | Speech recognition for free-conversation mode | gRPC (Docker container) |
| QTrobot Acapela | Both processes | Default text-to-speech with viseme-driven mouth sync | Robot Operating System service `/qt_robot/behavior/talkText` |
| Amazon Polly | Both processes (optional) | Alternative text-to-speech (paired with Pylips lipsync) | boto3 software development kit |

### 5.3 Vision services

| Service | Used in | Purpose | Interface |
|---------|---------|---------|-----------|
| Google Gemini Robotics ER 1.5 Preview | Web server | Held-object detection for the scene game | `scripts/gemini_analyze_image.py` (subprocess) |
| OpenCV hue/saturation/value thresholding | Web server | Red-card detection for child turn-taking | `_detect_red_card()` at `web_user_server.py:678` |
| DeepFace + RetinaFace | Robot Operating System brain only | Face detection and recognition | Local Python library |
| Silero | Robot Operating System brain only | Voice activity detection (paired with Riva) | Local PyTorch model |

### 5.4 Environment variables

```bash
# Language model and image generation keys
OPENAI_API_KEY=...              # Whisper speech recognition
GOOGLE_API_KEY=...              # Google Gemini (image generation + object detection)
GEMINI_API_KEY=...              # Google Gemini (alternative key name)

# Text-to-speech engine
TTS_ENGINE=qt                   # "qt" (default, with mouth synchronization) or "polly"
POLLY_VOICE=Ivy                 # Amazon Polly voice
POLLY_RATE=85%                  # Polly speech rate
POLLY_VOLUME=-10dB              # Polly volume

# Amazon Web Services (for Polly)
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_DEFAULT_REGION=us-east-1

# Robot connection
ROBOT_HOST=192.168.100.1
ROBOT_USER=developer
ROBOT_PASSWORD=qtrobot
ROBOT_SUDO_PASSWORD=qtrobot

# Whisper tuning
WHISPER_SILENCE_THRESHOLD=0.008 # Root-mean-square threshold for speech detection
WHISPER_SILENCE_DURATION=1.5    # Seconds of silence required to stop recording
WHISPER_MAX_RECORD=15.0         # Maximum recording seconds
WHISPER_STREAM_INTERVAL=2.0     # Partial transcription interval
WHISPER_PYTHON=/usr/bin/python3 # Python binary for the Whisper subprocess
```

---

## 6. ROS Interface

### 6.1 Services Used

| Service | Type | Purpose |
|---------|------|---------|
| `/qt_robot/behavior/talkText` | `behavior_talk_text` | Speak text (QT TTS) |
| `/qt_robot/behavior/talkAudio` | `behavior_talk_audio` | Play audio file |
| `/qt_robot/speech/config` | `speech_config` | Set TTS language/pitch/speed |
| `/qt_robot/emotion/show` | `emotion_show` | Display facial emotion |
| `/qt_robot/emotion/stop` | — | Stop current emotion |
| `/qt_robot/gesture/play` | `gesture_play` | Play arm gesture (name, speed) |
| `/qt_robot/gesture/list` | — | List available gestures |
| `/qt_robot/gesture/stop` | — | Stop current gesture |
| `/qt_robot/setting/setVolume` | `setting_setVolume` | Set speaker volume |
| `/qt_respeaker_app/tuning/set` | `tuning_set` | Configure microphone (AGC, etc.) |

### 6.2 Topics

| Topic | Type | Direction | Purpose |
|-------|------|-----------|---------|
| `/camera/color/image_raw` | `sensor_msgs/Image` | Subscribe | Camera feed |
| `/qt_robot/joints/state` | `sensor_msgs/JointState` | Subscribe | Joint positions |
| `/qt_respeaker_app/channel0` | `audio_common_msgs/AudioData` | Subscribe | Microphone audio |
| `/qt_robot/head_position/command` | `Float64MultiArray` | Publish | Head IK target |
| `/qt_robot/right_arm_position/command` | `Float64MultiArray` | Publish | Right arm IK target |
| `/qt_robot/left_arm_position/command` | `Float64MultiArray` | Publish | Left arm IK target |

### 6.3 Available Gestures (on QTRD)

**Location**: `qtrobot@QTRD:/home/qtrobot/robot/data/gestures/QT/`

**Positive / Celebration**: `happy`, `hoora`, `clapping`, `hi`, `nodding-yes`, `enjoy`, `yes1`, `bighi`, `one-arm-up`
**Encouragement**: `up_right`, `up_left`, `botharms`, `fast_hi`, `exactly`, `good_part`
**Empathy / Gentle**: `calm`, `shy`, `embrace`, `patience`, `relaxed`
**Correction / Gentle negative**: `slight_no`, `slight_no1`, `head-right-left`, `think`
**Emotions** (face): `happy`, `sad`, `surprised`, `afraid`, `angry`, `calm`, `disgusted`, `shy`, `hoora`
**Dance**: `Dance-1-1` through `Dance-4-6`
**Imitation**: `hands-on-belly`, `hands-on-head`, `hands-on-hip`, `hands-up`, `hands-side`, `nodding-yes`
**Pretend Play**: `Beep`, `Drive`, `Fly`, `Phone_call`
**Other**: `bye`, `kiss`, `peekaboo`, `monkey`, `sneezing`, `stretching`, `yawn`, `breathing_exercise`

**Gesture XML Format**:
```xml
<gesture>
    <name>happy</name>
    <parts>
        <part>left_arm</part>
        <part>right_arm</part>
    </parts>
    <duration>4.80</duration>
    <waypoints count="97">
        <point time="1549893042577720547">
            <LeftElbowRoll>-35.50</LeftElbowRoll>
            <LeftShoulderPitch>88.30</LeftShoulderPitch>
            <LeftShoulderRoll>-60.60</LeftShoulderRoll>
            <RightElbowRoll>-34.20</RightElbowRoll>
            <RightShoulderPitch>-87.60</RightShoulderPitch>
            <RightShoulderRoll>-58.60</RightShoulderRoll>
        </point>
        <!-- ... more waypoints ... -->
    </waypoints>
</gesture>
```

**6 Joints**: LeftElbowRoll, LeftShoulderPitch, LeftShoulderRoll, RightElbowRoll, RightShoulderPitch, RightShoulderRoll (all in degrees)

---

## 7. System Prompt Architecture

**File**: `documents/sar_system_prompt.md`

### Layer 1 — Core Values (hardcoded, never override)
1. **Child Wellbeing First** — safety over task completion
2. **Therapist Authority** — defer to clinical expert
3. **Honesty** — never deceive or fabricate
4. **Dignity** — respect regardless of behavior/ability
5. **Transparency of Limitations** — acknowledge uncertainty, escalate

### Layer 2 — Hard Constraints (hardcoded, never override)
- Never claim to be human
- Stop immediately if child distressed
- No age-inappropriate content
- No PII collection beyond session needs
- No medical/diagnostic advice
- No autonomous session continuation without therapist

### Layer 3 — Soft Constraints (defaults, therapist adjustable)
- Language complexity (target ages 4-10 default)
- Response length (1-3 sentences default)
- Encouragement style (effort-focused, not outcome-focused)
- Silence wait time (5 seconds default)
- Repetition limits (max 2 re-prompts)
- Error handling (never label response as "wrong")
- Topic boundaries (gentle redirect after 1 exchange)

### Layer 4 — Interaction Style (fully customizable per session)
- Persona: warm, curious, calm
- Session opening/closing scripts
- Unexpected input handling
- Robot self-reference (first person, no emotion claims)

### Session Context Injection Point
Insert child name, age, therapy type, session goals, sensitivities, and therapist overrides below Layer 4.

---

## 8. Data Flow Diagrams

### 8.1 Full Speech → Response → Action Cycle

```
[Microphone / ROS topic]
       │
       ▼
RivaSpeechRecognitionSilero
  ├── Silero VAD detects voice
  ├── Event.RECOGNIZING → HumanPresenceDetection.on_vad_trigged()
  │                      → acknowledge_human() → HumanTracking.track(speaker)
  └── Riva ASR returns text + language
       │
       ▼
QTAIDataAssistant._asr_callback(text, language)
  ├── State: LISTENING → PROCESSING
  ├── If PAUSED: check WakeupPrompt → resume or ignore
  └── ChatWithRAG.get_stream_response(text, user_context)
       │
       ▼
Ollama LLM Inference
  ├── System prompt (SAR 4-layer + ConversationPrompt)
  ├── Document context (RAG, if enabled)
  ├── Camera context (scene detection, if enabled)
  ├── Conversation memory (per-user ChatMemoryBuffer)
  └── Streams tokens → split_into_sentences()
       │
       ▼ (per complete sentence)
QTAIDataAssistant.proccess_response()
  ├── Try JSON parse → tool call?
  │     ├── {"command": "pause_interaction"} → set PAUSED state
  │     ├── {"command": "forget_conversation"} → clear memory
  │     ├── {"command": "set_language", "code": "fr-FR"} → change lang
  │     └── Other tool calls → CommandInterface.execute()
  └── Plain text → clean markdown → CommandInterface.execute([{"command": "talk", "message": text}])
       │
       ▼
CommandInterface._cmd_talk()
  └── ROS /qt_robot/behavior/talkText → Robot speaks (with mouth sync)
       │
       ▼
State: RESPONDING → IDLE
  ├── rest_robot_attention() (restore head position)
  ├── HumanTracking.untrack()
  └── IdleAttention.start()
```

### 8.2 Quiz Flow (Web Interface)

```
[Educational Quiz Page]
       │
       ├── loadQuiz(type) ──► GET /api/get_saved_quiz?type=yes_no
       │                       ├── Read user_data/<user>/quizzes/<type>/*.json
       │                       └── Merge learned_answers.json into accepted_answers
       │
       ├── (background) ──► POST /api/generate_quiz_feedback
       │                     ├── Load sar_system_prompt.md
       │                     ├── LLM generates 10 correct + 10 incorrect phrases
       │                     └── Return phrases (non-blocking)
       │
       ▼ Show question
       │
   ┌───┴───────────────────┐
   │                       │
[Click Yes/No]        [Click Mic 🎤]
   │                       │
   │                  POST /api/speech_recognize
   │                       ├── whisper.py --language en
   │                       ├── Parse FINAL: line
   │                       └── Return recognized text
   │                       │
   │                  Match "yes"/"no" or fill WH input
   │                       │
   └───────┬───────────────┘
           │
     Check answer
       ├── Yes/No: exact match
       ├── WH: normalize() + toSingular() + isAnswerAccepted()
       │        ├── Check against accepted_answers list
       │        ├── Strip articles (a/an/the/in/at/on)
       │        ├── Singular form match
       │        └── Containment match
       │
       ▼ Show feedback
       │
       ├── POST /api/speak_sentence ──► Robot speaks feedback phrase
       ├── POST /api/robot_gesture  ──► Robot plays gesture + emotion
       │     Correct: clapping/hoora/happy + happy face
       │     Wrong: patience/think/slight_no + calm face
       │
       └── [Teach Robot] button (WH only)
             ├── User types alternative answers
             └── POST /api/teach_quiz_answer
                  └── Save to user_data/<user>/quizzes/learned_answers.json
```

### 8.3 Object Detection Flow (Scene Game)

```
[Play Scene Page]
       │
       ▼
  GET /api/camera_frame ──► Camera feed (polling)
       │
  POST /api/camera_capture
       ├── Capture frame from ROS camera topic
       ├── Save to user_data/<user>/captured_scenes/
       ├── Invoke: python3.9 scripts/gemini_analyze_image.py --image <path>
       │     └── gemini-robotics-er-1.5-preview
       │          └── Returns [{"point": [y, x], "label": "carrot"}]
       ├── Parse detected label
       ├── Compare with target object (if provided)
       └── Robot voice feedback via tts_helper.speak()
```

### 8.4 Recovery Strategy Builder Flow

```
[Therapist drags components → builds recovery sequence]
       │
       ▼ serialize()
  { blocks: [{type:"step", component:"simple_attention", element:"name_call",
              text:"Hey! Soomin!", gesture:"hi", emotion:"QT/happy", useCamera:null},
             {type:"step", component:"question_attention", element:"toy_question",
              text:"...", gesture:"show_tablet", emotion:"QT/happy", useCamera:"toy"}],
    loop: 1 }
       │
       ├── [Test on Robot] → POST /api/activity/test (all steps, no pauses)
       ├── [Run] → step-by-step with therapist confirmation overlay
       └── [Save] → POST /api/activity/save → user_data/<user>/activities/

  Toy Question Step (useCamera="toy"):
  ┌──────────────────────────────────────────────────────┐
  │ 1. Capture camera → Gemini generates question        │
  │ 2. If toy seen: ask about it                         │
  │    If no toy: "What do you have? Show me!"           │
  │ 3. Watch camera every 3s for toy (up to 45s)         │
  │ 4. If toy appears: excited follow-up                 │
  │ 5. "Well done {name}!"                               │
  └──────────────────────────────────────────────────────┘
```

### 8.5 Conversation Flow Builder

```
[Therapist drags themes → sets follow-up count per theme]
       │
       ▼ serialize()
  { blocks: [{type:"step", theme:"greeting", text:"Hello Soomin!",
              gesture:"hi", emotion:"QT/happy", followups: 2},
             {type:"step", theme:"weather", text:"What's the weather?",
              gesture:"", emotion:"QT/happy", followups: 1}],
    loop: 1, activity_type: "conversation" }
       │
       ▼ Per theme step (e.g., greeting with 2 follow-ups):

  ┌─── Robot speaks opening ──────────────────────────────────┐
  │                                                            │
  │  _wait_until_robot_silent() → _enable_face_tracking()     │
  │  _signal_child_can_speak() → show_tablet gesture          │
  │                                                            │
  │  ┌── Follow-up 1 ──────────────────────────────────────┐  │
  │  │ ASR listens ◄──────────► Red card watcher (0.5s)    │  │
  │  │ (collects speech)         (polls camera in thread)   │  │
  │  │                  Red card detected!                  │  │
  │  │                     │                                │  │
  │  │  Filter English-only │                               │  │
  │  │                     ▼                                │  │
  │  │  Gemini generates follow-up question (ends with ?)   │  │
  │  │  Robot speaks follow-up                              │  │
  │  └─────────────────────────────────────────────────────┘  │
  │                                                            │
  │  ┌── Follow-up 2 ──────────────────────────────────────┐  │
  │  │ (same flow as above)                                 │  │
  │  └─────────────────────────────────────────────────────┘  │
  │                                                            │
  │  ┌── Closing ──────────────────────────────────────────┐  │
  │  │ ASR + red card → Gemini generates closing comment    │  │
  │  │ (no question, no farewell — warm acknowledgment)     │  │
  │  │ Robot speaks: "I really enjoyed hearing about that!" │  │
  │  └─────────────────────────────────────────────────────┘  │
  └────────────────────────────────────────────────────────────┘
```

### 8.6 Story Generation & Reading Flow

```
[Therapist clicks "Generate Story"]
       │ {child_name, age, gender, topics, goals}
       ▼
POST /api/generate_story[_stream]
       │
       ▼
StoryGenerator._build_prompt()
  ├── _get_age_tier(age)             # selects MASTER_TEMPLATE vs WH_MASTER_TEMPLATE
  ├── _get_theme_guidance(topics)    # season/school/family/friends/animals/adventure
  ├── _format_goals_section(goals)
  ├── PersonaRAG.build_story_prompt_fragment(age, disorder)
  └── _load_wh_examples(corpus)      # ages 4–6 only: few-shot from story_corpus.json
       │
       ▼
scripts/gemini_story.py (Python 3.9 subprocess, gemini-2.5-flash)
  └── streams CHUNK:<line> back to SSE / returns blocking text
       │
       ▼ Therapist approves
POST /api/save_story
  ├── Extract ** Title **, ** Takeaways ** blocks
  ├── If body > tier max_words → StoryGenerator.shorten_story()
  ├── _apply_emotion_tags_with_gemini()   # Gemini Flash re-inserts/repairs tags
  ├── _validate_tag_positions()           # snap tags to sentence boundaries
  ├── _split_story_into_pages(age)        # Gemini groups sentences into pages
  ├── _reinject_tags_into_pages()         # tags preserved across split
  ├── _split_into_paragraphs()
  ├── _identify_story_scenes()            # Gemini decides paragraphs sharing one image
  ├── _map_pages_to_paragraphs() → page_to_scene
  ├── _generate_story_questions()         # 3 comprehension MCQs
  ├── _generate_takeaway_questions()      # 1 MCQ per takeaway (ages 7+)
  ├── Persist user_data/<user>/stories/<file>.json
  └── ImageGenerator.generate_story_scene_image() per scene
       │ (Path A: direct google-genai if Py3.9; Path B: image_generator_worker.py subprocess)
       │ first image becomes the reference for subsequent scenes
       ▼
GET /read_story/<file>
       │
       ▼
GET /api/get_story_sentences   # returns pages + metadata + questions + takeaways
       │
       ▼ For each page:
GET /api/get_sentence_image     # resolves page_to_scene → story_scene_NNN_*.png
POST /api/speak_sentence
  ├── _split_page_into_segments()         # break at [gesture/emotion:…] tags
  ├── For each segment:
  │     ├── _play_tags(gestures, emotions)
  │     └── tts_helper.speak_story(sentence, language)   # per sentence
  └── Robot reads with inline gestures + facial expressions
       │
       ▼ End of story:
Comprehension + takeaway MCQs presented in UI
```

---

## 9. User Data Structure

```
user_data/
├── <username>/
│   ├── profile.json                    # {age, gender, disorder, learning_goals}
│   ├── chat_store.json                 # Persistent conversation memory (LlamaIndex)
│   ├── chat_history/                   # Individual conversation logs
│   ├── stories/
│   │   └── story_20260324_180000.json  # {story, metadata{child_name, age, age_tier, target_word_range, title},
│   │                                   #  pages[], paragraphs[], scenes[], page_to_scene[],
│   │                                   #  page_to_paragraph[], paragraph_to_scene[],
│   │                                   #  questions[{question, type, correct_answer, wrong_answers}],
│   │                                   #  takeaways[]}
│   ├── story_images/
│   │   └── story_20260324_180000/
│   │       ├── story_scene_000_<ts>_<uuid>.png   # one image per identified scene
│   │       ├── story_scene_001_<ts>_<uuid>.png   # (paragraphs sharing a scene reuse the same image)
│   │       └── ...                                # legacy stories use story_paragraph_NNN_*.png
│   ├── quizzes/
│   │   ├── yes_no/
│   │   │   └── quiz_20260324_180803.json  # [{question, type, correct_answer}]
│   │   ├── wh/
│   │   │   └── quiz_20260324_180803.json  # [{question, type, correct_answer, accepted_answers}]
│   │   └── learned_answers.json           # {"question text": ["alt1", "alt2", ...]}
│   ├── activities/
│   │   ├── activity_20260324_180000.json  # DIY: {blocks: [...], loop: 1}
│   │   └── activity_20260326_120000.json  # Conversation: {blocks: [...], loop: 1, activity_type: "conversation"}
│   ├── captured_scenes/                   # Camera captures for scene game & recovery questions
│   └── polly/                             # Polly TTS audio cache (if using polly)
│
├── activity_images/                       # Shared DIY builder images
│
└── users.json                             # Global user registry
    {
      "<username>": {
        "username": "...",
        "age": 5,
        "password_hash": "...",
        "created_at": "2026-03-24T...",
        "last_login": "2026-03-24T...",
        "display_name": "...",
        "gender": "...",
        "disorder": "...",
        "learning_goals": "..."
      }
    }
```

---

## 10. Python Runtime Architecture

The system requires **two Python versions** due to SDK compatibility:

```
┌─────────────────────────────┐    ┌──────────────────────────────┐
│    Python 3.8 (.venv)       │    │    Python 3.9 (.venv39)      │
│                             │    │                              │
│  - Flask web server         │    │  - qt_ai_data_assistant.py   │
│  - LlamaIndex / Ollama      │    │  - google-genai SDK          │
│  - DeepFace                 │    │  - Gemini API calls          │
│  - ROS Python bindings      │    │  - image_generator_worker.py │
│  - Whisper subprocess mgmt  │    │  - gemini_analyze_image.py   │
│  - TTSHelper                │    │  - robotics.py               │
│  - Human tracking/detection │    │                              │
└──────────┬──────────────────┘    └──────────────▲───────────────┘
           │                                      │
           │    subprocess.run / subprocess.Popen  │
           └──────────────────────────────────────┘
```

---

## 11. Threading & Concurrency

| Component | Thread Model | Sync Mechanism |
|-----------|-------------|----------------|
| QTAIDataAssistant | BaseNode loop thread | pause_event, state_lock |
| RivaSpeechRecognitionSilero | BaseNode loop thread | pause_event |
| HumanPresenceDetection | BaseNode loop thread | persons_lock |
| HumanTracking | Callback-driven | tracking state |
| IdleAttention | BaseNode loop thread | pause_event |
| SceneDetection | BaseNode loop thread | pause_event |
| CommandInterface | ThreadPoolExecutor | futures.wait() |
| Flask Web Server | Multi-threaded (default) | Flask session |
| Whisper ASR | Subprocess (blocking) | proc.wait() |
| Image Generator | Subprocess (Python 3.9) | proc.wait() |
| Red Card Watcher | Daemon thread per listen round | ThreadEvent (red_card_event) |
| Step Confirmation | Background activity thread | ThreadEvent (_step_confirm_event) |
| Conversation Follow-up | Gemini subprocess (Python 3.9) | subprocess.run + stdin pipe |

---

## 12. Interaction Modes

### Mode 1: Free Conversation (Robot Process)
Robot listens continuously → LLM responds → speaks + gestures → tracks gaze

### Mode 2: Educational Quiz (Web Interface)
Pre-generated questions → child answers via button/speech → robot gives varied feedback with gestures → "Teach Robot" for adaptive learning

### Mode 3: Story Reading (Web Interface)

Gemini generates an age-appropriate therapeutic story whose structure is selected by an age tier (`early_preschool` 3, `wh_question_format` 4–6, `early_school_age` 7–8, `school_age` 9–12). The story prompt is composed from theme guidance, the clinician's therapy goals, and a PersonaRAG fragment retrieved from `documents/personas_rag.json` based on the child's `disorder` field.

**Pipeline**:
1. `/api/generate_story[_stream]` → Gemini emits Title + body + (optional) Takeaways + Explanation, with inline `[gesture:…]` / `[emotion:QT/…]` tags
2. `/api/save_story` runs the post-processing pipeline (shorten if over word cap → Gemini re-tag emotions → split into age-appropriate pages → identify scenes at paragraph granularity → generate comprehension MCQs → generate one MCQ per takeaway for ages 7+)
3. `ImageGenerator` produces one illustration per **scene** (not per page), with the first image fed back as a style reference for the rest
4. `/read_story/<file>` → robot reads each page aloud through `tts_helper.speak_story()`; gesture/emotion tags fire on the segment they precede; the matching scene image is shown via `page_to_scene` mapping; comprehension questions are presented at the end

**Ages 4–6 (WH-format)** receive a short 3–4-sentence concrete vignette plus 5–7 WHO/WHAT/WHERE questions whose answers appear verbatim in the story. Few-shot examples are drawn from `documents/story for 4 to 6 years old/story_corpus.json`.

**Ages 7+** additionally receive 2–3 explicit takeaways (positive, actionable lessons) and a multiple-choice "what is one lesson from this story?" question per takeaway.

### Mode 4: Scene Game / Object Detection (Web Interface)
Camera feed shown → Gemini ER detects held objects → robot asks questions → validates answers

### Mode 5: Recovery Strategy Builder (Web Interface)
Therapist builds meltdown/disengagement recovery sequences via drag-and-drop. Four component types:
- **Simple Attention Bid** — name call or sound effect (age-appropriate text)
- **Question Attention Bid** — camera-based toy detection or child observation via Gemini
- **Modality Switch** — jumping jack, sing, or wiggling body
- **Graceful Withdrawal** — calm exit with age-appropriate language

**Toy question flow**: Robot captures camera → Gemini detects object → asks about it. If no object: asks child to show toy → watches camera every 3s → when toy appears, speaks excited follow-up → finishes with "Well done {name}!"

**Execution modes**:
- **Test**: All steps run sequentially (no pauses)
- **Run**: Step-by-step with therapist confirmation overlay between steps
- **Saved (run_saved)**: Server-side step-by-step with polling-based therapist confirmation

All text, gestures, and facial expressions are editable by the therapist. Text auto-generates based on child's age (4 tiers: 2-3, 4-5, 6-7, 8+).

### Mode 6: Conversation Flow Builder (Web Interface)
Therapist constructs conversational interaction flows from theme components (greeting, weather, weekend plan) via drag-and-drop.

**Per-theme step flow**:
1. Robot speaks opening line (greeting gets `hi` gesture; others have no gesture)
2. `_wait_until_robot_silent()` — mic stays off until TTS finishes + 1.5s cooldown
3. `_enable_face_tracking()` — robot follows child's face via camera + sound direction
4. `_signal_child_can_speak()` — robot performs `show_tablet` gesture to signal "your turn"
5. ASR listens while red card watcher polls camera every 0.5s in parallel
6. Child shows red card → ASR stops immediately → collected speech filtered to English-only
7. Gemini generates age-appropriate follow-up question (always ends with `?` unless closing)
8. Robot speaks follow-up → repeat from step 2 for configured number of rounds
9. Final round: closing comment (warm acknowledgment, no question, no farewell)

**Red card detection**: OpenCV HSV color thresholding (hue 0-10 & 165-180, saturation > 100), triggers when red area exceeds 3% of frame.

**Follow-up generation**: If child's speech is insufficient (< 2 English words), Gemini generates a new on-theme question instead of trying to reference unclear speech.

**Configurable**: 0-5 follow-up exchanges per theme. Even with 0 follow-ups, robot always listens once and gives a closing comment.

---

## 13. Security & Privacy

- **User isolation**: Separate data directories per user
- **Session-based auth**: Flask sessions, no cross-user access
- **API keys**: Environment variables only (not in code)
- **System prompt**: Explicitly forbids PII collection, sensitive data storage
- **No autonomous operation**: Robot stops if therapist leaves or becomes unresponsive
- **Child safety**: Hard constraints prevent harmful content generation
- **Data on disk**: Conversation memory, stories, quizzes stored locally (consider encryption for production)

---

## 14. File Index

```
version_1_llm_gemini/
├── config/
│   └── default.yaml                    # Application configuration
├── documents/
│   ├── sar_system_prompt.md            # 4-layer system prompt
│   ├── QTrobot.pdf                     # RAG document
│   ├── QTrobot_research_papers.txt     # RAG document
│   ├── personas_rag.json               # Persona profiles retrieved by PersonaRAG
│   └── story for 4 to 6 years old/
│       ├── story_corpus.json           # WH-question few-shot corpus (ages 4–6)
│       ├── story for kid.pdf
│       └── story with wh questions.pdf
├── scripts/
│   ├── autostart/
│   │   └── start_qt_ai_data_assitant.sh
│   ├── gemini_analyze_image.py         # Gemini ER object detection (Py3.9)
│   ├── gemini_story.py                 # Story generation via Gemini (Py3.9)
│   ├── gemini_general.py               # General-purpose Gemini text gen (Py3.9)
│   ├── gemini_recovery_question.py     # Camera-based toy/child question gen (Py3.9)
│   ├── gemini_conversation_followup.py # Conversation follow-up generation (Py3.9)
│   ├── gemini_wh_scene.py              # WH-question scene analysis (Py3.9)
│   └── install_ollama.sh
├── src/
│   ├── qt_ai_data_assistant.py         # Main robot brain
│   ├── web_user_server.py              # Flask web server (~2700 lines)
│   ├── command_interface.py            # ROS command execution
│   ├── llamaindex_interface.py         # LLM + RAG engine
│   ├── llm_prompts.py                  # Prompt definitions
│   ├── tts_helper.py                   # Text-to-speech
│   ├── whisper.py                      # OpenAI Whisper ASR
│   ├── riva_speech_recognition_vad.py  # Riva ASR + Silero VAD
│   ├── human_presence_detection.py     # Face detection
│   ├── human_tracking.py              # Gaze following
│   ├── idle_attention.py              # Idle gaze behavior
│   ├── scene_detection.py             # Camera scene understanding
│   ├── story_generator.py            # Therapeutic story generation (Gemini default, Ollama fallback)
│   ├── story_generator_ashley.py      # Legacy Ollama prototype — not wired into the web server
│   ├── persona_rag.py                # Persona retrieval + prompt fragment builder
│   ├── image_generator.py            # Gemini image generation (direct SDK + Py3.9 worker fallback)
│   ├── image_generator_worker.py      # Py3.9 image gen subprocess (gemini-2.5-flash-image)
│   ├── user_management.py            # Multi-user system
│   ├── user_interface.py             # User interface base
│   ├── user_cli_interface.py         # CLI user selection
│   ├── user_web_interface.py         # Web user interface
│   ├── version.py                    # Version info
│   ├── kinematics/
│   │   └── kinematic_interface.py    # IK for head + arms
│   ├── utils/
│   │   ├── base_node.py              # Threaded component base class
│   │   └── utils.py                  # Sentence splitting, timestamps
│   ├── user_data/                    # Per-user data storage
│   └── .env                          # Environment variables
├── templates/
│   ├── index.html                    # Login / registration / game selection
│   ├── dashboard.html                # Main dashboard
│   ├── play_games.html               # Game selection
│   ├── educational_quiz.html         # Quiz playing
│   ├── quiz_generation.html          # Quiz builder
│   ├── read_story.html               # Story reading
│   ├── play_scene.html               # Object detection game
│   ├── diy_builder.html              # Recovery strategy builder (drag-and-drop)
│   ├── conversation_builder.html     # Conversation flow builder (drag-and-drop)
│   ├── my_games.html                 # Saved activities & conversations
│   └── select_toy.html               # Toy selection
├── robotics.py                       # Google Gemini Robotics ER demo script
├── test.py                           # Google Gemini vision test
├── requirements.txt                  # Python dependencies
├── env.polly                         # Amazon Polly text-to-speech environment variables (source manually)
└── ARCHITECTURE.md                   # This file
```

---

## 15. Per-activity pipelines

This section walks through each user-facing activity end to end — what data goes in, which models are called in which order, what files are written, and what the robot does to close the loop. All pipelines below describe the active **web server path** unless explicitly labeled "Robot Operating System path."

### 15.1 Story telling

The story activity is split into a long authoring pipeline (run once, when the therapist asks for a story) and a short read-aloud loop (run page by page when the child reads).

```
┌─────────────────────────────────────────────────────────────────────┐
│ INPUTS                                                              │
│  • User profile: age, gender, disorder, learning_goals              │
│    (users.json + user_data/<user>/profile.json)                     │
│  • Therapist: topics[], extra goals                                 │
└────────────────────────────┬────────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 1. PERSONA RETRIEVAL                                                │
│    PersonaRAG.build_story_prompt_fragment(age, disorder)            │
│    → matched persona's interests / language targets / constraints   │
└────────────────────────────┬────────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 2. STORY GENERATION  (web_user_server.py:1132 / :1183 stream)       │
│    StoryGenerator → scripts/gemini_story.py  [gemini-2.5-flash]     │
│    Age tier picks the template:                                     │
│      3 → early_preschool   4-6 → with-question format               │
│      7-8 → early_school    9-12 → school_age                        │
│    Output: Title + body with inline [gesture:X] [emotion:QT/X]      │
│            + Takeaways (ages 7+)                                    │
└────────────────────────────┬────────────────────────────────────────┘
                             ▼
                    🧑 THERAPIST APPROVES
                             │
                             ▼ /api/save_story  (web_user_server.py:2530)
┌─────────────────────────────────────────────────────────────────────┐
│ POST-PROCESSING PIPELINE  (every step is Gemini Flash via           │
│                            scripts/gemini_general.py)                │
│                                                                     │
│  3. Shorten if over tier word cap   StoryGenerator.shorten_story()  │
│  4. Re-tag emotions and gestures    _apply_emotion_tags_with_gemini │
│  5. Snap tags to sentence bounds    _validate_tag_positions         │
│  6. Split into age-sized pages      _split_story_into_pages         │
│  7. Re-inject tags lost in split    _reinject_tags_into_pages       │
│  8. Paragraphs + scene grouping     _identify_story_scenes          │
│     → page_to_scene mapping                                         │
│  9. Comprehension questions (3)     _generate_story_questions       │
│ 10. One question per takeaway (7+)  _generate_takeaway_questions    │
└────────────────────────────┬────────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 11. PERSIST                                                         │
│     user_data/<user>/stories/story_<ts>.json                        │
│     {story, metadata, pages[], paragraphs[], scenes[],              │
│      page_to_scene[], questions[], takeaways[]}                     │
└────────────────────────────┬────────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 12. IMAGE GENERATION  (one image per scene, NOT per page)           │
│     ImageGenerator → gemini-2.5-flash-image                         │
│     The first image becomes the style reference for the rest        │
│     Saves: user_data/<user>/story_images/<file>/story_scene_NNN.png │
└────────────────────────────┬────────────────────────────────────────┘
                             ▼
        ═══════════ READ-ALOUD on /read_story/<file> ═══════════
                             │
                             ▼ per page → /api/speak_sentence (5202)
┌─────────────────────────────────────────────────────────────────────┐
│ For each segment between [gesture:] / [emotion:] tags:              │
│   ├── Fire Robot OS /qt_robot/gesture/play   _play_tags()  (5108)   │
│   ├── Fire Robot OS /qt_robot/emotion/show                          │
│   └── Per sentence: tts_helper.speak_story()                        │
│       → Robot OS /qt_robot/behavior/talkText                        │
│       (Whisper is suspended during text-to-speech                   │
│        via _with_asr_suspended)                                     │
│ HumanTracking follows the child's face throughout                   │
└────────────────────────────┬────────────────────────────────────────┘
                             ▼
        📝 User interface shows comprehension questions
            (multiple-choice; one correct answer + two
             plausible distractors per question)
```

---

### 15.2 Educational quiz

The quiz activity has an **authoring phase** (therapist generates and saves a quiz) and a **play phase** (child answers questions on screen with optional voice input).

```
═══════════════════ AUTHORING PHASE ═══════════════════

  Therapist picks: topic, age, type (yes_no | wh)
                   │
                   ▼  /api/generate_quiz  (1250)
  ┌──────────────────────────────────────────────────┐
  │ _GeminiQuizLLM → scripts/gemini_general.py       │
  │                  [gemini-2.5-flash]              │
  │ Returns JSON array of {question, type,           │
  │ correct_answer, accepted_answers[]}              │
  │ Special branch for "social rules" topic (1310)   │
  └────────────────────────┬─────────────────────────┘
                           ▼  /api/save_quiz  (1748)
  user_data/<user>/quizzes/{yes_no|wh}/quiz_<ts>.json

         ▼  optional: /api/generate_quiz_feedback (1540)
  Pre-generates 10 correct + 10 incorrect feedback phrases
  (varied praise / encouragement, also Gemini)


═══════════════════ PLAY PHASE ═══════════════════

  Browser /educational_quiz → loads quiz + merges
                              learned_answers.json
                   │
                   ▼ shows question
   ┌───────────────┴─────────────────┐
   │                                 │
[Yes/No tap]              [🎤 Mic — Whisper]
                              POST /api/speech_recognize
                              → src/whisper.py subprocess
                              → OpenAI gpt-4o-transcribe
                              → "FINAL:<text>" parsed
   │                                 │
   │                                 ▼
   │                  ┌──────────────────────────────┐
   │                  │ Match against accepted_answers│
   │                  │  ├ exact / strip articles     │
   │                  │  ├ singular form              │
   │                  │  └ containment match          │
   │                  │                                │
   │                  │ MISS? → _llm_canonicalize_heard│
   │                  │  ChatWithRAG model="gemma4:e4b"│
   │                  │  (Ollama) — "did the child    │
   │                  │  intend the expected word?"   │
   │                  │  ← only Ollama call in the    │
   │                  │    web server                  │
   │                  └──────────────┬─────────────────┘
   ▼                                 ▼
  ┌───── Correct ─────┐    ┌───── Wrong ─────┐
  │ Speak praise      │    │ Speak gentle hint│
  │ Gesture: clapping │    │ Gesture: think / │
  │   / hoora / happy │    │   patience       │
  │ Emotion: QT/happy │    │ Emotion: QT/calm │
  └─────────┬─────────┘    └─────────┬────────┘
            │                        │
            │       (with-questions  │
            │        only)           ▼
            │           [Teach Robot] button
            │           POST /api/teach_quiz_answer (1710)
            │           → append to learned_answers.json
            ▼
       Next question
```

---

### 15.3 Scene game (object detection)

The child holds an object up to the camera; the robot asks a question about what it sees, and Google Gemini Robotics decides whether the object matches.

```
   /api/scene_game/new_round  (3177)
              │
              ▼
   ┌──────────────────────────────────────────────────┐
   │ Load toy list → _load_scene_toys()               │
   │ Generate question → _scene_game_generate_question│
   │   gemini_general.py [gemini-2.5-flash]           │
   │   Age tier:                                      │
   │     2-3  → "find the carrot"                     │
   │     4-6  → color/shape criteria                  │
   │     7+   → inference riddle                      │
   │ Generate item card images via                    │
   │   ImageGenerator [gemini-2.5-flash-image]        │
   └──────────────────────┬───────────────────────────┘
                          ▼
              🤖 Robot speaks the question
                          │
                  Child holds an object up
                          │
                          ▼  /api/scene_game/answer  (3309)
   ┌──────────────────────────────────────────────────┐
   │ _get_ros_frame()           ← grab camera frame   │
   │ _run_gemini_detect_and_look(image)  (3341)       │
   │   subprocess scripts/gemini_analyze_image.py     │
   │   [gemini-robotics-er-1.5-preview]               │
   │   Returns {label, color, shape, point[y, x]}     │
   └──────────────────────┬───────────────────────────┘
                          ▼
   ┌──────────────────────────────────────────────────┐
   │ Compare detected vs target:                      │
   │   age 2-3   → exact label match                  │
   │   age 4+    → criteria match (color/shape/sub)   │
   └──────────────────────┬───────────────────────────┘
                          ▼
                    ┌─────┴──────┐
                  Correct       Wrong
                    │            │
            "That's correct!"  "No, try again"
              tts_helper.speak  tts_helper.speak
```

---

### 15.4 Recovery activity builder ("do-it-yourself" builder)

The therapist drags components onto a canvas to compose a recovery sequence (used when a child becomes distracted or distressed). Each component compiles down to a Step block on the server, optionally with a camera-driven question and a follow-up loop.

```
═══════════════════ AUTHORING (drag-and-drop user interface) ════════

  Therapist composes blocks. Server-side schema (web_user_server.py
  _execute_activity, line 4034) supports:
   • Step block        → speak + gesture + emotion (+ optional camera)
   • Logic block       → parallel speech recognizers → branch then-blocks
   • Loop wrapper      → run blocks N times

  Front-end "components" map to Step blocks with different settings:
   • simple_attention      → name call / sound, no camera
   • question_attention    → useCamera="toy" or "child"
   • modality_switch       → speak + dance/sing gesture
   • graceful_withdrawal   → calm exit script
                                   │
                                   ▼  /api/activity/save  (4473)
  user_data/<user>/activities/activity_<ts>.json


═══════════════════ EXECUTION ═══════════════════

  /api/activity/test       → _execute_activity, no pauses
  /api/activity/run_saved  → background thread, step-by-step

  Per Step block (4151-4370):

  ┌──────────────────────────────────────────────────────────┐
  │ 1. If useCamera → _generate_recovery_question (4161)     │
  │    _get_ros_frame() →                                    │
  │    scripts/gemini_recovery_question.py                   │
  │    [gemini-2.5-flash]                                    │
  │    --mode toy | child  --child-age N  --child-name X     │
  │    Returns {text, object}                                │
  ├──────────────────────────────────────────────────────────┤
  │ 2. tts_helper.speak_story(text)            🤖 SPEAK      │
  │ 3. Robot OS gesture/play  +  emotion/show  🤖 EMOTE      │
  ├──────────────────────────────────────────────────────────┤
  │ 4. If num_followups > 0:  per follow-up round            │
  │      a. Enable face tracking                             │
  │      b. Spawn red-card watcher thread                    │
  │           _detect_red_card(frame)                        │
  │           hue/saturation/value [0–10] ∪ [165–180]        │
  │           triggers when red area > 3% of frame           │
  │      c. _whisper_recognize_once()  → child speech        │
  │      d. Stop on red card OR speech-recognition end       │
  │      e. scripts/gemini_conversation_followup.py          │
  │           [gemini-2.5-flash]                             │
  │      f. Speak follow-up → loop                           │
  └──────────────────────┬───────────────────────────────────┘
                         ▼
  Therapist confirmation between steps
   ├── /api/activity/step_status   (poll: waiting/index/labels)
   └── /api/activity/confirm_step  (set _step_confirm_event)
```

---

### 15.5 Conversation flow builder

A structured back-and-forth that pairs the robot's prompts with red-card-driven turn-taking. The therapist defines each conversational theme and how many follow-up exchanges to allow per theme.

```
═══════════════════ AUTHORING ═══════════════════

  Therapist drags themes (greeting, weather, weekend...)
  Sets followups: 0-5 per theme
                                 │
                                 ▼
  user_data/<user>/activities/activity_<ts>.json
  {blocks:[{type:"step", theme, text, gesture, emotion,
            followups:N}], activity_type:"conversation"}


═══════════════════ EXECUTION (per theme step) ═══════════════════

   ┌─ Robot speaks opening (greeting → "hi" gesture) ─┐
   │                                                  │
   │ /api/conversation/wait_for_turn  (3652)          │
   │                                                  │
   │ 1. _wait_until_robot_silent()                    │
   │    (text-to-speech done + 1.5 s)                 │
   │ 2. _enable_face_tracking()                       │
   │ 3. _signal_child_can_speak()                     │
   │      → Robot OS gesture "show_tablet"            │
   │        (means: your turn)                        │
   │                                                  │
   │ 4. PARALLEL:                                     │
   │    ┌────────────────┐   ┌────────────────────┐   │
   │    │ Whisper speech │   │ Red-card watcher   │   │
   │    │ recognition,   │   │ thread, polls      │   │
   │    │ collects child │   │ camera every 0.5 s │   │
   │    │ speech rounds  │   │                    │   │
   │    └───────┬────────┘   └─────────┬──────────┘   │
   │            └────────┬─────────────┘              │
   │                 stop on red card                 │
   │                                                  │
   │ 5. Filter to English-only                        │
   │ 6. scripts/gemini_conversation_followup.py       │
   │    [gemini-2.5-flash]                            │
   │    stdin: {theme, robot_said, child_said,        │
   │            child_name, child_age, followup_n,    │
   │            total_followups, history, is_closing} │
   │ 7. Insufficient speech (< 2 English words)?      │
   │    → Gemini generates a new on-theme question    │
   │ 8. Robot speaks follow-up                        │
   │                                                  │
   │ 9. Repeat steps 1–8 for N follow-ups             │
   │                                                  │
   │10. CLOSING ROUND: same flow but is_closing=true  │
   │    → warm acknowledgment, no question, no        │
   │      farewell                                    │
   └──────────────────────────────────────────────────┘

  Red-card detection (web_user_server.py:678):
    Hue/saturation/value thresholds
                    hue ∈ [0, 10] ∪ [165, 180]
                    saturation > 100, value > 80
    Detected when red pixel ratio > 3% of frame
```

---

### 15.6 Free conversation (Robot Operating System path — separate process)

This is the open-ended chat mode that runs in `qt_ai_data_assistant.py`, **not** the web server. The Flask application does not invoke this state machine.

```
  Source: src/qt_ai_data_assistant.py
  Note: This is the Robot Operating System brain, separate from
        the web server.

  ┌──────────────────────────────────────────────────────────┐
  │  STATE: IDLE                                             │
  │   IdleAttention.start() → random gaze                    │
  │   asr.recognize_once()                                   │
  └──────────────────────┬───────────────────────────────────┘
                         │ Riva + Silero voice activity trigger
                         ▼
  ┌──────────────────────────────────────────────────────────┐
  │  STATE: LISTENING                                        │
  │   acknowledge_human() → HumanTracking.track(speaker)     │
  │   Riva returns text + language                           │
  └──────────────────────┬───────────────────────────────────┘
                         ▼
  ┌──────────────────────────────────────────────────────────┐
  │  STATE: PROCESSING                                       │
  │   ChatWithRAG.get_stream_response(text, user_context)    │
  │     • Language model: Ollama                             │
  │       (config default gemma4:e4b)                        │
  │     • Embeddings: mxbai-embed-large (Ollama)             │
  │     • Retrieval: VectorStoreIndex over documents/        │
  │     • Memory: per-user ChatMemoryBuffer                  │
  │     • Optional camera context (SceneDetection /          │
  │       Moondream only fires if enable_scene = true)       │
  └──────────────────────┬───────────────────────────────────┘
                         ▼ stream of sentences
  ┌──────────────────────────────────────────────────────────┐
  │  STATE: RESPONDING                                       │
  │   For each chunk:                                        │
  │     • JSON tool call?                                    │
  │         pause_interaction → PAUSED                       │
  │         forget_conversation → clear memory               │
  │         set_language → swap text-to-speech +             │
  │           re-initialize Riva                             │
  │     • Plain text → CommandInterface.execute(talk)        │
  │         → Robot OS /qt_robot/behavior/talkText           │
  │           (mouth synchronization)                        │
  └──────────────────────┬───────────────────────────────────┘
                         ▼
                  rest_robot_attention()
                         │
                         ▼
                   STATE: IDLE
```

---

### 15.7 Cross-activity model summary

| Activity | Authoring-time models | Run-time models |
|----------|----------------------|------------------|
| Story telling | Google Gemini 2.5 Flash (≥ 6 passes), Google Gemini 2.5 Flash Image | (none — robot reads pre-baked output) |
| Educational quiz | Google Gemini 2.5 Flash (questions + feedback phrases) | OpenAI gpt-4o-transcribe (Whisper); Ollama gemma4:e4b (mishearing correction) |
| Scene game | Google Gemini 2.5 Flash (questions); Google Gemini 2.5 Flash Image (cards) | Google Gemini Robotics ER 1.5 Preview (object detection) |
| Recovery activity builder | (none — text and gestures authored manually) | Google Gemini 2.5 Flash (recovery questions, follow-ups); Whisper |
| Conversation flow builder | (none) | Google Gemini 2.5 Flash (follow-ups); Whisper |
| Free conversation (Robot Operating System path) | (none) | Ollama (`gemma4:e4b` per current `config/default.yaml`); Riva speech recognition; optionally Moondream |
