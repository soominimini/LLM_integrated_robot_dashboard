# QTrobot AI Data Assistant — System Architecture

## 1. System Overview

A socially assistive robot (SAR) platform for pediatric therapeutic settings, built on QTrobot hardware. The system integrates multi-modal perception (speech, vision, presence detection), LLM-powered conversation with RAG, and expressive robot behavior (speech, gestures, gaze tracking) — all governed by a layered ethical system prompt designed for child safety.

```
┌──────────────────────────────────────────────────────────────────┐
│                        THERAPIST / CHILD                         │
│                    (Speech, Gestures, Objects)                    │
└──────────┬──────────────────────────────────────┬────────────────┘
           │ Audio / Visual Input                  │ Speech / Movement Output
           ▼                                       ▼
┌──────────────────────┐              ┌──────────────────────────┐
│   PERCEPTION LAYER   │              │    EXPRESSION LAYER      │
│  ┌────────────────┐  │              │  ┌────────────────────┐  │
│  │ Riva ASR (ROS) │  │              │  │ QT TTS (ROS)       │  │
│  │ Whisper (Web)  │  │              │  │ AWS Polly (optional)│  │
│  │ Silero VAD     │  │              │  │ Gestures (ROS)     │  │
│  │ DeepFace       │  │              │  │ Emotions (ROS)     │  │
│  │ Moondream      │  │              │  │ Head/Arm IK (ROS)  │  │
│  │ Gemini ER      │  │              │  │ Pylips Lipsync     │  │
│  └────────────────┘  │              │  └────────────────────┘  │
└──────────┬───────────┘              └──────────▲───────────────┘
           │                                     │
           ▼                                     │
┌──────────────────────────────────────────────────────────────────┐
│                       COGNITION LAYER                            │
│  ┌─────────────────┐  ┌──────────────┐  ┌───────────────────┐  │
│  │ Ollama LLM      │  │ LlamaIndex   │  │ Gemini Vision     │  │
│  │ (llama3.1,      │  │ RAG Engine   │  │ (robotics-er,     │  │
│  │  phi4:14b)      │  │ (documents)  │  │  2.5-flash-image) │  │
│  └─────────────────┘  └──────────────┘  └───────────────────┘  │
│  ┌─────────────────┐  ┌──────────────┐  ┌───────────────────┐  │
│  │ Chat Memory     │  │ System Prompt│  │ Scene Context     │  │
│  │ (per-user)      │  │ (4-layer)    │  │ (camera feed)     │  │
│  └─────────────────┘  └──────────────┘  └───────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
           │                                     ▲
           ▼                                     │
┌──────────────────────────────────────────────────────────────────┐
│                     ORCHESTRATION LAYER                          │
│  ┌──────────────────────────┐  ┌─────────────────────────────┐  │
│  │ QTAIDataAssistant        │  │ Flask Web Server             │  │
│  │ (ROS main node)          │  │ (Therapist/User interface)   │  │
│  │ State: IDLE → LISTENING  │  │ Routes: /api/*, pages        │  │
│  │  → PROCESSING → RESPOND  │  │ Session-based auth           │  │
│  └──────────────────────────┘  └─────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
           │                                     │
           ▼                                     ▼
┌──────────────────────────────────────────────────────────────────┐
│                        DATA LAYER                                │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────────────┐  │
│  │ User Profiles │  │ Chat Memory  │  │ Quizzes / Stories /   │  │
│  │ (users.json)  │  │ (per-user)   │  │ Activities / Learned  │  │
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

#### `src/llamaindex_interface.py` — LLM + RAG Engine

**Class**: `ChatWithRAG`

**Components**:
- **LLM**: Ollama (local, default `llama3.1`)
- **Embeddings**: OllamaEmbedding (`mxbai-embed-large:latest`)
- **Document Loader**: SimpleDirectoryReader (PDF, TXT, MD, DOCX)
- **Index**: VectorStoreIndex (in-memory)
- **Memory**: ChatMemoryBuffer → SimpleChatStore (persistent per-user)
- **Chat Engine**: CustomChatEngine (extends ContextChatEngine) with camera context injection

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

#### `src/riva_speech_recognition_vad.py` — Primary ASR (Robot Process)

**Class**: `RivaSpeechRecognitionSilero`

- **ASR Backend**: NVIDIA Riva (gRPC, Docker)
- **VAD**: Silero VAD (confidence threshold: 0.6)
- **Audio**: 16kHz, mono, from ROS topic `/qt_respeaker_app/channel0`
- **Languages**: en-US, en-GB, ar-AR, de-DE, es-ES, fr-FR, hi-IN, it-IT, ja-JP, ru-RU, ko-KR, pt-BR, zh-CN

**Event Flow**:
```
Audio chunks from ROS mic topic
  → Silero VAD detects voice activity
  → Event.RECOGNIZING fired → robot starts tracking speaker
  → Riva ASR processes audio stream
  → Returns recognized text + language
```

#### `src/whisper.py` — Web ASR (Quiz/Web Interface)

**Type**: Subprocess (called by web server)
**Backend**: OpenAI `gpt-4o-transcribe`

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

**Language**: Passed via `--language` CLI arg (ISO-639-1 code extracted from config).

#### `src/tts_helper.py` — Text-to-Speech

**Class**: `TTSHelper`

**Engines**:
| Engine | Service | Mouth Sync |
|--------|---------|------------|
| `qt` (default) | ROS `/qt_robot/behavior/talkText` (Acapela) | Built-in viseme support |
| `polly` | AWS Polly → SSH upload → robot playback | Via Pylips (socketio) |

**Joint Limits** (for movement during speech):
```
Head:      HeadYaw [-90, 90], HeadPitch [-15, 25]
Right Arm: ShoulderPitch [-140, 140], ShoulderRoll [-75, 7], ElbowRoll [-90, -7]
Left Arm:  ShoulderPitch [-140, 140], ShoulderRoll [-75, 7], ElbowRoll [-90, -7]
```

---

### 3.4 Vision & Perception

#### `src/human_presence_detection.py` — Face Detection

**Class**: `HumanPresenceDetection`

- **Backend**: DeepFace + RetinaFace
- **Input**: ROS camera topic
- **Output**: Per-person data: face bbox, 3D position (xyz), emotions, embeddings
- **Features**: Temporal filtering, external VAD trigger, callback-based

#### `src/human_tracking.py` — Gaze Following

**Class**: `HumanTracking`

- **Input**: HumanPresenceDetection callbacks
- **Output**: Smooth head movement to follow active speaker
- **Features**: Person ID tracking, absence memory (forgets after 10 min)

#### `src/idle_attention.py` — Idle Gaze

**Class**: `IdleAttention`

- Random gaze at detected persons or random directions
- Prevents staring; creates natural "looking around" behavior
- Active when robot is in IDLE state

#### `src/scene_detection.py` — Scene Understanding

**Class**: `SceneDetection`

- **Model**: Moondream (via Ollama)
- **Input**: Camera frames at configurable framerate (default: 0.1 FPS)
- **Output**: Scene description text → injected into LLM context
- **Prompt**: "Describe in details what you see. If you see people, also describe how they dressed and what they carry."

#### `scripts/gemini_analyze_image.py` — Physical Object Detection

**Model**: `gemini-robotics-er-1.5-preview`
**Runtime**: Python 3.9 (subprocess)

- **Input**: Image file path (`--image` argument)
- **Output**: JSON array of detected objects with normalized point coordinates
- **Format**: `[{"point": [y, x], "label": "object_name"}]` (coordinates 0-1000)
- **Prompt**: "Point to no more than 1 item a person is holding in the image."
- **Used by**: `/api/camera_capture` in web server for scene game / object validation

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

### 5.1 LLM Models

| Model | Provider | Runtime | Used For |
|-------|----------|---------|----------|
| `llama3.1` | Ollama (local) | Python 3.8/3.9 | Main conversation, RAG (default `ChatWithRAG` LLM) |
| `phi4:14b` | Ollama (local) | Python 3.8 | Quiz generation, feedback phrases |
| `mxbai-embed-large:latest` | Ollama (local) | Python 3.8 | Document embeddings for RAG |
| `moondream` | Ollama (local) | Python 3.8 | Camera scene understanding |
| (story Ollama fallback) | Ollama (local) | Python 3.8/3.9 | `StoryGenerator` can be reconfigured to a non-`gemini-*` Ollama model; selected via `StoryGenerator(llm_model=…)` |

### 5.2 Gemini Models

| Model | Purpose | Called From |
|-------|---------|------------|
| `gemini-robotics-er-1.5-preview` | Physical object detection + localization | `scripts/gemini_analyze_image.py` → `/api/camera_capture` |
| `gemini-2.5-flash-image` | Story scene illustration generation | `src/image_generator.py` / `src/image_generator_worker.py` |
| `gemini-2.5-flash` | Therapeutic story generation (age-tiered, themed, WH-format) | `scripts/gemini_story.py` → `StoryGenerator` → `/api/generate_story[_stream]` |
| `gemini-2.5-flash` | Story post-processing: emotion/gesture re-tagging, page splitting, scene identification, comprehension + takeaway MCQs | `scripts/gemini_general.py` (via `_gemini_generate()`) → `/api/save_story`, `/api/get_story_sentences` |
| `gemini-2.5-flash` | Recovery question generation (toy/child, age-appropriate) | `scripts/gemini_recovery_question.py` → `/api/recovery/generate_question` |
| `gemini-2.5-flash` | Conversation follow-up generation | `scripts/gemini_conversation_followup.py` → `/api/conversation/wait_for_turn` |
| `gemini-2.5-flash` | WH-question scene analysis | `scripts/gemini_wh_scene.py` → `/api/wh_scene/capture` |

### 5.3 Speech Services

| Service | Purpose | Interface |
|---------|---------|-----------|
| NVIDIA Riva | Primary ASR (robot process) | gRPC (Docker container) |
| OpenAI `gpt-4o-transcribe` | Web ASR (quiz/web interface) | REST API via subprocess |
| QT Acapela | Default TTS (with mouth sync) | ROS service |
| AWS Polly | Optional TTS | boto3 SDK |

### 5.4 Vision Services

| Service | Purpose | Interface |
|---------|---------|-----------|
| DeepFace + RetinaFace | Face detection & recognition | Local Python library |
| Silero VAD | Voice activity detection | Local PyTorch model |

### 5.5 Environment Variables

```bash
# LLM / AI APIs
OPENAI_API_KEY=...              # Whisper ASR
GOOGLE_API_KEY=...              # Gemini (image gen + object detection)
GEMINI_API_KEY=...              # Gemini (alternative key name)

# TTS Engine
TTS_ENGINE=qt                   # "qt" (default, with mouth sync) or "polly"
POLLY_VOICE=Ivy                 # AWS Polly voice
POLLY_RATE=85%                  # Polly speech rate
POLLY_VOLUME=-10dB              # Polly volume

# AWS (for Polly)
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_DEFAULT_REGION=us-east-1

# Robot Connection
ROBOT_HOST=192.168.100.1
ROBOT_USER=developer
ROBOT_PASSWORD=qtrobot
ROBOT_SUDO_PASSWORD=qtrobot

# Whisper Tuning
WHISPER_SILENCE_THRESHOLD=0.008 # RMS threshold for speech detection
WHISPER_SILENCE_DURATION=1.5    # Seconds of silence to stop recording
WHISPER_MAX_RECORD=15.0         # Max recording seconds
WHISPER_STREAM_INTERVAL=2.0     # Partial transcription interval
WHISPER_PYTHON=/usr/bin/python3 # Python binary for whisper subprocess
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
├── robotics.py                       # Gemini ER demo script
├── test.py                           # Gemini vision test
├── requirements.txt                  # Python dependencies
├── env.polly                         # Polly TTS env vars (source manually)
└── ARCHITECTURE.md                   # This file
```
