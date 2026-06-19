# Final results — prompt token usage vs. model context window (per activity)

## How to calculate the tokens

Token counts are **model-specific** — always count with the same model you will send
the prompt to. Two ways, in order of accuracy:

### A. Exact — provider `count_tokens` APIs (what the numbers on this page use)

`count_tokens` measures a prompt **without generating** (negligible cost). It needs
the API keys, which this project stores in `src/.env` as
`export GOOGLE_API_KEY=…` / `export GEMINI_API_KEY=…` / `export ANTHROPIC_API_KEY=…`.

Load the keys (note the `export` prefix):
```bash
set -a; . ./src/.env; set +a      # run from the demo root
```
or in Python:
```python
import os
for line in open("src/.env"):
    line = line.strip()
    if line.startswith("export "): line = line[7:]
    if "=" in line and not line.startswith("#"):
        k, v = line.split("=", 1)
        os.environ.setdefault(k, v.strip().strip('"').strip("'"))
```

**Claude** (Anthropic SDK) — count `system` + `messages` together:
```python
import anthropic
client = anthropic.Anthropic()                       # reads ANTHROPIC_API_KEY
n = client.messages.count_tokens(
    model="claude-sonnet-4-6",
    system=SYSTEM_PROMPT,
    messages=[{"role": "user", "content": USER_PROMPT}],
).input_tokens
```

**Gemini** (google-genai SDK) — text, and text + image:
```python
from google import genai
from google.genai import types
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])   # or GOOGLE_API_KEY

# text only
n = client.models.count_tokens(model="gemini-2.5-flash", contents=[PROMPT_TEXT]).total_tokens

# text + image — image tokens are MODEL-SPECIFIC (same image differs per model)
img = open("image_dataset/grape.jpg", "rb").read()
n_with_img = client.models.count_tokens(
    model="gemini-2.5-flash",
    contents=[types.Part.from_bytes(data=img, mime_type="image/jpeg"), PROMPT_TEXT],
).total_tokens
image_tokens = n_with_img - n
```

**Reproduce every number on this page** (assembles each activity's real prompt and
counts it; run from the demo root so the relative paths resolve):
```bash
cd /home/qtrobot/tutorials-master/demos/version_1_llm_gemini
.venv39/bin/python prompt_inventory/count_tokens.py
```

### B. Quick estimate — no key / no network

`tokens ≈ len(text) / 4` for English (±~10–15%); cross-check with `words / 0.75`.
**Do not use `tiktoken`** — it is OpenAI's tokenizer and undercounts Claude/Gemini.
Good enough to confirm “a tiny fraction of the context window,” not for exact billing.

### Then compare against the window

`utilization = prompt_tokens / context_window` — context windows are in
[`context_windows.md`](context_windows.md). For vision calls, add the image/video
tokens (additive, and model-specific — see the notes lower down).

---

## Method (provenance of the numbers below)

- Counts below are **exact**, from the providers' server-side `count_tokens` APIs
  (Anthropic `messages.count_tokens` for Claude; Google `models.count_tokens` for
  Gemini), using the API keys in `src/.env` (`ANTHROPIC_API_KEY`, `GEMINI_API_KEY`/
  `GOOGLE_API_KEY`). `count_tokens` only measures — it does not generate, and cost is
  negligible.
- Prompts were **fully assembled** with representative values (name “Alex”, ages
  3–9, gender boy, a ~95-word sample story, an 8-toy list) and the **real resolved
  knowledge-base fragment** from `LanguageInterestKB`. The **story** prompts are the
  genuine output of `StoryGenerator._build_prompt(...)`.
- **Images were measured** against a real sample frame (`image_dataset/grape.jpg`).
  Image token cost is **model-dependent** (see notes) and grows with image
  resolution; a larger uploaded scene photo costs more tiles.
- **One value is still an estimate:** the spatial-validation **video** worker (no
  sample MP4 on disk to measure) — flagged inline.
- These are **input** (prompt) tokens. Output is capped separately and never counts
  against the input window.

---

## Story prompt size, exact, per age (system + user)

| Child age | Tier | Prompt tokens | % of 1,000,000 ctx |
|---|---|---:|---:|
| 3 | early_preschool | 1,628 | 0.163% |
| 4 | wh_question_format | 1,877 | 0.188% |
| 5 | wh_question_format | 1,828 | 0.183% |
| 6 | early_school_age (+takeaways +WH) | 2,593 | 0.259% |
| 7 | early_school_age (+takeaways +WH) | 2,593 | 0.259% |
| 8 | school_age (+takeaways) | 1,889 | 0.189% |
| 9–10 | school_age (+takeaways) | 1,889 | 0.189% |

---

## Master comparison — every activity (exact)

“Image/video” is the measured additive media input. “Total %ctx” includes it.

| Activity | Model | Context window |     Text tokens | Image/video tokens | Total %ctx |
|---|---|---:|----------------:|---:|---:|
| Story generation (main) | `claude-sonnet-4-6` | 1,000,000 | 1,628–2,593  (system prompt + the assembled user prompt) | — | 0.16–0.26% |
| AI Conversation Assistant | `claude-sonnet-4-6` | 1,000,000 |  2,269 (system) | + RAG + memory (bounded, see notes) | 0.227% + |
| ASR intent correction | `claude-sonnet-4-6` | 1,000,000 |              93 | — | 0.009% |
| Story: comprehension questions | `gemini-2.5-flash` | 1,048,576 |             181 | — | 0.017% |
| Story: takeaway MCQs | `gemini-2.5-flash` | 1,048,576 |             177 | — | 0.017% |
| Story: gesture/emotion tagging | `gemini-2.5-flash` | 1,048,576 |             241 | — | 0.023% |
| Story: page splitting | `gemini-2.5-flash` | 1,048,576 |             148 | — | 0.014% |
| Story: scene identification | `gemini-2.5-flash` | 1,048,576 |             156 | — | 0.015% |
| Educational Quiz: generation | `gemini-2.5-flash` | 1,048,576 |              72 | — | 0.007% |
| Educational Quiz: feedback | `gemini-2.5-flash` | 1,048,576 |           2,134 | — | 0.204% |
| Educational Quiz: WH options | `gemini-2.5-flash` | 1,048,576 |              77 | — | 0.007% |
| Scene game: criteria question (4–6) | `gemini-2.5-flash` | 1,048,576 |             289 | — | 0.028% |
| Scene game: riddle (7+) | `gemini-2.5-flash` | 1,048,576 |             282 | — | 0.027% |
| Scene game: spatial validation (still) | `gemini-2.5-flash` | 1,048,576 |              99 | +258 (1 image) | 0.034% |
| Scene game: spatial validation (video) | `gemini-2.5-flash` | 1,048,576 |              99 | +~900–1,000 (~3 s, *est*) | ~0.10% |
| Scene game: object detection / pointing | `gemini-robotics-er-1.6-preview` | 131,072 |              86 | **+1,064 (1 image)** | 0.877% |
| Toy recovery question | `gemini-2.5-flash` | 1,048,576 |              92 | +258 (1 image) | 0.033% |
| Conversation follow-up | `gemini-2.5-flash` | 1,048,576 |             119 | + history (grows) | 0.011% + |
| WH Picture Scene: receptive | `gemini-2.5-flash` | 1,048,576 |              90 | +258 (1 image) | 0.033% |
| WH Picture Scene: expressive | `gemini-2.5-flash` | 1,048,576 |              85 | +258 (1 image) | 0.033% |
| Story sentence illustration | `gemini-2.5-flash-image` | 32,768 |              87 | +258 (ref image; larger PNGs cost more) | 1.053% |

---

## Media & variable inputs (measured facts + bounds)

- **Image tokenization is model-specific.** The *same* `grape.jpg` measured at
  **258 tokens on `gemini-2.5-flash`** but **1,064 tokens on
  `gemini-robotics-er-1.6-preview`** (the Gemini-3-based robotics model tiles images
  more finely). So the object-detection step is the heaviest single image cost in the
  system, and it runs on the smallest LLM window (131,072) — yet still only **0.88%**.
- **Bigger images cost more.** `grape.jpg` is a small frame (≈1 tile). A higher-res
  scene card / camera capture splits into more 768-px tiles, each ~258 tokens on Flash
  — still a few hundred tokens, far under 1%.
- **Video (spatial-validation video worker)** — not measured (no sample clip on disk).
  Gemini bills video at ~263 tok/s (1 fps) + audio, so a ~3 s clip ≈ ~900–1,000
  tokens → ~0.10% of ctx.
- **Story-illustration reference image** — I measured a small sample (258 tok). The
  real reference is a previously generated PNG (often ~1024 px); a worst-case ~1,290-
  token reference would put this activity at ≈ (87+1,290)/32,768 ≈ **4.2%** — the
  highest utilization in the system, still small.
- **AI assistant RAG + chat memory** — the only inputs that grow with use, and both
  are bounded: the LlamaIndex memory buffer is token-limited (`token_limit =
  max_tokens = 4096`) and RAG injects only retrieved top-k chunks (not whole PDFs).
  Realistic worst case ≈ ~10–15k tokens → ~1–1.5% of 1,000,000.

---

## Bottom line

- **Every activity uses well under 1% of its model's context window** — except the
  story-illustration model in a worst-case large-reference scenario (~4%), which is
  still tiny. No overflow risk anywhere.
- **Largest text prompts:** story at the 6–7 tier (**2,593 tok, 0.26%**) and
  quiz-feedback (**2,134 tok, 0.20%**, because it inlines the entire SAR system
  prompt). Both fit trivially.
- **Heaviest image cost:** object detection on the robotics-ER model (**1,064 image
  tokens**, 0.88% — small window + fine image tiling), worth knowing if that model is
  ever fed high-res frames.
- The estimates in the prior version were close: story within ~3%, assistant system
  was ~2,269 (not ~2,600), and the image/robotics image cost is the main thing pure
  `chars/4` could not have predicted.
