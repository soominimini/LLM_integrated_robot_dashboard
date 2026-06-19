# Helpers & shared prompt fragments

## H1. ASR intent correction (`_ensure_intent_llm` / `_llm_canonicalize_heard`)

- **Where used:** corrects ASR mishearings against an expected word/phrase (e.g. confirming a child said the target toy) across the games.
- **Model:** `claude-sonnet-4-6` via `ChatWithRAG(... disable_rag=True, max_tokens=128)`. **Context window: 1,000,000.** Only invoked when a quick fuzzy match (`difflib` ratio ≥ 0.85) fails.
- **Age-varied?** No.

### System role (verbatim)
```
You correct ASR mishearings for a child's therapy robot. Decide if the transcript likely intended the target word(s) given the immediate context. Be conservative; only match when highly likely. Respond strictly in compact JSON: {"match": true|false, "canonical": "<canonical or expected>"}.
```

### User prompt (verbatim)
```
Expected: '{expected}'
Heard: '{heard}'
Context: '{ctx}'
Answer in JSON only with keys match (true/false) and canonical.
```

---

## H2. Age/gender knowledge-base fragments (`src/knowledge_base.py`)

These are **not** standalone activities — they are prompt fragments **injected into
other activities' prompts** (`{persona_section}` in the story templates, and the
`persona_context` passed to the scene-game and comprehension-question generators).
They are the main reason age (and gender) changes generated content beyond the
explicit age tiers.

> **It IS in the prompt.** `knowledge_base = LanguageInterestKB()` is created at
> `web_user_server.py:242` and injected via `_persona_context_for(...)` into story
> generation (`:1319`/`:1391`), comprehension questions (`:2840`/`:5527`), and
> scene-game question generation (`:3743`/`:6164`). The section below shows only the
> wrapper **template**; for the **actual resolved data per age** (MLU ranges, target
> lists, interest themes) see [`knowledge_base_data.md`](knowledge_base_data.md).

- **Source data:** `documents/Simple_version_slp_codesign_knowledge_base.json`.
- **Driven by:** `language_age` (or chronological age) for the developmental level / MLU / language targets; chronological age + gender for interest themes. Built by `_persona_context_for(username, age, kind=...)` in the server.

### Story fragment (`build_story_prompt_fragment`) — template (verbatim)
```
--- LANGUAGE & INTEREST GUIDANCE (knowledge base) ---
Target developmental level: age {level_age}, approx MLU {mlu_range} words per utterance. Keep sentences at or near this length.

Weave these language targets naturally into narration and dialogue (model them in context; do not drill or quiz them in the story):
{targets}

Use these interest themes as story hooks, characters, and settings:
- {interests}
```

### Question fragment (`build_question_prompt_fragment`) — template (verbatim)
```
--- LANGUAGE & INTEREST GUIDANCE (knowledge base) ---
Target level: age {level_age}, approx MLU {mlu_range} words. Match question wording to this length.

Embed these language targets in the question wording where natural:
{targets}

Draw question content from these interests: {interests}
```

`{level_age}`, `{mlu_range}`, `{targets}` (a bulleted list of language targets +
examples), and `{interests}` are all resolved from the JSON by age/gender. The
developmental level is the highest entry whose `age` ≤ the child's age.

---

## H3. Persona RAG fragments (`src/persona_rag.py`) — legacy alternative

`persona_rag.py` is the **older** persona-matching approach (superseded by
`knowledge_base.py`, per that module's own docstring). It matches one of a few
fixed clinical personas by `(age, diagnosis)` and formats therapy goals/interests/
constraints into a prompt block. **It is not imported by `web_user_server.py`** —
only `LanguageInterestKB` is — so it is **not** on the active prompt path. Its
fragment headers are kept here for completeness only (verbatim):

### `build_story_context` header
```
--- PERSONA CONTEXT (retrieved reference profile) ---
Reference persona: {name} ({age_display}) — {dx_primary}.
Use this persona's therapy goals and interests to shape the story. Do not mention the persona's name; instead adapt the narrative to the actual child's name and age.

Therapy goals (high-level):
{goals_summary}

Structured language targets (weave naturally into narration and dialogue):
{structured_block}

Interests to use as story hooks and settings:
{interests_block}

Persona-specific constraints (must respect):
{constraints_block}
```

### `build_question_context` header
```
--- PERSONA CONTEXT (retrieved reference profile) ---
Reference persona: {name} ({age_display}) — {dx_primary}.
Tailor question wording and content using this persona's therapy goals and interests. Respect the constraints below.

Therapy goals (high-level):
{summary}

Structured language targets to embed:
{structured_block}

Interests to draw from: {interests_line}

Persona-specific constraints:
{constraints_block}
```

Source data: `documents/personas_rag.json`. Matching uses diagnosis-keyword
overlap + age proximity (2-point penalty per year of age difference).

---

## H4. General Gemini helper default system (`_gemini_generate`)

When a caller does not pass its own `--system`, `scripts/gemini_general.py` uses:
```
You are a helpful assistant. Return JSON only when asked.
```
In practice every story/quiz/scene caller passes its own `system` string (shown in
the respective activity files), so this default is rarely the effective one.
