# Activity 3 — Educational Quiz

- **Entry points:** pages `/quiz_generation`, `/educational_quiz`; APIs `/api/generate_quiz`, `/api/generate_quiz_feedback`, `/api/generate_wh_options`, `/api/teach_quiz_answer`, `/api/save_quiz`, `/api/get_saved_quiz`.
- **Model:** `gemini-2.5-flash` via `_GeminiQuizLLM` (→ `_gemini_generate` → `scripts/gemini_general.py`). **Context window: 1,048,576.** Quiz LLM created with `max_tokens=8192`; generation `temperature=0.3`.
  - *Note:* the route docstring says “Generate quiz questions using Llama,” but the implementation uses Gemini Flash (`_GeminiQuizLLM`). The comment is stale.
- **Age-varied prompt?** **Yes** — via the `difficulty` field, and a separate social-rules branch.

## Quiz LLM system role (`_GeminiQuizLLM.SYSTEM_ROLE`, verbatim)

```
You create short, child-friendly quiz questions. Return JSON only. Each item must be {"question": "...", "type": "yes_no"|"wh"}.
```
(Passed as `--system` to `gemini_general.py`. Individual routes below override the user prompt and may pass their own `system`.)

## 3a. Quiz generation (`/api/generate_quiz`) — **age-varied**

Age is set by `difficulty` → `{age_hint}`:
- `low` → `"Target ages 2-3."`
- `med` → `"Target ages 4-5."`
- `high` → `"Target ages 7+."`

`{rule_text}` depends on selected types:
- yes_no only → `"Rules: yes_no questions must be answerable with a clear yes or no (correct/incorrect)."`
- wh only → `"Rules: wh questions must begin with one of: what, when, where, why, who, how."`
- both → `"Rules: questions must be answerable with a clear yes or no (correct/incorrect). questions must begin with one of: what, when, where, why, who, how."`

`{goal_text}` / `{length_constraint}` have two variants:

**Default branch:**
```
Goal: Questions must be objectively True or False based on basic object functions or category labels. Avoid subjective questions like 'Do you like school?' or 'Are there toys?'.
```
length: `Constraint: Questions must be short (under 8 words).`

**Social-rules branch** (topic mentions social rules/etiquette/manners/kindness/behaviour AND yes_no selected; designed for age 7+):
```
Goal: Generate yes/no questions about social rules, etiquette, kindness, and basic social norms that a child should learn. Every question MUST have a clear, widely accepted yes-or-no answer — not an opinion or gray-area. The aim is to make children think about right and wrong behavior and learn social rules. Cover a mix of: physical kindness (no hitting/kicking/pushing), sharing and taking turns, polite words (please, thank you, sorry, excuse me), classroom behavior (listening, raising hands, waiting), respect for others' belongings, helping others, and basic honesty. Examples of GOOD questions and their answers: 'Is it okay to kick your friend?' → no. 'Should you say thank you when someone helps you?' → yes. 'Is it okay to take a toy without asking?' → no. 'Should you wait your turn in line?' → yes. 'Is it polite to interrupt someone speaking?' → no. 'Should you say sorry when you hurt someone?' → yes. 'Is it okay to laugh at someone who made a mistake?' → no. 'Should you share with a friend who has none?' → yes. AVOID opinion or vague questions like 'Do you like sharing?', 'Is school fun?', or 'Should you always be nice?' (the word 'always' makes it too strong).
```
length: `Constraint: Questions must be short (under 12 words) and use simple language a 7-year-old understands.`

### Full prompt template (verbatim)
```
Act as a pediatric educator. Create {count} questions about the topic(s) '{topic_text}'. {age_hint} Use only these types: {type_hint}. {goal_text} {length_constraint} Return Format: Respond with ONE JSON array ONLY. The first non-whitespace character of your response MUST be '[' and the last MUST be ']'. Do NOT wrap the array inside an object (e.g. do NOT use {"questions": [...]}). Do NOT add commentary, markdown, or code fences. Each array element must be an object with keys: 'question', 'type', 'correct_answer', 'accepted_answers'. For yes_no, correct_answer must be 'yes' or 'no' and accepted_answers should be omitted. For wh, correct_answer is the primary short answer and accepted_answers must be a list of all reasonably correct alternative answers (synonyms, related valid answers, plural/singular forms). Example: if question is 'Where do kids read books in school?', correct_answer is 'classroom' and accepted_answers could be ['classroom', 'library', 'reading room', 'classrooms']. {rule_text}
```

### Accepted-answers follow-up (`alt_prompt`, for WH items missing alternatives) — not age-varied
```
For each question below, generate a list of all reasonably correct alternative answers that a child might give. Include the original answer, synonyms, plural/singular forms, and semantically valid alternatives. Return a JSON array where each element is a list of accepted answer strings, in the same order as the input. Example input: [{"question": "Where do kids read books?", "correct_answer": "classroom"}] Example output: [["classroom", "classrooms", "library", "reading room", "school"]] Input: {alt_input_json}
```

## 3b. Quiz feedback phrases (`/api/generate_quiz_feedback`) — not age-varied

Prepends the full `documents/sar_system_prompt.md` text as `{system_context}` inside
the prompt (the SAR prompt is reproduced in
[`activity_01_ai_conversation_assistant.md`](activity_01_ai_conversation_assistant.md)).
Model `gemini-2.5-flash`.
```
You are a socially assistive robot in a pediatric therapeutic setting. Here is your system prompt for context:
{system_context}

Generate 10 short, varied, child-friendly phrases for when a child answers a quiz question CORRECTLY, and 10 short, varied, child-friendly phrases for when a child answers INCORRECTLY. Follow these rules:
- Each phrase must be 2-8 words maximum
- Use warm, encouraging, effort-focused language
- For incorrect: be gentle, never shaming. Encourage trying again
- Vary the style: some excited, some calm, some playful
- Do not use emojis
Return JSON only: {"correct": ["...", ...], "incorrect": ["...", ...]}
```

## 3c. WH multiple-choice distractor options (`/api/generate_wh_options`) — not age-varied

`{distractor_count}` = `num_options - 1` (2–3). `{llm_input}` is the list of WH questions + correct/accepted answers. Model `gemini-2.5-flash`.
```
You are creating multiple-choice options for a child's quiz. For each question below, generate exactly {distractor_count} short, plausible-but-WRONG answer options that a child might consider. Rules:
- Each option must be 1-3 words, child-friendly, and clearly different from the correct answer and from any of its accepted_answers.
- Options should be in the same category as the correct answer (e.g. if the answer is an animal, give other animals).
- Do NOT include the correct answer or any accepted answer in the distractors.
- Do NOT include duplicates.
Return JSON only: a list of lists, in the same order as the input. Each inner list must contain exactly {distractor_count} distractor strings.
Input: {llm_input_json}
```

## 3d. Teach answer / save quiz
`/api/teach_quiz_answer` and `/api/save_quiz` persist user-taught alternative
answers and the generated question sets to disk. They do **not** call the LLM, so
there is no prompt.
