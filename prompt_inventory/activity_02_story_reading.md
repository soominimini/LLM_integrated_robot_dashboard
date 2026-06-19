# Activity 2 — Story / Read Story

- **Entry points:** page `/read_story`; APIs `/api/generate_story`, `/api/generate_story_stream`; saved-story playback under `/api/read_*`.
- **Main model:** `claude-sonnet-4-6` via `StoryGenerator` → `scripts/claude_story.py`. **Context window: 1,000,000.** Output cap `max_tokens=4096`, `temperature` not sent (Sonnet 4.6 manages it).
  - *Configurable:* `StoryGenerator(llm_model=...)` is set to `claude-sonnet-4-6` (`web_user_server.py:241`). `gemini-*` routes to `scripts/gemini_story.py` (`gemini-2.5-flash`, ctx 1,048,576, `temperature=0.8`, `max_output_tokens=4096`); any other name → local Ollama.
- **Age-varied prompt?** **Yes — this is the most age-sensitive activity.** The prompt is assembled from a master template plus an **age tier** chosen by `language_age` (if set in the profile) else chronological `age`.

The full story prompt = **system prompt** (worker) + **assembled user prompt** (`StoryGenerator._build_prompt`).

---

## System prompt (both `claude_story.py` and `gemini_story.py`), verbatim

```
You are a clinical storyteller for pediatric speech-language therapy. You create personalized therapeutic stories that are age-calibrated, clinically grounded, and engaging. You follow word count constraints precisely. You integrate therapy goals into narrative and dialogue naturally, never as explicit lessons. You never add preamble, commentary, or text outside the requested output format.
```

---

## Age tiers (`StoryGenerator.AGE_TIERS`) — the per-age variation

Tier is selected by `language_age` if present, else `age` (`_get_age_tier`). Ages
below 3 fall to the lowest tier; above 10 fall to the highest.

### Tier **age 3** — `early_preschool`, word range **50–100**
```
Language level: Use 3–5 word sentences. Repeat key phrases and sentence patterns 2–3 times for reinforcement. Use only concrete, familiar objects and animals (ball, dog, tree, cup, cat). Use onomatopoeia freely (splash, whoosh, boom, moo). Avoid abstract concepts, idioms, or figurative language. Story structure: Use a simple linear sequence (first, then, finally) with no subplots. Characters: Maximum 2–3 named characters. Dialogue: Very short exchanges, 1 sentence per turn.
```

### Tier **ages 4–5** — `wh_question_format`, word range **40–70** (uses the WH master template, below)
```
Language level: Write 3–4 short, concrete sentences in the present tense. Use simple vocabulary a 4–5 year old already knows (ball, beach, dog, bus, kite, snow, library). Each sentence states a single observable fact (who is there, where, what is happening, what surprises them). Avoid abstract concepts, idioms, figurative language, and complex compound sentences. Story structure: One small everyday scene with a tiny twist or surprise — NOT a three-act journey. Examples of scenes: building a sandcastle and finding a crab; flying a kite and the wind blowing it onto a dog; dropping toothpaste on the bathroom floor; spotting an unusual lunch in the cafeteria. Characters: 1–3 named characters; refer to them by name (not pronouns) so questions are answerable. Follow-up questions: After the story, generate 5–7 WH-questions (WHO/WHAT/WHERE ONLY — no HOW/WHY at this age) whose answers are explicitly stated in the story text — one fact per question, in roughly story order.
```

### Tier **ages 6–7** — `early_school_age`, word range **80–120**, `requires_takeaways=True`, `requires_wh_questions=True`
```
Language level: Use varied sentence structures including relative clauses and embedded phrases. Include emotional vocabulary (frustrated, proud, nervous, relieved, grateful). Weave in 3–5 target vocabulary words with natural contextual support. Model question forms and conversational turn-taking in dialogue. Story structure: Three-act structure with a secondary challenge or emotional subplot. Characters: Up to 4–5 characters with motivations and feelings. Dialogue: Natural back-and-forth exchanges of 2–3 sentences, showing perspective-taking. Comprehension questions: After the story, generate 5–7 WH-questions mixing WHO/WHAT/WHERE (concrete recall, answers appear verbatim in the story) with 2–3 HOW/WHY questions (cause, motivation, process). HOW/WHY answers may require short inference, but the inference must be clearly supported by what happens in the story.
```

### Tier **ages 8–10** — `school_age`, word range **130–200**, `requires_takeaways=True`
```
Language level: Use complex sentences with subordinate clauses. Include nuanced emotional and social vocabulary (empathy, compromise, perseverance). Introduce figurative language gently (similes, simple metaphors) with clear context. Model inferencing and perspective-taking through character thoughts and dialogue. Story structure: Three-act structure with internal conflict and character growth. Characters: Realistic motivations and interpersonal dynamics. Dialogue: Extended exchanges showing negotiation, repair, and social problem-solving.
```

**Tier-gated blocks:** ages **7+** (`requires_takeaways`) add a `** Takeaways **`
section + the takeaways prompt block; ages **6–7** (`requires_wh_questions`) add a
`** Questions **` section + the WH-questions prompt block. Ages **4–5** use the
separate WH master template entirely.

---

## Master template (ages 3, 6–7, 8–10) — `MASTER_TEMPLATE`, verbatim

Placeholders: `{child_name}`, `{age}` (chronological), `{gender}`,
`{age_guidelines}` (the tier text above), `{theme_*}` (from `THEME_GUIDANCE`),
`{goals_section}`, `{persona_section}` (knowledge-base fragment — see
`helpers_and_shared.md`), `{takeaways_block}`, `{wh_questions_block}`,
`{output_format}`.

```
Write a short therapeutic story for a {age}-year-old {gender} named {child_name}, who has speech delay. The story should be developmentally appropriate, engaging, and supportive of early language development.

--- AGE-APPROPRIATE LANGUAGE REQUIREMENTS ---
{age_guidelines}

--- STORY SETTING AND STRUCTURE ---
{theme_setting}
{theme_obstacle}
{theme_resolution}

Use a clear three-act structure:
1. BEGINNING: Introduce {child_name}, the setting, and {child_name}'s goal or desire.
2. MIDDLE: {child_name} encounters an obstacle. {child_name} meets a supportive character who helps. Show the process of overcoming the challenge together.
3. END: {child_name} achieves the goal, learns something, and feels positive about the experience.

--- VOCABULARY AND LANGUAGE TARGETS ---
{theme_vocabulary}

{goals_section}
{persona_section}{takeaways_block}{wh_questions_block}--- TONE AND STYLE ---
- Warm, encouraging, and gently paced.
- Show, don't tell: use actions and dialogue to convey emotions rather than stating them.
- Include at least one moment of humor, wonder, or sensory delight.
- Use character names consistently (avoid pronoun ambiguity for young readers).

--- ROBOT GESTURES AND EMOTIONS ---
A robot will read this story aloud and physically act it out. Embed gesture or emotion tags INLINE in the story text so the robot's face and body match what is happening in the narrative.

Available gestures (use [gesture:NAME] format):
  hi, bye, nodding-yes, clapping, hoora, happy, calm, shy, embrace, patience,
  slight_no, think, sneezing, yawn, breathing_exercise, kiss, stretching

Available emotions (use [emotion:NAME] format) — use ONLY these exact names:
  QT/happy, QT/sad, QT/surprised, QT/afraid, QT/angry, QT/calm, QT/shy

Rules for tags:
- Tag EVERY clear emotional beat. Whenever a character smiles, laughs, giggles, or feels happy/proud/excited, insert [emotion:QT/happy]. Whenever they cry, frown, or feel sad/disappointed, insert [emotion:QT/sad]. Apply the same rule for surprised, afraid, angry, calm, and shy.
- Place the tag IMMEDIATELY BEFORE the sentence that depicts the emotion or action — not at the start of the paragraph.
- It is fine to use the same emotion multiple times in one paragraph if the character feels it more than once.
- Do NOT invent emotion names. If the feeling isn't in the list above, pick the closest available one (e.g. "relieved" or "proud" → QT/happy; "frustrated" → QT/angry; "nervous" → QT/afraid).
- Use gesture tags for physical actions (waving, clapping, nodding) where they fit the story.
- Example: 'Anna looked at the puppy. [emotion:QT/happy] She smiled brightly and laughed.'
- Example: '[gesture:nodding-yes] [emotion:QT/happy] "Yes, I can help!" said the rabbit.'
- Example: 'The wind blew hard. [emotion:QT/surprised] Suddenly, a big rainbow appeared in the sky!'

{output_format}
```

If `topics` are supplied, the builder appends:
`"\n\nIncorporate the following theme(s) prominently and naturally: <topics>."`

---

## WH master template (ages 4–5) — `WH_MASTER_TEMPLATE`, verbatim

`{wh_examples}` are 2 few-shot stories sampled from
`documents/story for 4 to 7 years old/story_corpus.json`.

```
Write a short illustrated-style story for a {age}-year-old {gender} named {child_name}, who has speech delay. The story will be used by a robot to practise WH-question comprehension (WHO / WHAT / WHERE) with the child.

--- STYLE REFERENCE (from the curated 4-to-7 corpus, WHO/WHAT/WHERE subset for this age) ---
Match the style, length, vocabulary, and structure of these reference stories EXACTLY. Each is a 3–4 sentence concrete vignette in the present tense, followed by 5–7 WH-questions whose answers appear verbatim in the story.

{wh_examples}
--- END STYLE REFERENCE ---

--- AGE-APPROPRIATE LANGUAGE REQUIREMENTS ---
{age_guidelines}

--- STORY SETTING AND THEME ---
{theme_setting}
{theme_obstacle}
{theme_resolution}

--- VOCABULARY FOCUS ---
{theme_vocabulary}

{goals_section}
{persona_section}
--- TONE AND STYLE ---
- Warm, simple, and concrete. Describe one small everyday scene with a tiny surprise or twist.
- {child_name} should appear by name in the story (not just "she" / "he"), so WHO-questions are answerable.
- Use observable facts only (who is there, where they are, what they are doing, what they see/find).
- Do NOT invent moral lessons, internal monologue, or three-act structure. Keep it to the corpus style.

--- ROBOT GESTURES AND EMOTIONS ---
A robot will read this aloud and act it out. Embed gesture or emotion tags INLINE in the story so the robot's face and body match the narrative. Keep these SPARSE (at most 2–3 tags total) so the story stays short.

Available gestures: [gesture:NAME] where NAME ∈ {hi, bye, nodding-yes, clapping, hoora, happy, calm, shy, embrace, patience, slight_no, think, sneezing, yawn, breathing_exercise, kiss, stretching}
Available emotions: [emotion:NAME] where NAME ∈ {QT/happy, QT/sad, QT/surprised, QT/afraid, QT/angry, QT/calm, QT/shy}

Place a tag IMMEDIATELY before the sentence that depicts the emotion or gesture. Do NOT include tags inside the WH-questions.

{output_format}
```

If `topics` are supplied, the builder appends:
`"\n\nIncorporate the following theme(s) into the scene: <topics>."`

---

## Output-format blocks (injected as `{output_format}`)

### `OUTPUT_FORMAT` (master template; word bounds + sections injected)
```
Return your answer EXACTLY in this format, with no other text before or after:

** Title **
<one short title>

<the full story text>

** End **
{takeaways_section}{wh_questions_section}** Explanation of the output **
<1–3 short sentences explaining how the story matches the selected topic(s) and the learning goals: {goals}>

STRICT RULES:
- HARD WORD LIMIT: The story text between ** Title ** and ** End ** MUST be {min_words}–{max_words} words. {max_words} is a CEILING you cannot exceed. Target around {mid_words} words so you have safety margin. Count as you write; if you approach {max_words}, wrap up immediately at the next natural sentence — do not start a new subplot, paragraph, or piece of dialogue.
- Brevity beats completeness. Cut adjectives, side details, and extra dialogue exchanges before going over.
- Do NOT include any preamble, commentary, or meta-text (no "Here is your story", "Sure!", etc.).
- Do NOT add sections beyond Title, story text, End,{takeaways_in_rule}{wh_questions_in_rule} and Explanation.
Do not add any other sections or extra text.
```

### `WH_OUTPUT_FORMAT` (ages 4–5)
```
Return your answer EXACTLY in this format, with no other text before or after:

** Title **
<one short title>

<the full story text — 3 to 4 short concrete sentences in the present tense>

** End **
** Questions **
1. <WH-question>
2. <WH-question>
3. <WH-question>
4. <WH-question>
5. <WH-question>
(6. and 7. optional)

** Explanation of the output **
<1–2 short sentences explaining how the story and questions match the selected topic(s) and the learning goals: {goals}>

STRICT RULES:
- Do NOT include any preamble, commentary, or meta-text (no "Here is your story", "Sure!", etc.).
- The story text (between Title and End) MUST be between {min_words} and {max_words} words.
- Generate {min_questions}–{max_questions} WH-questions using ONLY WHO / WHAT / WHERE (no HOW / WHY at this age tier).
- Every question's answer MUST appear word-for-word in the story text. Do NOT ask about anything not stated.
- Refer to characters by name (not pronouns) in the story so questions like "WHO is in the story?" are answerable.
- Do NOT add sections beyond Title, story text, End, Questions, and Explanation.
```

### `TAKEAWAYS_OUTPUT_BLOCK` (ages 7+, into `{takeaways_section}`)
```
** Takeaways **
- <takeaway 1: one short, kid-friendly sentence stating a lesson the child should learn>
- <takeaway 2: another lesson the story demonstrates>
- <takeaway 3: another lesson (optional)>
```

### `WH_QUESTIONS_OUTPUT_BLOCK` (ages 6–7, into `{wh_questions_section}`)
```
** Questions **
1. <WH-question — WHO/WHAT/WHERE/HOW/WHY>
2. <WH-question>
3. <WH-question>
4. <WH-question>
5. <WH-question>
(6. and 7. optional)
```

### `TAKEAWAYS_PROMPT_BLOCK` (ages 7+, into `{takeaways_block}`)
```
--- TAKEAWAYS / LESSONS ---
The story MUST teach 2–3 clear takeaways that the child can articulate after listening. Demonstrate them through what characters DO and the consequences they face — do NOT preach or moralise inside the narration. After the story ends, list the takeaways explicitly in a ** Takeaways ** section.

Takeaway rules:
- Each takeaway is ONE short sentence a 7-year-old can repeat back in their own words.
- Phrase them as positive, actionable lessons or values — not as "do not" prohibitions when a positive form is natural.
- Each takeaway must connect to a specific moment or decision in the story (cause → consequence).
- Cover values like honesty, kindness, perseverance, asking for help, sharing, courage, patience, listening, or fairness — pick the ones that fit the plot.

Examples of well-formed takeaways:
- "Asking for help is a sign of being smart, not weak."
- "Being patient often works better than rushing."
- "Telling the truth is hard but builds trust."
- "Treating others kindly makes them want to help you back."
```

### `WH_QUESTIONS_PROMPT_BLOCK` (ages 6–7, into `{wh_questions_block}`)
`{fable_examples}` are 2 few-shot HOW/WHY examples mined from `story_corpus.json`.
```
--- COMPREHENSION QUESTIONS (WHO/WHAT/WHERE + HOW/WHY) ---
After the story, generate 5–7 WH-questions a clinician or robot can use to check the child's comprehension:
- Include 3–4 WHO/WHAT/WHERE questions whose answers appear verbatim in the story (concrete recall).
- Include 2–3 HOW/WHY questions about cause, motivation, or process. Short inference is allowed, but the inference MUST be clearly supported by what happens in the story.
- Phrase each question so a 6–7 year old can answer it in 1 short sentence. Avoid yes/no questions.

HOW/WHY example questions (from the curated fable corpus — same style for {child_name}'s story):
{fable_examples}
```

---

## Theme guidance (`THEME_GUIDANCE`) — injected as `{theme_setting/obstacle/resolution/vocabulary}`

Selected by the chosen `topics`; multiple matching topics are merged; default used
if none match. Keys: `season`, `school`, `family`, `friends`, `animals`,
`adventure`. (Example — `adventure`:)
```
setting:    Set the story in an imaginative but safe environment (enchanted garden, friendly forest, treasure map).
obstacle:   The obstacle should involve solving a puzzle, finding something, or navigating a path.
resolution: The resolution should reward curiosity, bravery, and persistence.
vocabulary: Emphasize spatial and action vocabulary: behind, through, under, climb, discover, search.
```
Default (no topic match):
```
setting:    Set the story in a familiar, child-friendly environment.
obstacle:   The obstacle should involve a manageable challenge that requires help from others.
resolution: The resolution should leave the protagonist feeling proud, grateful, and connected.
vocabulary: Use rich descriptive vocabulary appropriate to the setting and characters.
```

## Goals section (`GOALS_SECTION_TEMPLATE` + `DEFAULT_GOALS`)

With clinician goals supplied:
```
Therapy goals to integrate naturally into the story (do NOT list them explicitly — weave them into narrative, dialogue, and action):

- Clinician-specified goals: <goals>
  Integrate these naturally through story events, character dialogue, and descriptive language.
  Also include the following foundational goals:
  1) learning descriptive words, 2) collaboration, 3) importance of friendships and relationships, 4) overcoming challenges.
```
Default (no goals):
```
Therapy goals to integrate naturally into the story (do NOT list them explicitly — weave them into narrative, dialogue, and action):

- Learning descriptive words (adjectives, spatial terms)
- Collaboration and turn-taking
- Importance of friendships and relationships
- Overcoming challenges with support
```

## Post-generation shortener (`SHORTEN_TEMPLATE`)

Run only when a saved story overshoots its age word cap. Same model as generation.
```
You are shortening a children's story so it fits a strict word cap.

ORIGINAL STORY (with inline [gesture:...] and [emotion:...] tags):
"""
{body}
"""

REWRITE RULES:
- Target length: {min_words}–{max_words} words. The {max_words} ceiling is HARD; aim for around {mid_words} words.
- Preserve the plot, character names ({names_hint}), beginning, and ending.
- Preserve the existing inline [gesture:...] and [emotion:...] tags — keep them attached to the same emotional/action beats.
- Cut adjectives, side details, redundant dialogue lines, and any subplot first.
- Do NOT add new characters, events, or sections.
- Do NOT add commentary, preamble, or markdown headings — just return the rewritten story body.

Return ONLY the rewritten story body. No "Here is" preamble. No ** Title **, ** End **, takeaways, or explanation — just the story text.
```

---

# Story sub-passes (post-processing) — all on `gemini-2.5-flash` (ctx 1,048,576) via `_gemini_generate`

These run after the story is generated, to build the interactive reading
experience. Each lists its own `system` string.

## 2a. Comprehension questions — **age-varied (3 bands)**

`complexity_age = language_age or child_age`. System:
`"You generate comprehension questions for children's stories. Return JSON only."`
(`max_tokens=4096`).

Band guidance (`{detail_guidance}` injected into the prompt):

- **complexity_age ≤ 4** → 3 questions:
```
- 1 main idea question (e.g. 'What was the story about?')
- 2 detail questions about characters, events, or objects in the story
Use very simple language with short sentences (3-6 words per question).
Keep answer options very short (1-5 words each).
```
- **complexity_age ≤ 6** → 4 questions:
```
- 1 main idea question (e.g. 'What was the main thing that happened in the story?')
- 3 detail questions about characters, events, settings, or objects
Use simple language appropriate for a 5-6 year old.
Keep answer options short (1-8 words each).
```
- **else (7–12)** → 5 questions incl. inference:
```
- 1 main idea question (e.g. 'What is the main message of this story?')
- 2 detail questions about characters, events, settings, or sequence of events
- 2 inference questions that ask the child to think deeper, such as:
  * 'Why do you think [character] felt that way?'
  * 'What do you think would have happened if...?'
  * 'How do you think [character] felt when...?'
  * 'Why did [character] decide to...?'
Use age-appropriate language for a 7-12 year old.
Keep answer options concise (1-12 words each).
```

Prompt template (verbatim):
```
You are creating comprehension questions for a story read by a {child_age}-year-old child named {child_name}.

Story:
{cleaned}
{persona_block}
Generate exactly {num_questions} questions:
{detail_guidance}

For each question, provide:
- 1 correct answer
- 2 plausible but incorrect answers
The wrong answers should be believable but clearly wrong based on the story.

Return ONLY a JSON array of objects. Each object has:
- "question": the question text
- "type": one of "main_idea", "detail", or "inference"
- "correct_answer": the correct answer text
- "wrong_answers": an array of exactly 2 incorrect answer texts

Example: [{"question": "What was the story about?", "type": "main_idea", "correct_answer": "A boy who helped his friend", "wrong_answers": ["A girl who went swimming", "A cat who got lost"]}]
```

## 2b. Takeaway multiple-choice questions — age in text (soft)

System: `"You write children's multiple-choice comprehension questions. Return JSON only."` (`max_tokens=1536`). `{numbered}` is the verbatim takeaways list.
```
You are creating multiple-choice LESSON questions for a {child_age}-year-old child named {child_name}. The story below has {len} takeaways. Create exactly ONE question per takeaway.

Story:
{cleaned_story}

Takeaways (each is the CORRECT answer for one question — use verbatim):
{numbered}

For EACH takeaway, produce one question with:
- A natural, kid-friendly QUESTION STEM. Vary the phrasing across questions. Pick whichever
  fits best:
    * "What can you learn from their behavior?"
    * "What is one lesson from this story?"
    * "What did [character name] show us by what they did?"
    * "Why was it a good idea to ...?"
    * "What can we do like [character]?"
- The CORRECT answer: the takeaway, exactly as given above (do NOT rephrase, paraphrase, or shorten it).
- 2 WRONG answers: plausible-but-clearly-wrong lessons. They must:
    - Sound like reasonable lessons in general but be wrong for THIS story.
    - Be similar length to the correct answer.
    - NOT be the OTHER takeaways from the list above (those are also correct).
    - NOT be opposites or trivially wrong ("You should never be kind").

Return ONLY a JSON array of {len} objects in the same order as the takeaways above. Each object has keys: "question", "correct_answer", "wrong_answers" (array of exactly 2 strings).
Example shape:
[{"question": "What can you learn from their behavior?", "correct_answer": "<takeaway 1 verbatim>", "wrong_answers": ["<distractor a>", "<distractor b>"]}, ...]
```

## 2c. Gesture / emotion tagging — not age-varied

System: `"You add inline gesture/emotion tags to children's stories. Return only the tagged story."` (`temperature=0.2`, `max_tokens=4096`).
```
You are tagging a children's story for a robot that will read it aloud while showing matching facial expressions and gestures.

TASK: Return the SAME story word-for-word, but with [gesture:NAME] and [emotion:NAME] tags inserted immediately before the sentence that depicts the emotional beat or physical action.

ALLOWED EMOTION NAMES (use ONLY these — exact match):
  QT/happy, QT/sad, QT/surprised, QT/afraid, QT/angry, QT/calm, QT/shy

ALLOWED GESTURE NAMES:
  hi, bye, nodding-yes, clapping, hoora, happy, calm, shy, embrace,
  patience, slight_no, think, sneezing, yawn, breathing_exercise,
   kiss, stretching

RULES:
- Tag EVERY clear emotional beat. Whenever a character smiles, laughs,
  giggles, or feels happy/proud/excited/relieved/grateful, insert
  [emotion:QT/happy]. Whenever they cry, frown, or feel
  sad/disappointed/lonely, insert [emotion:QT/sad]. Same rule for
  surprised, afraid (scared/nervous/worried), angry (frustrated/mad),
  calm (peaceful/content), shy (embarrassed/bashful).
- Never invent emotion names. If a feeling is not in the allowlist,
  pick the closest one.
- Place each tag IMMEDIATELY BEFORE the sentence it describes —
  not at the start of the paragraph. The same emotion may appear
  multiple times in one paragraph if the character feels it more
  than once.
- Use gesture tags for physical actions where they fit (waving,
  clapping, nodding, hugging, stretching, etc.).
- DO NOT change, add, remove, rephrase, or reorder ANY of the
  original words. Only insert tags.
- If the input already contains tags, KEEP correct ones, FIX invalid
  emotion names by remapping to the allowlist, and ADD missing tags
  for emotional beats that are currently untagged.
- Return ONLY the tagged story text. No JSON, no explanation, no
  preamble, no code fences.

STORY:
{story_text}
```

## 2d. Page splitting — **age-varied (3 bands)**

System: `"You split stories into pages. Return JSON only."` (`max_tokens=4096`).
`{sents_per_page}`: `age ≤4` → "about 1 to 2"; `≤6` → "about 2 to 3"; else "about 3 to 5".
```
Split the following story into pages for a {child_age}-year-old child.

PRIORITIES (in order):
1. Narrative flow and context come FIRST. Keep sentences that belong to
   the same scene, moment, or train of thought together on the SAME page.
   Never split a continuous scene across pages just to hit a sentence count.
2. A page break must fall at a natural scene/context shift — a change of
   setting, time, character focus, or action.
3. As a SOFT target, aim for {sents_per_page} sentences per page. It is
   acceptable to go slightly over (or under) this target when the scene
   demands it. Do NOT force a split mid-scene to satisfy the count.

HARD RULES:
- Keep every sentence intact and do NOT rephrase or change any words.
- Do not split a single sentence across two pages.
- PRESERVE all [gesture:...] and [emotion:...] tags VERBATIM in their
  exact original positions. Do NOT remove, move, rewrite, or reformat
  any tag. A tag stays attached to the sentence that follows it; if
  that sentence moves to a new page, the tag moves with it.
- Paragraph breaks in the input are strong scene hints — prefer to
  split at paragraph boundaries when possible.
- Return ONLY a JSON array of strings, where each string is one page.
- Example: ["Page 1 text here.", "[emotion:QT/happy] Page 2 text."]

Story:
{cleaned_for_llm}
```

## 2e. Scene identification (for choosing illustrations) — not age-varied

System: `"You analyze story structure. Return JSON only."` (`max_tokens=2048`). `{unit_label}` defaults to "paragraph".
```
You are choosing illustrations for a children's story that has been split into {len} {unit_label}s.

For each {unit_label}, decide which SCENE it depicts. Two {unit_label}s should share a scene ONLY IF they show essentially the same visual moment — same setting, same characters present, and similar action. If anything important changes (location, who is on screen, what they are doing), it is a NEW scene.

DEFAULT BIAS: assume each {unit_label} is its own scene. Only merge {unit_label}s when the same illustration would clearly work for both. Do not over-merge — we want roughly one image per {unit_label} unless they are truly the same visual.

{full_text}

Return ONLY a JSON object with:
- "scenes": an array of short visual descriptions (1-2 sentences each) describing what should be illustrated for each scene. Focus on setting, characters, and key action. There must be AT MOST {len} scenes.
- "chunk_to_scene": an array of {len} integers, where each integer is the 0-based scene index for that {unit_label}.

Example for 4 {unit_label}s with 3 scenes ({unit_label}s 0 and 1 share scene 0 because they're a single conversation in the same kitchen):
{"scenes": ["Mom and Lily sitting at a sunny kitchen table eating breakfast", "Lily walking to school carrying her red backpack along a tree-lined sidewalk", "Lily showing her drawing to the class at the front of the classroom"], "chunk_to_scene": [0, 0, 1, 2]}
```

## 2f. Sentence illustrations
See [`activity_07_image_generation.md`](activity_07_image_generation.md) — `gemini-2.5-flash-image`, ctx 32,768. Not age-varied.
