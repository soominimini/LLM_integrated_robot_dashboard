# Structured knowledge base — actual content & per-age resolved fragments

This documents the **structured knowledge base** itself (the data) and the exact
text it produces once resolved for a child's age/gender — i.e. the real content
that fills the `{persona_section}` / `persona_block` / `goals_clause` placeholders
referenced in the activity files.

- **Source data:** `documents/Simple_version_slp_codesign_knowledge_base.json`
- **Loader:** `src/knowledge_base.py` → `LanguageInterestKB`
- **Instantiated:** `web_user_server.py:242` → `knowledge_base = LanguageInterestKB()`
- **Builder used by the server:** `_persona_context_for(username, age, kind)` →
  `knowledge_base.build_story_prompt_fragment(...)` / `build_question_prompt_fragment(...)`

## Where it IS injected (confirmed in code)

| Activity | Call site | Placeholder it fills |
|---|---|---|
| Story generation | `web_user_server.py:1319` (and stream `:1391`) → `generate_story(..., persona_context=...)` | `{persona_section}` in the story master/WH templates |
| Story comprehension questions | `:2840` / `:5527` → `_generate_story_questions(..., persona_context=...)` | `persona_block` in the comprehension prompt |
| Scene / object-game question generation | `:3743` / `:6164` → `_scene_game_generate_question(..., persona_context=...)` | appended into `goals_clause` |

**Selection logic:** the developmental level (MLU + language targets) is chosen by
`language_age` if the profile sets it, else chronological age — “highest level
whose `age` ≤ the child’s age.” Interest themes are chosen by **chronological age +
gender**. If no level resolves (age below the lowest entry with empty data), the
fragment is an empty string and nothing is injected.

## Where it is NOT injected

Quiz (generation/feedback/WH options), WH Picture Scene, toy recovery, conversation
follow-up, story gesture-tagging / page-splitting / scene-identification, image
generation, and ASR intent correction receive **no** knowledge-base content.

---

## 1. The data (from the JSON)

### Developmental levels (age → MLU → target keys)
| Level age | MLU range | Target keys |
|---|---|---|
| 2 | 1-3 | action_verbs_basic, basic_adjectives, pronouns_early |
| 3 | 4 | action_verbs_higher_level, basic_adjectives, pronouns_early, location_prepositions, size_noun, person_noun |
| 4 | 5-6 | action_verbs_higher_level, basic_adjectives, pronouns_later, location_prepositions, size_noun, location_noun, ing_form |
| 5 | 6-8 | plural_forms, ing_form, pronouns_later, location_noun, simple_conjunction |
| 6 | 8-10 | plural_forms, ing_form, advanced_conjunction |
| 8 | 11+ | irregular_past_tense, advanced_conjunction, auxiliary_verbs, copula_verbs, verb_tense_concepts |

### Language targets (key → description → examples)
- `action_verbs_basic` — Common everyday actions: eat, drink, sleep, run, jump, sit
- `action_verbs_higher_level` — Less frequent and cognitively more complex verbs: climb, chase, carry, deliver, rescue, build, search, discover
- `basic_adjectives` — Descriptive words for size, color, and shape: big, small, long, short, red, blue, green, yellow, round, square, triangle-shaped
- `pronouns_early` — Early personal and possessive pronouns: I, me, my, mine, you
- `pronouns_later` — Later developing pronouns: he, she, they, her, his, their
- `location_prepositions` — Spatial relationships: in, on, under
- `size_noun` — Adjective + noun combinations: big pig, small monkey, big dinosaur
- `person_noun` — Possessive determiner + noun combinations: my apple, your car, his dinosaur
- `location_noun` — Location concept expressed with a noun phrase: in the box, on the table, under the chair
- `plural_forms` — Regular plural nouns: dogs, cars, apples, cats, toys
- `ing_form` — Present participle/progressive verb form: running, jumping, eating, playing
- `simple_conjunction` — Early conjunction use: and, or
- `intermeidate_conjunction` — pre-school conjunction use: because, but, so, if  *(defined but not referenced by any level)*
- `advanced_conjunction` — More complex conjunction structures: while, although, before, after
- `irregular_past_tense` — Irregular past tense verbs: went, ate, saw, gave
- `auxiliary_verbs` — Helping verbs: is, are, was, were, has, have
- `copula_verbs` — Linking verbs: is, am, are, was, were
- `verb_tense_concepts` — Past, present, future, and progressive tense usage: I play, I played, I will play, I am playing

### Interest themes (key → items)
animals (pig, dog, cat, horse, rabbit) · pets (dog, cat, hamster, fish) · vehicles
(car, truck, bus, train, airplane) · dinosaurs (trex, triceratops, stegosaurus) ·
princesses (princess, castle, crown) · fairies (fairy, magic wand, wings) ·
superheroes (superhero, cape, rescue mission) · video_game_characters (game hero,
game creature, game vehicle) · pretend_play (tea party, grocery store, doctor, chef)

### Interest preferences (age → girls / boys theme keys)
| Pref age | girls | boys |
|---|---|---|
| 2 | *(none)* | dinosaurs, vehicles |
| 3 | unicorns, animals, pets, princesses, pretend_play | pets, pretend_play, vehicles |
| 5 | fairies, princesses, pets, superheroes | pets, vehicles, dinosaurs, superheroes |
| 6 | fairies, superheroes | superheroes, vehicles |
| 8 | fairies | superheroes, video_game_characters, vehicles |

> Note: `unicorns` appears in the age-3 girls list but has **no** entry in `themes`,
> so it renders as the bare word “unicorns” (no item list). Unknown/blank gender
> merges both buckets (deduped, girls first).

**Age → which rows are picked** (highest entry ≤ age): level uses `[2,3,4,5,6,8]`;
interests use `[2,3,5,6,8]`.
- age 3 → level 3, interests 3 · age 4 → level **4**, interests 3 · age 5 → level 5, interests 5
- age 6 → level 6, interests 6 · age 7 → level **6**, interests **6** · age 8–10 → level 8, interests 8

---

## 2. Resolved STORY language-target block, per developmental level (verbatim)

This is the middle of the story fragment (`build_story_prompt_fragment`). The full
fragment wraps it as shown in §4.

**Level age 2 (MLU 1-3)**
```
- action verbs basic (Common everyday actions): e.g. eat, drink, sleep, run, jump, sit
- basic adjectives (Descriptive words for size, color, and shape): e.g. big, small, long, short, red, blue, green, yellow, round, square, triangle-shaped
- pronouns early (Early personal and possessive pronouns): e.g. I, me, my, mine, you
```
**Level age 3 (MLU 4)**
```
- action verbs higher level (Less frequent and cognitively more complex verbs): e.g. climb, chase, carry, deliver, rescue, build, search, discover
- basic adjectives (Descriptive words for size, color, and shape): e.g. big, small, long, short, red, blue, green, yellow, round, square, triangle-shaped
- pronouns early (Early personal and possessive pronouns): e.g. I, me, my, mine, you
- location prepositions (Spatial relationships): e.g. in, on, under
- size noun (Adjective + noun combinations): e.g. big pig, small monkey, big dinosaur
- person noun (Possessive determiner + noun combinations): e.g. my apple, your car, his dinosaur
```
**Level age 4 (MLU 5-6)**
```
- action verbs higher level (Less frequent and cognitively more complex verbs): e.g. climb, chase, carry, deliver, rescue, build, search, discover
- basic adjectives (Descriptive words for size, color, and shape): e.g. big, small, long, short, red, blue, green, yellow, round, square, triangle-shaped
- pronouns later (Later developing pronouns): e.g. he, she, they, her, his, their
- location prepositions (Spatial relationships): e.g. in, on, under
- size noun (Adjective + noun combinations): e.g. big pig, small monkey, big dinosaur
- location noun (Location concept expressed with a noun phrase): e.g. in the box, on the table, under the chair
- ing form (Present participle/progressive verb form): e.g. running, jumping, eating, playing
```
**Level age 5 (MLU 6-8)**
```
- plural forms (Regular plural nouns): e.g. dogs, cars, apples, cats, toys
- ing form (Present participle/progressive verb form): e.g. running, jumping, eating, playing
- pronouns later (Later developing pronouns): e.g. he, she, they, her, his, their
- location noun (Location concept expressed with a noun phrase): e.g. in the box, on the table, under the chair
- simple conjunction (Early conjunction use): e.g. and, or
```
**Level age 6 (MLU 8-10)**
```
- plural forms (Regular plural nouns): e.g. dogs, cars, apples, cats, toys
- ing form (Present participle/progressive verb form): e.g. running, jumping, eating, playing
- advanced conjunction (More complex conjunction structures): e.g. while, although, before, after
```
**Level age 8 (MLU 11+)**
```
- irregular past tense (Irregular past tense verbs): e.g. went, ate, saw, gave
- advanced conjunction (More complex conjunction structures): e.g. while, although, before, after
- auxiliary verbs (Helping verbs): e.g. is, are, was, were, has, have
- copula verbs (Linking verbs): e.g. is, am, are, was, were
- verb tense concepts (Past, present, future, and progressive tense usage): e.g. I play, I played, I will play, I am playing
```

## 3. Resolved interest line, per (interest age, gender)

| Interest age | girls | boys |
|---|---|---|
| 2 | *(none specified)* | `dinosaurs (trex, triceratops, stegosaurus); vehicles (car, truck, bus, train, airplane)` |
| 3 | `unicorns; animals (pig, dog, cat, horse, rabbit); pets (dog, cat, hamster, fish); princesses (princess, castle, crown); pretend play (tea party, grocery store, doctor, chef)` | `pets (dog, cat, hamster, fish); pretend play (tea party, grocery store, doctor, chef); vehicles (car, truck, bus, train, airplane)` |
| 5 | `fairies (fairy, magic wand, wings); princesses (princess, castle, crown); pets (dog, cat, hamster, fish); superheroes (superhero, cape, rescue mission)` | `pets (dog, cat, hamster, fish); vehicles (car, truck, bus, train, airplane); dinosaurs (trex, triceratops, stegosaurus); superheroes (superhero, cape, rescue mission)` |
| 6 | `fairies (fairy, magic wand, wings); superheroes (superhero, cape, rescue mission)` | `superheroes (superhero, cape, rescue mission); vehicles (car, truck, bus, train, airplane)` |
| 8 | `fairies (fairy, magic wand, wings)` | `superheroes (superhero, cape, rescue mission); video game characters (game hero, game creature, game vehicle); vehicles (car, truck, bus, train, airplane)` |

---

## 4. Fully-assembled STORY fragment — worked examples (verbatim)

These are exactly what is substituted for `{persona_section}` in the story prompt.

### Example: 3-year-old girl
```
--- LANGUAGE & INTEREST GUIDANCE (knowledge base) ---
Target developmental level: age 3, approx MLU 4 words per utterance. Keep sentences at or near this length.

Weave these language targets naturally into narration and dialogue (model them in context; do not drill or quiz them in the story):
- action verbs higher level (Less frequent and cognitively more complex verbs): e.g. climb, chase, carry, deliver, rescue, build, search, discover
- basic adjectives (Descriptive words for size, color, and shape): e.g. big, small, long, short, red, blue, green, yellow, round, square, triangle-shaped
- pronouns early (Early personal and possessive pronouns): e.g. I, me, my, mine, you
- location prepositions (Spatial relationships): e.g. in, on, under
- size noun (Adjective + noun combinations): e.g. big pig, small monkey, big dinosaur
- person noun (Possessive determiner + noun combinations): e.g. my apple, your car, his dinosaur

Use these interest themes as story hooks, characters, and settings:
- unicorns; animals (pig, dog, cat, horse, rabbit); pets (dog, cat, hamster, fish); princesses (princess, castle, crown); pretend play (tea party, grocery store, doctor, chef)
```

### Example: 5-year-old boy
```
--- LANGUAGE & INTEREST GUIDANCE (knowledge base) ---
Target developmental level: age 5, approx MLU 6-8 words per utterance. Keep sentences at or near this length.

Weave these language targets naturally into narration and dialogue (model them in context; do not drill or quiz them in the story):
- plural forms (Regular plural nouns): e.g. dogs, cars, apples, cats, toys
- ing form (Present participle/progressive verb form): e.g. running, jumping, eating, playing
- pronouns later (Later developing pronouns): e.g. he, she, they, her, his, their
- location noun (Location concept expressed with a noun phrase): e.g. in the box, on the table, under the chair
- simple conjunction (Early conjunction use): e.g. and, or

Use these interest themes as story hooks, characters, and settings:
- pets (dog, cat, hamster, fish); vehicles (car, truck, bus, train, airplane); dinosaurs (trex, triceratops, stegosaurus); superheroes (superhero, cape, rescue mission)
```

### Example: 8-year-old boy
```
--- LANGUAGE & INTEREST GUIDANCE (knowledge base) ---
Target developmental level: age 8, approx MLU 11+ words per utterance. Keep sentences at or near this length.

Weave these language targets naturally into narration and dialogue (model them in context; do not drill or quiz them in the story):
- irregular past tense (Irregular past tense verbs): e.g. went, ate, saw, gave
- advanced conjunction (More complex conjunction structures): e.g. while, although, before, after
- auxiliary verbs (Helping verbs): e.g. is, are, was, were, has, have
- copula verbs (Linking verbs): e.g. is, am, are, was, were
- verb tense concepts (Past, present, future, and progressive tense usage): e.g. I play, I played, I will play, I am playing

Use these interest themes as story hooks, characters, and settings:
- superheroes (superhero, cape, rescue mission); video game characters (game hero, game creature, game vehicle); vehicles (car, truck, bus, train, airplane)
```

---

## 5. Resolved QUESTION fragment (comprehension Qs + scene-game) — format & example

`build_question_prompt_fragment` is the compact variant (same data, different
wrapper). Template:
```
--- LANGUAGE & INTEREST GUIDANCE (knowledge base) ---
Target level: age {level_age}, approx MLU {mlu_range} words. Match question wording to this length.

Embed these language targets in the question wording where natural:
{targets_block}

Draw question content from these interests: {interests_line}
```

### Example: 5-year-old boy (as injected into the scene-game / comprehension prompt)
```
--- LANGUAGE & INTEREST GUIDANCE (knowledge base) ---
Target level: age 5, approx MLU 6-8 words. Match question wording to this length.

Embed these language targets in the question wording where natural:
- plural forms (Regular plural nouns): e.g. dogs, cars, apples, cats, toys
- ing form (Present participle/progressive verb form): e.g. running, jumping, eating, playing
- pronouns later (Later developing pronouns): e.g. he, she, they, her, his, their
- location noun (Location concept expressed with a noun phrase): e.g. in the box, on the table, under the chair
- simple conjunction (Early conjunction use): e.g. and, or

Draw question content from these interests: pets (dog, cat, hamster, fish); vehicles (car, truck, bus, train, airplane); dinosaurs (trex, triceratops, stegosaurus); superheroes (superhero, cape, rescue mission)
```

> In the **scene/object game**, this block is appended after any therapy-goal text
> into `goals_clause`. In **comprehension questions** it is inserted as `persona_block`
> right after the story text. See `activity_02` / `activity_04`.

---

## Legacy note
`src/persona_rag.py` (`PersonaRAG`, data `documents/personas_rag.json`) is the older
persona-matching approach. It is **not** imported by `web_user_server.py` (only
`LanguageInterestKB` is). Its fragment headers are in `helpers_and_shared.md` §H3
for completeness, but it is not on the active prompt path.
