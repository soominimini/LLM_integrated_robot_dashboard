# Structure of `restructured_knowledge_base_v2.json`

**Name:** Unified SLP Co-design Knowledge Base · **Version:** 2.2 · Documented **2026-07-04**.
Loaded by `src/knowledge_base.py` (`LanguageInterestKB`); consumed by story generation,
educational quiz, and the WH picture-scene activity (see "Who reads what" at the bottom).

## What's in this document

- **Top-level tree** — the 7 top-level sections at a glance
- **Per-framework detail** — all 5 frameworks with exact counts and field shapes:
  - `language`: 25 targets (21 grammar + 4 WH-question targets), 6 MLU levels with a table showing which levels carry `wh_question_guidance`, plus the `wh_question_hierarchy`
  - `speech_sound`: the 4 phoneme groups with their actual phonemes
  - `concept`: 21 targets, 6 levels (ages 2-3 → 9+), topic_mapping, the 4 question-template families
  - `social_communication`: 10 targets, the age-8 ASD level and its activity types
  - `interests`: 10 themes, noting the branded `{name, type, usable_themes}` entries and the three age-preference lists
- **`image_card_wh_question_generation_guidance`** — both age blocks including `default_for_age_5_plus` (exactly-one-why/how rule, added 2026-07-04) and how the code age-gates them (≤4 vs ≥5)
- **The 11 global rules** compressed to one paragraph
- **"Who reads what"** — a mapping from each KB section to the loader attribute and the activities that consume it, explicitly flagging that `concept` and `social_communication` are not consumed by any code yet (the two biggest untapped sections — natural candidates for the educational quiz and a future social-skills activity)
- **Editing tips** — the practical gotchas (target ids must appear in a level's list to take effect; only plain-string theme examples reach prompts; the WH precedence order: level guidance → hierarchy age buckets → image-card blocks)

## Top-level map

```
restructured_knowledge_base_v2.json
├── knowledge_base_metadata        name, version, description, source_note,
│                                  frameworks list, design_principles[6]
├── frameworks
│   ├── language                   wording complexity: grammar targets, MLU levels,
│   │                              WH-question hierarchy
│   ├── speech_sound               articulation: phoneme groups by developmental age
│   ├── concept                    conceptual-knowledge targets for quizzes
│   ├── social_communication       ASD peer-interaction / social-pragmatic targets
│   └── interests                  theme library for personalization
├── global_generation_rules        rule_1 … rule_11 (cross-activity guardrails)
├── example_activity_selection_workflow   7-step recipe: profile → domain → level →
│                                          theme → generate → check → WH hierarchy
├── example_child_profile_schema   field-by-field schema incl. wh_question_level,
│                                  allowed_wh_types
├── example_child_profile          worked age-5 example
└── image_card_wh_question_generation_guidance
    ├── default_for_age_4_5        gentle what/who(/where) selection steps + fallback
    ├── default_for_age_5_plus     exactly ONE evidence-supported why/how per set
    └── evidence_rules[4]          visibility requirements per WH type
```

## frameworks.language (v1.1)

| Key | Shape | Content |
|---|---|---|
| `targets` | dict, **25 targets** | 21 grammar/vocab targets (`{description, examples}`) + **4 WH-question targets** (`wh_questions_early/location/time/reasoning`, extra fields `{wh_types, difficulty_level, recommended_use}`) |
| `developmental_levels` | list[6] | one per language age — see below |
| `wh_question_hierarchy` | dict | `developmental_order` (what→who→where→when→why→how), `difficulty_levels[4]`, `age_guidance{age_2_3, age_4, age_5, age_6_8}`, `image_card_guidance` |

**Levels** (`level_id`, `age`, `mlu_range`, `targets[]`, + optional `optional_targets`, `wh_question_guidance`, `discourse_guidance`):

| age | MLU | targets | wh_guidance | notes |
|---|---|---|---|---|
| 2 | 1–3 | 3 | — | |
| 3 | 4 | 6 | — | |
| 4 | 5–6 | 9 | ✔ | primary what/who, secondary where |
| 5 | 6–8 | 9 (+1 opt) | ✔ | primary what/who/where; when w/ support; why/how emerging |
| 6 | 8–10 | 7 (+1 opt) | ✔ | + when recommended; why/how w/ support |
| 8 | 11+ | 12 | ✔ | all six WH types; + `discourse_guidance` |

## frameworks.speech_sound (v1.0)

- `targets` — 4 phoneme groups: `early_sounds` [p b m n d], `middle_sounds` [k g f],
  `later_sounds` [sh s], `advanced_sounds` [th r ch]; each `{difficulty_level, age_range,
  description, phonemes, example_words, example_activity_phrases}`
- `developmental_levels` — list[4]: age ranges 2-3 / 3-4 / 4-5 / 5+ → one target group each

## frameworks.concept (v1.1)

| Key | Shape | Content |
|---|---|---|
| `targets` | dict, **21 targets** | object_identity … prediction_inference; each `{difficulty_level, developmental_position, description, question_goal, conceptual_relation, recommended_age_range, examples, example_questions, generation_constraints}` |
| `developmental_levels` | list[6] | ages 2-3, 4, 5, 6-7, 8, 9+; each with `primary_targets`, `optional_targets`, `question_guidance` (incl. per-level `wh_question_guidance`), `example_questions` |
| `topic_mapping` | dict[5] | Fruit / School / Home / Food / Nature → `recommended_targets_age_4_5`, `recommended_targets_age_8`, `example_items`, `avoid_for_younger_children` |
| `question_templates` | dict[4] | `yes_no`, `wh`, `why_how_compare`, `wh_developmental_hierarchy` (level_1_what_who → level_4_why_how with templates + constraints) |
| `generation_rules` | rule_1…rule_11 | quiz-generation guardrails |

## frameworks.social_communication (v1.0)

- `targets` — 10 targets (reciprocal_conversation, emotion_understanding, conflict_resolution, …);
  each `{description, skills, example_activity}`
- `developmental_levels` — list[1]: `social_age_8_asd_peer_interaction` with
  `recommended_activity_types` (role_play, social_story, comic_strip_conversation, …)
- `generation_rules` — rule_1…rule_4 (no forced eye contact / masking; concrete scenarios)

## frameworks.interests (v1.1)

- `themes` — 10 themes; each `{description, generic_examples, specific_examples,
  generation_constraints}`. `superheroes` / `video_game_characters` carry **structured branded
  entries** (`{name, type, usable_themes}` — Superman, Roblox, Minecraft, Pokemon …) in
  `specific_examples`; branded content requires an explicit child-profile interest.
- `neutral_age_recommendations` — list[5] (ages 2,3,5,6,8): gender-neutral theme lists
- `co_design_observed_age_preferences` — list[5]: original girls/boys observations (ages 2–8)
- `selection_rules` — rule_1…rule_4 (clinical target before theme; gender-neutral application)
- `source_age_preferences_from_previous_kb` — list[5]: provenance copy of the v1.x lists

## image_card_wh_question_generation_guidance (v1.0)

Operational rules for the WH picture-scene activity:

- **`default_for_age_4_5`** (applies to language age **≤ 4** in code): start what/who; add
  where if clear; when only with visible cues; why/how only if simply supported; fallback =
  replace unsupported types with more what/who/where.
- **`default_for_age_5_plus`** (language age **≥ 5**, added 2026-07-04): include **exactly ONE**
  simple why/how per question set, chosen by visible evidence; what/who/where for the rest;
  fallback prefers a visually grounded why over dropping it.
- **`evidence_rules`** — who/what/where directly visible; when needs cues; why/how needs
  evidence; never assume hidden intentions.

## global_generation_rules (11)

Domain-first selection (1, 8) · wording ≠ concept level (2) · combine targets only when
clinically apt (3) · no diagnostic assumptions (4) · expected-answer metadata for yes/no (5) ·
safe theme use (6, 10) · speech-sound domain preserved (7) · developmental not chronological
age (9) · **rule_11: follow the WH-question hierarchy for image-card/WH generation**.

## Who reads what (`src/knowledge_base.py` → activities)

| KB section | Loader attribute | Used by |
|---|---|---|
| `frameworks.language.targets` | `_language_targets` | story fragment (WH targets excluded from narration), question fragments |
| `frameworks.language.developmental_levels` | `_levels` | MLU calibration everywhere; per-level `wh_question_guidance` via `resolve_wh_guidance()` |
| `frameworks.language.wh_question_hierarchy` | `_wh_hierarchy` | fallback age buckets for WH guidance |
| `frameworks.speech_sound.*` | `_articulation_targets`, `_speech_levels` | story + question speech-sound blocks |
| `frameworks.interests.themes` | `_themes` (normalized to string lists; branded dict entries skipped) | story hooks, question content |
| `frameworks.interests.co_design_observed_age_preferences` | `_age_prefs` | interest selection by age(+gender; unknown gender merges) |
| `image_card_wh_question_generation_guidance` | `_image_card_wh` | WH picture-scene receptive prompts (`build_wh_question_guidance_fragment(image_card=True)`) |
| `frameworks.concept.*` | — **not consumed by code yet** | (candidate: educational quiz topics/templates) |
| `frameworks.social_communication.*` | — **not consumed by code yet** | (candidate: social-skills activities) |
| `global_generation_rules`, examples/schema | — reference for authors/LLM designers, not parsed | |

**Editing tips:** new language/concept/speech target ids must be listed in a level's
`targets`/`primary_targets` to take effect; theme items live in `generic_examples` (plain
strings only reach prompts); WH behavior is governed by per-level `wh_question_guidance`
first, hierarchy `age_guidance` second, and — for picture cards — the
`image_card_wh_question_generation_guidance` blocks on top.
