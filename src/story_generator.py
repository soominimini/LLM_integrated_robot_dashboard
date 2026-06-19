#!/usr/bin/env python3.9

# Copyright (c) 2024 LuxAI S.A.
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import os
import json
import random
import subprocess
import tempfile
import requests
from typing import Dict, Any, Optional, List

class StoryGenerator:

    # ─────────────────────────────────────────────
    # AGE-TIER SYSTEM
    # Maps child age to language complexity, sentence
    # structure, and word count constraints.
    #
    # Evidence base for word ranges:
    # - Ages 3–4 (150–250 words): Aligned with early picture
    #   book norms (Goodnight Moon ~130w, Brown Bear ~200w,
    #   Very Hungry Caterpillar ~220w). Attention span research
    #   shows 3-year-olds sustain ~2–3 min focus on a picture
    #   book (Blue Bird Day, 2023), which at read-aloud pace
    #   with interactive pauses yields ~150–300 words of text.
    #   Language milestones: children at this age process 3–5
    #   word sentences (ASHA; Raising Children Network).
    # - Ages 5–6 (250–400 words): Standard young picture book
    #   range (Boyds Mills Press; Wheatmark). Attention spans
    #   of 6–10 min support 5–10 min read-alouds.
    # - Ages 7–8 (400–500 words): Upper picture book / early
    #   reader range (Literary Rambles; Mary Kole Editorial).
    # - Ages 9–12 (450–600 words): Early reader / short chapter
    #   book range. Supports complex narrative arcs.
    # ─────────────────────────────────────────────

    AGE_TIERS = {
        (3, 3): {
            "label": "early_preschool",
            "word_range": (50, 100),
            "guidelines": (
                "Language level: Use 3–5 word sentences. "
                "Repeat key phrases and sentence patterns 2–3 times for reinforcement. "
                "Use only concrete, familiar objects and animals (ball, dog, tree, cup, cat). "
                "Use onomatopoeia freely (splash, whoosh, boom, moo). "
                "Avoid abstract concepts, idioms, or figurative language. "
                "Story structure: Use a simple linear sequence (first, then, finally) with no subplots. "
                "Characters: Maximum 2–3 named characters. "
                "Dialogue: Very short exchanges, 1 sentence per turn."
            ),
        },
        # ─────────────────────────────────────────────
        # WH-QUESTION FORMAT TIER (ages 4–5)
        # Format is modelled on the curated corpus at
        # documents/story for 4 to 7 years old/story_corpus.json
        # ("30 Wh-Question Stories" — @TheSpeechAndSpecialNeedsSpot).
        # Each story is a 3–4 sentence vignette describing a concrete
        # everyday scene, followed by 5–7 WHO/WHAT/WHERE follow-up
        # questions a clinician/robot can ask the child.
        # ─────────────────────────────────────────────
        (4, 5): {
            "label": "wh_question_format",
            "word_range": (40, 70),
            "guidelines": (
                "Language level: Write 3–4 short, concrete sentences in the present tense. "
                "Use simple vocabulary a 4–5 year old already knows (ball, beach, dog, bus, kite, snow, library). "
                "Each sentence states a single observable fact (who is there, where, what is happening, what surprises them). "
                "Avoid abstract concepts, idioms, figurative language, and complex compound sentences. "
                "Story structure: One small everyday scene with a tiny twist or surprise — NOT a three-act journey. "
                "Examples of scenes: building a sandcastle and finding a crab; flying a kite and the wind blowing it onto a dog; "
                "dropping toothpaste on the bathroom floor; spotting an unusual lunch in the cafeteria. "
                "Characters: 1–3 named characters; refer to them by name (not pronouns) so questions are answerable. "
                "Follow-up questions: After the story, generate 5–7 WH-questions (WHO/WHAT/WHERE ONLY — no HOW/WHY at this age) "
                "whose answers are explicitly stated in the story text — one fact per question, in roughly story order."
            ),
        },
        # ─────────────────────────────────────────────
        # EARLY SCHOOL AGE TIER (ages 6–7)
        # Keeps the three-act narrative format AND appends a WH-question
        # comprehension block. At this age the question mix expands to
        # include HOW/WHY (cause, motivation, process). HOW/WHY few-shot
        # examples are loaded from the `fables[*].how_why_questions`
        # field in story_corpus.json.
        # ─────────────────────────────────────────────
        (6, 7): {
            "label": "early_school_age",
            "word_range": (80, 120),
            "requires_takeaways": True,
            "requires_wh_questions": True,
            "guidelines": (
                "Language level: Use varied sentence structures including relative clauses and embedded phrases. "
                "Include emotional vocabulary (frustrated, proud, nervous, relieved, grateful). "
                "Weave in 3–5 target vocabulary words with natural contextual support. "
                "Model question forms and conversational turn-taking in dialogue. "
                "Story structure: Three-act structure with a secondary challenge or emotional subplot. "
                "Characters: Up to 4–5 characters with motivations and feelings. "
                "Dialogue: Natural back-and-forth exchanges of 2–3 sentences, showing perspective-taking. "
                "Comprehension questions: After the story, generate 5–7 WH-questions mixing WHO/WHAT/WHERE (concrete recall, "
                "answers appear verbatim in the story) with 2–3 HOW/WHY questions (cause, motivation, process). "
                "HOW/WHY answers may require short inference, but the inference must be clearly supported by what happens in the story."
            ),
        },
        (8, 10): {
            "label": "school_age",
            "word_range": (130, 200),
            "requires_takeaways": True,
            "guidelines": (
                "Language level: Use complex sentences with subordinate clauses. "
                "Include nuanced emotional and social vocabulary (empathy, compromise, perseverance). "
                "Introduce figurative language gently (similes, simple metaphors) with clear context. "
                "Model inferencing and perspective-taking through character thoughts and dialogue. "
                "Story structure: Three-act structure with internal conflict and character growth. "
                "Characters: Realistic motivations and interpersonal dynamics. "
                "Dialogue: Extended exchanges showing negotiation, repair, and social problem-solving."
            ),
        },
    }

    # ─────────────────────────────────────────────
    # THEME GUIDANCE
    # Provides structural narrative direction per theme
    # (setting, obstacle, resolution, vocabulary focus)
    # rather than appending themes as an afterthought.
    # ─────────────────────────────────────────────

    THEME_GUIDANCE = {
        "season": {
            "setting": "Set the story outdoors with vivid seasonal details (falling leaves, snow, blooming flowers, warm sun).",
            "obstacle": "The obstacle should involve a change in weather or nature that the protagonist must adapt to.",
            "resolution": "The resolution should connect seasonal change to personal growth or emotional understanding.",
            "vocabulary_focus": "Emphasize sensory and nature vocabulary: rustling, shimmering, crisp, gentle, blooming.",
        },
        "school": {
            "setting": "Set the story in a school, classroom, or playground.",
            "obstacle": "The obstacle should involve a social or learning challenge (new routine, group activity, lost item).",
            "resolution": "The resolution should show that asking for help and cooperating leads to success.",
            "vocabulary_focus": "Emphasize school and social vocabulary: share, listen, take turns, try again, curious.",
        },
        "family": {
            "setting": "Center the story around home, family routines, or a family outing.",
            "obstacle": "The obstacle should involve helping a family member or navigating a family situation.",
            "resolution": "The resolution should reinforce love, support, and belonging within the family unit.",
            "vocabulary_focus": "Emphasize relational and home vocabulary: together, helpful, safe, proud, caring.",
        },
        "friends": {
            "setting": "Focus on meeting, playing with, or helping a friend in a familiar environment.",
            "obstacle": "The obstacle should require cooperation, sharing, or empathy between friends.",
            "resolution": "The resolution should show that friendship grows through kindness and working together.",
            "vocabulary_focus": "Emphasize social and emotional vocabulary: kind, brave, worried, excited, grateful.",
        },
        "animals": {
            "setting": "Set the story in a natural habitat, farm, or pet-friendly environment with animal characters.",
            "obstacle": "The obstacle should involve helping an animal or learning from animal behavior.",
            "resolution": "The resolution should connect caring for animals to empathy and responsibility.",
            "vocabulary_focus": "Emphasize animal and descriptive vocabulary: soft, furry, gentle, fast, tiny, enormous.",
        },
        "adventure": {
            "setting": "Set the story in an imaginative but safe environment (enchanted garden, friendly forest, treasure map).",
            "obstacle": "The obstacle should involve solving a puzzle, finding something, or navigating a path.",
            "resolution": "The resolution should reward curiosity, bravery, and persistence.",
            "vocabulary_focus": "Emphasize spatial and action vocabulary: behind, through, under, climb, discover, search.",
        },
    }

    # ─────────────────────────────────────────────
    # OUTPUT FORMAT
    # Word count bounds are injected at prompt-build time.
    # ─────────────────────────────────────────────

    OUTPUT_FORMAT = """
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
Do not add any other sections or extra text."""

    # Injected into OUTPUT_FORMAT when the age tier sets requires_takeaways=True.
    # Sits between ** End ** and ** Explanation of the output **, so existing
    # parsers that look for ** Title ** / ** End ** still work unchanged.
    TAKEAWAYS_OUTPUT_BLOCK = """** Takeaways **
- <takeaway 1: one short, kid-friendly sentence stating a lesson the child should learn>
- <takeaway 2: another lesson the story demonstrates>
- <takeaway 3: another lesson (optional)>

"""

    # Injected into OUTPUT_FORMAT when the age tier sets requires_wh_questions=True.
    # Sits between ** End ** / ** Takeaways ** and ** Explanation of the output **.
    # The question mix at ages 6-7 includes WHO/WHAT/WHERE (concrete recall) AND
    # HOW/WHY (inference about cause, motivation, or process).
    WH_QUESTIONS_OUTPUT_BLOCK = """** Questions **
1. <WH-question — WHO/WHAT/WHERE/HOW/WHY>
2. <WH-question>
3. <WH-question>
4. <WH-question>
5. <WH-question>
(6. and 7. optional)

"""

    # Injected into MASTER_TEMPLATE when the tier requires WH-questions.
    # Few-shot HOW/WHY examples (from fables) are placed where {fable_examples}
    # is substituted at build time.
    WH_QUESTIONS_PROMPT_BLOCK = """
--- COMPREHENSION QUESTIONS (WHO/WHAT/WHERE + HOW/WHY) ---
After the story, generate 5–7 WH-questions a clinician or robot can use to check the child's comprehension:
- Include 3–4 WHO/WHAT/WHERE questions whose answers appear verbatim in the story (concrete recall).
- Include 2–3 HOW/WHY questions about cause, motivation, or process. Short inference is allowed, but the inference MUST be clearly supported by what happens in the story.
- Phrase each question so a 6–7 year old can answer it in 1 short sentence. Avoid yes/no questions.

HOW/WHY example questions (from the curated fable corpus — same style for {child_name}'s story):
{fable_examples}
"""

    # Injected into MASTER_TEMPLATE when the tier requires takeaways.
    TAKEAWAYS_PROMPT_BLOCK = """
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
"""

    # WH-format output adds a Questions section between End and Explanation.
    # Title/End delimiters are preserved so existing parsers keep working.
    WH_OUTPUT_FORMAT = """
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
- Do NOT add sections beyond Title, story text, End, Questions, and Explanation."""

    # ─────────────────────────────────────────────
    # MASTER PROMPT TEMPLATE
    # Replaces all individual per-topic templates with a
    # single composable template. Theme-specific narrative
    # structure is injected via THEME_GUIDANCE fields.
    # ─────────────────────────────────────────────

    MASTER_TEMPLATE = """Write a short therapeutic story for a {age}-year-old {gender} named {child_name}, who has speech delay. The story should be developmentally appropriate, engaging, and supportive of early language development.

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

{output_format}"""

    # ─────────────────────────────────────────────
    # WH-FORMAT MASTER TEMPLATE (ages 4–5)
    # Mirrors the corpus style: short concrete scene + WH-questions.
    # Examples from story_corpus.json are injected into {wh_examples}.
    # ─────────────────────────────────────────────

    WH_MASTER_TEMPLATE = """Write a short illustrated-style story for a {age}-year-old {gender} named {child_name}, who has speech delay. The story will be used by a robot to practise WH-question comprehension (WHO / WHAT / WHERE) with the child.

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

Available gestures: [gesture:NAME] where NAME ∈ {{hi, bye, nodding-yes, clapping, hoora, happy, calm, shy, embrace, patience, slight_no, think, sneezing, yawn, breathing_exercise, kiss, stretching}}
Available emotions: [emotion:NAME] where NAME ∈ {{QT/happy, QT/sad, QT/surprised, QT/afraid, QT/angry, QT/calm, QT/shy}}

Place a tag IMMEDIATELY before the sentence that depicts the emotion or gesture. Do NOT include tags inside the WH-questions.

{output_format}"""

    # ─────────────────────────────────────────────
    # CORPUS REFERENCE (for ages 4–7 WH-question stories)
    # ─────────────────────────────────────────────

    CORPUS_PATH = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "documents", "story for 4 to 7 years old", "story_corpus.json",
    )

    # ─────────────────────────────────────────────
    # GOALS FORMATTING
    # ─────────────────────────────────────────────

    GOALS_SECTION_TEMPLATE = """Therapy goals to integrate naturally into the story (do NOT list them explicitly — weave them into narrative, dialogue, and action):

{formatted_goals}"""

    DEFAULT_GOALS = (
        "- Learning descriptive words (adjectives, spatial terms)\n"
        "- Collaboration and turn-taking\n"
        "- Importance of friendships and relationships\n"
        "- Overcoming challenges with support"
    )

    def __init__(self, llm_model: str = "claude-sonnet-4-6", disable_rag: bool = True):
        """
        Initialize the story generator.

        Routes to a subprocess worker based on the model name:
          - "claude-*" → scripts/claude_story.py (Anthropic Claude API)
          - "gemini-*" → scripts/gemini_story.py (Google Gemini API)
          - anything else → local Ollama at http://localhost:11434

        Args:
            llm_model: The model identifier (e.g. "claude-sonnet-4-6", "gemini-2.5-flash", "gemma4:e4b")
            disable_rag: Kept for API compatibility (unused with subprocess workers)
        """
        self.llm_model = llm_model
        self.disable_rag = disable_rag
        scripts_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'scripts',
        )
        self._gemini_script_path = os.path.join(scripts_dir, 'gemini_story.py')
        self._claude_script_path = os.path.join(scripts_dir, 'claude_story.py')
        self._worker_python = os.getenv(
            "IMAGE_WORKER_PYTHON",
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         ".venv39", "bin", "python"),
        )

    # ─────────────────────────────────────────────
    # INTERNAL HELPERS
    # ─────────────────────────────────────────────

    def _get_age_tier(self, age: int) -> Dict[str, Any]:
        """Look up the age tier. Falls back to nearest tier if outside defined ranges."""
        for (lo, hi), tier in self.AGE_TIERS.items():
            if lo <= age <= hi:
                return tier
        all_ranges = sorted(self.AGE_TIERS.keys())
        if age < all_ranges[0][0]:
            return self.AGE_TIERS[all_ranges[0]]
        return self.AGE_TIERS[all_ranges[-1]]

    def _get_theme_guidance(self, topics: Optional[List[str]] = None) -> Dict[str, str]:
        """
        Get structural theme guidance for the selected topics.
        If multiple topics match, merge their guidance.
        Falls back to a generic default if no topics match.
        """
        default_theme = {
            "setting": "Set the story in a familiar, child-friendly environment.",
            "obstacle": "The obstacle should involve a manageable challenge that requires help from others.",
            "resolution": "The resolution should leave the protagonist feeling proud, grateful, and connected.",
            "vocabulary_focus": "Use rich descriptive vocabulary appropriate to the setting and characters.",
        }
        if not topics:
            return default_theme

        merged = {"setting": [], "obstacle": [], "resolution": [], "vocabulary_focus": []}
        matched = False
        for topic in topics:
            key = str(topic).strip().lower()
            if key in self.THEME_GUIDANCE:
                matched = True
                for field in merged:
                    merged[field].append(self.THEME_GUIDANCE[key][field])
        if not matched:
            return default_theme
        return {k: " ".join(v) for k, v in merged.items()}

    def _load_wh_examples(self, n: int = 2, topics: Optional[List[str]] = None) -> str:
        """
        Load N example stories from the WH-question corpus and format them
        as few-shot examples for the prompt.

        If `topics` is provided, prefers examples whose setting field contains
        one of the topic keywords; otherwise samples randomly. Falls back
        silently to an empty string if the corpus cannot be read.
        """
        try:
            with open(self.CORPUS_PATH, "r", encoding="utf-8") as f:
                corpus = json.load(f)
            stories = corpus.get("wh_question_stories", [])
            if not stories:
                return ""

            pool = stories
            if topics:
                topic_keys = [str(t).strip().lower() for t in topics if str(t).strip()]
                matched = [s for s in stories
                           if any(k in str(s.get("setting", "")).lower() for k in topic_keys)]
                if matched:
                    pool = matched

            sample = random.sample(pool, min(n, len(pool)))

            blocks = []
            for i, s in enumerate(sample, 1):
                q_lines = "\n".join(f"  - {q['q']}" for q in s.get("questions", []))
                blocks.append(
                    f"Example {i} (setting: {s.get('setting', 'unknown')}):\n"
                    f"  Story: {s.get('text', '').strip()}\n"
                    f"  Questions:\n{q_lines}"
                )
            return "\n\n".join(blocks) + "\n"
        except (OSError, json.JSONDecodeError):
            return ""

    def _load_fable_inferential_examples(self, n: int = 2) -> str:
        """
        Load N fables that carry HOW/WHY question examples (set on the
        `how_why_questions` field) and format them as few-shot examples
        for the ages 6-7 comprehension-question block. Falls back to an
        empty string if the corpus cannot be read.
        """
        try:
            with open(self.CORPUS_PATH, "r", encoding="utf-8") as f:
                corpus = json.load(f)
            fables = [f for f in corpus.get("fables", []) if f.get("how_why_questions")]
            if not fables:
                return ""
            sample = random.sample(fables, min(n, len(fables)))
            blocks = []
            for i, f in enumerate(sample, 1):
                q_lines = "\n".join(
                    f"    - {q['q']}  →  {q['a']}" for q in f.get("how_why_questions", [])
                )
                blocks.append(
                    f"  Example {i} (\"{f.get('title', 'untitled')}\"):\n"
                    f"    Story summary: {f.get('text', '').strip()}\n"
                    f"    HOW/WHY questions:\n{q_lines}"
                )
            return "\n\n".join(blocks)
        except (OSError, json.JSONDecodeError):
            return ""

    def _format_goals_section(self, goals: Optional[str] = None) -> str:
        """Format therapy goals into the prompt section."""
        if goals:
            formatted = (
                f"- Clinician-specified goals: {goals}\n"
                "  Integrate these naturally through story events, character dialogue, and descriptive language.\n"
                "  Also include the following foundational goals:\n"
                "  1) learning descriptive words, 2) collaboration, "
                "3) importance of friendships and relationships, 4) overcoming challenges."
            )
        else:
            formatted = self.DEFAULT_GOALS
        return self.GOALS_SECTION_TEMPLATE.format(formatted_goals=formatted)

    def _build_prompt(
        self,
        child_name: str,
        age: int,
        gender: str,
        topics: Optional[List[str]] = None,
        goals: Optional[str] = None,
        persona_context: Optional[str] = None,
        language_age: Optional[int] = None,
    ) -> str:
        """
        Build the full story generation prompt from composable components:
        age tier -> theme guidance -> goals -> persona -> output format.

        Replaces the old _select_template() approach with a single
        master template that gets different content injected based
        on the child's age and selected topics.

        ``age`` is the chronological age used for the child's stated identity in
        the prose. ``language_age`` (when given) is the developmental/language
        age that selects the complexity tier — sentence length, structure, and
        word range. This decouples a child's identity from their language level
        (e.g. a 9-year-old targeted at an MLU-6-8 / language-age-5 tier).
        """
        tier_age = language_age if language_age is not None else age
        age_tier = self._get_age_tier(tier_age)
        min_words, max_words = age_tier["word_range"]
        theme = self._get_theme_guidance(topics)
        goals_section = self._format_goals_section(goals)

        persona_section = ''
        if persona_context and persona_context.strip():
            persona_section = "\n" + persona_context.strip() + "\n"

        # Ages 4–6: route to the WH-question short-story format,
        # using examples retrieved from the curated corpus.
        if age_tier.get("label") == "wh_question_format":
            wh_examples = self._load_wh_examples(n=2, topics=topics)
            output_format = self.WH_OUTPUT_FORMAT.format(
                goals=goals or "general speech-language therapy goals",
                min_words=min_words,
                max_words=max_words,
                min_questions=5,
                max_questions=7,
            )
            prompt = self.WH_MASTER_TEMPLATE.format(
                child_name=child_name,
                age=age,
                gender=gender,
                age_guidelines=age_tier["guidelines"],
                theme_setting=theme["setting"],
                theme_obstacle=theme["obstacle"],
                theme_resolution=theme["resolution"],
                theme_vocabulary=theme["vocabulary_focus"],
                goals_section=goals_section,
                persona_section=persona_section,
                wh_examples=wh_examples,
                output_format=output_format,
            )
            if topics:
                safe_topics = [str(t).strip() for t in topics if str(t).strip()]
                if safe_topics:
                    prompt += "\n\nIncorporate the following theme(s) into the scene: " + ", ".join(safe_topics) + "."
            return prompt

        # Tier-gated takeaways: ages 7+ get an explicit ** Takeaways ** section
        # and a prompt block telling the model to teach 2–3 lessons.
        if age_tier.get("requires_takeaways"):
            takeaways_section = self.TAKEAWAYS_OUTPUT_BLOCK
            takeaways_block = self.TAKEAWAYS_PROMPT_BLOCK
            takeaways_in_rule = " Takeaways,"
        else:
            takeaways_section = ""
            takeaways_block = ""
            takeaways_in_rule = ""

        # Tier-gated WH-questions: ages 6-7 append a ** Questions ** block
        # with mixed WHO/WHAT/WHERE + HOW/WHY questions. Few-shot HOW/WHY
        # examples are mined from the fables in story_corpus.json.
        if age_tier.get("requires_wh_questions"):
            wh_questions_section = self.WH_QUESTIONS_OUTPUT_BLOCK
            fable_examples = self._load_fable_inferential_examples(n=2)
            wh_questions_block = self.WH_QUESTIONS_PROMPT_BLOCK.format(
                child_name=child_name,
                fable_examples=fable_examples,
            )
            wh_questions_in_rule = " Questions,"
        else:
            wh_questions_section = ""
            wh_questions_block = ""
            wh_questions_in_rule = ""

        output_format = self.OUTPUT_FORMAT.format(
            goals=goals or "general speech-language therapy goals",
            min_words=min_words,
            max_words=max_words,
            mid_words=(min_words + max_words) // 2,
            takeaways_section=takeaways_section,
            takeaways_in_rule=takeaways_in_rule,
            wh_questions_section=wh_questions_section,
            wh_questions_in_rule=wh_questions_in_rule,
        )

        prompt = self.MASTER_TEMPLATE.format(
            child_name=child_name,
            age=age,
            gender=gender,
            age_guidelines=age_tier["guidelines"],
            theme_setting=theme["setting"],
            theme_obstacle=theme["obstacle"],
            theme_resolution=theme["resolution"],
            theme_vocabulary=theme["vocabulary_focus"],
            goals_section=goals_section,
            persona_section=persona_section,
            takeaways_block=takeaways_block,
            wh_questions_block=wh_questions_block,
            output_format=output_format,
        )

        # Append topic names as reinforcement (not primary integration)
        if topics:
            safe_topics = [str(t).strip() for t in topics if str(t).strip()]
            if safe_topics:
                prompt += "\n\nIncorporate the following theme(s) prominently and naturally: " + ", ".join(safe_topics) + "."

        return prompt

    # ─────────────────────────────────────────────
    # POST-GENERATION SHORTENING
    # Safety net for when the model overshoots its word cap despite the
    # prompt rule. Called by api_save_story when the saved body exceeds
    # max_words by a meaningful margin.
    # ─────────────────────────────────────────────

    def get_word_range_for_age(self, age: int):
        """Return (min_words, max_words) for the given age, used by callers
        that want to validate body length without rebuilding the prompt."""
        tier = self._get_age_tier(age)
        return tier["word_range"]

    SHORTEN_TEMPLATE = """You are shortening a children's story so it fits a strict word cap.

ORIGINAL STORY (with inline [gesture:...] and [emotion:...] tags):
\"\"\"
{body}
\"\"\"

REWRITE RULES:
- Target length: {min_words}–{max_words} words. The {max_words} ceiling is HARD; aim for around {mid_words} words.
- Preserve the plot, character names ({names_hint}), beginning, and ending.
- Preserve the existing inline [gesture:...] and [emotion:...] tags — keep them attached to the same emotional/action beats.
- Cut adjectives, side details, redundant dialogue lines, and any subplot first.
- Do NOT add new characters, events, or sections.
- Do NOT add commentary, preamble, or markdown headings — just return the rewritten story body.

Return ONLY the rewritten story body. No "Here is" preamble. No ** Title **, ** End **, takeaways, or explanation — just the story text."""

    def shorten_story(self, body: str, age: int, child_name: str = "") -> str:
        """Shorten an already-generated story body to fit the age tier's word range.

        Returns the shortened body on success, or the original body on any failure.
        """
        if not body:
            return body
        tier = self._get_age_tier(age)
        min_words, max_words = tier["word_range"]
        mid_words = (min_words + max_words) // 2

        names_hint = child_name or "the protagonist"

        prompt = self.SHORTEN_TEMPLATE.format(
            body=body,
            min_words=min_words,
            max_words=max_words,
            mid_words=mid_words,
            names_hint=names_hint,
        )
        try:
            shortened = self._run_llm(prompt, stream=False) or ""
            shortened = shortened.strip()
            # Defensive cleanup: drop any accidental markers the model added.
            for marker in ("** Title **", "** End **", "** Takeaways **",
                           "** Explanation of the output **"):
                idx = shortened.find(marker)
                if idx >= 0:
                    shortened = shortened[:idx].strip() if marker != "** Title **" else shortened[idx + len(marker):].strip()
            if not shortened:
                return body
            return shortened
        except Exception as e:
            print(f"[StoryGenerator] shorten_story failed: {e}")
            return body

    # ─────────────────────────────────────────────
    # PUBLIC API (unchanged signatures)
    # ─────────────────────────────────────────────

    def _run_subprocess_worker(self, script_path: str, prompt: str, stream: bool = False):
        """Run a subprocess story worker (Gemini or Claude).

        Both workers share the same CLI shape: --model, --prompt-file, --stream.
        Returns story text (non-stream) or a Popen object (stream).
        """
        # Write prompt to a temp file to avoid shell escaping issues
        tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
        try:
            tmp.write(prompt)
            tmp.close()

            cmd = [self._worker_python, script_path,
                   '--model', self.llm_model,
                   '--prompt-file', tmp.name]
            if stream:
                cmd.append('--stream')
                return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                        text=True, bufsize=1)

            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if proc.returncode != 0:
                raise RuntimeError(f"{os.path.basename(script_path)} failed: {proc.stderr}")
            return proc.stdout or ''
        finally:
            if not stream:
                try:
                    os.unlink(tmp.name)
                except OSError:
                    pass

    def _run_gemini(self, prompt: str, stream: bool = False):
        return self._run_subprocess_worker(self._gemini_script_path, prompt, stream=stream)

    def _run_claude(self, prompt: str, stream: bool = False):
        return self._run_subprocess_worker(self._claude_script_path, prompt, stream=stream)

    def _is_claude_model(self):
        """Check if the configured model is an Anthropic Claude API model."""
        return self.llm_model.startswith("claude")

    def _is_ollama_model(self):
        """Check if the configured model is a local Ollama model (not a cloud API model)."""
        return not (self.llm_model.startswith("gemini") or self.llm_model.startswith("claude"))

    def _run_ollama(self, prompt: str, stream: bool = False):
        """Run text generation via local Ollama API.

        Returns story text (non-stream) or a generator of chunks (stream).
        """
        url = "http://localhost:11434/api/generate"
        payload = {
            "model": self.llm_model,
            "prompt": prompt,
            "stream": stream,
            "options": {"num_predict": 2048, "temperature": 0.7},
        }
        if not stream:
            resp = requests.post(url, json=payload, timeout=180)
            resp.raise_for_status()
            return resp.json().get("response", "")
        else:
            # Return a generator for streaming
            resp = requests.post(url, json=payload, timeout=180, stream=True)
            resp.raise_for_status()
            return resp

    def _run_llm(self, prompt: str, stream: bool = False):
        """Route to Claude, Gemini, or local Ollama based on model name."""
        if self._is_claude_model():
            return self._run_claude(prompt, stream=stream)
        if self._is_ollama_model():
            return self._run_ollama(prompt, stream=stream)
        return self._run_gemini(prompt, stream=stream)

    def generate_story(
        self,
        child_name: str,
        age: int,
        gender: str,
        custom_prompt: Optional[str] = None,
        topics: Optional[List[str]] = None,
        goals: Optional[str] = None,
        persona_context: Optional[str] = None,
        language_age: Optional[int] = None,
    ) -> Dict[str, Any]:

        try:
            if custom_prompt:
                prompt = custom_prompt
            else:
                prompt = self._build_prompt(
                    child_name=child_name,
                    age=age,
                    gender=gender,
                    topics=topics,
                    goals=goals,
                    persona_context=persona_context,
                    language_age=language_age,
                )

            print("[StoryGenerator] prompt: ", prompt)

            story_text = self._run_llm(prompt, stream=False)

            tier_age = language_age if language_age is not None else age
            age_tier = self._get_age_tier(tier_age)
            story_metadata = {
                "child_name": child_name,
                "age": age,
                "language_age": language_age,
                "age_tier": age_tier["label"],
                "target_word_range": list(age_tier["word_range"]),
                "word_count": len(story_text.split()),
                "generated_at": None,
                "model": self.llm_model,
                "topics": topics or []
            }

            return {
                "success": True,
                "story": story_text,
                "metadata": story_metadata
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "story": None,
                "metadata": None
            }


    def generate_story_stream(
        self,
        child_name: str,
        age: int,
        gender: str,
        custom_prompt: Optional[str] = None,
        topics: Optional[List[str]] = None,
        goals: Optional[str] = None,
        persona_context: Optional[str] = None,
        language_age: Optional[int] = None,
    ):
        """
        Generate a therapeutic story with streaming response via Gemini API.

        Yields:
            Story text chunks as they are generated
        """
        tmp_path = None
        try:
            if custom_prompt:
                prompt = custom_prompt
            else:
                prompt = self._build_prompt(
                    child_name=child_name,
                    age=age,
                    gender=gender,
                    topics=topics,
                    goals=goals,
                    persona_context=persona_context,
                    language_age=language_age,
                )

            if self._is_ollama_model():
                resp = self._run_ollama(prompt, stream=True)
                for line in resp.iter_lines():
                    if line:
                        try:
                            chunk_data = json.loads(line)
                            text = chunk_data.get("response", "")
                            if text:
                                yield text
                        except json.JSONDecodeError:
                            pass
            else:
                # Claude and Gemini share the same CHUNK: line subprocess protocol
                proc = (self._run_claude(prompt, stream=True)
                        if self._is_claude_model()
                        else self._run_gemini(prompt, stream=True))
                # Save tmp path for cleanup (stored in proc.args)
                tmp_path = proc.args[proc.args.index('--prompt-file') + 1]

                for line in proc.stdout:
                    line = line.rstrip('\n')
                    if line.startswith('CHUNK:'):
                        yield line[6:] + '\n'

                proc.wait()
                if proc.returncode != 0:
                    err = proc.stderr.read() if proc.stderr else ''
                    yield f"Error generating story: {err}"

        except Exception as e:
            yield f"Error generating story: {str(e)}"
        finally:
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    def close(self):
        """Clean up resources"""
        pass
