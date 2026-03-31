#!/usr/bin/env python3.9

# Copyright (c) 2024 LuxAI S.A.
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import os
import json
import subprocess
import tempfile
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
        (3, 4): {
            "label": "early_preschool",
            "word_range": (150, 250),
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
        (5, 6): {
            "label": "late_preschool",
            "word_range": (250, 400),
            "guidelines": (
                "Language level: Use simple compound sentences with connectors (and, but, so, because). "
                "Include basic cause-and-effect relationships. "
                "Introduce 2–3 new descriptive words with context clues so meaning is inferrable. "
                "Begin using basic spatial and temporal language (behind, next to, before, after). "
                "Story structure: Three-act structure with a clear problem and resolution. "
                "Characters: Up to 3–4 named characters with simple traits. "
                "Dialogue: Short exchanges of 1–2 sentences per turn, modelling target language structures."
            ),
        },
        (7, 8): {
            "label": "early_school_age",
            "word_range": (400, 500),
            "guidelines": (
                "Language level: Use varied sentence structures including relative clauses and embedded phrases. "
                "Include emotional vocabulary (frustrated, proud, nervous, relieved, grateful). "
                "Weave in 3–5 target vocabulary words with natural contextual support. "
                "Model question forms and conversational turn-taking in dialogue. "
                "Story structure: Three-act structure with a secondary challenge or emotional subplot. "
                "Characters: Up to 4–5 characters with motivations and feelings. "
                "Dialogue: Natural back-and-forth exchanges of 2–3 sentences, showing perspective-taking."
            ),
        },
        (9, 12): {
            "label": "school_age",
            "word_range": (450, 600),
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
** Explanation of the output **
<1–3 short sentences explaining how the story matches the selected topic(s) and the learning goals: {goals}>

STRICT RULES:
- Do NOT include any preamble, commentary, or meta-text (no "Here is your story", "Sure!", etc.).
- Do NOT add sections beyond Title, story text, End, and Explanation.
- Count your words before finishing. The story text (between Title and End) MUST be between {min_words} and {max_words} words. If you are over {max_words}, shorten it. If under {min_words}, expand it.
Do not add any other sections or extra text."""

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

--- TONE AND STYLE ---
- Warm, encouraging, and gently paced.
- Show, don't tell: use actions and dialogue to convey emotions rather than stating them.
- Include at least one moment of humor, wonder, or sensory delight.
- Use character names consistently (avoid pronoun ambiguity for young readers).

{output_format}"""

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

    def __init__(self, llm_model: str = "gemini-2.5-flash", disable_rag: bool = True):
        """
        Initialize the story generator with Gemini API (via subprocess).

        Args:
            llm_model: The Gemini model to use (e.g. "gemini-2.5-flash")
            disable_rag: Kept for API compatibility (unused with Gemini)
        """
        self.llm_model = llm_model
        self.disable_rag = disable_rag
        self._script_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'scripts', 'gemini_story.py'
        )
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
    ) -> str:
        """
        Build the full story generation prompt from composable components:
        age tier -> theme guidance -> goals -> output format.

        Replaces the old _select_template() approach with a single
        master template that gets different content injected based
        on the child's age and selected topics.
        """
        age_tier = self._get_age_tier(age)
        min_words, max_words = age_tier["word_range"]
        theme = self._get_theme_guidance(topics)
        goals_section = self._format_goals_section(goals)

        output_format = self.OUTPUT_FORMAT.format(
            goals=goals or "general speech-language therapy goals",
            min_words=min_words,
            max_words=max_words,
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
            output_format=output_format,
        )

        # Append topic names as reinforcement (not primary integration)
        if topics:
            safe_topics = [str(t).strip() for t in topics if str(t).strip()]
            if safe_topics:
                prompt += "\n\nIncorporate the following theme(s) prominently and naturally: " + ", ".join(safe_topics) + "."

        return prompt

    # ─────────────────────────────────────────────
    # PUBLIC API (unchanged signatures)
    # ─────────────────────────────────────────────

    def _run_gemini(self, prompt: str, stream: bool = False):
        """Run the gemini_story.py script via subprocess.

        Returns story text (non-stream) or a Popen object (stream).
        """
        # Write prompt to a temp file to avoid shell escaping issues
        tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
        try:
            tmp.write(prompt)
            tmp.close()

            cmd = [self._worker_python, self._script_path,
                   '--model', self.llm_model,
                   '--prompt-file', tmp.name]
            if stream:
                cmd.append('--stream')
                return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                        text=True, bufsize=1)

            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if proc.returncode != 0:
                raise RuntimeError(f"gemini_story.py failed: {proc.stderr}")
            return proc.stdout or ''
        finally:
            if not stream:
                try:
                    os.unlink(tmp.name)
                except OSError:
                    pass

    def generate_story(
        self,
        child_name: str,
        age: int,
        gender: str,
        custom_prompt: Optional[str] = None,
        topics: Optional[List[str]] = None,
        goals: Optional[str] = None,
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
                )

            print("[StoryGenerator] prompt: ", prompt)

            story_text = self._run_gemini(prompt, stream=False)

            age_tier = self._get_age_tier(age)
            story_metadata = {
                "child_name": child_name,
                "age": age,
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
                )

            proc = self._run_gemini(prompt, stream=True)
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
