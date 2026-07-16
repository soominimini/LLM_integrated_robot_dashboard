#!/usr/bin/env python3.9

"""SLP knowledge-base loader for the SAR therapeutic system.

Replaces the older persona-matching RAG (``persona_rag.py``). Instead of
selecting one of a handful of fixed clinical personas by ``(age, diagnosis)``,
this loads ``documents/restructured_knowledge_base_v2.json`` (the five clinical
frameworks live under its top-level ``frameworks`` key) and, given a child's
age and gender, derives:

  * **language** — grammar/phrase-length targets with an MLU range, plus the
    WH-question developmental hierarchy (what/who -> where -> when -> why/how),
  * **speech_sound** — articulation targets (phonemes + example words),
  * **concept** — what a question may *test*, from object identity through
    category membership and food origin to causal/narrative reasoning,
  * **social_communication** — social-pragmatic targets (reciprocal
    conversation, peer entry, emotion understanding, perspective taking,
    conflict resolution, self-advocacy) and the KB's neurodiversity
    guardrails, and
  * **interests** — age/gender-appropriate themes, used for personalization
    only and never to determine the clinical target,

then formats them into prompt fragments ready to be injected into the story /
question generators.

The frameworks are deliberately independent, which is the KB's central design
principle: ``language`` controls *wording* level (MLU, grammar) while
``concept`` and ``social_communication`` control *content* level. A child can
be targeted at MLU 6-8 while reasoning at concept age 8 — hence the
``language_age`` parameter, which decouples wording from chronological age.

Level resolution is uniform across frameworks: take the highest level at or
below the child's age, falling back to the lowest level when the child is
younger than all of them. Levels are defined at sparse ages, so an age with no
level of its own reuses a neighbouring one rather than over-targeting (see
ARCHITECTURE.md §9.1 for the resolved age -> level table and its gaps).

Prompt fragments — each returns '' when the KB carries no relevant data:

    build_story_prompt_fragment(age, gender, language_age=, interest=,
                                social_theme=)                            -> str
    build_question_prompt_fragment(age, gender, language_age=,
                                   include_targets=)                      -> str
    build_concept_prompt_fragment(concept_age, max_targets=, types=)      -> str
    build_social_prompt_fragment(age, max_targets=, target_ids=)          -> str
    build_wh_question_guidance_fragment(age, language_age=, image_card=)  -> str

Social-theme selection for coverage rotation: ``social_theme_ids(age)`` lists
the level's target ids, ``social_theme(id)`` resolves one to
(id, description, skills), ``pick_random_social_theme(age)`` is the
no-memory fallback.

``describe(age, gender, language_age=)`` returns a compact dict of what was
derived, for logging.

The first two fragments intentionally mirror ``PersonaRAG``'s old interface so
the rest of the application only needed its data source swapped. The class name
``LanguageInterestKB`` predates the concept and social_communication
frameworks; it now covers all five.
"""

import json
import os
import random
from typing import Any, Dict, List, Optional, Tuple


DEFAULT_JSON_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'documents', 'restructured_knowledge_base_v2.json'
)


def _normalize_gender(gender: str) -> str:
    """Map free-form profile gender onto the KB's 'boys' / 'girls' buckets.

    Returns '' for unknown/blank, signalling the caller to merge both buckets.
    """
    g = (gender or '').strip().lower()
    if g in ('m', 'male', 'boy', 'boys', 'man'):
        return 'boys'
    if g in ('f', 'female', 'girl', 'girls', 'woman'):
        return 'girls'
    return ''


class LanguageInterestKB:

    def __init__(self, json_path: str = DEFAULT_JSON_PATH):
        self.json_path = json_path
        self._language_targets: Dict[str, Any] = {}
        self._levels: List[Dict[str, Any]] = []
        self._articulation_targets: Dict[str, Any] = {}
        self._speech_levels: List[Dict[str, Any]] = []
        self._themes: Dict[str, List[str]] = {}
        self._age_prefs: List[Dict[str, Any]] = []
        self._wh_hierarchy: Dict[str, Any] = {}
        self._image_card_wh: Dict[str, Any] = {}
        self._concept_targets: Dict[str, Any] = {}
        self._concept_levels: List[Dict[str, Any]] = []
        self._social_targets: Dict[str, Any] = {}
        self._social_levels: List[Dict[str, Any]] = []
        self._social_notes: List[str] = []
        self._load()

    def _load(self) -> None:
        try:
            with open(self.json_path, 'r') as f:
                data = json.load(f)
            fw = data.get('frameworks', {}) or {}
            lang = fw.get('language', {}) or {}
            self._language_targets = lang.get('targets', {}) or {}
            self._levels = lang.get('developmental_levels', []) or []
            speech = fw.get('speech_sound', {}) or {}
            self._articulation_targets = speech.get('targets', {}) or {}
            self._speech_levels = speech.get('developmental_levels', []) or []
            intf = fw.get('interests', {}) or {}
            # v2 themes are rich objects ({description, generic_examples,
            # specific_examples, generation_constraints}); downstream code only
            # needs the example items, so normalize to theme -> [items].
            # Dict-shaped specific_examples are branded characters (Superman,
            # Roblox, ...) which the KB's own rules reserve for children whose
            # profile explicitly names that interest — we have no such signal
            # here, so keep only plain-string items.
            self._themes = {}
            for key, spec in (intf.get('themes', {}) or {}).items():
                if isinstance(spec, dict):
                    items = [x for x in (spec.get('generic_examples', []) or [])
                             if isinstance(x, str)]
                    items += [x for x in (spec.get('specific_examples', []) or [])
                              if isinstance(x, str) and x not in items]
                    self._themes[key] = items
                else:  # tolerate the old plain-list shape
                    self._themes[key] = spec or []
            self._age_prefs = intf.get('co_design_observed_age_preferences', []) or []
            # WH-question developmental hierarchy + image-card generation rules.
            self._wh_hierarchy = lang.get('wh_question_hierarchy', {}) or {}
            self._image_card_wh = data.get(
                'image_card_wh_question_generation_guidance', {}) or {}
            # Conceptual-knowledge framework (educational quiz difficulty).
            concept = fw.get('concept', {}) or {}
            self._concept_targets = concept.get('targets', {}) or {}
            self._concept_levels = concept.get('developmental_levels', []) or []
            # Social-communication framework (peer interaction, emotion
            # understanding, perspective taking, self-advocacy). Drives the
            # social-pragmatic quiz topic.
            social = fw.get('social_communication', {}) or {}
            self._social_targets = social.get('targets', {}) or {}
            self._social_levels = social.get('developmental_levels', []) or []
            self._social_notes = social.get('notes', []) or []
        except (OSError, json.JSONDecodeError) as e:
            print(f"[KB] Failed to load {self.json_path}: {e}")

    # ─────────────────────────────────────────────
    # SELECTION
    # ─────────────────────────────────────────────

    @staticmethod
    def _pick_by_age(entries: List[Dict[str, Any]], age: Any) -> Optional[Dict[str, Any]]:
        """Highest entry whose ``age`` <= the child's age; else the lowest entry.

        Levels/preferences are defined at sparse ages (2,3,4,5,6,8 etc.), so we
        target at-or-just-below the child's age rather than over-targeting.
        """
        if not entries:
            return None
        try:
            a = int(age)
        except (TypeError, ValueError):
            a = 0
        eligible = [e for e in entries if int(e.get('age', 0)) <= a]
        if eligible:
            return max(eligible, key=lambda e: int(e.get('age', 0)))
        return min(entries, key=lambda e: int(e.get('age', 0)))

    def resolve_level(self, age: Any) -> Optional[Dict[str, Any]]:
        """The developmental level (age, mlu_range, targets) for this child."""
        return self._pick_by_age(self._levels, age)

    def resolve_targets(self, age: Any) -> List[Tuple[str, str, List[str]]]:
        """List of (key, description, examples) for the child's level.

        Target keys missing from ``language_targets`` are skipped gracefully.
        """
        level = self.resolve_level(age)
        out: List[Tuple[str, str, List[str]]] = []
        if not level:
            return out
        for key in level.get('targets', []) or []:
            spec = self._language_targets.get(key)
            if not spec:
                continue
            out.append((key, spec.get('description', ''), spec.get('examples', []) or []))
        return out

    @staticmethod
    def _range_lower_bound(age_range: str) -> int:
        """Leading integer of an age-range string like '2-3' or '5+' (0 if none)."""
        s = (age_range or '').strip()
        num = ''
        for ch in s:
            if ch.isdigit():
                num += ch
            else:
                break
        try:
            return int(num)
        except ValueError:
            return 0

    def resolve_speech_level(self, age: Any) -> Optional[Dict[str, Any]]:
        """The articulation level (age_range, difficulty_level, targets) for this child.

        Speech levels use overlapping ranges ('2-3','3-4','4-5','5+'); we target
        the stage that *begins* at or just below the child's age — i.e. the stage
        currently being developed — mirroring ``resolve_level``'s at-or-just-below
        philosophy for language.
        """
        if not self._speech_levels:
            return None
        try:
            a = int(age)
        except (TypeError, ValueError):
            a = 0
        eligible = [e for e in self._speech_levels
                    if self._range_lower_bound(e.get('age_range', '')) <= a]
        pool = eligible or self._speech_levels
        key = lambda e: self._range_lower_bound(e.get('age_range', ''))
        return (max if eligible else min)(pool, key=key)

    def resolve_articulation(
            self, age: Any) -> List[Tuple[str, str, List[str], List[str], List[str]]]:
        """List of (key, description, phonemes, example_words, example_phrases).

        Target keys missing from ``articulation_targets`` are skipped gracefully.
        """
        level = self.resolve_speech_level(age)
        out: List[Tuple[str, str, List[str], List[str], List[str]]] = []
        if not level:
            return out
        for key in level.get('targets', []) or []:
            spec = self._articulation_targets.get(key)
            if not spec:
                continue
            out.append((
                key,
                spec.get('description', ''),
                spec.get('phonemes', []) or [],
                spec.get('example_words', []) or [],
                spec.get('example_activity_phrases', []) or [],
            ))
        return out

    def resolve_interests(self, age: Any, gender: str) -> List[Tuple[str, List[str]]]:
        """List of (theme, items) appropriate to age + gender.

        Unknown/blank gender merges both buckets (deduped, order preserved).
        Theme keys missing from ``themes`` fall back to an empty item list, so
        the bare theme name is still usable downstream.
        """
        entry = self._pick_by_age(self._age_prefs, age)
        if not entry:
            return []
        g = _normalize_gender(gender)
        if g in ('boys', 'girls'):
            theme_keys = list(entry.get(g, []) or [])
        else:
            theme_keys = []
            seen = set()
            for k in list(entry.get('girls', []) or []) + list(entry.get('boys', []) or []):
                if k not in seen:
                    seen.add(k)
                    theme_keys.append(k)
        return [(k, self._themes.get(k, [])) for k in theme_keys]

    def pick_random_interest(self, age: Any, gender: str,
                             exclude: Optional[List[str]] = None
                             ) -> Optional[Tuple[str, List[str]]]:
        """Randomly pick ONE (theme, items) interest appropriate to age + gender.

        ``exclude`` lists theme keys to avoid (e.g. the previous story's
        interest) so consecutive stories rotate through different interests.
        If exclusion would empty the pool, the full pool is used instead.
        Returns None when the KB has no interests for this age/gender.
        """
        pool = self.resolve_interests(age, gender)
        if not pool:
            return None
        excluded = {str(t).strip().lower() for t in (exclude or []) if str(t).strip()}
        filtered = [(t, items) for (t, items) in pool if t.lower() not in excluded]
        return random.choice(filtered or pool)

    def social_theme_ids(self, age: Any) -> List[str]:
        """Ordered target ids of the resolved social level's primary targets
        (only those actually present in the targets dict)."""
        level = self.resolve_social_level(age)
        if not level:
            return []
        return [t for t in (level.get('primary_targets') or [])
                if t in self._social_targets]

    def social_theme(self, target_id: str
                     ) -> Optional[Tuple[str, str, List[str]]]:
        """(target_id, description, skills) for one social target, or None."""
        spec = self._social_targets.get(target_id)
        if not spec:
            return None
        return (target_id, spec.get('description', ''),
                list(spec.get('skills') or []))

    def pick_random_social_theme(self, age: Any
                                 ) -> Optional[Tuple[str, str, List[str]]]:
        """Randomly pick ONE social-communication target for story weaving.

        Returns (target_id, description, skills) from the resolved social
        level's primary targets, or None when the KB has no social framework.
        Callers gate by age (social story themes are for older children); the
        KB level itself resolves for any age via ``resolve_social_level``.
        Callers wanting no-repeat coverage should manage their own cycle over
        ``social_theme_ids`` and resolve with ``social_theme`` instead.
        """
        ids = self.social_theme_ids(age)
        if not ids:
            return None
        return self.social_theme(random.choice(ids))

    def resolve_wh_guidance(self, age: Any) -> Optional[Dict[str, Any]]:
        """WH-question guidance for this age.

        Prefers the developmental level's own ``wh_question_guidance`` block;
        falls back to ``wh_question_hierarchy.age_guidance`` buckets
        (age_2_3 / age_4 / age_5 / age_6_8) when the level carries none.
        """
        level = self.resolve_level(age)
        if level and level.get('wh_question_guidance'):
            return level['wh_question_guidance']
        buckets = self._wh_hierarchy.get('age_guidance', {}) or {}
        try:
            a = int(age)
        except (TypeError, ValueError):
            a = 0
        if a <= 3:
            key = 'age_2_3'
        elif a == 4:
            key = 'age_4'
        elif a == 5:
            key = 'age_5'
        else:
            key = 'age_6_8'
        return buckets.get(key)

    # ─────────────────────────────────────────────
    # FORMATTING
    # ─────────────────────────────────────────────

    def _targets_block(self, age: Any, include_wh: bool = True) -> str:
        lines: List[str] = []
        for key, desc, examples in self.resolve_targets(age):
            # WH-question targets are comprehension-question material, not
            # narration material — callers building story-narration guidance
            # exclude them (the WH hierarchy governs questions separately).
            if not include_wh and (self._language_targets.get(key) or {}).get('wh_types'):
                continue
            label = key.replace('_', ' ')
            ex = f" e.g. {', '.join(examples)}" if examples else ''
            if desc:
                lines.append(f"- {label} ({desc}):{ex}" if ex else f"- {label} ({desc})")
            else:
                lines.append(f"- {label}:{ex}" if ex else f"- {label}")
        return '\n'.join(lines)

    def _articulation_block(self, age: Any) -> str:
        lines: List[str] = []
        for key, desc, phonemes, words, _phrases in self.resolve_articulation(age):
            label = key.replace('_', ' ')
            head = f"{label} [{', '.join(phonemes)}]" if phonemes else label
            ex = f" e.g. {', '.join(words)}" if words else ''
            if desc:
                lines.append(f"- {head} ({desc}):{ex}" if ex else f"- {head} ({desc})")
            else:
                lines.append(f"- {head}:{ex}" if ex else f"- {head}")
        return '\n'.join(lines)

    def _articulation_line(self, age: Any) -> str:
        parts: List[str] = []
        for key, _desc, phonemes, words, _phrases in self.resolve_articulation(age):
            label = key.replace('_', ' ')
            head = f"{label} [{', '.join(phonemes)}]" if phonemes else label
            ex = f" (e.g. {', '.join(words[:3])})" if words else ''
            parts.append(f"{head}{ex}")
        return '; '.join(parts)

    def _interests_line(self, age: Any, gender: str) -> str:
        parts: List[str] = []
        for theme, items in self.resolve_interests(age, gender):
            label = theme.replace('_', ' ')
            parts.append(f"{label} ({', '.join(items)})" if items else label)
        return '; '.join(parts)

    def build_story_prompt_fragment(self, age: Any, gender: str = '',
                                    language_age: Any = None,
                                    interest: Optional[Tuple[str, List[str]]] = None,
                                    social_theme: Optional[Tuple[str, str, List[str]]] = None
                                    ) -> str:
        """Narrative-shaped guidance block for story generation.

        ``age`` is the child's chronological age and drives interest themes.
        ``language_age`` is the developmental / language age and drives the MLU
        target + language targets; when ``None`` it falls back to ``age``. This
        lets an older child with a language delay be targeted at a lower MLU
        (e.g. a 9-year-old with an MLU-6-8 target -> language_age 5).

        ``interest`` is an optional single (theme, items) tuple — typically from
        ``pick_random_interest`` — that the story must be built around. When
        ``None``, the fragment lists ALL age/gender-appropriate themes (the LLM
        then tends to gravitate to the same one every time).

        ``social_theme`` is an optional (target_id, description, skills) tuple —
        typically from ``pick_random_social_theme`` — woven into the story as a
        social-communication theme (KB frameworks.social_communication). Callers
        pass it for older children only.
        """
        lang_age = language_age if language_age is not None else age
        level = self.resolve_level(lang_age)
        if not level:
            return ''
        targets = self._targets_block(lang_age, include_wh=False)
        sounds = self._articulation_block(lang_age)
        sounds_section = (
            "\nPractise these target speech sounds by naturally featuring words "
            "that contain them (do not turn the story into a pronunciation drill):\n"
            f"{sounds}\n"
        ) if sounds else ""
        if interest:
            theme, items = interest
            label = theme.replace('_', ' ')
            interest_line = f"{label} ({', '.join(items)})" if items else label
            interests_section = (
                "\nBuild the story around this interest theme — use it for the "
                "story's hook, characters, and setting:\n"
                f"- {interest_line}\n"
            )
        else:
            interests = self._interests_line(age, gender)
            interests_section = (
                "\nUse these interest themes as story hooks, characters, and settings:\n"
                f"- {interests or '(none specified)'}\n"
            )
        social_section = ""
        if social_theme:
            tid, desc, skills = social_theme
            label = tid.replace('_', ' ')
            sk = f" Show characters {', '.join(skills[:4])}." if skills else ''
            social_section = (
                "\nWeave this social-communication theme into the story naturally — "
                "characters modelling the skills in a concrete situation, not a "
                "lecture or a moral tacked on at the end:\n"
                f"- {label}: {desc}{sk}\n"
            )
        return (
            "--- LANGUAGE & INTEREST GUIDANCE (knowledge base) ---\n"
            f"Target developmental level: age {level.get('age')}, "
            f"approx MLU {level.get('mlu_range', '')} words per utterance. "
            "Keep sentences at or near this length.\n\n"
            "Weave these language targets naturally into narration and dialogue "
            "(model them in context; do not drill or quiz them in the story):\n"
            f"{targets or '- (none specified)'}\n"
            f"{sounds_section}"
            f"{interests_section}"
            f"{social_section}"
        )

    def build_question_prompt_fragment(self, age: Any, gender: str = '',
                                       language_age: Any = None,
                                       include_targets: bool = True) -> str:
        """Compact guidance block for question generation.

        ``language_age`` (developmental age) drives MLU + targets; falls back to
        chronological ``age``. ``age`` drives interest selection.

        ``include_targets`` controls whether the speech-sound / grammar / interest
        targeting is included. When False, only the developmental wording-level
        (MLU length) calibration is returned — used by the educational quiz, where
        embedding target sounds/plurals into short questions distorts the content
        (e.g. "Do three cherries grow underground?").
        """
        lang_age = language_age if language_age is not None else age
        level = self.resolve_level(lang_age)
        if not level:
            return ''
        level_line = (
            f"Target level: age {level.get('age')}, approx MLU "
            f"{level.get('mlu_range', '')} words. Match question wording to this length.\n"
        )
        if not include_targets:
            return (
                "--- LANGUAGE GUIDANCE (knowledge base) ---\n"
                f"{level_line}"
            )
        targets = self._targets_block(lang_age)
        sounds_line = self._articulation_line(lang_age)
        interests = self._interests_line(age, gender)
        sounds_section = (
            "\nFavour words that use these target speech sounds where natural: "
            f"{sounds_line}\n"
        ) if sounds_line else ""
        return (
            "--- LANGUAGE & INTEREST GUIDANCE (knowledge base) ---\n"
            f"{level_line}\n"
            "Embed these language targets in the question wording where natural:\n"
            f"{targets or '- (none specified)'}\n"
            f"{sounds_section}"
            f"\nDraw question content from these interests: {interests or '(none specified)'}\n"
        )

    def build_wh_question_guidance_fragment(self, age: Any, language_age: Any = None,
                                            image_card: bool = False) -> str:
        """Compact WH-question TYPE guidance block from the KB hierarchy.

        Used by WH-flavoured generators (educational quiz `wh` questions, WH
        picture-scene) so question types follow the developmental order
        (what/who -> where -> when -> why/how) at the child's level.

        ``language_age`` (developmental age) drives selection; falls back to
        chronological ``age``. ``image_card`` appends the image-card selection
        steps and evidence rules. Returns '' when the KB carries no guidance.
        """
        lang_age = language_age if language_age is not None else age
        g = self.resolve_wh_guidance(lang_age) or {}
        if not g and not self._image_card_wh:
            return ''
        lines: List[str] = ["--- WH-QUESTION GUIDANCE (knowledge base) ---"]
        order = g.get('developmental_order') or \
            self._wh_hierarchy.get('developmental_order') or []
        if order:
            lines.append(
                "Developmental order (easiest first): " + " -> ".join(order) + ".")
        for key, label in (
                ('recommended_wh_types', 'Recommended WH types'),
                ('recommended_primary_wh_types', 'Primary WH types (prefer these)'),
                ('recommended_secondary_wh_types', 'Secondary WH types'),
                ('use_with_support', 'Use only with clear support'),
                ('emerging_or_optional', 'Emerging / optional'),
                ('avoid_or_use_with_support', 'Avoid or use only with support'),
                ('avoid_or_use_only_with_strong_support', 'Avoid without strong support'),
        ):
            vals = g.get(key) or []
            if vals:
                lines.append(f"{label}: {', '.join(vals)}.")
        for note in g.get('notes') or []:
            lines.append(f"- {note}")
        if image_card and self._image_card_wh:
            icw = self._image_card_wh
            # <=4: gentle what/who/where selection; >=5: additionally require
            # exactly ONE evidence-supported why/how question per set.
            try:
                a = int(lang_age)
            except (TypeError, ValueError):
                a = 0
            block = (icw.get('default_for_age_4_5', {}) if a <= 4
                     else icw.get('default_for_age_5_plus', {})
                     or icw.get('default_for_age_4_5', {})) or {}
            if block.get('question_selection'):
                lines.append("Question selection for this picture:")
                lines.extend(f"- {s}" for s in block['question_selection'])
            if block.get('fallback_rule'):
                lines.append(f"Fallback: {block['fallback_rule']}")
            if icw.get('evidence_rules'):
                lines.append("Evidence rules:")
                lines.extend(f"- {r}" for r in icw['evidence_rules'])
        return '\n'.join(lines) + '\n' if len(lines) > 1 else ''

    def resolve_concept_level(self, age: Any) -> Optional[Dict[str, Any]]:
        """Concept developmental level for this age.

        Concept levels use range strings ('2-3','4','5','6-7','8','9+'); pick
        the level whose lower bound is at or just below the age, mirroring
        ``resolve_speech_level``.
        """
        if not self._concept_levels:
            return None
        try:
            a = int(age)
        except (TypeError, ValueError):
            a = 0
        eligible = [e for e in self._concept_levels
                    if self._range_lower_bound(e.get('age_range', '')) <= a]
        pool = eligible or self._concept_levels
        key = lambda e: self._range_lower_bound(e.get('age_range', ''))
        return (max if eligible else min)(pool, key=key)

    _WH_WORDS = ('what', 'who', 'where', 'when', 'why', 'how', 'which')

    @classmethod
    def _question_form(cls, question: str) -> str:
        """Crude form classifier: 'wh' when the question starts with a WH word,
        'yes_no' otherwise."""
        words = (question or '').strip().strip('"').split()
        first = words[0].lower().rstrip(",'") if words else ''
        return 'wh' if first in cls._WH_WORDS else 'yes_no'

    def _pick_example(self, examples: List[str],
                      types: Optional[List[str]]) -> str:
        """First example whose form matches a requested question type; else the
        first example.

        Keeps the injected 'e.g.' aligned with the requested format — a
        yes/no-only quiz must not be shown a WH-form model question (and vice
        versa) for targets that carry examples of both forms.
        """
        if not examples:
            return ''
        if types:
            for q in examples:
                if self._question_form(q) in types:
                    return q
        return examples[0]

    def _suits_types(self, target_id: str, types: Optional[List[str]]) -> bool:
        """Whether a concept target suits any of the requested question types.

        Reads the target's ``suitable_question_types`` (yes_no / wh; why/how
        count as wh). A target that declares none is treated as suiting every
        type — the KB, not this filter, decides what is excluded.
        """
        if not types:
            return True
        declared = (self._concept_targets.get(target_id) or {}).get(
            'suitable_question_types')
        if not declared:
            return True
        return any(t in declared for t in types)

    def build_concept_prompt_fragment(self, concept_age: Any,
                                      max_targets: int = 12,
                                      types: Optional[List[str]] = None) -> str:
        """Conceptual-difficulty guidance block for the educational quiz.

        ``concept_age`` drives the concept level (typically the child's profile
        age, NOT their language age — wording level and concept level are
        independent KB domains). Targets are the level's ``primary_targets``.
        Returns '' when the KB has no concept framework.

        ``max_targets`` (12) is a safety backstop, sized above every level's
        curated list (the largest, age 8, carries 11) so no KB target is
        silently dropped; the per-type filter below is what actually trims.

        ``types`` are the requested question types (``yes_no`` / ``wh``). Targets
        are filtered to those whose ``suitable_question_types`` matches, so a
        yes/no quiz stops being handed reasoning targets the KB reserves for
        why/how — see ``concept.generation_rules.rule_6``. If the filter would
        empty the list the unfiltered list is used instead and a warning logged:
        an imprecise concept target beats none at all.
        """
        level = self.resolve_concept_level(concept_age)
        if not level:
            return ''
        lines: List[str] = ["--- CONCEPT GUIDANCE (knowledge base) ---"]
        desc = level.get('description', '')
        lines.append(f"Concept level (ages {level.get('age_range')}): {desc}")
        qg = level.get('question_guidance', {}) or {}
        if qg.get('wording'):
            lines.append(f"Wording: {qg['wording']}")
        if qg.get('recommended_question_length'):
            lines.append(f"Question length: {qg['recommended_question_length']}.")
        avoid = qg.get('avoid') or []
        if avoid:
            lines.append("Avoid: " + "; ".join(avoid))
        patterns = qg.get('recommended_question_patterns') or []
        if patterns:
            lines.append("Use these simple question frames (one clause each): "
                         + " | ".join(patterns))

        # Collect target ids from the level's primary targets, filtered by
        # question type.
        pool = list(level.get('primary_targets', []) or [])
        target_ids = [t for t in pool if self._suits_types(t, types)]
        if not target_ids:
            target_ids = list(pool)
            if target_ids:
                print(f"[KB] concept: no target suits types={types} at "
                      f"concept_age={concept_age}; falling back to the unfiltered list")
        dropped = [t for t in pool if t not in target_ids]
        if len(target_ids) > max_targets:
            dropped += target_ids[max_targets:]
            target_ids = target_ids[:max_targets]
        if dropped:
            print(f"[KB] concept: {len(dropped)} target(s) not sent "
                  f"(max_targets={max_targets}, types={types}): {dropped}")

        if target_ids:
            lines.append("Target these concepts (vary across the question set):")
            for tid in target_ids:
                spec = self._concept_targets.get(tid) or {}
                label = tid.replace('_', ' ')
                goal = spec.get('question_goal') or spec.get('description') or ''
                exq = self._pick_example(
                    spec.get('example_questions') or [], types)
                ex = f" e.g. {exq}" if exq else ''
                lines.append(f"- {label}: {goal}{ex}")
        for exq in (level.get('example_questions') or [])[:3]:
            lines.append(f"Example question at this level: {exq}")
        return '\n'.join(lines) + '\n'

    def resolve_social_level(self, age: Any) -> Optional[Dict[str, Any]]:
        """Social-communication level for this age.

        Levels use range strings ('8'); pick the level whose lower bound is at
        or just below the age, mirroring ``resolve_concept_level``. The KB
        currently defines a single age-8 level, so younger ages fall back onto
        it rather than resolving to nothing.
        """
        if not self._social_levels:
            return None
        try:
            a = int(age)
        except (TypeError, ValueError):
            a = 0
        eligible = [e for e in self._social_levels
                    if self._range_lower_bound(e.get('age_range', '')) <= a]
        pool = eligible or self._social_levels
        key = lambda e: self._range_lower_bound(e.get('age_range', ''))
        return (max if eligible else min)(pool, key=key)

    def build_social_prompt_fragment(self, age: Any,
                                     max_targets: Optional[int] = None,
                                     target_ids: Optional[List[str]] = None) -> str:
        """Social-communication guidance block for social-pragmatic questions.

        Supplies WHAT the social questions may cover: the level's primary
        targets (reciprocal conversation, emotion understanding, perspective
        taking, conflict resolution, self-advocacy, ...) with their component
        skills, plus the level's recommended contexts and its avoid rules —
        which include the KB's neurodiversity guardrails (no forced eye
        contact, no punishing autistic communication style). Returns '' when
        the KB has no social framework.

        ``target_ids`` restricts the fragment to those targets (order
        preserved, unknown ids skipped) — used by callers that rotate targets
        across generations for full coverage. When None, every primary target
        is listed. Each target line is prefixed with its id so the LLM can
        tag generated questions with ``social_target``.

        ``max_targets`` defaults to None (emit every listed target): a social
        level carries one fixed curated list, so capping it would silently
        drop clinical targets — the age-8 level has 10, and a cap of 8 always
        lost social_problem_solving and
        self_advocacy_and_communication_of_needs.
        """
        level = self.resolve_social_level(age)
        if not level:
            return ''
        lines: List[str] = ["--- SOCIAL-COMMUNICATION GUIDANCE (knowledge base) ---"]
        desc = level.get('description', '')
        lines.append(f"Social level (ages {level.get('age_range')}): {desc}")
        gg = level.get('generation_guidance', {}) or {}
        contexts = gg.get('recommended_contexts') or []
        if contexts:
            lines.append("Set questions in these contexts: " + ", ".join(contexts) + ".")
        avoid = gg.get('avoid') or []
        if avoid:
            lines.append("Avoid: " + "; ".join(avoid))
        if target_ids is not None:
            ids = [t for t in target_ids if t in self._social_targets]
        else:
            ids = list(level.get('primary_targets', []) or [])
        if max_targets is not None:
            ids = ids[:max_targets]
        if ids:
            lines.append("Cover a diverse mix of these social-communication targets "
                         "(do not repeat the same one):")
            for tid in ids:
                spec = self._social_targets.get(tid) or {}
                d = spec.get('description', '')
                skills = spec.get('skills') or []
                sk = f" Skills: {', '.join(skills[:4])}." if skills else ''
                lines.append(f"- {tid}: {d}{sk}")
        for note in self._social_notes[:3]:
            lines.append(f"- {note}")
        return '\n'.join(lines) + '\n'

    # ─────────────────────────────────────────────
    # DIAGNOSTICS
    # ─────────────────────────────────────────────

    def describe(self, age: Any, gender: str = '',
                 language_age: Any = None) -> Dict[str, Any]:
        """Compact summary of what was derived — handy for logging."""
        lang_age = language_age if language_age is not None else age
        level = self.resolve_level(lang_age)
        speech_level = self.resolve_speech_level(lang_age)
        return {
            'level_age': level.get('age') if level else None,
            'mlu_range': level.get('mlu_range') if level else None,
            'targets': [k for (k, _, _) in self.resolve_targets(lang_age)],
            'speech_age_range': speech_level.get('age_range') if speech_level else None,
            'speech_sounds': [k for (k, *_rest) in self.resolve_articulation(lang_age)],
            'interests': [t for (t, _) in self.resolve_interests(age, gender)],
        }
