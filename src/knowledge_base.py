#!/usr/bin/env python3.9

"""Language & Interest knowledge-base loader for the SAR therapeutic system.

Replaces the older persona-matching RAG (``persona_rag.py``). Instead of
selecting one of a handful of fixed clinical personas by ``(age, diagnosis)``,
this loads ``documents/SLP_codesign_knowledge_base_integrated_v1_1.json`` and,
given a child's age and gender, derives:

  * the developmentally-appropriate **language targets** (with an MLU range),
  * the developmentally-appropriate **articulation / speech-sound targets**
    (phonemes + example words), and
  * age / gender-appropriate **interest themes**,

then formats them into prompt fragments ready to be injected into the story /
question generators.

The public interface intentionally mirrors ``PersonaRAG`` so the rest of the
application only needs its data source swapped:

    build_story_prompt_fragment(age, gender)    -> str
    build_question_prompt_fragment(age, gender)  -> str
"""

import json
import os
from typing import Any, Dict, List, Optional, Tuple


DEFAULT_JSON_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'documents', 'SLP_codesign_knowledge_base_integrated_v1_1.json'
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
        self._load()

    def _load(self) -> None:
        try:
            with open(self.json_path, 'r') as f:
                data = json.load(f)
            ldf = data.get('language_development_framework', {}) or {}
            self._language_targets = ldf.get('language_targets', {}) or {}
            self._levels = ldf.get('developmental_levels', []) or []
            ssf = data.get('speech_sound_development_framework', {}) or {}
            self._articulation_targets = ssf.get('articulation_targets', {}) or {}
            self._speech_levels = ssf.get('developmental_levels', []) or []
            intf = data.get('interest_framework', {}) or {}
            self._themes = intf.get('themes', {}) or {}
            self._age_prefs = intf.get('age_preferences', []) or []
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

    # ─────────────────────────────────────────────
    # FORMATTING
    # ─────────────────────────────────────────────

    def _targets_block(self, age: Any) -> str:
        lines: List[str] = []
        for key, desc, examples in self.resolve_targets(age):
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
                                    language_age: Any = None) -> str:
        """Narrative-shaped guidance block for story generation.

        ``age`` is the child's chronological age and drives interest themes.
        ``language_age`` is the developmental / language age and drives the MLU
        target + language targets; when ``None`` it falls back to ``age``. This
        lets an older child with a language delay be targeted at a lower MLU
        (e.g. a 9-year-old with an MLU-6-8 target -> language_age 5).
        """
        lang_age = language_age if language_age is not None else age
        level = self.resolve_level(lang_age)
        if not level:
            return ''
        targets = self._targets_block(lang_age)
        sounds = self._articulation_block(lang_age)
        interests = self._interests_line(age, gender)
        sounds_section = (
            "\nPractise these target speech sounds by naturally featuring words "
            "that contain them (do not turn the story into a pronunciation drill):\n"
            f"{sounds}\n"
        ) if sounds else ""
        return (
            "--- LANGUAGE & INTEREST GUIDANCE (knowledge base) ---\n"
            f"Target developmental level: age {level.get('age')}, "
            f"approx MLU {level.get('mlu_range', '')} words per utterance. "
            "Keep sentences at or near this length.\n\n"
            "Weave these language targets naturally into narration and dialogue "
            "(model them in context; do not drill or quiz them in the story):\n"
            f"{targets or '- (none specified)'}\n"
            f"{sounds_section}"
            "\nUse these interest themes as story hooks, characters, and settings:\n"
            f"- {interests or '(none specified)'}\n"
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
