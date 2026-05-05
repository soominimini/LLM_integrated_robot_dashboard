#!/usr/bin/env python3.9

"""Persona RAG loader for the SAR therapeutic system.

Loads `documents/personas_rag.json` and, given a child's age and diagnosis
string (as stored in `profile.json` -> `disorder`), retrieves the closest
matching reference persona and formats its therapy goals, interests, and
constraints into prompt fragments ready to be injected into the story /
question generators.
"""

import json
import os
from typing import Any, Dict, List, Optional


DEFAULT_JSON_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'documents', 'personas_rag.json'
)


class PersonaRAG:

    def __init__(self, json_path: str = DEFAULT_JSON_PATH):
        self.json_path = json_path
        self._personas: List[Dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        try:
            with open(self.json_path, 'r') as f:
                data = json.load(f)
            self._personas = data.get('personas', []) or []
        except (OSError, json.JSONDecodeError) as e:
            print(f"[PersonaRAG] Failed to load {self.json_path}: {e}")
            self._personas = []

    @property
    def personas(self) -> List[Dict[str, Any]]:
        return self._personas

    # ─────────────────────────────────────────────
    # MATCHING
    # ─────────────────────────────────────────────

    @staticmethod
    def _score(persona: Dict[str, Any], age: int, disorder: str) -> float:
        """Higher is better. Combines diagnosis keyword match + age proximity."""
        score = 0.0

        disorder_norm = (disorder or '').lower()
        dx = persona.get('diagnosis', {}) or {}
        primary = (dx.get('primary') or '').lower()
        secondary = ' '.join(dx.get('secondary', []) or []).lower()
        persona_dx = f"{primary} {secondary}"

        # Specific severity / subtype keywords first (strongest signal).
        for kw in ('hfa', 'high-functioning', 'high functioning'):
            if kw in disorder_norm and ('high-functioning' in persona_dx or 'hfa' in persona_dx):
                score += 20
                break
        if 'moderate' in disorder_norm and 'moderate' in persona_dx:
            score += 15
        if 'mild' in disorder_norm and 'mild' in persona_dx:
            score += 15

        # Generic ASD / autism overlap.
        if ('asd' in disorder_norm or 'autism' in disorder_norm) and (
            'asd' in persona_dx or 'autism' in persona_dx
        ):
            score += 5

        # Speech / language delay overlap (common secondary).
        if 'speech' in disorder_norm or 'language' in disorder_norm:
            if 'speech' in persona_dx or 'language' in persona_dx:
                score += 3

        # Age proximity penalty (2 points per year of difference).
        persona_age = (persona.get('age') or {}).get('years', 0)
        try:
            score -= abs(int(age) - int(persona_age)) * 2
        except (TypeError, ValueError):
            pass

        return score

    def match(self, age: int, disorder: str = '') -> Optional[Dict[str, Any]]:
        """Return the persona that best matches (age, disorder), or None if none loaded."""
        if not self._personas:
            return None
        best = None
        best_score = float('-inf')
        for p in self._personas:
            s = self._score(p, age, disorder)
            if s > best_score:
                best_score = s
                best = p
        return best

    # ─────────────────────────────────────────────
    # PROMPT FRAGMENTS
    # ─────────────────────────────────────────────

    @staticmethod
    def _fmt_list(items: List[str]) -> str:
        return '\n'.join(f"- {x}" for x in items if str(x).strip())

    @staticmethod
    def _fmt_constraints(constraints: Dict[str, Any]) -> str:
        lines: List[str] = []
        for k, v in (constraints or {}).items():
            if isinstance(v, list):
                if v:
                    lines.append(f"- {k}: {', '.join(str(x) for x in v)}")
            elif isinstance(v, dict):
                # Flatten one level (rare in this JSON).
                inner = '; '.join(f"{ik}={iv}" for ik, iv in v.items())
                lines.append(f"- {k}: {inner}")
            elif v is not None and str(v).strip():
                lines.append(f"- {k}: {v}")
        return '\n'.join(lines)

    def build_story_context(self, persona: Dict[str, Any]) -> str:
        """Block of prompt text describing the persona for story generation."""
        if not persona:
            return ''

        name = persona.get('name', 'the child')
        age_display = (persona.get('age') or {}).get('display', '')
        dx_primary = (persona.get('diagnosis') or {}).get('primary', '')

        goals = persona.get('therapy_goals', {}) or {}
        goals_summary = self._fmt_list(goals.get('summary', []) or [])

        structured = goals.get('structured', []) or []
        structured_lines = []
        for item in structured:
            goal = item.get('goal', '').strip()
            sp = item.get('system_prompt', '').strip()
            examples = item.get('examples', []) or []
            ex = f" Examples: {', '.join(examples)}." if examples else ''
            if goal or sp:
                structured_lines.append(f"- {goal}: {sp}{ex}".strip())
        structured_block = '\n'.join(structured_lines)

        interest_lines = []
        for i in persona.get('interests', []) or []:
            topic = i.get('topic', '').strip()
            use = (
                i.get('storytelling_use')
                or i.get('therapeutic_use')
                or i.get('activity_use')
                or ''
            ).strip()
            if topic:
                interest_lines.append(f"- {topic}: {use}" if use else f"- {topic}")
        interests_block = '\n'.join(interest_lines)

        constraints_block = self._fmt_constraints(persona.get('constraints', {}) or {})

        return (
            "--- PERSONA CONTEXT (retrieved reference profile) ---\n"
            f"Reference persona: {name} ({age_display}) — {dx_primary}.\n"
            "Use this persona's therapy goals and interests to shape the story. "
            "Do not mention the persona's name; instead adapt the narrative to the "
            "actual child's name and age.\n\n"
            "Therapy goals (high-level):\n"
            f"{goals_summary or '- (none specified)'}\n\n"
            "Structured language targets (weave naturally into narration and dialogue):\n"
            f"{structured_block or '- (none specified)'}\n\n"
            "Interests to use as story hooks and settings:\n"
            f"{interests_block or '- (none specified)'}\n\n"
            "Persona-specific constraints (must respect):\n"
            f"{constraints_block or '- (none specified)'}\n"
        )

    def build_question_context(self, persona: Dict[str, Any]) -> str:
        """Compact block describing the persona for question generation."""
        if not persona:
            return ''

        name = persona.get('name', 'the child')
        age_display = (persona.get('age') or {}).get('display', '')
        dx_primary = (persona.get('diagnosis') or {}).get('primary', '')

        goals = persona.get('therapy_goals', {}) or {}
        summary = self._fmt_list(goals.get('summary', []) or [])

        # For questions, only keep the goal + one-liner system_prompt (no examples).
        structured = goals.get('structured', []) or []
        structured_lines = []
        for item in structured:
            goal = item.get('goal', '').strip()
            sp = item.get('system_prompt', '').strip()
            if goal:
                structured_lines.append(f"- {goal}: {sp}")
        structured_block = '\n'.join(structured_lines)

        interests = [
            i.get('topic', '').strip()
            for i in persona.get('interests', []) or []
            if i.get('topic')
        ]
        interests_line = ', '.join(interests)

        constraints_block = self._fmt_constraints(persona.get('constraints', {}) or {})

        return (
            "--- PERSONA CONTEXT (retrieved reference profile) ---\n"
            f"Reference persona: {name} ({age_display}) — {dx_primary}.\n"
            "Tailor question wording and content using this persona's therapy goals "
            "and interests. Respect the constraints below.\n\n"
            "Therapy goals (high-level):\n"
            f"{summary or '- (none specified)'}\n\n"
            "Structured language targets to embed:\n"
            f"{structured_block or '- (none specified)'}\n\n"
            f"Interests to draw from: {interests_line or '(none specified)'}\n\n"
            "Persona-specific constraints:\n"
            f"{constraints_block or '- (none specified)'}\n"
        )

    # ─────────────────────────────────────────────
    # CONVENIENCE
    # ─────────────────────────────────────────────

    def build_story_prompt_fragment(self, age: int, disorder: str = '') -> str:
        """One-shot: match + format for story generation."""
        persona = self.match(age=age, disorder=disorder)
        return self.build_story_context(persona) if persona else ''

    def build_question_prompt_fragment(self, age: int, disorder: str = '') -> str:
        """One-shot: match + format for question generation."""
        persona = self.match(age=age, disorder=disorder)
        return self.build_question_context(persona) if persona else ''
