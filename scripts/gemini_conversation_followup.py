#!/usr/bin/env python3.9

import os
import re
import sys
import json
from google import genai
from google.genai import types


def filter_english_only(text):
    """Remove non-English characters, keeping only ASCII letters, digits, basic punctuation, and spaces."""
    if not text:
        return ''
    # Keep only ASCII printable characters (English letters, digits, punctuation, spaces)
    filtered = re.sub(r'[^\x20-\x7E]', ' ', text)
    # Collapse multiple spaces
    filtered = re.sub(r'\s+', ' ', filtered).strip()
    return filtered


def main():
    MODEL_ID = os.getenv("GEMINI_VISION_MODEL", "gemini-2.5-flash")
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY or GOOGLE_API_KEY")

    # Read input from stdin as JSON
    input_data = json.loads(sys.stdin.read())
    theme = input_data.get("theme", "greeting")
    robot_said = input_data.get("robot_said", "")
    child_said_raw = input_data.get("child_said", "")
    child_name = input_data.get("child_name", "")
    child_age = input_data.get("child_age", 5)
    followup_number = input_data.get("followup_number", 1)
    total_followups = input_data.get("total_followups", 1)
    is_closing = input_data.get("is_closing", False)
    history = input_data.get("history", [])

    # Filter to English-only content
    child_said = filter_english_only(child_said_raw)
    # Check if there's enough meaningful content (at least 2 real words)
    meaningful_words = [w for w in child_said.split() if len(w) > 1 and w.isalpha()]
    has_meaningful_response = len(meaningful_words) >= 2

    # Build conversation history string
    history_text = ""
    for h in history:
        history_text += f"Robot: {h.get('robot', '')}\n"
        if h.get('child'):
            history_text += f"Child: {filter_english_only(h.get('child', ''))}\n"

    if is_closing:
        instruction = (
            "This is the CLOSING comment. Do NOT ask any question. "
            "Warmly acknowledge everything the child shared during this conversation. "
            "Summarize what they said positively and praise them for talking. "
            "Keep it general and encouraging — say something like 'I really enjoyed hearing about that!' or 'It was so nice talking with you!'. "
            "Do NOT say goodbye, bye bye, see you later, or any farewell. "
            "Do NOT say anything about leaving, ending, or going away. "
            "Just give a warm, positive comment about the conversation itself. "
            "Examples: 'That was so fun talking with you! You told me so many cool things!', "
            "'I really enjoyed hearing about that! You're such a great talker!', "
            "'It was so nice chatting! I loved hearing your stories!'"
        )
    else:
        instruction = (
            "You MUST end your response with a clear, simple question for the child to answer. "
            "The question should be easy for a {age}-year-old to understand and respond to. "
            "Do NOT end with a statement — always end with a question mark."
        ).format(age=child_age)

    if not has_meaningful_response and not is_closing:
        child_context = (
            f"The child's response was unclear or too short to understand (heard: '{child_said}'). "
            "Do NOT try to reference what the child said. Instead, generate a NEW question "
            f"related to the theme of {theme.replace('_', ' ')} that the child can easily answer. "
            "Be encouraging and make it fun."
        )
    else:
        child_context = (
            f"The child said: \"{child_said}\"\n"
            "Acknowledge what the child said to show you listened, then respond."
        )

    prompt = f"""You are a friendly social robot having a conversation with a child.

CONTEXT:
- Theme: {theme.replace('_', ' ')}
- Child's name: {child_name or 'the child'}
- Child's age: {child_age} years old
- This is {"the closing comment" if is_closing else f"follow-up {followup_number} of {total_followups}"}

CONVERSATION SO FAR:
{history_text}Robot: {robot_said}
{child_context}

TASK:
Generate a warm, natural response.
Rules:
1. Be age-appropriate for a {child_age}-year-old:
   - Ages 2-3: Very short, simple, enthusiastic (1-2 short sentences). Use simple yes/no or choice questions.
   - Ages 4-5: Simple sentences (2-3 max). Ask easy "what" or "do you like" questions.
   - Ages 6-7: Natural conversation (2-3 sentences). Ask slightly more open questions.
   - Ages 8+: More conversational (2-3 sentences). Ask open-ended questions showing genuine interest.
2. Stay on the theme of {theme.replace('_', ' ')}
3. {instruction}

Return ONLY a JSON object: {{"text": "<your response>", "acknowledged": "<brief summary of what child said or 'unclear'>"}}"""

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=MODEL_ID,
        contents=[prompt],
        config=types.GenerateContentConfig(
            temperature=0.7,
        )
    )

    raw = (response.text or '').strip()
    try:
        if raw.startswith('```'):
            raw = raw.strip('`').strip()
            if raw.startswith('json'):
                raw = raw[4:].strip()
        result = json.loads(raw)
        print(json.dumps(result))
    except Exception:
        print(json.dumps({"text": raw}))

    return 0


if __name__ == '__main__':
    sys.exit(main())
