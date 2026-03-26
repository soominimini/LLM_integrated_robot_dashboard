#!/usr/bin/env python3.9

import os
import argparse
import sys
import json
from google import genai
from google.genai import types


PROMPTS = {
    "toy": """Look at this image from a robot's camera. A child may be holding or showing a toy or object.
Identify the object and generate a short, warm, conversational sentence the robot should say to the child.
The sentence should mention the object and ask a simple engaging question about it.

IMPORTANT: The question must be age-appropriate for a {age}-year-old child.
- For ages 2-3: Use very simple words (1-2 syllables), yes/no questions, or single-choice questions. Example: "Is that a cookie? Yummy!"
- For ages 4-5: Use simple sentences with basic "what" or "do you like" questions. Example: "That's a cookie! Do you like cookies?"
- For ages 6-7: Use slightly more complex questions that invite short answers. Example: "I see a cookie! What's your favourite kind of cookie?"
- For ages 8+: Use open-ended questions that encourage conversation. Example: "That's a cookie! Have you ever tried baking cookies? What kind would you make?"

If you cannot clearly see any object being held or shown, set object to null.
When object is null, generate a gentle prompt like:
- "What do you have there? Can you show me?"
- "Do you have a toy? Show it to me!"

Return ONLY a JSON object: {{"text": "<the sentence>", "object": "<detected object name or null>"}}""",

    "toy_followup": """Look at this image from a robot's camera. A child is showing a toy or object.
Identify the object and generate a short, warm, positive follow-up statement about it.
This is a follow-up after the child responded to the robot's request to show a toy.
The tone should be excited and encouraging.

IMPORTANT: The response must be age-appropriate for a {age}-year-old child.
- For ages 2-3: Very short, simple, excited reactions with basic words. Example: "Wow, cookie! Yummy yummy!"
- For ages 4-5: Short excited sentences with simple words. Example: "Oh wow, a cookie toy! That looks yummy! I love cookies too!"
- For ages 6-7: Slightly richer responses with a relatable comment. Example: "A cookie! That looks delicious! I bet it's chocolate chip, those are the best!"
- For ages 8+: Engaging responses that show genuine interest. Example: "That's a cool cookie toy! It looks so realistic! Do you collect toy food?"

Return ONLY a JSON object: {{"text": "<the sentence>", "object": "<detected object name or null>"}}""",

    "child": """Look at this image from a robot's camera. Focus on the child's appearance:
their clothing, colors they are wearing, accessories, hairstyle, or anything visually notable.
Generate a short, warm, friendly compliment or observation the robot should say to the child.

IMPORTANT: The compliment must be age-appropriate for a {age}-year-old child.
- For ages 2-3: Very simple observations about colors or characters. Example: "Blue shirt! So pretty!"
- For ages 4-5: Simple, enthusiastic compliments. Example: "You're wearing a blue shirt today! Blue is such a cool color!"
- For ages 6-7: Slightly more detailed observations. Example: "I really like your blue shirt! Is blue your favourite color? It looks great on you!"
- For ages 8+: More conversational compliments. Example: "That's a nice blue shirt! It goes really well with your style today!"

If you cannot clearly see the child, respond with: "Hey there! You look great today!"

Return ONLY a JSON object: {{"text": "<the sentence>", "observation": "<what you noticed or null>"}}"""
}


def main():
    MODEL_ID = os.getenv("GEMINI_VISION_MODEL", "gemini-2.5-flash")
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY or GOOGLE_API_KEY")

    p = argparse.ArgumentParser(description="Generate recovery question from camera image")
    p.add_argument('--image', required=True, help='Path to image file (jpg/png)')
    p.add_argument('--mode', required=True, choices=['toy', 'toy_followup', 'child'], help='Question type')
    p.add_argument('--child-name', default='', help='Child name to personalize')
    p.add_argument('--child-age', type=int, default=5, help='Child age for age-appropriate language')
    args = p.parse_args()

    with open(args.image, 'rb') as f:
        image_bytes = f.read()

    prompt = PROMPTS[args.mode].format(age=args.child_age)
    if args.child_name:
        prompt += f"\n\nThe child's name is {args.child_name}. You may use their name naturally in the sentence."

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=MODEL_ID,
        contents=[
            types.Part.from_bytes(
                data=image_bytes,
                mime_type='image/jpeg',
            ),
            prompt
        ],
        config=types.GenerateContentConfig(
            temperature=0.7,
        )
    )

    raw = (response.text or '').strip()
    # Try to parse JSON from response
    try:
        if raw.startswith('```'):
            raw = raw.strip('`').strip()
            if raw.startswith('json'):
                raw = raw[4:].strip()
        result = json.loads(raw)
        print(json.dumps(result))
    except Exception:
        # Fallback: wrap raw text
        print(json.dumps({"text": raw}))

    return 0


if __name__ == '__main__':
    sys.exit(main())
