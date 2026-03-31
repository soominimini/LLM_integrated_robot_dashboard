#!/usr/bin/env python3.9
"""General-purpose Gemini text generation.

Accepts a prompt via --prompt or --prompt-file, returns the response to stdout.
Used by the server for tasks that need a general LLM (not quiz-specific).
"""

import os
import sys
import argparse
from google import genai
from google.genai import types


def main():
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: Missing GEMINI_API_KEY", file=sys.stderr)
        return 1

    p = argparse.ArgumentParser()
    p.add_argument('--model', default='gemini-2.5-flash')
    p.add_argument('--prompt', default=None)
    p.add_argument('--prompt-file', default=None)
    p.add_argument('--system', default='You are a helpful assistant. Return JSON only when asked.')
    p.add_argument('--max-tokens', type=int, default=2048)
    p.add_argument('--temperature', type=float, default=0.3)
    args = p.parse_args()

    if args.prompt_file:
        with open(args.prompt_file, 'r') as f:
            prompt = f.read()
    elif args.prompt:
        prompt = args.prompt
    else:
        prompt = sys.stdin.read()

    if not prompt.strip():
        print("ERROR: Empty prompt", file=sys.stderr)
        return 1

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=args.model,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=args.system,
            temperature=args.temperature,
            max_output_tokens=args.max_tokens,
        ),
    )
    print(response.text or '')
    return 0


if __name__ == '__main__':
    sys.exit(main())
