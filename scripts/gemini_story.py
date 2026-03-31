#!/usr/bin/env python3.9
"""Generate a therapeutic story using the Gemini API.

Called as a subprocess by StoryGenerator.
Accepts the prompt via --prompt-file (path to a text file) or --prompt (inline).
Prints the generated story text to stdout.

For streaming mode, pass --stream to print chunks as they arrive (one per line,
prefixed with "CHUNK:").
"""

import os
import sys
import argparse
from google import genai
from google.genai import types

SYSTEM_PROMPT = (
    "You are a clinical storyteller for pediatric speech-language therapy. "
    "You create personalized therapeutic stories that are age-calibrated, "
    "clinically grounded, and engaging. You follow word count constraints precisely. "
    "You integrate therapy goals into narrative and dialogue naturally, never as explicit lessons. "
    "You never add preamble, commentary, or text outside the requested output format."
)


def main():
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: Missing GEMINI_API_KEY", file=sys.stderr)
        return 1

    p = argparse.ArgumentParser()
    p.add_argument('--model', default='gemini-2.5-flash')
    p.add_argument('--prompt', default=None, help='Inline prompt text')
    p.add_argument('--prompt-file', default=None, help='Path to file containing prompt')
    p.add_argument('--stream', action='store_true', help='Stream output chunks')
    args = p.parse_args()

    if args.prompt_file:
        with open(args.prompt_file, 'r') as f:
            prompt = f.read()
    elif args.prompt:
        prompt = args.prompt
    else:
        print("ERROR: --prompt or --prompt-file required", file=sys.stderr)
        return 1

    client = genai.Client(api_key=api_key)

    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        temperature=0.8,
        max_output_tokens=4096,
    )

    if args.stream:
        response = client.models.generate_content_stream(
            model=args.model,
            contents=prompt,
            config=config,
        )
        for chunk in response:
            if chunk.text:
                # Print each chunk on its own line with prefix
                for line in chunk.text.split('\n'):
                    print(f"CHUNK:{line}", flush=True)
    else:
        response = client.models.generate_content(
            model=args.model,
            contents=prompt,
            config=config,
        )
        print(response.text or '')

    return 0


if __name__ == '__main__':
    sys.exit(main())
