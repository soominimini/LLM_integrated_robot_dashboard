#!/usr/bin/env python3.9
"""Generate a therapeutic story using the Claude API (Sonnet 4.6).

Called as a subprocess by StoryGenerator. Accepts the prompt via --prompt-file
(path to a text file) or --prompt (inline). Prints the generated story text to
stdout.

For streaming mode, pass --stream to print chunks as they arrive (one per line,
prefixed with "CHUNK:") — same protocol as scripts/gemini_story.py so the
parent's stream reader works unchanged.
"""

import os
import sys
import argparse
import anthropic

SYSTEM_PROMPT = (
    "You are a clinical storyteller for pediatric speech-language therapy. "
    "You create personalized therapeutic stories that are age-calibrated, "
    "clinically grounded, and engaging. You follow word count constraints precisely. "
    "You integrate therapy goals into narrative and dialogue naturally, never as explicit lessons. "
    "You never add preamble, commentary, or text outside the requested output format."
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='claude-sonnet-4-6')
    p.add_argument('--prompt', default=None, help='Inline prompt text')
    p.add_argument('--prompt-file', default=None, help='Path to file containing prompt')
    p.add_argument('--stream', action='store_true', help='Stream output chunks')
    args = p.parse_args()

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: Missing ANTHROPIC_API_KEY", file=sys.stderr)
        return 1

    if args.prompt_file:
        with open(args.prompt_file, 'r') as f:
            prompt = f.read()
    elif args.prompt:
        prompt = args.prompt
    else:
        print("ERROR: --prompt or --prompt-file required", file=sys.stderr)
        return 1

    client = anthropic.Anthropic(api_key=api_key)

    try:
        if args.stream:
            with client.messages.stream(
                model=args.model,
                max_tokens=4096,
                cache_control={"type": "ephemeral"},
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            ) as stream:
                for text in stream.text_stream:
                    if not text:
                        continue
                    for line in text.split('\n'):
                        print(f"CHUNK:{line}", flush=True)
        else:
            response = client.messages.create(
                model=args.model,
                max_tokens=4096,
                cache_control={"type": "ephemeral"},
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            text = next((b.text for b in response.content if b.type == "text"), "")
            print(text)
    except anthropic.APIError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
