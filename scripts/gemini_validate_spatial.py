#!/usr/bin/env python3.9
"""Validate a spatial-direction setup for the scene game's 'direction' mode.

The robot asks the child to arrange two named objects in a particular
spatial relation, e.g. "Put the banana under the blue block." This worker
takes a single camera frame and decides whether the arrangement matches.

Input (args):
    --image      path to the captured JPEG
    --obj-a      moving object the robot named first (e.g. "banana")
    --obj-b      reference object (e.g. "blue block")
    --relation   one of: next_to, above, under, behind, in_front_of
    --toy-list   optional comma-separated list of valid toys (constrains ID)

Output (stdout, single JSON line):
    {
      "obj_a_found":     true | false,
      "obj_b_found":     true | false,
      "actual_relation": "next_to" | "above" | "under" | "behind"
                       | "in_front_of" | "other",
      "correct":         true | false,
      "reason":          "<short child-friendly explanation>"
    }
"""

import os
import argparse
import sys
import json
from google import genai
from google.genai import types


# Canonical relation key -> phrase used by the robot when speaking the
# instruction. Single phrase here keeps prompt grounding unambiguous; the
# question generator on the server side may swap synonyms in the spoken
# sentence (e.g. "beside" instead of "next to") but always passes the
# canonical key to this worker.
RELATION_PHRASE = {
    "next_to": "next to",
    "above": "above",
    "under": "under",
    "behind": "behind",
    "in_front_of": "in front of",
    "in": "in",
    "out": "out of",
}


def main():
    # Gemini Robotics-ER is the embodied-reasoning model (also used by the
    # object detector); it is the default backend for single-frame spatial
    # validation. Override with SPATIAL_ER_MODEL.
    model_id = os.getenv("SPATIAL_ER_MODEL", "gemini-robotics-er-1.6-preview")
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY or GOOGLE_API_KEY")

    p = argparse.ArgumentParser(description="Validate spatial direction setup")
    p.add_argument('--image', required=True, help='Path to image file')
    p.add_argument('--obj-a', required=True, dest='obj_a',
                   help='Moving object named first in the instruction')
    p.add_argument('--obj-b', required=True, dest='obj_b',
                   help='Reference object the relation is measured against')
    p.add_argument('--relation', required=True,
                   choices=list(RELATION_PHRASE.keys()),
                   help='Canonical spatial relation key')
    p.add_argument('--toy-list', default='',
                   help='Comma-separated valid toys (optional)')
    args = p.parse_args()

    with open(args.image, 'rb') as f:
        image_bytes = f.read()

    rel_phrase = RELATION_PHRASE[args.relation]
    toy_clause = ""
    if args.toy_list:
        toy_clause = (
            f"The valid game objects are: {args.toy_list}. "
            "Only identify objects from this list.\n"
        )

    prompt = (
        "You are judging a children's spatial-direction game.\n"
        f"{toy_clause}"
        f"The child was asked to arrange the scene so that the {args.obj_a} is "
        f"{rel_phrase} the {args.obj_b}.\n"
        "\n"
        "The image is taken from the front of the child (camera-facing view).\n"
        "Decide:\n"
        f"1. Is the {args.obj_a} present in the scene?\n"
        f"2. Is the {args.obj_b} present in the scene?\n"
        f"3. What is the actual spatial relation of the {args.obj_a} TO the "
        f"{args.obj_b}? Pick ONE:\n"
        "   - next_to       (side by side, roughly same height)\n"
        "   - above         (higher than / on top of)\n"
        "   - under         (lower than / underneath)\n"
        "   - behind        (further from the camera, partially hidden)\n"
        "   - in_front_of   (closer to camera, may partially block the other)\n"
        "   - in            (inside / contained by the other; partially hidden by its walls or rim)\n"
        "   - out           (outside / not contained by the other; fully visible and separate)\n"
        "   - other         (none of the above clearly applies)\n"
        f"4. Does that match the requested relation '{args.relation}'?\n"
        "\n"
        "Tips:\n"
        "2D images cause spatial states to overlap (e.g., an object poking out 'under' a bowl also looks 'next_to' it). If the visual evidence supports multiple valid interpretations, give the benefit of the doubt and prioritize the child's requested relation over competing overlapping states."
        "- 'behind' means partially hidden by the reference object, or visibly\n"
        "  smaller/further along the camera's depth axis.\n"
        "- 'in_front_of' means the moving object partly occludes or sits\n"
        "  closer to the camera than the reference object.\n"
        "- 'in' means the moving object is contained by the reference object\n"
        "  (e.g. ball inside a cup or box) — typically partly hidden by the\n"
        "  rim/walls of the container.\n"
        "- 'out' means the moving object is clearly outside the reference\n"
        "  object, fully visible, with a visible gap between them.\n"
        "-  To be visible while 'under' a solid object (like a bowl), the moving object will be 'poking out.' Consider it 'under' if it is physically touching or partially hidden by the bottom edge or base of the reference object, even if it appears to be sitting next to or in front of it in 2D space.\n"
        "If you cannot tell confidently, return 'other'.\n"
        "\n"
        "Return ONLY a JSON object with no markdown fences:\n"
        "{\n"
        "  \"obj_a_found\": true|false,\n"
        "  \"obj_b_found\": true|false,\n"
        "  \"actual_relation\": \"next_to|above|under|behind|in_front_of|in|out|other\",\n"
        "  \"correct\": true|false,\n"
        "  \"reason\": \"<short, child-friendly explanation>\"\n"
        "}\n"
        "If either object is missing, set correct=false."
    )

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model_id,
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type='image/jpeg'),
            prompt,
        ],
        config=types.GenerateContentConfig(
            temperature=0.2,
            system_instruction=(
                "You judge a children's spatial-direction game from one photo. "
                "Return JSON only."),
        ),
    )

    raw = (response.text or '').strip()
    if raw.startswith('```'):
        raw = raw.strip('`').strip()
        if raw.startswith('json'):
            raw = raw[4:].strip()
    try:
        result = json.loads(raw)
    except Exception:
        result = {
            "obj_a_found": False,
            "obj_b_found": False,
            "actual_relation": "other",
            "correct": False,
            "reason": raw or "Could not parse vision response",
        }
    # Surface what was sent to / returned by Gemini so the UI can show the
    # therapist exactly how the verdict was reached.
    result["prompt"] = prompt
    result["raw_response"] = raw
    print(json.dumps(result))
    return 0


if __name__ == '__main__':
    sys.exit(main())
