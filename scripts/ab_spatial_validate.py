#!/usr/bin/env python3.9
"""A/B compare Claude vs Gemini Robotics-ER on the scene game's PHOTO spatial validator.

For each labeled frame it sends the EXACT prompt that
``_run_claude_validate_spatial`` uses in web_user_server.py, to both:
  - Claude    : claude-sonnet-4-6              (current production model)
  - Gemini ER : gemini-robotics-er-1.6-preview (candidate; already used for detection)

so the only variable is the model. Emits a side-by-side HTML report (frames
embedded) + a results JSON so you can adjudicate which model is right per frame.

Runs under .venv39 (which has BOTH `anthropic` and `google.genai`).
Keys are read from <demo>/src/.env (GEMINI_API_KEY, ANTHROPIC_API_KEY).

  .venv39/bin/python scripts/ab_spatial_validate.py --manifest scripts/ab_out/manifest.json

The prompt text below is copied verbatim from _run_claude_validate_spatial();
if that function changes, mirror it here so the A/B stays honest.
"""

import argparse
import base64
import html as _html
import json
import os
import sys
import time

# ── keys ─────────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_DEMO = os.path.dirname(_HERE)


def _load_env(path):
    """Minimal .env loader (no python-dotenv in .venv39). Handles `export K=V`,
    surrounding quotes, and stray whitespace. Does not override real env vars."""
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            if line.startswith("export "):
                line = line[len("export "):]
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip().strip('"').strip("'").strip()
            if k and k not in os.environ:
                os.environ[k] = v


_load_env(os.path.join(_DEMO, "src", ".env"))

CLAUDE_MODEL_DEFAULT = "claude-sonnet-4-6"
ER_MODEL_DEFAULT = "gemini-robotics-er-1.6-preview"

SYSTEM = ("You judge a children's spatial-direction game from one photo. "
          "Return JSON only.")

# Must stay identical to web_user_server._SPATIAL_RELATION_PHRASE
SPATIAL_RELATION_PHRASE = {
    "next_to": "next to",
    "above": "on top of",
    "under": "under",
    "behind": "behind",
    "in_front_of": "in front of",
    "in": "in",
    "out": "out of",
}


def build_prompt(obj_a, obj_b, relation, toy_list=None):
    """Byte-identical to _run_claude_validate_spatial()'s prompt."""
    rel_phrase = SPATIAL_RELATION_PHRASE.get(relation, relation)
    toy_clause = ""
    if toy_list:
        toys = ','.join(toy_list) if isinstance(toy_list, (list, tuple)) else str(toy_list)
        toy_clause = (
            f"The valid game objects are: {toys}. "
            "Only identify objects from this list.\n"
        )
    return (
        "You are judging a children's spatial-direction game.\n"
        f"{toy_clause}"
        f"The child was asked to arrange the scene so that the {obj_a} is "
        f"{rel_phrase} the {obj_b}.\n"
        "\n"
        "The image is taken from the front of the child (camera-facing view).\n"
        "Decide:\n"
        f"1. Is the {obj_a} present in the scene?\n"
        f"2. Is the {obj_b} present in the scene?\n"
        f"3. What is the actual spatial relation of the {obj_a} TO the "
        f"{obj_b}? Pick ONE:\n"
        "   - next_to       (side by side, roughly same height)\n"
        "   - above         (higher than / on top of)\n"
        "   - under         (lower than / underneath)\n"
        "   - behind        (further from the camera, partially hidden)\n"
        "   - in_front_of   (closer to camera, may partially block the other)\n"
        "   - in            (inside / contained by the other; partially hidden by its walls or rim)\n"
        "   - out           (outside / not contained by the other; fully visible and separate)\n"
        "   - other         (none of the above clearly applies)\n"
        f"4. Does that match the requested relation '{relation}'?\n"
        "\n"
        "Tips:\n"
        "- 'behind' means partially hidden by the reference object, or visibly\n"
        "  smaller/further along the camera's depth axis.\n"
        "- 'in_front_of' means the moving object partly occludes or sits\n"
        "  closer to the camera than the reference object.\n"
        "- 'in' means the moving object is contained by the reference object\n"
        "  (e.g. ball inside a cup or box) — typically partly hidden by the\n"
        "  rim/walls of the container.\n"
        "- 'out' means the moving object is clearly outside the reference\n"
        "  object, fully visible, with a visible gap between them.\n"
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


def _parse_json(raw):
    raw = (raw or "").strip()
    if raw.startswith("```"):
        raw = raw.strip("`").strip()
        if raw.startswith("json"):
            raw = raw[4:].strip()
    try:
        return json.loads(raw)
    except Exception:
        return {"_parse_error": True, "raw": raw}


# ── model callers ────────────────────────────────────────────────────────────
_anthropic = None
_genai = None
_genai_types = None


def run_claude(image_bytes, prompt, model):
    global _anthropic
    if _anthropic is None:
        import anthropic
        _anthropic = anthropic.Anthropic()
    b64 = base64.standard_b64encode(image_bytes).decode("ascii")
    resp = _anthropic.messages.create(
        model=model, max_tokens=1024, temperature=0.2, system=SYSTEM,
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64",
                                         "media_type": "image/jpeg", "data": b64}},
            {"type": "text", "text": prompt},
        ]}],
    )
    return "".join(getattr(b, "text", "") for b in resp.content
                   if getattr(b, "type", None) == "text").strip()


def run_er(image_bytes, prompt, model):
    global _genai, _genai_types
    if _genai is None:
        from google import genai
        from google.genai import types
        _genai = genai.Client(api_key=os.getenv("GEMINI_API_KEY")
                              or os.getenv("GOOGLE_API_KEY"))
        _genai_types = types
    types = _genai_types
    resp = _genai.models.generate_content(
        model=model,
        contents=[types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                  prompt],
        config=types.GenerateContentConfig(temperature=0.2,
                                           system_instruction=SYSTEM),
    )
    return (resp.text or "").strip()


def _call(fn, image_bytes, prompt, model):
    t0 = time.time()
    try:
        raw = fn(image_bytes, prompt, model)
        return {"ok": True, "ms": int((time.time() - t0) * 1000),
                "parsed": _parse_json(raw), "raw": raw}
    except Exception as e:
        return {"ok": False, "ms": int((time.time() - t0) * 1000),
                "parsed": {}, "raw": "", "error": f"{type(e).__name__}: {e}"}


# ── HTML report ──────────────────────────────────────────────────────────────
def _badge(txt, kind):
    colors = {"agree": "#1a7f37", "disagree": "#cf222e", "muted": "#57606a"}
    return (f'<span style="background:{colors.get(kind, "#57606a")};color:#fff;'
            f'padding:2px 8px;border-radius:10px;font-size:12px;">{_html.escape(str(txt))}</span>')


def _cell(res):
    if not res.get("ok"):
        return f'<div style="color:#cf222e">ERROR: {_html.escape(res.get("error", "?"))}</div>'
    p = res["parsed"]
    if p.get("_parse_error"):
        return f'<div style="color:#cf222e">unparseable:</div><pre>{_html.escape(p.get("raw", ""))}</pre>'
    rows = [
        ("actual_relation", p.get("actual_relation", "—")),
        ("correct", p.get("correct", "—")),
        ("obj_a_found", p.get("obj_a_found", "—")),
        ("obj_b_found", p.get("obj_b_found", "—")),
    ]
    body = "".join(
        f'<tr><td style="color:#57606a;padding-right:10px">{k}</td>'
        f'<td><b>{_html.escape(str(v))}</b></td></tr>' for k, v in rows)
    reason = _html.escape(str(p.get("reason", "")))
    return (f'<table style="font-size:14px">{body}</table>'
            f'<div style="margin-top:6px;color:#24292f"><i>{reason}</i></div>'
            f'<div style="margin-top:4px;color:#8c959f;font-size:11px">{res["ms"]} ms</div>')


def write_html(cases, out_path, claude_model, er_model):
    # summary
    n = len(cases)
    rel_agree = sum(1 for c in cases if c["agree_relation"])
    cor_agree = sum(1 for c in cases if c["agree_correct"])
    parts = [f"""<!doctype html><html><head><meta charset="utf-8">
<title>Spatial validator A/B — Claude vs Gemini ER</title>
<style>
 body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:24px;color:#24292f}}
 h1{{font-size:20px}} .sum{{background:#f6f8fa;border:1px solid #d0d7de;border-radius:8px;padding:12px 16px;margin-bottom:20px}}
 .case{{border:1px solid #d0d7de;border-radius:8px;padding:16px;margin-bottom:20px}}
 .grid{{display:grid;grid-template-columns:280px 1fr 1fr;gap:16px;align-items:start}}
 img{{max-width:280px;border-radius:6px;border:1px solid #d0d7de}}
 .col h3{{margin:0 0 8px;font-size:14px}} .ask{{font-size:14px;margin-bottom:10px}}
 pre{{white-space:pre-wrap;font-size:11px;background:#f6f8fa;padding:6px;border-radius:4px}}
</style></head><body>
<h1>Photo spatial validator — A/B</h1>
<div class="sum">
 <div><b>{n}</b> frames &nbsp;|&nbsp; Claude = <code>{_html.escape(claude_model)}</code> &nbsp;vs&nbsp; ER = <code>{_html.escape(er_model)}</code></div>
 <div style="margin-top:6px">Agree on <b>actual_relation</b>: {rel_agree}/{n} &nbsp;·&nbsp; Agree on <b>correct</b>: {cor_agree}/{n}</div>
 <div style="margin-top:6px;color:#57606a;font-size:13px">Disagreements are where a model choice would change the child's feedback — inspect those frames to decide the winner.</div>
</div>"""]
    for i, c in enumerate(cases, 1):
        try:
            with open(c["image"], "rb") as f:
                b64 = base64.standard_b64encode(f.read()).decode("ascii")
            img = f'<img src="data:image/jpeg;base64,{b64}">'
        except Exception as e:
            img = f'<div style="color:#cf222e">no image: {_html.escape(str(e))}</div>'
        rel_badge = _badge("relation: agree" if c["agree_relation"] else "relation: DIFFER",
                           "agree" if c["agree_relation"] else "disagree")
        cor_badge = _badge("correct: agree" if c["agree_correct"] else "correct: DIFFER",
                           "agree" if c["agree_correct"] else "disagree")
        ask = (f'Asked: <b>{_html.escape(c["obj_a"])}</b> '
               f'<b>{_html.escape(SPATIAL_RELATION_PHRASE.get(c["relation"], c["relation"]))}</b> '
               f'<b>{_html.escape(c["obj_b"])}</b> '
               f'&nbsp;·&nbsp; relation=<code>{_html.escape(c["relation"])}</code>')
        parts.append(f"""<div class="case">
 <div class="ask">#{i} &nbsp; {ask} &nbsp; {rel_badge} &nbsp; {cor_badge}
   <div style="color:#8c959f;font-size:11px;margin-top:2px">{_html.escape(os.path.basename(c["image"]))}</div></div>
 <div class="grid">
   <div>{img}</div>
   <div class="col"><h3>Claude ({_html.escape(claude_model)})</h3>{_cell(c["claude"])}</div>
   <div class="col"><h3>Gemini ER ({_html.escape(er_model)})</h3>{_cell(c["er"])}</div>
 </div></div>""")
    parts.append("</body></html>")
    with open(out_path, "w") as f:
        f.write("\n".join(parts))


def main():
    ap = argparse.ArgumentParser(description="A/B: Claude vs Gemini ER spatial validator")
    ap.add_argument("--manifest", required=True, help="JSON list of {image,obj_a,obj_b,relation,toy_list}")
    ap.add_argument("--out", default=os.path.join(_HERE, "ab_out", "ab_report.html"))
    ap.add_argument("--limit", type=int, default=0, help="cap number of cases (0 = all)")
    ap.add_argument("--claude-model", default=CLAUDE_MODEL_DEFAULT)
    ap.add_argument("--er-model", default=ER_MODEL_DEFAULT)
    args = ap.parse_args()

    with open(args.manifest) as f:
        manifest = json.load(f)
    if args.limit:
        manifest = manifest[:args.limit]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    cases = []
    for i, m in enumerate(manifest, 1):
        img = m["image"]
        if not os.path.exists(img):
            print(f"[{i}/{len(manifest)}] SKIP (no image): {img}")
            continue
        with open(img, "rb") as f:
            image_bytes = f.read()
        prompt = build_prompt(m["obj_a"], m["obj_b"], m["relation"], m.get("toy_list"))
        print(f"[{i}/{len(manifest)}] {os.path.basename(img)}  "
              f"{m['obj_a']} {m['relation']} {m['obj_b']} ...", flush=True)
        claude = _call(run_claude, image_bytes, prompt, args.claude_model)
        er = _call(run_er, image_bytes, prompt, args.er_model)
        cr = claude["parsed"].get("actual_relation")
        er_rel = er["parsed"].get("actual_relation")
        cc = claude["parsed"].get("correct")
        ec = er["parsed"].get("correct")
        case = {
            **m, "claude": claude, "er": er,
            "agree_relation": bool(cr is not None and cr == er_rel),
            "agree_correct": bool(cc is not None and cc == ec),
        }
        cases.append(case)
        print(f"      Claude: rel={cr} correct={cc} ({claude['ms']}ms)  |  "
              f"ER: rel={er_rel} correct={ec} ({er['ms']}ms)  "
              f"{'AGREE' if case['agree_relation'] else 'DIFFER'}", flush=True)

    if not cases:
        print("No cases evaluated.")
        return 1

    write_html(cases, args.out, args.claude_model, args.er_model)
    json_out = os.path.splitext(args.out)[0] + ".json"
    with open(json_out, "w") as f:
        json.dump(cases, f, indent=2, default=str)

    n = len(cases)
    print("\n=== SUMMARY ===")
    print(f"cases: {n}")
    print(f"agree on actual_relation: {sum(c['agree_relation'] for c in cases)}/{n}")
    print(f"agree on correct verdict: {sum(c['agree_correct'] for c in cases)}/{n}")
    print(f"HTML: {args.out}")
    print(f"JSON: {json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
