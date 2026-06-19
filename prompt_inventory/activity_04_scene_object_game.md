# Activity 4 — Scene / Object Detection Game

- **Entry points:** pages `/play_scene`, `/object_game_generate`, `/scene_game_config`; APIs `/api/scene/start`, `/api/scene_game/new_round|hint|answer`, `/api/scene_game/toys`.
- **Three distinct LLM steps**, each with its own model:
  1. **Question / riddle generation** — `gemini-2.5-flash` (ctx **1,048,576**) via `_gemini_generate`.
  2. **Spatial-relation validation** (still & video) — `gemini-2.5-flash` (ctx 1,048,576).
  3. **Held-object detection / pointing** — `gemini-robotics-er-1.6-preview` (ctx **131,072** effective).

---

## 4a. Question / riddle generation (`_scene_game_generate_question`) — **age-varied (3 tiers)**

`complexity_age = language_age or child_age`. System (when the LLM is used):
`"You generate game questions for children. Return JSON only."` (default temp 0.3).

### Tier A — `complexity_age ≤ 3`: **no LLM call**
A direct request is chosen locally from fixed templates (`{target}` = a random toy):
```
Show me the {target}!
Where is the {target}?
Can you find the {target}?
Let's find the {target}!
Show me {article} {target}!      (article = "a"/"an")
```

### Tier B — `complexity_age 4–6`: criteria prompt (verbatim)
`{goals_clause}` carries any therapy goals + persona/knowledge-base context. `{toy_list}` = comma-separated toy names.
```
You are generating a question for an object detection game for a {complexity_age}-year-old child.
Available physical toys: {toy_list}.
{goals_clause}Generate ONE inference-style request that lets the child figure
out the target object from its observable properties.

HARD RULE — the QUESTION text must NOT name the target object.
It must NEVER contain any of these noun names from the toy list:
  {toy_list}
Refer to the target only as "it", "something", "one", or by
a generic placeholder like "a fruit" or "a vehicle". The child
must INFER which object you mean from the description.

CRITICAL — the criteria must describe ONE simple, concrete object:
- A single noun (the object type) with at most ONE adjective
  describing color, size, or category.
- Good criteria: "banana", "red car", "green dinosaur",
  "tomato", "yellow fruit", "round ball".
- BAD criteria (do NOT produce these):
    * "red car moving block" (compound / multi-object)
    * "big round shiny red fruit on a tree" (too many properties)
    * "toy that you can stack" (function-based, vague)
- Do not chain multiple objects or stack three+ adjectives.

The criteria MUST match at least one toy from the list above.
Use simple, clear language appropriate for ages 4-6.
Good examples (target NOT named in the question):
- Question: "I want a red fruit!" (criteria: red fruit)
- Question: "Can you find something yellow?" (criteria: yellow)
- Question: "Show me something green that goes ROAR!"
  (criteria: green dinosaur)
BAD example (do NOT do this — names the target):
- Question: "Show me the red apple!" — "apple" is the target name.

Return ONLY a JSON object:
{"question": "<the sentence — must NOT contain any toy name>", "criteria": "<short criteria phrase: one noun + at most one adjective>"}
```

### Tier C — `complexity_age 7+`: riddle prompt (verbatim)
```
You are generating a question for an object detection game for a {complexity_age}-year-old child.
Available physical toys: {toy_list}.
{goals_clause}Generate ONE riddle that requires the child to reason about
properties (color, shape, size, function, where it is found) to
figure out the answer. Do NOT use a conversational tone.

HARD RULE — the QUESTION (riddle) text must NEVER name the target
object. It must NOT contain any of these noun names from the toy list:
  {toy_list}
Use only pronouns ("it", "I") and property descriptions. The
child must INFER the target from the clues.

CRITICAL — the underlying TARGET must be ONE simple, concrete object:
- A single noun (the object type), optionally with ONE color or
  size adjective. Examples of acceptable targets: "banana",
  "red car", "green dinosaur", "tomato".
- The riddle text may use 2-3 properties as clues, but the
  "criteria" field MUST be the simple target description (one
  noun + at most one adjective).
- Do NOT chain multiple objects or invent compound targets like
  "red car moving block" or "shiny round tree fruit".

The target MUST match at least one toy from the list.
Good example: "I am round and red, and I grow on a tree. What am I?"
  (criteria: "red apple") — note the riddle does NOT say "apple".
BAD example (do NOT do this — names the target):
- "Find the red apple that grows on a tree."

Return ONLY a JSON object:
{"question": "<the riddle — must NOT contain any toy name>", "criteria": "<simple target: one noun + at most one adjective>"}
```

**Retry strengthening** (appended to the same prompt if the first output leaks a toy name):
```
PREVIOUS ATTEMPT FAILED — your last question contained the
forbidden word "{leaked}". Rewrite the question so it contains
NONE of these words: {toy_list}. Refer to the target ...
```

> There is also a “direction” mode (spatial prepositions, e.g. “put the banana
> under the blue block”) generated server-side without an LLM; its arrangement is
> then checked by step 4b.

---

## 4b. Spatial-relation validation — not age-varied

`{rel_phrase}` is the canonical relation phrase (next_to → "next to", above → "on top of", under → "under", behind → "behind", in_front_of → "in front of", in → "in", out → "out of"). `{obj_a}` moving object, `{obj_b}` reference. Optional `{toy_clause}` constrains identification to a toy list. `temperature=0.2`. The full prompt and raw response are returned to the UI so a therapist can see how the verdict was reached.

### Still-frame worker (`scripts/gemini_validate_spatial.py`) — `gemini-2.5-flash`
```
You are judging a children's spatial-direction game.
{toy_clause}The child was asked to arrange the scene so that the {obj_a} is {rel_phrase} the {obj_b}.

The image is taken from the front of the child (camera-facing view).
Decide:
1. Is the {obj_a} present in the scene?
2. Is the {obj_b} present in the scene?
3. What is the actual spatial relation of the {obj_a} TO the {obj_b}? Pick ONE:
   - next_to       (side by side, roughly same height)
   - above         (higher than / on top of)
   - under         (lower than / underneath)
   - behind        (further from the camera, partially hidden)
   - in_front_of   (closer to camera, may partially block the other)
   - in            (inside / contained by the other; partially hidden by its walls or rim)
   - out           (outside / not contained by the other; fully visible and separate)
   - other         (none of the above clearly applies)
4. Does that match the requested relation '{relation}'?

Tips:
- 'behind' means partially hidden by the reference object, or visibly
  smaller/further along the camera's depth axis.
- 'in_front_of' means the moving object partly occludes or sits
  closer to the camera than the reference object.
- 'in' means the moving object is contained by the reference object
  (e.g. ball inside a cup or box) — typically partly hidden by the
  rim/walls of the container.
- 'out' means the moving object is clearly outside the reference
  object, fully visible, with a visible gap between them.
If you cannot tell confidently, return 'other'.

Return ONLY a JSON object with no markdown fences:
{
  "obj_a_found": true|false,
  "obj_b_found": true|false,
  "actual_relation": "next_to|above|under|behind|in_front_of|in|out|other",
  "correct": true|false,
  "reason": "<short, child-friendly explanation>"
}
If either object is missing, set correct=false.
```

### Video-clip worker (`scripts/gemini_validate_spatial_video.py`) — `gemini-2.5-flash`
Used for depth relations (behind / in_front_of) where a 3-second clip gives parallax. Uploads the clip via the Gemini Files API.
```
You are judging a children's spatial-direction game.
{toy_clause}The child was asked to arrange the scene so that the {obj_a} is {rel_phrase} the {obj_b}.

You are watching a short video (camera-facing view) of the child's
setup. Use motion ACROSS frames to infer depth and containment more
reliably than a single still frame would allow.

1. Is the {obj_a} present in the scene?
2. Is the {obj_b} present in the scene?
3. What is the actual spatial relation of the {obj_a} TO the {obj_b}? Pick ONE:
   - next_to       (side by side, similar height)
   - above         (higher than / on top of)
   - under         (lower than / underneath)
   - behind        (further from camera; partially hidden by the other)
   - in_front_of   (closer to camera; may partially block the other)
   - in            (inside / contained by the other; partially hidden by its walls or rim)
   - out           (outside / not contained by the other; fully visible and separate)
   - other         (can't tell confidently)
4. Does that match the requested relation '{relation}'?

Cues to use across frames:
- Parallax: an object CLOSER to the camera shifts MORE in the image
  than a FARTHER object for the same scene motion. (depth)
- Occlusion: an object that consistently hides part of another is
  in front of it. (depth)
- Containment ('in'): if the moving object stays inside the rim of
  the reference object across the clip — partly hidden by the
  container's walls — it is 'in'. If the child visibly LIFTS the
  moving object OUT of the container during the clip, classify by
  the END state of the video.
- Separation ('out'): if there is a clear visible gap between the
  two objects throughout the clip (or by the end of it), it is 'out'.
If you cannot tell confidently from any of these cues, return 'other'.

Return ONLY a JSON object with no markdown fences:
{
  "obj_a_found": true|false,
  "obj_b_found": true|false,
  "actual_relation": "next_to|above|under|behind|in_front_of|in|out|other",
  "correct": true|false,
  "reason": "<short, child-friendly explanation>"
}
If either object is missing, set correct=false.
```

---

## 4c. Held-object detection / pointing (`scripts/gemini_analyze_image.py`) — not age-varied

- **Model:** `gemini-robotics-er-1.6-preview` (hardcoded). **Context window: 131,072 effective (docs list 1,048,576).** `temperature=0.5`, `thinking_budget=0`.
- Used to locate the object a child is holding and point the robot's head at it / verify the child picked the right toy. The returned point is `[y, x]` normalized to 0–1000.

### Prompt (verbatim)
```
Point to no more than 1 item a person is holding in the image.
Return the object's identifying name, its dominant color, and its shape.
The answer should follow the json format:
[{"point": <point>, "label": <label>, "color": <color>, "shape": <shape>}, ...].
The points are in [y, x] format normalized to 0-1000.
```
