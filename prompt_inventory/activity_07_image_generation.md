# Activity 7 — Story sentence illustration (image generation)

- **Where used:** illustrations for the Read Story experience (`/api/get_sentence_image`, story scene images). Implemented in `src/image_generator.py` (with a Python-3.9 worker `src/image_generator_worker.py` for the 3.8 server).
- **Model:** `gemini-2.5-flash-image` (a.k.a. “Nano Banana”), from `GOOGLE_IMAGE_MODEL` env default. **Context window: 32,768 input / 8,192 output** — the smallest of any model in the project. Generates 1 image per call; an existing image in the output dir is passed as a style reference.
- **Age-varied?** No.

## Fixed style instruction (`style_instruction`, verbatim)
Prepended to every image prompt:
```
You are an illustrator. Use a consistent, children's-book illustration style: soft round shapes, pastel color palette, thick outlines, minimal shading. The image should feel warm and friendly.use the given image as a reference to keep the style the exact same, it is not reevant otherwise the image generation.do not describe the image
```
*(Reproduced exactly, including the original typos/spacing in the source string.)*

## How the per-image prompt is assembled
For a story sentence, `generate_story_scene_image(sentence, story_context)` builds:
```
prompt = "<sentence> <story_context>"        # story_context optional
prompt = prompt.replace("**", "").replace("*", "")   # strip markdown emphasis
```
Then `generate_image` sends:
```
full_prompt = "{style_instruction}\n\n{prompt}"
```
So the final prompt = the fixed style instruction + a blank line + the (tag-stripped)
story sentence and optional scene context. The scene context typically comes from
the story scene-identification pass (see
[`activity_02_story_reading.md`](activity_02_story_reading.md) §2e).
