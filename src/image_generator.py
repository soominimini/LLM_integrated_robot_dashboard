import os
import uuid
from datetime import datetime
from typing import Optional
from io import BytesIO
from PIL import Image

import json
import subprocess
import sys

try:
    # Available only when google-genai is installed (typically Python 3.9+)
    from google import genai  # type: ignore
    from google.genai import types  # type: ignore
    _HAS_GOOGLE_GENAI = True
except Exception:
    genai = None  # type: ignore
    types = None  # type: ignore
    _HAS_GOOGLE_GENAI = False


style_instruction = (
    "You are an illustrator. "
    "Use a consistent, children's-book illustration style: "
    "soft round shapes, pastel color palette, thick outlines, minimal shading. "
    "The image should feel warm and friendly."
    "use the given image as a reference to keep the style the exact same, it is not reevant otherwise the image generation."
    "do not describe the image"
)


class ImageGenerator:
    """
    Generate images using Google GenAI (nanobanana) for story scenes
    """

    def __init__(self):
        try:
            # Default image model (can be overridden via env)
            self.model = os.getenv("GOOGLE_IMAGE_MODEL", "gemini-2.5-flash-image")

            # If running under Python 3.8 (project's default), google-genai likely isn't installed.
            # In that case we fall back to a Python 3.9 worker process (./.venv39) for image generation.
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.worker_python = os.getenv(
                "IMAGE_WORKER_PYTHON",
                os.path.join(project_root, ".venv39", "bin", "python"),
            )
            self.worker_script = os.getenv(
                "IMAGE_WORKER_SCRIPT",
                os.path.join(project_root, "src", "image_generator_worker.py"),
            )

            self.client = None
            if _HAS_GOOGLE_GENAI:
                api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
                if not api_key:
                    print("Missing GOOGLE_API_KEY (or GEMINI_API_KEY)", file=sys.stderr)
                    return 3
                client = genai.Client(api_key=api_key)
                if not api_key:
                    raise RuntimeError("Missing GOOGLE_API_KEY (or GEMINI_API_KEY) environment variable")
                self.client = genai.Client(api_key=api_key)

            self._available = True
            print("Google GenAI image generator initialized")
        except Exception as e:
            print(f"Warning: Could not initialize Google GenAI image generator: {e}")
            self.client = None
            self.model = None
            self._available = False

    def generate_image(self, prompt: str, output_dir: str = "generated_images", filename_prefix: str = None) -> Optional[str]:
        # IMPORTANT:
        # - In the Python 3.8 server env, google-genai usually isn't installed, so self.client is None.
        # - We still want to generate images via the Python 3.9 worker.
        if not self._available:
            return None

        try:
            os.makedirs(output_dir, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_id = str(uuid.uuid4())[:8]
            if filename_prefix:
                filename = f"{filename_prefix}_{timestamp}_{unique_id}.png"
            else:
                filename = f"story_scene_{timestamp}_{unique_id}.png"
            filepath = os.path.join(output_dir, filename)

            full_prompt = f"{style_instruction}\n\n{prompt}".strip()
            print(f"Generating image for prompt: {full_prompt[:100]}...")

            img_bytes = None

            # Path A: direct SDK call (when google-genai is installed in this interpreter)
            if self.client is not None and _HAS_GOOGLE_GENAI:
                try:
                    resp = self.client.models.generate_images(
                        model=self.model,
                        prompt=full_prompt,
                        config=types.GenerateImagesConfig(number_of_images=1),
                    )
                    if hasattr(resp, "generated_images") and resp.generated_images:
                        gi = resp.generated_images[0]
                        if hasattr(gi, "image") and hasattr(gi.image, "image_bytes"):
                            img_bytes = gi.image.image_bytes
                        elif hasattr(gi, "image_bytes"):
                            img_bytes = gi.image_bytes
                except Exception as e:
                    print(f"generate_images() failed: {e}")

            # Path B: worker process (Python 3.9 venv) to avoid ImportError in Python 3.8 server
            if not img_bytes:
                try:
                    # If output_dir already has images, use the first one as a reference guide
                    reference_image = None
                    if os.path.isdir(output_dir):
                        existing = sorted(
                            f for f in os.listdir(output_dir)
                            if f.lower().endswith(".png")
                        )
                        if existing:
                            reference_image = os.path.join(output_dir, existing[0])

                    payload = {
                        "model": self.model,
                        "prompt": full_prompt,
                        "output_path": filepath,
                        "reference_image": reference_image,
                    }
                    proc = subprocess.run(
                        [self.worker_python, self.worker_script],
                        input=json.dumps(payload),
                        text=True,
                        capture_output=True,
                        check=False,
                        env=os.environ.copy(),
                    )
                    if proc.returncode != 0:
                        print(f"image worker failed (code={proc.returncode}): {proc.stderr.strip()}")
                        return None
                except Exception as e:
                    print(f"image worker invocation failed: {e}")
                    return None

            if img_bytes:
                image = Image.open(BytesIO(img_bytes)).convert("RGB")
                image.save(filepath)
                print(f"Image saved to: {filepath}")
                return filepath

            # If worker ran, it already saved the image at filepath.
            if os.path.exists(filepath):
                print(f"Image saved to: {filepath}")
                return filepath

            print("No image returned by model/worker")
            return None

        except Exception as e:
            print(f"Error generating image: {e}")
            return None

    def generate_story_scene_image(self, sentence: str, story_context: str = "", output_dir: str = "generated_images", filename_prefix: str = None) -> Optional[str]:
        if not sentence.strip():
            return None

        clean_sentence = sentence.strip()
        if story_context:
            prompt = f"{clean_sentence} {story_context}"
        else:
            prompt = f"{clean_sentence}"

        prompt = prompt.replace("**", "").replace("*", "")
        return self.generate_image(prompt, output_dir, filename_prefix)

    def is_available(self) -> bool:
        return self._available