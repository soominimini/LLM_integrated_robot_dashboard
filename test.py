import os
from google import genai
from google.genai import types

with open('image_dataset/capture-20251018-214625.jpg', 'rb') as f:
    image_bytes = f.read()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("Missing GOOGLE_API_KEY (or GEMINI_API_KEY)")
client = genai.Client(api_key=api_key)
response = client.models.generate_content(
    model='gemini-2.5-flash',
    contents=[
        types.Part.from_bytes(
            data=image_bytes,
            mime_type='image/jpeg',
        ),
        'Caption this image.'
    ]
)

print(response.text)
