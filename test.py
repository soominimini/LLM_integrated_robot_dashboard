from google import genai
from google.genai import types

with open('image_dataset/capture-20251018-214625.jpg', 'rb') as f:
    image_bytes = f.read()

client = genai.Client(api_key="***REMOVED***")
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
