from base64 import b64encode

from dotenv import load_dotenv

from langchain.chat_models import init_chat_model 

load_dotenv()

model = init_chat_model('google_genai:gemma-4-31b-it')

message = {
    'role': 'user',
    'content': [
        {'type': 'text', 'text': 'Describe the contents of this image.'},
        {'type': 'image', 'url': 'https://neuralnine.com/wp-content/uploads/2025/04/neuralnine_logo_transparent-1536x740.png'}
    ]
}


"""
image_base64 = base64.b64encode(response_img.content).decode("utf-8")

message = {
    "role": "user",
    "content": [
        {"type": "text", "text": "Describe the contents of this image."},
        {
            "type": "image",
            "source_type": "base64",
            "data": image_base64,
            "mime_type": "image/jpeg",
        },
    ],
}
"""

response = model.invoke([message])

print(response.content)