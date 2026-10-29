
import os
from dotenv import load_dotenv
from openai import OpenAI, APIStatusError

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_BASE_URL")
model = os.getenv("LLM_MODEL")

if not api_key or not base_url or not model:
    raise SystemExit(
        "Missing OPENAI_API_KEY, OPENAI_BASE_URL, or LLM_MODEL in .env"
    )

client = OpenAI(
    api_key=api_key,
    base_url=base_url,
    timeout=30.0,
)

try:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": "به فارسی بگو: اتصال موفق است."
            }
        ],
        max_tokens=50,
    )
    print("API connection successful.")
    print(response.choices[0].message.content)

except APIStatusError as exc:
    print("API request failed.")
    print("HTTP status:", exc.status_code)
    print("Error type:", type(exc).__name__)
    print("Check the provider, API key, model access, and endpoint.")

except Exception as exc:
    print("Request failed:", type(exc).__name__)
    print("Check network connectivity and API configuration.")
