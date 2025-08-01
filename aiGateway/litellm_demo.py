from litellm import completion
import os

## set ENV variables
os.environ["OPENAI_API_KEY"] = ""

response = completion(
  model="openai/gpt-4o",
  messages=[{ "content": "Hello, how are you?","role": "user"}]
)