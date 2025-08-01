import openai

client = openai.OpenAI(
    api_key="",
    base_url="http://0.0.0.0:4000"
)

response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": "This is a test request, write a short poem"}
    ]
)