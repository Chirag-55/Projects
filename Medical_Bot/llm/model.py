# llm/model.py

import requests

def ask_llm(prompt):
    headers = {
        "Authorization": "Bearer YOUR_API_KEY",  # Replace with your actual key
        "Content-Type": "application/json"
    }

    messages = [
        {"role": "system", "content": "You are a helpful and knowledgeable medical assistant."},
        {"role": "user", "content": prompt}
    ]

    payload = {
        "model": "llama3-8b-8192",
        "messages": messages,
        "temperature": 0.7
    }

    response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()
