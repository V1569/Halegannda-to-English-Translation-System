import requests

def verify_deepseek_key(api_key: str):
    url = "https://api.deepseek.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    data = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 5
    }

    response = requests.post(url, json=data, headers=headers)

    if response.status_code == 200:
        print("✅ API key is valid!")
    elif response.status_code == 401:
        print("❌ Invalid API key (401 Unauthorized).")
    else:
        print(f"⚠️ Unexpected response: {response.status_code}")
        print(response.text)


# Put your API key here on your computer ONLY
api_key = "sk-49e1c4e2a01542d69d626d05a0c76281"
verify_deepseek_key(api_key)
