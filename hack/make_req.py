import requests
import json

url = "http://localhost:8000/v1/chat/completions"
headers = {
    "Content-Type": "application/json",
}

data = {
    "model": "arc-easy",
    "messages": [
        {"role": "user", "content": "[Question]:\n{Name me cold-blooded animals}\n\n[Response]:"}
    ],
    "stream": True
}

# Make the API call
response = requests.post(url, headers=headers, json=data, stream=True)
# Process and print the "content" in real-time
concatenated_content = ""
for chunk in response.iter_content(chunk_size=128):
    if chunk:
        try:
            json_data = json.loads(chunk)
            content = json_data['choices'][0]['delta']['content']
            concatenated_content += content
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error processing response: {e}")

# Print the concatenated content
print(concatenated_content)