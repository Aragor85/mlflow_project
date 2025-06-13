import requests

url = "http://127.0.0.1:8000/predict"

texts = [
    "I love Air Paradis!",
    "Air Paradis was the worst experience of my life.",
    "It was okay, nothing special.",
    "Amazing service, I’ll fly again!",
    "Terrible flight, never again!",
]

for text in texts:
    response = requests.post(url, json={"text": text})
    print(f"Texte : {text}")
    print(f"Réponse : {response.json()}")
    print("-" * 40)
