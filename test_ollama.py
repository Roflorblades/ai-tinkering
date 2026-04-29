from ollama import Client
import os
from dotenv import load_dotenv

load_dotenv()

OLLAMA_IP = os.getenv("OLLAMA_IP")
OLLAMA_PORT = os.getenv("OLLAMA_PORT")

OLLAMA_HOST = f"http://{OLLAMA_IP}:{OLLAMA_PORT}"
client = Client(host=OLLAMA_HOST)

# --- Verfügbare Modelle anzeigen ---
print("=== Installierte Modelle ===")
for m in client.list().models:           # .models statt ["models"]
    print(f"  {m.model}  –  {round(m.size / 1e9, 1)} GB")   # m.model statt m['name']

# --- Einfacher Test mit Streaming ---
print("\n=== Test phi4-mini ===")
for chunk in client.chat(
    model="phi4-mini",
    messages=[{"role": "user", "content": "Sag Hallo auf Deutsch in einem Satz."}],
    stream=True,
):
    print(chunk.message.content, end="", flush=True)   # auch hier: .content statt ['content']

print("\n\n=== Test gemma3:4b ===")
for chunk in client.chat(
    model="gemma3:4b",
    messages=[{"role": "user", "content": "Sag Hallo auf Deutsch in einem Satz."}],
    stream=True,
):
    print(chunk.message.content, end="", flush=True)

print()