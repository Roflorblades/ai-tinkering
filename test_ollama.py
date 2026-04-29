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
for m in client.list().models:          
    print(f"  {m.model}  –  {round(m.size / 1e9, 1)} GB")   


def test_model(model_name: str):
    print(f"\n=== Test {model_name} ===")
    for chunk in client.chat(
        model=model_name,
        messages=[{"role": "user", "content": "Sag Hallo auf Deutsch in einem Satz."}],
        stream=True,
    ):
        print(chunk.message.content, end="", flush=True)
    print()


test_model("phi4-mini")
test_model("gemma3:4b")
test_model("gemma3:12b")