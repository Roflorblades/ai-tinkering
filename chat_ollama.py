from ollama import Client
import os
from dotenv import load_dotenv

load_dotenv()

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://192.168.0.203:11434")
client = Client(host=OLLAMA_HOST)

def select_model():
    print("\n=== Verfügbare Modelle ===")
    models = client.list().models
    for i, m in enumerate(models):
        print(f"  [{i}] {m.model}  –  {round(m.size / 1e9, 1)} GB")
    
    choice = input("\nModell wählen (Nummer): ")
    return models[int(choice)].model

def chat(model: str):
    history = []
    print(f"\n💬 Chat mit [{model}] — 'exit' zum Beenden, 'reset' für neues Gespräch\n")
    
    while True:
        user_input = input("Du: ").strip()
        
        if user_input.lower() == "exit":
            print("Tschüss!")
            break
        if user_input.lower() == "reset":
            history = []
            print("--- Gesprächsverlauf gelöscht ---\n")
            continue
        if not user_input:
            continue

        history.append({"role": "user", "content": user_input})

        print(f"\n[{model}]: ", end="", flush=True)
        response_text = ""

        for chunk in client.chat(
            model=model,
            messages=history,
            stream=True,
            keep_alive="30m",
        ):
            token = chunk.message.content
            print(token, end="", flush=True)
            response_text += token

        print("\n")
        history.append({"role": "assistant", "content": response_text})

if __name__ == "__main__":
    model = select_model()
    chat(model)