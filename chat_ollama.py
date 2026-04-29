from ollama import Client
import os
from dotenv import load_dotenv


load_dotenv()


OLLAMA_IP = os.getenv("OLLAMA_IP", "localhost")
OLLAMA_PORT = os.getenv("OLLAMA_PORT", "11434")


OLLAMA_HOST = f"http://{OLLAMA_IP}:{OLLAMA_PORT}"
client = Client(host=OLLAMA_HOST)

PREFERRED_MODEL = "gemma3:12b"


def select_model():
    print("\n=== Verfügbare Modelle ===")
    models = client.list().models
    default_idx = 0

    for i, m in enumerate(models):
        marker = ""
        if m.model == PREFERRED_MODEL:
            marker = "  ★ Standard"
            default_idx = i
        print(f"  [{i}] {m.model}  –  {round(m.size / 1e9, 1)} GB{marker}")

    while True:
        choice = input(f"\nModell wählen (Nummer, Enter = [{default_idx}]): ").strip()
        if choice == "":
            return models[default_idx].model
        if choice.isdigit() and 0 <= int(choice) < len(models):
            return models[int(choice)].model
        print(f"  Ungültige Eingabe. Bitte eine Zahl zwischen 0 und {len(models) - 1} eingeben.")


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
            keep_alive="5m",
        ):
            token = chunk.message.content
            print(token, end="", flush=True)
            response_text += token

        print("\n")
        history.append({"role": "assistant", "content": response_text})


if __name__ == "__main__":
    model = select_model()
    chat(model)