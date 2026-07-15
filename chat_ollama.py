from ollama import Client
from dotenv import load_dotenv

from ollama_compare import (
    compare_responses,
    create_clients,
    run_chat,
    select_model,
)


load_dotenv()

primary_client, secondary_client = create_clients()
model = select_model(primary_client)
options = select_model_options(model)


def chat_single(client: Client, model: str, options: dict[str, Any]) -> None:
    history: list[dict] = []
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
        response_text = run_chat(client, model, history, keep_alive="5m", options=options)

        print(f"\n[{model}]: {response_text}\n")
        history.append({"role": "assistant", "content": response_text})


def chat_compare(primary_client: Client, secondary_client: Client, model: str) -> None:
    history_a: list[dict] = []
    history_b: list[dict] = []

    print(
        f"\n💬 Vergleichschat mit zwei Instanzen [{model}] — 'exit' zum Beenden, 'reset' für neues Gespräch\n"
    )

    while True:
        user_input = input("Du: ").strip()

        if user_input.lower() == "exit":
            print("Tschüss!")
            break
        if user_input.lower() == "reset":
            history_a = []
            history_b = []
            print("--- Gesprächsverlauf gelöscht ---\n")
            continue
        if not user_input:
            continue

        history_a.append({"role": "user", "content": user_input})
        history_b.append({"role": "user", "content": user_input})

        results = compare_responses(
            primary_client,
            secondary_client,
            model,
            history_a,
            labels=("Instanz 1", "Instanz 2"),
            keep_alive="5m",
            options=options,
        )

        for label, response in results.items():
            print(f"\n[{label}] {response}\n")

        history_a.append({"role": "assistant", "content": results["Instanz 1"]})
        history_b.append({"role": "assistant", "content": results["Instanz 2"]})


if __name__ == "__main__":
    if secondary_client is not None:
        chat_compare(primary_client, secondary_client, model)
    else:
        chat_single(primary_client, model)
