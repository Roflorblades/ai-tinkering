from dotenv import load_dotenv

from ollama_compare import compare_responses, create_clients, run_chat, select_model, select_model_options


load_dotenv()

primary_client, secondary_client = create_clients()
model = select_model(primary_client)
options = select_model_options(model)


def test_prompt(prompt: str, model_name: str, options: dict[str, Any]) -> None:
    if secondary_client is None:
        print(f"\n=== Test auf einer Instanz ({model_name}) ===")
        response = run_chat(
            primary_client,
            model_name,
            [{"role": "user", "content": prompt}],
            options=options,
        )
        print(response)
        return

    print(f"\n=== Vergleichstest ({model_name}) ===")
    results = compare_responses(
        primary_client,
        secondary_client,
        model_name,
        [{"role": "user", "content": prompt}],
        labels=("Instanz 1", "Instanz 2"),
        options=options,
    )
    for label, response in results.items():
        print(f"\n[{label}]\n{response}")


if __name__ == "__main__":
    print(f"Ausgewähltes Modell: {model}")
    print("Drücke Enter, um den Standardprompt zu verwenden, oder gib deinen eigenen Testtext ein.")

    while True:
        prompt = input("\nTestprompt ('exit' zum Beenden): ").strip()
        if prompt.lower() == "exit":
            break
        if prompt == "":
            prompt = "Sag Hallo auf Deutsch in einem Satz."

        test_prompt(prompt, model, options)
