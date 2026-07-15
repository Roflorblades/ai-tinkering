from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

try:
    from ollama import Client
except ModuleNotFoundError:  # pragma: no cover - fallback for test environments without the package
    Client = Any


def make_host_url(ip: str, port: str) -> str:
    return f"http://{ip}:{port}"


def create_client(host: str) -> Client:
    return Client(host=host)


def default_host_port() -> Tuple[str, str]:
    return (
        os.getenv("OLLAMA_IP", "localhost"),
        os.getenv("OLLAMA_PORT", "11434"),
    )


def prompt_host(label: str, default_ip: str, default_port: str) -> Tuple[str, str]:
    ip = input(f"{label} IP-Adresse (Enter = {default_ip}): ").strip() or default_ip
    port = input(f"{label} Port (Enter = {default_port}): ").strip() or default_port
    return ip, port


def create_primary_client() -> Client:
    ip, port = default_host_port()
    return create_client(make_host_url(ip, port))


def prompt_for_second_client(default_ip: Optional[str] = None, default_port: Optional[str] = None) -> Optional[Client]:
    if default_ip is None or default_port is None:
        default_ip, default_port = default_host_port()

    answer = input("Möchtest du eine zweite OLLAMA-Instanz zum Vergleichen hinzufügen? [j/N]: ").strip().lower()
    if answer not in ("j", "y", "ja", "yes"):
        return None

    ip, port = prompt_host("Zweite Instanz", default_ip, default_port)
    return create_client(make_host_url(ip, port))


def create_clients() -> Tuple[Client, Optional[Client]]:
    primary_client = create_primary_client()
    secondary_client = prompt_for_second_client()
    return primary_client, secondary_client


def list_models(client: Client) -> List:
    return list(client.list().models)


def select_model(client: Client) -> str:
    models = list_models(client)
    print("\n=== Verfügbare Modelle ===")
    for index, model in enumerate(models, start=1):
        print(f"  {index}. {model.model}  –  {round(model.size / 1e9, 1)} GB")

    while True:
        choice = input(f"\nModell wählen (1-{len(models)}): ").strip()
        if choice.isdigit():
            index = int(choice) - 1
            if 0 <= index < len(models):
                return models[index].model

        print(f"Ungültige Auswahl. Bitte eine Zahl zwischen 1 und {len(models)} eingeben.")


def select_model_options(model: str) -> dict[str, Any]:
    options: dict[str, Any] = {}
    print(f"\nOptionen für Modell {model}:")
    if model.startswith("qwen"):
        answer = input("Soll das Modell weniger 'thinking' verwenden? [j/N]: ").strip().lower()
        if answer in ("j", "y", "ja", "yes"):
            level = input("Denkintensität wählen [low/medium/high] (Enter = low): ").strip().lower() or "low"
            if level not in ("low", "medium", "high"):
                print("Ungültige Wahl. Verwende 'low'.")
                level = "low"
            options["think"] = level
    return options


def run_chat(
    client: Client,
    model: str,
    messages: Sequence[dict],
    keep_alive: Optional[str] = None,
    options: Optional[dict[str, Any]] = None,
) -> str:
    response = ""
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": True,
    }
    if keep_alive is not None:
        kwargs["keep_alive"] = keep_alive
    if options:
        kwargs.update(options)

    for chunk in client.chat(**kwargs):
        response += chunk.message.content
    return response


def safe_run_chat(
    client: Client,
    model: str,
    messages: Sequence[dict],
    keep_alive: Optional[str] = None,
    options: Optional[dict[str, Any]] = None,
    return_timing: bool = False,
) -> Union[str, Tuple[str, float]]:
    started = time.perf_counter()
    try:
        response = run_chat(client, model, messages, keep_alive=keep_alive, options=options)
    except Exception as exc:
        response = f"[Fehler] {exc}"

    elapsed_seconds = time.perf_counter() - started
    if return_timing:
        return response, round(elapsed_seconds, 3)
    return response


def compare_responses(
    primary_client: Client,
    secondary_client: Client,
    model: str,
    messages: Sequence[dict],
    labels: Tuple[str, str] = ("Instanz 1", "Instanz 2"),
    keep_alive: Optional[str] = None,
    options: Optional[dict[str, Any]] = None,
    return_timing: bool = False,
) -> Dict[str, Any]:
    results: Dict[str, Any] = {}
    for label, client in ((labels[0], primary_client), (labels[1], secondary_client)):
        response, elapsed_seconds = safe_run_chat(
            client,
            model,
            messages,
            keep_alive=keep_alive,
            options=options,
            return_timing=True,
        )
        if return_timing:
            results[label] = {"response": response, "elapsed_seconds": elapsed_seconds}
        else:
            results[label] = response
    return results
