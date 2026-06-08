"""Proofreader module using local LM Studio server."""

from __future__ import annotations

import requests

from preprocess.tts_script_builder import PodcastScript

DEFAULT_API_URL = "http://localhost:1234/v1/chat/completions"


def get_available_models(api_url: str) -> list[str]:
    """Fetch the list of available model IDs from LM Studio."""
    try:
        models_url = api_url.replace("/chat/completions", "/models")
        r = requests.get(models_url, timeout=5)
        r.raise_for_status()
        models = r.json().get("data", [])
        return [m["id"] for m in models]
    except Exception:
        return []


def proofread_text(text: str, model: str, api_url: str) -> str:
    """Send text to local LLM to fix Polish spelling, grammar, and inflection."""
    prompt = (
        "Jesteś profesjonalnym korektorem i redaktorem tekstu. Popraw wszelkie błędy ortograficzne, "
        "gramatyczne, interpunkcyjne oraz stylistyczne w poniższym tekście w języku polskim. "
        "Zwróć szczególną uwagę na poprawną odmianę (fleksję) wyrazów w zdaniach. "
        "Upewnij się, że tekst brzmi naturalnie i płynnie podczas czytania na głos. "
        "Bardzo ważne: Zwróć WYŁĄCZNIE poprawiony tekst. Nie dodawaj żadnego wstępu, wyjaśnień, "
        "komentarzy ani cudzysłowów.\n\n"
        f"Tekst do poprawy:\n{text}"
    )

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
    }

    try:
        response = requests.post(api_url, json=payload, timeout=60)
        response.raise_for_status()
        result = response.json()
        corrected = result["choices"][0]["message"]["content"].strip()
        # Clean up any potential markdown code blocks or quotes returned by the LLM
        if corrected.startswith("```"):
            lines = corrected.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines[-1].strip() == "```":
                lines = lines[:-1]
            corrected = "\n".join(lines).strip()
        return corrected if corrected else text
    except Exception as e:
        print(f"  [Ostrzeżenie] Nie udało się poprawić tekstu przez LLM ({e}). Używam oryginalnego.")
        return text


def proofread_script(
    script: PodcastScript,
    preferred_model: str = "google/gemma-4-26b-a4b-qat",
    api_url: str = DEFAULT_API_URL,
) -> None:
    """Proofread all chunks in the script in-place using LM Studio."""
    models = get_available_models(api_url)

    # Try to find the preferred model, or fallback to first available, or fallback to preferred_model string
    model = preferred_model
    if models:
        # Check if the preferred model is exact match or substring
        matched = [m for m in models if preferred_model in m]
        if matched:
            model = matched[0]
        else:
            model = models[0]
            print(f"Preferowany model '{preferred_model}' nie jest załadowany. Używam '{model}'.")
    else:
        print(f"Nie udało się połączyć z LM Studio pod adresem {api_url}. Pomijam automatyczną korektę LLM.")
        return

    print(f"Rozpoczynam korektę językową z użyciem modelu '{model}'...")
    total = len(script.chunks)
    for idx, chunk in enumerate(script.chunks):
        text = chunk.text.strip()
        if not text:
            continue

        print(f"[{idx + 1}/{total}] Korygowanie fragmentu: {chunk.id} ({len(text)} znaków)...")
        corrected = proofread_text(text, model, api_url)
        if corrected != text:
            chunk.text = corrected
            print("  -> Tekst poprawiony.")
    print("Korekta językowa zakończona.")
