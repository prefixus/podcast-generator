#!/usr/bin/env python3
"""Script to proofread and correct Polish grammar/inflection in generated podcast scripts using local LM Studio."""

import json
import argparse
from pathlib import Path
import requests

DEFAULT_API_URL = "http://localhost:1234/v1/chat/completions"

def get_active_model() -> str:
    """Fetch the first available model from LM Studio."""
    try:
        r = requests.get("http://localhost:1234/v1/models", timeout=5)
        r.raise_for_status()
        models = r.json().get("data", [])
        if models:
            return models[0]["id"]
    except Exception:
        pass
    return "default"

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
        "messages": [
            {"role": "user", "content": prompt}
        ],
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

def main() -> None:
    parser = argparse.ArgumentParser(description="Popraw błędy językowe w skrypcie podcastu przy użyciu lokalnego modelu w LM Studio.")
    parser.add_argument("--input", default="output/Seksuologia_opracowane_tezy_script.json", help="Ścieżka do wejściowego pliku JSON")
    parser.add_argument("--output", default="output/Seksuologia_opracowane_tezy_script_proofread.json", help="Ścieżka do wyjściowego pliku JSON")
    parser.add_argument("--url", default=DEFAULT_API_URL, help="Adres URL API LM Studio")
    
    args = parser.parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if not input_path.exists():
        print(f"Błąd: Plik wejściowy {input_path} nie istnieje.")
        sys.exit(1)
        
    print(f"Wczytywanie skryptu z {input_path}...")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    model = get_active_model()
    print(f"Używanie modelu LM Studio: {model}")
    
    chunks = data.get("chunks", [])
    total = len(chunks)
    
    # We only proofread chunks that are read (intro, transitions, titles, body chunks)
    # and have actual content.
    print(f"Rozpoczęcie korekty {total} fragmentów...")
    for idx, chunk in enumerate(chunks):
        text = chunk.get("text", "").strip()
        if not text:
            continue
            
        print(f"[{idx+1}/{total}] Korekta fragmentu: {chunk['id']} ({len(text)} znaków)...")
        corrected = proofread_text(text, model, args.url)
        if corrected != text:
            print(f"  -> Zmieniono tekst.")
            chunk["text"] = corrected
            
    # Save the proofread JSON
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        
    print(f"\nGotowe! Poprawiony skrypt został zapisany do {output_path}")

if __name__ == "__main__":
    import sys
    main()
