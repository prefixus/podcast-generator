"""Postprocessing audio merging and format conversion."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
import wave
from preprocess.tts_script_builder import PodcastScript


def merge_wav_files(audio_files: list[Path], output_wav: Path) -> bool:
    """Concatenate multiple WAV files into a single WAV file using Python's wave module."""
    if not audio_files:
        print("Brak plików audio do złączenia.")
        return False

    # Check if files exist
    valid_files = [f for f in audio_files if f.exists()]
    if not valid_files:
        print("Żaden z podanych plików audio nie istnieje.")
        return False

    print(f"Łączenie {len(valid_files)} plików WAV do: {output_wav}...")
    try:
        output_wav.parent.mkdir(parents=True, exist_ok=True)
        
        # Read the params from the first valid file
        with wave.open(str(valid_files[0]), 'rb') as first_file:
            params = first_file.getparams()

        # Write all frames to output WAV
        with wave.open(str(output_wav), 'wb') as merged_file:
            merged_file.setparams(params)
            for file_path in valid_files:
                with wave.open(str(file_path), 'rb') as f:
                    # Verify matching audio parameters
                    if f.getparams()[:3] != params[:3]:
                        print(f"Ostrzeżenie: Plik {file_path.name} ma inne parametry audio niż pierwszy plik. Może brzmieć nieprawidłowo.")
                    merged_file.writeframes(f.readframes(f.getnframes()))
        
        print("Pomyślnie złączono pliki WAV.")
        return True
    except Exception as e:
        print(f"Błąd podczas łączenia plików WAV: {e}")
        return False


def convert_wav_to_mp3_or_aac(input_wav: Path, output_file: Path) -> bool:
    """Convert WAV to MP3 or AAC using ffmpeg if installed."""
    suffix = output_file.suffix.lower()
    if suffix not in (".mp3", ".m4a", ".aac"):
        print(f"Nieobsługiwany format docelowy: {suffix}. Obsługiwane są: .mp3, .m4a, .aac")
        return False

    # Check if ffmpeg is available
    try:
        # Check if ffmpeg runs
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except (subprocess.SubprocessError, FileNotFoundError):
        print("Ostrzeżenie: Narzędzie 'ffmpeg' nie jest zainstalowane na tym systemie.")
        print(f"Nie można skonwertować do formatu {suffix.upper()}. Końcowy plik WAV pozostanie bez konwersji.")
        print("Aby zainstalować ffmpeg na macOS, użyj komendy: brew install ffmpeg")
        return False

    print(f"Konwertowanie {input_wav.name} do {output_file.name} przy użyciu ffmpeg...")
    try:
        # Build ffmpeg command based on format
        if suffix == ".mp3":
            cmd = ["ffmpeg", "-y", "-i", str(input_wav), "-codec:a", "libmp3lame", "-qscale:a", "2", str(output_file)]
        else:  # .m4a or .aac
            cmd = ["ffmpeg", "-y", "-i", str(input_wav), "-c:a", "aac", "-b:a", "192k", str(output_file)]

        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode == 0:
            print(f"Pomyślnie wygenerowano plik: {output_file}")
            # Optionally remove temporary WAV file
            if input_wav.exists() and input_wav != output_file:
                try:
                    os.remove(input_wav)
                except OSError:
                    pass
            return True
        else:
            print(f"Błąd ffmpeg (kod {result.returncode}):\n{result.stderr}")
            return False
    except Exception as e:
        print(f"Wystąpił błąd podczas konwersji: {e}")
        return False


def postprocess_podcast_audio(
    script: PodcastScript,
    audio_map: dict[str, str],
    output_dir: Path,
    output_format: str = "mp3"
) -> Path | None:
    """Merge all chunk audio files into a single file and convert to requested format (mp3/m4a/wav)."""
    # Order files according to the script chunks
    audio_files: list[Path] = []
    for chunk in script.chunks:
        file_path_str = audio_map.get(chunk.id)
        if file_path_str:
            audio_files.append(Path(file_path_str))

    if not audio_files:
        print("Brak wygenerowanych plików audio do przetworzenia.")
        return None

    stem = Path(script.title or "podcast").stem
    # Replace spaces and special chars in filename
    safe_stem = "".join([c if c.isalnum() or c in ("-", "_") else "_" for c in stem])
    
    temp_wav = output_dir / f"{safe_stem}_temp_merged.wav"
    
    # First merge WAV files
    success = merge_wav_files(audio_files, temp_wav)
    if not success:
        return None

    # Handle formats
    output_format = output_format.lower().strip(".")
    if output_format == "wav":
        # Rename temp to final WAV
        final_wav = output_dir / f"{safe_stem}.wav"
        if temp_wav.exists():
            if final_wav.exists():
                os.remove(final_wav)
            temp_wav.rename(final_wav)
            print(f"Pomyślnie zapisano plik: {final_wav}")
            return final_wav
        return None

    # Convert to MP3 or AAC (M4A)
    ext = f".{output_format}"
    if output_format in ("aac", "m4a", "mp3"):
        final_output = output_dir / f"{safe_stem}{ext}"
        if output_format == "aac":
            # ffmpeg aac works better inside m4a container for phone playback
            final_output = output_dir / f"{safe_stem}.m4a"
            
        conv_success = convert_wav_to_mp3_or_aac(temp_wav, final_output)
        if conv_success:
            return final_output
        else:
            # Fallback: keep the temp WAV as the final result
            fallback_wav = output_dir / f"{safe_stem}.wav"
            if temp_wav.exists():
                if fallback_wav.exists():
                    os.remove(fallback_wav)
                temp_wav.rename(fallback_wav)
                print(f"Zapisano plik WAV jako fallback pod adresem: {fallback_wav}")
                return fallback_wav

    return temp_wav
