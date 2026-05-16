"""
Qur'an Reciter Recognition — Proof of Concept
==============================================
Week 1 experiment. Goal: identify reciter from a short clip.

SETUP (run once in terminal):
    pip install resemblyzer numpy scipy soundfile librosa

FOLDER STRUCTURE expected:
    dataset/
        alafasy/       <- put .mp3 or .wav clips here
        dosari/        <- put .mp3 or .wav clips here
        muaiqly/       <- put .mp3 or .wav clips here
    test_clip.wav      <- the clip you want to identify

HOW TO GET AUDIO:
    everyayah.com  — download per-ayah MP3s by reciter
    download.quranicaudio.com — full recitation zips

USAGE:
    1. python quran_recognition_experiment.py --build
       (builds voice profiles from your dataset folder)

    2. python quran_recognition_experiment.py --identify test_clip.wav
       (identifies which reciter the clip belongs to)

    3. python quran_recognition_experiment.py --stress
       (runs full stress test across all test clips)
"""

import os
import sys
import json
import argparse
import warnings
import csv
from datetime import datetime
import numpy as np

warnings.filterwarnings("ignore")


# ─── CONFIG ──────────────────────────────────────────────────────────────────

DATASET_DIR   = "dataset"       # folder with subfolders per reciter
PROFILES_FILE = "profiles.json" # saved voice profiles
RESULTS_FILE  = "results.csv"   # stress test log
SAMPLE_RATE   = 16000           # 16kHz is standard for voice work


# ─── AUDIO LOADING ───────────────────────────────────────────────────────────

def load_audio(path, sr=SAMPLE_RATE):
    """Load any audio file to mono numpy array at target sample rate."""
    try:
        import librosa
        audio, _ = librosa.load(path, sr=sr, mono=True)
        return audio
    except Exception as e:
        print(f"  [error] Could not load {path}: {e}")
        return None


# ─── FEATURE EXTRACTION ──────────────────────────────────────────────────────

def extract_embedding(audio, sr=SAMPLE_RATE):
    """
    Extract a d-vector (voice embedding) using Resemblyzer.
    This captures the unique vocal characteristics of a speaker.
    Works better than audio fingerprinting for distinguishing reciters
    who recite the same words with similar melody.
    """
    try:
        from resemblyzer import VoiceEncoder, preprocess_wav
        encoder = VoiceEncoder()
        wav = preprocess_wav(audio, source_sr=sr)
        embedding = encoder.embed_utterance(wav)
        return embedding
    except ImportError:
        print("[error] resemblyzer not installed. Run: pip install resemblyzer")
        sys.exit(1)
    except Exception as e:
        print(f"  [error] Embedding failed: {e}")
        return None


def extract_mfcc_embedding(audio, sr=SAMPLE_RATE):
    """
    Fallback: MFCC-based embedding if Resemblyzer fails.
    Less accurate but requires no model download.
    """
    import librosa
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    return np.mean(mfcc, axis=1)  # average over time -> 40-dim vector


# ─── PROFILE BUILDING ────────────────────────────────────────────────────────

def build_profiles():
    """
    Scan dataset/ folder, extract embeddings for each reciter,
    average them into a single voice profile per reciter.
    Save to profiles.json.
    """
    if not os.path.exists(DATASET_DIR):
        print(f"[error] No '{DATASET_DIR}' folder found.")
        print("Create it with subfolders: dataset/alafasy/, dataset/dosari/, etc.")
        return

    reciters = [d for d in os.listdir(DATASET_DIR)
                if os.path.isdir(os.path.join(DATASET_DIR, d))]

    if not reciters:
        print("[error] No reciter subfolders found inside dataset/")
        return

    print(f"\nFound {len(reciters)} reciters: {', '.join(reciters)}")
    profiles = {}

    for reciter in reciters:
        folder = os.path.join(DATASET_DIR, reciter)
        files  = [f for f in os.listdir(folder)
                  if f.lower().endswith(('.mp3', '.wav', '.m4a', '.ogg'))]

        if not files:
            print(f"  [{reciter}] No audio files found, skipping.")
            continue

        print(f"\n  [{reciter}] Processing {len(files)} clip(s)...")
        embeddings = []

        for fname in files:
            path  = os.path.join(folder, fname)
            audio = load_audio(path)
            if audio is None:
                continue

            print(f"    -> {fname} ({len(audio)/SAMPLE_RATE:.1f}s)")
            emb = extract_embedding(audio)
            if emb is not None:
                embeddings.append(emb)

        if embeddings:
            # Average all clips into one profile vector
            profile = np.mean(embeddings, axis=0)
            profiles[reciter] = profile.tolist()
            print(f"  [{reciter}] Profile built from {len(embeddings)} clip(s).")
        else:
            print(f"  [{reciter}] No valid embeddings. Skipping.")

    if profiles:
        with open(PROFILES_FILE, "w") as f:
            json.dump(profiles, f)
        print(f"\nProfiles saved to {PROFILES_FILE}")
        print(f"Ready to identify. Run: python {sys.argv[0]} --identify your_clip.wav")
    else:
        print("\n[error] No profiles were built. Check your audio files.")


# ─── IDENTIFICATION ───────────────────────────────────────────────────────────

def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def identify_clip(clip_path, verbose=True):
    """
    Identify which reciter a clip belongs to.
    Returns dict with ranked results and confidence scores.
    """
    if not os.path.exists(PROFILES_FILE):
        print(f"[error] No profiles found. Run: python {sys.argv[0]} --build first.")
        return None

    with open(PROFILES_FILE) as f:
        profiles = json.load(f)

    if verbose:
        print(f"\nIdentifying: {clip_path}")

    audio = load_audio(clip_path)
    if audio is None:
        return None

    clip_len = len(audio) / SAMPLE_RATE
    if verbose:
        print(f"Clip length: {clip_len:.1f} seconds")
        if clip_len < 5:
            print("  [warning] Very short clip — accuracy may be low.")

    emb = extract_embedding(audio)
    if emb is None:
        return None

    # Score against every reciter profile
    scores = {}
    for reciter, profile in profiles.items():
        scores[reciter] = cosine_similarity(emb, profile)

    # Convert to percentage confidence
    total = sum(max(0, s) for s in scores.values())
    confidences = {}
    for reciter, score in scores.items():
        confidences[reciter] = round(max(0, score) / total * 100, 1) if total > 0 else 0

    ranked = sorted(confidences.items(), key=lambda x: x[1], reverse=True)

    if verbose:
        print("\n─── RESULT ─────────────────────────────")
        for i, (reciter, pct) in enumerate(ranked):
            bar   = "█" * int(pct / 5)
            label = " <-- best match" if i == 0 else ""
            print(f"  {reciter:<20} {pct:5.1f}%  {bar}{label}")
        print("─────────────────────────────────────────")

        top_reciter, top_pct = ranked[0]
        if top_pct >= 60:
            verdict = f"Likely: {top_reciter}  ({top_pct}% confidence)"
        elif top_pct >= 40:
            verdict = f"Uncertain — possibly {top_reciter}, but low confidence."
        else:
            verdict = "Could not identify reliably. Clip may be too short or noisy."

        print(f"\n  {verdict}\n")

    return {"ranked": ranked, "clip_length": clip_len}


# ─── STRESS TEST ─────────────────────────────────────────────────────────────

def stress_test():
    """
    Run identification on every clip in dataset/ and log accuracy.
    Each clip's true label is its parent folder name.
    Saves results to results.csv.
    """
    if not os.path.exists(PROFILES_FILE):
        print(f"[error] Run --build first.")
        return

    print("\nRunning stress test across all dataset clips...\n")

    rows     = []
    correct  = 0
    total    = 0

    reciters = [d for d in os.listdir(DATASET_DIR)
                if os.path.isdir(os.path.join(DATASET_DIR, d))]

    for reciter in reciters:
        folder = os.path.join(DATASET_DIR, reciter)
        files  = [f for f in os.listdir(folder)
                  if f.lower().endswith(('.mp3', '.wav', '.m4a', '.ogg'))]

        for fname in files:
            path   = os.path.join(folder, fname)
            result = identify_clip(path, verbose=False)
            if result is None:
                continue

            ranked     = result["ranked"]
            predicted  = ranked[0][0]
            confidence = ranked[0][1]
            is_correct = (predicted == reciter)

            if is_correct:
                correct += 1
            total += 1

            status = "CORRECT" if is_correct else f"WRONG (got {predicted})"
            print(f"  [{reciter}] {fname:<40} {status}  ({confidence:.0f}%)")

            rows.append({
                "timestamp"  : datetime.now().isoformat(),
                "true_reciter": reciter,
                "clip"       : fname,
                "predicted"  : predicted,
                "confidence" : confidence,
                "correct"    : is_correct,
                "clip_length": result["clip_length"],
            })

    # Save CSV
    if rows:
        with open(RESULTS_FILE, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    accuracy = (correct / total * 100) if total > 0 else 0
    print(f"\n─── STRESS TEST SUMMARY ─────────────────")
    print(f"  Total clips : {total}")
    print(f"  Correct     : {correct}")
    print(f"  Accuracy    : {accuracy:.1f}%")
    print(f"  Results saved to {RESULTS_FILE}")
    print(f"─────────────────────────────────────────")

    if accuracy >= 75:
        print("\n  Strong result. Technical foundation is promising.")
    elif accuracy >= 50:
        print("\n  Moderate result. Try more clips per reciter to improve profiles.")
    else:
        print("\n  Low accuracy. Consider: longer clips, cleaner audio, more training data.")


# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Qur'an reciter recognition experiment")
    parser.add_argument("--build",    action="store_true", help="Build voice profiles from dataset/")
    parser.add_argument("--identify", metavar="CLIP",      help="Identify a single audio clip")
    parser.add_argument("--stress",   action="store_true", help="Run accuracy test on all dataset clips")
    args = parser.parse_args()

    if args.build:
        build_profiles()
    elif args.identify:
        identify_clip(args.identify)
    elif args.stress:
        stress_test()
    else:
        parser.print_help()
        print("\nQuick start:")
        print("  1. Add clips to dataset/alafasy/, dataset/dosari/, dataset/muaiqly/")
        print("  2. python quran_recognition_experiment.py --build")
        print("  3. python quran_recognition_experiment.py --identify test_clip.wav")


if __name__ == "__main__":
    main()
