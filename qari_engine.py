"""
Qari Engine — MFCC-Based Reciter Recognition
=============================================
Version 2 — proper script built on proven MFCC approach.

USAGE:
    python qari_engine.py --build
    python qari_engine.py --identify test_clip.mp3
    python qari_engine.py --stress
    python qari_engine.py --add-reciter <name> <folder>

FOLDER STRUCTURE:
    dataset/
        alafasy/
        dosari/
        muaiqly/
    test_clip.mp3
    qari_engine.py
"""

import os
import sys
import json
import argparse
import csv
import warnings
from datetime import datetime

import numpy as np
import librosa

warnings.filterwarnings("ignore")


# ─── CONFIG ──────────────────────────────────────────────────────────────────

DATASET_DIR   = "dataset"
PROFILES_FILE = "profiles_v2.json"
RESULTS_FILE  = "results_v2.csv"
SAMPLE_RATE   = 16000
N_MFCC        = 40       # number of MFCC coefficients
MIN_DURATION  = 3.0      # skip clips shorter than this (seconds)


# ─── AUDIO + FEATURES ────────────────────────────────────────────────────────

def load_audio(path):
    try:
        audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
        return audio
    except Exception as e:
        print(f"  [error] Could not load {path}: {e}")
        return None


def extract_mfcc(audio):
    """
    Extract MFCC feature vector from audio.
    Returns a 40-dimensional vector representing the audio's tonal fingerprint.
    This is the core of the recognition engine.
    """
    mfcc        = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=N_MFCC)
    delta       = librosa.feature.delta(mfcc)         # rate of change
    delta2      = librosa.feature.delta(mfcc, order=2) # acceleration
    combined    = np.concatenate([
        np.mean(mfcc,   axis=1),
        np.std(mfcc,    axis=1),
        np.mean(delta,  axis=1),
        np.mean(delta2, axis=1),
    ])
    return combined


def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# ─── BUILD PROFILES ──────────────────────────────────────────────────────────

def build_profiles():
    """
    Scan dataset/ and build an MFCC voice profile for each reciter.
    Profiles are saved to profiles_v2.json.
    """
    if not os.path.exists(DATASET_DIR):
        print(f"[error] No '{DATASET_DIR}' folder found.")
        return

    reciters = sorted([
        d for d in os.listdir(DATASET_DIR)
        if os.path.isdir(os.path.join(DATASET_DIR, d))
    ])

    if not reciters:
        print("[error] No reciter folders found inside dataset/")
        return

    print(f"\nBuilding profiles for {len(reciters)} reciter(s): {', '.join(reciters)}\n")

    profiles = {}
    for reciter in reciters:
        folder = os.path.join(DATASET_DIR, reciter)
        files  = [f for f in os.listdir(folder)
                  if f.lower().endswith(('.mp3', '.wav', '.m4a', '.ogg'))]

        if not files:
            print(f"  [{reciter}] No audio files — skipping.")
            continue

        print(f"  [{reciter}] Processing {len(files)} clip(s)...")
        vectors = []

        for fname in files:
            path  = os.path.join(folder, fname)
            audio = load_audio(path)
            if audio is None:
                continue

            duration = len(audio) / SAMPLE_RATE
            if duration < MIN_DURATION:
                print(f"    -> {fname} ({duration:.1f}s) — too short, skipping.")
                continue

            vec = extract_mfcc(audio)
            vectors.append(vec)
            print(f"    -> {fname} ({duration:.1f}s) ✓")

        if vectors:
            profiles[reciter] = np.mean(vectors, axis=0).tolist()
            print(f"  [{reciter}] Profile built from {len(vectors)} clip(s).\n")
        else:
            print(f"  [{reciter}] No valid clips found.\n")

    if profiles:
        with open(PROFILES_FILE, "w") as f:
            json.dump(profiles, f, indent=2)
        print(f"Profiles saved to {PROFILES_FILE}")
        print(f"Ready. Run: python qari_engine.py --identify your_clip.mp3\n")
    else:
        print("[error] No profiles built.")


# ─── IDENTIFY ────────────────────────────────────────────────────────────────

def identify(clip_path, verbose=True):
    """
    Identify the reciter of a given audio clip.
    Returns ranked results with confidence scores.
    """
    if not os.path.exists(PROFILES_FILE):
        print(f"[error] No profiles found. Run --build first.")
        return None

    with open(PROFILES_FILE) as f:
        profiles = json.load(f)

    if not os.path.exists(clip_path):
        print(f"[error] File not found: {clip_path}")
        return None

    audio = load_audio(clip_path)
    if audio is None:
        return None

    duration = len(audio) / SAMPLE_RATE

    if verbose:
        print(f"\nIdentifying: {clip_path}  ({duration:.1f}s)\n")
        if duration < 5:
            print("  [warning] Very short clip — accuracy may be lower.\n")

    test_vec = extract_mfcc(audio)

    # Score against every reciter
    scores = {
        reciter: cosine_similarity(test_vec, profile)
        for reciter, profile in profiles.items()
    }

    # Normalise to percentage
    total = sum(max(0, s) for s in scores.values())
    confidences = {
        r: round(max(0, s) / total * 100, 1) if total > 0 else 0
        for r, s in scores.items()
    }
    ranked = sorted(confidences.items(), key=lambda x: x[1], reverse=True)

    if verbose:
        print("─── RESULT ─────────────────────────────────")
        for i, (reciter, pct) in enumerate(ranked):
            bar   = "█" * int(pct / 3)
            label = "  ← best match" if i == 0 else ""
            print(f"  {reciter:<22} {pct:5.1f}%  {bar}{label}")
        print("─────────────────────────────────────────────")

        top, top_pct     = ranked[0]
        second, sec_pct  = ranked[1]
        gap              = top_pct - sec_pct

        if gap >= 10:
            verdict = f"Confident:  {top}  ({top_pct}%)"
        elif gap >= 5:
            verdict = f"Likely:     {top}  ({top_pct}%)  — gap is narrow, consider longer clip"
        else:
            verdict = f"Uncertain  — {top} vs {second} too close ({gap:.1f}% gap)"

        print(f"\n  {verdict}\n")

    return {
        "ranked"      : ranked,
        "top"         : ranked[0][0],
        "top_pct"     : ranked[0][1],
        "gap"         : ranked[0][1] - ranked[1][1],
        "duration"    : duration,
    }


# ─── STRESS TEST ─────────────────────────────────────────────────────────────

def stress_test():
    """
    Test every clip in the dataset and report accuracy.
    Each clip's true label = its parent folder name.
    Saves detailed results to results_v2.csv.
    """
    if not os.path.exists(PROFILES_FILE):
        print("[error] Run --build first.")
        return

    print("\nRunning stress test...\n")

    rows    = []
    correct = 0
    total   = 0

    reciters = sorted([
        d for d in os.listdir(DATASET_DIR)
        if os.path.isdir(os.path.join(DATASET_DIR, d))
    ])

    for reciter in reciters:
        folder = os.path.join(DATASET_DIR, reciter)
        files  = [f for f in os.listdir(folder)
                  if f.lower().endswith(('.mp3', '.wav', '.m4a', '.ogg'))]

        for fname in files:
            path   = os.path.join(folder, fname)
            result = identify(path, verbose=False)
            if result is None:
                continue

            predicted  = result["top"]
            confidence = result["top_pct"]
            gap        = result["gap"]
            is_correct = (predicted == reciter)

            if is_correct:
                correct += 1
            total += 1

            status = "✓" if is_correct else f"✗ (got {predicted})"
            print(f"  [{reciter}]  {fname:<30}  {status}  {confidence:.0f}%  gap={gap:.1f}%")

            rows.append({
                "timestamp"    : datetime.now().isoformat(),
                "true_reciter" : reciter,
                "clip"         : fname,
                "predicted"    : predicted,
                "confidence"   : confidence,
                "gap"          : gap,
                "correct"      : is_correct,
                "duration"     : result["duration"],
            })

    if rows:
        with open(RESULTS_FILE, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    accuracy = (correct / total * 100) if total > 0 else 0
    print(f"\n─── STRESS TEST SUMMARY ─────────────────────")
    print(f"  Reciters tested : {len(reciters)}")
    print(f"  Total clips     : {total}")
    print(f"  Correct         : {correct}")
    print(f"  Accuracy        : {accuracy:.1f}%")
    print(f"  Results saved   : {RESULTS_FILE}")
    print(f"─────────────────────────────────────────────")

    if accuracy >= 80:
        print("\n  Strong result. Ready to scale up.")
    elif accuracy >= 60:
        print("\n  Decent baseline. Add more clips per reciter to improve.")
    else:
        print("\n  Low accuracy. Try adding more and longer clips.")


# ─── ADD RECITER ─────────────────────────────────────────────────────────────

def add_reciter(name, folder):
    """
    Add a single new reciter to existing profiles without rebuilding everything.
    """
    if not os.path.exists(PROFILES_FILE):
        print("[error] No profiles file found. Run --build first.")
        return

    if not os.path.exists(folder):
        print(f"[error] Folder not found: {folder}")
        return

    files = [f for f in os.listdir(folder)
             if f.lower().endswith(('.mp3', '.wav', '.m4a', '.ogg'))]

    if not files:
        print(f"[error] No audio files found in {folder}")
        return

    print(f"\nAdding reciter: {name} from {folder}\n")
    vectors = []

    for fname in files:
        path  = os.path.join(folder, fname)
        audio = load_audio(path)
        if audio is None:
            continue
        duration = len(audio) / SAMPLE_RATE
        if duration < MIN_DURATION:
            print(f"  -> {fname} too short, skipping.")
            continue
        vec = extract_mfcc(audio)
        vectors.append(vec)
        print(f"  -> {fname} ({duration:.1f}s) ✓")

    if not vectors:
        print("[error] No valid clips.")
        return

    with open(PROFILES_FILE) as f:
        profiles = json.load(f)

    profiles[name] = np.mean(vectors, axis=0).tolist()

    with open(PROFILES_FILE, "w") as f:
        json.dump(profiles, f, indent=2)

    print(f"\n  {name} added. Total reciters: {len(profiles)}")
    all_reciters = ', '.join(sorted(profiles.keys()))
    print(f"  Current reciters: {all_reciters}\n")


# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Qari Engine — MFCC-based Qur'an reciter recognition"
    )
    parser.add_argument("--build",       action="store_true",  help="Build profiles from dataset/")
    parser.add_argument("--identify",    metavar="CLIP",       help="Identify a single clip")
    parser.add_argument("--stress",      action="store_true",  help="Test accuracy across all clips")
    parser.add_argument("--add-reciter", nargs=2, metavar=("NAME", "FOLDER"), help="Add a new reciter")
    args = parser.parse_args()

    if args.build:
        build_profiles()
    elif args.identify:
        identify(args.identify)
    elif args.stress:
        stress_test()
    elif args.add_reciter:
        add_reciter(args.add_reciter[0], args.add_reciter[1])
    else:
        parser.print_help()
        print("\nQuick start:")
        print("  python qari_engine.py --build")
        print("  python qari_engine.py --identify test_clip.mp3")
        print("  python qari_engine.py --stress")


if __name__ == "__main__":
    main()
