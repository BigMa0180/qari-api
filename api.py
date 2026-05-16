"""
Qari API Server
===============
FastAPI server that accepts audio and returns reciter identification.

USAGE:
    python api.py

Then test it at:
    http://localhost:8000/docs   <- interactive test UI
    http://localhost:8000/health <- health check
"""

import os
import json
import tempfile
import warnings

import numpy as np
import librosa
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

warnings.filterwarnings("ignore")


# ─── CONFIG ──────────────────────────────────────────────────────────────────

PROFILES_FILE = "profiles_v2.json"
SAMPLE_RATE   = 16000
N_MFCC        = 40
MIN_DURATION  = 2.0


# ─── APP ─────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Qari API",
    description="Identify Qur'an reciters from audio clips.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── LOAD PROFILES ───────────────────────────────────────────────────────────

def load_profiles():
    if not os.path.exists(PROFILES_FILE):
        raise RuntimeError(f"No profiles file found: {PROFILES_FILE}. Run qari_engine.py --build first.")
    with open(PROFILES_FILE) as f:
        return json.load(f)

profiles = load_profiles()
print(f"Loaded {len(profiles)} reciter profiles: {', '.join(sorted(profiles.keys()))}")


# ─── FEATURE EXTRACTION ──────────────────────────────────────────────────────

def extract_mfcc(audio):
    mfcc   = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=N_MFCC)
    delta  = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    return np.concatenate([
        np.mean(mfcc,   axis=1),
        np.std(mfcc,    axis=1),
        np.mean(delta,  axis=1),
        np.mean(delta2, axis=1),
    ])

def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def identify_audio(audio):
    vec    = extract_mfcc(audio)
    scores = {r: cosine_similarity(vec, p) for r, p in profiles.items()}
    total  = sum(max(0, s) for s in scores.values())
    confidences = {
        r: round(max(0, s) / total * 100, 1) if total > 0 else 0
        for r, s in scores.items()
    }
    ranked = sorted(confidences.items(), key=lambda x: x[1], reverse=True)
    return ranked


# ─── ROUTES ──────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status"   : "ok",
        "reciters" : len(profiles),
        "reciter_list": sorted(profiles.keys())
    }


@app.post("/identify")
async def identify(file: UploadFile = File(...)):
    """
    Upload an audio file (mp3, wav, m4a) and get back the identified reciter.
    """
    # Save uploaded file to temp location
    suffix = os.path.splitext(file.filename)[-1] or ".mp3"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # Load audio
        audio, _ = librosa.load(tmp_path, sr=SAMPLE_RATE, mono=True)
        duration  = len(audio) / SAMPLE_RATE

        if duration < MIN_DURATION:
            raise HTTPException(
                status_code=400,
                detail=f"Audio too short ({duration:.1f}s). Minimum is {MIN_DURATION}s."
            )

        # Identify
        ranked = identify_audio(audio)
        top_reciter, top_pct = ranked[0]
        gap = ranked[0][1] - ranked[1][1]

        # Confidence verdict
        if gap >= 10:
            confidence_label = "high"
        elif gap >= 5:
            confidence_label = "medium"
        else:
            confidence_label = "low"

        return {
            "success"         : True,
            "top_reciter"     : top_reciter,
            "confidence_pct"  : top_pct,
            "confidence_label": confidence_label,
            "gap"             : round(gap, 2),
            "duration_seconds": round(duration, 1),
            "all_scores"      : [
                {"reciter": r, "score": pct}
                for r, pct in ranked
            ]
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.unlink(tmp_path)


@app.get("/reciters")
def list_reciters():
    """List all reciters in the database."""
    return {
        "count"   : len(profiles),
        "reciters": sorted(profiles.keys())
    }


# ─── RUN ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
