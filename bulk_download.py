"""
Bulk Dataset Downloader
=======================
Downloads 50 clips per reciter from Surah Al-Baqarah (long ayahs).
Covers the 10 most popular reciters for the Qari app.

USAGE:
    python bulk_download.py
"""

import urllib.request
import os
import time

# ─── RECITERS ─────────────────────────────────────────────────────────────────

RECITERS = {
    "alafasy"  : "Alafasy_128kbps",
    "dussary"  : "Yasser_Ad-Dussary_128kbps",
    "muaiqly"  : "MaherAlMuaiqly128kbps",
    "sudais"   : "Abdurrahmaan_As-Sudais_192kbps",
    "qatami"   : "Nasser_Alqatami_128kbps",
    "shatri"   : "Abu_Bakr_Ash-Shaatree_128kbps",
    "ghamdi"   : "Saad_Al-Ghamdi_128kbps",
    "basfar"   : "Abdullah_Basfar_192kbps",
    "jalil"    : "Khalid_Abdullaah_al-Qahtaanee_192kbps",
    "bukhatir" : "Salah_Al_Budair_128kbps",
}

# Surah Al-Baqarah — long ayahs, good for voice profiling
SURAH = 2
START = 2
END   = 51  # 50 ayahs total

BASE_URL = "https://everyayah.com/data"


# ─── DOWNLOAD ─────────────────────────────────────────────────────────────────

def download_reciter(name, url_folder):
    folder = os.path.join("dataset", name)
    os.makedirs(folder, exist_ok=True)

    print(f"\n{'─'*50}")
    print(f"  Downloading: {name}")
    print(f"  Surah {SURAH}, Ayahs {START}–{END}")
    print(f"{'─'*50}")

    downloaded = 0
    failed     = 0
    skipped    = 0

    for ayah in range(START, END + 1):
        filename  = f"{SURAH:03d}{ayah:03d}.mp3"
        url       = f"{BASE_URL}/{url_folder}/{filename}"
        save_path = os.path.join(folder, f"{name}_s{SURAH}_a{ayah}.mp3")

        if os.path.exists(save_path) and os.path.getsize(save_path) > 2000:
            print(f"  [ayah {ayah:3d}] Already exists.")
            skipped += 1
            downloaded += 1
            continue

        try:
            headers = {"User-Agent": "Mozilla/5.0"}
            req     = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=15) as response:
                data = response.read()

            if len(data) < 2000:
                print(f"  [ayah {ayah:3d}] ✗  Too small — likely missing")
                failed += 1
                continue

            with open(save_path, "wb") as f:
                f.write(data)

            print(f"  [ayah {ayah:3d}] ✓  ({len(data)//1024}KB)")
            downloaded += 1
            time.sleep(0.35)

        except Exception as e:
            print(f"  [ayah {ayah:3d}] ✗  {e}")
            failed += 1
            time.sleep(0.5)

    print(f"\n  {name} done — Downloaded: {downloaded}  Failed: {failed}  Skipped: {skipped}")
    return downloaded, failed


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    print("Qari Bulk Dataset Downloader")
    print(f"Reciters : {len(RECITERS)}")
    print(f"Clips per reciter : {END - START + 1}")
    print(f"Total clips : {len(RECITERS) * (END - START + 1)}")
    print(f"Surah : Al-Baqarah ({SURAH}), Ayahs {START}–{END}")

    total_downloaded = 0
    total_failed     = 0

    for name, url_folder in RECITERS.items():
        d, f = download_reciter(name, url_folder)
        total_downloaded += d
        total_failed     += f

    print(f"\n{'='*50}")
    print(f"  ALL DONE")
    print(f"  Total downloaded : {total_downloaded}")
    print(f"  Total failed     : {total_failed}")
    print(f"{'='*50}")
    print("\nNext steps:")
    print("  python qari_engine.py --build")
    print("  python qari_engine.py --stress")


if __name__ == "__main__":
    main()
