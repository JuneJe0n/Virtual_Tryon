# save as: extract_bare_ffhq_makeup.py
from pathlib import Path
import shutil

INPUT_DIR  = Path("/home/jiyoon/data/FFHQ_Makeup")      
OUTPUT_DIR = Path("/home/jiyoon/data/FFHQ")  
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# match names like bare.jpg / BARE.JPG / bare_face.jpeg etc.
BARE_PREFIXES = ("bare", "bareface", "bare_face", "no_makeup", "nomakeup")
BARE_EXTS = (".jpg", ".jpeg", ".png", ".webp")

n_ok = 0
missing = []

for person_dir in sorted([p for p in INPUT_DIR.iterdir() if p.is_dir()]):
    # find a candidate bare image in this folder
    candidate = None
    for f in person_dir.iterdir():
        name_low = f.name.lower()
        if f.suffix.lower() in BARE_EXTS and any(name_low.startswith(pref) for pref in BARE_PREFIXES):
            candidate = f
            break

    if candidate and candidate.is_file():
        out_path = OUTPUT_DIR / f"{person_dir.name}.jpg"
        # copy and normalize extension to .jpg (no re-encode, just file copy & rename)
        shutil.copy2(candidate, out_path)
        n_ok += 1
    else:
        missing.append(person_dir.name)

print(f"✅ Copied {n_ok} bare images to {OUTPUT_DIR}")
if missing:
    print(f"⚠️ Folders without a detected bare image: {len(missing)}")
    # print first few for a quick peek
    print(", ".join(missing[:20]), "..." if len(missing) > 20 else "")
