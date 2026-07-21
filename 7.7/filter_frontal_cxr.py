import csv
from collections import Counter
from pathlib import Path

# ===== paths =====
CXR_ROOT = Path("/path/to/mimic_cxr/mimic_cxr_jpg")
METADATA_CSV = CXR_ROOT / "mimic-cxr-2.0.0-metadata.csv"

OUT_DIR = Path("./7.7")
OUT_CSV = OUT_DIR / "frontal_cxr_metadata.csv"
OUT_COUNTS = OUT_DIR / "frontal_cxr_counts.txt"

# MIMIC-CXR frontal views
FRONTAL_VIEWS = {"AP", "PA"}

total_rows = 0
kept_rows = 0
view_counter = Counter()
kept_view_counter = Counter()
patient_set = set()
study_set = set()

with open(METADATA_CSV, "r", newline="") as f_in, open(OUT_CSV, "w", newline="") as f_out:
    reader = csv.DictReader(f_in)
    fieldnames = reader.fieldnames

    if fieldnames is None:
        raise RuntimeError("Metadata CSV has no header.")

    writer = csv.DictWriter(f_out, fieldnames=fieldnames)
    writer.writeheader()

    for row in reader:
        total_rows += 1

        view = (row.get("ViewPosition") or "").strip().upper()
        view_counter[view] += 1

        if view not in FRONTAL_VIEWS:
            continue

        writer.writerow(row)
        kept_rows += 1
        kept_view_counter[view] += 1

        patient_set.add(row.get("subject_id"))
        study_set.add(row.get("study_id"))

summary = []
summary.append(f"Metadata CSV: {METADATA_CSV}")
summary.append(f"Output CSV: {OUT_CSV}")
summary.append("")
summary.append(f"Total metadata rows: {total_rows}")
summary.append(f"Kept frontal rows AP/PA: {kept_rows}")
summary.append(f"Unique frontal patients: {len(patient_set)}")
summary.append(f"Unique frontal studies: {len(study_set)}")
summary.append("")
summary.append("All ViewPosition counts:")
for view, count in view_counter.most_common():
    summary.append(f"  {repr(view)}: {count}")
summary.append("")
summary.append("Kept frontal ViewPosition counts:")
for view, count in kept_view_counter.most_common():
    summary.append(f"  {view}: {count}")

text = "\n".join(summary)

with open(OUT_COUNTS, "w") as f:
    f.write(text + "\n")

print(text)
