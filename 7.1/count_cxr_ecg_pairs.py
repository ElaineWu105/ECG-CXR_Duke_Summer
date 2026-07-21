import csv
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict
from bisect import bisect_left, bisect_right

# ============================================================
# Paths
# ============================================================

CXR_METADATA_PATH = "/path/to/mimic_cxr/mimic_cxr_jpg/mimic-cxr-2.0.0-metadata.csv"
ECG_ROOT = "/path/to/MIMIC_waveform/MIMIC_IV_ECG_Matched/files"

OUT_SUMMARY = "cxr_ecg_pair_counts_summary.csv"
OUT_CXR = "cxr_study_times.csv"
OUT_ECG = "ecg_record_times.csv"
OUT_BAD_ECG = "bad_ecg_headers_in_pair_count.csv"

N_VALUES = [0, 2, 4, 6, 8, 10, 12]


# ============================================================
# Helper functions
# ============================================================

def parse_cxr_datetime(study_date, study_time):
    """
    CXR StudyDate example:
    21800506

    CXR StudyTime example:
    213014.53100000002
    """
    if not study_date or not study_time:
        return None

    date_str = str(study_date).strip()
    time_str = str(study_time).strip().split(".")[0]
    time_str = time_str.zfill(6)

    try:
        return datetime.strptime(date_str + time_str, "%Y%m%d%H%M%S")
    except Exception:
        return None


def parse_ecg_datetime(date_part, time_part):
    """
    ECG header dates should be parsed as DD/MM/YYYY.

    Evidence from all 800,035 ECG headers:
    - 484,076 are definitely DD/MM/YYYY
    - 0 are definitely MM/DD/YYYY
    - 315,959 are ambiguous

    Therefore ambiguous dates should also be parsed as DD/MM/YYYY.
    """
    date_part = date_part.strip()
    time_part = time_part.strip()

    try:
        return datetime.strptime(date_part + " " + time_part, "%d/%m/%Y %H:%M:%S")
    except Exception:
        return None


def parse_ecg_header(hea_path):
    """
    ECG first line example:
    00000000 12 500 5000 18:09:00 02/10/2117

    Subject line example:
    # <subject_id>: 00000000
    """
    record_id = hea_path.stem
    ecg_time = None
    subject_id = None
    first_line = ""

    try:
        with open(hea_path, "r", errors="ignore") as f:
            lines = f.readlines()
    except Exception:
        return None, {
            "path": str(hea_path),
            "status": "cannot_read_file",
            "first_line": ""
        }

    if not lines:
        return None, {
            "path": str(hea_path),
            "status": "empty_file",
            "first_line": ""
        }

    first_line = lines[0].strip()
    first = first_line.split()

    # Expected:
    # record_id, n_leads, fs, n_samples, HH:MM:SS, DD/MM/YYYY
    if len(first) >= 6:
        time_part = first[4]
        date_part = first[5]
        ecg_time = parse_ecg_datetime(date_part, time_part)

    for line in lines:
        line = line.strip()
        if line.startswith("# <subject_id>:"):
            subject_id = line.split(":")[-1].strip()
            break

    if subject_id is None and ecg_time is None:
        status = "missing_subject_and_time"
    elif subject_id is None:
        status = "missing_subject"
    elif ecg_time is None:
        status = "missing_time"
    else:
        status = "ok"

    if status != "ok":
        return None, {
            "path": str(hea_path),
            "status": status,
            "first_line": first_line
        }

    return {
        "subject_id": subject_id,
        "record_id": record_id,
        "ecg_time": ecg_time,
        "path": str(hea_path)
    }, None


def count_times_in_window(sorted_times, start, end):
    """
    Count timestamps in inclusive window [start, end].
    """
    if start > end:
        return 0

    left = bisect_left(sorted_times, start)
    right = bisect_right(sorted_times, end)
    return right - left


# ============================================================
# Load CXR study-level times
# ============================================================

print("===== Loading CXR metadata =====")

cxr_by_subject = defaultdict(list)
seen_cxr_studies = set()

total_cxr_rows = 0
valid_cxr_studies = 0
bad_cxr_rows = 0

with open(CXR_METADATA_PATH, "r", newline="") as f:
    reader = csv.DictReader(f)

    for row in reader:
        total_cxr_rows += 1

        subject_id = row["subject_id"].strip()
        study_id = row["study_id"].strip()

        key = (subject_id, study_id)
        if key in seen_cxr_studies:
            continue

        cxr_time = parse_cxr_datetime(row["StudyDate"], row["StudyTime"])

        if cxr_time is None:
            bad_cxr_rows += 1
            continue

        seen_cxr_studies.add(key)
        cxr_by_subject[subject_id].append({
            "study_id": study_id,
            "cxr_time": cxr_time
        })
        valid_cxr_studies += 1

for subject_id in cxr_by_subject:
    cxr_by_subject[subject_id].sort(key=lambda x: x["cxr_time"])

print("Total CXR metadata rows:", total_cxr_rows)
print("Valid unique CXR studies:", valid_cxr_studies)
print("Bad CXR rows:", bad_cxr_rows)
print("CXR patients:", len(cxr_by_subject))


with open(OUT_CXR, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["subject_id", "study_id", "cxr_time"])

    for subject_id, studies in cxr_by_subject.items():
        for s in studies:
            writer.writerow([
                subject_id,
                s["study_id"],
                s["cxr_time"].isoformat(sep=" ")
            ])

print("Saved:", OUT_CXR)


# ============================================================
# Load ECG record times
# ============================================================

print("\n===== Loading ECG headers =====")

ecg_by_subject = defaultdict(list)
bad_ecg_rows = []

hea_files = list(Path(ECG_ROOT).rglob("*.hea"))
print("Total .hea files found:", len(hea_files))

parsed_ecg = 0
bad_ecg = 0

for i, hea_path in enumerate(hea_files, start=1):
    info, bad_info = parse_ecg_header(hea_path)

    if info is None:
        bad_ecg += 1
        if bad_info is not None:
            bad_ecg_rows.append([
                bad_info["path"],
                bad_info["status"],
                bad_info["first_line"]
            ])
        continue

    subject_id = info["subject_id"]
    ecg_by_subject[subject_id].append(info)
    parsed_ecg += 1

    if i % 50000 == 0:
        print(f"Processed {i} / {len(hea_files)} ECG headers...")

for subject_id in ecg_by_subject:
    ecg_by_subject[subject_id].sort(key=lambda x: x["ecg_time"])

print("Parsed ECG records with valid time:", parsed_ecg)
print("Bad ECG headers:", bad_ecg)
print("ECG patients with at least one valid ECG time:", len(ecg_by_subject))


with open(OUT_ECG, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["subject_id", "record_id", "ecg_time", "path"])

    for subject_id, records in ecg_by_subject.items():
        for r in records:
            writer.writerow([
                subject_id,
                r["record_id"],
                r["ecg_time"].isoformat(sep=" "),
                r["path"]
            ])

print("Saved:", OUT_ECG)


with open(OUT_BAD_ECG, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["hea_path", "status", "first_line"])
    writer.writerows(bad_ecg_rows)

print("Saved:", OUT_BAD_ECG)


# ============================================================
# Count CXR-ECG pairs
# ============================================================

print("\n===== Counting CXR-ECG pairs =====")

common_subjects = sorted(set(cxr_by_subject.keys()) & set(ecg_by_subject.keys()))

print("Patients with both CXR and valid ECG time:", len(common_subjects))

# ------------------------------------------------------------
# Criterion 1:
# CXR at time t, ECG in [t-12-n, t-n]
# ------------------------------------------------------------

criterion1_pair_counts = {n: 0 for n in N_VALUES}
criterion1_cxr_with_at_least_one_ecg = {n: 0 for n in N_VALUES}

# ------------------------------------------------------------
# Criterion 2:
# CXR at time t2,
# CXR at t1 in [t2-24, t2),
# ECG in [max(t2-12-n, t1), t2-n]
#
# We compute three interpretations:
#
# 1. nearest:
#    t1 = closest previous CXR to t2
#
# 2. earliest:
#    t1 = earliest previous CXR within [t2-24, t2),
#    i.e. closest to t2 - 24h
#
# 3. all_t1:
#    all previous CXRs within [t2-24, t2)
# ------------------------------------------------------------

criterion2_nearest_pair_counts = {n: 0 for n in N_VALUES}
criterion2_nearest_cxr_with_at_least_one_ecg = {n: 0 for n in N_VALUES}
criterion2_nearest_eligible_cxr_count = 0

criterion2_earliest_pair_counts = {n: 0 for n in N_VALUES}
criterion2_earliest_cxr_with_at_least_one_ecg = {n: 0 for n in N_VALUES}
criterion2_earliest_eligible_cxr_count = 0

criterion2_all_t1_pair_counts = {n: 0 for n in N_VALUES}
criterion2_all_t1_windows_with_at_least_one_ecg = {n: 0 for n in N_VALUES}
criterion2_all_t1_window_count = 0

total_cxr_considered = 0
total_ecg_considered = 0

for subject_id in common_subjects:
    cxrs = cxr_by_subject[subject_id]
    ecgs = ecg_by_subject[subject_id]

    cxr_times = [x["cxr_time"] for x in cxrs]
    ecg_times = [x["ecg_time"] for x in ecgs]

    total_cxr_considered += len(cxr_times)
    total_ecg_considered += len(ecg_times)

    # -------------------------
    # Criterion 1
    # -------------------------
    for t in cxr_times:
        for n in N_VALUES:
            start = t - timedelta(hours=12 + n)
            end = t - timedelta(hours=n)

            c = count_times_in_window(ecg_times, start, end)

            criterion1_pair_counts[n] += c
            if c > 0:
                criterion1_cxr_with_at_least_one_ecg[n] += 1

    # -------------------------
    # Criterion 2
    # -------------------------
    for t2 in cxr_times:
        prev_start = t2 - timedelta(hours=24)

        # previous CXR only, excluding the current t2 itself
        left = bisect_left(cxr_times, prev_start)
        right = bisect_left(cxr_times, t2)

        previous_t1_list = cxr_times[left:right]

        if not previous_t1_list:
            continue

        # -------------------------
        # Criterion 2A: nearest previous CXR
        # -------------------------
        nearest_t1 = previous_t1_list[-1]
        criterion2_nearest_eligible_cxr_count += 1

        for n in N_VALUES:
            start = max(t2 - timedelta(hours=12 + n), nearest_t1)
            end = t2 - timedelta(hours=n)

            c = count_times_in_window(ecg_times, start, end)

            criterion2_nearest_pair_counts[n] += c
            if c > 0:
                criterion2_nearest_cxr_with_at_least_one_ecg[n] += 1

        # -------------------------
        # Criterion 2B: earliest previous CXR within 24h
        # This is the t1 closest to t2 - 24h.
        # -------------------------
        earliest_t1 = previous_t1_list[0]
        criterion2_earliest_eligible_cxr_count += 1

        for n in N_VALUES:
            start = max(t2 - timedelta(hours=12 + n), earliest_t1)
            end = t2 - timedelta(hours=n)

            c = count_times_in_window(ecg_times, start, end)

            criterion2_earliest_pair_counts[n] += c
            if c > 0:
                criterion2_earliest_cxr_with_at_least_one_ecg[n] += 1

        # -------------------------
        # Criterion 2C: all previous CXRs within 24h
        # This counts CXR-CXR-ECG windows/triples.
        # -------------------------
        for t1 in previous_t1_list:
            criterion2_all_t1_window_count += 1

            for n in N_VALUES:
                start = max(t2 - timedelta(hours=12 + n), t1)
                end = t2 - timedelta(hours=n)

                c = count_times_in_window(ecg_times, start, end)

                criterion2_all_t1_pair_counts[n] += c
                if c > 0:
                    criterion2_all_t1_windows_with_at_least_one_ecg[n] += 1


print("Total CXR studies considered among common patients:", total_cxr_considered)
print("Total ECG records considered among common patients:", total_ecg_considered)
print("Criterion 2 nearest eligible CXR t2 count:", criterion2_nearest_eligible_cxr_count)
print("Criterion 2 earliest eligible CXR t2 count:", criterion2_earliest_eligible_cxr_count)
print("Criterion 2 all-t1 CXR-CXR window count:", criterion2_all_t1_window_count)


# ============================================================
# Save summary
# ============================================================

with open(OUT_SUMMARY, "w", newline="") as f:
    writer = csv.writer(f)

    writer.writerow([
        "n_hours",

        "criterion1_pair_count",
        "criterion1_cxr_with_at_least_one_ecg",

        "criterion2_nearest_pair_count",
        "criterion2_nearest_cxr_with_at_least_one_ecg",

        "criterion2_earliest_pair_count",
        "criterion2_earliest_cxr_with_at_least_one_ecg",

        "criterion2_all_t1_pair_or_triple_count",
        "criterion2_all_t1_windows_with_at_least_one_ecg"
    ])

    for n in N_VALUES:
        writer.writerow([
            n,

            criterion1_pair_counts[n],
            criterion1_cxr_with_at_least_one_ecg[n],

            criterion2_nearest_pair_counts[n],
            criterion2_nearest_cxr_with_at_least_one_ecg[n],

            criterion2_earliest_pair_counts[n],
            criterion2_earliest_cxr_with_at_least_one_ecg[n],

            criterion2_all_t1_pair_counts[n],
            criterion2_all_t1_windows_with_at_least_one_ecg[n]
        ])

print("\n===== Summary by n =====")
print(
    "n | "
    "criterion1_pairs | "
    "criterion1_CXR_with_ECG | "
    "criterion2_nearest_pairs | "
    "criterion2_earliest_pairs | "
    "criterion2_all_t1_pairs/triples"
)

for n in N_VALUES:
    print(
        n,
        criterion1_pair_counts[n],
        criterion1_cxr_with_at_least_one_ecg[n],
        criterion2_nearest_pair_counts[n],
        criterion2_earliest_pair_counts[n],
        criterion2_all_t1_pair_counts[n],
        sep=" | "
    )

print("\nSaved summary:", OUT_SUMMARY)
print("Done.")
