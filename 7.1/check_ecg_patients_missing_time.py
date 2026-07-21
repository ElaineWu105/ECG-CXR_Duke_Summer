from pathlib import Path
from collections import defaultdict
from datetime import datetime
import csv

ECG_ROOT = "/path/to/MIMIC_waveform/MIMIC_IV_ECG_Matched/files"

OUT_PATIENT_SUMMARY = "ecg_patient_time_check_summary.csv"
OUT_BAD_HEADERS = "ecg_headers_missing_time_or_subject.csv"


def get_subject_from_path(path):
    """
    Example:
    /path/to/example.hea

    patient-level folder:
    p00000000 -> subject_id 00000000
    """
    candidates = []

    for part in path.parts:
        if part.startswith("p") and len(part) > 1 and part[1:].isdigit():
            candidates.append(part[1:])

    if candidates:
        return max(candidates, key=len)

    return None


def parse_datetime_flexible(date_part, time_part):
    """
    ECG headers appear to contain mixed date formats.

    Examples:
    02/10/2117 18:09:00
    23/07/2180 09:54:00

    We support both:
    MM/DD/YYYY
    DD/MM/YYYY
    """
    date_part = date_part.strip()
    time_part = time_part.strip()

    for fmt in ["%m/%d/%Y %H:%M:%S", "%d/%m/%Y %H:%M:%S"]:
        try:
            return datetime.strptime(date_part + " " + time_part, fmt)
        except Exception:
            pass

    return None


def parse_ecg_header_time_and_subject(hea_path):
    """
    First line example:
    00000000 12 500 5000 18:09:00 02/10/2117

    Subject line example:
    # <subject_id>: 00000000
    """
    header_subject = None
    ecg_time = None
    first_line = ""

    try:
        with open(hea_path, "r", errors="ignore") as f:
            lines = f.readlines()
    except Exception:
        return None, None, "cannot_read_file", ""

    if not lines:
        return None, None, "empty_file", ""

    first_line = lines[0].strip()
    parts = first_line.split()

    if len(parts) >= 6:
        time_part = parts[4]
        date_part = parts[5]
        ecg_time = parse_datetime_flexible(date_part, time_part)

    for line in lines:
        line = line.strip()
        if line.startswith("# <subject_id>:"):
            header_subject = line.split(":")[-1].strip()
            break

    if header_subject is None and ecg_time is None:
        status = "missing_subject_and_time"
    elif header_subject is None:
        status = "missing_subject"
    elif ecg_time is None:
        status = "missing_time"
    else:
        status = "ok"

    return header_subject, ecg_time, status, first_line


print("Scanning ECG headers...")
hea_files = list(Path(ECG_ROOT).rglob("*.hea"))
print("Total .hea files found:", len(hea_files))

patient_stats = defaultdict(lambda: {
    "total_headers": 0,
    "headers_with_time": 0,
    "headers_missing_time": 0,
    "headers_with_subject": 0,
    "headers_missing_subject": 0,
    "headers_ok": 0,
})

bad_rows = []

for i, hea_path in enumerate(hea_files, start=1):
    path_subject = get_subject_from_path(hea_path)
    header_subject, ecg_time, status, first_line = parse_ecg_header_time_and_subject(hea_path)

    subject_id = header_subject if header_subject is not None else path_subject

    if subject_id is None:
        subject_id = "UNKNOWN"

    patient_stats[subject_id]["total_headers"] += 1

    if ecg_time is not None:
        patient_stats[subject_id]["headers_with_time"] += 1
    else:
        patient_stats[subject_id]["headers_missing_time"] += 1

    if header_subject is not None:
        patient_stats[subject_id]["headers_with_subject"] += 1
    else:
        patient_stats[subject_id]["headers_missing_subject"] += 1

    if status == "ok":
        patient_stats[subject_id]["headers_ok"] += 1
    else:
        bad_rows.append([
            subject_id,
            str(hea_path),
            status,
            first_line
        ])

    if i % 50000 == 0:
        print(f"Processed {i} / {len(hea_files)} headers...")


patients_total = len(patient_stats)
patients_all_have_time = 0
patients_some_missing_time = 0
patients_no_valid_time = 0

for subject_id, s in patient_stats.items():
    if s["headers_with_time"] == s["total_headers"]:
        patients_all_have_time += 1
    elif s["headers_with_time"] == 0:
        patients_no_valid_time += 1
    else:
        patients_some_missing_time += 1


headers_with_time = sum(s["headers_with_time"] for s in patient_stats.values())
headers_missing_time = sum(s["headers_missing_time"] for s in patient_stats.values())
headers_missing_subject = sum(s["headers_missing_subject"] for s in patient_stats.values())

print("\n===== Patient-level summary =====")
print("Total ECG patients found from headers/paths:", patients_total)
print("Patients where all ECG headers have time:", patients_all_have_time)
print("Patients with some ECG headers missing time:", patients_some_missing_time)
print("Patients with no ECG headers having valid time:", patients_no_valid_time)

print("\n===== Header-level summary =====")
print("Total ECG headers:", len(hea_files))
print("Headers with valid time:", headers_with_time)
print("Headers missing time:", headers_missing_time)
print("Headers missing subject:", headers_missing_subject)
print("Bad headers total:", len(bad_rows))

print("\n===== Internal checks =====")
print("Patient categories sum:", patients_all_have_time + patients_some_missing_time + patients_no_valid_time)
print("Header time categories sum:", headers_with_time + headers_missing_time)


with open(OUT_PATIENT_SUMMARY, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "subject_id",
        "total_headers",
        "headers_with_time",
        "headers_missing_time",
        "headers_with_subject",
        "headers_missing_subject",
        "headers_ok"
    ])

    for subject_id, s in sorted(patient_stats.items()):
        writer.writerow([
            subject_id,
            s["total_headers"],
            s["headers_with_time"],
            s["headers_missing_time"],
            s["headers_with_subject"],
            s["headers_missing_subject"],
            s["headers_ok"]
        ])

with open(OUT_BAD_HEADERS, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "subject_id",
        "hea_path",
        "status",
        "first_line"
    ])
    writer.writerows(bad_rows)

print("\nSaved:", OUT_PATIENT_SUMMARY)
print("Saved:", OUT_BAD_HEADERS)
print("Done.")
