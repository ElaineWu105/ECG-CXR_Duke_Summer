from pathlib import Path
from collections import defaultdict
from datetime import datetime
import csv
import statistics


CXR_METADATA = Path("/path/to/mimic_cxr/mimic_cxr_jpg/mimic-cxr-2.0.0-metadata.csv")


def parse_cxr_datetime(study_date, study_time):
    """
    Convert StudyDate + StudyTime into a Python datetime object.
    """
    if not study_date or not study_time:
        return None

    study_date = str(study_date).strip()
    study_time = str(study_time).strip()

    if study_date == "" or study_time == "":
        return None

    # Example: "084400.000" -> "084400"
    study_time = study_time.split(".")[0]

    # Example: "84400" -> "084400"
    study_time = study_time.zfill(6)

    try:
        return datetime.strptime(study_date + study_time, "%Y%m%d%H%M%S")
    except ValueError:
        return None


def percentile(values, p):
    """
    Simple percentile calculation.
    """
    values = sorted(values)

    if not values:
        return None

    idx = int(p * (len(values) - 1))
    return values[idx]


def main():
    print("Task 3: CXR time interval between consecutive studies")
    print("CXR metadata:", CXR_METADATA)

    if not CXR_METADATA.exists():
        print("Metadata file does not exist. Please check the path.")
        return

    # studies_by_subject:
    # subject_id -> {study_id -> datetime}
    #
    studies_by_subject = defaultdict(dict)

    total_rows = 0
    valid_rows = 0
    bad_time_rows = 0

    with CXR_METADATA.open("r", newline="") as f:
        reader = csv.DictReader(f)

        print("\nMetadata columns:")
        print(reader.fieldnames)

        required_cols = {"subject_id", "study_id", "StudyDate", "StudyTime"}
        missing_cols = required_cols - set(reader.fieldnames)

        if missing_cols:
            print("Missing required columns:", missing_cols)
            return

        for row in reader:
            total_rows += 1

            subject_id = row["subject_id"].strip()
            study_id = row["study_id"].strip()
            study_date = row["StudyDate"].strip()
            study_time = row["StudyTime"].strip()

            if subject_id == "" or study_id == "":
                continue

            dt = parse_cxr_datetime(study_date, study_time)

            if dt is None:
                bad_time_rows += 1
                continue

            valid_rows += 1

            # One CXR study can have multiple images.
            # Keep one timestamp per subject_id + study_id.
            studies_by_subject[subject_id][study_id] = dt

    print("\n===== Data loading summary =====")
    print("Total metadata rows:", total_rows)
    print("Rows with valid datetime:", valid_rows)
    print("Rows with bad/missing datetime:", bad_time_rows)
    print("Number of patients with at least one CXR study:", len(studies_by_subject))

    intervals_days = []
    patients_with_multiple_studies = 0

    example_printed = 0

    for subject_id, study_dict in studies_by_subject.items():
        # Sort all CXR studies of this patient by time.
        study_times = sorted(study_dict.values())

        if len(study_times) < 2:
            continue

        patients_with_multiple_studies += 1

        # Compute consecutive intervals:
        # t2 - t1, t3 - t2, ...
        for i in range(1, len(study_times)):
            delta = study_times[i] - study_times[i - 1]
            days = delta.total_seconds() / (24 * 3600)

            if days < 0:
                continue

            intervals_days.append(days)

        # Print a few examples to check the logic.
        if example_printed < 5:
            print("\nExample patient:", subject_id)
            print("Number of CXR studies:", len(study_times))
            print("First few study times:")
            for t in study_times[:5]:
                print(" ", t)
            example_printed += 1

    print("\n===== Task 3 CXR interval summary =====")
    print("Patients with at least 2 CXR studies:", patients_with_multiple_studies)
    print("Number of consecutive intervals computed:", len(intervals_days))

    if len(intervals_days) == 0:
        print("No intervals found.")
        return

    print("Mean interval days:", statistics.mean(intervals_days))
    print("Median interval days:", statistics.median(intervals_days))
    print("Min interval days:", min(intervals_days))
    print("Max interval days:", max(intervals_days))
    print("25th percentile days:", percentile(intervals_days, 0.25))
    print("75th percentile days:", percentile(intervals_days, 0.75))

    intervals_hours = [x * 24 for x in intervals_days]

    print("\n===== Same results in hours =====")
    print("Mean interval hours:", statistics.mean(intervals_hours))
    print("Median interval hours:", statistics.median(intervals_hours))
    print("Min interval hours:", min(intervals_hours))
    print("Max interval hours:", max(intervals_hours))


if __name__ == "__main__":
    main()