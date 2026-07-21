from pathlib import Path
import csv


ECG_ROOT = Path("/path/to/MIMIC_waveform/MIMIC_IV_ECG_Matched")
CXR_METADATA = Path("/path/to/mimic_cxr/mimic_cxr_jpg/mimic-cxr-2.0.0-metadata.csv")


def get_ecg_subjects():
    """
    Get all subject_id values from ECG .hea file paths.
    """
    ecg_subjects = set()

    hea_files = ECG_ROOT.rglob("*.hea")

    for path in hea_files:
        # Example:
        # .../files/p0000/p00000000/s00000000/00000000.hea
        subject_folder = path.parent.parent.name  # p00000000

        if subject_folder.startswith("p"):
            subject_id = subject_folder.lstrip("p")
            ecg_subjects.add(subject_id)

    return ecg_subjects


def get_cxr_subjects():
    """
    Get all subject_id values from CXR metadata.
    """
    cxr_subjects = set()

    with CXR_METADATA.open("r", newline="") as f:
        reader = csv.DictReader(f)

        if "subject_id" not in reader.fieldnames:
            raise ValueError("subject_id column not found in CXR metadata")

        for row in reader:
            subject_id = row["subject_id"]
            if subject_id != "":
                cxr_subjects.add(subject_id)

    return cxr_subjects


def main():
    print("Comparing ECG and CXR patient sets...")
    print("ECG_ROOT:", ECG_ROOT)
    print("CXR_METADATA:", CXR_METADATA)

    ecg_subjects = get_ecg_subjects()
    cxr_subjects = get_cxr_subjects()

    intersection = ecg_subjects & cxr_subjects
    ecg_only = ecg_subjects - cxr_subjects
    cxr_only = cxr_subjects - ecg_subjects

    print("\n===== Patient counts =====")
    print("Number of ECG patients:", len(ecg_subjects))
    print("Number of CXR patients:", len(cxr_subjects))
    print("Number of patients with both ECG and CXR:", len(intersection))
    print("Number of ECG-only patients:", len(ecg_only))
    print("Number of CXR-only patients:", len(cxr_only))

    print("\n===== Is CXR smaller than ECG? =====")
    print("CXR patients < ECG patients:", len(cxr_subjects) < len(ecg_subjects))

    print("\n===== Percentages =====")
    print("CXR patients / ECG patients:", len(cxr_subjects) / len(ecg_subjects))
    print("Matched patients / ECG patients:", len(intersection) / len(ecg_subjects))
    print("Matched patients / CXR patients:", len(intersection) / len(cxr_subjects))

    print("\nFirst 10 ECG-only patients:")
    for subject_id in list(ecg_only)[:10]:
        print(subject_id)

    print("\nFirst 10 CXR-only patients:")
    for subject_id in list(cxr_only)[:10]:
        print(subject_id)


if __name__ == "__main__":
    main()
