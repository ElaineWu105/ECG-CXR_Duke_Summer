import csv
from collections import defaultdict, Counter
from pathlib import Path

# MIMIC-CXR metadata path
metadata_path = Path.home() / "mimic_cxr_jpg" / "mimic-cxr-2.0.0-metadata.csv"

print("CXR metadata:", metadata_path)

# patient_id -> set of study_id
patient_studies = defaultdict(set)

with open(metadata_path, "r", newline="") as f:
    reader = csv.DictReader(f)

    print("\nMetadata columns:")
    print(reader.fieldnames)

    for row in reader:
        subject_id = row["subject_id"]
        study_id = row["study_id"]
        patient_studies[subject_id].add(study_id)

patient_cxr_counts = {
    subject_id: len(studies)
    for subject_id, studies in patient_studies.items()
}

distribution = Counter(patient_cxr_counts.values())

print("\n===== Summary =====")
print("Number of patients with at least one CXR study:", len(patient_cxr_counts))
print("Total unique CXR studies:", sum(patient_cxr_counts.values()))

print("\n===== Distribution: number of CXR studies per patient =====")
print("cxr_study_count,num_patients")

for cxr_count in sorted(distribution):
    print(f"{cxr_count},{distribution[cxr_count]}")

with open("cxr_count_per_patient.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["subject_id", "num_cxr_studies"])

    for subject_id, count in sorted(patient_cxr_counts.items()):
        writer.writerow([subject_id, count])

with open("cxr_count_distribution.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["num_cxr_studies", "num_patients"])

    for cxr_count in sorted(distribution):
        writer.writerow([cxr_count, distribution[cxr_count]])

print("\nSaved files:")
print("  cxr_count_per_patient.csv")
print("  cxr_count_distribution.csv")
