import csv
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from bisect import bisect_left, bisect_right

CXR_TIMES_CSV = "cxr_study_times.csv"
ECG_TIMES_CSV = "ecg_record_times.csv"

OUT_DISTRIBUTION_LONG = "cxr_ecg_pair_distribution_long.csv"
OUT_DISTRIBUTION_WIDE = "cxr_ecg_pair_distribution_wide.csv"

N_VALUES = [0, 2, 4, 6, 8, 10, 12]


def parse_dt(x):
    return datetime.fromisoformat(x)


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
# Load CXR study times
# ============================================================

print("===== Loading CXR study times =====")

cxr_by_subject = defaultdict(list)

with open(CXR_TIMES_CSV, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        subject_id = row["subject_id"]
        cxr_time = parse_dt(row["cxr_time"])
        cxr_by_subject[subject_id].append(cxr_time)

for subject_id in cxr_by_subject:
    cxr_by_subject[subject_id].sort()

print("CXR patients:", len(cxr_by_subject))
print("Total CXR studies:", sum(len(v) for v in cxr_by_subject.values()))


# ============================================================
# Load ECG record times
# ============================================================

print("\n===== Loading ECG record times =====")

ecg_by_subject = defaultdict(list)

with open(ECG_TIMES_CSV, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        subject_id = row["subject_id"]
        ecg_time = parse_dt(row["ecg_time"])
        ecg_by_subject[subject_id].append(ecg_time)

for subject_id in ecg_by_subject:
    ecg_by_subject[subject_id].sort()

print("ECG patients:", len(ecg_by_subject))
print("Total ECG records:", sum(len(v) for v in ecg_by_subject.values()))


# ============================================================
# Common patients
# ============================================================

common_subjects = sorted(set(cxr_by_subject.keys()) & set(ecg_by_subject.keys()))

print("\n===== Common cohort =====")
print("Patients with both CXR and ECG:", len(common_subjects))
print("CXR studies among common patients:", sum(len(cxr_by_subject[s]) for s in common_subjects))
print("ECG records among common patients:", sum(len(ecg_by_subject[s]) for s in common_subjects))


# ============================================================
# Distribution containers
# ============================================================

# Each distribution is:
# distribution[criterion_name][n][num_matched_ecg] = number of CXR/windows
distribution = defaultdict(lambda: {n: Counter() for n in N_VALUES})

# Criterion 2 eligible window counts
criterion2_nearest_eligible = 0
criterion2_earliest_eligible = 0
criterion2_all_t1_windows = 0


# ============================================================
# Count distributions
# ============================================================

print("\n===== Counting ECG-per-CXR distributions =====")

for subject_id in common_subjects:
    cxr_times = cxr_by_subject[subject_id]
    ecg_times = ecg_by_subject[subject_id]

    # --------------------------------------------------------
    # Criterion 1:
    # CXR at time t,
    # ECG in [t - 12 - n, t - n]
    # Unit counted in distribution:
    # one CXR study
    # --------------------------------------------------------
    for t in cxr_times:
        for n in N_VALUES:
            start = t - timedelta(hours=12 + n)
            end = t - timedelta(hours=n)

            c = count_times_in_window(ecg_times, start, end)

            distribution["criterion1"][n][c] += 1

    # --------------------------------------------------------
    # Criterion 2:
    # CXR at t2,
    # prior CXR t1 in [t2 - 24, t2),
    # ECG in [max(t2 - 12 - n, t1), t2 - n]
    #
    # We compute:
    # 1. nearest prior t1
    # 2. earliest prior t1 within 24h
    # 3. all prior t1 windows
    # --------------------------------------------------------
    for t2 in cxr_times:
        prev_start = t2 - timedelta(hours=24)

        left = bisect_left(cxr_times, prev_start)
        right = bisect_left(cxr_times, t2)  # exclude current t2

        previous_t1_list = cxr_times[left:right]

        if not previous_t1_list:
            continue

        # -------------------------
        # Criterion 2A: nearest prior CXR
        # Unit counted:
        # one eligible t2 CXR study
        # -------------------------
        nearest_t1 = previous_t1_list[-1]
        criterion2_nearest_eligible += 1

        for n in N_VALUES:
            start = max(t2 - timedelta(hours=12 + n), nearest_t1)
            end = t2 - timedelta(hours=n)

            c = count_times_in_window(ecg_times, start, end)

            distribution["criterion2_nearest"][n][c] += 1

        # -------------------------
        # Criterion 2B: earliest prior CXR within 24h
        # Unit counted:
        # one eligible t2 CXR study
        # -------------------------
        earliest_t1 = previous_t1_list[0]
        criterion2_earliest_eligible += 1

        for n in N_VALUES:
            start = max(t2 - timedelta(hours=12 + n), earliest_t1)
            end = t2 - timedelta(hours=n)

            c = count_times_in_window(ecg_times, start, end)

            distribution["criterion2_earliest"][n][c] += 1

        # -------------------------
        # Criterion 2C: all prior CXR t1 windows
        # Unit counted:
        # one t1-t2 window
        # -------------------------
        for t1 in previous_t1_list:
            criterion2_all_t1_windows += 1

            for n in N_VALUES:
                start = max(t2 - timedelta(hours=12 + n), t1)
                end = t2 - timedelta(hours=n)

                c = count_times_in_window(ecg_times, start, end)

                distribution["criterion2_all_t1"][n][c] += 1


print("Criterion 2 nearest eligible t2 CXR count:", criterion2_nearest_eligible)
print("Criterion 2 earliest eligible t2 CXR count:", criterion2_earliest_eligible)
print("Criterion 2 all-t1 CXR-CXR window count:", criterion2_all_t1_windows)


# ============================================================
# Save long-format distribution
# ============================================================

with open(OUT_DISTRIBUTION_LONG, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "criterion",
        "n_hours",
        "num_matched_ecg",
        "num_cxr_or_windows"
    ])

    for criterion in sorted(distribution.keys()):
        for n in N_VALUES:
            counter = distribution[criterion][n]
            for num_matched_ecg in sorted(counter.keys()):
                writer.writerow([
                    criterion,
                    n,
                    num_matched_ecg,
                    counter[num_matched_ecg]
                ])

print("\nSaved:", OUT_DISTRIBUTION_LONG)


# ============================================================
# Save wide-format distribution
# Easier to read in terminal.
# Rows = criterion + n
# Columns = ECG_0, ECG_1, ECG_2, ...
# ============================================================

max_k = 0
for criterion in distribution:
    for n in N_VALUES:
        if distribution[criterion][n]:
            max_k = max(max_k, max(distribution[criterion][n].keys()))

with open(OUT_DISTRIBUTION_WIDE, "w", newline="") as f:
    writer = csv.writer(f)

    header = ["criterion", "n_hours"] + [f"ECG_{k}" for k in range(max_k + 1)]
    writer.writerow(header)

    for criterion in sorted(distribution.keys()):
        for n in N_VALUES:
            counter = distribution[criterion][n]
            row = [criterion, n] + [counter.get(k, 0) for k in range(max_k + 1)]
            writer.writerow(row)

print("Saved:", OUT_DISTRIBUTION_WIDE)


# ============================================================
# Print compact summary for ECG >= 1 only
# ============================================================

print("\n===== Compact distribution: CXR/windows with 1, 2, 3, ... matched ECGs =====")

for criterion in sorted(distribution.keys()):
    print(f"\n--- {criterion} ---")
    for n in N_VALUES:
        counter = distribution[criterion][n]

        nonzero_items = [(k, v) for k, v in sorted(counter.items()) if k >= 1]

        total_with_at_least_one = sum(v for k, v in nonzero_items)
        total_pairs = sum(k * v for k, v in nonzero_items)

        print(f"n={n}: units_with_>=1_ECG={total_with_at_least_one}, total_pairs={total_pairs}")

        # Print detailed distribution up to 10 ECGs, then group >10
        shown = []
        more_than_10 = 0

        for k, v in nonzero_items:
            if k <= 10:
                shown.append(f"{k} ECG: {v}")
            else:
                more_than_10 += v

        if more_than_10 > 0:
            shown.append(f">10 ECG: {more_than_10}")

        print("  " + "; ".join(shown))

print("\nDone.")
