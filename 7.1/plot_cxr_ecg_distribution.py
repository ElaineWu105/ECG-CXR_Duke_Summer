import csv
import os
import matplotlib.pyplot as plt

INPUT_CSV = "cxr_ecg_pair_distribution_wide.csv"
OUT_DIR = "figures_cxr_ecg_distribution"

os.makedirs(OUT_DIR, exist_ok=True)

# You can change this if you want to show more ECG-count bins
MAX_ECG_BIN_TO_SHOW = 6

# Main criteria to plot
CRITERIA_TO_PLOT = [
    "criterion1",
    "criterion2_earliest",
    "criterion2_nearest",
    "criterion2_all_t1",
]


def load_rows(path):
    rows = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def get_ecg_columns(row):
    cols = []
    for k in row.keys():
        if k.startswith("ECG_"):
            num = int(k.replace("ECG_", ""))
            cols.append((num, k))
    cols.sort()
    return cols


def make_distribution_for_row(row, max_bin):
    """
    Return labels and values.

    ECG_0, ECG_1, ..., ECG_max_bin, ECG_>max_bin
    """
    ecg_cols = get_ecg_columns(row)

    labels = []
    values = []

    more_than_max = 0

    for num, col in ecg_cols:
        val = int(row[col])

        if num <= max_bin:
            labels.append(str(num))
            values.append(val)
        else:
            more_than_max += val

    if more_than_max > 0:
        labels.append(f">{max_bin}")
        values.append(more_than_max)

    return labels, values


def plot_single_row(row):
    criterion = row["criterion"]
    n = row["n_hours"]

    labels, values = make_distribution_for_row(row, MAX_ECG_BIN_TO_SHOW)

    plt.figure(figsize=(9, 5))
    plt.bar(labels, values)

    plt.xlabel("Number of matched ECG records per CXR/window")
    plt.ylabel("Number of CXR studies / windows")
    plt.title(f"{criterion}, n={n}: ECG matches per CXR/window")

    plt.tight_layout()

    out_path = os.path.join(OUT_DIR, f"{criterion}_n{n}_distribution.png")
    plt.savefig(out_path, dpi=200)
    plt.close()

    return out_path


def plot_all_n_for_criterion(rows, criterion):
    """
    For one criterion, make one figure per n.
    """
    out_paths = []

    criterion_rows = [r for r in rows if r["criterion"] == criterion]
    criterion_rows.sort(key=lambda r: int(r["n_hours"]))

    for row in criterion_rows:
        out_paths.append(plot_single_row(row))

    return out_paths


def plot_compact_summary(rows, criterion):
    """
    Plot a compact summary across n:
    x-axis = n
    y-axis = number of CXR/windows with exactly 1, 2, 3, >3 ECGs.

    This is often easier to include in a report/email than many separate figures.
    """
    criterion_rows = [r for r in rows if r["criterion"] == criterion]
    criterion_rows.sort(key=lambda r: int(r["n_hours"]))

    n_values = [int(r["n_hours"]) for r in criterion_rows]

    series = {
        "1 ECG": [],
        "2 ECG": [],
        "3 ECG": [],
        ">3 ECG": [],
    }

    for row in criterion_rows:
        ecg_1 = int(row.get("ECG_1", 0))
        ecg_2 = int(row.get("ECG_2", 0))
        ecg_3 = int(row.get("ECG_3", 0))

        gt3 = 0
        for k, v in row.items():
            if k.startswith("ECG_"):
                num = int(k.replace("ECG_", ""))
                if num > 3:
                    gt3 += int(v)

        series["1 ECG"].append(ecg_1)
        series["2 ECG"].append(ecg_2)
        series["3 ECG"].append(ecg_3)
        series[">3 ECG"].append(gt3)

    plt.figure(figsize=(9, 5))

    for label, vals in series.items():
        plt.plot(n_values, vals, marker="o", label=label)

    plt.xlabel("n hours")
    plt.ylabel("Number of CXR studies / windows")
    plt.title(f"{criterion}: distribution of matched ECG count across n")
    plt.legend()
    plt.tight_layout()

    out_path = os.path.join(OUT_DIR, f"{criterion}_compact_across_n.png")
    plt.savefig(out_path, dpi=200)
    plt.close()

    return out_path


def main():
    rows = load_rows(INPUT_CSV)

    print("Loaded rows:", len(rows))
    print("Output directory:", OUT_DIR)

    all_outputs = []

    for criterion in CRITERIA_TO_PLOT:
        criterion_rows = [r for r in rows if r["criterion"] == criterion]

        if not criterion_rows:
            print("Skipping missing criterion:", criterion)
            continue

        print("\nPlotting:", criterion)

        # One compact summary plot across n
        out = plot_compact_summary(rows, criterion)
        print("Saved:", out)
        all_outputs.append(out)

        # Separate distribution plots for each n
        out_paths = plot_all_n_for_criterion(rows, criterion)
        for p in out_paths:
            print("Saved:", p)
        all_outputs.extend(out_paths)

    print("\nDone.")
    print("Generated", len(all_outputs), "figures.")


if __name__ == "__main__":
    main()
