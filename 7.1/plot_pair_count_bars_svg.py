import csv
import html
from pathlib import Path

INPUT_CSV = "cxr_ecg_pair_counts_summary.csv"
OUT_DIR = Path("pair_count_bar_figures")
OUT_DIR.mkdir(exist_ok=True)


def load_summary(path):
    rows = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "n": int(row["n_hours"]),
                "criterion1": int(row["criterion1_pair_count"]),
                "criterion2_nearest": int(row["criterion2_nearest_pair_count"]),
                "criterion2_earliest": int(row["criterion2_earliest_pair_count"]),
                "criterion2_all_t1": int(row["criterion2_all_t1_pair_or_triple_count"]),
            })
    return rows


def make_single_bar_svg(title, subtitle, data, value_key):
    width = 900
    height = 500
    margin_left = 85
    margin_right = 35
    margin_top = 90
    margin_bottom = 75

    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    max_val = max(d[value_key] for d in data)
    y_max = max_val * 1.15 if max_val > 0 else 1

    bar_gap = 18
    n_bars = len(data)
    bar_width = (plot_width - bar_gap * (n_bars - 1)) / n_bars

    x0 = margin_left
    y0 = margin_top + plot_height

    svg = []
    svg.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">')
    svg.append(f'<rect width="{width}" height="{height}" fill="white"/>')

    # Title
    svg.append(f'<text x="{width/2}" y="32" text-anchor="middle" font-size="22" font-family="Arial" font-weight="bold">{html.escape(title)}</text>')
    svg.append(f'<text x="{width/2}" y="58" text-anchor="middle" font-size="14" font-family="Arial" fill="#555">{html.escape(subtitle)}</text>')

    # Axes
    svg.append(f'<line x1="{x0}" y1="{margin_top}" x2="{x0}" y2="{y0}" stroke="#333" stroke-width="1"/>')
    svg.append(f'<line x1="{x0}" y1="{y0}" x2="{margin_left + plot_width}" y2="{y0}" stroke="#333" stroke-width="1"/>')

    # Y ticks
    for i in range(6):
        val = y_max * i / 5
        y = y0 - (val / y_max) * plot_height
        svg.append(f'<line x1="{x0}" y1="{y}" x2="{margin_left + plot_width}" y2="{y}" stroke="#e8e8e8" stroke-width="1"/>')
        svg.append(f'<text x="{x0 - 10}" y="{y + 4}" text-anchor="end" font-size="12" font-family="Arial" fill="#555">{int(val):,}</text>')

    # Bars
    for i, d in enumerate(data):
        x = margin_left + i * (bar_width + bar_gap)
        value = d[value_key]
        bar_h = (value / y_max) * plot_height
        y = y0 - bar_h

        svg.append(f'<rect x="{x}" y="{y}" width="{bar_width}" height="{bar_h}" rx="4" fill="#4C78A8"/>')
        svg.append(f'<text x="{x + bar_width/2}" y="{y - 8}" text-anchor="middle" font-size="12" font-family="Arial" fill="#333">{value:,}</text>')
        svg.append(f'<text x="{x + bar_width/2}" y="{y0 + 25}" text-anchor="middle" font-size="13" font-family="Arial" fill="#333">n={d["n"]}</text>')

    # Axis labels
    svg.append(f'<text x="{margin_left + plot_width/2}" y="{height - 22}" text-anchor="middle" font-size="14" font-family="Arial" fill="#333">n hours</text>')
    svg.append(f'<text x="22" y="{margin_top + plot_height/2}" text-anchor="middle" font-size="14" font-family="Arial" fill="#333" transform="rotate(-90 22 {margin_top + plot_height/2})">Pair count</text>')

    svg.append('</svg>')
    return "\n".join(svg)


def make_grouped_svg(title, subtitle, data):
    width = 1050
    height = 540
    margin_left = 85
    margin_right = 40
    margin_top = 95
    margin_bottom = 85

    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    keys = [
        ("criterion2_nearest", "nearest t1", "#4C78A8"),
        ("criterion2_earliest", "earliest t1", "#F58518"),
        ("criterion2_all_t1", "all t1", "#54A24B"),
    ]

    max_val = max(d[k] for d in data for k, _, _ in keys)
    y_max = max_val * 1.15 if max_val > 0 else 1

    n_groups = len(data)
    group_gap = 26
    group_width = (plot_width - group_gap * (n_groups - 1)) / n_groups
    inner_gap = 5
    bar_width = (group_width - inner_gap * (len(keys) - 1)) / len(keys)

    x0 = margin_left
    y0 = margin_top + plot_height

    svg = []
    svg.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">')
    svg.append(f'<rect width="{width}" height="{height}" fill="white"/>')

    svg.append(f'<text x="{width/2}" y="32" text-anchor="middle" font-size="22" font-family="Arial" font-weight="bold">{html.escape(title)}</text>')
    svg.append(f'<text x="{width/2}" y="58" text-anchor="middle" font-size="14" font-family="Arial" fill="#555">{html.escape(subtitle)}</text>')

    # Legend
    for idx, (_, label, color) in enumerate(keys):
        lx = margin_left + idx * 160
        ly = 75
        svg.append(f'<rect x="{lx}" y="{ly}" width="14" height="14" fill="{color}"/>')
        svg.append(f'<text x="{lx + 20}" y="{ly + 12}" font-size="13" font-family="Arial" fill="#333">{label}</text>')

    # Axes
    svg.append(f'<line x1="{x0}" y1="{margin_top}" x2="{x0}" y2="{y0}" stroke="#333" stroke-width="1"/>')
    svg.append(f'<line x1="{x0}" y1="{y0}" x2="{margin_left + plot_width}" y2="{y0}" stroke="#333" stroke-width="1"/>')

    # Y ticks
    for i in range(6):
        val = y_max * i / 5
        y = y0 - (val / y_max) * plot_height
        svg.append(f'<line x1="{x0}" y1="{y}" x2="{margin_left + plot_width}" y2="{y}" stroke="#e8e8e8" stroke-width="1"/>')
        svg.append(f'<text x="{x0 - 10}" y="{y + 4}" text-anchor="end" font-size="12" font-family="Arial" fill="#555">{int(val):,}</text>')

    # Bars
    for i, d in enumerate(data):
        group_x = margin_left + i * (group_width + group_gap)

        for j, (key, label, color) in enumerate(keys):
            x = group_x + j * (bar_width + inner_gap)
            value = d[key]
            bar_h = (value / y_max) * plot_height
            y = y0 - bar_h

            svg.append(f'<rect x="{x}" y="{y}" width="{bar_width}" height="{bar_h}" rx="3" fill="{color}"/>')
            svg.append(f'<text x="{x + bar_width/2}" y="{y - 5}" text-anchor="middle" font-size="10" font-family="Arial" fill="#333">{value:,}</text>')

        svg.append(f'<text x="{group_x + group_width/2}" y="{y0 + 25}" text-anchor="middle" font-size="13" font-family="Arial" fill="#333">n={d["n"]}</text>')

    svg.append(f'<text x="{margin_left + plot_width/2}" y="{height - 25}" text-anchor="middle" font-size="14" font-family="Arial" fill="#333">n hours</text>')
    svg.append(f'<text x="22" y="{margin_top + plot_height/2}" text-anchor="middle" font-size="14" font-family="Arial" fill="#333" transform="rotate(-90 22 {margin_top + plot_height/2})">Pair count</text>')

    svg.append('</svg>')
    return "\n".join(svg)


def main():
    data = load_summary(INPUT_CSV)

    charts = [
        (
            "criterion1_pair_count_by_n.svg",
            make_single_bar_svg(
                "Criterion 1: CXR–ECG pair count by n",
                "CXR at time t; ECG in [t-12-n, t-n].",
                data,
                "criterion1"
            )
        ),
        (
            "criterion2_nearest_pair_count_by_n.svg",
            make_single_bar_svg(
                "Criterion 2 nearest t1: pair count by n",
                "For each t2, choose the nearest prior CXR t1 within [t2-24h, t2).",
                data,
                "criterion2_nearest"
            )
        ),
        (
            "criterion2_earliest_pair_count_by_n.svg",
            make_single_bar_svg(
                "Criterion 2 earliest t1: pair count by n",
                "For each t2, choose the earliest prior CXR t1 within [t2-24h, t2).",
                data,
                "criterion2_earliest"
            )
        ),
        (
            "criterion2_three_versions_pair_count_by_n.svg",
            make_grouped_svg(
                "Criterion 2: pair count comparison by n",
                "Nearest t1, earliest t1, and all-t1 interpretations.",
                data
            )
        ),
    ]

    for filename, svg in charts:
        out_path = OUT_DIR / filename
        out_path.write_text(svg, encoding="utf-8")
        print("Saved:", out_path)

    print("\nDone. Open the SVG files in VS Code preview:")
    print(OUT_DIR.resolve())


if __name__ == "__main__":
    main()
