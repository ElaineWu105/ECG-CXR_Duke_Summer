import csv
import html
from pathlib import Path

INPUT_CSV = "cxr_ecg_pair_counts_summary.csv"
OUT_HTML = "cxr_ecg_pair_count_bar_charts.html"


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


def make_bar_chart(title, subtitle, data, value_key, color="#4C78A8"):
    """
    Create a simple SVG vertical bar chart.
    No external packages required.
    """
    width = 900
    height = 460
    margin_left = 80
    margin_right = 30
    margin_top = 70
    margin_bottom = 70

    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    max_val = max(d[value_key] for d in data)
    if max_val == 0:
        max_val = 1

    # Add headroom
    y_max = max_val * 1.12

    bar_gap = 18
    n_bars = len(data)
    bar_width = (plot_width - bar_gap * (n_bars - 1)) / n_bars

    svg = []
    svg.append(f'<h2>{html.escape(title)}</h2>')
    svg.append(f'<p class="subtitle">{html.escape(subtitle)}</p>')
    svg.append(f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}">')

    # Background
    svg.append(f'<rect x="0" y="0" width="{width}" height="{height}" fill="white"/>')

    # Axes
    x0 = margin_left
    y0 = margin_top + plot_height
    svg.append(f'<line x1="{x0}" y1="{margin_top}" x2="{x0}" y2="{y0}" stroke="#333" stroke-width="1"/>')
    svg.append(f'<line x1="{x0}" y1="{y0}" x2="{margin_left + plot_width}" y2="{y0}" stroke="#333" stroke-width="1"/>')

    # Y grid/ticks
    n_ticks = 5
    for i in range(n_ticks + 1):
        val = y_max * i / n_ticks
        y = y0 - (val / y_max) * plot_height

        svg.append(f'<line x1="{x0}" y1="{y}" x2="{margin_left + plot_width}" y2="{y}" stroke="#eee" stroke-width="1"/>')
        svg.append(
            f'<text x="{x0 - 10}" y="{y + 4}" text-anchor="end" '
            f'font-size="12" fill="#555">{int(val):,}</text>'
        )

    # Bars
    for i, d in enumerate(data):
        x = margin_left + i * (bar_width + bar_gap)
        bar_h = (d[value_key] / y_max) * plot_height
        y = y0 - bar_h

        svg.append(
            f'<rect x="{x}" y="{y}" width="{bar_width}" height="{bar_h}" '
            f'rx="4" fill="{color}"/>'
        )

        # Value label
        svg.append(
            f'<text x="{x + bar_width/2}" y="{y - 8}" text-anchor="middle" '
            f'font-size="12" fill="#333">{d[value_key]:,}</text>'
        )

        # X label
        svg.append(
            f'<text x="{x + bar_width/2}" y="{y0 + 25}" text-anchor="middle" '
            f'font-size="13" fill="#333">n={d["n"]}</text>'
        )

    # Axis labels
    svg.append(
        f'<text x="{margin_left + plot_width/2}" y="{height - 20}" '
        f'text-anchor="middle" font-size="14" fill="#333">n hours</text>'
    )
    svg.append(
        f'<text x="20" y="{margin_top + plot_height/2}" '
        f'text-anchor="middle" font-size="14" fill="#333" '
        f'transform="rotate(-90 20 {margin_top + plot_height/2})">Pair count</text>'
    )

    svg.append('</svg>')
    return "\n".join(svg)


def make_grouped_bar_chart(title, subtitle, data):
    """
    Grouped bar chart for criterion2 nearest / earliest / all_t1.
    """
    width = 1000
    height = 500
    margin_left = 80
    margin_right = 40
    margin_top = 80
    margin_bottom = 80

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

    svg = []
    svg.append(f'<h2>{html.escape(title)}</h2>')
    svg.append(f'<p class="subtitle">{html.escape(subtitle)}</p>')
    svg.append(f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}">')
    svg.append(f'<rect x="0" y="0" width="{width}" height="{height}" fill="white"/>')

    x0 = margin_left
    y0 = margin_top + plot_height

    # Axes
    svg.append(f'<line x1="{x0}" y1="{margin_top}" x2="{x0}" y2="{y0}" stroke="#333" stroke-width="1"/>')
    svg.append(f'<line x1="{x0}" y1="{y0}" x2="{margin_left + plot_width}" y2="{y0}" stroke="#333" stroke-width="1"/>')

    # Y ticks
    n_ticks = 5
    for i in range(n_ticks + 1):
        val = y_max * i / n_ticks
        y = y0 - (val / y_max) * plot_height
        svg.append(f'<line x1="{x0}" y1="{y}" x2="{margin_left + plot_width}" y2="{y}" stroke="#eee" stroke-width="1"/>')
        svg.append(
            f'<text x="{x0 - 10}" y="{y + 4}" text-anchor="end" '
            f'font-size="12" fill="#555">{int(val):,}</text>'
        )

    # Bars
    for i, d in enumerate(data):
        group_x = margin_left + i * (group_width + group_gap)

        for j, (key, label, color) in enumerate(keys):
            x = group_x + j * (bar_width + inner_gap)
            bar_h = (d[key] / y_max) * plot_height
            y = y0 - bar_h

            svg.append(
                f'<rect x="{x}" y="{y}" width="{bar_width}" height="{bar_h}" '
                f'rx="3" fill="{color}"/>'
            )

            svg.append(
                f'<text x="{x + bar_width/2}" y="{y - 5}" text-anchor="middle" '
                f'font-size="10" fill="#333">{d[key]:,}</text>'
            )

        # X label
        svg.append(
            f'<text x="{group_x + group_width/2}" y="{y0 + 25}" text-anchor="middle" '
            f'font-size="13" fill="#333">n={d["n"]}</text>'
        )

    # Legend
    legend_x = margin_left
    legend_y = 25
    for idx, (_, label, color) in enumerate(keys):
        lx = legend_x + idx * 150
        svg.append(f'<rect x="{lx}" y="{legend_y}" width="14" height="14" fill="{color}"/>')
        svg.append(f'<text x="{lx + 20}" y="{legend_y + 12}" font-size="13" fill="#333">{html.escape(label)}</text>')

    # Axis labels
    svg.append(
        f'<text x="{margin_left + plot_width/2}" y="{height - 25}" '
        f'text-anchor="middle" font-size="14" fill="#333">n hours</text>'
    )
    svg.append(
        f'<text x="20" y="{margin_top + plot_height/2}" '
        f'text-anchor="middle" font-size="14" fill="#333" '
        f'transform="rotate(-90 20 {margin_top + plot_height/2})">Pair count</text>'
    )

    svg.append('</svg>')
    return "\n".join(svg)


def main():
    data = load_summary(INPUT_CSV)

    html_parts = []
    html_parts.append("""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>CXR-ECG Pair Count Bar Charts</title>
<style>
body {
    font-family: Arial, sans-serif;
    margin: 32px;
    background: #fafafa;
    color: #222;
}
.chart-card {
    background: white;
    border: 1px solid #ddd;
    border-radius: 12px;
    padding: 24px;
    margin-bottom: 32px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}
h1 {
    margin-bottom: 8px;
}
h2 {
    margin-top: 0;
    margin-bottom: 4px;
}
.subtitle {
    color: #666;
    margin-top: 0;
    margin-bottom: 16px;
}
.note {
    color: #555;
    font-size: 14px;
    line-height: 1.5;
}
</style>
</head>
<body>
<h1>CXR–ECG Pair Count Bar Charts</h1>
<p class="note">
These charts show the basic pair-count results by n. They are based on
cxr_ecg_pair_counts_summary.csv. The y-axis is the number of CXR–ECG pairs.
</p>
""")

    # Criterion 1
    html_parts.append('<div class="chart-card">')
    html_parts.append(make_bar_chart(
        title="Criterion 1: CXR–ECG pair count by n",
        subtitle="CXR at time t; ECG in [t-12-n, t-n].",
        data=data,
        value_key="criterion1",
        color="#4C78A8"
    ))
    html_parts.append('</div>')

    # Criterion 2 nearest
    html_parts.append('<div class="chart-card">')
    html_parts.append(make_bar_chart(
        title="Criterion 2 nearest t1: pair count by n",
        subtitle="For each t2, choose the nearest prior CXR t1 within [t2-24h, t2).",
        data=data,
        value_key="criterion2_nearest",
        color="#4C78A8"
    ))
    html_parts.append('</div>')

    # Criterion 2 earliest
    html_parts.append('<div class="chart-card">')
    html_parts.append(make_bar_chart(
        title="Criterion 2 earliest t1: pair count by n",
        subtitle="For each t2, choose the earliest prior CXR t1 within [t2-24h, t2), closest to t2-24h.",
        data=data,
        value_key="criterion2_earliest",
        color="#F58518"
    ))
    html_parts.append('</div>')

    # Criterion 2 grouped comparison
    html_parts.append('<div class="chart-card">')
    html_parts.append(make_grouped_bar_chart(
        title="Criterion 2 comparison: nearest vs earliest vs all t1",
        subtitle="Grouped bar chart comparing the three interpretations of t1 selection."
    , data=data))
    html_parts.append('</div>')

    html_parts.append("""
</body>
</html>
""")

    Path(OUT_HTML).write_text("\n".join(html_parts), encoding="utf-8")

    print("Saved:", OUT_HTML)
    print("Open this file in VS Code or a browser:")
    print(str(Path(OUT_HTML).resolve()))


if __name__ == "__main__":
    main()
