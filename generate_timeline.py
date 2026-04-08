#!/usr/bin/env python3
"""Generate a publication timeline plot from the README paper tables."""

import re
from collections import Counter, defaultdict
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

README_PATH = "README.md"
OUTPUT_PATH = "Assets/paper_timeline.png"


def parse_papers_from_readme(path):
    """Extract (date_str, category) tuples from all markdown tables in README."""
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    # Split into sections by <details> blocks and headings
    # We want to map each table to its category (LLM/VLM/MLLM/DM/WM) and type (Gen/Analysis)
    section_pattern = re.compile(
        r'<summary><strong>(.*?)</strong></summary>\s*\n\s*\n'
        r'\|.*?\n\|.*?\n((?:\|.*\n)*)',
        re.MULTILINE
    )

    papers = []
    for match in section_pattern.finditer(content):
        section_name = match.group(1).strip()
        table_content = match.group(2)

        # Determine category
        if "LLM" in section_name and "VLM" not in section_name and "MLLM" not in section_name:
            category = "LLM"
        elif "MLLM" in section_name:
            category = "MLLM"
        elif "VLM" in section_name:
            category = "VLM"
        elif "Diffusion" in section_name:
            category = "Diffusion Model"
        elif "World" in section_name:
            category = "World Model"
        else:
            category = "Other"

        # Determine type
        if "Generation" in section_name:
            task_type = "Generation"
        elif "Analysis" in section_name:
            task_type = "Analysis"
        else:
            task_type = "Other"

        # Extract dates from table rows
        for row in table_content.strip().split("\n"):
            cols = row.split("|")
            if len(cols) >= 4:
                date_str = cols[2].strip()
                # Match YYYY-MM format
                date_match = re.match(r'(\d{4})-(\d{2})', date_str)
                if date_match:
                    papers.append({
                        "date": date_str,
                        "year": int(date_match.group(1)),
                        "month": int(date_match.group(2)),
                        "category": category,
                        "task": task_type,
                    })

    return papers


def generate_timeline_plot(papers, output_path):
    """Generate a stacked bar chart showing paper counts per month by FM category."""

    # Create year-month keys and count by category
    categories = ["LLM", "VLM", "MLLM", "Diffusion Model", "World Model"]
    colors = {
        "LLM": "#4ECDC4",
        "VLM": "#FF6B6B",
        "MLLM": "#45B7D1",
        "Diffusion Model": "#96CEB4",
        "World Model": "#FFEAA7",
    }

    # Find date range
    min_year, min_month = min((p["year"], p["month"]) for p in papers)
    max_year, max_month = max((p["year"], p["month"]) for p in papers)

    # Generate all months in range
    months = []
    y, m = min_year, min_month
    while (y, m) <= (max_year, max_month):
        months.append((y, m))
        m += 1
        if m > 12:
            m = 1
            y += 1

    month_labels = [f"{y}-{m:02d}" for y, m in months]

    # Count papers per month per category
    counts = {cat: [] for cat in categories}
    for ym in months:
        for cat in categories:
            count = sum(1 for p in papers if (p["year"], p["month"]) == ym and p["category"] == cat)
            counts[cat].append(count)

    # Also compute cumulative total
    total_per_month = [sum(counts[cat][i] for cat in categories) for i in range(len(months))]
    cumulative = np.cumsum(total_per_month)

    # Plot
    fig, ax1 = plt.subplots(figsize=(16, 6))

    x = np.arange(len(months))
    bar_width = 0.8

    # Stacked bars
    bottom = np.zeros(len(months))
    for cat in categories:
        values = np.array(counts[cat])
        ax1.bar(x, values, bar_width, bottom=bottom, label=cat, color=colors[cat], edgecolor="white", linewidth=0.3)
        bottom += values

    ax1.set_xlabel("Date", fontsize=12)
    ax1.set_ylabel("Number of Papers (per month)", fontsize=12, color="black")

    # X-axis: show every 3rd month label
    tick_positions = list(range(0, len(months), 3))
    ax1.set_xticks(tick_positions)
    ax1.set_xticklabels([month_labels[i] for i in tick_positions], rotation=45, ha="right", fontsize=9)

    # Cumulative line on secondary axis
    ax2 = ax1.twinx()
    ax2.plot(x, cumulative, color="#E74C3C", linewidth=2.5, marker="", label="Cumulative Total")
    ax2.set_ylabel("Cumulative Number of Papers", fontsize=12, color="#E74C3C")
    ax2.tick_params(axis="y", labelcolor="#E74C3C")

    # Add total count annotation at the end
    ax2.annotate(
        f"{int(cumulative[-1])} papers",
        xy=(x[-1], cumulative[-1]),
        xytext=(x[-1] - 3, cumulative[-1] + 15),
        fontsize=11,
        fontweight="bold",
        color="#E74C3C",
        arrowprops=dict(arrowstyle="->", color="#E74C3C", lw=1.5),
    )

    # Legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=10, framealpha=0.9)

    ax1.set_title("Publication Timeline: Foundation Models for AD Scenario Generation & Analysis", fontsize=14, fontweight="bold", pad=15)
    ax1.grid(axis="y", alpha=0.3)
    ax1.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved timeline plot to {output_path}")
    print(f"Total papers: {int(cumulative[-1])}")

    # Print summary
    for cat in categories:
        total = sum(counts[cat])
        print(f"  {cat}: {total}")


if __name__ == "__main__":
    papers = parse_papers_from_readme(README_PATH)
    print(f"Parsed {len(papers)} papers from README")
    generate_timeline_plot(papers, OUTPUT_PATH)
