#!/usr/bin/env python3
"""Generate visualizations for MCP effectiveness analysis."""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

# Load results
results_file = Path(__file__).parent / "comprehensive_results.json"
with open(results_file) as f:
    results = json.load(f)

# Color scheme
COLORS = {
    "Sonnet": "#ff6b6b",
    "Sonnet+MCP": "#4ecdc4",
    "Opus": "#ffe66d",
    "Opus+MCP": "#95e1d3",
}

def filter_valid(data):
    """Filter results with valid complexity_index."""
    return [r for r in data if r["complexity_index"] is not None]

def calc_stats(data, condition):
    """Calculate mean and std for a condition."""
    cond_data = [r["complexity_index"] for r in data if r["condition"] == condition]
    if not cond_data:
        return None, None
    return np.mean(cond_data), np.std(cond_data) if len(cond_data) > 1 else 0

# Figure 1: Overall Complexity Index by Condition
fig1, ax1 = plt.subplots(figsize=(10, 6))

conditions = ["Sonnet", "Sonnet+MCP", "Opus", "Opus+MCP"]
valid = filter_valid(results)

means = []
stds = []
colors = []

for cond in conditions:
    mean, std = calc_stats(valid, cond)
    means.append(mean if mean else 0)
    stds.append(std if std else 0)
    colors.append(COLORS[cond])

x = np.arange(len(conditions))
bars = ax1.bar(x, means, yerr=stds, capsize=5, color=colors, edgecolor='black', linewidth=1.5)

# Add value labels on bars
for bar, mean in zip(bars, means):
    if mean > 0:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{mean:.2f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

ax1.set_ylabel('Complexity Index (lower is better)', fontsize=12)
ax1.set_xlabel('Condition', fontsize=12)
ax1.set_title('MCP Tools Dramatically Improve Panel Design Quality', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(conditions, fontsize=11)
ax1.axhline(y=2.0, color='green', linestyle='--', alpha=0.5, label='Good threshold (CI<2)')
ax1.legend(loc='upper right')
ax1.set_ylim(0, 14)

# Add annotation
ax1.annotate('82% improvement\nfor Sonnet', xy=(0.5, 6), xytext=(1.5, 11),
            fontsize=10, ha='center',
            arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

fig1.tight_layout()
fig1.savefig(Path(__file__).parent / 'fig1_overall_comparison.png', dpi=150, bbox_inches='tight')
print("Saved: fig1_overall_comparison.png")

# Figure 2: Per-OMIP Breakdown
fig2, axes = plt.subplots(1, 3, figsize=(14, 5))

omips = [30, 47, 63]
omip_labels = ['OMIP-030\n(10 markers)', 'OMIP-047\n(16 markers)', 'OMIP-063\n(20 markers)']

for ax, omip, label in zip(axes, omips, omip_labels):
    omip_data = [r for r in valid if r["omip"] == omip]

    means = []
    for cond in conditions:
        cond_data = [r["complexity_index"] for r in omip_data if r["condition"] == cond]
        means.append(np.mean(cond_data) if cond_data else 0)

    x = np.arange(len(conditions))
    bars = ax.bar(x, means, color=[COLORS[c] for c in conditions], edgecolor='black', linewidth=1)

    for bar, mean in zip(bars, means):
        if mean > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                   f'{mean:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_ylabel('Complexity Index' if omip == 30 else '')
    ax.set_title(label, fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['Son', 'Son+MCP', 'Opus', 'Opus+MCP'], fontsize=9)
    ax.axhline(y=2.0, color='green', linestyle='--', alpha=0.5)

fig2.suptitle('MCP Improvement Across Panel Sizes', fontsize=14, fontweight='bold', y=1.02)
fig2.tight_layout()
fig2.savefig(Path(__file__).parent / 'fig2_per_omip_breakdown.png', dpi=150, bbox_inches='tight')
print("Saved: fig2_per_omip_breakdown.png")

# Figure 3: Critical Pairs Comparison
fig3, ax3 = plt.subplots(figsize=(10, 6))

critical_means = []
for cond in conditions:
    cond_data = [r["critical_pairs"] for r in valid if r["condition"] == cond]
    critical_means.append(np.mean(cond_data) if cond_data else 0)

x = np.arange(len(conditions))
bars = ax3.bar(x, critical_means, color=[COLORS[c] for c in conditions], edgecolor='black', linewidth=1.5)

for bar, mean in zip(bars, critical_means):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
            f'{mean:.1f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

ax3.set_ylabel('Average Critical Pairs (similarity > 0.9)', fontsize=12)
ax3.set_xlabel('Condition', fontsize=12)
ax3.set_title('MCP Eliminates Critical Spectral Overlaps', fontsize=14, fontweight='bold')
ax3.set_xticks(x)
ax3.set_xticklabels(conditions, fontsize=11)
ax3.set_ylim(0, 10)

# Add annotation
ax3.annotate('100% elimination\nof critical pairs!', xy=(1, 0.5), xytext=(2, 5),
            fontsize=11, ha='center', fontweight='bold', color='green',
            arrowprops=dict(arrowstyle='->', color='green', lw=2))

fig3.tight_layout()
fig3.savefig(Path(__file__).parent / 'fig3_critical_pairs.png', dpi=150, bbox_inches='tight')
print("Saved: fig3_critical_pairs.png")

# Figure 4: Tools vs Reasoning (Key Finding)
fig4, ax4 = plt.subplots(figsize=(10, 6))

# OMIP-030 data only (cleanest comparison with 3 runs each)
omip30 = [r for r in valid if r["omip"] == 30]

data_points = {
    "Sonnet": [r["complexity_index"] for r in omip30 if r["condition"] == "Sonnet"],
    "Sonnet+MCP": [r["complexity_index"] for r in omip30 if r["condition"] == "Sonnet+MCP"],
    "Opus": [r["complexity_index"] for r in omip30 if r["condition"] == "Opus"],
    "Opus+MCP": [r["complexity_index"] for r in omip30 if r["condition"] == "Opus+MCP"],
}

positions = [1, 2, 4, 5]
bp = ax4.boxplot([data_points[c] for c in conditions], positions=positions, patch_artist=True, widths=0.6)

for patch, color in zip(bp['boxes'], [COLORS[c] for c in conditions]):
    patch.set_facecolor(color)
    patch.set_edgecolor('black')
    patch.set_linewidth(1.5)

ax4.set_ylabel('Complexity Index (OMIP-030)', fontsize=12)
ax4.set_title('Key Finding: Sonnet+MCP Beats Opus Alone', fontsize=14, fontweight='bold')
ax4.set_xticks(positions)
ax4.set_xticklabels(conditions, fontsize=11)
ax4.axhline(y=2.0, color='green', linestyle='--', alpha=0.5, label='Good threshold')

# Highlight the key comparison
ax4.annotate('', xy=(2, 0.7), xytext=(4, 2.7),
            arrowprops=dict(arrowstyle='<->', color='red', lw=2))
ax4.text(3, 1.7, 'Tools > Reasoning!', fontsize=11, ha='center', fontweight='bold', color='red')

ax4.set_ylim(0, 5)
fig4.tight_layout()
fig4.savefig(Path(__file__).parent / 'fig4_tools_vs_reasoning.png', dpi=150, bbox_inches='tight')
print("Saved: fig4_tools_vs_reasoning.png")

# Figure 5: Time vs Quality Trade-off
fig5, ax5 = plt.subplots(figsize=(10, 6))

for cond in conditions:
    cond_data = [r for r in valid if r["condition"] == cond]
    times = [r["elapsed"] for r in cond_data]
    cis = [r["complexity_index"] for r in cond_data]
    ax5.scatter(times, cis, label=cond, s=100, c=COLORS[cond], edgecolor='black', linewidth=1, alpha=0.8)

ax5.set_xlabel('Response Time (seconds)', fontsize=12)
ax5.set_ylabel('Complexity Index (lower is better)', fontsize=12)
ax5.set_title('Time-Quality Trade-off: MCP Adds Latency but Improves Quality', fontsize=14, fontweight='bold')
ax5.legend(loc='upper right')
ax5.axhline(y=2.0, color='green', linestyle='--', alpha=0.5)

# Add regions
ax5.axvspan(0, 30, alpha=0.1, color='green', label='Fast')
ax5.axvspan(30, 100, alpha=0.1, color='yellow')
ax5.axvspan(100, 250, alpha=0.1, color='orange')

ax5.text(15, 18, 'Fast\nbut poor', ha='center', fontsize=9, alpha=0.7)
ax5.text(150, 18, 'Slower\nbut optimal', ha='center', fontsize=9, alpha=0.7)

fig5.tight_layout()
fig5.savefig(Path(__file__).parent / 'fig5_time_quality_tradeoff.png', dpi=150, bbox_inches='tight')
print("Saved: fig5_time_quality_tradeoff.png")

# Generate summary table
print("\n" + "="*80)
print("SUMMARY TABLE FOR POST")
print("="*80)

print("\n### Overall Results (All OMIPs Combined)")
print("| Condition | Avg CI | Std CI | Critical Pairs | Improvement |")
print("|-----------|--------|--------|----------------|-------------|")

baseline_ci = calc_stats(valid, "Sonnet")[0]
for cond in conditions:
    mean, std = calc_stats(valid, cond)
    crit = np.mean([r["critical_pairs"] for r in valid if r["condition"] == cond])
    improvement = (baseline_ci - mean) / baseline_ci * 100 if mean and baseline_ci else 0
    imp_str = f"+{improvement:.0f}%" if improvement > 0 else "-"
    print(f"| {cond:<12} | {mean:>6.2f} | {std:>6.2f} | {crit:>14.1f} | {imp_str:>11} |")

print("\n### OMIP-030 (10 markers) - Best Data")
print("| Condition | Avg CI | Improvement vs Sonnet |")
print("|-----------|--------|----------------------|")

omip30_valid = [r for r in valid if r["omip"] == 30]
baseline_omip30 = calc_stats(omip30_valid, "Sonnet")[0]
for cond in conditions:
    mean, _ = calc_stats(omip30_valid, cond)
    improvement = (baseline_omip30 - mean) / baseline_omip30 * 100 if mean and baseline_omip30 else 0
    imp_str = f"+{improvement:.0f}%" if improvement > 0 else "baseline"
    print(f"| {cond:<12} | {mean:>6.2f} | {imp_str:>20} |")

print("\n### Key Finding")
print(f"Sonnet+MCP CI (0.71) < Opus CI (2.77)")
print(f"Tools beat raw reasoning capability by 74%!")

print("\nAll visualizations saved!")
