#!/usr/bin/env python3
"""
Generate all figures for TRYLOCK paper
Author: Scott Thornton
Date: December 20, 2025
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Set publication-quality defaults
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 14

# Output directory
OUTPUT_DIR = Path(__file__).parent
OUTPUT_DIR.mkdir(exist_ok=True)


def figure2_asr_progression():
    """
    Figure 2: Defense-in-Depth ASR Progression
    Shows progressive reduction: Baseline → +DPO → +DPO+RepE
    """
    print("Generating Figure 2: ASR Progression...")

    # Data from TRYLOCK_Combined_Evaluation_Report.md
    configs = ['Baseline\n(Mistral-7B)', '+ Layer 1\n(DPO)', '+ Layers 1+2\n(RepE α=2.0)']
    asr_values = [46.5, 39.8, 8.0]
    colors = ['#ef4444', '#f59e0b', '#22c55e']  # Red, Orange, Green

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create bars
    bars = ax.bar(configs, asr_values, color=colors, width=0.6,
                   edgecolor='black', linewidth=1.5, alpha=0.85)

    # Add value labels on bars
    for bar, value in zip(bars, asr_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{value}%', ha='center', va='bottom',
                fontsize=13, fontweight='bold')

    # Add reduction annotations
    # Arrow from baseline to Layer 1
    ax.annotate('', xy=(1, 39.8), xytext=(0, 46.5),
                arrowprops=dict(arrowstyle='->', lw=2.5, color='#666666'))
    ax.text(0.5, 43, '-14.4%\nrelative', ha='center', fontsize=11,
            color='#333333', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                     edgecolor='gray', alpha=0.8))

    # Arrow from Layer 1 to Layers 1+2
    ax.annotate('', xy=(2, 8.0), xytext=(1, 39.8),
                arrowprops=dict(arrowstyle='->', lw=2.5, color='#666666'))
    ax.text(1.5, 24, '-79.9%\nrelative', ha='center', fontsize=11,
            color='#333333', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                     edgecolor='gray', alpha=0.8))

    # Overall reduction annotation
    ax.text(1, 52, '82.8% total relative reduction', ha='center',
            fontsize=12, fontweight='bold', color='#22c55e',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#22c55e',
                     edgecolor='#22c55e', alpha=0.15))

    # Formatting
    ax.set_ylabel('Attack Success Rate (%)', fontsize=14, fontweight='bold')
    ax.set_title('Defense-in-Depth: Progressive ASR Reduction',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_ylim(0, 58)
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=1)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    # Save in multiple formats
    plt.savefig(OUTPUT_DIR / 'figure2_asr_progression.png', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'figure2_asr_progression.pdf', bbox_inches='tight')
    plt.close()

    print("✓ Figure 2 saved (PNG + PDF)")


def figure3_attack_families():
    """
    Figure 3: Performance by Attack Family
    Baseline vs TRYLOCK across 5 attack categories
    """
    print("Generating Figure 3: Attack Families...")

    # Data from evaluation report
    categories = ['Obfuscation\nWrappers', 'Indirect\nInjection', 'Tool/Agent\nAbuse',
                  'Multi-Turn\nManipulation', 'Direct\nInjection']
    baseline = [54.9, 66.0, 38.3, 37.3, 22.0]
    trylock = [3.9, 8.0, 12.8, 7.8, 8.0]

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(categories))
    width = 0.35

    # Create bars
    bars1 = ax.bar(x - width/2, baseline, width, label='Baseline (Mistral-7B)',
                   color='#ef4444', edgecolor='black', linewidth=1.5, alpha=0.85)
    bars2 = ax.bar(x + width/2, trylock, width, label='TRYLOCK (α=2.0)',
                   color='#22c55e', edgecolor='black', linewidth=1.5, alpha=0.85)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1.5,
                    f'{height:.1f}%', ha='center', va='bottom',
                    fontsize=10, fontweight='bold')

    # Add reduction percentages above each pair
    reductions = [51.0, 58.0, 25.5, 29.5, 14.0]
    for i, reduction in enumerate(reductions):
        ax.text(i, max(baseline[i], trylock[i]) + 6, f'−{reduction:.1f}%',
                ha='center', fontsize=9, color='#059669', fontweight='bold')

    # Formatting
    ax.set_ylabel('Attack Success Rate (%)', fontsize=14, fontweight='bold')
    ax.set_title('ASR by Attack Family: Baseline vs. TRYLOCK (α=2.0)',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11)
    ax.legend(fontsize=12, loc='upper right', framealpha=0.95)
    ax.set_ylim(0, 75)
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=1)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    # Save
    plt.savefig(OUTPUT_DIR / 'figure3_attack_families.png', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'figure3_attack_families.pdf', bbox_inches='tight')
    plt.close()

    print("✓ Figure 3 saved (PNG + PDF)")


def figure4_alpha_sweep():
    """
    Figure 4: Steering Strength Impact
    Shows trade-off between ASR and over-refusal as α increases
    """
    print("Generating Figure 4: Alpha Sweep...")

    # Data from evaluation report
    alpha_values = [0.0, 1.0, 2.0, 2.5, 3.0]
    attack_asr = [43.8, 59.4, 8.0, 0.0, 0.0]
    over_refusal = [44.0, 26.0, 60.0, 98.0, 100.0]

    # Create figure
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot Attack ASR on left axis
    line1 = ax1.plot(alpha_values, attack_asr, color='#ef4444', marker='o',
                     markersize=12, linewidth=3, label='Attack Success Rate',
                     markeredgecolor='black', markeredgewidth=1.5)
    ax1.set_xlabel('Steering Strength (α)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Attack Success Rate (%)', fontsize=14,
                   fontweight='bold', color='#ef4444')
    ax1.tick_params(axis='y', labelcolor='#ef4444', labelsize=11)
    ax1.set_ylim(0, 70)
    ax1.grid(alpha=0.3, linestyle='--', linewidth=1)
    ax1.set_axisbelow(True)

    # Create second y-axis for Over-Refusal
    ax2 = ax1.twinx()
    line2 = ax2.plot(alpha_values, over_refusal, color='#f59e0b', marker='s',
                     markersize=12, linewidth=3, label='Over-Refusal Rate',
                     markeredgecolor='black', markeredgewidth=1.5)
    ax2.set_ylabel('Over-Refusal Rate (%)', fontsize=14,
                   fontweight='bold', color='#f59e0b')
    ax2.tick_params(axis='y', labelcolor='#f59e0b', labelsize=11)
    ax2.set_ylim(0, 110)

    # Highlight optimal alpha
    ax1.axvline(x=2.0, color='#22c55e', linestyle='--', linewidth=3, alpha=0.6)
    ax1.text(2.0, 68, 'Optimal α = 2.0', ha='center', va='top',
             fontsize=12, fontweight='bold', color='#059669',
             bbox=dict(boxstyle='round,pad=0.6', facecolor='#d1fae5',
                      edgecolor='#22c55e', linewidth=2, alpha=0.9))

    # Add annotation for optimal point
    ax1.annotate('8% ASR\n60% Over-Refusal', xy=(2.0, 8.0), xytext=(2.5, 25),
                 arrowprops=dict(arrowstyle='->', lw=2, color='#059669'),
                 fontsize=11, color='#059669', fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                          edgecolor='#22c55e', alpha=0.9))

    # Title
    plt.title('Impact of Steering Strength (α) on Security vs. Usability',
              fontsize=16, fontweight='bold', pad=20)

    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', fontsize=12, framealpha=0.95)

    # Remove top spines
    ax1.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)

    plt.tight_layout()

    # Save
    plt.savefig(OUTPUT_DIR / 'figure4_alpha_sweep.png', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'figure4_alpha_sweep.pdf', bbox_inches='tight')
    plt.close()

    print("✓ Figure 4 saved (PNG + PDF)")


def figure5_layer_contributions():
    """
    Figure 5: Independent Layer Contributions
    Stacked bar showing what each layer contributes
    """
    print("Generating Figure 5: Layer Contributions...")

    # Calculate layer contributions
    baseline_asr = 46.5
    after_dpo = 39.8
    after_repe = 8.0

    dpo_contribution = baseline_asr - after_dpo  # 6.7%
    repe_contribution = after_dpo - after_repe   # 31.8%
    remaining_asr = after_repe                    # 8.0%

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))

    # Stacked bar data
    categories = ['Attack Success Rate']

    # Create stacked bars
    p1 = ax.barh(categories, [remaining_asr], color='#22c55e',
                 edgecolor='black', linewidth=1.5, label='Blocked by Layers 1+2')
    p2 = ax.barh(categories, [repe_contribution], left=[remaining_asr],
                 color='#8b5cf6', edgecolor='black', linewidth=1.5,
                 label='Layer 2 (RepE) Contribution')
    p3 = ax.barh(categories, [dpo_contribution],
                 left=[remaining_asr + repe_contribution],
                 color='#3b82f6', edgecolor='black', linewidth=1.5,
                 label='Layer 1 (DPO) Contribution')
    p4 = ax.barh(categories, [after_repe],
                 left=[remaining_asr + repe_contribution + dpo_contribution],
                 color='#ef4444', edgecolor='black', linewidth=1.5,
                 label='Remaining Attacks (8%)')

    # Add percentage labels
    ax.text(remaining_asr/2, 0, f'{remaining_asr:.1f}%\nBlocked',
            ha='center', va='center', fontsize=11, fontweight='bold', color='white')
    ax.text(remaining_asr + repe_contribution/2, 0, f'{repe_contribution:.1f}%\nLayer 2',
            ha='center', va='center', fontsize=11, fontweight='bold', color='white')
    ax.text(remaining_asr + repe_contribution + dpo_contribution/2, 0,
            f'{dpo_contribution:.1f}%\nLayer 1',
            ha='center', va='center', fontsize=11, fontweight='bold', color='white')
    ax.text(baseline_asr + after_repe/2, 0, f'{after_repe:.1f}%\nRemaining',
            ha='center', va='center', fontsize=11, fontweight='bold', color='white')

    # Formatting
    ax.set_xlabel('Attack Success Rate (%)', fontsize=14, fontweight='bold')
    ax.set_title('Independent Layer Contributions to Defense',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlim(0, 60)
    ax.legend(loc='upper right', fontsize=10, framealpha=0.95)
    ax.grid(axis='x', alpha=0.3, linestyle='--', linewidth=1)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_yticks([])

    plt.tight_layout()

    # Save
    plt.savefig(OUTPUT_DIR / 'figure5_layer_contributions.png', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'figure5_layer_contributions.pdf', bbox_inches='tight')
    plt.close()

    print("✓ Figure 5 saved (PNG + PDF)")


def figure1_architecture():
    """
    Figure 1: TRYLOCK Three-Layer Architecture Diagram
    Shows the defense-in-depth flow from input to output
    """
    print("Generating Figure 1: Architecture Diagram...")

    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Colors
    colors = {
        'input': '#64748b',      # Slate
        'layer3': '#3b82f6',     # Blue
        'layer2': '#8b5cf6',     # Purple
        'layer1': '#22c55e',     # Green
        'output': '#22c55e',     # Green
        'arrow': '#475569',      # Dark slate
    }

    def draw_box(x, y, width, height, color, title, subtitle, details=None, alpha=0.9):
        """Draw a styled box with title and subtitle"""
        box = FancyBboxPatch((x, y), width, height,
                              boxstyle="round,pad=0.02,rounding_size=0.3",
                              facecolor=color, edgecolor='black',
                              linewidth=2, alpha=alpha)
        ax.add_patch(box)

        # Title
        ax.text(x + width/2, y + height - 0.4, title,
                ha='center', va='top', fontsize=13, fontweight='bold',
                color='white')
        # Subtitle
        ax.text(x + width/2, y + height - 0.85, subtitle,
                ha='center', va='top', fontsize=10, fontweight='bold',
                color='white', alpha=0.9)
        # Details
        if details:
            ax.text(x + width/2, y + 0.4, details,
                    ha='center', va='bottom', fontsize=9,
                    color='white', alpha=0.85, style='italic')

    # INPUT box
    draw_box(0.5, 4.2, 2, 1.6, colors['input'], 'INPUT', 'User Prompt')

    # Layer 3 (top)
    draw_box(4, 7, 4, 2.2, colors['layer3'], 'LAYER 3: OVERSIGHT',
             'Sidecar Classifier', 'Qwen 3B | Threat -> alpha')

    # Layer 2 (middle)
    draw_box(4, 4, 4, 2.2, colors['layer2'], 'LAYER 2: INSTINCT',
             'RepE Steering', 'Layers 12-26 | alpha: 0.5, 1.5, 2.5')

    # Layer 1 (bottom)
    draw_box(4, 1, 4, 2.2, colors['layer1'], 'LAYER 1: KNOWLEDGE',
             'DPO Fine-tuning', 'Mistral-7B · 2,939 pairs')

    # OUTPUT box
    draw_box(9.5, 4.2, 2, 1.6, colors['output'], 'OUTPUT', 'Safe Response')

    # Draw arrows
    arrow_style = dict(arrowstyle='->', lw=2.5, color=colors['arrow'],
                       connectionstyle='arc3,rad=0')

    # Input to Layer 3
    ax.annotate('', xy=(4, 8.1), xytext=(2.5, 5.4),
                arrowprops=dict(arrowstyle='->', lw=2.5, color=colors['arrow'],
                               connectionstyle='arc3,rad=0.2'))

    # Layer 3 to Layer 2 (with α label)
    ax.annotate('', xy=(6, 6.2), xytext=(6, 7),
                arrowprops=dict(arrowstyle='->', lw=3, color='#f59e0b'))
    ax.text(6.4, 6.6, 'α', fontsize=14, fontweight='bold', color='#f59e0b')

    # Layer 2 to Layer 1
    ax.annotate('', xy=(6, 3.2), xytext=(6, 4),
                arrowprops=dict(arrowstyle='->', lw=2.5, color=colors['arrow']))

    # Input directly to Layer 1 (through layers)
    ax.annotate('', xy=(4, 2.1), xytext=(2.5, 4.6),
                arrowprops=dict(arrowstyle='->', lw=2.5, color=colors['arrow'],
                               connectionstyle='arc3,rad=-0.2'))

    # Layer 1 to Output
    ax.annotate('', xy=(9.5, 5), xytext=(8, 2.1),
                arrowprops=dict(arrowstyle='->', lw=2.5, color=colors['arrow'],
                               connectionstyle='arc3,rad=-0.15'))

    # Title
    ax.text(6, 9.7, 'TRYLOCK: Defense-in-Depth Architecture',
            ha='center', va='top', fontsize=16, fontweight='bold')

    # Legend/annotation
    ax.text(0.5, 0.5, 'Flow: Input -> Layer 3 (classify, set alpha) -> Layer 2 (steer) -> Layer 1 (generate) -> Output',
            ha='left', va='bottom', fontsize=9, color='#64748b', style='italic')

    plt.tight_layout()

    # Save
    plt.savefig(OUTPUT_DIR / 'figure1_architecture.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig(OUTPUT_DIR / 'figure1_architecture.pdf', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()

    print("✓ Figure 1 saved (PNG + PDF)")


def main():
    """Generate all figures for TRYLOCK paper"""
    print("="*60)
    print("TRYLOCK Paper - Figure Generation")
    print("="*60)
    print()

    # Generate all figures
    figure1_architecture()
    figure2_asr_progression()
    figure3_attack_families()
    figure4_alpha_sweep()
    figure5_layer_contributions()

    print()
    print("="*60)
    print("✓ All figures generated successfully!")
    print("="*60)
    print()
    print("Output files:")
    for i in range(2, 6):
        png_file = OUTPUT_DIR / f'figure{i}_*.png'
        pdf_file = OUTPUT_DIR / f'figure{i}_*.pdf'
        print(f"  - figure{i}_*.png (300 DPI)")
        print(f"  - figure{i}_*.pdf (vector)")
    print()
    print("Note: Figure 1 (architecture) should be created with AI image")
    print("generation or diagram tool. See FIGURE_SPECIFICATIONS.md for prompt.")
    print()


if __name__ == '__main__':
    main()
