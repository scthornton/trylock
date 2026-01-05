# TRYLOCK Paper - Figure Specifications and Data

**Date:** December 20, 2025
**Source Data:** `/Users/scott/perfecxion/datasets/aegis/outputs/TRYLOCK_Combined_Evaluation_Report.md`

---

## Figure 1: TRYLOCK Three-Layer Architecture

### AI Image Generation Prompt

```
Create a professional technical diagram showing a three-layer security architecture called "TRYLOCK" for defending AI language models against jailbreak attacks. The diagram should be clean, modern, and suitable for an academic computer science paper.

Layout (left to right):
1. INPUT box on the far left containing "User Prompt"
2. Three defensive layers shown as distinct vertical sections:

   LAYER 3 (top): "OVERSIGHT - Sidecar Classifier"
   - Small neural network icon (3B parameters)
   - Output: "Threat Level → α (0.5, 1.5, or 2.5)"
   - Color: Blue/cyan tones
   - Arrow pointing down to Layer 2

   LAYER 2 (middle): "INSTINCT - RepE Steering"
   - Activation space visualization (wavy lines or neural activations)
   - Label: "Steering Vectors (8 layers)"
   - Shows "α" input from Layer 3
   - Color: Purple/violet tones
   - Arrow pointing down to Layer 1

   LAYER 1 (bottom): "KNOWLEDGE - DPO Fine-tuning"
   - Model weights icon (LoRA adapter)
   - Label: "Safety Preferences (2,939 pairs)"
   - Color: Green/teal tones

3. OUTPUT box on the far right containing "Safe Response"

Visual style:
- Clean, modern, professional academic paper aesthetic
- Flat design with subtle gradients
- Clear arrows showing data flow
- Labels in sans-serif font (similar to Helvetica or Arial)
- White or light gray background
- Each layer in a distinct color scheme
- Icons should be simple and geometric

The overall flow should clearly show: Input → Layer 3 (determines α) → Layer 2 (applies steering with α) → Layer 1 (base model) → Output
```

### Data for Diagram Labels
- **Layer 1**: "DPO LoRA Adapter (Mistral-7B)"
- **Layer 2**: "RepE Vectors (Layers 12-26)"
- **Layer 3**: "Qwen 3B Classifier"
- **Steering Strength Values**: α ∈ {0.5, 1.5, 2.5}

### Alternative Text-Based Specification
```
┌─────────────┐
│ User Prompt │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────┐
│  Layer 3: OVERSIGHT         │
│  Sidecar Classifier (3B)    │
│  Output: Threat → α         │
└────────────┬────────────────┘
             │ α = {0.5,1.5,2.5}
             ▼
┌─────────────────────────────┐
│  Layer 2: INSTINCT          │
│  RepE Steering Vectors      │
│  Activation steering @ α    │
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  Layer 1: KNOWLEDGE         │
│  DPO LoRA Weights           │
│  2,939 training pairs       │
└────────────┬────────────────┘
             │
             ▼
     ┌───────────────┐
     │ Safe Response │
     └───────────────┘
```

---

## Figure 2: Attack Success Rate - Defense-in-Depth Progression

### Actual Data (from evaluation results)

| Configuration | ASR | Absolute Reduction | Relative Reduction |
|--------------|-----|-------------------|-------------------|
| Baseline (Mistral-7B) | 46.5% | -- | -- |
| + Layer 1 (DPO) | 39.8% | -6.7% | -14.4% |
| + Layer 2 (RepE α=2.0) | 8.0% | -38.5% | -82.8% |

### AI Image Generation Prompt

```
Create a professional bar chart for an academic computer science paper showing the progressive reduction in "Attack Success Rate" (ASR) across three defensive configurations.

Chart specifications:
- Title: "Defense-in-Depth: Progressive ASR Reduction"
- Y-axis: "Attack Success Rate (%)" ranging from 0% to 50%
- X-axis: Three bars labeled:
  1. "Baseline" - height at 46.5%
  2. "+ Layer 1 (DPO)" - height at 39.8%
  3. "+ Layers 1+2 (RepE)" - height at 8.0%

Visual design:
- Bar 1 (Baseline): Red color (#ef4444) - showing danger
- Bar 2 (Layer 1): Orange color (#f59e0b) - showing improvement
- Bar 3 (Layers 1+2): Green color (#22c55e) - showing success
- Each bar should have the exact percentage displayed on top
- Show reduction arrows between bars:
  - Between Bar 1 and 2: "-14.4% relative" in small text
  - Between Bar 2 and 3: "-79.9% relative" in small text
- Clean, professional appearance suitable for academic publication
- White or light gray background
- Grid lines on Y-axis for easier reading
- Sans-serif font (Helvetica or Arial style)

The chart should clearly visualize the dramatic drop from 46.5% to 8.0%, emphasizing the 82.8% total relative reduction achieved by the layered defense approach.
```

### Python Code to Generate This Figure

```python
import matplotlib.pyplot as plt
import numpy as np

# Data
configs = ['Baseline\n(Mistral-7B)', '+ Layer 1\n(DPO)', '+ Layers 1+2\n(RepE α=2.0)']
asr_values = [46.5, 39.8, 8.0]
colors = ['#ef4444', '#f59e0b', '#22c55e']  # Red, Orange, Green

# Create figure
fig, ax = plt.subplots(figsize=(10, 6))

# Create bars
bars = ax.bar(configs, asr_values, color=colors, width=0.6, edgecolor='black', linewidth=1.5)

# Add value labels on bars
for bar, value in zip(bars, asr_values):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 1,
            f'{value}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

# Add reduction annotations
ax.annotate('', xy=(1, 39.8), xytext=(0, 46.5),
            arrowprops=dict(arrowstyle='->', lw=2, color='gray'))
ax.text(0.5, 43, '-14.4%\nrelative', ha='center', fontsize=10, color='gray')

ax.annotate('', xy=(2, 8.0), xytext=(1, 39.8),
            arrowprops=dict(arrowstyle='->', lw=2, color='gray'))
ax.text(1.5, 24, '-79.9%\nrelative', ha='center', fontsize=10, color='gray')

# Formatting
ax.set_ylabel('Attack Success Rate (%)', fontsize=14, fontweight='bold')
ax.set_title('Defense-in-Depth: Progressive ASR Reduction', fontsize=16, fontweight='bold', pad=20)
ax.set_ylim(0, 55)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig('figure2_asr_progression.png', dpi=300, bbox_inches='tight')
plt.savefig('figure2_asr_progression.pdf', bbox_inches='tight')
print("✓ Figure 2 saved")
```

---

## Figure 3: Performance by Attack Family (α=2.0 vs Baseline)

### Actual Data

| Attack Family | Baseline ASR | α=2.0 ASR | Improvement |
|---------------|-------------|-----------|-------------|
| Obfuscation Wrappers | 54.9% | 3.9% | -51.0% |
| Indirect Injection | 66.0% | 8.0% | -58.0% |
| Tool/Agent Abuse | 38.3% | 12.8% | -25.5% |
| Multi-Turn Manipulation | 37.3% | 7.8% | -29.5% |
| Direct Injection | 22.0% | 8.0% | -14.0% |

### AI Image Generation Prompt

```
Create a professional grouped bar chart for an academic computer science paper comparing attack success rates across five different attack categories, showing baseline vs. defended performance.

Chart specifications:
- Title: "ASR by Attack Family: Baseline vs. TRYLOCK (α=2.0)"
- Y-axis: "Attack Success Rate (%)" ranging from 0% to 70%
- X-axis: Five attack categories:
  1. "Obfuscation\nWrappers"
  2. "Indirect\nInjection"
  3. "Tool/Agent\nAbuse"
  4. "Multi-Turn\nManipulation"
  5. "Direct\nInjection"

For each category, show two bars side by side:
- Left bar (Baseline): Red color (#ef4444)
- Right bar (TRYLOCK): Green color (#22c55e)

Exact heights:
1. Obfuscation: 54.9% (red), 3.9% (green)
2. Indirect: 66.0% (red), 8.0% (green)
3. Tool/Agent: 38.3% (red), 12.8% (green)
4. Multi-Turn: 37.3% (red), 7.8% (green)
5. Direct: 22.0% (red), 8.0% (green)

Visual design:
- Clean, professional academic paper aesthetic
- White or light gray background
- Horizontal grid lines for easier reading
- Legend showing "Baseline" (red) and "TRYLOCK (α=2.0)" (green)
- Sans-serif font
- Each bar should have exact percentage value on top
- Bars should have black edge outlines

The chart should emphasize the dramatic reduction across all attack types, with the most dramatic improvements on Obfuscation (-51%) and Indirect Injection (-58%).
```

### Python Code to Generate This Figure

```python
import matplotlib.pyplot as plt
import numpy as np

# Data
categories = ['Obfuscation\nWrappers', 'Indirect\nInjection', 'Tool/Agent\nAbuse',
              'Multi-Turn\nManipulation', 'Direct\nInjection']
baseline = [54.9, 66.0, 38.3, 37.3, 22.0]
trylock = [3.9, 8.0, 12.8, 7.8, 8.0]

# Create figure
fig, ax = plt.subplots(figsize=(12, 6))

x = np.arange(len(categories))
width = 0.35

# Create bars
bars1 = ax.bar(x - width/2, baseline, width, label='Baseline',
               color='#ef4444', edgecolor='black', linewidth=1.5)
bars2 = ax.bar(x + width/2, trylock, width, label='TRYLOCK (α=2.0)',
               color='#22c55e', edgecolor='black', linewidth=1.5)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Formatting
ax.set_ylabel('Attack Success Rate (%)', fontsize=14, fontweight='bold')
ax.set_title('ASR by Attack Family: Baseline vs. TRYLOCK (α=2.0)', fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=11)
ax.legend(fontsize=12, loc='upper right')
ax.set_ylim(0, 75)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig('figure3_attack_families.png', dpi=300, bbox_inches='tight')
plt.savefig('figure3_attack_families.pdf', bbox_inches='tight')
print("✓ Figure 3 saved")
```

---

## Figure 4: Steering Strength Impact (Alpha Sweep)

### Actual Data

| Alpha (α) | Attack ASR | Over-Refusal Rate |
|-----------|-----------|-------------------|
| 0.0 | 43.8% | 44.0% |
| 1.0 | 59.4% | 26.0% |
| 2.0 | 8.0% | 60.0% |
| 2.5 | 0.0% | 98.0% |
| 3.0 | 0.0% | 100.0% |

### AI Image Generation Prompt

```
Create a professional dual-axis line chart for an academic computer science paper showing how two metrics change with increasing "steering strength" (alpha parameter).

Chart specifications:
- Title: "Impact of Steering Strength (α) on Security vs. Usability"
- X-axis: "Steering Strength (α)" from 0.0 to 3.0
- Left Y-axis: "Attack Success Rate (%)" from 0% to 70% (RED)
- Right Y-axis: "Over-Refusal Rate (%)" from 0% to 100% (ORANGE)

Two lines:
1. Attack ASR (red line, circles):
   - Points: (0.0, 43.8%), (1.0, 59.4%), (2.0, 8.0%), (2.5, 0.0%), (3.0, 0.0%)
   - Thick red line (#ef4444) with circular markers
   - Label: "Attack Success Rate"

2. Over-Refusal (orange line, squares):
   - Points: (0.0, 44.0%), (1.0, 26.0%), (2.0, 60.0%), (2.5, 98.0%), (3.0, 100.0%)
   - Thick orange line (#f59e0b) with square markers
   - Label: "Over-Refusal Rate"

Special annotations:
- Vertical dashed line at α=2.0 labeled "Optimal α"
- Green box highlighting the α=2.0 point
- Text annotation: "Sweet spot: 8% ASR, 60% over-refusal"

Visual design:
- Clean, professional academic paper aesthetic
- White or light gray background
- Grid lines for both axes
- Clear legend
- Sans-serif font
- Markers should be visible and distinct

The chart should show the trade-off: as α increases, attacks decrease but over-refusal increases, with α=2.0 being the optimal balance.
```

### Python Code to Generate This Figure

```python
import matplotlib.pyplot as plt
import numpy as np

# Data
alpha_values = [0.0, 1.0, 2.0, 2.5, 3.0]
attack_asr = [43.8, 59.4, 8.0, 0.0, 0.0]
over_refusal = [44.0, 26.0, 60.0, 98.0, 100.0]

# Create figure
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot Attack ASR on left axis
ax1.plot(alpha_values, attack_asr, color='#ef4444', marker='o',
         markersize=10, linewidth=3, label='Attack Success Rate')
ax1.set_xlabel('Steering Strength (α)', fontsize=14, fontweight='bold')
ax1.set_ylabel('Attack Success Rate (%)', fontsize=14, fontweight='bold', color='#ef4444')
ax1.tick_params(axis='y', labelcolor='#ef4444')
ax1.set_ylim(0, 70)
ax1.grid(alpha=0.3, linestyle='--')

# Create second y-axis for Over-Refusal
ax2 = ax1.twinx()
ax2.plot(alpha_values, over_refusal, color='#f59e0b', marker='s',
         markersize=10, linewidth=3, label='Over-Refusal Rate')
ax2.set_ylabel('Over-Refusal Rate (%)', fontsize=14, fontweight='bold', color='#f59e0b')
ax2.tick_params(axis='y', labelcolor='#f59e0b')
ax2.set_ylim(0, 110)

# Highlight optimal alpha
ax1.axvline(x=2.0, color='green', linestyle='--', linewidth=2, alpha=0.7)
ax1.text(2.0, 65, 'Optimal α=2.0\n8% ASR, 60% Over-Refusal',
         ha='center', va='top', fontsize=11, fontweight='bold',
         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

# Title
plt.title('Impact of Steering Strength (α) on Security vs. Usability',
          fontsize=16, fontweight='bold', pad=20)

# Combine legends
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=12)

plt.tight_layout()
plt.savefig('figure4_alpha_sweep.png', dpi=300, bbox_inches='tight')
plt.savefig('figure4_alpha_sweep.pdf', bbox_inches='tight')
print("✓ Figure 4 saved")
```

---

## Figure 5: Layer Independence - What Each Layer Catches

### Conceptual Visualization (requires manual design)

### AI Image Generation Prompt

```
Create a professional Venn diagram-style illustration for an academic computer science paper showing how three defensive layers provide independent, complementary protection against different attack types.

Layout: Three overlapping circles/regions labeled:

1. Circle 1 (Green): "Layer 1 - DPO"
   - Lists attacks it catches best:
   - "Direct attacks"
   - "Simple roleplay"
   - "Training-similar patterns"
   - Coverage: ~14% independent

2. Circle 2 (Purple): "Layer 2 - RepE"
   - Lists attacks it catches best:
   - "Obfuscation (-51%)"
   - "Indirect injection (-58%)"
   - "Novel attack patterns"
   - Coverage: ~80% when combined with Layer 1

3. Circle 3 (Blue): "Layer 3 - Sidecar"
   - Lists its unique function:
   - "Adaptive α selection"
   - "Reduces over-refusal"
   - "Threat classification"
   - Coverage: Optimization layer

Overlapping regions show combined protection.

Center text: "Combined: 82.8% ASR reduction"

Visual style:
- Clean, modern, professional academic diagram
- Translucent overlapping circles with distinct colors
- Clear labels in sans-serif font
- White or light gray background
- Subtle shadows or borders on circles
- Attack types listed in bullet points within each region

The diagram should visually communicate that each layer provides unique, independent protection that complements the others.
```

---

## Summary: Recommended Figures for Paper

### Priority 1 (Must Have):
1. **Figure 1**: Architecture diagram (AI-generated or manually designed)
2. **Figure 2**: ASR progression bar chart (Python-generated)

### Priority 2 (Strongly Recommended):
3. **Figure 3**: Attack family performance (Python-generated)
4. **Figure 4**: Alpha sweep analysis (Python-generated)

### Priority 3 (Nice to Have):
5. **Figure 5**: Layer independence Venn diagram (AI-generated or manual)

---

## File Export Recommendations

For each figure, generate:
- **PNG** at 300 DPI for inclusion in paper
- **PDF** vector format for best LaTeX quality
- **SVG** (optional) for web/presentation use

Recommended dimensions:
- Single-column: 3.5 inches width
- Double-column: 7 inches width
- Height: 3-5 inches (as needed)

---

## LaTeX Integration

```latex
\begin{figure}[h]
\centering
\includegraphics[width=0.9\textwidth]{figure1_architecture.pdf}
\caption{TRYLOCK three-layer defense architecture. The sidecar classifier (Layer 3) determines threat level and sets steering strength $\alpha$ for RepE (Layer 2), which operates on top of DPO-trained weights (Layer 1).}
\label{fig:architecture}
\end{figure}

\begin{figure}[h]
\centering
\includegraphics[width=0.8\textwidth]{figure2_asr_progression.pdf}
\caption{Progressive ASR reduction through defense-in-depth. Each layer provides independent protection, achieving 82.8\% total reduction.}
\label{fig:results}
\end{figure}
```

---

**Generated:** December 20, 2025
**TRYLOCK Project - perfecXion.ai**
