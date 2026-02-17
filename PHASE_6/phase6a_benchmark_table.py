"""
Phase 6a: Benchmark Table & Visualization
Generates a professional benchmark comparison table image
suitable for a Design & Concept proposal.

Run from PHASE_6/ folder:
  python3 phase6a_benchmark_table.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import os

print("="*80)
print("PHASE 6a: BENCHMARK TABLE & VISUALIZATION")
print("="*80)

PHASE6_DIR = './'
def p6(f): return os.path.join(PHASE6_DIR, f)

# ============================================================================
# ALL RESULTS
# ============================================================================
RF = {
    'binary': {
        'accuracy': 99.64, 'precision': 91.80,
        'recall': 99.84,   'f1': 95.65,
        'roc_auc': 99.98,  'train_time': 299.69, 'inf_ms': 0.0061,
    },
    'multi': {
        'accuracy': 98.85, 'precision': 99.30,
        'recall': 98.85,   'f1_w': 99.05,
        'f1_macro': 68.69, 'train_time': 267.54, 'inf_ms': 0.0097,
    },
    'per_class': {
        'Analysis': 0.16, 'Backdoor': 0.15, 'Benign': 1.00,
        'DoS': 0.47,      'Exploits': 0.86, 'Fuzzers': 0.84,
        'Generic': 0.91,  'Reconnaissance': 0.83,
        'Shellcode': 0.92,'Worms': 0.76,
    }
}

MLP = {
    'binary': {
        'accuracy': 99.60, 'precision': 92.01,
        'recall': 98.55,   'f1': 95.17,
        'roc_auc': 99.95,  'train_time': 324.93, 'inf_ms': 0.0453,
    },
    'multi': {
        'accuracy': 98.68, 'precision': 98.78,
        'recall': 98.68,   'f1_w': 98.70,
        'f1_macro': 54.93, 'train_time': 341.19, 'inf_ms': 0.0442,
    },
    'per_class': {
        'Analysis': 0.14, 'Backdoor': 0.19, 'Benign': 1.00,
        'DoS': 0.15,      'Exploits': 0.79, 'Fuzzers': 0.77,
        'Generic': 0.84,  'Reconnaissance': 0.80,
        'Shellcode': 0.82,'Worms': 0.00,
    }
}

# ============================================================================
# CHART 1: Master Benchmark Table (publication-ready)
# ============================================================================
print("\n[1] Generating Master Benchmark Table...")

fig, ax = plt.subplots(figsize=(14, 9))
ax.axis('off')

# Table data
col_labels = ['Metric', 'Random Forest', 'MLP (Neural Net)', 'Winner']
row_data = [
    # Binary
    ['BINARY CLASSIFICATION', '', '', ''],
    ['  Accuracy (%)',           f"{RF['binary']['accuracy']:.2f}",  f"{MLP['binary']['accuracy']:.2f}",  'RF'],
    ['  Precision (%)',          f"{RF['binary']['precision']:.2f}", f"{MLP['binary']['precision']:.2f}", 'MLP'],
    ['  Recall (%)',             f"{RF['binary']['recall']:.2f}",    f"{MLP['binary']['recall']:.2f}",    'RF'],
    ['  F1-Score (%)',           f"{RF['binary']['f1']:.2f}",        f"{MLP['binary']['f1']:.2f}",        'RF'],
    ['  ROC-AUC (%)',            f"{RF['binary']['roc_auc']:.2f}",   f"{MLP['binary']['roc_auc']:.2f}",   'RF'],
    ['  Training Time (s)',      f"{RF['binary']['train_time']:.1f}",f"{MLP['binary']['train_time']:.1f}",'RF'],
    ['  Inference (ms/sample)',  f"{RF['binary']['inf_ms']:.4f}",    f"{MLP['binary']['inf_ms']:.4f}",    'RF'],
    # Multi
    ['MULTI-CLASS CLASSIFICATION', '', '', ''],
    ['  Accuracy (%)',           f"{RF['multi']['accuracy']:.2f}",   f"{MLP['multi']['accuracy']:.2f}",   'RF'],
    ['  Precision - Weighted (%)' ,f"{RF['multi']['precision']:.2f}",f"{MLP['multi']['precision']:.2f}",  'RF'],
    ['  Recall - Weighted (%)',  f"{RF['multi']['recall']:.2f}",     f"{MLP['multi']['recall']:.2f}",     'RF'],
    ['  F1 Weighted (%)',        f"{RF['multi']['f1_w']:.2f}",       f"{MLP['multi']['f1_w']:.2f}",       'RF'],
    ['  F1 Macro (%) *',        f"{RF['multi']['f1_macro']:.2f}",   f"{MLP['multi']['f1_macro']:.2f}",   'RF'],
    ['  Training Time (s)',      f"{RF['multi']['train_time']:.1f}", f"{MLP['multi']['train_time']:.1f}", 'RF'],
    ['  Inference (ms/sample)',  f"{RF['multi']['inf_ms']:.4f}",     f"{MLP['multi']['inf_ms']:.4f}",     'RF'],
]

col_widths = [0.38, 0.20, 0.22, 0.14]
col_x      = [0.01, 0.40, 0.61, 0.84]
row_height = 0.054
start_y    = 0.97

# Colors
HEADER_BG   = '#1F4E79'
HEADER_FG   = 'white'
SECTION_BG  = '#BDD7EE'
SECTION_FG  = '#1F4E79'
ROW_ODD     = '#F2F7FC'
ROW_EVEN    = 'white'
RF_COLOR    = '#2E75B6'
MLP_COLOR   = '#C55A11'
WIN_RF      = '#E8F4FD'
WIN_MLP     = '#FDE9D9'
BORDER      = '#CCCCCC'

# Draw header
for i, (label, cx, cw) in enumerate(zip(col_labels, col_x, col_widths)):
    rect = FancyBboxPatch((cx, start_y - row_height), cw - 0.005, row_height,
                          boxstyle="square,pad=0", linewidth=0.5,
                          edgecolor=BORDER, facecolor=HEADER_BG,
                          transform=ax.transAxes, clip_on=False)
    ax.add_patch(rect)
    ax.text(cx + cw/2 - 0.002, start_y - row_height/2,
            label, transform=ax.transAxes,
            ha='center', va='center',
            fontsize=10, fontweight='bold', color=HEADER_FG)

# Draw rows
for row_i, row in enumerate(row_data):
    y = start_y - row_height * (row_i + 2)
    is_section = row[1] == ''

    for col_i, (cell, cx, cw) in enumerate(zip(row, col_x, col_widths)):
        if is_section:
            bg = SECTION_BG
        elif row_i % 2 == 0:
            bg = ROW_ODD
        else:
            bg = ROW_EVEN

        # Highlight winner column
        if not is_section and col_i == 3:
            bg = WIN_RF if cell == 'RF' else (WIN_MLP if cell == 'MLP' else bg)

        rect = FancyBboxPatch((cx, y), cw - 0.005, row_height,
                              boxstyle="square,pad=0", linewidth=0.5,
                              edgecolor=BORDER, facecolor=bg,
                              transform=ax.transAxes, clip_on=False)
        ax.add_patch(rect)

        # Text
        ha = 'left' if col_i == 0 else 'center'
        tx = (cx + 0.01) if col_i == 0 else (cx + cw/2 - 0.002)

        color = 'black'
        bold  = is_section
        fs    = 9.5

        if not is_section and col_i == 1:  # RF values
            color = RF_COLOR
        elif not is_section and col_i == 2:  # MLP values
            color = MLP_COLOR
        elif col_i == 3 and cell == 'RF':
            color = RF_COLOR; bold = True
        elif col_i == 3 and cell == 'MLP':
            color = MLP_COLOR; bold = True

        if is_section:
            fs = 9.5

        ax.text(tx, y + row_height/2, cell,
                transform=ax.transAxes,
                ha=ha, va='center',
                fontsize=fs,
                fontweight='bold' if bold else 'normal',
                color=SECTION_FG if is_section else color)

# Title
ax.text(0.5, 0.995, 'Benchmark Comparison: Random Forest vs MLP',
        transform=ax.transAxes, ha='center', va='top',
        fontsize=13, fontweight='bold', color='#1F4E79')

# Footnote
ax.text(0.01, 0.005,
        '* F1 Macro is the most honest metric — treats all 10 classes equally '
        'regardless of class size (Benign=96% of data).',
        transform=ax.transAxes, ha='left', va='bottom',
        fontsize=8, color='gray', style='italic')

# Legend
rf_patch  = mpatches.Patch(color=RF_COLOR,  label='Random Forest')
mlp_patch = mpatches.Patch(color=MLP_COLOR, label='MLP (Neural Net)')
ax.legend(handles=[rf_patch, mlp_patch], loc='lower right',
          bbox_to_anchor=(0.99, 0.02), fontsize=9, framealpha=0.9)

plt.tight_layout(pad=0)
plt.savefig(p6('benchmark_table.png'), dpi=250, bbox_inches='tight',
            facecolor='white')
plt.close()
print("  Saved -> benchmark_table.png")

# ============================================================================
# CHART 2: F1 Score Summary Bar (clean, publication-ready)
# ============================================================================
print("\n[2] Generating F1 Summary Bar Chart...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('RF vs MLP — F1 Score Comparison', fontsize=14,
             fontweight='bold', color='#1F4E79', y=1.01)

# Binary F1
ax1 = axes[0]
cats = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
rf_b  = [RF['binary']['accuracy'], RF['binary']['precision'],
         RF['binary']['recall'],   RF['binary']['f1'], RF['binary']['roc_auc']]
mlp_b = [MLP['binary']['accuracy'], MLP['binary']['precision'],
         MLP['binary']['recall'],   MLP['binary']['f1'], MLP['binary']['roc_auc']]

x = np.arange(len(cats)); w = 0.35
b1 = ax1.bar(x - w/2, rf_b,  w, label='Random Forest', color='#2E75B6',
             edgecolor='white', linewidth=0.8)
b2 = ax1.bar(x + w/2, mlp_b, w, label='MLP',           color='#C55A11',
             edgecolor='white', linewidth=0.8)
ax1.set_title('Binary Classification', fontweight='bold', fontsize=12,
              color='#1F4E79')
ax1.set_ylim(88, 102)
ax1.set_xticks(x); ax1.set_xticklabels(cats, fontsize=9)
ax1.set_ylabel('Score (%)', fontsize=10)
ax1.legend(fontsize=9); ax1.grid(axis='y', alpha=0.25, linestyle='--')
ax1.spines[['top','right']].set_visible(False)
for b, v in zip(b1, rf_b):
    ax1.text(b.get_x()+b.get_width()/2, v+0.1, f'{v:.2f}',
             ha='center', va='bottom', fontsize=7.5, color='#2E75B6',
             fontweight='bold')
for b, v in zip(b2, mlp_b):
    ax1.text(b.get_x()+b.get_width()/2, v+0.1, f'{v:.2f}',
             ha='center', va='bottom', fontsize=7.5, color='#C55A11',
             fontweight='bold')

# Multi-class F1
ax2 = axes[1]
cats2 = ['Accuracy', 'Precision\n(W)', 'Recall\n(W)', 'F1\nWeighted',
         'F1 Macro\n(Honest)']
rf_m  = [RF['multi']['accuracy'],  RF['multi']['precision'],
         RF['multi']['recall'],     RF['multi']['f1_w'],
         RF['multi']['f1_macro']]
mlp_m = [MLP['multi']['accuracy'], MLP['multi']['precision'],
         MLP['multi']['recall'],    MLP['multi']['f1_w'],
         MLP['multi']['f1_macro']]

x = np.arange(len(cats2)); w = 0.35
b3 = ax2.bar(x - w/2, rf_m,  w, label='Random Forest', color='#2E75B6',
             edgecolor='white', linewidth=0.8)
b4 = ax2.bar(x + w/2, mlp_m, w, label='MLP',           color='#C55A11',
             edgecolor='white', linewidth=0.8)

# Highlight F1 Macro column
ax2.axvspan(3.5, 4.5, alpha=0.07, color='gold')
ax2.set_title('Multi-class Classification', fontweight='bold', fontsize=12,
              color='#1F4E79')
ax2.set_ylim(40, 104)
ax2.set_xticks(x); ax2.set_xticklabels(cats2, fontsize=9)
ax2.set_ylabel('Score (%)', fontsize=10)
ax2.legend(fontsize=9); ax2.grid(axis='y', alpha=0.25, linestyle='--')
ax2.spines[['top','right']].set_visible(False)
ax2.text(4, 42, 'Most\nHonest', ha='center', fontsize=8,
         color='#B8860B', fontstyle='italic')
for b, v in zip(b3, rf_m):
    ax2.text(b.get_x()+b.get_width()/2, v+0.3, f'{v:.2f}',
             ha='center', va='bottom', fontsize=7.5, color='#2E75B6',
             fontweight='bold')
for b, v in zip(b4, mlp_m):
    ax2.text(b.get_x()+b.get_width()/2, v+0.3, f'{v:.2f}',
             ha='center', va='bottom', fontsize=7.5, color='#C55A11',
             fontweight='bold')

plt.tight_layout()
plt.savefig(p6('benchmark_bar_chart.png'), dpi=250, bbox_inches='tight',
            facecolor='white')
plt.close()
print("  Saved -> benchmark_bar_chart.png")

# ============================================================================
# CHART 3: Per-Class F1 Heatmap
# ============================================================================
print("\n[3] Generating Per-Class F1 Heatmap...")

classes  = list(RF['per_class'].keys())
rf_vals  = [RF['per_class'][c]  for c in classes]
mlp_vals = [MLP['per_class'][c] for c in classes]

data = np.array([rf_vals, mlp_vals])

fig, ax = plt.subplots(figsize=(13, 3.5))
im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

ax.set_xticks(np.arange(len(classes)))
ax.set_yticks([0, 1])
ax.set_xticklabels(classes, fontsize=10, rotation=20, ha='right')
ax.set_yticklabels(['Random Forest', 'MLP'], fontsize=11, fontweight='bold')

# Annotate cells
for i in range(2):
    for j in range(len(classes)):
        val = data[i, j]
        color = 'black' if 0.3 < val < 0.8 else 'white' if val <= 0.3 else 'black'
        ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                fontsize=10, fontweight='bold', color=color)

ax.set_title('Per-Class F1 Score Heatmap (Multi-class Classification)',
             fontsize=13, fontweight='bold', color='#1F4E79', pad=12)

cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
cbar.set_label('F1 Score', fontsize=10)
cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])

plt.tight_layout()
plt.savefig(p6('benchmark_heatmap.png'), dpi=250, bbox_inches='tight',
            facecolor='white')
plt.close()
print("  Saved -> benchmark_heatmap.png")

# ============================================================================
# CHART 4: Speed Comparison
# ============================================================================
print("\n[4] Generating Speed Comparison Chart...")

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
fig.suptitle('Computational Performance: RF vs MLP', fontsize=13,
             fontweight='bold', color='#1F4E79')

# Training time
ax1 = axes[0]
tasks   = ['Binary', 'Multi-class']
rf_tt   = [RF['binary']['train_time'],  RF['multi']['train_time']]
mlp_tt  = [MLP['binary']['train_time'], MLP['multi']['train_time']]
x = np.arange(2); w = 0.35
ax1.bar(x-w/2, rf_tt,  w, label='RF',  color='#2E75B6', edgecolor='white')
ax1.bar(x+w/2, mlp_tt, w, label='MLP', color='#C55A11', edgecolor='white')
ax1.set_title('Training Time (seconds)', fontweight='bold')
ax1.set_xticks(x); ax1.set_xticklabels(tasks)
ax1.set_ylabel('Seconds'); ax1.legend(); ax1.grid(axis='y', alpha=0.25)
ax1.spines[['top','right']].set_visible(False)
for i, (rv, mv) in enumerate(zip(rf_tt, mlp_tt)):
    ax1.text(i-w/2, rv+2, f'{rv:.0f}s', ha='center', fontsize=9,
             color='#2E75B6', fontweight='bold')
    ax1.text(i+w/2, mv+2, f'{mv:.0f}s', ha='center', fontsize=9,
             color='#C55A11', fontweight='bold')

# Inference speed
ax2 = axes[1]
rf_inf  = [RF['binary']['inf_ms'],  RF['multi']['inf_ms']]
mlp_inf = [MLP['binary']['inf_ms'], MLP['multi']['inf_ms']]
ax2.bar(x-w/2, rf_inf,  w, label='RF',  color='#2E75B6', edgecolor='white')
ax2.bar(x+w/2, mlp_inf, w, label='MLP', color='#C55A11', edgecolor='white')
ax2.set_title('Inference Speed (ms per sample)', fontweight='bold')
ax2.set_xticks(x); ax2.set_xticklabels(tasks)
ax2.set_ylabel('Milliseconds'); ax2.legend(); ax2.grid(axis='y', alpha=0.25)
ax2.spines[['top','right']].set_visible(False)
for i, (rv, mv) in enumerate(zip(rf_inf, mlp_inf)):
    ax2.text(i-w/2, rv+0.001, f'{rv:.4f}', ha='center', fontsize=8,
             color='#2E75B6', fontweight='bold')
    ax2.text(i+w/2, mv+0.001, f'{mv:.4f}', ha='center', fontsize=8,
             color='#C55A11', fontweight='bold')

plt.tight_layout()
plt.savefig(p6('benchmark_speed.png'), dpi=250, bbox_inches='tight',
            facecolor='white')
plt.close()
print("  Saved -> benchmark_speed.png")

# ============================================================================
print("\n" + "="*80)
print("PHASE 6a COMPLETE!")
print("="*80)
print("\n  Files saved to PHASE_6/:")
print("  benchmark_table.png       <- Master comparison table")
print("  benchmark_bar_chart.png   <- Binary & multi-class bar charts")
print("  benchmark_heatmap.png     <- Per-class F1 heatmap")
print("  benchmark_speed.png       <- Training & inference speed")
print("\n  Use these images directly in your proposal document!")
print("="*80)
