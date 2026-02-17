"""
Phase 5: RF vs MLP - Final Comparison & Visualization
Dataset: NF-UNSW-NB15-v2

Reads results from Phase 3 (RF) and Phase 4 (MLP) summaries
and produces a research-ready comparison report + charts.

Folder structure:
  src/
  ├── PHASE_3/  <- RF results
  ├── PHASE_4/  <- MLP results
  └── PHASE_5/  <- run from here, outputs saved here
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

print("="*80)
print("PHASE 5: RF vs MLP FINAL COMPARISON")
print("="*80)

# ============================================================================
# PATHS
# ============================================================================
PHASE3_DIR   = '../PHASE_3/'
PHASE4_DIR   = '../PHASE_4/'
PHASE5_DIR   = './'
SUMMARY_FILE = os.path.join(PHASE5_DIR, 'final_comparison_report.txt')

def p3(f): return os.path.join(PHASE3_DIR, f)
def p4(f): return os.path.join(PHASE4_DIR, f)
def p5(f): return os.path.join(PHASE5_DIR, f)

def append_summary(text):
    with open(SUMMARY_FILE, 'a') as f:
        f.write(text)

# ============================================================================
# RESULTS — copied from Phase 3 & Phase 4 summaries
# ============================================================================

results = {

    # ── BINARY ──────────────────────────────────────────────────────────────
    'binary': {
        'RF': {
            'accuracy'   : 99.6390,
            'precision'  : 91.7980,
            'recall'     : 99.8422,
            'f1'         : 95.6513,
            'roc_auc'    : 99.9779,
            'train_time' : 299.69,
            'inf_ms'     : 0.0061,
            'cm'         : [[457347, 1696], [30, 18982]],
        },
        'MLP': {
            'accuracy'   : 99.6021,
            'precision'  : 92.0104,
            'recall'     : 98.5535,
            'f1'         : 95.1696,
            'roc_auc'    : 99.9517,
            'train_time' : 324.93,
            'inf_ms'     : 0.0453,
            'cm'         : [[457416, 1627], [275, 18737]],
        },
    },

    # ── MULTI-CLASS ─────────────────────────────────────────────────────────
    'multiclass': {
        'RF': {
            'accuracy'   : 98.8518,
            'precision'  : 99.2959,
            'recall'     : 98.8518,
            'f1_weighted': 99.0454,
            'f1_macro'   : 68.69,
            'train_time' : 267.54,
            'inf_ms'     : 0.0097,
            # Per-class F1
            'per_class_f1': {
                'Analysis'      : 0.16,
                'Backdoor'      : 0.15,
                'Benign'        : 1.00,
                'DoS'           : 0.47,
                'Exploits'      : 0.86,
                'Fuzzers'       : 0.84,
                'Generic'       : 0.91,
                'Reconnaissance': 0.83,
                'Shellcode'     : 0.92,
                'Worms'         : 0.76,
            },
        },
        'MLP': {
            'accuracy'   : 98.6847,
            'precision'  : 98.7802,
            'recall'     : 98.6847,
            'f1_weighted': 98.7048,
            'f1_macro'   : 54.93,
            'train_time' : 341.19,
            'inf_ms'     : 0.0442,
            'per_class_f1': {
                'Analysis'      : 0.14,
                'Backdoor'      : 0.19,
                'Benign'        : 1.00,
                'DoS'           : 0.15,
                'Exploits'      : 0.79,
                'Fuzzers'       : 0.77,
                'Generic'       : 0.84,
                'Reconnaissance': 0.80,
                'Shellcode'     : 0.82,
                'Worms'         : 0.00,
            },
        },
    },
}

class_names = list(results['multiclass']['RF']['per_class_f1'].keys())

# ============================================================================
# CHART 1: Binary Classification Comparison Bar Chart
# ============================================================================
print("\n[CHART 1] Binary Classification Comparison...")

metrics_bin  = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
rf_vals_bin  = [
    results['binary']['RF']['accuracy'],
    results['binary']['RF']['precision'],
    results['binary']['RF']['recall'],
    results['binary']['RF']['f1'],
    results['binary']['RF']['roc_auc'],
]
mlp_vals_bin = [
    results['binary']['MLP']['accuracy'],
    results['binary']['MLP']['precision'],
    results['binary']['MLP']['recall'],
    results['binary']['MLP']['f1'],
    results['binary']['MLP']['roc_auc'],
]

x     = np.arange(len(metrics_bin))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 6))
bars_rf  = ax.bar(x - width/2, rf_vals_bin,  width, label='Random Forest',
                  color='steelblue', edgecolor='black', alpha=0.85)
bars_mlp = ax.bar(x + width/2, mlp_vals_bin, width, label='MLP',
                  color='coral',     edgecolor='black', alpha=0.85)

ax.set_title('Binary Classification: RF vs MLP', fontsize=15, fontweight='bold')
ax.set_ylabel('Score (%)', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(metrics_bin, fontsize=11)
ax.set_ylim(88, 101)
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)

# Value labels on bars
for bar in bars_rf:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
            f'{bar.get_height():.2f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')
for bar in bars_mlp:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
            f'{bar.get_height():.2f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')

plt.tight_layout()
plt.savefig(p5('chart1_binary_comparison.png'), dpi=200, bbox_inches='tight')
plt.close()
print("  Saved -> chart1_binary_comparison.png")

# ============================================================================
# CHART 2: Multi-class Key Metrics Comparison
# ============================================================================
print("\n[CHART 2] Multi-class Key Metrics Comparison...")

metrics_multi  = ['Accuracy', 'Precision\n(weighted)', 'Recall\n(weighted)',
                  'F1 Weighted', 'F1 Macro\n(honest)']
rf_vals_multi  = [
    results['multiclass']['RF']['accuracy'],
    results['multiclass']['RF']['precision'],
    results['multiclass']['RF']['recall'],
    results['multiclass']['RF']['f1_weighted'],
    results['multiclass']['RF']['f1_macro'],
]
mlp_vals_multi = [
    results['multiclass']['MLP']['accuracy'],
    results['multiclass']['MLP']['precision'],
    results['multiclass']['MLP']['recall'],
    results['multiclass']['MLP']['f1_weighted'],
    results['multiclass']['MLP']['f1_macro'],
]

x     = np.arange(len(metrics_multi))
width = 0.35

fig, ax = plt.subplots(figsize=(13, 6))
bars_rf  = ax.bar(x - width/2, rf_vals_multi,  width, label='Random Forest',
                  color='steelblue', edgecolor='black', alpha=0.85)
bars_mlp = ax.bar(x + width/2, mlp_vals_multi, width, label='MLP',
                  color='coral',     edgecolor='black', alpha=0.85)

ax.set_title('Multi-class Classification: RF vs MLP', fontsize=15, fontweight='bold')
ax.set_ylabel('Score (%)', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(metrics_multi, fontsize=10)
ax.set_ylim(40, 103)
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)

# Highlight the F1 Macro bar (honest metric)
ax.axvspan(3.5, 4.5, alpha=0.08, color='gold', label='Honest metric')
ax.text(4, 42, '← Most honest\n    metric', ha='center', fontsize=9,
        color='darkorange', fontweight='bold')

for bar in bars_rf:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f'{bar.get_height():.2f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')
for bar in bars_mlp:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f'{bar.get_height():.2f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')

plt.tight_layout()
plt.savefig(p5('chart2_multiclass_comparison.png'), dpi=200, bbox_inches='tight')
plt.close()
print("  Saved -> chart2_multiclass_comparison.png")

# ============================================================================
# CHART 3: Per-Class F1 Score Comparison (Multi-class)
# ============================================================================
print("\n[CHART 3] Per-Class F1 Score Comparison...")

rf_f1_per_class  = [results['multiclass']['RF']['per_class_f1'][c]  for c in class_names]
mlp_f1_per_class = [results['multiclass']['MLP']['per_class_f1'][c] for c in class_names]

x     = np.arange(len(class_names))
width = 0.35

fig, ax = plt.subplots(figsize=(14, 6))
bars_rf  = ax.bar(x - width/2, rf_f1_per_class,  width, label='Random Forest',
                  color='steelblue', edgecolor='black', alpha=0.85)
bars_mlp = ax.bar(x + width/2, mlp_f1_per_class, width, label='MLP',
                  color='coral',     edgecolor='black', alpha=0.85)

ax.set_title('Per-Class F1 Score: RF vs MLP (Multi-class)', fontsize=14, fontweight='bold')
ax.set_ylabel('F1 Score', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(class_names, rotation=30, ha='right', fontsize=10)
ax.set_ylim(0, 1.12)
ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.4, label='0.5 threshold')
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)

for bar in bars_rf:
    if bar.get_height() > 0:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=7.5)
for bar in bars_mlp:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=7.5)

plt.tight_layout()
plt.savefig(p5('chart3_perclass_f1_comparison.png'), dpi=200, bbox_inches='tight')
plt.close()
print("  Saved -> chart3_perclass_f1_comparison.png")

# ============================================================================
# CHART 4: Training Time & Inference Speed Comparison
# ============================================================================
print("\n[CHART 4] Training Time & Inference Speed...")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Training time
categories  = ['Binary', 'Multi-class']
rf_times    = [results['binary']['RF']['train_time'],
               results['multiclass']['RF']['train_time']]
mlp_times   = [results['binary']['MLP']['train_time'],
               results['multiclass']['MLP']['train_time']]

x     = np.arange(len(categories))
width = 0.35
axes[0].bar(x - width/2, rf_times,  width, label='RF',  color='steelblue',
            edgecolor='black', alpha=0.85)
axes[0].bar(x + width/2, mlp_times, width, label='MLP', color='coral',
            edgecolor='black', alpha=0.85)
axes[0].set_title('Training Time (seconds)', fontsize=13, fontweight='bold')
axes[0].set_ylabel('Seconds', fontsize=11)
axes[0].set_xticks(x)
axes[0].set_xticklabels(categories)
axes[0].legend()
axes[0].grid(axis='y', alpha=0.3)
for i, (rv, mv) in enumerate(zip(rf_times, mlp_times)):
    axes[0].text(i - width/2, rv + 3, f'{rv:.0f}s', ha='center', fontsize=9, fontweight='bold')
    axes[0].text(i + width/2, mv + 3, f'{mv:.0f}s', ha='center', fontsize=9, fontweight='bold')

# Inference speed (ms per sample)
rf_inf  = [results['binary']['RF']['inf_ms'],
           results['multiclass']['RF']['inf_ms']]
mlp_inf = [results['binary']['MLP']['inf_ms'],
           results['multiclass']['MLP']['inf_ms']]

axes[1].bar(x - width/2, rf_inf,  width, label='RF',  color='steelblue',
            edgecolor='black', alpha=0.85)
axes[1].bar(x + width/2, mlp_inf, width, label='MLP', color='coral',
            edgecolor='black', alpha=0.85)
axes[1].set_title('Inference Speed (ms per sample)', fontsize=13, fontweight='bold')
axes[1].set_ylabel('Milliseconds', fontsize=11)
axes[1].set_xticks(x)
axes[1].set_xticklabels(categories)
axes[1].legend()
axes[1].grid(axis='y', alpha=0.3)
for i, (rv, mv) in enumerate(zip(rf_inf, mlp_inf)):
    axes[1].text(i - width/2, rv + 0.001, f'{rv:.4f}', ha='center', fontsize=9, fontweight='bold')
    axes[1].text(i + width/2, mv + 0.001, f'{mv:.4f}', ha='center', fontsize=9, fontweight='bold')

plt.suptitle('Computational Performance: RF vs MLP', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(p5('chart4_performance_comparison.png'), dpi=200, bbox_inches='tight')
plt.close()
print("  Saved -> chart4_performance_comparison.png")

# ============================================================================
# CHART 5: Radar Chart - Overall Model Comparison
# ============================================================================
print("\n[CHART 5] Radar Chart - Overall Comparison...")

# Normalize all metrics to 0-1 scale for radar
# Using binary + multiclass macro F1 + inference speed (inverted)
radar_labels = [
    'Binary F1', 'Binary\nROC-AUC', 'Multi F1\n(Macro)',
    'Multi F1\n(Weighted)', 'Inference\nSpeed'
]

# Inference speed score: lower ms = better, normalize as (1 - ms/max_ms)
max_inf = max(results['binary']['RF']['inf_ms'], results['binary']['MLP']['inf_ms'])
rf_inf_score  = 1 - (results['binary']['RF']['inf_ms']  / (max_inf * 10))
mlp_inf_score = 1 - (results['binary']['MLP']['inf_ms'] / (max_inf * 10))

rf_radar  = [
    results['binary']['RF']['f1']               / 100,
    results['binary']['RF']['roc_auc']          / 100,
    results['multiclass']['RF']['f1_macro']     / 100,
    results['multiclass']['RF']['f1_weighted']  / 100,
    rf_inf_score,
]
mlp_radar = [
    results['binary']['MLP']['f1']              / 100,
    results['binary']['MLP']['roc_auc']         / 100,
    results['multiclass']['MLP']['f1_macro']    / 100,
    results['multiclass']['MLP']['f1_weighted'] / 100,
    mlp_inf_score,
]

N     = len(radar_labels)
theta = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
theta += theta[:1]  # close the circle

rf_radar  += rf_radar[:1]
mlp_radar += mlp_radar[:1]

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
ax.plot(theta, rf_radar,  'o-', linewidth=2, color='steelblue', label='Random Forest')
ax.fill(theta, rf_radar,  alpha=0.15, color='steelblue')
ax.plot(theta, mlp_radar, 's-', linewidth=2, color='coral',     label='MLP')
ax.fill(theta, mlp_radar, alpha=0.15, color='coral')

ax.set_xticks(theta[:-1])
ax.set_xticklabels(radar_labels, fontsize=11)
ax.set_ylim(0, 1)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=12)
ax.set_title('RF vs MLP - Overall Performance Radar',
             fontsize=14, fontweight='bold', pad=20)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(p5('chart5_radar_comparison.png'), dpi=200, bbox_inches='tight')
plt.close()
print("  Saved -> chart5_radar_comparison.png")

# ============================================================================
# FINAL REPORT
# ============================================================================
print("\n[REPORT] Writing Final Comparison Report...")

with open(SUMMARY_FILE, 'w') as f:
    f.write("="*80 + "\n")
    f.write("FINAL COMPARISON REPORT: RANDOM FOREST vs MLP\n")
    f.write("Dataset  : NF-UNSW-NB15-v2 (University of Queensland)\n")
    f.write("Task     : Network Intrusion Detection (Forensic Investigation)\n")
    f.write("Approach : Supervised Learning - Binary & Multi-class Classification\n")
    f.write("="*80 + "\n\n")

    # ── Binary ──────────────────────────────────────────────────────────────
    f.write("BINARY CLASSIFICATION (Benign vs Attack)\n")
    f.write("-"*80 + "\n")
    f.write(f"  {'Metric':<20} {'Random Forest':>16} {'MLP':>16} {'Winner':>10}\n")
    f.write(f"  {'-'*65}\n")

    bin_rows = [
        ('Accuracy (%)',   results['binary']['RF']['accuracy'],
                           results['binary']['MLP']['accuracy']),
        ('Precision (%)',  results['binary']['RF']['precision'],
                           results['binary']['MLP']['precision']),
        ('Recall (%)',     results['binary']['RF']['recall'],
                           results['binary']['MLP']['recall']),
        ('F1-Score (%)',   results['binary']['RF']['f1'],
                           results['binary']['MLP']['f1']),
        ('ROC-AUC (%)',    results['binary']['RF']['roc_auc'],
                           results['binary']['MLP']['roc_auc']),
        ('Train Time (s)', results['binary']['RF']['train_time'],
                           results['binary']['MLP']['train_time'],),
        ('Infer (ms/smp)', results['binary']['RF']['inf_ms'],
                           results['binary']['MLP']['inf_ms']),
    ]

    for row in bin_rows:
        name, rf_v, mlp_v = row
        # For time metrics lower is better
        if 'Time' in name or 'Infer' in name:
            winner = 'RF' if rf_v < mlp_v else 'MLP'
        else:
            winner = 'RF' if rf_v > mlp_v else 'MLP'
        f.write(f"  {name:<20} {rf_v:>16.4f} {mlp_v:>16.4f} {winner:>10}\n")

    f.write(f"\n  {'Verdict':<20} RF wins on F1, Recall, ROC-AUC and speed\n\n")

    # ── Multi-class ──────────────────────────────────────────────────────────
    f.write("MULTI-CLASS CLASSIFICATION (10 Attack Categories)\n")
    f.write("-"*80 + "\n")
    f.write(f"  {'Metric':<20} {'Random Forest':>16} {'MLP':>16} {'Winner':>10}\n")
    f.write(f"  {'-'*65}\n")

    multi_rows = [
        ('Accuracy (%)',      results['multiclass']['RF']['accuracy'],
                              results['multiclass']['MLP']['accuracy']),
        ('Precision-W (%)',   results['multiclass']['RF']['precision'],
                              results['multiclass']['MLP']['precision']),
        ('Recall-W (%)',      results['multiclass']['RF']['recall'],
                              results['multiclass']['MLP']['recall']),
        ('F1 Weighted (%)',   results['multiclass']['RF']['f1_weighted'],
                              results['multiclass']['MLP']['f1_weighted']),
        ('F1 Macro (%) **',  results['multiclass']['RF']['f1_macro'],
                              results['multiclass']['MLP']['f1_macro']),
        ('Train Time (s)',    results['multiclass']['RF']['train_time'],
                              results['multiclass']['MLP']['train_time']),
        ('Infer (ms/smp)',    results['multiclass']['RF']['inf_ms'],
                              results['multiclass']['MLP']['inf_ms']),
    ]

    for row in multi_rows:
        name, rf_v, mlp_v = row
        if 'Time' in name or 'Infer' in name:
            winner = 'RF' if rf_v < mlp_v else 'MLP'
        else:
            winner = 'RF' if rf_v > mlp_v else 'MLP'
        f.write(f"  {name:<20} {rf_v:>16.4f} {mlp_v:>16.4f} {winner:>10}\n")

    f.write(f"\n  ** F1 Macro is the honest metric - treats all classes equally\n")
    f.write(f"     regardless of class size. RF (68.69%) >> MLP (54.93%)\n\n")
    f.write(f"  {'Verdict':<20} RF clearly wins on all metrics\n\n")

    # ── Per-class breakdown ─────────────────────────────────────────────────
    f.write("PER-CLASS F1 SCORE BREAKDOWN\n")
    f.write("-"*80 + "\n")
    f.write(f"  {'Class':<18} {'RF F1':>10} {'MLP F1':>10} {'Winner':>10} {'Note'}\n")
    f.write(f"  {'-'*70}\n")

    for cls in class_names:
        rf_f1  = results['multiclass']['RF']['per_class_f1'][cls]
        mlp_f1 = results['multiclass']['MLP']['per_class_f1'][cls]
        if rf_f1 == mlp_f1:
            winner = 'Tie'
        else:
            winner = 'RF' if rf_f1 > mlp_f1 else 'MLP'

        # Add notes for interesting cases
        note = ''
        if cls == 'Worms':
            note = '<-- MLP completely failed (0.00)'
        elif cls == 'Backdoor' and mlp_f1 > rf_f1:
            note = '<-- MLP slightly better'
        elif cls == 'Benign':
            note = 'Both perfect'
        elif rf_f1 < 0.5 and mlp_f1 < 0.5:
            note = '<-- Both struggle (rare class)'

        f.write(f"  {cls:<18} {rf_f1:>10.2f} {mlp_f1:>10.2f} {winner:>10}  {note}\n")

    # ── Key findings ─────────────────────────────────────────────────────────
    f.write("\n" + "="*80 + "\n")
    f.write("KEY FINDINGS\n")
    f.write("-"*80 + "\n")
    f.write("""
  1. BINARY CLASSIFICATION
     Both models perform excellently (>99% accuracy, >99.9% ROC-AUC).
     RF has a slight edge in F1 (95.65% vs 95.17%) and is 7x faster
     at inference (0.006ms vs 0.045ms per sample).

  2. MULTI-CLASS CLASSIFICATION (honest F1 Macro metric)
     RF significantly outperforms MLP: 68.69% vs 54.93% F1 Macro.
     This confirms RF is better suited for tabular network flow data
     with severe class imbalance.

  3. CLASS IMBALANCE IMPACT
     Both models struggle with rare classes (Analysis, Backdoor, DoS, Worms).
     Worms (30 test samples) was completely missed by MLP (F1=0.00).
     RF handled Worms better (F1=0.76) due to its ensemble voting nature.

  4. FEATURE IMPORTANCE (RF insight)
     Top features: MIN_TTL, MAX_TTL, MIN_IP_PKT_LEN, SHORTEST_FLOW_PKT
     These TTL and packet length features are the strongest indicators
     of malicious traffic in this dataset.

  5. COMPUTATIONAL COST
     Both models have similar training times (~300s each on CPU).
     RF is significantly faster at inference - important for real-time
     forensic detection systems.

  6. FORENSIC INVESTIGATION SUITABILITY
     RF is recommended for this use case because:
     - Higher macro F1 (detects more attack types reliably)
     - Faster inference (real-time detection)
     - Feature importance (explainability for forensic reports)
     - More robust to class imbalance
""")

    # ── Recommendation ───────────────────────────────────────────────────────
    f.write("="*80 + "\n")
    f.write("RECOMMENDATION\n")
    f.write("-"*80 + "\n")
    f.write("""
  For network intrusion detection in forensic investigation:

  WINNER: Random Forest

  Reasons:
  1. Higher F1 Macro (68.69% vs 54.93%) - detects more attack types
  2. Better on rare/minority attack classes critical for forensics
  3. 7x faster inference speed - suitable for real-time detection
  4. Feature importance output - provides explainability for court/reports
  5. More robust to the severe class imbalance in real network data

  MLP is competitive for binary detection but falls behind on the
  more realistic and challenging multi-class scenario.
""")

    # ── Summary table ────────────────────────────────────────────────────────
    f.write("="*80 + "\n")
    f.write("EXPERIMENT SUMMARY\n")
    f.write("-"*80 + "\n")
    f.write(f"  Dataset        : NF-UNSW-NB15-v2\n")
    f.write(f"  Total samples  : 2,390,275\n")
    f.write(f"  Train samples  : 1,912,220 (80%)\n")
    f.write(f"  Test samples   :   478,055 (20%)\n")
    f.write(f"  Features used  : 41 NetFlow features\n")
    f.write(f"  Classes        : 10 (1 Benign + 9 attack types)\n")
    f.write(f"  Imbalance ratio: 13,995:1 (Benign:Worms)\n")
    f.write(f"  Hardware       : Pentium Silver N5000 @ 1.1GHz, 4GB RAM\n")
    f.write(f"  RF library     : scikit-learn RandomForestClassifier\n")
    f.write(f"  MLP library    : scikit-learn MLPClassifier (partial_fit)\n\n")

    f.write("  OUTPUT CHARTS\n")
    f.write(f"  chart1_binary_comparison.png\n")
    f.write(f"  chart2_multiclass_comparison.png\n")
    f.write(f"  chart3_perclass_f1_comparison.png\n")
    f.write(f"  chart4_performance_comparison.png\n")
    f.write(f"  chart5_radar_comparison.png\n")
    f.write("="*80 + "\n")

print("  Saved -> final_comparison_report.txt")

# ============================================================================
# Final print
# ============================================================================
print("\n" + "="*80)
print("PHASE 5 COMPLETE - EXPERIMENT DONE!")
print("="*80)

print("""
  FINAL SCORECARD:
  ┌─────────────────────┬──────────────┬──────────────┬──────────┐
  │ Metric              │ Random Forest│     MLP      │  Winner  │
  ├─────────────────────┼──────────────┼──────────────┼──────────┤
  │ Binary Accuracy     │   99.64%     │   99.60%     │   RF     │
  │ Binary F1           │   95.65%     │   95.17%     │   RF     │
  │ Binary ROC-AUC      │   99.98%     │   99.95%     │   RF     │
  │ Multi F1 (Weighted) │   99.05%     │   98.70%     │   RF     │
  │ Multi F1 (Macro)    │   68.69%     │   54.93%     │   RF     │
  │ Inference Speed     │   0.006ms    │   0.045ms    │   RF     │
  │ Training Time       │   ~284s      │   ~333s      │   RF     │
  └─────────────────────┴──────────────┴──────────────┴──────────┘

  Overall Winner: RANDOM FOREST

  Files saved to PHASE_5/:
  - final_comparison_report.txt
  - chart1_binary_comparison.png
  - chart2_multiclass_comparison.png
  - chart3_perclass_f1_comparison.png
  - chart4_performance_comparison.png
  - chart5_radar_comparison.png
""")
print("="*80)