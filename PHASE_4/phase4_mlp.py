"""
Phase 4: MLP (Deep Learning) Training - scikit-learn version
Dataset: NF-UNSW-NB15-v2

WHY scikit-learn MLP instead of TensorFlow:
  - TensorFlow 2.6+ requires AVX2 CPU instructions
  - Pentium Silver N5000 does NOT support AVX2
  - sklearn MLPClassifier runs on ANY CPU, no issues
  - Still a valid deep learning benchmark vs Random Forest

Memory strategy:
  - Uses partial_fit() to train in chunks (never loads full dataset)
  - Safe for 4GB RAM systems
  - Evaluation done in chunks too

Folder structure:
  src/
  ├── PHASE_2/  <- input files
  ├── PHASE_3/  <- RF results
  └── PHASE_4/  <- run from here, outputs saved here
"""

import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
    roc_auc_score
)
import joblib
import time
import gc
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

print("="*80)
print("PHASE 4: MLP TRAINING (scikit-learn - CPU Compatible)")
print("="*80)

# ============================================================================
# PATHS
# ============================================================================
PHASE2_DIR   = '../PHASE_2/'
PHASE4_DIR   = './'
SUMMARY_FILE = os.path.join(PHASE4_DIR, 'mlp_results_summary.txt')

def p2(f): return os.path.join(PHASE2_DIR, f)
def p4(f): return os.path.join(PHASE4_DIR, f)

def append_summary(text):
    with open(SUMMARY_FILE, 'a') as f:
        f.write(text)

# ============================================================================
# CHECK Phase 2 files
# ============================================================================
print("\n[CHECK] Verifying Phase 2 files...")
print("-"*80)

required = [
    'train_binary.csv', 'test_binary.csv',
    'train_multiclass.csv', 'test_multiclass.csv',
    'label_encoder.pkl', 'feature_names.pkl',
    'class_weights_binary.pkl', 'class_weights_multiclass.pkl',
]

all_found = True
for f in required:
    exists = os.path.exists(p2(f))
    size   = os.path.getsize(p2(f)) / (1024*1024) if exists else 0
    status = f"OK ({size:.1f} MB)" if exists else "MISSING"
    print(f"  {f:<35} {status}")
    if not exists:
        all_found = False

if not all_found:
    print(f"\n  ERROR: Missing files. Check PHASE2_DIR = '{os.path.abspath(PHASE2_DIR)}'")
    exit()

print(f"\n  All files found!")
print(f"  Reading from : {os.path.abspath(PHASE2_DIR)}")
print(f"  Saving to    : {os.path.abspath(PHASE4_DIR)}")

# ============================================================================
# STEP 1: Load metadata
# ============================================================================
print("\n[STEP 1] Loading Metadata...")
print("-"*80)

label_encoder  = joblib.load(p2('label_encoder.pkl'))
feature_names  = joblib.load(p2('feature_names.pkl'))
class_names    = label_encoder.classes_
n_features     = len(feature_names)
n_classes      = len(class_names)

print(f"  Features : {n_features}")
print(f"  Classes  : {n_classes} -> {list(class_names)}")

# ============================================================================
# Init summary file
# ============================================================================
with open(SUMMARY_FILE, 'w') as f:
    f.write("="*80 + "\n")
    f.write("MLP RESULTS SUMMARY (scikit-learn MLPClassifier)\n")
    f.write("Dataset : NF-UNSW-NB15-v2\n")
    f.write("Input   : ../PHASE_2/\n")
    f.write("Output  : ./PHASE_4/\n")
    f.write("="*80 + "\n\n")
    f.write("NOTE: Using sklearn MLPClassifier (TensorFlow incompatible\n")
    f.write("      with Pentium Silver N5000 - no AVX2 support)\n\n")
    f.write("MEMORY STRATEGY\n")
    f.write("-"*40 + "\n")
    f.write("  Uses partial_fit() - trains one chunk at a time\n")
    f.write("  Never loads full dataset into RAM\n")
    f.write("  Safe peak RAM: ~500MB-1GB\n\n")

print("\n  Summary file initialised -> mlp_results_summary.txt")

# ============================================================================
# Helpers
# ============================================================================
CHUNK_SIZE = 100_000  # 100k rows per chunk (~40MB per chunk in RAM)

def save_confusion_matrix(cm, labels, title, filepath):
    plt.figure(figsize=(max(8, len(labels)), max(6, len(labels)-1)))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges',
                xticklabels=labels, yticklabels=labels)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel('Actual', fontsize=12)
    plt.xlabel('Predicted', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(filepath, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filepath}")

def plot_loss_curve(loss_curve, title, filepath):
    """Plot the MLP training loss curve."""
    plt.figure(figsize=(10, 5))
    plt.plot(loss_curve, color='blue', linewidth=2)
    plt.title(f'{title} - Training Loss Curve', fontsize=14, fontweight='bold')
    plt.xlabel('Iterations', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filepath, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filepath}")

def train_mlp_chunked(model, filepath, label_col, all_classes, chunk_size=CHUNK_SIZE):
    """
    Train MLP using partial_fit() chunk by chunk.
    Goes through the dataset multiple times (epochs).
    Never loads full dataset into RAM.
    """
    EPOCHS = 5   # Number of passes through the full dataset

    print(f"  Training for {EPOCHS} passes through dataset...")
    print(f"  Chunk size: {chunk_size:,} rows | ~{chunk_size*41*4/1024/1024:.0f}MB per chunk")

    for epoch in range(EPOCHS):
        epoch_start = time.time()
        total = 0
        skipped = 0

        for chunk in pd.read_csv(filepath, chunksize=chunk_size):
            y_chunk = chunk[label_col].values
            X_chunk = chunk.drop(columns=[label_col]).values.astype('float32')

            # Skip chunks missing some classes (e.g. pure-Benign chunks)
            # This avoids the warm_start class mismatch error in partial_fit
            if not set(all_classes).issubset(set(np.unique(y_chunk))):
                skipped += len(chunk)
                del chunk, X_chunk, y_chunk
                gc.collect()
                continue

            model.partial_fit(X_chunk, y_chunk, classes=all_classes)

            total += len(chunk)
            print(f"  Epoch {epoch+1}/{EPOCHS} | trained: {total:,} "
                  f"| skipped: {skipped:,} (missing classes)...", end='\r')

            del chunk, X_chunk, y_chunk
            gc.collect()

        epoch_time = time.time() - epoch_start
        loss = model.loss_ if hasattr(model, 'loss_') else 'N/A'
        print(f"  Epoch {epoch+1}/{EPOCHS} complete | "
              f"Time: {epoch_time:.1f}s | Loss: {loss:.6f}" if isinstance(loss, float)
              else f"  Epoch {epoch+1}/{EPOCHS} complete | Time: {epoch_time:.1f}s")
        print(f"  Epoch {epoch+1} summary: {total:,} trained | {skipped:,} skipped")

    return model

def evaluate_chunked(model, filepath, label_col, is_binary, chunk_size=CHUNK_SIZE):
    """
    Evaluate model chunk by chunk - no RAM crash.
    Returns y_true, y_pred, y_proba.
    """
    y_true_list, y_pred_list, y_proba_list = [], [], []
    total = 0

    for chunk in pd.read_csv(filepath, chunksize=chunk_size):
        y_true_chunk = chunk[label_col].values
        X_chunk      = chunk.drop(columns=[label_col]).values.astype('float32')

        y_pred_chunk  = model.predict(X_chunk)
        y_proba_chunk = model.predict_proba(X_chunk)

        y_true_list.append(y_true_chunk)
        y_pred_list.append(y_pred_chunk)
        y_proba_list.append(y_proba_chunk)

        total += len(chunk)
        print(f"  Evaluated {total:,} rows...", end='\r')

        del chunk, X_chunk, y_pred_chunk
        gc.collect()

    print(f"  Evaluated {total:,} rows total.      ")

    y_true  = np.concatenate(y_true_list)
    y_pred  = np.concatenate(y_pred_list)
    y_proba = np.concatenate(y_proba_list)

    del y_true_list, y_pred_list, y_proba_list
    gc.collect()
    return y_true, y_pred, y_proba

# ============================================================================
# PART A: BINARY CLASSIFICATION
# ============================================================================
print("\n" + "="*80)
print("PART A: BINARY CLASSIFICATION (Benign vs Attack)")
print("="*80)

print("\n[STEP 2A] Building MLP Model (Binary)...")
print("-"*80)

# Architecture: 41 -> 256 -> 128 -> 64 -> 1
# Same architecture as TensorFlow version
mlp_binary = MLPClassifier(
    hidden_layer_sizes = (256, 128, 64),  # 3 hidden layers
    activation         = 'relu',
    solver             = 'adam',
    alpha              = 0.0001,           # L2 regularization (like Dropout)
    batch_size         = 1024,             # Mini-batch size for partial_fit
    learning_rate      = 'adaptive',       # Reduces LR when stuck
    learning_rate_init = 0.001,
    max_iter           = 1,                # 1 iter per partial_fit call
    warm_start         = True,             # Keep weights between partial_fit calls
    random_state       = 42,
    verbose            = False
)

print("  Architecture : 41 -> 256 -> 128 -> 64 -> 1")
print("  Activation   : ReLU")
print("  Solver       : Adam (lr=0.001, adaptive)")
print("  Regularize   : L2 alpha=0.0001")
print("  Batch size   : 1024")
print("  Passes       : 5 through full dataset")

append_summary("MODEL ARCHITECTURE - BINARY\n")
append_summary("-"*40 + "\n")
append_summary("  Framework      : scikit-learn MLPClassifier\n")
append_summary("  Architecture   : 41 -> Dense(256) -> Dense(128) -> Dense(64) -> 1\n")
append_summary("  Activation     : ReLU\n")
append_summary("  Solver         : Adam (lr=0.001, adaptive)\n")
append_summary("  Regularization : L2 alpha=0.0001\n")
append_summary("  Batch size     : 1024\n")
append_summary("  Training passes: 5 (chunked partial_fit)\n\n")

print("\n[STEP 3A] Training MLP (Binary) - chunked partial_fit...")
print("-"*80)

binary_classes = np.array([0, 1])

t0 = time.time()
mlp_binary = train_mlp_chunked(
    mlp_binary,
    p2('train_binary.csv'),
    label_col   = 'Label',
    all_classes = binary_classes
)
train_time_bin = time.time() - t0
print(f"\n  Training complete in {train_time_bin:.1f}s ({train_time_bin/60:.1f} mins)")

joblib.dump(mlp_binary, p4('mlp_binary_model.pkl'))
print(f"  Model saved -> mlp_binary_model.pkl")

# Plot loss curve
if hasattr(mlp_binary, 'loss_curve_'):
    plot_loss_curve(mlp_binary.loss_curve_,
                    'MLP Binary', p4('mlp_binary_loss_curve.png'))

print("\n[STEP 4A] Evaluating Binary (chunked)...")
print("-"*80)

t0 = time.time()
y_test_bin, y_pred_bin, y_proba_bin = evaluate_chunked(
    mlp_binary, p2('test_binary.csv'), 'Label', is_binary=True
)
inf_time_bin = time.time() - t0

acc_bin    = accuracy_score(y_test_bin, y_pred_bin)
prec_bin   = precision_score(y_test_bin, y_pred_bin, zero_division=0)
rec_bin    = recall_score(y_test_bin, y_pred_bin, zero_division=0)
f1_bin     = f1_score(y_test_bin, y_pred_bin, zero_division=0)
roc_bin    = roc_auc_score(y_test_bin, y_proba_bin[:, 1])
cm_bin     = confusion_matrix(y_test_bin, y_pred_bin)
report_bin = classification_report(y_test_bin, y_pred_bin,
               target_names=['Benign','Attack'], zero_division=0)

print(f"\n  Accuracy  : {acc_bin*100:.4f}%")
print(f"  Precision : {prec_bin*100:.4f}%")
print(f"  Recall    : {rec_bin*100:.4f}%")
print(f"  F1-Score  : {f1_bin*100:.4f}%")
print(f"  ROC-AUC   : {roc_bin*100:.4f}%")
print(f"  Train: {train_time_bin:.1f}s  |  Inference: {inf_time_bin:.2f}s")

save_confusion_matrix(cm_bin, ['Benign','Attack'],
    'MLP Binary - Confusion Matrix', p4('mlp_binary_confusion_matrix.png'))

# Write binary results immediately
append_summary("BINARY CLASSIFICATION RESULTS\n")
append_summary("-"*40 + "\n")
append_summary(f"  Accuracy        : {acc_bin*100:.4f}%\n")
append_summary(f"  Precision       : {prec_bin*100:.4f}%\n")
append_summary(f"  Recall          : {rec_bin*100:.4f}%\n")
append_summary(f"  F1-Score        : {f1_bin*100:.4f}%\n")
append_summary(f"  ROC-AUC         : {roc_bin*100:.4f}%\n")
append_summary(f"  Training Time   : {train_time_bin:.2f} seconds\n")
append_summary(f"  Inference Time  : {inf_time_bin:.2f} seconds\n")
append_summary(f"  ms per sample   : {inf_time_bin/len(y_test_bin)*1000:.4f}ms\n\n")
append_summary(f"  Confusion Matrix:\n  {cm_bin}\n\n")
append_summary(f"  Classification Report:\n{report_bin}\n")
print("  Binary results written to summary.")

del y_test_bin, y_pred_bin, y_proba_bin
gc.collect()

# ============================================================================
# PART B: MULTI-CLASS CLASSIFICATION
# ============================================================================
print("\n" + "="*80)
print("PART B: MULTI-CLASS CLASSIFICATION (10 Attack Types)")
print("="*80)

print("\n[STEP 2B] Building MLP Model (Multi-class)...")
print("-"*80)

mlp_multi = MLPClassifier(
    hidden_layer_sizes = (256, 128, 64),
    activation         = 'relu',
    solver             = 'adam',
    alpha              = 0.0001,
    batch_size         = 1024,
    learning_rate      = 'adaptive',
    learning_rate_init = 0.001,
    max_iter           = 1,
    warm_start         = True,
    random_state       = 42,
    verbose            = False
)

print("  Architecture : 41 -> 256 -> 128 -> 64 -> 10")
print("  Activation   : ReLU")
print("  Solver       : Adam (lr=0.001, adaptive)")
print("  Regularize   : L2 alpha=0.0001")
print("  Passes       : 5 through full dataset")

append_summary("MODEL ARCHITECTURE - MULTI-CLASS\n")
append_summary("-"*40 + "\n")
append_summary("  Framework      : scikit-learn MLPClassifier\n")
append_summary(f"  Architecture   : 41 -> Dense(256) -> Dense(128) -> Dense(64) -> {n_classes}\n")
append_summary("  Activation     : ReLU\n")
append_summary("  Solver         : Adam (lr=0.001, adaptive)\n")
append_summary("  Regularization : L2 alpha=0.0001\n")
append_summary("  Batch size     : 1024\n")
append_summary("  Training passes: 5 (chunked partial_fit)\n\n")

print("\n[STEP 3B] Training MLP (Multi-class) - chunked partial_fit...")
print("-"*80)

multi_classes = np.arange(n_classes)

t0 = time.time()
mlp_multi = train_mlp_chunked(
    mlp_multi,
    p2('train_multiclass.csv'),
    label_col   = 'Attack',
    all_classes = multi_classes
)
train_time_multi = time.time() - t0
print(f"\n  Training complete in {train_time_multi:.1f}s ({train_time_multi/60:.1f} mins)")

joblib.dump(mlp_multi, p4('mlp_multiclass_model.pkl'))
print(f"  Model saved -> mlp_multiclass_model.pkl")

if hasattr(mlp_multi, 'loss_curve_'):
    plot_loss_curve(mlp_multi.loss_curve_,
                    'MLP Multi-class', p4('mlp_multiclass_loss_curve.png'))

print("\n[STEP 4B] Evaluating Multi-class (chunked)...")
print("-"*80)

t0 = time.time()
y_test_multi, y_pred_multi, y_proba_multi = evaluate_chunked(
    mlp_multi, p2('test_multiclass.csv'), 'Attack', is_binary=False
)
inf_time_multi = time.time() - t0

acc_multi    = accuracy_score(y_test_multi, y_pred_multi)
prec_multi   = precision_score(y_test_multi, y_pred_multi, average='weighted', zero_division=0)
rec_multi    = recall_score(y_test_multi, y_pred_multi, average='weighted', zero_division=0)
f1_multi     = f1_score(y_test_multi, y_pred_multi, average='weighted', zero_division=0)
f1_macro     = f1_score(y_test_multi, y_pred_multi, average='macro', zero_division=0)
cm_multi     = confusion_matrix(y_test_multi, y_pred_multi)
report_multi = classification_report(y_test_multi, y_pred_multi,
                 target_names=class_names, zero_division=0)

print(f"\n  Accuracy    : {acc_multi*100:.4f}%")
print(f"  Precision   : {prec_multi*100:.4f}% (weighted)")
print(f"  Recall      : {rec_multi*100:.4f}% (weighted)")
print(f"  F1 Weighted : {f1_multi*100:.4f}%")
print(f"  F1 Macro    : {f1_macro*100:.4f}%  <- honest metric")
print(f"  Train: {train_time_multi:.1f}s  |  Inference: {inf_time_multi:.2f}s")
print(f"\n  Per-Class Report:\n{report_multi}")

save_confusion_matrix(cm_multi, class_names,
    'MLP Multi-class - Confusion Matrix', p4('mlp_multiclass_confusion_matrix.png'))

append_summary("MULTI-CLASS CLASSIFICATION RESULTS\n")
append_summary("-"*40 + "\n")
append_summary(f"  Accuracy        : {acc_multi*100:.4f}%\n")
append_summary(f"  Precision       : {prec_multi*100:.4f}% (weighted)\n")
append_summary(f"  Recall          : {rec_multi*100:.4f}% (weighted)\n")
append_summary(f"  F1 Weighted     : {f1_multi*100:.4f}%\n")
append_summary(f"  F1 Macro        : {f1_macro*100:.4f}%  <- honest metric\n")
append_summary(f"  Training Time   : {train_time_multi:.2f} seconds\n")
append_summary(f"  Inference Time  : {inf_time_multi:.2f} seconds\n")
append_summary(f"  ms per sample   : {inf_time_multi/len(y_test_multi)*1000:.4f}ms\n\n")
append_summary(f"  Confusion Matrix:\n  {cm_multi}\n\n")
append_summary(f"  Classification Report:\n{report_multi}\n")
print("  Multi-class results written to summary.")

del y_test_multi, y_pred_multi, y_proba_multi
gc.collect()

# ============================================================================
# Finalise summary
# ============================================================================
append_summary("="*80 + "\n")
append_summary("OUTPUT FILES (saved in PHASE_4/)\n")
append_summary("-"*40 + "\n")
append_summary("  mlp_binary_model.pkl\n")
append_summary("  mlp_multiclass_model.pkl\n")
append_summary("  mlp_binary_confusion_matrix.png\n")
append_summary("  mlp_multiclass_confusion_matrix.png\n")
append_summary("  mlp_binary_loss_curve.png\n")
append_summary("  mlp_multiclass_loss_curve.png\n")
append_summary("  mlp_results_summary.txt\n")
append_summary("\n" + "="*80 + "\n")
append_summary("NEXT STEPS\n")
append_summary("-"*40 + "\n")
append_summary("  Phase 5: Compare RF vs MLP side by side\n")
append_summary("="*80 + "\n")

# ============================================================================
# Final print
# ============================================================================
print("\n" + "="*80)
print("PHASE 4 COMPLETE - MLP DONE!")
print("="*80)
print(f"\n  {'Metric':<15} {'Binary':>12} {'Multi-class':>12}")
print(f"  {'-'*42}")
print(f"  {'Accuracy':<15} {acc_bin*100:>11.4f}% {acc_multi*100:>11.4f}%")
print(f"  {'Precision':<15} {prec_bin*100:>11.4f}% {prec_multi*100:>11.4f}%")
print(f"  {'Recall':<15} {rec_bin*100:>11.4f}% {rec_multi*100:>11.4f}%")
print(f"  {'F1 Weighted':<15} {f1_bin*100:>11.4f}% {f1_multi*100:>11.4f}%")
print(f"  {'F1 Macro':<15} {'N/A':>12} {f1_macro*100:>11.4f}%")
print(f"  {'Train Time':<15} {train_time_bin:>10.1f}s {train_time_multi:>10.1f}s")

print("\n  Files in PHASE_4/:")
print("  mlp_binary_model.pkl")
print("  mlp_multiclass_model.pkl")
print("  mlp_binary_confusion_matrix.png")
print("  mlp_multiclass_confusion_matrix.png")
print("  mlp_binary_loss_curve.png")
print("  mlp_multiclass_loss_curve.png")
print("  mlp_results_summary.txt")
print("\n  NEXT: Phase 5 - RF vs MLP Comparison!")
print("="*80)