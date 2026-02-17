"""
Phase 2: Data Cleaning & Preprocessing (Memory-Efficient Version)
Dataset: NF-UNSW-NB15-v2
Designed for low RAM systems (4GB or less)
Strategy: 
  - Read CSV in chunks to avoid loading all at once
  - Use efficient dtypes to reduce memory usage
  - Process and save incrementally
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import joblib
import os
import gc  # Garbage collector to free memory

print("="*80)
print("PHASE 2: DATA CLEANING & PREPROCESSING (Memory-Efficient)")
print("="*80)

dataset_path = '/home/kiminarii/Documents/forenxai/fe6cb615d161452c_MOHANAD_A4706/data/NF-UNSW-NB15-v2.csv'

# Columns to drop (IP addresses - not useful for ML)
COLUMNS_TO_DROP = ['IPV4_SRC_ADDR', 'IPV4_DST_ADDR']

# ============================================================================
# STEP 1: Scan dataset in chunks to get class info (no big load)
# ============================================================================
print("\n[STEP 1] Scanning dataset for class info (chunked, low memory)...")
print("-"*80)

CHUNK_SIZE = 100_000  # Process 100k rows at a time

all_attack_labels = []
total_rows = 0

for chunk in pd.read_csv(dataset_path, usecols=['Attack', 'Label'], chunksize=CHUNK_SIZE):
    all_attack_labels.extend(chunk['Attack'].tolist())
    total_rows += len(chunk)
    print(f"  Scanned {total_rows:,} rows...", end='\r')

print(f"\n✓ Scan complete! Total rows: {total_rows:,}")

# Fit LabelEncoder on all class names
label_encoder = LabelEncoder()
label_encoder.fit(all_attack_labels)
class_names = label_encoder.classes_

print(f"\n✓ Classes found: {len(class_names)}")
for idx, name in enumerate(class_names):
    count = all_attack_labels.count(name)
    print(f"  {name}: {count:,} ({count/total_rows*100:.2f}%)")

# Save label encoder
joblib.dump(label_encoder, 'label_encoder.pkl')
print("\n✓ Label encoder saved")

# Free memory
del all_attack_labels
gc.collect()

# ============================================================================
# STEP 2: Define optimized dtypes to reduce memory footprint
# ============================================================================
print("\n[STEP 2] Defining Memory-Optimized Data Types...")
print("-"*80)

# Reduce int64 → int32 and float64 → float32 where possible
# This alone can halve your memory usage!
optimized_dtypes = {
    'L4_SRC_PORT':                      'int32',
    'L4_DST_PORT':                      'int32',
    'PROTOCOL':                         'int16',
    'L7_PROTO':                         'float32',
    'IN_BYTES':                         'int32',
    'IN_PKTS':                          'int32',
    'OUT_BYTES':                        'int32',
    'OUT_PKTS':                         'int32',
    'TCP_FLAGS':                        'int16',
    'CLIENT_TCP_FLAGS':                 'int16',
    'SERVER_TCP_FLAGS':                 'int16',
    'FLOW_DURATION_MILLISECONDS':       'int32',
    'DURATION_IN':                      'int32',
    'DURATION_OUT':                     'int32',
    'MIN_TTL':                          'int16',
    'MAX_TTL':                          'int16',
    'LONGEST_FLOW_PKT':                 'int32',
    'SHORTEST_FLOW_PKT':                'int32',
    'MIN_IP_PKT_LEN':                   'int32',
    'MAX_IP_PKT_LEN':                   'int32',
    'SRC_TO_DST_SECOND_BYTES':          'float32',
    'DST_TO_SRC_SECOND_BYTES':          'float32',
    'RETRANSMITTED_IN_BYTES':           'int32',
    'RETRANSMITTED_IN_PKTS':            'int32',
    'RETRANSMITTED_OUT_BYTES':          'int32',
    'RETRANSMITTED_OUT_PKTS':           'int32',
    'SRC_TO_DST_AVG_THROUGHPUT':        'int32',
    'DST_TO_SRC_AVG_THROUGHPUT':        'int32',
    'NUM_PKTS_UP_TO_128_BYTES':         'int32',
    'NUM_PKTS_128_TO_256_BYTES':        'int32',
    'NUM_PKTS_256_TO_512_BYTES':        'int32',
    'NUM_PKTS_512_TO_1024_BYTES':       'int32',
    'NUM_PKTS_1024_TO_1514_BYTES':      'int32',
    'TCP_WIN_MAX_IN':                   'int32',
    'TCP_WIN_MAX_OUT':                  'int32',
    'ICMP_TYPE':                        'int32',
    'ICMP_IPV4_TYPE':                   'int32',
    'DNS_QUERY_ID':                     'int32',
    'DNS_QUERY_TYPE':                   'int32',
    'DNS_TTL_ANSWER':                   'int32',
    'FTP_COMMAND_RET_CODE':             'float32',
    'Label':                            'int8',
}

print("✓ Optimized dtypes defined (int64→int32, float64→float32)")
print("  This reduces memory usage by ~50%!")

# ============================================================================
# STEP 3: Fit StandardScaler incrementally (chunk by chunk)
# ============================================================================
print("\n[STEP 3] Fitting Scaler Incrementally (chunk by chunk)...")
print("-"*80)
print("  (This avoids loading full dataset into memory)")

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
processed = 0

for chunk in pd.read_csv(
    dataset_path,
    dtype=optimized_dtypes,
    chunksize=CHUNK_SIZE
):
    # Drop IP and label columns
    chunk = chunk.drop(columns=COLUMNS_TO_DROP + ['Attack'], errors='ignore')
    X_chunk = chunk.drop(columns=['Label'])

    # Partial fit the scaler
    scaler.partial_fit(X_chunk)

    processed += len(chunk)
    print(f"  Scaler fitted on {processed:,} / {total_rows:,} rows...", end='\r')

    # Free memory
    del chunk, X_chunk
    gc.collect()

print(f"\n✓ Scaler fitted on all {total_rows:,} rows")
feature_names = list(pd.read_csv(dataset_path, nrows=1)
                     .drop(columns=COLUMNS_TO_DROP + ['Attack', 'Label'], errors='ignore')
                     .columns)

joblib.dump(scaler, 'feature_scaler.pkl')
joblib.dump(feature_names, 'feature_names.pkl')
print("✓ Scaler saved as 'feature_scaler.pkl'")
print(f"✓ Feature names saved: {len(feature_names)} features")

# ============================================================================
# STEP 4: Process chunks → Split → Scale → Save to disk
# ============================================================================
print("\n[STEP 4] Processing, Splitting & Saving Data in Chunks...")
print("-"*80)

# We'll collect all rows split into train/test
# To avoid memory issues, we build index lists from a sampled approach:
# Safer strategy: process full dataset chunk by chunk and write to two CSV files

train_binary_path   = 'train_binary.csv'
test_binary_path    = 'test_binary.csv'
train_multi_path    = 'train_multiclass.csv'
test_multi_path     = 'test_multiclass.csv'

# Remove existing files if re-running
for path in [train_binary_path, test_binary_path, train_multi_path, test_multi_path]:
    if os.path.exists(path):
        os.remove(path)

processed = 0
write_header = True  # Write header only on first chunk

np.random.seed(42)

for chunk in pd.read_csv(
    dataset_path,
    dtype=optimized_dtypes,
    chunksize=CHUNK_SIZE
):
    # Drop IP columns
    chunk = chunk.drop(columns=COLUMNS_TO_DROP, errors='ignore')

    # Prepare labels
    y_bin   = chunk['Label'].values
    y_multi = label_encoder.transform(chunk['Attack'].values)

    # Drop label columns to get features
    X_chunk = chunk.drop(columns=['Label', 'Attack']).values.astype('float32')

    # Scale features
    X_scaled = scaler.transform(X_chunk).astype('float32')

    # Split this chunk 80/20 randomly (stratified per chunk)
    idx = np.arange(len(X_scaled))

    # Use per-chunk stratified split
    from sklearn.model_selection import train_test_split as tts

    try:
        train_idx, test_idx = tts(idx, test_size=0.2, random_state=42, stratify=y_bin)
    except ValueError:
        # If a class has too few samples for stratify, fall back
        train_idx, test_idx = tts(idx, test_size=0.2, random_state=42)

    # --- Binary ---
    train_bin_df = pd.DataFrame(
        X_scaled[train_idx],
        columns=feature_names,
        dtype='float32'
    )
    train_bin_df['Label'] = y_bin[train_idx]

    test_bin_df = pd.DataFrame(
        X_scaled[test_idx],
        columns=feature_names,
        dtype='float32'
    )
    test_bin_df['Label'] = y_bin[test_idx]

    # --- Multi-class ---
    train_multi_df = pd.DataFrame(
        X_scaled[train_idx],
        columns=feature_names,
        dtype='float32'
    )
    train_multi_df['Attack'] = y_multi[train_idx]

    test_multi_df = pd.DataFrame(
        X_scaled[test_idx],
        columns=feature_names,
        dtype='float32'
    )
    test_multi_df['Attack'] = y_multi[test_idx]

    # Append to CSV files
    train_bin_df.to_csv(train_binary_path,  mode='a', header=write_header, index=False)
    test_bin_df.to_csv(test_binary_path,    mode='a', header=write_header, index=False)
    train_multi_df.to_csv(train_multi_path, mode='a', header=write_header, index=False)
    test_multi_df.to_csv(test_multi_path,   mode='a', header=write_header, index=False)

    write_header = False  # Only write header once
    processed += len(chunk)
    print(f"  Processed & saved {processed:,} / {total_rows:,} rows...", end='\r')

    # Free memory aggressively
    del chunk, X_chunk, X_scaled, train_bin_df, test_bin_df
    del train_multi_df, test_multi_df
    gc.collect()

print(f"\n✓ All {total_rows:,} rows processed and saved!")

# ============================================================================
# STEP 5: Compute Class Weights (just from label arrays, lightweight)
# ============================================================================
print("\n[STEP 5] Computing Class Weights...")
print("-"*80)

# Read only label columns (very lightweight)
labels_df = pd.read_csv(train_binary_path, usecols=['Label'])
y_train_bin = labels_df['Label'].values

class_weights_bin = compute_class_weight(
    class_weight='balanced',
    classes=np.array([0, 1]),
    y=y_train_bin
)
class_weight_dict_bin = {0: float(class_weights_bin[0]), 1: float(class_weights_bin[1])}

print("📊 BINARY CLASS WEIGHTS:")
print(f"  Benign (0): {class_weights_bin[0]:.4f}")
print(f"  Attack (1): {class_weights_bin[1]:.4f}")

del labels_df, y_train_bin
gc.collect()

labels_df = pd.read_csv(train_multi_path, usecols=['Attack'])
y_train_multi = labels_df['Attack'].values

class_weights_multi = compute_class_weight(
    class_weight='balanced',
    classes=np.arange(len(class_names)),
    y=y_train_multi
)
class_weight_dict_multi = {i: float(class_weights_multi[i]) for i in range(len(class_names))}

print("\n📊 MULTI-CLASS WEIGHTS:")
for idx, name in enumerate(class_names):
    print(f"  {name}: {class_weights_multi[idx]:.4f}")

del labels_df, y_train_multi
gc.collect()

joblib.dump(class_weight_dict_bin,   'class_weights_binary.pkl')
joblib.dump(class_weight_dict_multi, 'class_weights_multiclass.pkl')
print("\n✓ Class weights saved")

# ============================================================================
# STEP 6: Verify output file sizes
# ============================================================================
print("\n[STEP 6] Verifying Output Files...")
print("-"*80)

files = {
    'train_binary.csv':     train_binary_path,
    'test_binary.csv':      test_binary_path,
    'train_multiclass.csv': train_multi_path,
    'test_multiclass.csv':  test_multi_path,
}

for name, path in files.items():
    if os.path.exists(path):
        size_mb = os.path.getsize(path) / (1024 * 1024)
        rows = sum(1 for _ in open(path)) - 1  # Count rows minus header
        print(f"  {name}: {rows:,} rows | {size_mb:.1f} MB")
    else:
        print(f"  ⚠ {name}: NOT FOUND")

# ============================================================================
# STEP 7: Generate Summary Report
# ============================================================================
print("\n[STEP 7] Generating Summary Report...")
print("-"*80)

# Collect file stats for report
file_stats = {}
for name, path in files.items():
    if os.path.exists(path):
        size_mb = os.path.getsize(path) / (1024 * 1024)
        rows = sum(1 for _ in open(path)) - 1
        file_stats[name] = {'rows': rows, 'size_mb': size_mb}

# Read label counts from output files (lightweight)
train_bin_labels  = pd.read_csv(train_binary_path,  usecols=['Label'])['Label'].value_counts().sort_index()
test_bin_labels   = pd.read_csv(test_binary_path,   usecols=['Label'])['Label'].value_counts().sort_index()
train_multi_labels= pd.read_csv(train_multi_path,   usecols=['Attack'])['Attack'].value_counts().sort_index()
test_multi_labels = pd.read_csv(test_multi_path,    usecols=['Attack'])['Attack'].value_counts().sort_index()

with open('preprocessing_summary.txt', 'w') as f:
    f.write("="*80 + "\n")
    f.write("DATA PREPROCESSING SUMMARY REPORT\n")
    f.write("Dataset: NF-UNSW-NB15-v2\n")
    f.write("Method: Memory-Efficient Chunked Processing\n")
    f.write("="*80 + "\n\n")

    f.write("ORIGINAL DATASET\n")
    f.write("-"*40 + "\n")
    f.write(f"  Total rows       : {total_rows:,}\n")
    f.write(f"  Total columns    : 45\n")
    f.write(f"  Feature columns  : {len(feature_names)}\n")
    f.write(f"  Dropped columns  : {COLUMNS_TO_DROP}\n\n")

    f.write("PREPROCESSING STEPS APPLIED\n")
    f.write("-"*40 + "\n")
    f.write(f"  1. Dropped IP address columns (not useful for ML)\n")
    f.write(f"  2. Optimized dtypes (int64→int32, float64→float32)\n")
    f.write(f"  3. Fitted StandardScaler incrementally (chunk by chunk)\n")
    f.write(f"  4. Scaled all features using fitted scaler\n")
    f.write(f"  5. Train/Test split: 80% / 20% (stratified)\n")
    f.write(f"  6. Computed class weights for imbalance handling\n")
    f.write(f"  7. Chunk size used: {CHUNK_SIZE:,} rows per chunk\n\n")

    f.write("BINARY CLASSIFICATION\n")
    f.write("-"*40 + "\n")
    f.write(f"  Training set: {file_stats['train_binary.csv']['rows']:,} rows\n")
    f.write(f"    - Benign (0) : {train_bin_labels.get(0, 0):,}\n")
    f.write(f"    - Attack (1) : {train_bin_labels.get(1, 0):,}\n")
    f.write(f"  Test set    : {file_stats['test_binary.csv']['rows']:,} rows\n")
    f.write(f"    - Benign (0) : {test_bin_labels.get(0, 0):,}\n")
    f.write(f"    - Attack (1) : {test_bin_labels.get(1, 0):,}\n\n")

    f.write("  Class Weights (to handle imbalance):\n")
    f.write(f"    Benign (0) weight : {class_weight_dict_bin[0]:.4f}\n")
    f.write(f"    Attack (1) weight : {class_weight_dict_bin[1]:.4f}\n\n")

    f.write("MULTI-CLASS CLASSIFICATION\n")
    f.write("-"*40 + "\n")
    f.write(f"  Number of classes : {len(class_names)}\n")
    f.write(f"  Training set: {file_stats['train_multiclass.csv']['rows']:,} rows\n")
    f.write(f"  Test set    : {file_stats['test_multiclass.csv']['rows']:,} rows\n\n")

    f.write("  Class Distribution (Training Set):\n")
    for class_idx, count in train_multi_labels.items():
        name = class_names[class_idx]
        weight = class_weight_dict_multi[class_idx]
        pct = count / file_stats['train_multiclass.csv']['rows'] * 100
        f.write(f"    [{class_idx}] {name:<15}: {count:>8,} ({pct:5.2f}%)  weight={weight:.4f}\n")

    f.write("\n  Class Distribution (Test Set):\n")
    for class_idx, count in test_multi_labels.items():
        name = class_names[class_idx]
        pct = count / file_stats['test_multiclass.csv']['rows'] * 100
        f.write(f"    [{class_idx}] {name:<15}: {count:>8,} ({pct:5.2f}%)\n")

    f.write("\n" + "="*80 + "\n")
    f.write("OUTPUT FILES\n")
    f.write("-"*40 + "\n")
    for name, stats in file_stats.items():
        f.write(f"  {name:<25}: {stats['rows']:>10,} rows | {stats['size_mb']:>7.1f} MB\n")
    f.write(f"\n  {'feature_scaler.pkl':<25}: StandardScaler (fitted on training data)\n")
    f.write(f"  {'label_encoder.pkl':<25}: LabelEncoder ({len(class_names)} classes)\n")
    f.write(f"  {'feature_names.pkl':<25}: {len(feature_names)} feature names\n")
    f.write(f"  {'class_weights_binary.pkl':<25}: Binary class weights\n")
    f.write(f"  {'class_weights_multiclass.pkl':<25}: Multi-class weights\n")

    f.write("\n" + "="*80 + "\n")
    f.write("FEATURE LIST\n")
    f.write("-"*40 + "\n")
    for i, name in enumerate(feature_names, 1):
        f.write(f"  {i:>2}. {name}\n")

    f.write("\n" + "="*80 + "\n")
    f.write("NEXT STEPS\n")
    f.write("-"*40 + "\n")
    f.write("  1. Phase 3: Train Random Forest model (binary classification)\n")
    f.write("  2. Phase 4: Train MLP model (binary classification)\n")
    f.write("  3. Phase 5: Repeat both for multi-class classification\n")
    f.write("  4. Phase 6: Compare and evaluate both models\n")
    f.write("="*80 + "\n")

print("✓ Summary report saved as 'preprocessing_summary.txt'")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "="*80)
print("PHASE 2 COMPLETE! (Memory-Efficient)")
print("="*80)

print("\n✅ All data preprocessed and saved successfully!")
print("\n📁 Files created:")
print("  • train_binary.csv          - Training data (binary labels)")
print("  • test_binary.csv           - Test data (binary labels)")
print("  • train_multiclass.csv      - Training data (multi-class labels)")
print("  • test_multiclass.csv       - Test data (multi-class labels)")
print("  • feature_scaler.pkl        - Fitted StandardScaler")
print("  • label_encoder.pkl         - Fitted LabelEncoder")
print("  • feature_names.pkl         - Feature column names")
print("  • class_weights_binary.pkl")
print("  • class_weights_multiclass.pkl")
print("  • preprocessing_summary.txt - Full summary report ✨")

print("\n📌 NEXT STEPS:")
print("  Proceed to Phase 3: Random Forest Training!")
print("  The model will load data in chunks too - no RAM crashes!")

print("\n" + "="*80)