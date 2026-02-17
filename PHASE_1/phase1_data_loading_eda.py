"""
Phase 1: Data Loading & Exploratory Data Analysis (EDA)
Dataset: NF-UNSW-NB15-v2
Author: Network Intrusion Detection Experiment
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Set display options for better readability
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

print("="*80)
print("PHASE 1: DATA LOADING & EXPLORATORY DATA ANALYSIS")
print("="*80)

# ============================================================================
# STEP 1: Load the Dataset
# ============================================================================
print("\n[STEP 1] Loading Dataset...")
print("-"*80)

# Update this path to where you saved your CSV file
dataset_path = '/home/kiminarii/Documents/forenxai/fe6cb615d161452c_MOHANAD_A4706/data/NF-UNSW-NB15-v2.csv'  # Change this to your actual file path

try:
    df = pd.read_csv(dataset_path)
    print(f"✓ Dataset loaded successfully!")
    print(f"✓ Dataset shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
except FileNotFoundError:
    print(f"✗ Error: File not found at '{dataset_path}'")
    print("Please update the 'dataset_path' variable with the correct path.")
    exit()

# ============================================================================
# STEP 2: Initial Inspection
# ============================================================================
print("\n[STEP 2] Initial Data Inspection")
print("-"*80)

print("\n📋 First 5 rows:")
print(df.head())

print("\n📋 Column names and data types:")
print(df.dtypes)

print(f"\n📋 Total columns: {len(df.columns)}")
print(f"Column names: {list(df.columns)}")

# ============================================================================
# STEP 3: Check for Missing Values
# ============================================================================
print("\n[STEP 3] Missing Values Analysis")
print("-"*80)

missing_values = df.isnull().sum()
missing_percentage = (missing_values / len(df)) * 100

missing_df = pd.DataFrame({
    'Missing Count': missing_values,
    'Percentage': missing_percentage
})

# Show only columns with missing values
missing_df_filtered = missing_df[missing_df['Missing Count'] > 0]

if len(missing_df_filtered) > 0:
    print("\n⚠ Columns with missing values:")
    print(missing_df_filtered)
else:
    print("\n✓ No missing values found!")

# ============================================================================
# STEP 4: Check for Duplicates
# ============================================================================
print("\n[STEP 4] Duplicate Records Analysis")
print("-"*80)

duplicates = df.duplicated().sum()
print(f"Number of duplicate rows: {duplicates:,}")

if duplicates > 0:
    print(f"⚠ Warning: {duplicates:,} duplicate rows found ({(duplicates/len(df))*100:.2f}%)")
else:
    print("✓ No duplicate rows found!")

# ============================================================================
# STEP 5: Class Distribution Analysis
# ============================================================================
print("\n[STEP 5] Class Distribution Analysis")
print("-"*80)

# Assuming the label column is named 'Attack' or 'Label' - adjust if different
# Common names: 'Attack', 'Label', 'attack_cat', 'label'
label_column = None
possible_label_names = ['Attack', 'Label', 'attack_cat', 'label', 'Attack_type', 'Class']

for col in possible_label_names:
    if col in df.columns:
        label_column = col
        break

if label_column is None:
    print("⚠ Warning: Could not automatically detect label column.")
    print("Available columns:", list(df.columns))
    print("\nPlease check the column names and update 'label_column' variable manually.")
else:
    print(f"✓ Label column detected: '{label_column}'")
    
    print("\n📊 Class Distribution:")
    class_counts = df[label_column].value_counts()
    print(class_counts)
    
    print("\n📊 Class Distribution (Percentage):")
    class_percentages = df[label_column].value_counts(normalize=True) * 100
    print(class_percentages)
    
    # Visualize class distribution
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    class_counts.plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title('Class Distribution (Count)', fontsize=14, fontweight='bold')
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    plt.subplot(1, 2, 2)
    class_percentages.plot(kind='bar', color='lightcoral', edgecolor='black')
    plt.title('Class Distribution (Percentage)', fontsize=14, fontweight='bold')
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('Percentage (%)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    plt.savefig('class_distribution.png', dpi=300, bbox_inches='tight')
    print("\n✓ Class distribution plot saved as 'class_distribution.png'")
    
    # Check for class imbalance
    print("\n⚖ Class Imbalance Check:")
    if len(class_counts) > 1:
        imbalance_ratio = class_counts.max() / class_counts.min()
        print(f"Imbalance ratio (max/min): {imbalance_ratio:.2f}")
        
        if imbalance_ratio > 10:
            print("⚠ Significant class imbalance detected! Consider using:")
            print("  - Class weights")
            print("  - SMOTE (Synthetic Minority Over-sampling)")
            print("  - Stratified sampling")
        else:
            print("✓ Classes are relatively balanced")

# ============================================================================
# STEP 6: Statistical Summary
# ============================================================================
print("\n[STEP 6] Statistical Summary")
print("-"*80)

print("\n📊 Numerical Features Summary:")
print(df.describe())

# Check for infinite values
print("\n🔍 Checking for infinite values...")
inf_counts = np.isinf(df.select_dtypes(include=[np.number])).sum()
inf_columns = inf_counts[inf_counts > 0]

if len(inf_columns) > 0:
    print("\n⚠ Columns with infinite values:")
    print(inf_columns)
else:
    print("✓ No infinite values found!")

# ============================================================================
# STEP 7: Data Type Analysis
# ============================================================================
print("\n[STEP 7] Data Type Analysis")
print("-"*80)

print("\n📋 Feature types breakdown:")
print(f"Numerical features: {len(df.select_dtypes(include=[np.number]).columns)}")
print(f"Categorical features: {len(df.select_dtypes(include=['object']).columns)}")

categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
if len(categorical_cols) > 0:
    print(f"\nCategorical columns: {categorical_cols}")

# ============================================================================
# STEP 8: Save Summary Report
# ============================================================================
print("\n[STEP 8] Saving Summary Report")
print("-"*80)

with open('eda_summary_report.txt', 'w') as f:
    f.write("="*80 + "\n")
    f.write("EXPLORATORY DATA ANALYSIS SUMMARY REPORT\n")
    f.write("Dataset: NF-UNSW-NB15-v2\n")
    f.write("="*80 + "\n\n")
    
    f.write(f"Dataset Shape: {df.shape[0]:,} rows × {df.shape[1]} columns\n\n")
    
    f.write("Missing Values:\n")
    f.write(str(missing_df_filtered) + "\n\n")
    
    f.write(f"Duplicate Rows: {duplicates:,}\n\n")
    
    if label_column:
        f.write("Class Distribution:\n")
        f.write(str(class_counts) + "\n\n")
        f.write("Class Distribution (%):\n")
        f.write(str(class_percentages) + "\n\n")
    
    f.write("Statistical Summary:\n")
    f.write(str(df.describe()) + "\n")

print("✓ Summary report saved as 'eda_summary_report.txt'")

# ============================================================================
# STEP 9: Key Findings & Recommendations
# ============================================================================
print("\n" + "="*80)
print("KEY FINDINGS & NEXT STEPS")
print("="*80)

print("\n✅ EDA Complete! Here's what we found:")
print(f"  • Total records: {df.shape[0]:,}")
print(f"  • Total features: {df.shape[1]}")
print(f"  • Missing values: {'Yes' if missing_values.sum() > 0 else 'No'}")
print(f"  • Duplicates: {'Yes' if duplicates > 0 else 'No'}")
if label_column:
    print(f"  • Number of classes: {len(class_counts)}")
    print(f"  • Class imbalance: {'Yes' if imbalance_ratio > 10 else 'Moderate' if imbalance_ratio > 3 else 'No'}")

print("\n📌 NEXT STEPS:")
print("  1. Review the 'eda_summary_report.txt' file")
print("  2. Check the 'class_distribution.png' visualization")
print("  3. Proceed to Phase 2: Data Cleaning & Preprocessing")

print("\n" + "="*80)
print("Phase 1 Complete!")
print("="*80)