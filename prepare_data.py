#!/usr/bin/env python3
"""
Data Preparation Script
Processes the Excel file and generates the CSV dataset with features.
"""

import pandas as pd
import os

print("="*70)
print("DATA PREPARATION FOR BLOOD GROUP & GENOTYPE PREDICTION")
print("="*70)
print()

# Create datasets directory
os.makedirs('datasets', exist_ok=True)

# Load the Excel file
print("Loading Excel file...")
try:
    df = pd.read_excel('BLD GRP & GENE (1).xlsx')
    print(f"Loaded successfully: {len(df)} records")
except FileNotFoundError:
    print("ERROR: 'BLD GRP & GENE (1).xlsx' not found in current directory")
    exit(1)
except Exception as e:
    print(f"ERROR: {e}")
    exit(1)

# Clean columns
print("\nCleaning data...")
df['Blood Group'] = df['Blood Group'].astype(str).str.strip().str.replace('＋', '+').str.replace('－', '-')
df['Blood Group'] = df['Blood Group'].str.replace(' ', '')
df['Genotype'] = df['Genotype'].astype(str).str.strip().str.upper()

# Generate blood group test parameters
print("Generating blood group features...")
df['Anti_A'] = df['Blood Group'].apply(
    lambda x: 1 if 'A' in x and 'AB' not in x else (1 if 'AB' in x else 0)
)
df['Anti_B'] = df['Blood Group'].apply(lambda x: 1 if 'B' in x else 0)
df['Anti_D'] = df['Blood Group'].apply(lambda x: 1 if '+' in x else 0)

# Generate genotype test parameters
print("Generating genotype features...")
df['Sickling'] = df['Genotype'].map({
    'AA': 'No',
    'AS': 'Few',
    'SS': 'Yes',
    'AC': 'No',
    'CC': 'No',
    'SC': 'Few'
})

df['Solubility'] = df['Genotype'].map({
    'AA': 'Clear',
    'AS': 'Cloudy',
    'SS': 'Cloudy',
    'AC': 'Clear',
    'CC': 'Clear',
    'SC': 'Cloudy'
})

df['Bands'] = df['Genotype'].map({
    'AA': 'A',
    'AS': 'A and S',
    'SS': 'S',
    'AC': 'A and C',
    'CC': 'C',
    'SC': 'S and C'
})

# Add Sample ID
df.insert(0, 'Sample_ID', range(1, len(df) + 1))

# Select and order columns
df = df[['Sample_ID', 'Blood Group', 'Anti_A', 'Anti_B', 'Anti_D',
         'Genotype', 'Sickling', 'Solubility', 'Bands']]

# Save CSV
output_path = 'datasets/raw_blood_and_genotype_dataset.csv'
df.to_csv(output_path, index=False)

print(f"\nDataset saved: {output_path}")
print(f"\nDataset Summary:")
print(f"  Total samples: {len(df)}")
print(f"  Blood groups: {df['Blood Group'].nunique()} types")
print(f"  Genotypes: {df['Genotype'].nunique()} types")
print()

print("Blood Group Distribution:")
for bg, count in df['Blood Group'].value_counts().items():
    print(f"  {bg}: {count} ({count/len(df)*100:.1f}%)")

print("\nGenotype Distribution:")
for gt, count in df['Genotype'].value_counts().items():
    print(f"  {gt}: {count} ({count/len(df)*100:.1f}%)")

print()
print("="*70)
print("DATA PREPARATION COMPLETE!")
print("="*70)
print("\nNext step: Run 'python3 complete_blood_prediction_system.py'")
print()
