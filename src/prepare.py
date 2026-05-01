import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GroupShuffleSplit
from datetime import datetime
import os
import json

# Buat folder output
os.makedirs('data/prepared', exist_ok=True)
os.makedirs('plots', exist_ok=True)

# ============================================================
# DATA ACQUISITION
# ============================================================
print("="*50)
print("DATA ACQUISITION")
print("="*50)
df = pd.read_csv('data/raw/phishing.csv')
print(f"Total baris  : {df.shape[0]:,}")
print(f"Total kolom  : {df.shape[1]}")
print(f"Missing value: {df.isnull().sum().sum()}")
print(f"Distribusi label:\n{df['label'].value_counts()}")

# ============================================================
# PREPROCESSING
# ============================================================
print("\n" + "="*50)
print("PREPROCESSING")
print("="*50)

# 🔴 FIX: pastikan URL ada sebelum dipakai
if 'URL' not in df.columns:
    raise ValueError("Kolom 'URL' tidak ditemukan, tidak bisa anti-leakage!")

url_backup = df['URL'].copy()

kolom_dibuang = ['FILENAME', 'URL', 'Domain', 'TLD', 'Title']
df_clean = df.drop(columns=kolom_dibuang, errors='ignore')
df_clean = df_clean.select_dtypes(include=[np.number])

X = df_clean.drop(columns=['label'])
y = df_clean['label']

print(f"Fitur (X): {X.shape}")
print(f"Target (y): {y.shape}")

# ============================================================
# SPLITTING (ANTI LEAKAGE)
# ============================================================
print("\n" + "="*50)
print("SPLITTING")
print("="*50)

groups = url_backup

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups))

X_train = X.iloc[train_idx]
y_train = y.iloc[train_idx]

X_test = X.iloc[test_idx]
y_test = y.iloc[test_idx]

# VALIDATION SPLIT
gss2 = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
train_idx2, val_idx = next(gss2.split(X_train, y_train, groups.iloc[train_idx]))

X_val = X_train.iloc[val_idx]
y_val = y_train.iloc[val_idx]

X_train = X_train.iloc[train_idx2]
y_train = y_train.iloc[train_idx2]

print(f"Train : {len(X_train):,} baris (72%)")
print(f"Val   : {len(X_val):,} baris  (8%)")
print(f"Test  : {len(X_test):,} baris  (20%)")

# Simpan hasil split
X_train.join(y_train).to_csv('data/prepared/train.csv', index=False)
X_test.join(y_test).to_csv('data/prepared/test.csv', index=False)
X_val.join(y_val).to_csv('data/prepared/val.csv', index=False)
print("✓ Data split tersimpan di data/prepared/")

# ============================================================
# CEK DATA LEAKAGE (BUKTI)
# ============================================================
print("\n" + "="*50)
print("CEK DATA LEAKAGE")
print("="*50)

# 🔴 FIX: pastikan mapping index benar
url_train = url_backup.iloc[train_idx].iloc[train_idx2]
url_val   = url_backup.iloc[train_idx].iloc[val_idx]
url_test  = url_backup.iloc[test_idx]

overlap_train_test = set(url_train) & set(url_test)
overlap_train_val  = set(url_train) & set(url_val)
overlap_val_test   = set(url_val) & set(url_test)

print(f"Overlap train-test : {len(overlap_train_test)}")
print(f"Overlap train-val  : {len(overlap_train_val)}")
print(f"Overlap val-test   : {len(overlap_val_test)}")

if len(overlap_train_test) == 0 and len(overlap_train_val) == 0 and len(overlap_val_test) == 0:
    print("✓ TIDAK ADA DATA LEAKAGE (VALID)")
else:
    print("⚠️ MASIH ADA DATA LEAKAGE!")

# ============================================================
# EDA
# ============================================================
print("\n" + "="*50)
print("EDA")
print("="*50)

# Plot 1: Distribusi kelas
counts = y.value_counts().sort_index()

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].bar(['Phishing (0)', 'Legitimate (1)'], counts.values,
            color=['#e74c3c', '#2ecc71'], width=0.5)
axes[0].set_title('Distribusi Kelas')
axes[0].set_ylabel('Jumlah URL')

for i, v in enumerate(counts.values):
    axes[0].text(i, v + 500, f'{v:,}', ha='center', fontweight='bold')

axes[1].pie(counts.values, labels=['Phishing', 'Legitimate'],
            autopct='%1.1f%%', colors=['#e74c3c', '#2ecc71'])
axes[1].set_title('Proporsi Kelas')

plt.suptitle('EDA — Distribusi Label PhiUSIIL')
plt.tight_layout()
plt.savefig('plots/01_distribusi_kelas.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Plot 1 disimpan: plots/01_distribusi_kelas.png")

# Plot 2: Distribusi fitur
fitur_tampil = [
    'URLLength', 'DomainLength', 'URLSimilarityIndex',
    'IsHTTPS', 'TLDLegitimateProb', 'NoOfSubDomain',
    'CharContinuationRate', 'URLCharProb'
]

fitur_ada = [f for f in fitur_tampil if f in X.columns]

fig, axes = plt.subplots(2, 4, figsize=(20, 8))

for ax, fitur in zip(axes.flatten(), fitur_ada):
    sns.histplot(data=df_clean, x=fitur, hue='label',
                 bins=40, kde=True, ax=ax,
                 palette={0: '#e74c3c', 1: '#2ecc71'}, alpha=0.6)
    ax.set_title(fitur, fontsize=11)

plt.suptitle('EDA — Distribusi Fitur per Kelas')
plt.tight_layout()
plt.savefig('plots/02_distribusi_fitur.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Plot 2 disimpan: plots/02_distribusi_fitur.png")

# Plot 3: Korelasi
df_numerik = df_clean.select_dtypes(include=[np.number])

korelasi = df_numerik.corr()['label'].drop('label')
korelasi_sorted = korelasi.abs().sort_values(ascending=False)
top15 = korelasi_sorted.head(15)

colors = ['#2ecc71' if korelasi[i] > 0 else '#e74c3c' for i in top15.index]

plt.figure(figsize=(10, 6))
plt.barh(top15.index[::-1], top15.values[::-1], color=colors[::-1])
plt.xlabel('Nilai Absolut Korelasi')
plt.title('Top 15 Fitur Paling Berkorelasi dengan Label')
plt.tight_layout()
plt.savefig('plots/03_korelasi_fitur.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Plot 3 disimpan: plots/03_korelasi_fitur.png")

# ============================================================
# DATASET TRACKING
# ============================================================
print("\n" + "="*50)
print("DATASET TRACKING")
print("="*50)

tracking = {
    'sumber'             : 'Kaggle / UCI Machine Learning Repository',
    'nama_dataset'       : 'PhiUSIIL Phishing URL Dataset',
    'doi'                : 'https://doi.org/10.1016/j.cose.2023.103545',
    'tanggal_akses'      : datetime.now().strftime('%Y-%m-%d %H:%M'),
    'total_baris'        : df.shape[0],
    'total_kolom_raw'    : df.shape[1],
    'missing_values'     : int(df.isnull().sum().sum()),
    'kelas_legitimate'   : int((y == 1).sum()),
    'kelas_phishing'     : int((y == 0).sum()),
    'kolom_dibuang'      : str(kolom_dibuang),
    'total_fitur_X'      : X.shape[1],
    'ukuran_train'       : len(X_train),
    'ukuran_val'         : len(X_val),
    'ukuran_test'        : len(X_test),
}

for key, value in tracking.items():
    print(f"{key:<25} : {value}")

with open('dataset_tracking.json', 'w') as f:
    json.dump(tracking, f, indent=4)

print("\n✓ dataset_tracking.json tersimpan")
print("\n✓ SEMUA TAHAP SELESAI!")

