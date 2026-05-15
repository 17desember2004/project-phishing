import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os

os.makedirs('plots/shap', exist_ok=True)

# ============================================================
# LOAD MODEL DAN FITUR
# ============================================================
print("="*50)
print("LOAD MODEL DAN DATA")
print("="*50)

model = joblib.load('models/model.pkl')

with open('models/feature_list.json') as f:
    fitur_tersedia = json.load(f)

print(f"✓ Model loaded")
print(f"✓ Fitur: {len(fitur_tersedia)} fitur URL-only")

# Load test data
test  = pd.read_csv('data/prepared/test.csv')
DROP_COLS = ['label','URL','url','Domain','TLD','Title','FILENAME']
X_test = test.drop(columns=DROP_COLS, errors='ignore').select_dtypes(include=[np.number])
X_test = X_test[fitur_tersedia]
y_test = test['label']

print(f"✓ Test data: {X_test.shape}")

# ============================================================
# HITUNG SHAP VALUES
# ============================================================
print("\n" + "="*50)
print("HITUNG SHAP VALUES")
print("="*50)

explainer   = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

print(f"✓ SHAP values berhasil dihitung")
print(f"  Shape: {np.array(shap_values).shape}")

# ============================================================
# PLOT 1: SHAP SUMMARY BAR
# ============================================================
print("\n" + "="*50)
print("PLOT 1: SHAP SUMMARY BAR")
print("="*50)

plt.figure(figsize=(10, 8))
shap.summary_plot(
    shap_values, X_test,
    feature_names=fitur_tersedia,
    plot_type="bar",
    show=False
)
plt.title("SHAP Feature Importance — Phishing Detection\n(Bar Plot)", fontsize=13)
plt.tight_layout()
plt.savefig('plots/shap/shap_summary_bar.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ plots/shap/shap_summary_bar.png")

# ============================================================
# PLOT 2: SHAP SUMMARY DOT (BEESWARM)
# ============================================================
print("\n" + "="*50)
print("PLOT 2: SHAP SUMMARY DOT")
print("="*50)

plt.figure(figsize=(10, 8))
shap.summary_plot(
    shap_values, X_test,
    feature_names=fitur_tersedia,
    show=False
)
plt.title("SHAP Summary Plot — Phishing Detection\n(Beeswarm)", fontsize=13)
plt.tight_layout()
plt.savefig('plots/shap/shap_summary_dot.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ plots/shap/shap_summary_dot.png")

# ============================================================
# PLOT 3: WATERFALL — CONTOH URL PHISHING
# ============================================================
print("\n" + "="*50)
print("PLOT 3: WATERFALL URL PHISHING")
print("="*50)

idx_phishing = y_test[y_test == 0].index[0]
idx_local    = X_test.index.get_loc(idx_phishing)

shap_exp_phishing = shap.Explanation(
    values       = shap_values[idx_local],
    base_values  = explainer.expected_value,
    data         = X_test.iloc[idx_local],
    feature_names= fitur_tersedia
)

plt.figure(figsize=(10, 7))
shap.waterfall_plot(shap_exp_phishing, show=False)
plt.title("SHAP Waterfall — Contoh URL Phishing", fontsize=13)
plt.tight_layout()
plt.savefig('plots/shap/shap_waterfall_phishing.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ plots/shap/shap_waterfall_phishing.png")

# ============================================================
# PLOT 4: WATERFALL — CONTOH URL LEGITIMATE
# ============================================================
print("\n" + "="*50)
print("PLOT 4: WATERFALL URL LEGITIMATE")
print("="*50)

idx_legit  = y_test[y_test == 1].index[0]
idx_local2 = X_test.index.get_loc(idx_legit)

shap_exp_legit = shap.Explanation(
    values       = shap_values[idx_local2],
    base_values  = explainer.expected_value,
    data         = X_test.iloc[idx_local2],
    feature_names= fitur_tersedia
)

plt.figure(figsize=(10, 7))
shap.waterfall_plot(shap_exp_legit, show=False)
plt.title("SHAP Waterfall — Contoh URL Legitimate", fontsize=13)
plt.tight_layout()
plt.savefig('plots/shap/shap_waterfall_legit.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ plots/shap/shap_waterfall_legit.png")

# ============================================================
# PLOT 5: FORCE PLOT HTML (interaktif)
# ============================================================
print("\n" + "="*50)
print("PLOT 5: FORCE PLOT HTML")
print("="*50)

force_html = shap.force_plot(
    explainer.expected_value,
    shap_values[idx_local],
    X_test.iloc[idx_local],
    feature_names=fitur_tersedia
)
shap.save_html('plots/shap/shap_force_phishing.html', force_html)
print("✓ plots/shap/shap_force_phishing.html")

# ============================================================
# RINGKASAN SHAP
# ============================================================
print("\n" + "="*50)
print("RINGKASAN SHAP")
print("="*50)

mean_shap = np.abs(shap_values).mean(axis=0)
shap_df   = pd.DataFrame({
    'Fitur'           : fitur_tersedia,
    'Mean SHAP Value' : mean_shap
}).sort_values('Mean SHAP Value', ascending=False)

print("\nTop 10 Fitur paling berpengaruh (SHAP):")
print(shap_df.head(10).to_string(index=False))

shap_df.to_csv('plots/shap/shap_importance.csv', index=False)
print("\n✓ plots/shap/shap_importance.csv")
print("\n✓ SEMUA PLOT SHAP SELESAI!")
print(f"  Tersimpan di: plots/shap/")