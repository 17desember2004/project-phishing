import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import lightgbm as lgb
import optuna
import joblib
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score,
    confusion_matrix, classification_report,
    RocCurveDisplay, PrecisionRecallDisplay
)

# ============================================================
# SETUP FOLDER
# ============================================================

os.makedirs('models', exist_ok=True)
os.makedirs('plots',  exist_ok=True)

mlflow.set_experiment("Phishing-Detection-LightGBM-Final")

# ============================================================
# LOAD DATA
# ============================================================

print("="*55)
print("LOAD DATA")
print("="*55)

train = pd.read_csv('data/prepared/train.csv')
val   = pd.read_csv('data/prepared/val.csv')
test  = pd.read_csv('data/prepared/test.csv')

print(f"Train : {train.shape}")
print(f"Val   : {val.shape}")
print(f"Test  : {test.shape}")

print("\n✓ Anti leakage sudah diverifikasi di prepare.py")

# ============================================================
# PISAHKAN FITUR DAN LABEL
# ============================================================

DROP_COLS = ['label']

X_train_full = train.drop(columns=DROP_COLS, errors='ignore')
y_train      = train['label']

X_val_full   = val.drop(columns=DROP_COLS, errors='ignore')
y_val        = val['label']

X_test_full  = test.drop(columns=DROP_COLS, errors='ignore')
y_test       = test['label']

# ============================================================
# CEK KORELASI
# ============================================================

print("\n" + "="*55)
print("CEK KORELASI FITUR vs LABEL")
print("="*55)

corr_all = X_train_full.corrwith(y_train).abs().sort_values(ascending=False)

print("Top 15 fitur berkorelasi dengan label:")
print(corr_all.head(15).to_string())

# ============================================================
# FITUR URL-ONLY
# ============================================================

FITUR_URL_ONLY = [

    'URLLength',
    'DomainLength',
    'TLDLength',

    'IsDomainIP',
    'NoOfSubDomain',

    'NoOfLettersInURL',
    'LetterRatioInURL',

    'NoOfDegitsInURL',
    'DegitRatioInURL',

    'NoOfEqualsInURL',
    'NoOfQMarkInURL',
    'NoOfAmpersandInURL',

    'NoOfOtherSpecialCharsInURL',
    'SpacialCharRatioInURL',

    'CharContinuationRate',
    'URLCharProb',

    'TLDLegitimateProb',

    'HasObfuscation',
    'NoOfObfuscatedChar',
    'ObfuscationRatio',
]

fitur_tersedia = [
    f for f in FITUR_URL_ONLY
    if f in X_train_full.columns
]

print(f"\nJumlah fitur dipakai: {len(fitur_tersedia)}")

print("Daftar fitur digunakan:")
for fitur in fitur_tersedia:
    print(f"- {fitur}")

X_train = X_train_full[fitur_tersedia]
X_val   = X_val_full[fitur_tersedia]
X_test  = X_test_full[fitur_tersedia]

corr_dipakai = X_train.corrwith(y_train).abs().sort_values(ascending=False)

print(f"\nKorelasi tertinggi: {corr_dipakai.max():.4f}")

if corr_dipakai.max() > 0.85:
    print("⚠️ Korelasi sangat tinggi — cek leakage")
else:
    print("✓ Korelasi aman")

# ============================================================
# SIMPAN FEATURE LIST
# ============================================================

with open('models/feature_list.json', 'w') as f:
    json.dump(fitur_tersedia, f, indent=2)

# ============================================================
# TAHAP 1: BASELINE MODEL
# ============================================================

print("\n" + "="*55)
print("TAHAP 1: LIGHTGBM BASELINE")
print("="*55)

params_baseline = {
    'objective'     : 'binary',
    'learning_rate' : 0.1,
    'num_leaves'    : 31,
    'max_depth'     : -1,
    'n_estimators'  : 100,
    'random_state'  : 42,
    'verbose'       : -1,
    'n_jobs'        : -1,
}

with mlflow.start_run(run_name="baseline"):

    mlflow.log_params(params_baseline)
    mlflow.log_param("tahap", "baseline")
    mlflow.log_param("n_features", len(fitur_tersedia))

    model_baseline = lgb.LGBMClassifier(**params_baseline)

    model_baseline.fit(X_train, y_train)

    yv      = model_baseline.predict(X_val)
    yv_prob = model_baseline.predict_proba(X_val)[:, 1]

    b_acc = accuracy_score(y_val, yv)
    b_f1  = f1_score(y_val, yv)
    b_auc = roc_auc_score(y_val, yv_prob)
    b_pre = precision_score(y_val, yv)
    b_rec = recall_score(y_val, yv)

    mlflow.log_metric("val_accuracy",  b_acc)
    mlflow.log_metric("val_f1",        b_f1)
    mlflow.log_metric("val_auc",       b_auc)
    mlflow.log_metric("val_precision", b_pre)
    mlflow.log_metric("val_recall",    b_rec)

    mlflow.sklearn.log_model(model_baseline, "model")

    print(f"Baseline VAL — Acc:{b_acc:.4f} F1:{b_f1:.4f} AUC:{b_auc:.4f}")
    print(f"              Pre:{b_pre:.4f} Rec:{b_rec:.4f}")

# ============================================================
# TAHAP 2: OPTUNA HYPERPARAMETER TUNING
# ============================================================

print("\n" + "="*55)
print("TAHAP 2: OPTUNA TUNING")
print("="*55)

def objective(trial):

    params = {

        'objective'        : 'binary',
        'verbose'          : -1,
        'random_state'     : 42,
        'n_jobs'           : -1,

        'learning_rate'    : trial.suggest_float('learning_rate', 0.01, 0.2),
        'num_leaves'       : trial.suggest_int('num_leaves', 15, 63),
        'max_depth'        : trial.suggest_int('max_depth', 3, 8),
        'n_estimators'     : trial.suggest_int('n_estimators', 100, 500),

        'min_child_samples': trial.suggest_int('min_child_samples', 20, 100),

        'subsample'        : trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree' : trial.suggest_float('colsample_bytree', 0.6, 1.0),

        'reg_alpha'        : trial.suggest_float('reg_alpha', 0.0, 2.0),
        'reg_lambda'       : trial.suggest_float('reg_lambda', 0.0, 2.0),
    }

    model = lgb.LGBMClassifier(**params)

    model.fit(X_train, y_train)

    pred = model.predict(X_val)

    return f1_score(y_val, pred)

optuna.logging.set_verbosity(optuna.logging.WARNING)

study = optuna.create_study(direction='maximize')

study.optimize(
    objective,
    n_trials=50,
    show_progress_bar=True
)

print(f"\nBest F1 (val) : {study.best_value:.4f}")
print(f"Best Params   : {study.best_params}")

# ============================================================
# TAHAP 3: MODEL TERBAIK + LOG MLFLOW
# ============================================================

print("\n" + "="*55)
print("TAHAP 3: MODEL TERBAIK + MLFLOW LOG")
print("="*55)

best_params = study.best_params

best_params.update({
    'objective'   : 'binary',
    'verbose'     : -1,
    'random_state': 42,
    'n_jobs'      : -1
})

# ============================================================
# RETRAIN TRAIN + VAL
# ============================================================

X_final_train = pd.concat([X_train, X_val], axis=0)
y_final_train = pd.concat([y_train, y_val], axis=0)

with mlflow.start_run(run_name="best_model_optuna"):

    mlflow.log_params(best_params)

    mlflow.log_param("tahap", "best_model_optuna")
    mlflow.log_param("n_features", len(fitur_tersedia))
    mlflow.log_param("features_used", str(fitur_tersedia))
    mlflow.log_param("max_corr", round(float(corr_dipakai.max()), 4))

    best_model = lgb.LGBMClassifier(**best_params)

    best_model.fit(X_final_train, y_final_train)

    # ========================================================
    # EVALUASI TEST
    # ========================================================

    yt_pred = best_model.predict(X_test)
    yt_prob = best_model.predict_proba(X_test)[:, 1]

    test_acc = accuracy_score(y_test, yt_pred)
    test_f1  = f1_score(y_test, yt_pred)
    test_auc = roc_auc_score(y_test, yt_prob)
    test_pre = precision_score(y_test, yt_pred)
    test_rec = recall_score(y_test, yt_pred)

    mlflow.log_metric("test_accuracy",  test_acc)
    mlflow.log_metric("test_f1",        test_f1)
    mlflow.log_metric("test_auc",       test_auc)
    mlflow.log_metric("test_precision", test_pre)
    mlflow.log_metric("test_recall",    test_rec)

    print(f"\n[TEST] Accuracy  : {test_acc:.4f}")
    print(f"[TEST] F1 Score : {test_f1:.4f}")
    print(f"[TEST] AUC ROC  : {test_auc:.4f}")
    print(f"[TEST] Precision: {test_pre:.4f}")
    print(f"[TEST] Recall   : {test_rec:.4f}")

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        yt_pred,
        target_names=['Phishing', 'Legitimate']
    ))

    if test_acc >= 0.999:
        print("⚠️ Akurasi terlalu tinggi — cek leakage")
    else:
        print("✓ Hasil realistis")

    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    cm = confusion_matrix(y_test, yt_pred)

    tn, fp, fn, tp = cm.ravel()

    print("\nConfusion Matrix Detail:")
    print(f"TN (Phishing benar)     : {tn:,}")
    print(f"FP (Phishing → aman)    : {fp:,}")
    print(f"FN (Aman → phishing)    : {fn:,}")
    print(f"TP (Aman benar)         : {tp:,}")

    fig, ax = plt.subplots(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Phishing', 'Legitimate'],
        yticklabels=['Phishing', 'Legitimate'],
        ax=ax
    )

    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')

    plt.tight_layout()

    plt.savefig(
        'plots/confusion_matrix.png',
        dpi=150
    )

    plt.close()

    mlflow.log_artifact('plots/confusion_matrix.png')

    print("✓ Confusion matrix tersimpan")

    # ========================================================
    # ROC CURVE
    # ========================================================

    fig, ax = plt.subplots(figsize=(6, 5))

    RocCurveDisplay.from_predictions(
        y_test,
        yt_prob,
        ax=ax,
        name='LightGBM + Optuna'
    )

    ax.plot([0, 1], [0, 1], 'k--')

    ax.set_title(f'ROC Curve (AUC={test_auc:.4f})')

    plt.tight_layout()

    plt.savefig(
        'plots/roc_curve.png',
        dpi=150
    )

    plt.close()

    mlflow.log_artifact('plots/roc_curve.png')

    print("✓ ROC curve tersimpan")

    # ========================================================
    # PRECISION RECALL CURVE
    # ========================================================

    fig, ax = plt.subplots(figsize=(6, 5))

    PrecisionRecallDisplay.from_predictions(
        y_test,
        yt_prob,
        ax=ax,
        name='LightGBM + Optuna'
    )

    ax.set_title(f'Precision Recall Curve (F1={test_f1:.4f})')

    plt.tight_layout()

    plt.savefig(
        'plots/precision_recall_curve.png',
        dpi=150
    )

    plt.close()

    mlflow.log_artifact('plots/precision_recall_curve.png')

    print("✓ Precision Recall curve tersimpan")

    # ========================================================
    # FEATURE IMPORTANCE
    # ========================================================

    feat_imp = pd.Series(
        best_model.feature_importances_,
        index=fitur_tersedia
    ).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(9, 7))

    feat_imp.plot(
        kind='barh',
        ax=ax
    )

    ax.set_title('Feature Importance')
    ax.set_xlabel('Importance Score')

    plt.tight_layout()

    plt.savefig(
        'plots/feature_importance.png',
        dpi=150
    )

    plt.close()

    mlflow.log_artifact('plots/feature_importance.png')

    print("✓ Feature importance tersimpan")

    # ========================================================
    # KORELASI FITUR
    # ========================================================

    corr_sorted = corr_dipakai.sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(9, 7))

    corr_sorted.plot(
        kind='barh',
        ax=ax
    )

    ax.axvline(
        x=0.85,
        color='red',
        linestyle='--',
        linewidth=1.5,
        label='Threshold leakage'
    )

    ax.set_title('Korelasi Fitur dengan Label')

    ax.legend()

    plt.tight_layout()

    plt.savefig(
        'plots/korelasi_fitur_dipakai.png',
        dpi=150
    )

    plt.close()

    mlflow.log_artifact('plots/korelasi_fitur_dipakai.png')

    print("✓ Korelasi fitur tersimpan")

    # ========================================================
    # SIMPAN MODEL
    # ========================================================

    joblib.dump(
        best_model,
        'models/model.pkl'
    )

    mlflow.sklearn.log_model(
        best_model,
        "model"
    )

    mlflow.log_artifact(
        'models/feature_list.json'
    )

    print("✓ Model tersimpan")

    # ========================================================
    # SIMPAN METRICS
    # ========================================================

    metrics_out = {

        "baseline_val_f1"    : round(b_f1, 4),
        "baseline_val_acc"   : round(b_acc, 4),

        "test_accuracy"      : round(test_acc, 4),
        "test_f1"            : round(test_f1, 4),
        "test_auc"           : round(test_auc, 4),
        "test_precision"     : round(test_pre, 4),
        "test_recall"        : round(test_rec, 4),

        "peningkatan_f1"     : round(study.best_value - b_f1, 4),

        "confusion_matrix"   : {
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
            "TP": int(tp)
        },

        "n_features"         : len(fitur_tersedia),

        "max_corr"           : round(float(corr_dipakai.max()), 4),

        "best_optuna_params" : study.best_params,
    }

    with open('metrics.json', 'w') as f:
        json.dump(metrics_out, f, indent=4)

    mlflow.log_artifact('metrics.json')

    print("✓ Metrics tersimpan")

    run_id = mlflow.active_run().info.run_id

    print(f"\n✓ MLflow run_id : {run_id}")

# ============================================================
# RINGKASAN AKHIR
# ============================================================

print("\n" + "="*55)
print("RINGKASAN AKHIR")
print("="*55)

print(f"Dataset      : PhiUSIIL")
print(f"Algoritma    : LightGBM + Optuna")
print(f"Fitur        : {len(fitur_tersedia)} fitur URL-only")
print(f"Korelasi max : {corr_dipakai.max():.4f}")
print(f"")

print(f"BASELINE:")
print(f"  Val F1  : {b_f1:.4f}")

print(f"\nOPTUNA:")
print(f"  Best Val F1 : {study.best_value:.4f}")

print(f"\nTEST FINAL:")
print(f"  Accuracy  : {test_acc:.4f}")
print(f"  F1 Score  : {test_f1:.4f}")
print(f"  AUC ROC   : {test_auc:.4f}")
print(f"  Precision : {test_pre:.4f}")
print(f"  Recall    : {test_rec:.4f}")

print(f"\nPLOTS:")
print(f"- plots/confusion_matrix.png")
print(f"- plots/roc_curve.png")
print(f"- plots/precision_recall_curve.png")
print(f"- plots/feature_importance.png")
print(f"- plots/korelasi_fitur_dipakai.png")

print(f"\nMODEL:")
print(f"- models/model.pkl")

print(f"\nMLFLOW:")
print(f"mlflow ui → http://localhost:5000")
print(f"Experiment: Phishing-Detection-LightGBM-Final")

