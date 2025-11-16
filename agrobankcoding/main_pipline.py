import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve,
                             accuracy_score, precision_score, recall_score, f1_score)
import warnings

warnings.filterwarnings('ignore')

# Import clean_dataset function (bu sizning funksiyangiz)
from dataset_cleaning import clean_dataset

print("=" * 80)
print(" " * 10 + "KREDIT DEFAULT BASHORAT QILISH MODELI")
print(" " * 15 + "(Train & Evaluation Pipeline)")
print("=" * 80)

# ============================================================================
# 1. TRAINING DATA - TOZALASH VA YUKLASH
# ============================================================================
print("\n" + "=" * 80)
print("QISM 1: TRAINING DATA")
print("=" * 80)

# Training datasets
train_data_list = [
    'datasets/application_metadata.csv',
    'datasets/demographics.csv',
    'datasets/financial_ratios.jsonl',
    'datasets/geographic_data.xml',
    'datasets/loan_details.xlsx',
    'datasets/credit_history.parquet'
]


eval_data_list = [
    'evaluation/application_metadata.csv',
    'evaluation/demographics.csv',
    'evaluation/financial_ratios.jsonl',
    'evaluation/geographic_data.xml',
    'evaluation/loan_details.xlsx',
    'evaluation/credit_history.parquet'
]

print("\n📂 Training datasetlarni tozalash...")
train_result = clean_dataset(train_data_list, output_list=['results/old_cleaned_dataset.csv', 'results/old_cleaned_features.csv'])
df_train = train_result['df']

print(f"\n✅ Training data tayyor: {df_train.shape}")
print(f"   Default rate: {df_train['default'].mean() * 100:.2f}%")

# ============================================================================
# 2. FEATURE SELECTION - OPTIMAL 35 FEATURES
# ============================================================================
print("\n" + "=" * 80)
print("QISM 2: FEATURE SELECTION")
print("=" * 80)

# Optimal features (ChatGPT tahlili asosida)
OPTIMAL_FEATURES = [
    'age',
    'annual_income',
    'employment_length',
    'num_dependents',
    'monthly_income',
    'existing_monthly_debt',
    'monthly_payment',
    'debt_to_income_ratio',
    'payment_to_income_ratio',
    'total_debt_amount',
    'monthly_free_cash_flow',
    'credit_score',
    'num_credit_accounts',
    'credit_utilization',
    'available_credit',
    'num_delinquencies_2yrs',
    'num_inquiries_6mo',
    'recent_inquiry_count',
    'loan_amount',
    'loan_term',
    'interest_rate',
    'loan_to_value_ratio'
]

# Add encoded categorical features
CATEGORICAL_FEATURES = ['employment_type', 'education', 'marital_status']

print("\n🔧 Categorical features ni encode qilish...")
label_encoders = {}

for col in CATEGORICAL_FEATURES:
    if col in df_train.columns:
        le = LabelEncoder()
        df_train[col + '_encoded'] = le.fit_transform(df_train[col].astype(str))
        label_encoders[col] = le
        OPTIMAL_FEATURES.append(col + '_encoded')

# Filter existing features
feature_cols = [col for col in OPTIMAL_FEATURES if col in df_train.columns]

print(f"\n✅ Jami features: {len(feature_cols)}")
print(f"   Numerical: {len(feature_cols) - len(CATEGORICAL_FEATURES)}")
print(f"   Categorical (encoded): {len(CATEGORICAL_FEATURES)}")

# ============================================================================
# 3. DATA PREPARATION
# ============================================================================
print("\n" + "=" * 80)
print("QISM 3: DATA PREPARATION")
print("=" * 80)

# Remove rows with missing target
df_train_clean = df_train.dropna(subset=['default'])

X = df_train_clean[feature_cols].copy()
y = df_train_clean['default'].copy()

# Fill missing values with median
X = X.fillna(X.median())

print(f"\n✅ Training samples: {len(X)}")
print(f"✅ Features: {len(feature_cols)}")
print(f"✅ Defaults: {y.sum()} ({y.mean() * 100:.2f}%)")
print(f"✅ Non-defaults: {(1 - y).sum()} ({(1 - y.mean()) * 100:.2f}%)")

# ============================================================================
# 4. TRAIN-TEST SPLIT
# ============================================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n📊 Train set: {X_train.shape[0]} samples")
print(f"📊 Test set: {X_test.shape[0]} samples")

# ============================================================================
# 5. FEATURE SCALING
# ============================================================================
print("\n⚙️  Feature scaling...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================================
# 6. MODEL TRAINING
# ============================================================================
print("\n" + "=" * 80)
print("QISM 4: MODEL TRAINING")
print("=" * 80)

models = {}

# 1. Random Forest
print("\n📈 1. Random Forest Training...")
rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight='balanced',
    max_depth=12,
    min_samples_split=20,
    min_samples_leaf=10,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)
models['Random Forest'] = rf_model
print("   ✅ Random Forest trained")

# 2. Gradient Boosting
print("\n📈 2. Gradient Boosting Training...")
gb_model = GradientBoostingClassifier(
    n_estimators=200,
    random_state=42,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8
)
gb_model.fit(X_train, y_train)
models['Gradient Boosting'] = gb_model
print("   ✅ Gradient Boosting trained")

# 3. XGBoost
print("\n📈 3. XGBoost Training...")
xgb_model = XGBClassifier(
    n_estimators=200,
    random_state=42,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='logloss',
    use_label_encoder=False
)
xgb_model.fit(X_train, y_train)
models['XGBoost'] = xgb_model
print("   ✅ XGBoost trained")

# ============================================================================
# 7. MODEL EVALUATION
# ============================================================================
print("\n" + "=" * 80)
print("QISM 5: MODEL EVALUATION (TEST SET)")
print("=" * 80)

results = []

for name, model in models.items():
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_pred_proba)

    results.append({
        'Model': name,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-Score': f1,
        'ROC-AUC': auc
    })

    print(f"\n{'=' * 50}")
    print(f"🎯 {name}")
    print(f"{'=' * 50}")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"ROC-AUC:   {auc:.4f}")

results_df = pd.DataFrame(results)
print(f"\n📊 MODEL COMPARISON:")
print(results_df.to_string(index=False))

# Select best model
best_idx = results_df['ROC-AUC'].idxmax()
best_model_name = results_df.loc[best_idx, 'Model']
best_model = models[best_model_name]

print(f"\n🏆 BEST MODEL: {best_model_name}")
print(f"   ROC-AUC: {results_df.loc[best_idx, 'ROC-AUC']:.4f}")

# ============================================================================
# 8. FEATURE IMPORTANCE
# ============================================================================
print("\n" + "=" * 80)
print("QISM 6: FEATURE IMPORTANCE")
print("=" * 80)

feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': best_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n🔝 TOP 15 ENG MUHIM OMILLAR:\n")
print(feature_importance.head(15).to_string(index=False))

# ============================================================================
# 9. SAVE MODEL & ARTIFACTS
# ============================================================================
print("\n" + "=" * 80)
print("QISM 7: MODEL VA ARTIFACTS NI SAQLASH")
print("=" * 80)

# Save model
model_path = 'models/best_model.pkl'
with open(model_path, 'wb') as f:
    pickle.dump(best_model, f)
print(f"✅ Model saqlandi: {model_path}")

# Save scaler
scaler_path = 'models/scaler.pkl'
with open(scaler_path, 'wb') as f:
    pickle.dump(scaler, f)
print(f"✅ Scaler saqlandi: {scaler_path}")

# Save label encoders
encoders_path = 'models/label_encoders.pkl'
with open(encoders_path, 'wb') as f:
    pickle.dump(label_encoders, f)
print(f"✅ Label encoders saqlandi: {encoders_path}")

# Save feature columns
features_path = 'models/feature_columns.pkl'
with open(features_path, 'wb') as f:
    pickle.dump(feature_cols, f)
print(f"✅ Feature columns saqlandi: {features_path}")

# Save model performance
results_df.to_csv('results/model_performance.csv', index=False)
print(f"✅ Model performance saqlandi: results/model_performance.csv")

# ============================================================================
# 10. EVALUATION DATA - YANGI MA'LUMOTLARNI FORECAST QILISH
# ============================================================================
print("\n" + "=" * 80)
print("QISM 8: EVALUATION DATA - FORECAST")
print("=" * 80)

# Evaluation datasets (harxil ma'lumotlar)


print("\n📂 Evaluation datasetlarni tozalash...")
eval_result = clean_dataset(eval_data_list, output_list=['results/new_cleaned_dataset.csv', 'results/new_cleaned_features.csv'])
df_eval = eval_result['df']

print(f"\n✅ Evaluation data tayyor: {df_eval.shape}")

# ============================================================================
# 11. PREPARE EVALUATION DATA
# ============================================================================
print("\n🔧 Evaluation data ni tayyorlash...")

# Encode categorical features using saved encoders
for col in CATEGORICAL_FEATURES:
    if col in df_eval.columns and col in label_encoders:
        le = label_encoders[col]
        # Handle unseen categories
        df_eval[col + '_encoded'] = df_eval[col].astype(str).apply(
            lambda x: le.transform([x])[0] if x in le.classes_ else -1
        )

# Extract features
X_eval = df_eval[feature_cols].copy()
X_eval = X_eval.fillna(X_eval.median())

print(f"✅ Evaluation samples: {len(X_eval)}")

# ============================================================================
# 12. MAKE PREDICTIONS
# ============================================================================
print("\n🎯 Bashorat qilish...")

# Predict using best model
predictions = best_model.predict(X_eval)
prediction_proba = best_model.predict_proba(X_eval)[:, 1]

print(f"✅ Bashoratlar tayyor")

# ============================================================================
# 13. CREATE RESULTS
# ============================================================================
print("\n📊 Natijalarni yaratish...")

# Create results dataframe
results_eval = pd.DataFrame({
    'customer_id': df_eval['customer_ref'].values,
    'prob': prediction_proba,
    'default': predictions
})

# Sort by customer_id
results_eval = results_eval.sort_values('customer_id').reset_index(drop=True)

# ============================================================================
# 14. SAVE RESULTS
# ============================================================================
print("\n💾 Natijalarni saqlash...")

# Save main results
output_file = 'results.csv'
results_eval.to_csv(output_file, index=False)
print(f"✅ Predictions saqlandi: {output_file}")


# Save detailed results with risk categories
def categorize_risk(prob):
    if prob < 0.3:
        return 'Low Risk'
    elif prob < 0.5:
        return 'Medium Risk'
    elif prob < 0.7:
        return 'High Risk'
    else:
        return 'Very High Risk'


results_detailed = results_eval.copy()
results_detailed['risk_category'] = results_detailed['prob'].apply(categorize_risk)


# Add recommendation
def get_recommendation(row):
    if row['prob'] < 0.3:
        return 'AUTO-APPROVE'
    elif row['prob'] < 0.5:
        return 'MANUAL REVIEW'
    elif row['prob'] < 0.7:
        return 'HIGH RISK - CAREFUL'
    else:
        return 'AUTO-REJECT'


results_detailed['recommendation'] = results_detailed.apply(get_recommendation, axis=1)

detailed_output = 'results/predictions_detailed.csv'
results_detailed.to_csv(detailed_output, index=False)
print(f"✅ Detailed predictions saqlandi: {detailed_output}")

# ============================================================================
# 15. STATISTICS & SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("QISM 9: NATIJALAR STATISTIKASI")
print("=" * 80)

print(f"\n📋 FORECAST RESULTS:")
print(f"   Jami mijozlar: {len(results_eval)}")
print(f"   Kredit BERILSIN (default=0): {(predictions == 0).sum()} ta ({(predictions == 0).mean() * 100:.2f}%)")
print(f"   Kredit BERILMASIN (default=1): {(predictions == 1).sum()} ta ({(predictions == 1).mean() * 100:.2f}%)")
print(f"   O'rtacha probability: {prediction_proba.mean():.4f}")

# Risk distribution
risk_counts = results_detailed['risk_category'].value_counts()
print(f"\n📊 RISK TAQSIMOTI:")
for category in ['Low Risk', 'Medium Risk', 'High Risk', 'Very High Risk']:
    count = risk_counts.get(category, 0)
    pct = count / len(results_detailed) * 100
    print(f"   {category:20s}: {count:5d} ta ({pct:5.1f}%)")

# Recommendation distribution
rec_counts = results_detailed['recommendation'].value_counts()
print(f"\n💡 TAVSIYALAR:")
for rec in ['AUTO-APPROVE', 'MANUAL REVIEW', 'HIGH RISK - CAREFUL', 'AUTO-REJECT']:
    count = rec_counts.get(rec, 0)
    pct = count / len(results_detailed) * 100
    print(f"   {rec:20s}: {count:5d} ta ({pct:5.1f}%)")

# Sample predictions
print(f"\n📋 NATIJADAN NAMUNA (birinchi 10 ta):")
print(results_detailed.head(10).to_string(index=False))

# ============================================================================
# 16. VISUALIZATION
# ============================================================================
print("\n" + "=" * 80)
print("QISM 10: VIZUALIZATSIYA")
print("=" * 80)

fig = plt.figure(figsize=(20, 12))

# 1. Feature Importance
ax1 = plt.subplot(3, 3, 1)
top_features = feature_importance.head(15)
colors = plt.cm.viridis(np.linspace(0, 1, len(top_features)))
ax1.barh(range(len(top_features)), top_features['importance'], color=colors)
ax1.set_yticks(range(len(top_features)))
ax1.set_yticklabels(top_features['feature'], fontsize=9)
ax1.set_xlabel('Importance')
ax1.set_title('Top 15 Feature Importance', fontsize=12, fontweight='bold')
ax1.invert_yaxis()

# 2. ROC Curves (Training)
ax2 = plt.subplot(3, 3, 2)
for name, model in models.items():
    y_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    ax2.plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})', linewidth=2)
ax2.plot([0, 1], [0, 1], 'k--', label='Random')
ax2.set_xlabel('False Positive Rate')
ax2.set_ylabel('True Positive Rate')
ax2.set_title('ROC Curves (Training)', fontsize=12, fontweight='bold')
ax2.legend(fontsize=8)
ax2.grid(alpha=0.3)

# 3. Prediction Probability Distribution
ax3 = plt.subplot(3, 3, 3)
ax3.hist(prediction_proba, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
ax3.axvline(0.5, color='red', linestyle='--', linewidth=2, label='Threshold')
ax3.set_xlabel('Default Probability')
ax3.set_ylabel('Count')
ax3.set_title('Prediction Probability Distribution', fontsize=12, fontweight='bold')
ax3.legend()

# 4. Risk Category Distribution
ax4 = plt.subplot(3, 3, 4)
risk_categories = ['Low Risk', 'Medium Risk', 'High Risk', 'Very High Risk']
risk_values = [risk_counts.get(cat, 0) for cat in risk_categories]
colors_risk = ['green', 'yellow', 'orange', 'red']
bars = ax4.bar(range(len(risk_categories)), risk_values, color=colors_risk, alpha=0.7, edgecolor='black')
ax4.set_xticks(range(len(risk_categories)))
ax4.set_xticklabels(risk_categories, rotation=45, ha='right')
ax4.set_ylabel('Count')
ax4.set_title('Risk Category Distribution', fontsize=12, fontweight='bold')
for i, v in enumerate(risk_values):
    ax4.text(i, v + 10, str(v), ha='center', fontweight='bold')

# 5. Recommendation Distribution
ax5 = plt.subplot(3, 3, 5)
recs = ['AUTO-APPROVE', 'MANUAL REVIEW', 'HIGH RISK', 'AUTO-REJECT']
rec_values = [rec_counts.get('AUTO-APPROVE', 0),
              rec_counts.get('MANUAL REVIEW', 0),
              rec_counts.get('HIGH RISK - CAREFUL', 0),
              rec_counts.get('AUTO-REJECT', 0)]
colors_rec = ['green', 'yellow', 'orange', 'red']
ax5.bar(range(len(recs)), rec_values, color=colors_rec, alpha=0.7, edgecolor='black')
ax5.set_xticks(range(len(recs)))
ax5.set_xticklabels(recs, rotation=45, ha='right', fontsize=8)
ax5.set_ylabel('Count')
ax5.set_title('Recommendation Distribution', fontsize=12, fontweight='bold')
for i, v in enumerate(rec_values):
    ax5.text(i, v + 10, str(v), ha='center', fontweight='bold', fontsize=8)

# 6. Confusion Matrix (Test Set)
ax6 = plt.subplot(3, 3, 6)
y_pred_test = best_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred_test)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax6, cbar=False)
ax6.set_xlabel('Predicted')
ax6.set_ylabel('Actual')
ax6.set_title(f'Confusion Matrix - {best_model_name}', fontsize=12, fontweight='bold')

# 7. Model Performance Comparison
ax7 = plt.subplot(3, 3, 7)
x_pos = np.arange(len(results_df))
ax7.bar(x_pos, results_df['ROC-AUC'], color='steelblue', alpha=0.7, edgecolor='black')
ax7.set_xticks(x_pos)
ax7.set_xticklabels(results_df['Model'], rotation=45, ha='right')
ax7.set_ylabel('ROC-AUC Score')
ax7.set_title('Model Performance Comparison', fontsize=12, fontweight='bold')
ax7.set_ylim([0, 1])
for i, v in enumerate(results_df['ROC-AUC']):
    ax7.text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')

# 8. Probability vs Risk
ax8 = plt.subplot(3, 3, 8)
scatter = ax8.scatter(range(len(prediction_proba[:100])),
                      sorted(prediction_proba[:100]),
                      c=sorted(prediction_proba[:100]),
                      cmap='RdYlGn_r', s=50, alpha=0.6)
ax8.axhline(0.3, color='green', linestyle='--', alpha=0.5, label='Low Risk')
ax8.axhline(0.5, color='yellow', linestyle='--', alpha=0.5, label='Medium Risk')
ax8.axhline(0.7, color='orange', linestyle='--', alpha=0.5, label='High Risk')
ax8.set_xlabel('Sample Index (sorted by probability)')
ax8.set_ylabel('Default Probability')
ax8.set_title('Probability Distribution (First 100)', fontsize=12, fontweight='bold')
ax8.legend(fontsize=8)
plt.colorbar(scatter, ax=ax8)

# 9. Default Rate Summary
ax9 = plt.subplot(3, 3, 9)
summary_data = {
    'Train Set': y_train.mean(),
    'Test Set': y_test.mean(),
    'Predictions': predictions.mean()
}
bars = ax9.bar(summary_data.keys(), summary_data.values(),
               color=['blue', 'green', 'red'], alpha=0.7, edgecolor='black')
ax9.set_ylabel('Default Rate')
ax9.set_title('Default Rate Comparison', fontsize=12, fontweight='bold')
ax9.set_ylim([0, max(summary_data.values()) * 1.2])
for bar in bars:
    height = bar.get_height()
    ax9.text(bar.get_x() + bar.get_width() / 2., height,
             f'{height:.2%}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
viz_path = 'results/analysis_visualization.png'
plt.savefig(viz_path, dpi=300, bbox_inches='tight')
print(f"✅ Vizualizatsiya saqlandi: {viz_path}")

# ============================================================================
# 17. FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("✅ PIPELINE MUVAFFAQIYATLI YAKUNLANDI!")
print("=" * 80)

print(f"\n📁 YARATILGAN FAYLLAR:")
print(f"   1. models/best_model.pkl - O'qitilgan model")
print(f"   2. models/scaler.pkl - Feature scaler")
print(f"   3. models/label_encoders.pkl - Categorical encoders")
print(f"   4. models/feature_columns.pkl - Feature nomi")
print(f"   5. results/predictions.csv - Asosiy natijalar")
print(f"   6. results/predictions_detailed.csv - Batafsil natijalar")
print(f"   7. results/model_performance.csv - Model performance")
print(f"   8. results/analysis_visualization.png - Grafiklar")

print(f"\n🏆 Best Model: {best_model_name}")
print(f"📊 ROC-AUC: {results_df.loc[best_idx, 'ROC-AUC']:.4f}")
print(f"📈 Total Predictions: {len(results_eval)}")

print("\n💡 QANDAY ISHLATISH:")
print("   1. Training: datasets/ papkasidagi datasetlar bilan model o'qitildi")
print("   2. Evaluation: evaluation/ papkasidagi datasetlar forecast qilindi")
print("   3. Results: results/ papkasida barcha natijalar saqlandi")
print("   4. Model: models/ papkasida keyingi forecast uchun saqlandi")

print("\n" + "=" * 80)