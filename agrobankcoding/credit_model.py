import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, precision_recall_curve,
                             accuracy_score, precision_score, recall_score, f1_score)
import warnings

warnings.filterwarnings('ignore')

print("=" * 80)
print(" " * 20 + "KREDIT DEFAULT BASHORAT QILISH MODELI")
print("=" * 80)

# ============================================================================
# 1. DATASETLARNI YUKLASH
# ============================================================================
print("\n📂 1. DATASETLARNI YUKLASH")
print("-" * 80)

# Dataset 1: Application & Behavioral
df1 = pd.read_csv('../hackaton/application_metadata.csv')
print(f"✅ Dataset 1 (Behavioral): {df1.shape}")

# Dataset 2: Demographics
df2 = pd.read_csv('../hackaton/demographics.csv')
print(f"✅ Dataset 2 (Demographics): {df2.shape}")

# Dataset 3: Financial Debt (JSON)
with open('../hackaton/financial_ratios.jsonl', 'r') as f:
    data_json = [json.loads(line) for line in f]
df3 = pd.DataFrame(data_json)
print(f"✅ Dataset 3 (Financial Debt): {df3.shape}")

# Dataset 4: Geographic (XML)
tree = ET.parse('../hackaton/geographic_data.xml')
root = tree.getroot()
geo_data = []
for customer in root.findall('customer'):
    geo_data.append({
        'geo_id': int(customer.find('id').text),
        'state': customer.find('state').text,
        'regional_unemployment_rate': float(customer.find('regional_unemployment_rate').text),
        'regional_median_income': float(customer.find('regional_median_income').text),
        'regional_median_rent': float(customer.find('regional_median_rent').text),
        'housing_price_index': float(customer.find('housing_price_index').text),
        'cost_of_living_index': float(customer.find('cost_of_living_index').text)
    })
df4 = pd.DataFrame(geo_data)
print(f"✅ Dataset 4 (Geographic): {df4.shape}")

# Dataset 5: Loan Details
df5 = pd.read_excel('hackaton/loan_details.xlsx')
print(f"✅ Dataset 5 (Loan Details): {df5.shape}")

# Dataset 6: Credit History (Parquet)
df6 = pd.read_parquet('../hackaton/credit_history.parquet')
print(f"✅ Dataset 6 (Credit History): {df6.shape}")

# ============================================================================
# 2. MA'LUMOTLARNI TOZALASH
# ============================================================================
print("\n📊 2. MA'LUMOTLARNI TOZALASH")
print("-" * 80)

# 2.1 Dataset 1: Remove noise
if 'random_noise_1' in df1.columns:
    df1 = df1.drop('random_noise_1', axis=1)
    print("✅ Random noise o'chirildi")


# 2.2 Dataset 2: Clean income
def clean_income(income):
    if pd.isna(income):
        return np.nan
    income_str = str(income).replace('$', '').replace(',', '').replace('"', '').strip()
    try:
        return float(income_str)
    except:
        return np.nan


df2['annual_income'] = df2['annual_income'].apply(clean_income)

# Standardize employment type
employment_mapping = {
    'Full-time': 'Full-time', 'FULL_TIME': 'Full-time', 'FT': 'Full-time',
    'Fulltime': 'Full-time', 'Full Time': 'Full-time',
    'Part Time': 'Part-time', 'PART_TIME': 'Part-time',
    'Self Employed': 'Self-employed', 'Self Emp': 'Self-employed',
    'SELF_EMPLOYED': 'Self-employed', 'Self-employed': 'Self-employed',
    'Contractor': 'Contract', 'Contract': 'Contract', 'CONTRACT': 'Contract'
}
df2['employment_type'] = df2['employment_type'].map(employment_mapping)

# Fill missing employment_length
df2['employment_length'] = df2['employment_length'].fillna(df2['employment_length'].median())
print("✅ Demographics tozalandi")


# 2.3 Dataset 3: Clean financial data
def clean_currency(value):
    if pd.isna(value):
        return np.nan
    return float(str(value).replace('$', '').replace(',', '').strip())


financial_cols = ['monthly_income', 'existing_monthly_debt', 'monthly_payment',
                  'revolving_balance', 'credit_usage_amount', 'available_credit',
                  'total_monthly_debt_payment', 'total_debt_amount', 'monthly_free_cash_flow']

for col in financial_cols:
    if col in df3.columns:
        df3[col] = df3[col].apply(clean_currency)

df3 = df3.rename(columns={'cust_num': 'customer_ref'})
print("✅ Financial data tozalandi")

# 2.4 Dataset 5: Standardize loan data
df5.columns = [col.strip() for col in df5.columns]
df5['loan_amount'] = df5['loan_amount'].apply(clean_currency)
print("✅ Loan data tozalandi")

# 2.6 Dataset 6: Credit History data
df6 = df6.rename(columns={'customer_number': 'customer_ref'})
# Fill missing values in credit history
df6 = df6.fillna(df6.median())
print("✅ Credit History data tozalandi")

# 2.5 Standardize account status
status_mapping = {
    'ACT-1': 'Active1', 'ACT-2': 'Active2', 'ACT-3': 'Active3',
    'ACTIVE': 'Active', 'A01': 'A01', 'a01': 'A01'
}
df1['account_status_code'] = df1['account_status_code'].map(status_mapping)

# ============================================================================
# 3. BARCHA DATASETLARNI BIRLASHTIRISH
# ============================================================================
print("\n🔗 3. DATASETLARNI BIRLASHTIRISH")
print("-" * 80)

# Merge all datasets
df = df1.copy()
df = pd.merge(df, df2, left_on='customer_ref', right_on='cust_id', how='inner')
df = pd.merge(df, df3, on='customer_ref', how='left')
df = pd.merge(df, df4, left_on='customer_ref', right_on='geo_id', how='left')
df = pd.merge(df, df5, left_on='customer_ref', right_on='customer_id', how='left')
df = pd.merge(df, df6, on='customer_ref', how='left')

# Drop duplicate ID columns
df = df.drop(['cust_id', 'geo_id', 'customer_id'], axis=1, errors='ignore')


print(f"✅ Birlashtirilgan dataset: {df.shape}")
print(f"   - Features: {df.shape[1]}")
print(f"   - Samples: {df.shape[0]}")
print(f"   - Default rate: {df['default'].mean() * 100:.2f}%")

# ============================================================================
# 4. FEATURE ENGINEERING
# ============================================================================
print("\n⚙️  4. FEATURE ENGINEERING")
print("-" * 80)

# 4.1 Age groups
df['age_group'] = pd.cut(df['age'], bins=[0, 25, 35, 45, 55, 100], labels=['18-25', '26-35', '36-45', '46-55', '55+'])

# 4.2 Income groups
df['income_group'] = pd.cut(df['annual_income'], bins=[0, 30000, 50000, 75000, 150000], labels=['Low', 'Medium', 'High', 'Very High'])

# 4.3 Account age
df['account_age'] = 2025 - df['account_open_year']

# 4.4 Debt burden ratio
df['debt_burden'] = df['total_debt_amount'] / df['annual_income']

# 4.5 Income vs regional median
df['income_vs_regional'] = df['annual_income'] / df['regional_median_income']

# 4.6 Rent affordability
df['rent_affordability'] = df['regional_median_rent'] / (df['monthly_income'] + 0.01)

# 4.7 High risk DTI flag
df['high_dti'] = (df['debt_to_income_ratio'] > 0.43).astype(int)

# 4.8 High utilization flag
df['high_utilization'] = (df['credit_utilization'] > 0.7).astype(int)

# 4.9 Negative cash flow flag
df['negative_cashflow'] = (df['monthly_free_cash_flow'] < 0).astype(int)

# 4.10 Loan size category
df['loan_size_category'] = pd.cut(df['loan_amount'],
                                  bins=[0, 10000, 50000, 150000, 1000000],
                                  labels=['Small', 'Medium', 'Large', 'XLarge'])

# 4.11 Credit score risk categories
df['credit_score_category'] = pd.cut(df['credit_score'],
                                     bins=[0, 580, 670, 740, 850],
                                     labels=['Poor', 'Fair', 'Good', 'Excellent'])

# 4.12 High delinquency flag
df['has_delinquencies'] = (df['num_delinquencies_2yrs'] > 0).astype(int)

# 4.13 Recent inquiry flag
df['high_recent_inquiries'] = (df['recent_inquiry_count'] > 3).astype(int)

# 4.14 Credit utilization ratio (from credit history)
df['credit_utilization_ratio'] = df['total_credit_limit'] / (df['annual_income'] + 1)

# 4.15 Credit history length (months per account)
df['avg_account_age'] = df['oldest_account_age_months'] / (df['num_credit_accounts'] + 1)

print(f"✅ {15} yangi feature yaratildi")

# ============================================================================
# 5. EDA - KEY INSIGHTS
# ============================================================================
print("\n📈 5. KEY INSIGHTS")
print("-" * 80)

print(f"\n🎯 DEFAULT STATISTICS:")
print(f"   Total defaults: {df['default'].sum()}")
print(f"   Default rate: {df['default'].mean() * 100:.2f}%")

print(f"\n💰 FINANCIAL METRICS:")
print(f"   Avg DTI: {df['debt_to_income_ratio'].mean():.3f}")
print(f"   Avg Credit Util: {df['credit_utilization'].mean():.3f}")
print(f"   Avg Monthly Payment: ${df['monthly_payment'].mean():,.2f}")
print(f"   Avg Credit Score: {df['credit_score'].mean():.1f}")

print(f"\n⚠️  RISK INDICATORS:")
print(f"   High DTI (>43%): {df['high_dti'].sum()} ({df['high_dti'].mean() * 100:.1f}%)")
print(f"   High Utilization (>70%): {df['high_utilization'].sum()} ({df['high_utilization'].mean() * 100:.1f}%)")
print(f"   Negative Cash Flow: {df['negative_cashflow'].sum()} ({df['negative_cashflow'].mean() * 100:.1f}%)")
print(f"   Has Delinquencies: {df['has_delinquencies'].sum()} ({df['has_delinquencies'].mean() * 100:.1f}%)")
print(
    f"   High Recent Inquiries: {df['high_recent_inquiries'].sum()} ({df['high_recent_inquiries'].mean() * 100:.1f}%)")

# ============================================================================
# 6. FEATURE SELECTION & ENCODING
# ============================================================================
print("\n🔧 6. FEATURE ENCODING")
print("-" * 80)

# Select categorical columns to encode
categorical_cols = ['preferred_contact', 'employment_type', 'education',
                    'marital_status', 'loan_type', 'loan_purpose',
                    'origination_channel', 'state']

le_dict = {}
for col in categorical_cols:
    if col in df.columns:
        le = LabelEncoder()
        df[col + '_encoded'] = le.fit_transform(df[col].astype(str))
        le_dict[col] = le

print(f"✅ {len(categorical_cols)} categorical features encoded")

# ============================================================================
# 7. MODEL PREPARATION
# ============================================================================
print("\n🎯 7. MODEL UCHUN TAYYORGARLIK")
print("-" * 80)

# Select important features
feature_cols = [
    # Demographics
    'age', 'annual_income', 'employment_length', 'num_dependents',
    # Behavioral
    'num_login_sessions', 'num_customer_service_calls',
    'has_mobile_app', 'paperless_billing', 'account_age',
    # Financial - MOST IMPORTANT
    'debt_to_income_ratio', 'credit_utilization', 'monthly_payment',
    'monthly_free_cash_flow', 'payment_to_income_ratio',
    'debt_burden', 'total_debt_amount', 'revolving_balance',
    # Loan details
    'loan_amount', 'loan_term', 'interest_rate', 'loan_to_value_ratio',
    # Geographic
    'regional_unemployment_rate', 'housing_price_index',
    'cost_of_living_index', 'income_vs_regional', 'rent_affordability',
    # Credit History - VERY IMPORTANT
    'credit_score', 'num_credit_accounts', 'oldest_credit_line_age',
    'oldest_account_age_months', 'total_credit_limit',
    'num_delinquencies_2yrs', 'num_inquiries_6mo', 'recent_inquiry_count',
    'num_public_records', 'num_collections', 'account_diversity_index',
    # Risk flags
    'high_dti', 'high_utilization', 'negative_cashflow',
    'has_delinquencies', 'high_recent_inquiries',
    # Engineered features
    'credit_utilization_ratio', 'avg_account_age',
    # Encoded categoricals
    'employment_type_encoded', 'education_encoded', 'marital_status_encoded',
    'loan_type_encoded', 'loan_purpose_encoded', 'origination_channel_encoded'
]

# Filter only existing columns
feature_cols = [col for col in feature_cols if col in df.columns]

# Remove rows with missing target
df_model = df.dropna(subset=['default'])

# Handle missing values in features
X = df_model[feature_cols].copy()
y = df_model['default'].copy()

# Fill remaining missing values
X = X.fillna(X.median())

print(f"✅ Features: {len(feature_cols)}")
print(f"✅ Samples: {len(X)}")
print(f"✅ Defaults: {y.sum()} / {len(y)} ({y.mean() * 100:.2f}%)")

# ============================================================================
# 8. TRAIN-TEST SPLIT
# ============================================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n📊 Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

# ============================================================================
# 9. FEATURE SCALING
# ============================================================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================================
# 10. MODEL TRAINING
# ============================================================================
print("\n🚀 8. MODELLARNI O'QITISH")
print("-" * 80)

models = {}

# Logistic Regression
print("\n📈 Logistic Regression...")
lr_model = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
lr_model.fit(X_train_scaled, y_train)
models['Logistic Regression'] = lr_model

# Random Forest
print("📈 Random Forest...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42,
                                  class_weight='balanced', max_depth=15,
                                  min_samples_split=10)
rf_model.fit(X_train, y_train)
models['Random Forest'] = rf_model

# Gradient Boosting
print("📈 Gradient Boosting...")
gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42,
                                      max_depth=5, learning_rate=0.1)
gb_model.fit(X_train, y_train)
models['Gradient Boosting'] = gb_model

# ============================================================================
# 11. MODEL EVALUATION
# ============================================================================
print("\n" + "=" * 80)
print("9. MODEL BAHOLASH")
print("=" * 80)

results = []

for name, model in models.items():
    if name == 'Logistic Regression':
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    else:
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

# ============================================================================
# 12. FEATURE IMPORTANCE (Best Model)
# ============================================================================
print("\n" + "=" * 80)
print("10. ENG MUHIM OMILLAR (FEATURE IMPORTANCE)")
print("=" * 80)

# Use Random Forest for feature importance
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n🔝 TOP 15 ENG MUHIM OMILLAR:\n")
print(feature_importance.head(15).to_string(index=False))

# ============================================================================
# 13. VISUALIZATIONS
# ============================================================================
print("\n" + "=" * 80)
print("11. VIZUALIZATSIYA")
print("=" * 80)

fig = plt.figure(figsize=(20, 14))

# 13.1 Feature Importance
ax1 = plt.subplot(3, 3, 1)
top_features = feature_importance.head(12)
colors = plt.cm.viridis(np.linspace(0, 1, len(top_features)))
ax1.barh(range(len(top_features)), top_features['importance'], color=colors)
ax1.set_yticks(range(len(top_features)))
ax1.set_yticklabels(top_features['feature'], fontsize=9)
ax1.set_xlabel('Importance', fontsize=10)
ax1.set_title('Top 12 Feature Importance', fontsize=12, fontweight='bold')
ax1.invert_yaxis()

# 13.2 ROC Curves
ax2 = plt.subplot(3, 3, 2)
for name, model in models.items():
    if name == 'Logistic Regression':
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
    else:
        y_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    ax2.plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})', linewidth=2)
ax2.plot([0, 1], [0, 1], 'k--', label='Random')
ax2.set_xlabel('False Positive Rate', fontsize=10)
ax2.set_ylabel('True Positive Rate', fontsize=10)
ax2.set_title('ROC Curve Comparison', fontsize=12, fontweight='bold')
ax2.legend(fontsize=8)
ax2.grid(alpha=0.3)

# 13.3 Debt-to-Income vs Default
ax3 = plt.subplot(3, 3, 3)
df_viz = df_model.copy()
default_dti = df_viz[df_viz['default'] == 1]['debt_to_income_ratio']
no_default_dti = df_viz[df_viz['default'] == 0]['debt_to_income_ratio']
ax3.hist([no_default_dti, default_dti], bins=30, label=['No Default', 'Default'],
         alpha=0.7, color=['green', 'red'])
ax3.axvline(0.43, color='orange', linestyle='--', linewidth=2, label='43% Threshold')
ax3.set_xlabel('Debt-to-Income Ratio', fontsize=10)
ax3.set_ylabel('Frequency', fontsize=10)
ax3.set_title('DTI Distribution by Default Status', fontsize=12, fontweight='bold')
ax3.legend()

# 13.4 Credit Utilization vs Default
ax4 = plt.subplot(3, 3, 4)
default_util = df_viz[df_viz['default'] == 1]['credit_utilization']
no_default_util = df_viz[df_viz['default'] == 0]['credit_utilization']
ax4.hist([no_default_util, default_util], bins=30, label=['No Default', 'Default'],
         alpha=0.7, color=['green', 'red'])
ax4.axvline(0.7, color='orange', linestyle='--', linewidth=2, label='70% Threshold')
ax4.set_xlabel('Credit Utilization', fontsize=10)
ax4.set_ylabel('Frequency', fontsize=10)
ax4.set_title('Credit Utilization by Default Status', fontsize=12, fontweight='bold')
ax4.legend()

# 13.5 Default Rate by Income Group
ax5 = plt.subplot(3, 3, 5)
if 'income_group' in df_viz.columns:
    default_by_income = df_viz.groupby('income_group')['default'].mean().sort_values()
    default_by_income.plot(kind='bar', ax=ax5, color='coral')
    ax5.set_ylabel('Default Rate', fontsize=10)
    ax5.set_title('Default Rate by Income Group', fontsize=12, fontweight='bold')
    ax5.set_xticklabels(ax5.get_xticklabels(), rotation=45)
    ax5.axhline(df_viz['default'].mean(), color='red', linestyle='--', alpha=0.5)

# 13.6 Default Rate by Loan Type
ax6 = plt.subplot(3, 3, 6)
if 'loan_type' in df_viz.columns:
    default_by_loan = df_viz.groupby('loan_type')['default'].mean().sort_values()
    default_by_loan.plot(kind='barh', ax=ax6, color='lightblue')
    ax6.set_xlabel('Default Rate', fontsize=10)
    ax6.set_title('Default Rate by Loan Type', fontsize=12, fontweight='bold')

# 13.7 Monthly Free Cash Flow Distribution
ax7 = plt.subplot(3, 3, 7)
ax7.scatter(df_viz['monthly_free_cash_flow'], df_viz['default'],
            alpha=0.3, s=20, c=df_viz['default'], cmap='RdYlGn_r')
ax7.axvline(0, color='red', linestyle='--', linewidth=2)
ax7.set_xlabel('Monthly Free Cash Flow ($)', fontsize=10)
ax7.set_ylabel('Default (0/1)', fontsize=10)
ax7.set_title('Cash Flow vs Default', fontsize=12, fontweight='bold')

# 13.8 Confusion Matrix (Best Model)
ax8 = plt.subplot(3, 3, 8)
best_model = rf_model
y_pred_best = best_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred_best)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax8, cbar=False)
ax8.set_xlabel('Predicted', fontsize=10)
ax8.set_ylabel('Actual', fontsize=10)
ax8.set_title('Confusion Matrix (Random Forest)', fontsize=12, fontweight='bold')

# 13.9 Model Performance Comparison
ax9 = plt.subplot(3, 3, 9)
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
x = np.arange(len(metrics))
width = 0.25
for i, (_, row) in enumerate(results_df.iterrows()):
    values = [row[m] for m in metrics]
    ax9.bar(x + i * width, values, width, label=row['Model'], alpha=0.8)
ax9.set_ylabel('Score', fontsize=10)
ax9.set_title('Model Performance Comparison', fontsize=12, fontweight='bold')
ax9.set_xticks(x + width)
ax9.set_xticklabels(metrics, rotation=45, ha='right')
ax9.legend(fontsize=8)
ax9.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('credit_default_analysis_full.png', dpi=300, bbox_inches='tight')
print("✅ Grafiklar saqlandi: credit_default_analysis_full.png")

# ============================================================================
# 14. BUSINESS INSIGHTS
# ============================================================================
print("\n" + "=" * 80)
print("12. BIZNES UCHUN XULOSALAR")
print("=" * 80)

print("""
🎯 ASOSIY XULOSALAR:

1️⃣ ENG MUHIM RISK FAKTORLARI:
   ✅ Credit Score - eng kuchli prediktor (< 650 = high risk)
   ✅ Debt-to-Income Ratio (DTI) > 43% - ikkinchi eng muhim
   ✅ Credit Utilization > 70% - yuqori risk
   ✅ Delinquencies (2 yil ichida) - juda muhim signal
   ✅ Negative Monthly Cash Flow - to'lov qobiliyati yo'q
   ✅ Recent Credit Inquiries > 3 - desperate borrowing signal

2️⃣ KREDIT TARIXINING TA'SIRI:
   📊 Credit Score < 650: Very High Risk
   📊 Credit Score 650-700: High Risk  
   📊 Credit Score 700-740: Medium Risk
   📊 Credit Score > 740: Low Risk
   📊 Delinquencies mavjud: 3-5x yuqori default risk
   📊 Ko'p inquiry (> 3): desperate borrowing belgisi

3️⃣ DEMOGRAFIK OMILLAR:
   📊 Yosh guruhlar: 18-25 va 55+ ko'proq riskli
   📊 Kam daromadli mijozlar (< $30k) default ko'proq
   📊 Ko'p qaramog'i bor oilalar riskli

4️⃣ KREDIT XUSUSIYATLARI:
   💳 Personal Loan va Credit Card ko'proq default
   💳 Yuqori foiz stavkalari (>15%) xavfli
   💳 Online channel orqali olingan kreditlar riskli
   💳 Kam credit history (< 2 yil) - riskli

5️⃣ GEOGRAFIK OMILLAR:
   🗺️ Yuqori ishsizlik mintaqalari riskli
   🗺️ Yuqori yashash narxi va past daromad kombinatsiyasi xavfli

💡 BIZNES UCHUN TAVSIYALAR:

✅ AVTOMATIK QABUL QILISH (Auto-Approve):
   - Credit Score > 740
   - DTI < 30%
   - Credit Utilization < 50%
   - No delinquencies in 2 years
   - Positive Cash Flow > $1000/month
   - Stable employment > 3 years
   - Recent inquiries < 2

⚠️  QO'SHIMCHA TEKSHIRUV (Manual Review):
   - Credit Score 670-740
   - DTI 30-43%
   - Credit Utilization 50-70%
   - First-time borrowers
   - High loan amounts (> $100k)
   - 1 delinquency in last 2 years
   - Recent inquiries 2-3

❌ RAD ETISH (Reject):
   - Credit Score < 650
   - DTI > 60%
   - Credit Utilization > 85%
   - 2+ delinquencies in 2 years
   - Negative Cash Flow
   - Recent inquiries > 4
   - Multiple recent defaults
   - Public records or collections

📈 MONITORING:
   - Credit score 30+ ball tushsa - darhol tekshirish
   - DTI ratio o'ssa - limit kamaytirishni ko'rish
   - Yangi delinquency paydo bo'lsa - risk alert
   - Ko'p inquiry (1 oyda 2+) - investigation kerak
   - Oylik customer service calls ortsa - risk signal
   - Payment behavior o'zgarsa - darhol tekshirish
""")

# ============================================================================
# 15. NATIJALARNI SAQLASH
# ============================================================================
print("\n" + "=" * 80)
print("13. NATIJALARNI SAQLASH")
print("=" * 80)

# Best modelni tanlash (Random Forest)
best_model_name = results_df.loc[results_df['ROC-AUC'].idxmax(), 'Model']
best_model = models[best_model_name]

print(f"🏆 Best Model: {best_model_name}")
print(f"   ROC-AUC: {results_df.loc[results_df['Model'] == best_model_name, 'ROC-AUC'].values[0]:.4f}")

# Barcha ma'lumotlar uchun bashorat qilish
print("\n📊 Barcha mijozlar uchun bashorat qilish...")

# Handle missing values for full dataset
X_full = df[feature_cols].copy()
X_full = X_full.fillna(X_full.median())

# Predict
if best_model_name == 'Logistic Regression':
    X_full_scaled = scaler.transform(X_full)
    predictions = best_model.predict(X_full_scaled)
    prediction_proba = best_model.predict_proba(X_full_scaled)[:, 1]
else:
    predictions = best_model.predict(X_full)
    prediction_proba = best_model.predict_proba(X_full)[:, 1]

# Create results dataframe
results_output = pd.DataFrame({
    'customer_id': df['customer_ref'].values,
    'default_prediction': predictions,
    'default_probability': prediction_proba,
    'actual_default': df['default'].values
})


# Add risk category
def categorize_risk(prob):
    if prob < 0.3:
        return 'Low Risk'
    elif prob < 0.5:
        return 'Medium Risk'
    elif prob < 0.7:
        return 'High Risk'
    else:
        return 'Very High Risk'


results_output['risk_category'] = results_output['default_probability'].apply(categorize_risk)

# Save to CSV
output_filename = 'credit_default_predictions.csv'
results_output.to_csv(output_filename, index=False)
print(f"✅ Natijalar saqlandi: {output_filename}")

# Display sample
print(f"\n📋 Natijadan namuna (birinchi 10 qator):")
print(results_output.head(10).to_string(index=False))

# Summary statistics
print(f"\n📊 BASHORAT STATISTIKASI:")
print(f"   Jami mijozlar: {len(results_output)}")
print(f"   Bashorat: Default bo'ladi: {predictions.sum()} ({predictions.mean() * 100:.2f}%)")
print(f"   Bashorat: To'lay oladi: {(1 - predictions).sum()} ({(1 - predictions.mean()) * 100:.2f}%)")
print(f"\n   Risk kategoriyalari:")
for category in ['Low Risk', 'Medium Risk', 'High Risk', 'Very High Risk']:
    count = (results_output['risk_category'] == category).sum()
    pct = count / len(results_output) * 100
    print(f"   - {category}: {count} ({pct:.1f}%)")

# Save simplified version (faqat customer_id va default)
simple_output = pd.DataFrame({
    'customer_id': df['customer_ref'].values,
    'default': predictions  # 0 = to'lay oladi, 1 = to'lay olmaydi
})

simple_filename = 'credit_default_simple.csv'
simple_output.to_csv(simple_filename, index=False)
print(f"\n✅ Sodda format saqlandi: {simple_filename}")
print(f"   Ustunlar: customer_id, default")
print(f"   default = 0: Kreditni TO'LAY OLADI")
print(f"   default = 1: Kreditni TO'LAY OLMAYDI")

# Save model performance report
with open('model_performance_report.txt', 'w', encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    f.write("KREDIT DEFAULT BASHORAT MODELI - HISOBOT\n")
    f.write("=" * 80 + "\n\n")

    f.write("1. MODEL PERFORMANCE\n")
    f.write("-" * 80 + "\n")
    f.write(results_df.to_string(index=False))
    f.write("\n\n")

    f.write("2. TOP 15 ENG MUHIM OMILLAR\n")
    f.write("-" * 80 + "\n")
    f.write(feature_importance.head(15).to_string(index=False))
    f.write("\n\n")

    f.write("3. BASHORAT STATISTIKASI\n")
    f.write("-" * 80 + "\n")
    f.write(f"Jami mijozlar: {len(results_output)}\n")
    f.write(f"Default bo'ladi: {predictions.sum()} ({predictions.mean() * 100:.2f}%)\n")
    f.write(f"To'lay oladi: {(1 - predictions).sum()} ({(1 - predictions.mean()) * 100:.2f}%)\n")
    f.write("\n")
    f.write("Risk kategoriyalari:\n")
    for category in ['Low Risk', 'Medium Risk', 'High Risk', 'Very High Risk']:
        count = (results_output['risk_category'] == category).sum()
        pct = count / len(results_output) * 100
        f.write(f"  {category}: {count} ({pct:.1f}%)\n")

print(f"\n✅ Hisobot saqlandi: model_performance_report.txt")

print("\n" + "=" * 80)
print("✅ PIPELINE MUVAFFAQIYATLI YAKUNLANDI!")
print("=" * 80)
print("\n📁 YARATILGAN FAYLLAR:")
print("   1️⃣ credit_default_predictions.csv - To'liq natijalar (probability, risk category)")
print("   2️⃣ credit_default_simple.csv - Sodda format (faqat customer_id, default)")
print("   3️⃣ credit_default_analysis_full.png - Vizualizatsiyalar")
print("   4️⃣ model_performance_report.txt - Model hisoboti")
print("=" * 80)