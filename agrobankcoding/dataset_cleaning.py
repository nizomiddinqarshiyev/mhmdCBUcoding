import pandas as pd
import numpy as np
import json
import xml.etree.ElementTree as ET

import warnings

warnings.filterwarnings('ignore')
data_list = ['datasets/application_metadata.csv',
             'datasets/demographics.csv',
             'datasets/financial_ratios.jsonl',
             'datasets/geographic_data.xml',
             'datasets/loan_details.xlsx'
             'datasets/credit_history.parquet'
             ]

def clean_dataset(data_list, output_list):
    print("=" * 80)
    print(" " * 20 + "DATASETLARNI TOZALASH")
    print("=" * 80)

    # ============================================================================
    # 1. DATASETLARNI YUKLASH
    # ============================================================================
    print("\n📂 1. DATASETLARNI YUKLASH")
    print("-" * 80)

    # Dataset 1: Application & Behavioral
    df1 = pd.read_csv(data_list[0])
    print(f"✅ Dataset 1 (Behavioral): {df1.shape}")

    # Dataset 2: Demographics
    df2 = pd.read_csv(data_list[1])
    print(f"✅ Dataset 2 (Demographics): {df2.shape}")

    # Dataset 3: Financial Debt (JSON)
    with open(data_list[2], 'r') as f:
        data_json = [json.loads(line) for line in f]
    df3 = pd.DataFrame(data_json)
    print(f"✅ Dataset 3 (Financial Debt): {df3.shape}")

    # Dataset 4: Geographic (XML)
    tree = ET.parse(data_list[3])
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
    df5 = pd.read_excel(data_list[4])
    print(f"✅ Dataset 5 (Loan Details): {df5.shape}")

    print(df5.columns)

    # Dataset 6: Credit History (Parquet)
    df6 = pd.read_parquet(data_list[5])
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
    df2['employment_type'] = df2['employment_type'].map(employment_mapping).fillna('Unknown')

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
    # loan_type ustunini stringga o'tkazish va bo'sh joylarni olib tashlash
    df5['loan_type'] = df5['loan_type'].astype(str).str.strip().str.lower()

    # Standartlashtirish uchun map yaratish
    loan_type_map = {
        'personal': 'Personal',
        'personal loan': 'Personal',
        'mortgage': 'Mortgage',
        'home loan': 'Mortgage',
        'credit card': 'Credit Card',
        'cc': 'Credit Card',
        '': 'Unknown'
    }

    # Qo'llash
    df5['loan_type'] = df5['loan_type'].map(loan_type_map).fillna('Unknown')

    # Natijani tekshirish
    print(df5['loan_type'].unique())

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
    print(df.columns)

    col = [
        'age',
        'annual_income',
        'employment_length',
        'employment_type',
        'education',
        'marital_status',
        'num_dependents',
        'monthly_income',
        'existing_monthly_debt',
        'monthly_payment',
        'debt_to_income_ratio',
        'payment_to_income_ratio',
        'credit_score',
        'num_credit_accounts',
        'num_delinquencies_2yrs',
        'num_inquiries_6mo',
        'recent_inquiry_count',
        'credit_utilization',
        'available_credit',
        'loan_amount',
        'loan_term',
        'interest_rate',
        'loan_to_value_ratio',
        'total_debt_amount',
        'monthly_free_cash_flow'
    ]
    df['revolving_balance'] = df['revolving_balance'].fillna(0)
    print(df['revolving_balance'].isnull().sum())
    print(df.isnull().sum().sum())
    features_path = 'results/cleaned_features.csv'
    df.to_csv(output_list[0], index=False)
    df.to_csv(output_list[1], index=False, columns=col)

    print(
        f"============================={len(df)} ta toza data datasets fayli ichiga saqlandi ============================")

    return {'df': df, 'df_ft_path': features_path}


