Loyiha agrobankcoding fayli ichida joylashgan.

results.csv  ======> natijalar

main_pipeline.py model uchun yozilgan kodlar

dataset_cleaning.py datasetlarni tozalash, birlashtirish va yangi faylga saqlash uchun

models/ bu modellarni saqlash uchun



================================================================================
          KREDIT DEFAULT BASHORAT QILISH MODELI
               (Train & Evaluation Pipeline)
================================================================================

================================================================================
QISM 1: TRAINING DATA
================================================================================

📂 Training datasetlarni tozalash...
================================================================================
                    DATASETLARNI TOZALASH
================================================================================

📂 1. DATASETLARNI YUKLASH
--------------------------------------------------------------------------------
✅ Dataset 1 (Behavioral): (89999, 14)

✅ Dataset 2 (Demographics): (89999, 8)

✅ Dataset 3 (Financial Debt): (89999, 16)

✅ Dataset 4 (Geographic): (89999, 7)

✅ Dataset 5 (Loan Details): (89999, 10)

✅ Dataset 6 (Credit History): (89999, 12)

📊 2. MA'LUMOTLARNI TOZALASH
--------------------------------------------------------------------------------
✅ Random noise o'chirildi

✅ Demographics tozalandi

✅ Financial data tozalandi

✅ Loan data tozalandi

✅ Credit History data tozalandi

🔗 3. DATASETLARNI BIRLASHTIRISH

=============================89999 ta toza data datasets fayli ichiga saqlandi ============================

✅ Training data tayyor: (89999, 61)

   Default rate: 5.10%

================================================================================
QISM 2: FEATURE SELECTION
================================================================================

🔧 Categorical features ni encode qilish...

✅ Jami features: 25

   Numerical: 22
   
   Categorical (encoded): 3

================================================================================
QISM 3: DATA PREPARATION
================================================================================

✅ Training samples: 89999

✅ Features: 25

✅ Defaults: 4594 (5.10%)

✅ Non-defaults: 85405 (94.90%)

📊 Train set: 71999 samples

📊 Test set: 18000 samples

⚙️  Feature scaling...

================================================================================
QISM 4: MODEL TRAINING
================================================================================

📈 1. Random Forest Training...

   ✅ Random Forest trained

📈 2. Gradient Boosting Training...

   ✅ Gradient Boosting trained

📈 3. XGBoost Training...

   ✅ XGBoost trained

================================================================================
QISM 5: MODEL EVALUATION (TEST SET)
================================================================================

==================================================
🎯 Random Forest
==================================================
Accuracy:  0.8764

Precision: 0.2065

Recall:    0.4995

F1-Score:  0.2922

ROC-AUC:   0.7974

==================================================
🎯 Gradient Boosting
==================================================
Accuracy:  0.9496
Precision: 0.5462
Recall:    0.0707
F1-Score:  0.1252
ROC-AUC:   0.8068

==================================================
🎯 XGBoost
==================================================

Accuracy:  0.9496

Precision: 0.5652

Recall:    0.0566

F1-Score:  0.1029

ROC-AUC:   0.8032

📊 MODEL COMPARISON:
            Model  Accuracy  Precision   Recall  F1-Score  ROC-AUC
    Random Forest  0.876444   0.206478 0.499456  0.292171 0.797397
Gradient Boosting  0.949556   0.546218 0.070729  0.125241 0.806850
          XGBoost  0.949611   0.565217 0.056583  0.102868 0.803162

🏆 BEST MODEL: Gradient Boosting

   ROC-AUC: 0.8068

================================================================================
QISM 6: FEATURE IMPORTANCE
================================================================================

🔝 TOP 15 ENG MUHIM OMILLAR:

                feature  importance
                
           credit_score    0.207769
           
   debt_to_income_ratio    0.140056
   
 monthly_free_cash_flow    0.113930
 
                    age    0.082817
                    
payment_to_income_ratio    0.054574

       available_credit    0.045657
       
     credit_utilization    0.044740
     
          interest_rate    0.044561
          
    num_credit_accounts    0.042738
    
  existing_monthly_debt    0.034782
  
      total_debt_amount    0.033112
      
        monthly_payment    0.024283
        
            loan_amount    0.022207
            
      employment_length    0.021863
      
    loan_to_value_ratio    0.018431

================================================================================
QISM 7: MODEL VA ARTIFACTS NI SAQLASH
================================================================================

✅ Model saqlandi: models/best_model.pkl

✅ Scaler saqlandi: models/scaler.pkl

✅ Label encoders saqlandi: models/label_encoders.pkl

✅ Feature columns saqlandi: models/feature_columns.pkl

✅ Model performance saqlandi: results/model_performance.csv

================================================================================
QISM 8: EVALUATION DATA - FORECAST
================================================================================

📂 Evaluation datasetlarni tozalash...
================================================================================
                    DATASETLARNI TOZALASH
================================================================================

📂 1. DATASETLARNI YUKLASH
--------------------------------------------------------------------------------

✅ Dataset 1 (Behavioral): (10001, 13)

✅ Dataset 2 (Demographics): (10001, 8)

✅ Dataset 3 (Financial Debt): (10001, 16)

✅ Dataset 4 (Geographic): (10001, 7)

✅ Dataset 5 (Loan Details): (10001, 10)

✅ Dataset 6 (Credit History): (10001, 12)

📊 2. MA'LUMOTLARNI TOZALASH
--------------------------------------------------------------------------------
✅ Random noise o'chirildi

✅ Demographics tozalandi

✅ Financial data tozalandi

✅ Loan data tozalandi

✅ Credit History data tozalandi

🔗 3. DATASETLARNI BIRLASHTIRISH
--------------------------------------------------------------------------------
0

0

============================= 10001 ta toza data datasets fayli ichiga saqlandi ============================

✅ Evaluation data tayyor: (10001, 60)

🔧 Evaluation data ni tayyorlash...

✅ Evaluation samples: 10001

🎯 Bashorat qilish...

✅ Bashoratlar tayyor

📊 Natijalarni yaratish...

💾 Natijalarni saqlash...

✅ Predictions saqlandi: results.csv

✅ Detailed predictions saqlandi: results/predictions_detailed.csv

================================================================================
QISM 9: NATIJALAR STATISTIKASI
================================================================================

📋 FORECAST RESULTS:

   Jami mijozlar: 10001
   
   Kredit BERILSIN (default=0): 9919 ta (99.18%)
   
   Kredit BERILMASIN (default=1): 82 ta (0.82%)
   
   O'rtacha probability: 0.0515

📊 RISK TAQSIMOTI:

   Low Risk            :  9781 ta ( 97.8%)
   
   Medium Risk         :   138 ta (  1.4%)
   
   High Risk           :    58 ta (  0.6%)
   
   Very High Risk      :    24 ta (  0.2%)
   

💡 TAVSIYALAR:

   AUTO-APPROVE        :  9781 ta ( 97.8%)
   
   MANUAL REVIEW       :   138 ta (  1.4%)
   
   HIGH RISK - CAREFUL :    58 ta (  0.6%)
   
   AUTO-REJECT         :    24 ta (  0.2%)

📋 NATIJADAN NAMUNA (birinchi 10 ta):

 customer_id     prob  default risk_category recommendation
 
      100000 0.008988        0      Low Risk   AUTO-APPROVE
      
      100001 0.018375        0      Low Risk   AUTO-APPROVE
      
      100002 0.083635        0      Low Risk   AUTO-APPROVE
      
      100003 0.059499        0      Low Risk   AUTO-APPROVE
      
      100004 0.044157        0      Low Risk   AUTO-APPROVE
      
      100005 0.022641        0      Low Risk   AUTO-APPROVE
      
      100006 0.011188        0      Low Risk   AUTO-APPROVE
      
      100007 0.029375        0      Low Risk   AUTO-APPROVE
      
      100008 0.007117        0      Low Risk   AUTO-APPROVE
      
      100009 0.030507        0      Low Risk   AUTO-APPROVE

================================================================================
QISM 10: VIZUALIZATSIYA
================================================================================

✅ Vizualizatsiya saqlandi: results/analysis_visualization.png

================================================================================

✅ PIPELINE MUVAFFAQIYATLI YAKUNLANDI!

================================================================================

📁 YARATILGAN FAYLLAR:

   1. models/best_model.pkl - O'qitilgan model
   
   2. models/scaler.pkl - Feature scaler

   3. models/label_encoders.pkl - Categorical encoders

   4. models/feature_columns.pkl - Feature nomi

   5. results/predictions.csv - Asosiy natijalar

   6. results/predictions_detailed.csv - Batafsil natijalar

   7. results/model_performance.csv - Model performance

   8. results/analysis_visualization.png - Grafiklar

🏆 Best Model: Gradient Boosting

📊 ROC-AUC: 0.8068

📈 Total Predictions: 10001

💡 QANDAY ISHLATISH:

   1. Training: datasets/ papkasidagi datasetlar bilan model o'qitildi
      
   2. Evaluation: evaluation/ papkasidagi datasetlar forecast qilindi

   3. Results: results/ papkasida barcha natijalar saqlandi
    
   4. Model: models/ papkasida keyingi forecast uchun saqlandi

================================================================================

Process finished with exit code 0

