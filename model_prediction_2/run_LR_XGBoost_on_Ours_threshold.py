import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score, confusion_matrix
import pickle  # 用于保存和加载模型
import sys
# Load data
from sklearn.preprocessing import PolynomialFeatures


import pandas as pd

# 指定需要读取的列
use_cols = ['age', 'race', 'veteran', 'gender', 'BMI', 'previous_antibiotic_exposure_cephalosporin',
       'previous_antibiotic_exposure_carbapenem',
       'previous_antibiotic_exposure_fluoroquinolone',
       'previous_antibiotic_exposure_polymyxin',
       'previous_antibiotic_exposure_aminoglycoside',
       'previous_antibiotic_exposure_nitrofurantoin',
       'previous_antibiotic_resistance_ciprofloxacin',
       'previous_antibiotic_resistance_levofloxacin',
       'previous_antibiotic_resistance_nitrofurantoin',
       'previous_antibiotic_resistance_sulfamethoxazole','resistance_nitrofurantoin', 'resistance_sulfamethoxazole',
       'resistance_ciprofloxacin', 'resistance_levofloxacin', 'source',
       'patience_id', 'encounter_id', 'sample_id', 'dept_ER', 'dept_ICU',
       'dept_IP', 'dept_OP', 'dept_nan',
       'Enterococcus_faecium', 'Staphylococcus_aureus',
       'Klebsiella_pneumoniae', 'Acinetobacter_baumannii',
       'Pseudomonas_aeruginosa', 'Enterobacter', 'organism_other',
       'organism_NA', 'note_embedding_pca_0', 'note_embedding_pca_1',
       'note_embedding_pca_2', 'note_embedding_pca_3', 'note_embedding_pca_4',
       'note_embedding_pca_5', 'note_embedding_pca_6', 'note_embedding_pca_7',
       'note_embedding_pca_8', 'note_embedding_pca_9', 'note_embedding_pca_10',
       'note_embedding_pca_11', 'note_embedding_pca_12',
       'note_embedding_pca_13', 'note_embedding_pca_14',
       'note_embedding_pca_15', 'note_embedding_pca_16',
       'note_embedding_pca_17', 'note_embedding_pca_18',
       'note_embedding_pca_19'
       ]  

# todo ： 'previous_infecting_organisms', 'current_empirical_antibiotics'

final_df = pd.read_csv('/ibex/project/c2205/AMR_dataset_peijun/integrate/final_additional_note_all_ner_present_note_embeddings_20_organism_name.csv', 
                        header=0, 
                        nrows = 300,
                        usecols=use_cols)


# final_df = pd.read_csv('/ibex/project/c2205/AMR_dataset_peijun/integrate/final_additional_note_all_ner_present_note_embeddings_20_organism_name.csv', header=0, nrows = 300)
# final_df = final_df[:10000]

# Prepare features
features = final_df.drop(columns=[ 'source',
                                   'resistance_nitrofurantoin', 'resistance_sulfamethoxazole',
                                   'resistance_ciprofloxacin', 'resistance_levofloxacin',
                                   ])

print(features.columns)
print(features[:1])
# Clean and convert BMI
features['BMI'] = features['BMI'].replace('12,073.88', np.nan)
features['BMI'] = pd.to_numeric(features['BMI'], errors='coerce')
features.fillna(-1, inplace=True)

output_folder = 'model'
sys.stdout.flush()
# Define prescription columns
prescription_columns = ['resistance_nitrofurantoin', 
                        'resistance_sulfamethoxazole', 
                        'resistance_ciprofloxacin', 
                        'resistance_levofloxacin']

results_dict = {}

# Function to calculate evaluation metrics
def calculate_metrics(y_true, y_pred, y_proba):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1_score = 2 * (ppv * sensitivity) / (ppv + sensitivity) if (ppv + sensitivity) > 0 else 0
    
    auroc = roc_auc_score(y_true, y_proba)
    auprc = average_precision_score(y_true, y_proba)

    return {
        "AUROC": np.round(auroc, 3),
        "AUPRC": np.round(auprc, 3),
        "Sensitivity": np.round(sensitivity, 3),
        "Specificity": np.round(specificity, 3),
        "PPV": np.round(ppv, 3),
        "F1 Score": np.round(f1_score, 3)
    }

# Define thresholds for each model and prescription
thresholds_lr = {'resistance_nitrofurantoin': 0.2041, 
                  'resistance_sulfamethoxazole': 0.2176, 
                  'resistance_ciprofloxacin': 0.2351, 
                  'resistance_levofloxacin': 0.2131}

thresholds_xgb = {'resistance_nitrofurantoin': 0.2161, 
                   'resistance_sulfamethoxazole': 0.2296, 
                   'resistance_ciprofloxacin': 0.2521, 
                   'resistance_levofloxacin': 0.2416}

# Loop through each prescription column for binary classification
for i, prescription in enumerate(prescription_columns):
    print(f"Training models for {prescription} ({i + 1}/{len(prescription_columns)})...")
    sys.stdout.flush()
    # Prepare labels for binary classification
    y = final_df[prescription]
    
    # Keep only rows where the label is not NA
    valid_indices = y.notna()
    X = features[valid_indices]
    y_binary = y[valid_indices]
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.2, random_state=42)

    # Logistic Regression with GridSearchCV
    lr_model = LogisticRegression(max_iter=1000)
    lr_param_grid = {'C': [0.001, 0.01, 0.1, 1, 10]}
    lr_grid_search = GridSearchCV(lr_model, lr_param_grid, cv=5)
    lr_grid_search.fit(X_train, y_train)

    # Save Logistic Regression model as pickle
    with open(f'/home/linp0a/AMR_prediction_pipeline/model_prediction/model/lr_model_{prescription}.pkl', 'wb') as f:
        pickle.dump(lr_grid_search.best_estimator_, f)
    print("Finished LR")
    sys.stdout.flush()
    # Predictions and probabilities for Logistic Regression
    y_proba_lr = lr_grid_search.predict_proba(X_test)[:, 1]
    
    # Apply threshold for Logistic Regression
    y_pred_lr_thresholded = (y_proba_lr >= thresholds_lr[prescription]).astype(int)
    
    # Store metrics for Logistic Regression
    results_dict[prescription] = {'Logistic Regression': calculate_metrics(y_test, y_pred_lr_thresholded, y_proba_lr)}

    # XGBoost with GridSearchCV using GPU
    xgb_model = XGBClassifier(use_label_encoder=False, tree_method='gpu_hist', gpu_id=0)
    xgb_param_grid = {'n_estimators': [10, 50, 100], 'max_depth': [3, 5, 10]}
    xgb_grid_search = GridSearchCV(xgb_model, xgb_param_grid, cv=5)
    xgb_grid_search.fit(X_train, y_train)

    # Save XGBoost model as pickle
    
    with open(f'/home/linp0a/AMR_prediction_pipeline/model_prediction/model/xgb_model_{prescription}.pkl', 'wb') as f:
        pickle.dump(xgb_grid_search.best_estimator_, f)
    print("Finished xgboost")
    sys.stdout.flush()
    # Predictions and probabilities for XGBoost
    y_proba_xgb = xgb_grid_search.predict_proba(X_test)[:, 1]
    
    # Apply threshold for XGBoost
    y_pred_xgb_thresholded = (y_proba_xgb >= thresholds_xgb[prescription]).astype(int)
    
    # Store metrics for XGBoost
    results_dict[prescription]['XGBoost'] = calculate_metrics(y_test, y_pred_xgb_thresholded, y_proba_xgb)




    # 生成二次多项式特征
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    # Logistic Regression 使用多项式特征
    lr_model_poly = LogisticRegression(max_iter=1000)
    lr_param_grid_poly = {'C': [0.001, 0.01, 0.1, 1, 10]}
    lr_grid_search_poly = GridSearchCV(lr_model_poly, lr_param_grid_poly, cv=5)
    lr_grid_search_poly.fit(X_train_poly, y_train)

    # 保存多项式 Logistic Regression 模型
    with open(f'/home/linp0a/AMR_prediction_pipeline/model_prediction/model/lr_model_poly_{prescription}.pkl', 'wb') as f:
        pickle.dump(lr_grid_search_poly.best_estimator_, f)

    # 预测及阈值调整
    y_proba_lr_poly = lr_grid_search_poly.predict_proba(X_test_poly)[:, 1]
    y_pred_lr_thresholded_poly = (y_proba_lr_poly >= thresholds_lr[prescription]).astype(int)

    # 存储多项式 Logistic Regression 的评估指标
    results_dict[prescription]['Logistic Regression (Polynomial)'] = calculate_metrics(y_test, y_pred_lr_thresholded_poly, y_proba_lr_poly)

    print(f"Finished training models for {prescription}.")
    sys.stdout.flush()

# Print results for each prescription column and model
for prescription in results_dict:
    print(f"\"{prescription}\": {{")
    for model in results_dict[prescription]:
        metrics = results_dict[prescription][model]
        print(f"\"{model}\": {metrics}")
    print("}")

# 加载模型的示例
# with open('lr_model_resistance_nitrofurantoin.pkl', 'rb') as f:
#     loaded_lr_model = pickle.load(f)
# with open('xgb_model_resistance_nitrofurantoin.pkl', 'rb') as f:
#     loaded_xgb_model = pickle.load(f)
