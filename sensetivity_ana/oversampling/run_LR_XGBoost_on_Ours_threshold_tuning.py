import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score, confusion_matrix, precision_recall_curve, roc_curve
import pickle  # 用于保存和加载模型
import sys
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
import os


output_loss_dir = '/home/linp0a/AMR_prediction_pipeline/loss_curves_balanced_Ours/'
if not os.path.exists(output_loss_dir):
    os.makedirs(output_loss_dir)  # 如果目录不存在，自动创建


output_dir = '/home/linp0a/AMR_prediction_pipeline/model_prediction/model_Ours_final_balanced'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)  # 如果目录不存在，自动创建


# Load data
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
        'dept_ER', 'dept_ICU',
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
       'note_embedding_pca_19', 'specimen_type_urine', 'specimen_type_blood',
        'specimen_type_respiratory', 'specimen_type_skin',
        'specimen_type_stool', 'specimen_type_vaginal_swab', 'specimen_type_biopsy', 
        'specimen_type_fluid', 'specimen_type_throat', 'specimen_type_other', 'specimen_type_na']

final_df = pd.read_csv('/ibex/project/c2205/AMR_dataset_peijun/integrate/final_additional_note_all_ner_present_note_embeddings_20_organism_name_specimen_type.csv', 
                        header=0, 
                        usecols=use_cols)
# final_df = []

# Prepare features
features = final_df.drop(columns=['source', 'resistance_nitrofurantoin', 'resistance_sulfamethoxazole', 'resistance_ciprofloxacin', 'resistance_levofloxacin'])
print(features.columns)

# Clean and convert BMI
features['BMI'] = features['BMI'].replace('12,073.88', np.nan)
features['BMI'] = pd.to_numeric(features['BMI'], errors='coerce')
features.fillna(-1, inplace=True)

# Define prescription columns
prescription_columns = ['resistance_nitrofurantoin', 'resistance_sulfamethoxazole', 'resistance_ciprofloxacin', 'resistance_levofloxacin']

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

# Function to optimize thresholds
def optimize_thresholds(y_true, y_proba, model_name, prescription):
    thresholds = np.linspace(0, 1, 100)
    best_threshold = 0.0
    best_f1_score = 0.0
    best_metrics = None

    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        metrics = calculate_metrics(y_true, y_pred, y_proba)
        
        if metrics['F1 Score'] > best_f1_score:
            best_f1_score = metrics['F1 Score']
            best_threshold = threshold
            best_metrics = metrics

    print(f"Best threshold for {prescription} using {model_name}: {best_threshold}")
    print(f"Metrics: {best_metrics}")
    
    return best_threshold, best_metrics

# Train models and optimize thresholds
for i, prescription in enumerate(prescription_columns):
    print(f"Training models for {prescription} ({i + 1}/{len(prescription_columns)})...")
    sys.stdout.flush()

    # Prepare labels for binary classification
    y = final_df[prescription]
    
    valid_indices = y.notna()
    X = features[valid_indices]
    y_binary = y[valid_indices]

     # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.2, random_state=42)
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    # Print shapes
    print("Before oversampling:", X_train.shape)
    print("After oversampling:", X_train_resampled.shape)
    sys.stdout.flush()
    
    # Logistic Regression with GridSearchCV
    lr_model = LogisticRegression(max_iter=1000)
    lr_param_grid = {'C': [0.001, 0.01, 0.1, 1, 10]}
    lr_grid_search = GridSearchCV(lr_model, lr_param_grid, cv=5)
    lr_grid_search.fit(X_train_resampled, y_train_resampled)

    # Save Logistic Regression model
    with open(os.path.join(output_dir, f'lr_model_{prescription}.pkl'), 'wb') as f:
        pickle.dump(lr_grid_search.best_estimator_, f)

    # Predictions for Logistic Regression
    y_proba_lr = lr_grid_search.predict_proba(X_test)[:, 1]
    
    # Optimize threshold for Logistic Regression
    best_threshold_lr, best_metrics_lr = optimize_thresholds(y_test, y_proba_lr, "Logistic Regression", prescription)

    results_dict[prescription] = {'Logistic Regression': best_metrics_lr}

    # XGBoost with GridSearchCV using GPU
    xgb_model = XGBClassifier(use_label_encoder=False, tree_method='gpu_hist', gpu_id=0)
    xgb_param_grid = {'n_estimators': [10, 50, 100, 200], 'max_depth': [3, 5, 10, 20]}
    
    
    # Define eval_set for tracking train and test losses
    eval_set = [(X_train_resampled, y_train_resampled), (X_test, y_test)]
    
    # Train XGBoost with evaluation set
    xgb_grid_search = GridSearchCV(xgb_model, xgb_param_grid, cv=5)
    xgb_grid_search.fit(X_train_resampled, y_train_resampled, eval_set=eval_set, eval_metric="logloss", early_stopping_rounds=10)

    # Save XGBoost model
    with open(os.path.join(output_dir, f'xgboost_model_{prescription}.pkl'), 'wb') as f:
        pickle.dump(xgb_grid_search.best_estimator_, f)

    # Predictions for XGBoost
    y_proba_xgb = xgb_grid_search.predict_proba(X_test)[:, 1]
    
    # Optimize threshold for XGBoost
    best_threshold_xgb, best_metrics_xgb = optimize_thresholds(y_test, y_proba_xgb, "XGBoost", prescription)

    results_dict[prescription]['XGBoost'] = best_metrics_xgb

    # Get and plot training and test log loss for XGBoost
    results = xgb_grid_search.best_estimator_.evals_result()
    
    epochs = len(results['validation_0']['logloss'])
    x_axis = range(epochs)
    
    # Plot log loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(x_axis, results['validation_0']['logloss'], label='Train')
    plt.plot(x_axis, results['validation_1']['logloss'], label='Test')
    plt.title(f'Log Loss for {prescription}')
    plt.xlabel('Epochs')
    plt.ylabel('Log Loss')
    plt.legend()
    plt.savefig(os.path.join(output_loss_dir , f'xgb_loss_curve_{prescription}.png'))  # Save to specified directory

    plt.show()
    
    print(f"Finished training models for {prescription}.")

# Print results for each prescription column and model
for prescription in results_dict:
    print(f"\"{prescription}\": {{")
    for model in results_dict[prescription]:
        metrics = results_dict[prescription][model]
        print(f"\"{model}\": {metrics}")
    print("}")

