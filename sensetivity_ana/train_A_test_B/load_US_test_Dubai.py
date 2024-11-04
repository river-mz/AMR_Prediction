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
from xgboost import plot_importance
import os
from sklearn.metrics import roc_auc_score, classification_report
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from skorch import NeuralNetClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import auc


model_dir = '/home/linp0a/AMR_prediction_pipeline/model_prediction/model_Ours_US_Oct_5'

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
       'note_embedding_pca_19', 'entity_present_pca_0', 'entity_present_pca_1',
       'entity_present_pca_2', 'entity_present_pca_3', 'entity_present_pca_4',
       'entity_present_pca_5', 'entity_present_pca_6', 'entity_present_pca_7',
       'entity_present_pca_8', 'entity_present_pca_9', 'entity_present_pca_10',
       'entity_present_pca_11', 'entity_present_pca_12',
       'entity_present_pca_13', 'entity_present_pca_14',
       'entity_present_pca_15', 'entity_present_pca_16',
       'entity_present_pca_17', 'entity_present_pca_18',
       'entity_present_pca_19']

final_df = pd.read_csv('/ibex/project/c2205/AMR_dataset_peijun/integrate/final_note_embeddings_20_organism_name_specimen_type_all_ner_present20_2.csv', 
                        header=0,
                        usecols=use_cols)

final_df = final_df[final_df['source']=='country1']
features = final_df.drop(columns=['source', 'resistance_nitrofurantoin', 'resistance_sulfamethoxazole', 'resistance_ciprofloxacin', 'resistance_levofloxacin'])
print(features.columns)

# Clean and convert BMI
features['BMI'] = features['BMI'].replace('12,073.88', np.nan)
features['BMI'] = pd.to_numeric(features['BMI'], errors='coerce')
features.fillna(-1, inplace=True)

# Define prescription columns
prescription_columns = ['resistance_nitrofurantoin', 'resistance_sulfamethoxazole', 'resistance_ciprofloxacin', 'resistance_levofloxacin']

results_dict = {}



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


# Step 2: Build a Logistic Regression model using PyTorch
class LogisticRegressionModule(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegressionModule, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        logits = self.linear(x)
        return logits  # 注意，这里不再应用 sigmoid

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

    # print(f"Best threshold for {prescription} using {model_name}: {best_threshold}")
    # print(f"Metrics: {best_metrics}")
    
    return best_threshold, best_metrics



# Train models and optimize thresholds
for i, prescription in enumerate(prescription_columns):
    print(f"Testing models for {prescription} ({i + 1}/{len(prescription_columns)})...")
    sys.stdout.flush()

        # Prepare labels for binary classification
    y = final_df[prescription]
    
    valid_indices = y.notna()

    if valid_indices.sum() == 0:
        print('Skipping testing for', prescription)
        continue
    X = features[valid_indices]
    y_binary = y[valid_indices]


    # Load Logistic Regression model
    lr_model_path = os.path.join(model_dir, f'lr_model_{prescription}.pth')
    lr_model = torch.load(lr_model_path)
    lr_model.eval()  # Set the model to evaluation mode

    # Convert X to a tensor for PyTorch
    X_tensor = torch.tensor(X.values, dtype=torch.float32)

    # Get predictions from the Logistic Regression model
    with torch.no_grad():
        lr_logits = lr_model(X_tensor)
        lr_proba = torch.sigmoid(lr_logits).numpy()  # Apply sigmoid to get probabilities

    # Load XGBoost model
    xgb_model_path = os.path.join(model_dir, f'xgb_model_{prescription}.pkl')
    with open(xgb_model_path, 'rb') as f:
        xgb_model = pickle.load(f)

    # Get predictions from the XGBoost model
    xgb_proba = xgb_model.predict_proba(X)[:, 1]  # Get probability of the positive class

    # Optimize thresholds for both models
    lr_best_threshold, lr_best_metrics = optimize_thresholds(y_binary.values, lr_proba, 'Logistic Regression', prescription)
    xgb_best_threshold, xgb_best_metrics = optimize_thresholds(y_binary.values, xgb_proba, 'XGBoost', prescription)

    results_dict[prescription] = {
        'Logistic Regression': {
            'Best Threshold': lr_best_threshold,
            'Metrics': lr_best_metrics
        },
        'XGBoost': {
            'Best Threshold': xgb_best_threshold,
            'Metrics': xgb_best_metrics
        }
    }

    print(f"Results for {prescription}:")
    # print(f"Logistic Regression - Best Threshold: {lr_best_threshold}, Metrics: {lr_best_metrics}")
    # print(f"XGBoost - Best Threshold: {xgb_best_threshold}, Metrics: {xgb_best_metrics}")
    print(f"Logistic Regression: {lr_best_metrics}")
    print(f"XGBoost: {xgb_best_metrics}")