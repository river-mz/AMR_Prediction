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
import argparse  # Import argparse for command-line arguments
import os
from sklearn.metrics import roc_auc_score, classification_report
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from skorch import NeuralNetClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import auc

# output_fig_dir = '/home/linp0a/AMR_prediction_pipeline/loss_auc_curves_Ours_Oct_5/'
# if not os.path.exists(output_fig_dir):
#     os.makedirs(output_fig_dir)  # 如果目录不存在，自动创建


# output_dir = '/home/linp0a/AMR_prediction_pipeline/model_prediction/model_Ours_Oct_5'
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)  # 如果目录不存在，自动创建



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

# Prepare features
features = final_df.drop(columns=['source', 'resistance_nitrofurantoin',
                                   'resistance_sulfamethoxazole',
                                   'resistance_ciprofloxacin',
                                   'resistance_levofloxacin'])

print(features.columns)

# Clean and convert BMI features
features['BMI'] = features['BMI'].replace('12,073.88', np.nan)
features['BMI'] = pd.to_numeric(features['BMI'], errors='coerce')
features.fillna(0, inplace=True)

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

class LogisticRegressionModule(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegressionModule, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        logits = self.linear(x)
        return logits  # 注意，这里不再应用 sigmoid
    
# 设置5-fold分割
n_splits = 5
kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

best_lr_params = {
    'resistance_nitrofurantoin':  {'batch_size': 64, 'lr': 0.0001, 'max_epochs': 50, 'optimizer__weight_decay': 0},

    'resistance_sulfamethoxazole': {'batch_size': 64, 'lr': 0.0001, 'max_epochs': 100, 'optimizer__weight_decay': 0},

    'resistance_ciprofloxacin': {'batch_size': 64, 'lr': 0.0001, 'max_epochs': 50, 'optimizer__weight_decay': 0},

    'resistance_levofloxacin': {'batch_size': 32, 'lr': 0.0001, 'max_epochs': 100, 'optimizer__weight_decay': 0}
}


best_xgb_params = {
    'resistance_nitrofurantoin': {'max_depth': 3, 'n_estimators': 50, 'subsample': 0.8},
    'resistance_sulfamethoxazole': {'max_depth': 3, 'n_estimators': 50, 'subsample': 0.8},
    'resistance_ciprofloxacin': {'max_depth': 3, 'n_estimators': 50, 'subsample': 0.8},
    'resistance_levofloxacin': {'max_depth': 5, 'n_estimators': 10, 'subsample': 0.8},
}

# Train models and optimize thresholds
for i, prescription in enumerate(prescription_columns):
    print()
    print(f"Training models for {prescription} ({i + 1}/{len(prescription_columns)})...")
    sys.stdout.flush()

    # Prepare labels for binary classification
    y = final_df[prescription]
    
    valid_indices = y.notna()

    if valid_indices.sum() == 0:
        print('Skipping training for', prescription)
        continue
    X = features[valid_indices]
    y_binary = y[valid_indices]
    
    fold_results_lr = []  # 保存Logistic Regression每个fold的metrics
    fold_results_xgb = []  # 保存XGBoost每个fold的metrics



    # 对5折交叉验证进行循环
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X, y_binary)):
        # print(f"Fold {fold_idx + 1}/{n_splits} for {prescription}")

        # 划分训练集和测试集
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y_binary.iloc[train_idx], y_binary.iloc[test_idx]



        best_lr_model = LogisticRegressionModule(input_dim=X.shape[1])
        optimizer = optim.Adam(best_lr_model.parameters(), lr=best_lr_params[prescription]['lr'], weight_decay=best_lr_params[prescription]['optimizer__weight_decay'])
        criterion = nn.BCEWithLogitsLoss()



        train_data = TensorDataset(torch.tensor(X_train.values.astype(np.float32)), torch.tensor(y_train.values.astype(np.float32)))
        test_data = TensorDataset(torch.tensor(X_test.values.astype(np.float32)), torch.tensor(y_test.values.astype(np.float32)))
        train_loader = DataLoader(train_data, batch_size=best_lr_params[prescription]['batch_size'],shuffle=True)
        test_loader = DataLoader(test_data, batch_size=best_lr_params[prescription]['batch_size'], shuffle=False)


        train_loss_list = []
        test_loss_list = []
        train_auroc_list = []
        test_auroc_list = []

        for epoch in range(best_lr_params[prescription]['max_epochs']):
            # Training phase
            best_lr_model.train()
            total_loss = 0
            y_train_proba = []
            y_train_true = []

            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                y_pred_logits = best_lr_model(X_batch)

                loss = criterion(y_pred_logits, y_batch.unsqueeze(1))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                # 在这个循环内获取        y_train_proba 

                y_pred_proba = torch.sigmoid(y_pred_logits).detach().numpy()  # 用 Sigmoid 将 logits 转为概率
                y_train_proba.extend(y_pred_proba)
                y_train_true.extend(y_batch.numpy())
            
            avg_train_loss = total_loss / len(train_loader)
            train_loss_list.append(avg_train_loss)
            
            # Evaluation phase
            best_lr_model.eval()
            total_test_loss = 0

            y_test_proba = []
            y_test_true = []

            with torch.no_grad():
                for X_batch, y_batch in test_loader:
                    y_pred_logits = best_lr_model(X_batch)

                    loss = criterion(y_pred_logits, y_batch.unsqueeze(1)) 
                    total_test_loss += loss.item()

                    y_pred_proba = torch.sigmoid(y_pred_logits).detach().numpy()
                    y_test_proba.extend(y_pred_proba)
                    y_test_true.extend(y_batch.numpy())

            avg_test_loss = total_test_loss / len(test_loader)
            test_loss_list.append(avg_test_loss)

            # Calculate AUROC
            train_auroc = roc_auc_score(y_train_true, y_train_proba)
            test_auroc = roc_auc_score(y_test_true, y_test_proba)
            train_auroc_list.append(train_auroc)
            test_auroc_list.append(test_auroc)


        y_pred = best_lr_model(torch.tensor(X_test.values.astype(np.float32)))
        y_proba_lr = torch.sigmoid(y_pred).detach().numpy()  # Apply sigmoid to the tensor and convert to numpy


        # Optimize threshold for Logistic Regression
        best_threshold_lr, best_metrics_lr = optimize_thresholds(y_test, y_proba_lr, "Logistic Regression", prescription)

        fold_results_lr.append(best_metrics_lr)
        results_dict[prescription] = {'Logistic Regression': best_metrics_lr}
        # print('LR finished!')

        # xgboost 


        best_xgb_model = XGBClassifier(**best_xgb_params[prescription], use_label_encoder=False, tree_method='gpu_hist',  device = "cuda")

        best_xgb_model.fit(X_train, y_train)
        y_proba_xgb = best_xgb_model.predict_proba(X_test)[:, 1]

        best_threshold_xgb, best_metrics_xgb = optimize_thresholds(y_test, y_proba_xgb, "XGBoost", prescription)

        results_dict[prescription]['XGBoost'] = best_metrics_xgb
        # print('xgb finished!')

        # print(f"Finished training models for {prescription}.")
        fold_results_xgb.append(best_metrics_xgb)


    print(f"\nResults for {prescription}: Logistic Regression")
    for fold_idx, metrics in enumerate(fold_results_lr):
        print(f"Fold {fold_idx + 1}: {metrics}")

    print(f"\nResults for {prescription}: XGBoost")
    for fold_idx, metrics in enumerate(fold_results_xgb):
        print(f"Fold {fold_idx + 1}: {metrics}")

   

# # Print results for each prescription column and model
# for prescription in results_dict:
#     print(f"\"{prescription}\": {{")
#     for model in results_dict[prescription]:
#         metrics = results_dict[prescription][model]
#         print(f"\"{model}\": {metrics}")
#     print("}")

