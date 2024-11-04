import sys
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pickle
import seaborn as sns
import time  # Import time module to track execution time
from pandas.plotting import parallel_coordinates
from matplotlib_venn import venn2
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import parallel_coordinates

import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import parallel_coordinates
from scipy.stats import spearmanr

# 设置输出目录
output_dir = '/home/linp0a/AMR_prediction_pipeline/model_prediction/shap_Oct_30'
os.makedirs(output_dir, exist_ok=True)  # 创建目录

# 加载数据
use_cols = ['age', 'race', 'veteran', 'gender', 'BMI', 'previous_antibiotic_exposure_cephalosporin',
       'previous_antibiotic_exposure_carbapenem', 'previous_antibiotic_exposure_fluoroquinolone',
       'previous_antibiotic_exposure_polymyxin', 'previous_antibiotic_exposure_aminoglycoside',
       'previous_antibiotic_exposure_nitrofurantoin', 'previous_antibiotic_resistance_ciprofloxacin',
       'previous_antibiotic_resistance_levofloxacin', 'previous_antibiotic_resistance_nitrofurantoin',
       'previous_antibiotic_resistance_sulfamethoxazole', 'resistance_nitrofurantoin', 
       'resistance_sulfamethoxazole', 'resistance_ciprofloxacin', 'resistance_levofloxacin', 'source',
       'dept_ER', 'dept_ICU', 'dept_IP', 'dept_OP', 'dept_nan', 'Enterococcus_faecium', 
       'Staphylococcus_aureus', 'Klebsiella_pneumoniae', 'Acinetobacter_baumannii',
       'Pseudomonas_aeruginosa', 'Enterobacter', 'organism_other', 'organism_NA', 
       'note_embedding_pca_0', 'note_embedding_pca_1', 'note_embedding_pca_2', 
       'note_embedding_pca_3', 'note_embedding_pca_4', 'note_embedding_pca_5', 
       'note_embedding_pca_6', 'note_embedding_pca_7', 'note_embedding_pca_8', 
       'note_embedding_pca_9', 'note_embedding_pca_10', 'note_embedding_pca_11', 
       'note_embedding_pca_12', 'note_embedding_pca_13', 'note_embedding_pca_14', 
       'note_embedding_pca_15', 'note_embedding_pca_16', 'note_embedding_pca_17', 
       'note_embedding_pca_18', 'note_embedding_pca_19', 'entity_present_pca_0', 
       'entity_present_pca_1', 'entity_present_pca_2', 'entity_present_pca_3', 
       'entity_present_pca_4', 'entity_present_pca_5', 'entity_present_pca_6', 
       'entity_present_pca_7', 'entity_present_pca_8', 'entity_present_pca_9', 
       'entity_present_pca_10', 'entity_present_pca_11', 'entity_present_pca_12', 
       'entity_present_pca_13', 'entity_present_pca_14', 'entity_present_pca_15', 
       'entity_present_pca_16', 'entity_present_pca_17', 'entity_present_pca_18', 
       'entity_present_pca_19']

final_df = pd.read_csv('/ibex/project/c2205/AMR_dataset_peijun/integrate/final_note_embeddings_20_organism_name_specimen_type_all_ner_present20_2.csv', 
                        usecols=use_cols)
us_df = final_df[final_df['source'] == 'US']
country1_df = final_df[final_df['source'] == 'country1']

# 清理和转换 BMI
final_df['BMI'] = pd.to_numeric(final_df['BMI'].replace('12,073.88', np.nan), errors='coerce').fillna(-1)

# 定义处方列
prescription_columns = ['resistance_nitrofurantoin', 
                       'resistance_ciprofloxacin', 'resistance_levofloxacin']


# Step 2: Build a Logistic Regression model using PyTorch
class LogisticRegressionModule(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegressionModule, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        logits = self.linear(x)
        return logits  # 注意，这里不再应用 sigmoid

    
# 加载模型
model_dir_us = '/home/linp0a/AMR_prediction_pipeline/model_prediction/model_Ours_US_Oct_5'
lr_models_us = {prescription: torch.load(os.path.join(model_dir_us, f'lr_model_{prescription}.pth')) for prescription in prescription_columns}
xgb_models_us = {prescription: pickle.load(open(os.path.join(model_dir_us, f'xgb_model_{prescription}.pkl'), 'rb')) for prescription in prescription_columns}

model_dir_country1 = '/home/linp0a/AMR_prediction_pipeline/model_prediction/model_Ours_country1_Oct_5'
lr_models_country1 = {prescription: torch.load(os.path.join(model_dir_country1, f'lr_model_{prescription}.pth')) for prescription in prescription_columns}
xgb_models_country1 = {prescription: pickle.load(open(os.path.join(model_dir_country1, f'xgb_model_{prescription}.pkl'), 'rb')) for prescription in prescription_columns}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# 定义保存LR特征重要性和排名
def calculate_lr_feature_importance(prescription, data):
    if data == 'US': 
        lr_model = lr_models_us[prescription].to(device)
    else:
        lr_model = lr_models_country1[prescription].to(device)

    lr_model.eval()
    X = final_df.drop(columns=['source', 'resistance_nitrofurantoin', 'resistance_sulfamethoxazole', 'resistance_ciprofloxacin', 'resistance_levofloxacin'])

    # 提取逻辑回归的权重
    with torch.no_grad():
        feature_importance = lr_model.linear.weight.cpu().numpy().flatten()  # 提取模型的系数
        feature_names = X.columns  # 特征名称
    
    # 创建一个 DataFrame 并按重要性排序，取前20个最重要的特征
    importance_df_ori = pd.DataFrame({
        'Feature': feature_names,
        'Importance': np.abs(feature_importance)  # 使用绝对值作为重要性度量
    }).sort_values(by='Importance', ascending=False)

    
    
    # 为特征添加排名
    importance_df_ori['Rank'] = range(1, len(importance_df_ori) + 1)
    
    importance_df = importance_df_ori[:20]
    # 保存排名
    importance_df.to_csv(os.path.join(output_dir, f"lr_top20_{prescription}_{data}.csv"), index=False)
    
    # 打印和保存特征重要性图
    plt.figure(figsize=(8, 12))
    sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis', orient='h')
    plt.title(f"Top 20 Features in Logistic Regression for {prescription} in {data} Dataset", fontsize=16)
    plt.xlabel("Absolute Coefficient Value", fontsize=12)
    plt.ylabel("Feature", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"lr_coefficients_{prescription}_{data}.png"))
    plt.close()
    
    print(f'Feature importance for {prescription} in {data} calculated and saved.')
    return importance_df_ori  # 返回 DataFrame 以用于并行坐标图

# 保存XGBoost SHAP值和排名
def calculate_xgb_shap_values(prescription, data):
    valid_indices = final_df[prescription].notna()
    X = final_df.loc[valid_indices].drop(columns=['source', 'resistance_nitrofurantoin', 'resistance_sulfamethoxazole', 'resistance_ciprofloxacin', 'resistance_levofloxacin'])
    y_binary = final_df.loc[valid_indices, prescription]
    X.fillna(-1, inplace=True)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.2, random_state=42)

    if data == 'US': 
        xgb_model = xgb_models_us[prescription]
    else:
        xgb_model = xgb_models_country1[prescription]

    # XGBoost SHAP calculation
    explainer_xgb = shap.TreeExplainer(xgb_model)
    shap_values_xgb = explainer_xgb.shap_values(X_test)

    # 计算每个特征的平均绝对SHAP值
    shap_importance = np.abs(shap_values_xgb).mean(axis=0)
    
    # 创建一个 DataFrame 并按重要性排序，取前20个最重要的特征
    shap_df_ori = pd.DataFrame({
        'Feature': X_test.columns,
        'Importance': shap_importance
    }).sort_values(by='Importance', ascending=False)[:20]

    
    
    # 为特征添加排名
    shap_df_ori['Rank'] = range(1, len(shap_df_ori) + 1)
    
    shap_df = shap_df_ori

    # 保存排名
    shap_df.to_csv(os.path.join(output_dir, f"xgb_shap_top20_{prescription}_{data}.csv"), index=False)

    # 保存SHAP图
    plt.figure()
    shap.summary_plot(shap_values_xgb, X_test, plot_type="dot", feature_names=X.columns)
    plt.title(f"XGBoost SHAP on {prescription}")
    plt.savefig(os.path.join(output_dir, f"shap_xgb_{prescription}_{data}.png"))
    plt.close()

    print(f'XGBoost SHAP for {prescription} calculated and saved.')
    return shap_df_ori  # 返回 DataFrame 以用于并行坐标图


def plot_parallel_coordinates1(us_df, country1_df, prescription, correlation, p_value, model):
    # 合并两个DataFrame，根据Feature列匹配
    # 先对lr_df 和shap_df 进行升序排列
    us_df = us_df.sort_values(by='US Rank', ascending=True)
    country1_df = country1_df.sort_values(by='country1 Rank', ascending=True)
    
    merged_df = pd.merge(us_df[['Feature', 'US Rank']], country1_df[['Feature', 'country1 Rank']], on='Feature', how='inner')
    # merged_df = merged_df
    print(merged_df)
    # 为并行坐标图添加一列以区分Logistic Regression和XGBoost
    merged_df['Data'] = 'US vs country1'

    # 创建并行坐标图
    fig, ax = plt.subplots(figsize=(12, 8))

    # 绘制并行坐标图
    parallel_coordinates(merged_df, 'Data', cols=['US Rank', 'country1 Rank'], color=('#556270',), alpha=0.5, linestyle='--')

    # 美化图形
    plt.title(f"Parallel Coordinates Plot for {prescription} {model}", fontsize=16, pad=20)
    # plt.xlabel("Model", fontsize=12)
    
    # 设置y轴刻度为1到20并翻转
    ax.set_yticks(range(1, 21))
    ax.set_yticklabels(range(1, 21), fontsize=10)  # 反向显示刻度

    # 在左右两个y轴上分别标出每个模型对应的特征名
    us_features = us_df['Feature'].tolist()
    country1_features = country1_df['Feature'].tolist()

    # 在左侧（LR）添加特征名，位置向右偏移
    for i, feature in enumerate(us_features):
        ax.text(-0.15, i + 1, feature, horizontalalignment='right', fontsize=9, color='blue')

    # 在右侧（XGB）添加特征名，位置向左偏移
    for i, feature in enumerate(country1_features):
        ax.text(1.15, i + 1, feature, horizontalalignment='left', fontsize=9, color='green')

    # 限制y轴范围
    ax.set_ylim(1, 20)

    # 去掉图例
    ax.get_legend().remove()

    plt.text(0.7, -0.3, f'Spearman Correlation: {correlation:.4f}', 
             ha='center', va='top', fontsize=9, bbox=dict(facecolor='white', alpha=0.5))
    plt.text(0.15, -0.3, f'p-value: {p_value:.4f}', 
             ha='center', va='top', fontsize=9, bbox=dict(facecolor='white', alpha=0.5))

    # 调整布局
    plt.tight_layout()

    # 保存图像
    plt.savefig(os.path.join(output_dir, f"parallel_coordinates_{prescription}_{model}.png"))
    plt.close()

    print(f'Parallel Coordinates Plot for {model} {prescription} saved.')



def plot_venn_diagram(df_us, df_country1, prescription, model):
    # 提取特征集合
    us_features = set(df_us['Feature'])
    country1_features = set(df_country1['Feature'])

    # 创建Venn图
    plt.figure(figsize=(8, 6))
    venn = venn2([us_features, country1_features], ('US', 'country1'))

    # 获取Venn图的区域
    us_circle = venn.get_label_by_id('10')  # 仅在LR中的特征
    country1_circle = venn.get_label_by_id('01')  # 仅在XGBoost中的特征
    overlap_circle = venn.get_label_by_id('11')  # 两者共有的特征

    # 在每个区域中添加特征
    us_only = us_features - country1_features
    country1_only = country1_features - us_features
    overlap = us_features & country1_features

    # 标注LR特征
    if us_only:
        us_circle.set_text('\n'.join(us_only))
    if country1_only:
        country1_circle.set_text('\n'.join(country1_only))
    if overlap:
        overlap_circle.set_text('\n'.join(overlap))

    plt.title(f'Venn Diagram of Features for {model} {prescription}', fontsize=16)
    plt.savefig(os.path.join(output_dir, f'{model}_venn_diagram_{prescription} in US vs country1.png'))
    plt.close()
    print(f'Venn diagram for {prescription} saved.')

# to fix the column name
def calculate_spearman_correlation(us_df, country1_df, prescription):
    # 合并两个 DataFrame
    merged_df = pd.merge(us_df[['Feature', 'US Rank']], country1_df[['Feature', 'country1 Rank']], on='Feature', how='inner')

    # 计算Spearman相关性
    correlation, p_value = spearmanr(merged_df['US Rank'], merged_df['country1 Rank'])
    
    print(f'Spearman correlation for {prescription}: {correlation}, p-value: {p_value}')
    return correlation, p_value

# 运行主程序，处理每个处方
if __name__ == '__main__':
    for i, prescription in enumerate(prescription_columns):
        print(f"Processing {prescription} ({i + 1}/{len(prescription_columns)})...")

        # 计算Logistic Regression的特征重要性
        us_lr_df = calculate_lr_feature_importance(prescription, 'US').rename(columns = {'Rank': 'US Rank'})
        country1_lr_df = calculate_lr_feature_importance(prescription, 'country1').rename(columns = {'Rank': 'country1 Rank'})
        plot_venn_diagram(us_lr_df[:20], country1_lr_df[:20], prescription, 'LR')
        correlation, p_value = calculate_spearman_correlation(us_lr_df, country1_lr_df, prescription)
        plot_parallel_coordinates1(us_lr_df[:20], country1_lr_df[:20], prescription, correlation, p_value, 'LR')


        # 计算XGBoost的SHAP值
        us_xgb_df = calculate_xgb_shap_values(prescription, 'US').rename(columns = {'Rank': 'US Rank'})
        country1_xgb_df = calculate_xgb_shap_values(prescription, 'country1').rename(columns = {'Rank': 'country1 Rank'})
        plot_venn_diagram(us_xgb_df[:20], country1_xgb_df[:20], prescription, 'XGB')
        # 绘制并保存Parallel Coordinates Plot
        correlation, p_value = calculate_spearman_correlation(us_xgb_df, country1_xgb_df, prescription)
        plot_parallel_coordinates1(us_xgb_df[:20], country1_xgb_df[:20], prescription, correlation, p_value, 'XGB')
