import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

# 读取之前生成的top10_pca_entity_contributions.csv文件
top_contributions_file = 'top10_pca_entity_contributions.csv'
top_contributions_df = pd.read_csv(top_contributions_file)

# 定义保存图像的文件夹
output_folder = 'top10_contribution_to_pca_entity_present'
os.makedirs(output_folder, exist_ok=True)  # 如果文件夹不存在则创建

# 设置seaborn样式
sns.set(style='whitegrid')

# 循环处理每个PCA component的top10贡献
for i in range(20):  # 因为有20个PCA维度
    # 过滤当前component的数据
    component_name = f'entity_present_pca_{i}'
    component_data = top_contributions_df[top_contributions_df['Component'] == component_name]
    
    # 创建绘图
    plt.figure(figsize=(8, 6))
    sns.barplot(x='Entity', y='Contribution', data=component_data, palette='Blues_d')
    
    # 设置图的标题和标签
    plt.title(f'Top 10 Entity Contributions to {component_name}', fontsize=16)
    plt.xlabel('Entity', fontsize=14)
    plt.ylabel('Contribution', fontsize=14)
    plt.xticks(rotation=45, ha='right')  # X轴标签倾斜，避免重叠
    
    # 保存图像到指定的文件夹
    output_path = os.path.join(output_folder, f'{component_name}_top10_contributions.png')
    plt.tight_layout()  # 自动调整子图间的间距
    plt.savefig(output_path, dpi=300)
    plt.close()  # 关闭图表，释放内存
    
    # 输出提示
    print(f'Saved plot for {component_name} to {output_path}')

print("All plots saved successfully.")
