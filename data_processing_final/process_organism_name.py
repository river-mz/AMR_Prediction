import pandas as pd
import os
import sys

# 定义文件路径
file = 'final_additional_note_all_ner_present_note_embeddings_2.csv'
dir = '/ibex/project/c2205/AMR_dataset_peijun/integrate/'
output_file = 'final_additional_note_all_ner_present_note_embeddings_2_organism_name.csv'
path = os.path.join(dir, file)

# 定义ESKAPE病原体并转换为单数小写形式
eskape_pathogens = {
    'Enterococcus faecium': 'Enterococcus_faecium',
    'Staphylococcus aureus': 'Staphylococcus_aureus',
    'Klebsiella pneumoniae': 'Klebsiella_pneumoniae',
    'Acinetobacter baumannii': 'Acinetobacter_baumannii',
    'Pseudomonas aeruginosa': 'Pseudomonas_aeruginosa',
    'Enterobacter': 'Enterobacter'
}

# 将键和值都转换为小写，方便后续匹配
eskape_pathogens = {key.lower(): value for key, value in eskape_pathogens.items()}

# 去除复数形式（例如，移除 "species"）
def normalize_name(organism_name):
    if pd.isna(organism_name):
        return organism_name
    organism_name = organism_name.lower()
    # 进一步处理逻辑可以根据需求添加
    return organism_name

# 对每一行的'organism_name'列进行分类
def classify_organism(row):
    organism = row['organism_name']
    
    if pd.isna(organism):  # 如果为空值
        row['organism_NA'] = 1
    else:
        organism_lower = normalize_name(organism)  # 进行单复数和术语归一化处理
        found = False
        
        # 遍历ESKAPE病原体，使用包含逻辑匹配
        for pathogen, col_name in eskape_pathogens.items():
            if pathogen in organism_lower:  # 部分匹配
                row[col_name] = 1
                found = True
                break  # 一旦匹配到就停止搜索

        if not found:  # 如果未匹配到ESKAPE病原体
            row['organism_other'] = 1

    return row

# 分块读取和处理
chunk_size = 5000  # 可根据内存情况调整
chunks = pd.read_csv(path, chunksize=chunk_size)

# 对每个块进行处理
for i, chunk in enumerate(chunks):
    # 创建新的ESKAPE病原体列，初始值为0
    for pathogen in eskape_pathogens.values():
        chunk[pathogen] = 0
    # 创建organism_other和organism_NA列
    chunk['organism_other'] = 0
    chunk['organism_NA'] = 0

    # 应用分类函数
    chunk = chunk.apply(classify_organism, axis=1)
    
    # 将结果保存到CSV
    output_path = os.path.join(dir, output_file)
    if i == 0:
        chunk.to_csv(output_path, index=False, mode='w')  # 第一次写入，包含header
    else:
        chunk.to_csv(output_path, index=False, mode='a', header=False)  # 追加写入，不包含header
    
    print(f"Processed chunk {i + 1}")
    sys.stdout.flush()

print("Processing complete.")
