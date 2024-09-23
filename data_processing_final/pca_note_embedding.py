import pandas as pd
import numpy as np
from sklearn.decomposition import IncrementalPCA
import os
import sys

# 文件路径
file = 'final_additional_note_all_ner_present_note_embeddings_2_organism_name.csv'
dir = '/ibex/project/c2205/AMR_dataset_peijun/integrate/'
path = os.path.join(dir, file)

# 初始化Incremental PCA
n_components = 20
chunk_size = 5000  # 可根据内存情况调整
ipca = IncrementalPCA(n_components=n_components)

# 输出文件路径
output_pca_file = os.path.join(dir, 'final_additional_note_all_ner_present_note_embeddings_20_organism_name.csv')

# 分块处理CSV文件以拟合IPCA
for chunk in pd.read_csv(path, chunksize=chunk_size):
    # 处理note_embedding
    chunk['note_embedding'] = chunk['note_embedding'].apply(lambda x: np.fromstring(x.strip('[]'), sep=' '))

    # Fit Incremental PCA on the chunk
    ipca.partial_fit(np.stack(chunk['note_embedding'].values))

print('PCA finished')
sys.stdout.flush()
# 第二次分块处理以获取PCA降维表示并立即存储
for i, chunk in enumerate(pd.read_csv(path, chunksize=chunk_size)):
    chunk['note_embedding'] = chunk['note_embedding'].apply(lambda x: np.fromstring(x.strip('[]'), sep=' '))
    embeddings_matrix = np.stack(chunk['note_embedding'].values)
    
    # Transform the chunk with the fitted IPCA
    X_reduced = ipca.transform(embeddings_matrix)
    
    # 将降维结果转换为DataFrame
    pca_df = pd.DataFrame(X_reduced, columns=[f'note_embedding_pca_{i}' for i in range(n_components)])
    
    # 读取原始数据的其他列
    original_columns = pd.read_csv(path, nrows=0).columns.tolist()
    original_columns.remove('note_embedding')
    original_data = chunk[original_columns].reset_index(drop=True)
    
    # 合并原始数据与PCA结果
    final_chunk = pd.concat([original_data, pca_df], axis=1)
    
    # 追加保存到CSV文件
    final_chunk.to_csv(output_pca_file, mode='a', header=not os.path.exists(output_pca_file), index=False)
    print("finished "+ str(i) + " chunk!")
    sys.stdout.flush()

print(f"PCA reduced embeddings saved to {output_pca_file}")
