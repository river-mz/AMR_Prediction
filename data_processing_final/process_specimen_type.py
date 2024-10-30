import pandas as pd
from collections import Counter
import os
import sys

# 定义文件和目录路径
file = '/ibex/project/c2205/AMR_dataset_peijun/integrate/final_additional_note_all_ner_present_note_embeddings_20_organism_name.csv'
dir = '/ibex/project/c2205/AMR_dataset_peijun/integrate/'
path = os.path.join(dir, file)

# use_cols = ['specimen_type', 'source']

# 初始化Counter对象和处理后的输出文件

output_file = os.path.join(dir, 'final_additional_note_all_ner_present_note_embeddings_20_organism_name_specimen_type.csv')
# output_file = 'test_tempt2.csv'
# 设置chunk大小
chunk_size =5000  # 可根据系统内存大小调整

merge_mapping = {
    'specimen_type_urine': ['urine', 'mid stream urine', 'clean catch', 'catheterized (indwelling catheter) urine'],
    'specimen_type_blood': ['blood', 'blood peripheral vein'],
    'specimen_type_respiratory': ['respiratory', 'sputum'],
    'specimen_type_skin': ['wound', 'abscess', 'pus', 'eye', 'ear'],
    'specimen_type_stool': ['stool'],
    'specimen_type_vaginal_swab': ['high vaginal swab', 'low vaginal swab'],
    'specimen_type_biopsy': ['biopsy'],
    'specimen_type_fluid': ['fluid'],
    'specimen_type_throat': ['throat']
}


# 第一次读取数据，用于统计specimen_type
# chunks = pd.read_csv(path, chunksize=chunk_size, nrows = 300)



# for chunk in chunks:
#     for specimen in chunk['specimen_type'].dropna():
#         specimen_types = [s.strip() for s in specimen.split(';')]
#         counter.update(specimen_types)

# 获取出现频率最高的19个类型
# top_19 = [item[0] for item in counter.most_common(19)]

# 定义处理函数，用于分块处理并分类specimen_type
def classify_specimen_type(chunk):
    
    # 创建新的specimen_type列并初始化为0
    for specimen_type in merge_mapping.keys():
        chunk[specimen_type] = 0
    
    # 创建specimen_type_other和specimen_type_na列并初始化为0
    chunk['specimen_type_other'] = 0
    chunk['specimen_type_na'] = 0

    if chunk['source'].iloc[0] == 'US' and len(set(chunk['source'])) == 1:
        chunk['specimen_type_na'] = 1
        return chunk

    # 对每一行的specimen_type进行分类
    for i, row in chunk.iterrows():
        specimen = row['specimen_type']
        
        if pd.isna(specimen):  # 如果为空值
            chunk.at[i, 'specimen_type_na'] = 1
        else:
            specimen_types = specimen.lower()  # 转为小写
            found = False
            
            # 遍历top_19的类型
            for top_type, subtypes in merge_mapping.items():

                top_type = top_type.lower()
                for subtype in subtypes:
                    if subtype in specimen_types:  # 如果匹配到了top_19中的任意一个类型
                        chunk.at[i, top_type] = 1
                        found = True
            
            if not found:  # 如果没有匹配到任何top_19类型
                chunk.at[i, 'specimen_type_other'] = 1
    
    return chunk

# 分块处理和逐步写入CSV
chunks = pd.read_csv(path, chunksize=chunk_size)
for i, chunk in enumerate(chunks):

    chunk = classify_specimen_type(chunk)
    
    # 第一个块写入时包括表头，后续的块则不包括表头
    if i == 0:
        chunk.to_csv(output_file, index=False, mode='w')
    else:
        chunk.to_csv(output_file, index=False, mode='a', header=False)

    print("finished chunk", i)
    sys.stdout.flush()


print("Processing complete. Output saved to:", output_file)
