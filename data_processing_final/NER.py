import spacy
import nltk
import sys
import scispacy
import pandas as pd
import numpy as np
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords 
import re
from sklearn.decomposition import PCA
from tqdm import tqdm

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('wordnet')

def name_extractor(text):
    tokens = nltk.word_tokenize(text)
    tagged = nltk.pos_tag(tokens)
    output = []
    for i in range(len(tagged)):
        if tagged[i][1] in ['NN', 'NNP', 'NNS', 'NNPS']:
            output.append(tagged[i][0])
    output_sentence = ' '.join(output)
    return output_sentence

def processor(sentence, minimum_length=3):
    stop_words = set(stopwords.words('english')) 
    lemmatizer = WordNetLemmatizer()
    sentence_lower = sentence.lower()
    tmp = [re.sub('[^A-Za-z]+', '', i) for i in word_tokenize(sentence_lower)]
    sentence_lower_nospecialchar = [i for i in tmp if len(i) > 0]
    sentence_lower_nospecialchar_nostopword = [w for w in sentence_lower_nospecialchar if w not in stop_words] 
    sentence_lower_nospecialchar_nostopword_lemmatized = [lemmatizer.lemmatize(i, pos='n') for i in sentence_lower_nospecialchar_nostopword] 
    sentence_lower_nospecialchar_nostopword_lemmatized_deduplicated = list(set(sentence_lower_nospecialchar_nostopword_lemmatized))
    sentence_lower_nospecialchar_nostopword_lemmatized_deduplicated_fileteredshorts = [i for i in sentence_lower_nospecialchar_nostopword_lemmatized_deduplicated if len(i) > minimum_length]
    output_sentence = ' '.join(sentence_lower_nospecialchar_nostopword_lemmatized_deduplicated_fileteredshorts)
    return output_sentence

# 加载 NER 模型
nlp_sci_lg = spacy.load("en_core_sci_lg")
nlp_bc5cdr = spacy.load("en_ner_bc5cdr_md")
nlp_bionlp13cg = spacy.load("en_ner_bionlp13cg_md")


path = 'country1_NER_5atb_merge_history_and_notes.csv'

output_path = '/ibex/project/c2205/AMR_dataset_peijun/Saudi/MRSA/anti_pt_meta_cmb_notes_ner_ebd.csv'

input_df = pd.read_csv(path, header=0)

# # 对 'additional_note' 列进行去重操作
# input_df_unique = input_df.drop_duplicates(subset=['additional_note'])

# 创建一个字典来存储每个唯一note的NER结果
ner_results = {
    'output_terms_nlp_sci_lg': [],
    'output_terms_nlp_bc5cdr': [],
    'output_terms_nlp_bionlp13cg': []
}

# 处理去重后的数据并显示进度
total_notes = len(input_df)
for i, nt in enumerate(input_df['additional_note']):
    if isinstance(nt, str):
        input_sentence = processor(name_extractor(nt))
        
        # 处理 en_core_sci_lg 模型
        doc_sci_lg = nlp_sci_lg(input_sentence)
        ner_results['output_terms_nlp_sci_lg'].append([ent.text for ent in doc_sci_lg.ents])
        
        # 处理 en_ner_bc5cdr_md 模型
        doc_bc5cdr = nlp_bc5cdr(input_sentence)
        ner_results['output_terms_nlp_bc5cdr'].append([ent.text for ent in doc_bc5cdr.ents])
        
        # 处理 en_ner_bionlp13cg_md 模型
        doc_bionlp13cg = nlp_bionlp13cg(input_sentence)
        ner_results['output_terms_nlp_bionlp13cg'].append([ent.text for ent in doc_bionlp13cg.ents])
    else:
        ner_results['output_terms_nlp_sci_lg'].append(np.nan)
        ner_results['output_terms_nlp_bc5cdr'].append(np.nan)
        ner_results['output_terms_nlp_bionlp13cg'].append(np.nan)
    
    # 打印进度
    print(f"Progress: {100 * (i + 1) / total_notes:.2f}%")
    sys.stdout.flush()

# 将NER结果添加回去重后的数据框
input_df['output_terms_nlp_sci_lg'] = ner_results['output_terms_nlp_sci_lg']
input_df['output_terms_nlp_bc5cdr'] = ner_results['output_terms_nlp_bc5cdr']
input_df['output_terms_nlp_bionlp13cg'] = ner_results['output_terms_nlp_bionlp13cg']


# 保存最终结果
input_df.to_csv(output_path, index=0)

print("Merging completed and saved to:", output_path)
sys.stdout.flush()

