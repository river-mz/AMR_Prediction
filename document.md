# Document for RiAMR

RiAMR is a machine learning based robust and interpretable antimicrobial resistance prediction tool from Electronic Health Records (EHR). This document will record all the related codes during development.





## Data: 
The data used to train and test our models sources from 2 dataset. The first one is AMR-UTI dataset from US (https://physionet.org/content/antimicrobial-resistance-uti/1.0.0/) and another is collected from Dubai. All the data was stored under /ibex/project/c2205/AMR_dataset_peijun/ .

- US-UTI: this dataset includes 116,902 EHR and coresponding AMR records. The antibiotics include: nitrofurantoin, sulfamethoxazole, ciprofloxacin, levofloxacin. The dataset without preprocessing is stored under /ibex/project/c2205/AMR_dataset_peijun/US-UTI/original_dataset

- Dubai: this dataset includes 34,488 EHR and coresponding AMR records. The antibiotics include: nitrofurantoin,  ciprofloxacin, levofloxacin. The dataset without preprocessing is stored under /ibex/project/c2205/AMR_dataset_peijun/Dubai/ .

    - The AMR data are stored in /ibex/project/c2205/AMR_dataset_peijun/Dubai/microculture folder. In the folder, there are  MicroCultureData from 2017 to 2021. All the MicroCultureData were merged into a csv file: integrated_microculture.csv

    - The EHR data are stored in GeneralData.csv, AntibioticsOP.csv and AntibioticsIP.csv under /ibex/project/c2205/AMR_dataset_peijun/Dubai/ folder.


- Saudi: this dataset was not used for training and test yet due to lacking of some important features. This dataset consists of three pathogens, including PA, KP, MRSA.

    - PA dataset was stored in /ibex/project/c2205/AMR_dataset_peijun/Saudi/PA Metadat+workflowsheet JUL 10

    - MRSA dataset was stored in /ibex/project/c2205/AMR_dataset_peijun/Saudi/MRSA Data 2023

    - KP dataset was stored in /ibex/project/c2205/AMR_dataset_peijun/Saudi/KP metadata 6June

    

## Data Preprocessing:
To transform the EHR data into numerous features, a series of feature preprocessing steps were performed, including extracting common features between two datasets and merging them, standardizing organism_name and converting it to one-hot encoding, and extracting textual embeddings from all notes in the EHRs. All the data preprocessing code is stored in the AMR_Prediction/data_processing_final folder.

- AMR-UTI (US) dataset preprocessing: 

    1. Extract prior_antibiotic_resistance	prior_antibiotic_exposures	prior_infecting_organisms as text from micro - prev resistance [ANTIBIOTIC] 90, medication, ab subtype, ab class 90 - [ANTIBIOTIC] and micro - prev organism [PATHOGEN NAME] 90

    2. Extract department_type from Hospital department type (IP/OP/ER/ICU) (4 columns)

    3. Setting gender as female

    4. Change columns' name to adapt to the unified dataset.  The adapted df was stored in all_feature_extracted_finished.csv


    5. Converting 'comorbiditiy_180', 'prior_procedures',  'prior_nursing_facilities_180','uncomplicated' into text and merge into additional note. The full dataframe with additional note is stored in all_feature_extracted_finished_notes.csv


    6. Extract previous AMR and previous antibiotics exposure. Result was stored in all_uti_features_previous_AMR_and_previous_exposure_ner.csv 

    7. Merge the result with label: final_clean_uti_ner_ebd_6atb_previous_AMR_exposure_label.csv'



- Dubai dataset preprocessing: 
    1. merge general data with antibioticsIP and antibioticsOP based on PT_NO
    2. merge antibiotics with microculture, based on PT_NO as well as SPECIMEN_DATE_COLLECTED(microculture) and Date (antibiotics) (within 10day)
    3. merge speciman_request, diagnosis, problem_list, admission_diagnosis and clinical_diagnosis as additional_note
    4. conduct NER on additional note 
    5. merging records of same patients to get previous antibiotics explosure or resistance:
        - for every patient, obtain the most recent AST test record 

        - using the patient id and the most recent test date to find drug usage on 6 classes of antibiotcs (cephalosporin, carbapenem, fluoroquinolone, polymyxin, aminoglycoside, nitrofurantoin) in the last 90 day record in ip and op file 

        - based on the last 90 day drug usages, derive the previous antibiotics exposure

        - the result was saved in dubai_NER_6atb_merge_history_and_notes_version2.csv

- Saudi dataset preprocessing: 

    - link all the PA files into 1 files: 
        1. merged AST file (workflow) with "All PA sample" based on 'Sample ID' to get the date of collection feature.
        2. merged the result from step.1 with the diagnosis file based on date of collection(from workflow) and DGNS_REG_DT(from diagnosis), and the time window is 14 days.
        3. merged the result from step.2 with IP and OP file based on the PT_ID. To further filter the result, only retain records whose "Date of Collection" is 7 days near "MED_DT" in op
        4. found rows in demo have same PT_ID but different demo features, assuming those records are wrong and deleted them.
        5. merged the results from step.3 and step.4 (demo) based on PT_NO
        5. merged the results in step.3 with the demo in step.4 to get the final results.

    - the codes were stored under data/data_explore/saudi_pa_Oct_8_final_merge.py


- Dataset merging: comman features in US and dubai dataset and corresponding discriptions are recorded in AMR_Prediction/final_feature_description.xlsx

- Standardizing organism_name and converting into one-hot: there are various organism recorded in the dataset. In order to transform this feature into numerical vector, we only extracted eskape pathogens and created another column named. The codes were stored in AMR_Prediction/data_processing_final/process_organism_name.py

- Standardizing specimen_type and converting into one-hot: similar with organism_name, we extract the main specimen types, including urine, blood, respiratory, skin, stool, vaginal_swab, fluid, throat and others. The codes were stored in AMR_Prediction/data_processing_final/process_specimen_type.py


- Extracting textual embeddings from all notes in the EHRs：

    0. Name entity recognition (NER): used en_ner_bionlp13cg_md, en_ner_bionlp13cg_md, en_core_sci_lg model to conduct NER on additional notes column in order to extract all the medical entitys. The codes were stored in AMR_Prediction/data_processing_final/NER.py

    1. Present of entitys: filtered the entitys in step1 that appear more than 5 times in the merged dataset and formed a entity dictionary. For 'all_ner' in each record, calcuated the present of the entity in the dictionary. In oder to reduce the dimention, ran pca on the resulted vectors and reducted them into 20 dimension. The codes were stored under AMR_Prediction/data_processing_final/present_of_entity.py
        - note1: the pca kernel was saved in AMR_Prediction/data_processing_final/tool/trained_ipca_model_entity_present.joblib

        - note2: the dictionary was stored under AMR_Prediction/data_processing_final/pca_present_of_entity.py

    2. Note embedding: another way to convert the textual data is using large language model to transform the clinical note into numerical embedding. Here I used Bio_ClinicalBERT model, which can transform clinical note into vector. The codes for note embedding were stored in AMR_Prediction/data_processing_final/note_embedding.py. We further use pca to reduct the dimension into 20 and the corresponding codes were under AMR_Prediction/data_processing_final/pca_on_note_embedding.py
        - notes: the pca kernel was saved in AMR_Prediction/data_processing_final/tool/trained_ipca_model_note_embedding.joblib

- The preprocessed merged final dataset was stored under /ibex/project/c2205/AMR_dataset_peijun/integrate/final_note_embeddings_20_organism_name_specimen_type_all_ner_present20_2.csv


## Model Training
Used the merged dataset to rrain XGBoost and LR models. The codes were stored in AMR_Prediction/model_prediction_final/model_training.py and the model parameters are saved in AMR_Prediction/model_prediction_final/model_Ours_Oct_5

- Models trained on dubai:  only use dubai dataset to train the models. The codes were stored in AMR_Prediction/model_prediction_final/model_training_us_dubai.py and the trained models were in AMR_Prediction/model_prediction_final/model_Ours_Dubai_Oct_5. 
    - Notes: in dubai dataset, there is not model for sulfamethoxazole resistance prediction since the dataset lacks of corresponding sulfamethoxazole AST records.


- Models trained on us: only use AMR_UTI (us) dataset to train the models. The codes were stored in AMR_Prediction/model_prediction_final/model_training_us.py and the trained models were in AMR_Prediction/model_prediction_final/model_Ours_US_Oct_5




## Sensetivity Analysis: 

1. In order to explore the generalization ability of the trained models. We have tried to train models on one dataset and test on another.

    - Trained on Dubai test on Us: the codes were stored in AMR_Prediction/sensetivity_ana/load_Dubai_test_Us.py

    - Trained on Us and test on Dubai: the codes were stored in AMR_Prediction/sensetivity_ana/load_US_test_Dubai.py

2. We also try to explore the impact of diversity and size of training dataset on the models' performance. The implementation logic is to control the ratio of us and dubai datasets. The codes was stored in AMR_Prediction/sensetivity_ana/run_LR_XGBoost_with_ratio_error.py

3. To explore the impact of balanced and unbalanced dataset on the models' performance, we train and test the models before and after oversampling. 



4. To understand each modality's effect on the prediction, we conduct the ablation experiment on different modality. The codes and the results were stored under AMR_Prediction/sensetivity_ana/modality. The modalities include:
    - demography features: ['age', 'race', 'veteran', 'gender', 'BMI' ]
    - textual features: note_embedding and entity_present
    - clinical features: previous_antibiotic_exposure, previous_antibiotic_resistance, specimen type, hospital department


5. To explore how the NA-value and its filling strategy will influence the result, we try different ways to deal with NA value. The code was stored in AMR_Prediction/sensetivity_ana/NA_filling





## Result Interpretation:

- Entity present embedding interpretation: to interpret each dimension of entity present embedding, we reversed the pca calculation process and traced back to the top10 entities. The figures were stored under AMR_Prediction/result_interpret/entity_present_interpretation/top10_contribution_to_pca_entity_present. The codes were stored under AMR_Prediction/result_interpret/entity_present_interpretation/visualize_contribution_entity_present.py


- Shap running, features correlation calculating, parallel coordinate figure drawing: The codes were stored in AMR_Prediction/result_interpret/parallel_coordinate_fig/load_pytorch_LR_weight_XGBoost_shap_Ours_rank_plot_venn_pearson_cmp_dataset2.py. And the figures were stored under AMR_Prediction/result_interpret/shap/shap_Oct_30


## Streamlit Running
Deploy the trained models in a streamlit-based website for users to run our AMR tool

- The website was built in the AMR_Prediction/streamlit_running/streamlit_website.py file. 

- The models used was stored under AMR_Prediction/streamlit_running/save_models_for_CI

