import pandas as pd
from collections import Counter
import os
import ast
import sys
import numpy as np
from sklearn.decomposition import IncrementalPCA

# Define file and directory paths
file = '/ibex/project/c2205/AMR_dataset_peijun/integrate/final_additional_note_all_ner_present_note_embeddings_20_organism_name_specimen_type.csv'
dir = '/ibex/project/c2205/AMR_dataset_peijun/integrate/'
path = os.path.join(dir, file)



# Function to safely convert and preprocess the 'all_ner' field
def convert_and_preprocess(ner_str):
    try:
        entities = ast.literal_eval(ner_str)
        return list(set(entity.strip().rstrip(',') for entity in entities if entity.strip()))
    except (ValueError, SyntaxError):
        return []

# Initialize a counter for entities across all chunks
overall_entity_counts = Counter()

# Process CSV in chunks
chunk_size = 1000  # Adjust this based on memory constraints
chunks = pd.read_csv(path, dtype={'all_ner': 'str', 'additional_note': 'str'}, chunksize=chunk_size)

# Loop over each chunk
for i, chunk in enumerate(chunks):
    # Apply the conversion and preprocessing to the 'all_ner' column
    chunk['all_ner'] = chunk['all_ner'].map(convert_and_preprocess)
    
    # Explode lists into rows and count occurrences efficiently
    exploded_entities = chunk['all_ner'].explode()
    
    # Update the overall entity count with the current chunk
    overall_entity_counts.update(exploded_entities)
    
    # Output progress
    print(f"Processed chunk {i + 1}. Current unique entities count: {len(overall_entity_counts)}")
    sys.stdout.flush()

# Filter out entities that appear only once
filtered_entities = {entity: count for entity, count in overall_entity_counts.items() if count > 5}

# Initialize Incremental PCA for the final dimensionality reduction
n_components = 20
ipca = IncrementalPCA(n_components=n_components)

# Second processing of CSV in chunks, now with PCA
output_pca_file = os.path.join(dir, 'final_note_embeddings_20_organism_name_specimen_type_all_ner_present20_2.csv')

chunks = pd.read_csv(path, dtype={'all_ner': 'str', 'additional_note': 'str'}, chunksize=chunk_size)

# Fit the Incremental PCA on the entity counts
for i, chunk in enumerate(chunks):
    chunk['all_ner'] = chunk['all_ner'].map(convert_and_preprocess)
    
    # Create binary count columns for filtered entities
    entity_counts_matrix = np.array([[entities.count(entity) for entity in filtered_entities.keys()] for entities in chunk['all_ner']])
    
    # Fit Incremental PCA on the current chunk
    ipca.partial_fit(entity_counts_matrix)
    print(f"Fitted PCA on chunk {i + 1}")
    sys.stdout.flush()

# Second pass: Transform the entity counts with PCA and save the result
chunks = pd.read_csv(path, dtype={'all_ner': 'str', 'additional_note': 'str'}, chunksize=chunk_size)

for i, chunk in enumerate(chunks):
    chunk['all_ner'] = chunk['all_ner'].map(convert_and_preprocess)
    
    # Create binary count columns for filtered entities
    entity_counts_matrix = np.array([[entities.count(entity) for entity in filtered_entities.keys()] for entities in chunk['all_ner']])
    
    # Transform the entity counts with the fitted PCA
    X_reduced = ipca.transform(entity_counts_matrix)
    
    # Convert the PCA-transformed data into a DataFrame
    pca_df = pd.DataFrame(X_reduced, columns=[f'entity_present_pca_{j}' for j in range(n_components)])
    
    # Combine the reduced data with the original chunk (excluding 'entity_counts')
    original_columns = chunk.columns.tolist()
    original_columns.remove('all_ner')
    original_data = chunk[original_columns].reset_index(drop=True)
    
    final_chunk = pd.concat([original_data, pca_df], axis=1)
    
    final_chunk = final_chunk.drop(columns=['additional_note', 'embedding','specimen_type','organism_name',
                     'entity_counts', ])
    # Append the final chunk to the output CSV file
    final_chunk.to_csv(output_pca_file, mode='a', header=not os.path.exists(output_pca_file), index=False)
    
    print(f"Saved chunk {i + 1}")
    sys.stdout.flush()

print("PCA transformation complete. Results saved to:", output_pca_file)

# Save the filtered entities to a CSV
filtered_entities_df = pd.DataFrame(list(filtered_entities.items()), columns=['Entity', 'Count'])
filtered_entities_df.to_csv(os.path.join(dir, 'filtered_entities.csv'), index=False)

print("Filtered entities have been saved to CSV.")
