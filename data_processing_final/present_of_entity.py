import pandas as pd
from collections import Counter
import os
import ast
import sys
# Define file and directory paths
file = 'final_all_ner_additional_note.csv'
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
filtered_entities = {entity: count for entity, count in overall_entity_counts.items() if count > 1}

chunks = pd.read_csv(path, dtype={'all_ner': 'str', 'additional_note': 'str'}, chunksize=chunk_size)

# Process again to count occurrences of filtered entities and write results chunk by chunk
for i, chunk in enumerate(chunks):
    chunk['all_ner'] = chunk['all_ner'].map(convert_and_preprocess)
    chunk['entity_counts'] = chunk['all_ner'].map(lambda entities: [bool(entities.count(entity)) for entity in filtered_entities.keys()])
    
    # Append to CSV file
    output_file = os.path.join(dir, 'final_additional_note_all_ner_present.csv')
    if i == 0:
        chunk.to_csv(output_file, index=False, mode='w')  # Write header for the first chunk
    else:
        chunk.to_csv(output_file, index=False, mode='a', header=False)  # Append without header
    
    print('Save', i)
    sys.stdout.flush()
    
print("Processing complete. The length of the filtered entity dict is:", len(filtered_entities))
sys.stdout.flush()

# 将filtered_entities转换为DataFrame
filtered_entities_df = pd.DataFrame(list(filtered_entities.items()), columns=['Entity', 'Count'])

# 存储为CSV文件
filtered_entities_df.to_csv(os.path.join(dir, 'filtered_entities.csv'), index=False)

print("Filtered entities have been saved to CSV.")
