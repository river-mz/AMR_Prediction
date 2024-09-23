import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
import os
import sys

# CSV file path
file = 'final_additional_note_all_ner_present.csv'
dir = '/ibex/project/c2205/AMR_dataset_peijun/integrate/'
path = os.path.join(dir, file)

# Load ClinicalBERT model and tokenizer
model_name = "emilyalsentzer/Bio_ClinicalBERT"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

# Function to convert notes to embeddings
def get_note_embedding(note):
    if pd.isna(note):
        return None
    inputs = tokenizer(note, padding=True, max_length=512, truncation=True, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    return outputs.last_hidden_state[:, 0, :].squeeze().numpy()

# Process CSV in chunks
# chunk_size = 10000  # Adjust based on memory constraints
chunk_size = 5000  # Adjust based on memory constraints

output_file = os.path.join(dir, 'final_additional_note_all_ner_present_note_embeddings_2.csv')

# Initialize the CSV for writing embeddings
for i, chunk in enumerate(pd.read_csv(path, dtype={'additional_note': 'str'}, chunksize=chunk_size)):
    # Initialize a new column for embeddings
    if i <23:
        continue

    chunk['note_embedding'] = None
    
    # Generate embeddings for each note in the chunk
    chunk['note_embedding'] = chunk['additional_note'].apply(get_note_embedding)
    
    # Save the chunk with embeddings to CSV
    if i == 0:
        chunk.to_csv(output_file, index=False, mode='w')  # Write header for the first chunk
    else:
        chunk.to_csv(output_file, index=False, mode='a', header=False)  # Append without header
    
    print(f'Processed chunk {i + 1} and saved to CSV.')
    sys.stdout.flush()
# Print completion message
print("Processing complete. Embeddings saved to:", output_file)
