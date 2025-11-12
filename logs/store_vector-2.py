import sys
import os
import time
import pandas as pd
import numpy as np
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import VectorParams, Distance, PointStruct, Batch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import config
from helpers.helper import df_to_json, format_output, combine_disease_data_for_llm
import streamlit as st


# --- DEVICE SELECTION FOR APPLE SILICON (M1/M2/M3) ---
try:
    import torch
    if torch.backends.mps.is_available():
        DEVICE = 'mps' # Use Apple Silicon GPU (fastest option for your Mac)
        print("Using Apple Silicon (MPS) GPU for fastest encoding.")
    elif torch.cuda.is_available():
        DEVICE = 'cuda' # For non-Mac/NVIDIA GPUs
        print("Using CUDA GPU for encoding.")
    else:
        DEVICE = 'cpu' # Fallback
        print("Using CPU for encoding (consider installing PyTorch with MPS/CUDA support).")
except ImportError:
    DEVICE = 'cpu'
    print("Warning: PyTorch not imported. Encoding will run on CPU.")

# --- Configuration (using config.py) ---
QDRANT_HOST = st.secrets['QDRANT_HOST']
# QDRANT_PORT = config.__QDRANT_PORT__
FINAL_COLLECTION_NAME = config.__EMBEDDING_COLLECTION_NAME__
EMBEDDING_MODEL_NAME = config.__EMBEDDING_MODEL__
CHECKPOINT_FILE = '/tmp/check'
DATA_FILE_PATH = config.__DISEASE_LOOKUP_FILE__

# Fixed vector size for the chosen model
VECTOR_SIZE = 768 
DISTANCE_METRIC = Distance.COSINE 
BATCH_SIZE = 50

# --- Helper Functions ---

def load_checkpoint():
    return 0
    """Reads the last indexed lookup_id from the checkpoint file."""
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r') as f:
            try:
                return int(f.read().strip())
            except ValueError:
                return 0
    return 0

def save_checkpoint(last_id):
    """Writes the last indexed lookup_id to the checkpoint file."""
    with open(CHECKPOINT_FILE, 'w') as f:
        f.write(str(last_id))

def get_data(clin_sig_arg):
    """
    Loads and filters the main ClinVar data based on Clinical Significance,
    and converts necessary columns for embedding.
    """
    print(f"Loading data from {DATA_FILE_PATH}...")
    try:
        # Load the entire dataset
        df = pd.read_csv(DATA_FILE_PATH, sep='\t', low_memory=False)
    except FileNotFoundError:
        print(f"Error: Data file not found at {DATA_FILE_PATH}")
        return None

    # Filter by the command-line argument for Clinical Significance
    # We filter before checkpointing to ensure we only process the relevant subset
    df_filtered = df[
        df['Clinical_Significance'].astype(str).str.lower().str.contains(clin_sig_arg, na=False)
    ].copy()

    # --- Feature Engineering for Embedding (Context Building) ---
    
    # 1. CREATE MUTATION_TYPE (New addition based on your question)
    # Assumes an 'IS_INDEL' column exists where 1=INDEL and 0=SNP
    df_filtered['Mutation_Type'] = df_filtered['IS_INDEL'].apply(
        lambda x: "Insertion/Deletion (INDEL)" if x == 1 else "Single Nucleotide Polymorphism (SNP)"
    )

    # 2. Standardize Disease Names (using the same logic as disease_name_lookup.py)
    df_filtered['Clinical_Disease_Name'] = df_filtered['Clinical_Disease_Name'].fillna('Not Provided').str.replace('|', ' | ', regex=False)

    # 3. Map Encoded Features back to Human-Readable Labels for Context
    # Note: Requires loading the maps from config.py
    chromo_id_label_dd = config.json_maps.get('chromo_id_label_dd', {})
    
    # Use the maps to create human-readable strings
    df_filtered['Chromosome_Label'] = df_filtered['Chromosome_Encoded'].astype(str).map(chromo_id_label_dd).fillna(df_filtered['Gene_Symbol'])
    
    # Add other encoded maps here if available (e.g., Gene_Symbol_Label)

    # 4. Create the final text context for the RAG embedding
    df_filtered['text_context'] = (
        "Variant Type: " + df_filtered['Mutation_Type'] + 
        " | Gene: " + df_filtered['Gene_Symbol'] + 
        " | Chromosome: " + df_filtered['Chromosome_Label'] + 
        " | Clinical Significance: " + df_filtered['Clinical_Significance'] + 
        " | Associated Diseases: " + df_filtered['Clinical_Disease_Name'] +
        " | Review Status: " + df_filtered['Clinical_Review_Status'].fillna('N/A')
    )
    
    return df_filtered


def run_indexing():
    if len(sys.argv) < 2:
        print("Usage: python index_qdrant.py <clinical_significance_substring>")
        print("Example: python index_qdrant.py pathogenic")
        sys.exit(1)

    # 1. Get the clinical significance argument (for filtering/labeling data)
    clin_sig_arg = sys.argv[1].lower().strip()
    
    print(f"--- Starting indexing for: {clin_sig_arg} into collection: {FINAL_COLLECTION_NAME} ---")

    try:
        # 2. Setup Client and Model
        # client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        client = QdrantClient(url=st.secrets['QDRANT_HOST'], api_key=st.secrets['QDRANT_KEY'])
        model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=DEVICE)
        
        # 3. ENSURE COLLECTION EXISTS NON-DESTRUCTIVELY (The core fix for appending)
        # We check if the collection exists first. If it does, we skip creation (and destruction).
        collections = client.get_collections().collections
        
        if FINAL_COLLECTION_NAME not in [c.name for c in collections]:
            print(f"Collection '{FINAL_COLLECTION_NAME}' not found. Creating it now...")
            client.create_collection(
                collection_name=FINAL_COLLECTION_NAME,
                vectors_config=VectorParams(size=VECTOR_SIZE, distance=DISTANCE_METRIC),
                # Add index configuration (e.g., payload index for filtering) if needed
                optimizers_config=models.OptimizersConfigDiff(indexing_threshold=20000)
            )
            print("Collection created.")
        else:
             print(f"Collection '{FINAL_COLLECTION_NAME}' already exists. Appending new data.")

        # 4. Load Data and Checkpoint
        df = get_data(clin_sig_arg)
        if df is None or df.empty:
            print(f"No relevant data found for '{clin_sig_arg}'. Exiting.")
            return

        last_indexed_id = load_checkpoint()
        
        # The lookup_id column must be used for tracking progress across all runs
        df_to_index = df[df['lookup_id'] > last_indexed_id].sort_values('lookup_id').reset_index(drop=True)
        
        if df_to_index.empty:
            print(f"Checkpoint reached: All {clin_sig_arg} data points (up to ID {last_indexed_id}) have been indexed.")
            return

        print(f"Indexing {len(df_to_index)} new points, starting after lookup_id {last_indexed_id}...")

        # 5. Batch Indexing Loop
        for i in tqdm(range(0, len(df_to_index), BATCH_SIZE), desc="Indexing Batches"):
            batch = df_to_index.iloc[i:i + BATCH_SIZE]
            
            # 5a. Generate Embeddings for the text_context
            texts = batch['text_context'].tolist()
            vectors = model.encode(texts, convert_to_numpy=True, device=DEVICE ).astype(np.float32)

            # 5b. Prepare Payload (metadata)
            # Use original ClinVar columns for payload, converted to standard Python types
            payloads = batch.drop(columns=['text_context']).apply(lambda row: row.to_dict(), axis=1).tolist()
            
            # 5c. Prepare Points
            points = []
            for j, row in batch.iterrows():
                point_id = int(row['lookup_id'])
                vector = vectors[j - i]
                payload = payloads[j - i]
                
                points.append(PointStruct(
                    id=point_id,
                    vector=vector.tolist(),
                    payload=payload
                ))

            # 6. UPSERT DATA (The appending operation)
            client.upsert(
                collection_name=FINAL_COLLECTION_NAME,
                points=points,
                wait=True
            )
            
            # 7. Update Checkpoint
            if not batch.empty:
                new_last_id = batch['lookup_id'].max()
                save_checkpoint(new_last_id)
        
        print(f"\n--- Indexing complete for {clin_sig_arg}. Collection size: {client.count(FINAL_COLLECTION_NAME, exact=True).count} ---")
        
    except Exception as e:
        print(f"An error occurred during indexing for {clin_sig_arg}: {e}")
        # Log the exception for debugging
        # config.logging.exception(f"Indexing failed for {clin_sig_arg}")

if __name__ == "__main__":
    run_indexing()