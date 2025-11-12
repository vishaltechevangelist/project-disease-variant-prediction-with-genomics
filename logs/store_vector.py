import config
import pandas as pd
import traceback
import os
import sys
import numpy as np
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
import streamlit as st
from qdrant_client.http.models import VectorParams, Distance, PointStruct, Batch

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

# --- CONFIGURATION ---
FILE_PATH = config.__DISEASE_LOOKUP_FILE__
QDRANT_HOST = st.secrets['QDRANT_HOST']
QDRANT_PORT = 6333
COLLECTION_NAME = config.__EMBEDDING_COLLECTION_NAME__
EMBEDDING_MODEL_NAME = config.__EMBEDDING_MODEL__
CHECKPOINT_FILE = '/tmp/checkpoint.log'
BATCH_SIZE = 50  # INCREASED: Larger batches reduce I/O overhead
VECTOR_DIMENSION = 768 # S-BioBert dimension
FINAL_COLLECTION_NAME = config.__EMBEDDING_COLLECTION_NAME__
VECTOR_SIZE = 768
DISTANCE_METRIC = Distance.COSINE 

# --- CHECKPOINTING FUNCTIONS ---

def get_last_indexed_id() -> int:
    """Reads the last successfully indexed lookup_id from the checkpoint file."""
    print(CHECKPOINT_FILE)
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, 'r') as f:
                content = f.read().strip()
                if content and content.isdigit():
                    return int(content)
                elif content:
                    print(f"Warning: Checkpoint file contains non-numeric content: '{content}'. Starting from ID 0.")
        except Exception as e:
            print(f"Warning: Could not read checkpoint file. Starting from ID 0. Error: {e}")
    return 0

def save_checkpoint_id(lookup_id: int):
    """Writes the last successfully indexed lookup_id to the checkpoint file."""
    try:
        with open(CHECKPOINT_FILE, 'w') as f:
            f.write(str(lookup_id))
    except Exception as e:
        print(f"Error: Failed to write checkpoint ID {lookup_id}. Manual resume might be needed.")
        print(traceback.format_exc())


# --- QDRANT & EMBEDDING FUNCTIONS ---

def create_vector_text(row: pd.Series) -> str:
    """
    Creates the comprehensive text string that will be converted into an embedding.
    """
    disease_name = str(row['Clinical_Disease_Name']).replace('_', ' ').replace('|', ' and ')
    review_status = str(row['Clinical_Review_Status']).replace('_', ' ')
    variant_type = 'INDEL' if (row['IS_INDEL'] == 1 or row['IS_INDEL'] == True) else 'SNP'
    
    return f"""
    ClinVar record for gene {row['Gene_Symbol']} ({variant_type} at position {row['Chromosome']}) 
    is classified as {row['Clinical_Significance']}. 
    The associated condition is '{disease_name}'. 
    The mutation is a {row['Missense_Variant']} and the evidence review status is '{review_status}'.
    """

def process_and_upload_batch(client: QdrantClient, model: SentenceTransformer, batch: pd.DataFrame):
    """
    Generates embeddings and payload for a batch of data and uploads to Qdrant,
    then updates the checkpoint.
    """
    if batch.empty:
        return

    # 1. Prepare texts for embedding
    texts = batch.apply(create_vector_text, axis=1).tolist()
    
    # 2. Generate embeddings
    print(f"Generating embeddings for batch of size {len(batch)} on {DEVICE}...")
    
    # Use the model's device
    vectors = model.encode(
        texts, 
        convert_to_numpy=True, 
        show_progress_bar=False,
        # Setting device is sometimes necessary even if initialized with it
        device=DEVICE 
    )
    
    # 3. Prepare payload and points
    points = []
    for i, row in batch.iterrows():
        point_id = int(row['lookup_id'])
        payload = row.to_dict()
        
        # Qdrant Point Structure
        points.append(
            models.PointStruct(
                id=point_id,
                vector=vectors[points.__len__()], 
                payload=payload
            )
        )

    # 4. Upload to Qdrant
    print(f"Uploading batch of {len(points)} points to Qdrant...")
    operation_info = client.upsert(
        collection_name=COLLECTION_NAME,
        wait=True, 
        points=points,
    )
    # print(operation_info)
    # sys.exit(1)
    # 5. Checkpoint: Only save ID if upload was successful
    if operation_info.status.name == 'COMPLETED':
        last_id_in_batch = batch['lookup_id'].max()
        save_checkpoint_id(last_id_in_batch)
        print(f"Upload completed. Last indexed ID saved: {last_id_in_batch}. Status: {operation_info.status.name}")
    else:
        print(f"Warning: Qdrant upload failed for batch. Status: {operation_info.status.name}. Checkpoint NOT updated. Retrying this batch is required.")


def index_data():
    """Main function to orchestrate the indexing and resumption process."""
    print(f"Starting Qdrant Indexing for file: {FILE_PATH}")
    print(f"Using Embedding Model: {EMBEDDING_MODEL_NAME}")
    print(f"Encoding will use device: {DEVICE}")
    print(f"Qdrant Batch Size: {BATCH_SIZE} points per upload")
    
    try:
        # 1. Load data
        print("Loading data file...")
        # Note: If your system is low on RAM, consider using chunking here.
        df = pd.read_csv(FILE_PATH, sep='\t', low_memory=False)
        df = df[df['Clinical_Significance'].astype(str).str.contains('pathogenic', case=False, na=False)].copy()
        df.columns = df.columns.str.replace(' ', '_') 
        df['lookup_id'] = df['lookup_id'].astype(int)
        
        # 2. Checkpoint Logic: Determine where to resume
        last_indexed_id = get_last_indexed_id()
        # print(last_indexed_id)
        # sys.exit(1)
        
        if last_indexed_id > 0:
            print(f"\n--- RESUMING INDEXING ---")
            df_to_process = df[df['lookup_id'] > last_indexed_id].copy()
            total_rows_original = len(df)
            remaining_rows = len(df_to_process)
            
            if remaining_rows == 0:
                print("Checkpoint indicates all data is already indexed. Exiting.")
                return
                
            print(f"Total rows in file: {total_rows_original}. Rows remaining to process: {remaining_rows}.")
        else:
            print("\n--- STARTING NEW INDEX ---")
            df_to_process = df.copy()
            total_rows = len(df_to_process)
            print(f"Total rows to process: {total_rows}.")

    except Exception as e:
        print(f"\n--- ERROR LOADING DATA ---")
        print(f"Could not load data from {FILE_PATH}. Check file path, separator, and encoding.")
        print(traceback.format_exc())
        return

    try:
        # 3. Initialize Qdrant and Model
        client = QdrantClient(url=st.secrets['QDRANT_HOST'], api_key=st.secrets['QDRANT_KEY'])
        print(f"Connected to Qdrant at {QDRANT_HOST}")

        # Load model and explicitly pass the device
        print(f"Loading Sentence Transformer model {EMBEDDING_MODEL_NAME}...")
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

        # # 4. Create collection if it doesn't exist
        # print(f"Checking/Creating collection '{COLLECTION_NAME}'...")
        # client.recreate_collection(
        #     collection_name=COLLECTION_NAME,
        #     vectors_config=models.VectorParams(size=VECTOR_DIMENSION, distance=models.Distance.COSINE),
        # )
        # print(f"Collection '{COLLECTION_NAME}' ready.")

        # 5. Process and upload data in batches
        num_batches = (len(df_to_process) + BATCH_SIZE - 1) // BATCH_SIZE

        from tqdm import tqdm
        
        for i in tqdm(range(num_batches), desc="Indexing Batches"):
            start_index = i * BATCH_SIZE
            end_index = min((i + 1) * BATCH_SIZE, len(df_to_process))
            
            batch = df_to_process.iloc[start_index:end_index] 
            
            # The actual work happens here
            process_and_upload_batch(client, model, batch)
            
        print("\n*** INDEXING COMPLETE ***")
        final_count = client.get_collection(COLLECTION_NAME).points_count
        print(f"Total points indexed: {final_count}")

    except Exception as e:
        print(f"\n--- CRITICAL ERROR DURING QDRANT/MODEL OPERATION ---")
        print(f"Error occurred. Checkpoint file '{CHECKPOINT_FILE}' contains the last successfully indexed ID.")
        print(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    index_data()