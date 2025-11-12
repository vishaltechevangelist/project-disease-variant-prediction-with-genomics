##** One time Script to populate the Qdrant for semantic search **##
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config, traceback
import pandas as pd
from classes.embedding_generator import embedding_generator
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import streamlit as st

import torch
if torch.backends.mps.is_available():
    DEVICE = 'mps' # Use Apple Silicon GPU (fastest option for your Mac)
    print("Using Apple Silicon (MPS) GPU for fastest encoding.")

try:
    df = pd.read_csv(config.__DISEASE_LOOKUP_FILE__, sep='\t', low_memory=False)
    ##** Data is large and have limited storage on Qdrant hence focus on pathogenic data only **##
    df = df[df['Clinical_Significance'].astype(str).str.contains('pathogenic', case=False, na=False)].copy()
    # print(df.info())
    # sys.exit(1)
    embedding_generator_obj = embedding_generator(SentenceTransformer, config.__EMBEDDING_MODEL__, DEVICE)
    client = QdrantClient(url=st.secrets['QDRANT_HOST'], api_key=st.secrets['QDRANT_KEY'])
    for idx, row in df.iterrows():
            text_for_embedding = f"Variant in gene {row['Gene_Symbol']} is classified as {row['Clinical_Significance']}. \
                Associated with disease: '{row['Clinical_Disease_Name']}'. Review status: {row['Clinical_Review_Status']}."
            vector_to_insert = embedding_generator_obj.get_embedding(text_for_embedding)
            payload = row.to_dict()
            client.upsert(collection_name=config.__EMBEDDING_COLLECTION_NAME__, 
                                points=[{
                                     "id": int(row['lookup_id']),
                                     "vector": vector_to_insert,
                                     "payload": payload
                                     }])
            print(idx)
            # if idx == 5:
            #      break
except Exception as e:
    print(traceback.format_exc())