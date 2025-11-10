import streamlit as st
import sys, os, traceback
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import __EMBEDDING_MODEL__, __EMBEDDING_COLLECTION_NAME__
# print(config.dspy_config)
from sentence_transformers import SentenceTransformer

try:
    # embed_model = SentenceTransformer(__EMBEDDING_MODEL__)
    # embed_model = SentenceTransformer("pritamdeka/S-BioBERT-snli-stsb")
    embed_model = SentenceTransformer("pritamdeka/S-BioBert-snli-multinli-stsb")
    dummy_input = {
            "lookup_id":1,
            "genetic_input_feature": {"gene":"BRCA1","variant":"c.68_69del"},
            "text": "BRCA1 INDEL mutation associated with breast cancer"
    }
    # text = "FUCA1 INDEL mutation associated with fucosidosis"
    # embed_vector = embed_model.encode(text)
    embed_vector = embed_model.encode(dummy_input["text"])
    # print(embed_vector)

    from qdrant_client import QdrantClient
    from qdrant_client.http.models import VectorParams, Distance, PointStruct

    vector_client = QdrantClient(url=st.secrets['QDRANT_HOST'], api_key=st.secrets['QDRANT_KEY'])
    vector_client.recreate_collection(collection_name=__EMBEDDING_COLLECTION_NAME__, vectors_config=VectorParams(size=768, distance=Distance.COSINE))
    points = PointStruct(id=dummy_input['lookup_id'], vector=embed_vector, payload=dummy_input['genetic_input_feature'])
    vector_client.upsert(collection_name=__EMBEDDING_COLLECTION_NAME__, points=[points])
except Exception as e:
    print(traceback.format_exc())