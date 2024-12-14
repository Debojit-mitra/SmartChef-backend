# app/services/bert_model.py

import torch
from transformers import BertTokenizer, BertModel
import numpy as np
from typing import List, Dict, Optional, Tuple
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import logging
import pickle
from pathlib import Path
from datetime import datetime
import json
from tqdm import tqdm
from functools import lru_cache
import os

class BERTRecipeModel:
    def __init__(self, cache_dir: Path):
        """Initialize the BERT recipe model"""
        self.logger = logging.getLogger(__name__)
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(exist_ok=True)
        
        # Cache paths
        self.embeddings_cache = self.cache_dir / "embeddings.pkl"
        self.data_cache = self.cache_dir / "processed_data.pkl"
        self.metadata_cache = self.cache_dir / "metadata.json"
        
        # Initialize device and model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"Using device: {self.device}")
        
        self._initialize_model()

    def _initialize_model(self):
        """Initialize BERT model and tokenizer"""
        try:
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.model = BertModel.from_pretrained('bert-base-uncased')
            self.model.to(self.device)
            self.model.eval()
            self.logger.info("BERT model initialized successfully")
        except Exception as e:
            self.logger.error(f"Error initializing BERT model: {str(e)}")
            raise

    def _load_cache_metadata(self) -> Dict:
        """Load cache metadata from file"""
        try:
            if self.metadata_cache.exists():
                with open(self.metadata_cache, 'r') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            self.logger.error(f"Error loading cache metadata: {str(e)}")
            return {}

    def _update_cache_metadata(self, cache_path: Path):
        """Update cache metadata with current timestamp"""
        try:
            metadata = self._load_cache_metadata()
            metadata[str(cache_path)] = datetime.now().timestamp()
            with open(self.metadata_cache, 'w') as f:
                json.dump(metadata, f)
        except Exception as e:
            self.logger.error(f"Error updating cache metadata: {str(e)}")

    # def _should_refresh_cache(self, cache_path: Path) -> bool:
    #     """Check if cache should be refreshed"""
    #     try:
    #         if not cache_path.exists():
    #             return True
            
    #         cache_metadata = self._load_cache_metadata()
    #         cache_mtime = cache_metadata.get(str(cache_path), 0)
            
    #         # Check if cache is older than 24 hours
    #         current_time = datetime.now().timestamp()
    #         cache_age = current_time - cache_mtime
            
    #         return cache_age > 86400  # 24 hours in seconds
    #     except Exception as e:
    #         self.logger.error(f"Error checking cache freshness: {str(e)}")
    #         return True

    @staticmethod
    def _parse_r_vector(text: str) -> List[str]:
        """Parse R-style vector strings into Python lists"""
        if pd.isna(text):
            return []
        if isinstance(text, list):
            return text
        if isinstance(text, str) and text.startswith('c(') and text.endswith(')'):
            text = text[2:-1]
            items = [item.strip().strip('"').strip("'") for item in text.split(',')]
            return [item for item in items if item]
        return []

    def save_embeddings(self, embeddings: np.ndarray, recipe_data: Dict = None):
        """Save embeddings and associated data to cache"""
        try:
            with open(self.embeddings_cache, 'wb') as f:
                pickle.dump(embeddings, f)
            
            if recipe_data is not None:
                with open(self.data_cache, 'wb') as f:
                    pickle.dump(recipe_data, f)
            
            self._update_cache_metadata(self.embeddings_cache)
            if recipe_data is not None:
                self._update_cache_metadata(self.data_cache)
                
            self.logger.info("Saved embeddings and data to cache")
        except Exception as e:
            self.logger.error(f"Error saving embeddings: {str(e)}")
            raise

    def load_embeddings(self) -> Optional[np.ndarray]:
        """Load embeddings from cache"""
        try:
            if self.embeddings_cache.exists():
                with open(self.embeddings_cache, 'rb') as f:
                    embeddings = pickle.load(f)
                self.logger.info("Loaded embeddings from cache")
                return embeddings
            return None
        except Exception as e:
            self.logger.error(f"Error loading embeddings: {str(e)}")
            return None

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate BERT embeddings for a list of texts"""
        try:
            batch_size = 32
            embeddings = []
            
            with torch.no_grad():
                for i in tqdm(range(0, len(texts), batch_size), desc="Generating BERT embeddings"):
                    batch_texts = texts[i:i+batch_size]
                    inputs = self.tokenizer(
                        batch_texts,
                        padding=True,
                        truncation=True,
                        max_length=512,
                        return_tensors='pt'
                    )
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    outputs = self.model(**inputs)
                    batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
                    embeddings.extend(batch_embeddings)
            
            embeddings = np.vstack(embeddings)
            self.logger.info(f"Generated embeddings with shape: {embeddings.shape}")
            return embeddings
            
        except Exception as e:
            self.logger.error(f"Error generating embeddings: {str(e)}")
            raise

    @lru_cache(maxsize=1024)
    def get_query_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for a single query text"""
        try:
            with torch.no_grad():
                inputs = self.tokenizer(
                    text,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors='pt'
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self.model(**inputs)
                return outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        except Exception as e:
            self.logger.error(f"Error generating query embedding: {str(e)}")
            raise

    def find_similar_recipes(self, query_embedding: np.ndarray, recipe_embeddings: np.ndarray, 
                           top_n: int) -> Tuple[np.ndarray, np.ndarray]:
        """Find similar recipes using cosine similarity"""
        try:
            similarities = cosine_similarity(query_embedding, recipe_embeddings)[0]
            top_indices = similarities.argsort()[-top_n:][::-1]
            top_similarities = similarities[top_indices]
            
            return top_indices, top_similarities
        except Exception as e:
            self.logger.error(f"Error finding similar recipes: {str(e)}")
            raise

    def clear_cache(self):
        """Clear all caches"""
        try:
            cache_files = [
                self.embeddings_cache,
                self.data_cache,
                self.metadata_cache
            ]
            
            for cache_file in cache_files:
                if cache_file.exists():
                    os.remove(cache_file)
                    self.logger.info(f"Cleared cache: {cache_file}")
            
            # Clear function cache
            self.get_query_embedding.cache_clear()
            
            self.logger.info("Cache clearing completed")
        except Exception as e:
            self.logger.error(f"Error clearing cache: {str(e)}")
            raise

    def get_model_info(self) -> Dict:
        """Get information about the model and its current state"""
        return {
            "model_type": "BERT",
            "model_name": "bert-base-uncased",
            "device": str(self.device),
            "cache_directory": str(self.cache_dir),
            "embeddings_cache_exists": self.embeddings_cache.exists(),
            "data_cache_exists": self.data_cache.exists(),
            "metadata_cache_exists": self.metadata_cache.exists()
        }