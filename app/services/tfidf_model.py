# app/services/tfidf_model.py
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Set, Tuple
import logging
from pathlib import Path
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from annoy import AnnoyIndex
import joblib
from tqdm import tqdm
from fuzzywuzzy import fuzz, process

# Download necessary NLTK data
nltk.download('stopwords', quiet=True)

class TFIDFRecipeModel:
    def __init__(self, cache_dir: Path):
        """Initialize the TF-IDF recipe model"""
        self.logger = logging.getLogger(__name__)
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(exist_ok=True)
        
        # Cache file paths
        self.preprocessed_cache = self.cache_dir / 'preprocessed_data.joblib'
        self.tfidf_matrix_cache = self.cache_dir / 'tfidf_matrix.joblib'
        self.reduced_matrix_cache = self.cache_dir / 'reduced_matrix.joblib'
        self.recipe_info_cache = self.cache_dir / 'recipe_info.joblib'
        self.annoy_index_cache = self.cache_dir / 'recipe_annoy_index.ann'
        self.vectorizer_cache = self.cache_dir / 'vectorizer.joblib'
        self.svd_model_cache = self.cache_dir / 'svd_model.joblib'
        self.ingredient_index_cache = self.cache_dir / 'ingredient_index.joblib'


        # Model parameters
        self.max_features = 10000
        self.n_components = 300
        self.n_trees = 10

        self._initialize_model()

    def _initialize_model(self):
        """Initialize TF-IDF vectorizer, SVD, and Annoy index"""
        try:
            self.vectorizer = TfidfVectorizer(
                max_features=self.max_features,
                ngram_range=(1, 2)
            )
            self.svd = TruncatedSVD(
                n_components=self.n_components,
                random_state=42
            )
            self.annoy_index = None
            self.logger.info("TF-IDF model initialized successfully")
        except Exception as e:
            self.logger.error(f"Error initializing TF-IDF model: {str(e)}")
            raise

    def _preprocess_text(self, text: str, stem: bool = True) -> str:
        """Preprocess text with optional stemming"""
        try:
            text = str(text).lower()
            text = re.sub(r'[^a-zA-Z\s]', '', text)
            stop_words = set(stopwords.words('english'))
            text = ' '.join([word for word in text.split() if word not in stop_words])
            if stem:
                stemmer = PorterStemmer()
                text = ' '.join([stemmer.stem(word) for word in text.split()])
            return text
        except Exception as e:
            self.logger.error(f"Error preprocessing text: {str(e)}")
            raise

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate TF-IDF embeddings with dimensionality reduction"""
        try:
            # Preprocess texts
            self.logger.info("Preprocessing texts...")
            preprocessed_texts = [self._preprocess_text(text, stem=False) for text in tqdm(texts, desc="Preprocessing")]
            
            # Generate TF-IDF matrix
            self.logger.info("Generating TF-IDF matrix...")
            tfidf_matrix = self.vectorizer.fit_transform(tqdm(preprocessed_texts, desc="TF-IDF"))
            
            # Perform dimensionality reduction
            self.logger.info("Performing dimensionality reduction...")
            reduced_matrix = self.svd.fit_transform(tfidf_matrix)
            
            # Build Annoy index
            self.logger.info("Building Annoy index...")
            self.annoy_index = AnnoyIndex(self.n_components, 'angular')
            for i in tqdm(range(reduced_matrix.shape[0]), desc="Building index"):
                self.annoy_index.add_item(i, reduced_matrix[i])
            self.annoy_index.build(self.n_trees)
            
            # Save the TF-IDF matrix
            joblib.dump(tfidf_matrix, self.tfidf_matrix_cache)
            
            self.logger.info(f"Generated TF-IDF embeddings with shape: {reduced_matrix.shape}")
            return reduced_matrix
            
        except Exception as e:
            self.logger.error(f"Error generating TF-IDF embeddings: {str(e)}")
            raise

    def save_embeddings(self, reduced_matrix: np.ndarray) -> bool:
        """Save embeddings and models to cache"""
        try:
            joblib.dump(reduced_matrix, self.reduced_matrix_cache)
            joblib.dump(self.vectorizer, self.vectorizer_cache)
            joblib.dump(self.svd, self.svd_model_cache)
            if self.annoy_index:
                self.annoy_index.save(str(self.annoy_index_cache))
            return True
        except Exception as e:
            self.logger.error(f"Error saving embeddings: {str(e)}")
            return False

    def load_embeddings(self) -> Tuple[Optional[np.ndarray], bool]:
        """Load embeddings and models from cache"""
        try:
            if all(path.exists() for path in [
                self.reduced_matrix_cache,
                self.vectorizer_cache,
                self.svd_model_cache
            ]):
                reduced_matrix = joblib.load(self.reduced_matrix_cache)
                self.vectorizer = joblib.load(self.vectorizer_cache)
                self.svd = joblib.load(self.svd_model_cache)
                
                if self.annoy_index_cache.exists():
                    self.annoy_index = AnnoyIndex(self.n_components, 'angular')
                    self.annoy_index.load(str(self.annoy_index_cache))
                
                return reduced_matrix, True
                
            return None, False
            
        except Exception as e:
            self.logger.error(f"Error loading embeddings: {str(e)}")
            return None, False
        
    def build_ingredient_index(self, df: pd.DataFrame) -> Dict[str, Set[int]]:
        """Build ingredient to recipe index mapping"""
        try:
            if self.ingredient_index_cache.exists():
                self.logger.info("Loading ingredient index from cache...")
                self.ingredient_index = joblib.load(self.ingredient_index_cache)
                return self.ingredient_index

            self.logger.info("Building ingredient index...")
            self.ingredient_index = {}
            
            # Process each recipe
            for idx, row in tqdm(df.iterrows(), total=len(df), desc="Indexing ingredients"):
                try:
                    # Get ingredients as a list
                    ingredients = self.parse_list_field(row['RecipeIngredientParts'])
                    
                    # Process each ingredient
                    for ingredient in ingredients:
                        if isinstance(ingredient, str):
                            # Clean and normalize ingredient
                            ingredient = ingredient.lower().strip()
                            if ingredient:
                                # Add to index
                                if ingredient not in self.ingredient_index:
                                    self.ingredient_index[ingredient] = set()
                                self.ingredient_index[ingredient].add(idx)
                                
                                # Add partial matches for better fuzzy matching
                                words = ingredient.split()
                                if len(words) > 1:
                                    for word in words:
                                        if len(word) > 3:  # Only index meaningful words
                                            partial_key = f"partial_{word}"
                                            if partial_key not in self.ingredient_index:
                                                self.ingredient_index[partial_key] = set()
                                            self.ingredient_index[partial_key].add(idx)
                                
                except Exception as e:
                    self.logger.warning(f"Error processing ingredients for recipe {idx}: {str(e)}")
                    continue
            
            # Save to cache
            self.logger.info("Saving ingredient index to cache...")
            joblib.dump(self.ingredient_index, self.ingredient_index_cache)
            
            self.logger.info(f"Built ingredient index with {len(self.ingredient_index)} unique ingredients")
            return self.ingredient_index
            
        except Exception as e:
            self.logger.error(f"Error building ingredient index: {str(e)}")
            raise

    def get_query_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for a single query text"""
        try:
            preprocessed_text = self._preprocess_text(text, stem=False)
            query_tfidf = self.vectorizer.transform([preprocessed_text])
            query_embedding = self.svd.transform(query_tfidf)
            return query_embedding
        except Exception as e:
            self.logger.error(f"Error generating query embedding: {str(e)}")
            raise

    def find_similar_recipes(self, query_embedding: np.ndarray, recipe_embeddings: np.ndarray, 
                           top_n: int) -> Tuple[np.ndarray, np.ndarray]:
        """Find similar recipes using Annoy index with improved scoring"""
        try:
            if self.annoy_index is None:
                # Fallback to cosine similarity if Annoy index is not available
                similarities = cosine_similarity(query_embedding, recipe_embeddings)[0]
                top_indices = similarities.argsort()[-top_n:][::-1]
                top_similarities = similarities[top_indices]
            else:
                # Use Annoy index for faster similarity search with more candidates
                n_candidates = min(top_n * 20, recipe_embeddings.shape[0])
                top_indices, distances = self.annoy_index.get_nns_by_vector(
                    query_embedding[0],
                    n_candidates,
                    include_distances=True
                )
                
                # Convert distances to similarities and normalize
                similarities = 1 - np.array(distances) / 2
                
                # Sort by similarity and get top_n
                sorted_indices = np.argsort(similarities)[-top_n:][::-1]
                top_indices = np.array(top_indices)[sorted_indices]
                top_similarities = similarities[sorted_indices]

            return np.array(top_indices), top_similarities
            
        except Exception as e:
            self.logger.error(f"Error finding similar recipes: {str(e)}")
            raise

    def clear_cache(self):
        """Clear all cached files"""
        try:
            cache_files = [
                self.preprocessed_cache,
                self.tfidf_matrix_cache,
                self.reduced_matrix_cache,
                self.recipe_info_cache,
                self.annoy_index_cache,
                self.vectorizer_cache,
                self.svd_model_cache,
                self.ingredient_index_cache
            ]
            
            for cache_file in cache_files:
                if cache_file.exists():
                    cache_file.unlink()
            
            self.logger.info("Cache cleared successfully")
        except Exception as e:
            self.logger.error(f"Error clearing cache: {str(e)}")
            raise

    def get_model_info(self) -> Dict:
        """Get information about the model and its current state"""
        return {
            "model_type": "TF-IDF",
            "max_features": self.max_features,
            "n_components": self.n_components,
            "n_trees": self.n_trees,
            "cache_directory": str(self.cache_dir),
            "preprocessed_cache_exists": self.preprocessed_cache.exists(),
            "tfidf_matrix_cache_exists": self.tfidf_matrix_cache.exists(),
            "reduced_matrix_cache_exists": self.reduced_matrix_cache.exists(),
            "recipe_info_cache_exists": self.recipe_info_cache.exists(),
            "annoy_index_cache_exists": self.annoy_index_cache.exists(),
            "vectorizer_cache_exists": self.vectorizer_cache.exists(),
            "svd_model_cache_exists": self.svd_model_cache.exists(),
            "ingredient_index_cache_exists": self.ingredient_index_cache.exists()
        }