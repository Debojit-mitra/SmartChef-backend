import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging
from pathlib import Path
import re
import nltk
from nltk.corpus import stopwords
import ast
from nltk.stem import PorterStemmer
import torch
import pickle
from datetime import datetime
from app.services.bert_model import BERTRecipeModel
from app.services.tfidf_model import TFIDFRecipeModel
from app.models.schemas import RecipeResponse, NutritionalInfo
from app.services.image_service import RecipeImageService
from app.services.recipeStepsEnhancer import EnhancedRecipeParser
from app.core.config import settings
from tqdm import tqdm
from fuzzywuzzy import fuzz, process

# Download necessary NLTK data
nltk.download('stopwords', quiet=True)

class ModelManager:
    def __init__(self):
        """Initialize the Recipe Recommendation System"""
        self.logger = logging.getLogger(__name__)
        self.bert_model = BERTRecipeModel(settings.BERT_CACHE_DIR)
        self.tfidf_model = TFIDFRecipeModel(settings.TFIDF_CACHE_DIR)
        self.image_service = RecipeImageService(settings.CACHE_DIR / "images")  # Initialize image service
        self.load_data_and_models()
        self.instruction_parser = EnhancedRecipeParser()
        self.logger.info("Model manager initialized successfully")

    def load_data_and_models(self):
        """Load recipe data and initialize/load models from cache if available, otherwise from source"""
        try:
            # Check if cache files exist
            bert_cache_exists = (
                (self.bert_model.data_cache).exists() and 
                (self.bert_model.embeddings_cache).exists()
            )

            tfidf_cache_exists = all(path.exists() for path in [
                self.tfidf_model.reduced_matrix_cache,
                self.tfidf_model.vectorizer_cache,
                self.tfidf_model.svd_model_cache,
                self.tfidf_model.ingredient_index_cache  # Add check for ingredient index cache

            ])

            if bert_cache_exists and tfidf_cache_exists:
                # Load from cache
                try:
                    with open(self.bert_model.data_cache, 'rb') as f:
                        cache_data = pickle.load(f)
                        self.df = cache_data['df']
                        self.recipe_index = cache_data['recipe_index']
                        self.ingredient_index = cache_data['ingredient_index']
                    
                    # Load BERT embeddings from cache
                    bert_embeddings = self.bert_model.load_embeddings()
                    # Load TF-IDF embeddings
                    tfidf_embeddings, tfidf_success = self.tfidf_model.load_embeddings()

                    if bert_embeddings is not None and tfidf_success:
                        self.bert_embeddings = bert_embeddings
                        self.tfidf_embeddings = tfidf_embeddings
                        self.logger.info("Successfully loaded all model data from cache.")
                        return
                    else:
                        self.logger.warning("Cache invalid or incomplete. Falling back to full data load.")
                except Exception as e:
                    self.logger.warning(f"Error loading from cache: {str(e)}. Falling back to full data load.")
            
            # If cache doesn't exist or is invalid, load from source
            self.df = pd.read_csv(settings.RECIPE_DATA_FILE)
            self.logger.info(f"Loaded {len(self.df)} recipes from dataset")
            
            # Create name and ingredient indices
            self._build_indices()
            
            # Prepare text data for models
            self.df['combined_text'] = self.df.apply(self._combine_recipe_text, axis=1)
            
            # Generate new BERT embeddings
            self.logger.info("Generating new BERT embeddings...")
            bert_embeddings = self.bert_model.generate_embeddings(self.df['combined_text'].tolist())
            self.bert_embeddings = bert_embeddings

            # Generate TF-IDF embeddings
            self.logger.info("Generating TF-IDF embeddings...")
            tfidf_embeddings = self.tfidf_model.generate_embeddings(self.df['combined_text'].tolist())
            self.tfidf_embeddings = tfidf_embeddings

            # Build TF-IDF ingredient index
            self.logger.info("Building TF-IDF ingredient index...")
            self.tfidf_model.build_ingredient_index(self.df)
            
            # Save to cache
            try:
                cache_data = {
                    'df': self.df,
                    'recipe_index': self.recipe_index,
                    'ingredient_index': self.ingredient_index
                }
                
                # Ensure cache directory exists
                Path(settings.DATA_CACHE_FILE).parent.mkdir(parents=True, exist_ok=True)
                
                with open(settings.DATA_CACHE_FILE, 'wb') as f:
                    pickle.dump(cache_data, f)
                
                self.bert_model.save_embeddings(bert_embeddings)
                self.tfidf_model.save_embeddings(tfidf_embeddings)

                self.logger.info("Successfully saved all model data to cache")
                
            except Exception as e:
                self.logger.warning(f"Error saving to cache: {str(e)}")
                
        except Exception as e:
            self.logger.error(f"Error in data and model loading: {str(e)}")
            raise

    def _build_indices(self):
        """Build search indices for faster lookup"""
        try:
            # Recipe name index
            self.recipe_index = {
                name.lower(): idx for idx, name in enumerate(self.df['Name'])
            }
            
            # Ingredient index
            self.ingredient_index = {}
            for idx, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Building indices"):
                ingredients = self._parse_list_field(row['RecipeIngredientParts'])
                for ingredient in ingredients:
                    ingredient = ingredient.lower().strip()
                    if ingredient:
                        if ingredient not in self.ingredient_index:
                            self.ingredient_index[ingredient] = set()
                        self.ingredient_index[ingredient].add(idx)
                        
            self.logger.info(f"Built indices: {len(self.recipe_index)} recipes, {len(self.ingredient_index)} ingredients")
        except Exception as e:
            self.logger.error(f"Error building indices: {str(e)}")
            raise

    def _combine_recipe_text(self, row: pd.Series) -> str:
        """Combine relevant recipe text fields for embedding"""
        text_parts = [
            str(row['Name']),
            str(row.get('RecipeCategory', '')),
            ' '.join(self._parse_list_field(row['RecipeIngredientParts'])),
            ' '.join(self._parse_list_field(row['RecipeInstructions']))
        ]
        return ' '.join(filter(None, text_parts))

    def _parse_list_field(self, field_value: str) -> List[str]:
        """Parse string representation of list fields with improved handling"""
        if pd.isna(field_value):
            return []
        if isinstance(field_value, list):
            return field_value
        try:
            # clean recipe steps
            if any(instruction_marker in str(field_value).lower() 
                  for instruction_marker in ['step', 'preheat', 'mix', 'cook', 'bake']):
                parsed_instructions = self.instruction_parser.parse_instructions(field_value)
                return [instr['text'] for instr in parsed_instructions]
            # Handle R-style vectors
            if isinstance(field_value, str):
                # Check if it's a c() format string
                if field_value.startswith('c(') and field_value.endswith(')'):
                    field_value = field_value[2:-1]
                    # Split by comma and clean up each item
                    items = [item.strip().strip('"\'') for item in field_value.split(',')]
                    return [item for item in items if item]
                    
                # Check if it's a bracketed list string
                elif field_value.startswith('[') and field_value.endswith(']'):
                    # Try to evaluate as a literal list
                    try:
                        items = eval(field_value)
                        if isinstance(items, list):
                            return [str(item).strip() for item in items if item]
                    except:
                        # If eval fails, strip brackets and split
                        field_value = field_value[1:-1]
                        items = [item.strip().strip('"\'') for item in field_value.split(',')]
                        return [item for item in items if item]
                        
                # Handle single quoted strings
                elif (field_value.startswith('"') and field_value.endswith('"')) or \
                    (field_value.startswith("'") and field_value.endswith("'")):
                    return [field_value.strip('"\'').strip()]
                    
                # Handle period-separated instructions
                elif '. ' in field_value:
                    instructions = []
                    current = ""
                    for char in field_value:
                        current += char
                        if char == '.' and len(current.strip()) > 0:
                            # Check if period is part of a number
                            if not (current.strip()[-2:].replace('.', '').isdigit()):
                                instructions.append(current.strip())
                                current = ""
                    if current.strip():
                        instructions.append(current.strip())
                    return instructions
                    
                # Try to evaluate string or split by comma if simple format
                try:
                    evaluated = eval(field_value)
                    if isinstance(evaluated, (list, tuple)):
                        return [str(item).strip() for item in evaluated if item]
                except:
                    if ',' in field_value:
                        items = [item.strip().strip('"\'') for item in field_value.split(',')]
                        return [item for item in items if item]
                    return [field_value.strip()]
                    
            # If all else fails, return as single item if non-empty
            if str(field_value).strip():
                return [str(field_value).strip()]
            return []
            
        except Exception as e:
            self.logger.warning(f"Error parsing list field: {str(e)}")
            return []

    def _format_recipe_responses(self, recipes_data: List[Tuple[int, float]]) -> RecipeResponse:
        """Format recipe data into RecipeResponse model"""
        try:
            recipe_names = []
            recipes = []
            for idx, score in recipes_data:
                recipe = self.df.iloc[idx]
                recipes.append((recipe, score))
                recipe_names.append(str(recipe.get('Name', '')))

            image_urls = self.image_service.get_recipe_images(recipe_names)

            
            responses = []
            for(recipe, score), recipe_name in zip(recipes, recipe_names):
                def safe_get(key, default=None):
                    try:
                        value = recipe.get(key, default)
                        if isinstance(value, pd.Series):
                            if len(value) == 1:
                                value = value.iloc[0]
                            else:
                                value = value.iloc[0]
                        if isinstance(value, np.ndarray):
                            if value.size == 1:
                                value = value.item()
                            else:
                                value = value[0]
                        if pd.isna(value) or value is pd.NA:
                            return default
                        return value
                    except Exception as e:
                        self.logger.warning(f"Error getting value for {key}: {str(e)}")
                        return default

                # Direct DataFrame access for list fields
                ingredient_parts = []
                ingredient_quantities = []
                structured_instructions = []
                equipment_needed = []

                try:
                    # Access RecipeIngredientParts
                    if 'RecipeIngredientParts' in recipe.index:
                        parts = recipe['RecipeIngredientParts']
                        if isinstance(parts, str):
                            # Try to evaluate string representation of list
                            try:
                                if parts.startswith('c('):
                                    parts = parts[2:-1]
                                ingredient_parts = eval(parts)
                            except:
                                ingredient_parts = [x.strip() for x in parts.split(',') if x.strip()]
                        elif isinstance(parts, (list, np.ndarray)):
                            ingredient_parts = parts.tolist() if isinstance(parts, np.ndarray) else parts

                    # Access RecipeIngredientQuantities
                    if 'RecipeIngredientQuantities' in recipe.index:
                        quantities = recipe['RecipeIngredientQuantities']
                        if isinstance(quantities, str):
                            try:
                                if quantities.startswith('c('):
                                    quantities = quantities[2:-1]
                                ingredient_quantities = eval(quantities)
                            except:
                                ingredient_quantities = [x.strip() for x in quantities.split(',') if x.strip()]
                        elif isinstance(quantities, (list, np.ndarray)):
                            ingredient_quantities = quantities.tolist() if isinstance(quantities, np.ndarray) else quantities

                    # Access RecipeInstructions
                    if 'RecipeInstructions' in recipe.index:
                        instrs = recipe['RecipeInstructions']
                        if isinstance(instrs, str):
                            try:
                                if instrs.startswith('c('):
                                    instrs = instrs[2:-1]
                                raw_instructions  = eval(instrs)
                            except:
                                raw_instructions  = [x.strip() for x in instrs.split(',') if x.strip()]
                        elif isinstance(instrs, (list, np.ndarray)):
                            raw_instructions  = instrs.tolist() if isinstance(instrs, np.ndarray) else instrs

                        # Use enhanced parser for instructions
                        structured_instructions = self.instruction_parser.parse_instructions(raw_instructions)
                        equipment_needed = self.instruction_parser.get_equipment_needed(structured_instructions)

                    # Convert all items to strings and remove empty ones
                    ingredient_parts = [str(x).strip() for x in ingredient_parts if x is not None and str(x).strip()]
                    ingredient_quantities = [str(x).strip() for x in ingredient_quantities if x is not None and str(x).strip()]

                except Exception as e:
                    self.logger.error(f"Error processing recipe lists: {str(e)}")

                # Create NutritionalInfo with explicit type conversion
                nutritional_info = NutritionalInfo(
                    calories=float(safe_get('Calories')) if safe_get('Calories') is not None else None,
                    fat_content=float(safe_get('FatContent')) if safe_get('FatContent') is not None else None,
                    saturated_fat_content=float(safe_get('SaturatedFatContent')) if safe_get('SaturatedFatContent') is not None else None,
                    cholesterol_content=float(safe_get('CholesterolContent')) if safe_get('CholesterolContent') is not None else None,
                    sodium_content=float(safe_get('SodiumContent')) if safe_get('SodiumContent') is not None else None,
                    carbohydrate_content=float(safe_get('CarbohydrateContent')) if safe_get('CarbohydrateContent') is not None else None,
                    fiber_content=float(safe_get('FiberContent')) if safe_get('FiberContent') is not None else None,
                    sugar_content=float(safe_get('SugarContent')) if safe_get('SugarContent') is not None else None,
                    protein_content=float(safe_get('ProteinContent')) if safe_get('ProteinContent') is not None else None
                )

                
                image_url = image_urls.get(recipe_name)
                

                # Create RecipeResponse with explicit type handling
                response = RecipeResponse(
                    recipe_id=str(safe_get('RecipeId')),
                    name=str(safe_get('Name', '')),
                    cook_time=self._parse_time(safe_get('CookTime')),
                    prep_time=self._parse_time(safe_get('PrepTime')),
                    total_time=self._parse_time(safe_get('TotalTime')),
                    image_url=image_url,  #str(safe_get('Images')) if safe_get('Images') is not None else None,
                    recipe_category=str(safe_get('RecipeCategory')) if safe_get('RecipeCategory') is not None else None,
                    ingredient_quantities=ingredient_quantities,
                    ingredient_parts=ingredient_parts,
                    aggregated_rating=float(safe_get('AggregatedRating')) if safe_get('AggregatedRating') is not None else None,
                    rating_count=float(safe_get('ReviewCount')) if safe_get('ReviewCount') is not None else None,
                    nutritional_info=nutritional_info,
                    recipe_servings=int(safe_get('RecipeServings')) if safe_get('RecipeServings') is not None else None,
                    recipe_instructions=structured_instructions,
                    equipment_needed=equipment_needed,
                    similarity_score=float(score)
                )
                responses.append(response)
                
            
            return responses

        except Exception as e:
            self.logger.error(f"Error formatting recipe response: {str(e)}")
            raise

    def _parse_time(self, time_str: Optional[str]) -> Optional[str]:
        """Parse ISO duration format to human readable format"""
        try:
            if time_str is None or pd.isna(time_str):
                return None
                
            time_str = str(time_str).upper()
            if not time_str.startswith('PT'):
                return time_str
                
            time_str = time_str[2:]  # Remove PT
            hours = 0
            minutes = 0
            
            if 'H' in time_str:
                hours = int(time_str.split('H')[0])
                time_str = time_str.split('H')[1]
            if 'M' in time_str:
                minutes = int(time_str.split('M')[0])
                
            if hours and minutes:
                return f"{hours}h {minutes}m"
            elif hours:
                return f"{hours}h"
            elif minutes:
                return f"{minutes}m"
            return None
        except Exception as e:
            self.logger.warning(f"Error parsing time {time_str}: {str(e)}")
            return None

    def get_recommendations_by_ingredients(self, ingredients: List[str], model_type: str, top_n: int) -> List[RecipeResponse]:
        """Get recommendations based on ingredient list using specified model"""
        try:
            if model_type.lower() == "bert":
                return self._get_bert_recommendations_by_ingredients(ingredients, top_n)
            elif model_type.lower() == "tfidf":
                return self._get_tfidf_recommendations_by_ingredients(ingredients, top_n)
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
                
        except Exception as e:
            self.logger.error(f"Error in ingredient-based recommendation: {str(e)}")
            raise
    
    def _get_bert_recommendations_by_ingredients(self, ingredients: List[str], top_n: int) -> List[RecipeResponse]:
        """Get recommendations based on ingredient list using BERT model"""
        try:
            # Normalize ingredients
            ingredients_lower = [ing.lower().strip() for ing in ingredients]
            
            # Find matching recipes using fuzzy matching
            matching_recipes = set()
            matched_ingredients_map = {}  # Map original ingredients to their matches
            
            # Get all unique ingredients from the index
            all_ingredients = set(self.ingredient_index.keys())
            
            # Fuzzy match ingredients and collect matches
            for ingredient in ingredients_lower:
                # Get fuzzy matches for each ingredient
                matches = process.extractBests(
                    ingredient,
                    all_ingredients,
                    scorer=fuzz.ratio,
                    score_cutoff=80,
                    limit=3
                )
                
                # Store matches for this original ingredient
                matched_ingredients_map[ingredient] = {matched_ing for matched_ing, _ in matches}
                
                # Process each match
                for matched_ing, score in matches:
                    # Safely get recipe indices for this ingredient
                    recipe_indices = self.ingredient_index.get(matched_ing, set())
                    if isinstance(recipe_indices, (np.ndarray, pd.Series)):
                        recipe_indices = set(recipe_indices.tolist())
                    matching_recipes.update(recipe_indices)
            
            # Score recipes based on ingredient matches
            recipe_scores = []
            for recipe_idx in matching_recipes:
                try:
                    # Safely get recipe ingredients
                    recipe_row = self.df.iloc[recipe_idx]
                    recipe_ingredients_raw = recipe_row['RecipeIngredientParts']
                    
                    # Handle various possible formats of recipe ingredients
                    if isinstance(recipe_ingredients_raw, str):
                        recipe_ingredients = set(ing.lower().strip() for ing in self._parse_list_field(recipe_ingredients_raw))
                    elif isinstance(recipe_ingredients_raw, (list, np.ndarray)):
                        recipe_ingredients = set(str(ing).lower().strip() for ing in recipe_ingredients_raw)
                    else:
                        recipe_ingredients = set()
                    
                    # Calculate match score
                    matched_count = 0
                    for original_ingredient in ingredients_lower:
                        # If any of the matches for this original ingredient are in the recipe
                        if recipe_ingredients.intersection(matched_ingredients_map[original_ingredient]):
                            matched_count += 1
                    
                    # Score is now guaranteed to be between 0 and 1
                    score = matched_count / len(ingredients_lower) if ingredients_lower else 0
                    
                    recipe_scores.append((int(recipe_idx), float(score)))
                    
                except Exception as e:
                    self.logger.warning(f"Error processing recipe {recipe_idx}: {str(e)}")
                    continue
            
            # Sort and get top matches
            recipe_scores.sort(key=lambda x: x[1], reverse=True)
            top_recipes = recipe_scores[:top_n]
            
            return self._format_recipe_responses(top_recipes)
            
        except Exception as e:
            self.logger.error(f"Error in BERT ingredient-based recommendation: {str(e)}")
            raise

    def _get_tfidf_recommendations_by_ingredients(self, ingredients: List[str], top_n: int) -> List[RecipeResponse]:
        """Get recommendations using improved TF-IDF model based on ingredients"""
        try:
            # Normalize ingredients
            ingredients_lower = [ing.lower().strip() for ing in ingredients]
            
            # Join ingredients into a query string and preprocess
            query = ", ".join(ingredients)
            preprocessed_query = self.tfidf_model._preprocess_text(query, stem=False)
            query_terms = preprocessed_query.split()
            
            # Get initial candidates using TF-IDF similarity
            query_vector = self.tfidf_model.get_query_embedding(query)
            indices, similarities = self.tfidf_model.find_similar_recipes(
                query_vector,
                self.tfidf_embeddings,
                top_n * 20  # Get more candidates for better filtering
            )
            
            # Get candidate recipes
            candidates = self.df.iloc[indices].copy()
            candidates['tfidf_similarity'] = similarities

            # Score candidates based on ingredient matches with improved matching
            def score_recipe(row: pd.Series) -> Dict:
                try:
                    # Safely get recipe ingredients as a list of strings
                    recipe_ingredients_raw = row.get('RecipeIngredientParts', [])
                    
                    # Convert to list if string representation
                    if isinstance(recipe_ingredients_raw, str):
                        try:
                            # Handle various string formats
                            if recipe_ingredients_raw.startswith('[') and recipe_ingredients_raw.endswith(']'):
                                recipe_ingredients = eval(recipe_ingredients_raw)
                            elif recipe_ingredients_raw.startswith('c(') and recipe_ingredients_raw.endswith(')'):
                                recipe_ingredients = eval(recipe_ingredients_raw[2:-1])
                            else:
                                recipe_ingredients = recipe_ingredients_raw.split(',')
                        except:
                            recipe_ingredients = recipe_ingredients_raw.split(',')
                    elif isinstance(recipe_ingredients_raw, (list, np.ndarray)):
                        recipe_ingredients = list(recipe_ingredients_raw)
                    else:
                        recipe_ingredients = []

                    # Normalize recipe ingredients
                    recipe_ingredients_lower = [
                        str(ing).lower().strip() 
                        for ing in recipe_ingredients 
                        if ing is not None and str(ing).strip()
                    ]
                    
                    # Initialize scoring variables
                    matched_ingredients = 0
                    total_score = 0
                    
                    # Score each query ingredient against recipe ingredients
                    for query_ing in ingredients_lower:
                        best_match_score = 0
                        
                        # Find best matching ingredient
                        for recipe_ing in recipe_ingredients_lower:
                            # Calculate different similarity scores
                            token_ratio = fuzz.token_set_ratio(query_ing, recipe_ing)
                            partial_ratio = fuzz.partial_ratio(query_ing, recipe_ing)
                            
                            # Combine scores with weights
                            match_score = (token_ratio * 0.7 + partial_ratio * 0.3)
                            
                            if match_score > best_match_score and match_score >= 70:  # Threshold for matching
                                best_match_score = match_score
                        
                        # If a good match was found
                        if best_match_score > 0:
                            matched_ingredients += 1
                            total_score += best_match_score

                    # Calculate coverage and similarity scores
                    ingredient_coverage = matched_ingredients / len(ingredients_lower) if ingredients_lower else 0
                    avg_match_score = total_score / len(ingredients_lower) if ingredients_lower else 0
                    
                    # Get TF-IDF similarity as a scalar
                    tfidf_sim = float(row['tfidf_similarity']) if isinstance(row['tfidf_similarity'], (np.ndarray, pd.Series)) else row['tfidf_similarity']
                    
                    # Final score combines:
                    # - Ingredient coverage (50%)
                    # - Average match quality (30%)
                    # - TF-IDF similarity (20%)
                    final_score = (
                        ingredient_coverage * 0.5 +
                        (avg_match_score / 100) * 0.3 +
                        tfidf_sim * 0.2
                    )
                    
                    return {
                        'final_score': float(round(final_score, 2)),
                        'matched_count': matched_ingredients
                    }
                    
                except Exception as e:
                    self.logger.warning(f"Error scoring recipe: {str(e)}")
                    return {
                        'final_score': 0.0,
                        'matched_count': 0
                    }

            # Score all candidates
            self.logger.info("Scoring candidate recipes...")
            recipe_scores = []
            for idx, row in candidates.iterrows():
                try:
                    score_info = score_recipe(row)
                    if score_info['matched_count'] > 0:  # Only include recipes with at least one match
                        recipe_scores.append({
                            'idx': int(idx),
                            'score': score_info['final_score'],
                            'matched_count': score_info['matched_count']
                        })
                except Exception as e:
                    self.logger.warning(f"Error processing recipe {idx}: {str(e)}")
                    continue

            # Sort recipes by score
            recipe_scores.sort(key=lambda x: (x['matched_count'], x['score']), reverse=True)
            
            return self._format_recipe_responses([(r['idx'], r['score']) for r in recipe_scores[:top_n]])

            
        except Exception as e:
            self.logger.error(f"Error in TF-IDF ingredient-based recommendation: {str(e)}")
            raise

    def get_recipe_by_name(self, recipe_name: str, model_type: str = "bert", top_n: int = 5) -> List[RecipeResponse]:
        """Get recipe by name with specified model"""
        try:
            if model_type.lower() == "bert":
                return self._get_bert_recipe_by_name(recipe_name, top_n)
            elif model_type.lower() == "tfidf":
                return self._get_tfidf_recipe_by_name(recipe_name, top_n)
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
                
        except Exception as e:
            self.logger.error(f"Error getting recipe by name: {str(e)}")
            raise

    def _get_tfidf_recipe_by_name(self, recipe_name: str, top_n: int = 5) -> List[RecipeResponse]:
        """Get recipe by name using TF-IDF with BERT-style name matching"""
        try:
            recipe_name_lower = recipe_name.lower().strip()
            search_terms = set(recipe_name_lower.split())
            
            # Get initial matches using rapid fuzzy matching
            closest_matches = process.extractBests(
                recipe_name_lower,
                self.recipe_index.keys(),
                scorer=fuzz.token_sort_ratio,
                score_cutoff=60,
                limit=20  # Get more initial matches to ensure diversity
            )
            
            if not closest_matches:
                return []
                    
            # Score matches more precisely
            scored_matches = []
            for name, base_score in closest_matches:
                name_lower = name.lower()
                final_score = base_score  # starts with fuzzy match score (0-100)
                
                # Calculate bonus points
                bonus = 0
                
                # Exact match bonus
                if name_lower == recipe_name_lower:
                    bonus += 20
                # Contains exact phrase bonus
                elif recipe_name_lower in name_lower:
                    bonus += 10
                
                # Matching terms bonus (up to 20 points)
                name_terms = set(name_lower.split())
                matching_terms = len(search_terms & name_terms)
                terms_bonus = min(20, matching_terms * 5)  # Cap at 20 points
                bonus += terms_bonus
                
                # Apply bonus while ensuring score doesn't exceed 100
                final_score = min(100, final_score + bonus)
                
                scored_matches.append((name, final_score))
            
            # Sort by final score and get unique top matches
            scored_matches.sort(key=lambda x: x[1], reverse=True)
            unique_matches = []
            seen_base_names = set()
            
            for name, score in scored_matches:
                # Add if not too similar to existing matches
                base_name = ' '.join(sorted(name.lower().split()))
                if base_name not in seen_base_names:
                    unique_matches.append((name, score))
                    seen_base_names.add(base_name)
                
                if len(unique_matches) >= top_n:
                    break
                    
        
            return self._format_recipe_responses([(self.recipe_index[name.lower()], score/100) for name, score in unique_matches])
                
        except Exception as e:
            self.logger.error(f"Error in TF-IDF name search: {str(e)}")
            raise

    def _get_bert_recipe_by_name(
        self, 
        recipe_name: str,
        top_n: int = 5
    ) -> List[RecipeResponse]:
        """
        Get recipe by name with optimized fuzzy matching and similar recipes using BERT.
        
        Args:
            recipe_name: Name of the recipe to search for
            top_n: Number of similar recipes to return
        
        Returns:
            List of RecipeResponse objects, with the best matches first followed by similar recipes
        """
        try:
            recipe_name_lower = recipe_name.lower().strip()
            search_terms = set(recipe_name_lower.split())
            
            # Get initial matches using rapid fuzzy matching
            closest_matches = process.extractBests(
                recipe_name_lower,
                self.recipe_index.keys(),
                scorer=fuzz.token_sort_ratio,
                score_cutoff=60,
                limit=20  # Get more initial matches to ensure diversity
            )
            
            if not closest_matches:
                return []
                
            # Score matches more precisely
            scored_matches = []
            for name, base_score in closest_matches:
                name_lower = name.lower()
                final_score = base_score  # starts with fuzzy match score (0-100)
                
                # Calculate bonus points
                bonus = 0
                
                # Exact match bonus
                if name_lower == recipe_name_lower:
                    bonus += 20
                # Contains exact phrase bonus
                elif recipe_name_lower in name_lower:
                    bonus += 10
                
                # Matching terms bonus (up to 20 points)
                name_terms = set(name_lower.split())
                matching_terms = len(search_terms & name_terms)
                terms_bonus = min(20, matching_terms * 5)  # Cap at 20 points
                bonus += terms_bonus
                
                # Apply bonus while ensuring score doesn't exceed 100
                final_score = min(100, final_score + bonus)
                
                scored_matches.append((name, final_score))
            
            # Sort by final score and get unique top matches
            scored_matches.sort(key=lambda x: x[1], reverse=True)
            unique_matches = []
            seen_base_names = set()
            
            for name, score in scored_matches:
                # Add if not too similar to existing matches
                base_name = ' '.join(sorted(name.lower().split()))
                if base_name not in seen_base_names:
                    unique_matches.append((name, score))
                    seen_base_names.add(base_name)
                
                if len(unique_matches) >= top_n:
                    break
            
      
            return self._format_recipe_responses([(self.recipe_index[name.lower()], score/100) for name, score in unique_matches])
            
        except Exception as e:
            self.logger.error(f"Error getting recipe by name: {str(e)}")
            raise

    def refresh_models(self):
        """Refresh models by clearing cache and retraining"""
        try:
            # Clear cache
            self.bert_model.clear_cache()
            self.tfidf_model.clear_cache()
            
            # Reload everything
            self.load_data_and_models()
            self.logger.info("Successfully refreshed models")
        except Exception as e:
            self.logger.error(f"Error refreshing models: {str(e)}")
            raise

    def get_model_info(self) -> Dict:
        """Get information about the model and data"""
        return {
            "total_recipes": len(self.df),
            "total_ingredients": len(self.ingredient_index),
            "bert_embedding_shape": self.bert_embeddings.shape,
            "bert_info": self.bert_model.get_model_info(),
            "tfidf_info": self.tfidf_model.get_model_info(),
            "device": str(self.bert_model.device)
        }