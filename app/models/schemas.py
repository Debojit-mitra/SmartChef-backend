 
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Union, Any

class NutritionalInfo(BaseModel):
    calories: Optional[float] = Field(None, description="Calories per serving")
    fat_content: Optional[float] = Field(None, description="Total fat in grams")
    saturated_fat_content: Optional[float] = Field(None, description="Saturated fat in grams")
    cholesterol_content: Optional[float] = Field(None, description="Cholesterol in milligrams")
    sodium_content: Optional[float] = Field(None, description="Sodium in milligrams")
    carbohydrate_content: Optional[float] = Field(None, description="Total carbohydrates in grams")
    fiber_content: Optional[float] = Field(None, description="Dietary fiber in grams")
    sugar_content: Optional[float] = Field(None, description="Sugar content in grams")
    protein_content: Optional[float] = Field(None, description="Protein in grams")

class RecipeResponse(BaseModel):
    recipe_id: str
    name: str
    cook_time: Optional[str]
    prep_time: Optional[str]
    total_time: Optional[str]
    image_url: Optional[str]
    recipe_category: Optional[str]
    ingredient_quantities: List[str]
    ingredient_parts: List[str]
    aggregated_rating: Optional[float]
    rating_count: Optional[int]
    nutritional_info: NutritionalInfo
    recipe_servings: Optional[int]
    recipe_instructions: List[Dict[str, Any]] #List[str]
    equipment_needed: Optional[List[str]]
    similarity_score: float

class RecipeQuery(BaseModel):
    query: str
    model_type: str = Field("bert", description="Model type: 'bert' or 'tfidf'")
    top_n: int = Field(5, ge=1, le=20, description="Number of recommendations")

class IngredientsQuery(BaseModel):
    ingredients: List[str]
    model_type: str = Field("bert", description="Model type: 'bert' or 'tfidf'")
    top_n: int = Field(5, ge=1, le=20, description="Number of recommendations")

class RecommendationResponse(BaseModel):
    recommendations: List[RecipeResponse]
    query_time: float
    model_used: str

class ChatResponse(BaseModel):
    chat_id: str
    message: str
    query_time: float