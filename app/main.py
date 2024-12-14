# app/main.py

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from app.services.chat_service import ChatService
from fastapi import WebSocket, WebSocketDisconnect
import uuid
from fastapi.responses import JSONResponse
from app.models.schemas import (
    RecipeQuery,
    IngredientsQuery,
    RecipeResponse,
    RecommendationResponse
)
from app.services.model_manager import ModelManager
from app.core.logging_config import setup_logging
from app.core.config import settings
from app.core.cache import RecipeCache
import time
import logging
from typing import List, Optional
import uvicorn
import json
import asyncio

# Setup logging
logger = setup_logging()

# Initialize FastAPI app
app = FastAPI(
    title="Recipe Recommendation API",
    description="API for recipe recommendations using BERT and TF-IDF models",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Modify this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize global variables
model_manager = None
chat_service = None
recipe_cache = None

@app.on_event("startup")
async def startup_event():
    """Initialize models and cache on startup"""
    global model_manager, chat_service, recipe_cache
    try:
        logger.info("Initializing model manager...")
        model_manager = ModelManager()
        logger.info("Model manager initialized successfully")
        
        logger.info("Initializing chat service...")
        chat_service = ChatService()
        logger.info("ChatService initialized successfully")
        
        logger.info("Initializing recipe cache...")
        recipe_cache = RecipeCache(
            max_size=settings.CACHE_MAX_SIZE,
            ttl=settings.CACHE_TTL
        )
        logger.info("Recipe cache initialized successfully")
        
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")
        raise

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests and their processing time"""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    
    logger.info(
        f"Path: {request.url.path} | "
        f"Method: {request.method} | "
        f"Processing Time: {process_time:.4f}s | "
        f"Status Code: {response.status_code}"
    )
    
    return response

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if model_manager is None or recipe_cache is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    return {
        "status": "healthy",
        "models_loaded": True,
        "model_info": model_manager.get_model_info(),
        "cache_info": {
            "size": len(recipe_cache.cache),
            "max_size": recipe_cache.max_size,
            "ttl": recipe_cache.ttl
        }
    }

@app.post("/api/v1/recommend/ingredients", response_model=RecommendationResponse)
async def recommend_by_ingredients(query: IngredientsQuery):
    """Get recipe recommendations based on ingredients with caching"""
    if not query.ingredients:
        raise HTTPException(
            status_code=400,
            detail="No ingredients provided"
        )
        
    cache_key = recipe_cache._generate_key(
        'ingredients',
        sorted(query.ingredients),
        query.model_type,
        query.top_n
    )
    
    # Try to get from cache
    cached_result = recipe_cache.get(cache_key)
    if cached_result:
        logger.info("Serving ingredients recommendation from cache")
        return RecommendationResponse(**cached_result)
    
    # Generate new recommendations
    start_time = time.time()
    logger.info(f"Ingredient recommendation request - Ingredients: {query.ingredients}, Model: {query.model_type}")
    
    try:
        recommendations = model_manager.get_recommendations_by_ingredients(
            query.ingredients,
            query.model_type,
            query.top_n
        )
        
        process_time = time.time() - start_time
        logger.info(f"Ingredient recommendation completed in {process_time:.4f}s")
        process_time = round(process_time, 2)
        
        response = RecommendationResponse(
            recommendations=recommendations,
            query_time=process_time,
            model_used=query.model_type
        )
        
        # Cache the result
        recipe_cache.set(cache_key, response.dict())
        
        return response
    
    except Exception as e:
        logger.error(f"Error processing ingredient recommendation: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@app.get("/api/v1/recipe/{recipe_name}")
async def get_recipe(
    recipe_name: str, 
    model_type: str = "bert", 
    top_n: int = 5
):
    """Get recipes matching the name with caching"""
    if not recipe_name.strip():
        raise HTTPException(
            status_code=400,
            detail="Recipe name cannot be empty"
        )
        
    cache_key = recipe_cache._generate_key('recipe', recipe_name, model_type, top_n)
    
    # Try to get from cache
    cached_result = recipe_cache.get(cache_key)
    if cached_result:
        logger.info("Serving recipe lookup from cache")
        return RecommendationResponse(**cached_result)
    
    start_time = time.time()
    logger.info(f"Recipe lookup request - Name: {recipe_name}, Model: {model_type}, Top_N: {top_n}")
    
    try:
        recipes = model_manager.get_recipe_by_name(
            recipe_name,
            model_type=model_type,
            top_n=top_n
        )
        if not recipes:
            raise HTTPException(
                status_code=404,
                detail=f"Recipe not found: {recipe_name}"
            )
        
        process_time = time.time() - start_time
        logger.info(f"Recipe recommendation completed in {process_time:.4f}s")
        process_time = round(process_time, 2)

        response = RecommendationResponse(
            recommendations=recipes,
            query_time=process_time,
            model_used=model_type
        )
        
        # Cache the result
        recipe_cache.set(cache_key, response.dict())
        
        return response
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting recipe: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@app.websocket("/ws/chat")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for chat functionality"""
    client_id = str(uuid.uuid4())
    
    try:
        await chat_service.manager.connect(websocket, client_id)
        
        while True:
            try:
                # Receive message with timeout
                data = await websocket.receive_json()
                
                # Handle heartbeat response
                if data.get("type") == "heartbeat":
                    continue
                
                message = data.get("message")
                chat_id = data.get("chat_id")
                uname = data.get("users_name")
                dateOfBirth = data.get("users_dob")
                gender = data.get("users_gender")
                cusine_pref = data.get("cuisinePreferences")
                diet_pref = data.get("dietPreference")
                medical_condi = data.get("conditions")
                health_consi = data.get("healthConscious")
                
                if not message:
                    await chat_service.manager.send_message(client_id, {
                        "type": "error",
                        "error": "No message provided"
                    })
                    continue
                
                # Process message
                await chat_service.process_message(
                    websocket,
                    client_id,
                    message,
                    uname,
                    dateOfBirth,
                    gender,
                    health_consi,
                    chat_id,
                    cusine_pref,
                    diet_pref,
                    medical_condi
                )
                
            except WebSocketDisconnect:
                await chat_service.manager.disconnect(client_id)
                break
            except json.JSONDecodeError:
                await chat_service.manager.send_message(client_id, {
                    "type": "error",
                    "error": "Invalid JSON format"
                })
            except Exception as e:
                logger.error(f"Error processing message: {str(e)}")
                await chat_service.manager.send_message(client_id, {
                    "type": "error",
                    "error": str(e)
                })
                
    except Exception as e:
        logger.error(f"Error in websocket connection: {e}")
        await chat_service.manager.disconnect(client_id)

# Cache management endpoints
@app.post("/api/v1/cache/clear")
async def clear_cache():
    """Clear the recipe cache"""
    recipe_cache.clear()
    return {"message": "Cache cleared successfully"}

@app.get("/api/v1/cache/stats")
async def get_cache_stats():
    """Get cache statistics"""
    return {
        "size": len(recipe_cache.cache),
        "max_size": recipe_cache.max_size,
        "ttl": recipe_cache.ttl,
        "hit_rate": recipe_cache.get_hit_rate() if hasattr(recipe_cache, 'get_hit_rate') else None
    }

@app.post("/api/v1/models/refresh")
async def refresh_models():
    """Refresh the models and clear cache"""
    try:
        logger.info("Starting model refresh")
        model_manager.refresh_models()
        recipe_cache.clear()
        logger.info("Model refresh completed successfully")
        return {"message": "Models refreshed and cache cleared successfully"}
    except Exception as e:
        logger.error(f"Error refreshing models: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error refreshing models: {str(e)}"
        )

# run.py
if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )