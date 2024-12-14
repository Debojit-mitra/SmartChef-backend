from functools import lru_cache
from typing import Dict, List, Optional
import time
from app.models.schemas import RecipeResponse, RecommendationResponse
import json
from dataclasses import asdict
import hashlib

class RecipeCache:
    def __init__(self, max_size: int = 1000, ttl: int = 3600):
        self.cache: Dict = {}
        self.max_size = max_size
        self.ttl = ttl  # Time to live in seconds

    def _generate_key(self, *args, **kwargs) -> str:
        """Generate a cache key from arguments"""
        key_parts = [str(arg) for arg in args]
        key_parts.extend(f"{k}:{v}" for k, v in sorted(kwargs.items()))
        key_string = "|".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()

    def get(self, key: str) -> Optional[dict]:
        """Get item from cache if it exists and hasn't expired"""
        if key in self.cache:
            item = self.cache[key]
            if time.time() - item['timestamp'] < self.ttl:
                return item['data']
            else:
                del self.cache[key]
        return None

    def set(self, key: str, value: dict):
        """Set item in cache with timestamp"""
        if len(self.cache) >= self.max_size:
            # Remove oldest item
            oldest = min(self.cache.items(), key=lambda x: x[1]['timestamp'])
            del self.cache[oldest[0]]
        
        self.cache[key] = {
            'data': value,
            'timestamp': time.time()
        }

    def clear(self):
        """Clear all cached items"""
        self.cache.clear()