from typing import Dict, List, Optional, Tuple
import requests
from bs4 import BeautifulSoup
import logging
from pathlib import Path
import json
import time
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures

class RecipeImageService:
    def __init__(self, cache_dir: Path = Path("cache/images")):
        """Initialize the image service with caching capability"""
        self.logger = logging.getLogger(__name__)
        self.cache_dir = cache_dir
        self.cache_file = cache_dir / "image_cache.json"
        self.cache_duration = timedelta(days=7)  # Cache images for 7 days
        self._init_cache()
        
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
        }
        
        # Maximum number of concurrent threads
        self.max_workers = 5

    def _init_cache(self):
        """Initialize the image cache"""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        if not self.cache_file.exists():
            self._save_cache({})
        self.cache = self._load_cache()

    def _load_cache(self) -> Dict:
        """Load the image cache from disk"""
        try:
            with open(self.cache_file, 'r') as f:
                cache = json.load(f)
            return cache
        except Exception as e:
            self.logger.error(f"Error loading cache: {e}")
            return {}

    def _save_cache(self, cache: Dict):
        """Save the image cache to disk"""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(cache, f)
        except Exception as e:
            self.logger.error(f"Error saving cache: {e}")

    def _is_cache_valid(self, cache_entry: Dict) -> bool:
        """Check if a cache entry is still valid"""
        if not cache_entry or 'timestamp' not in cache_entry:
            return False
        cache_time = datetime.fromisoformat(cache_entry['timestamp'])
        return datetime.now() - cache_time < self.cache_duration

    def _clean_url(self, url: str) -> str:
        """
        Clean the URL by removing unwanted characters and normalizing format
        
        Args:
            url: Raw URL string
            
        Returns:
            Cleaned URL string
        """
        if not url:
            return ""
        
        # Remove all double quotes from the string
        url = url.replace('"', '')
        
        # Remove escaped quotes and backslashes
        url = url.strip('\'')
        
        # Remove any whitespace
        url = url.strip()
        
        # Clean any remaining quote artifacts
        url = url.replace('""', '')
        url = url.replace("''", '')
        
        # Ensure proper URL formatting
        if url and not url.startswith(('http://', 'https://')):
            url = 'https://' + url
            
        return url

    def get_recipe_images(self, recipe_names: List[str]) -> Dict[str, str]:
        """
        Get images for multiple recipes in parallel
        
        Args:
            recipe_names: List of recipe names to fetch images for
            
        Returns:
            Dictionary mapping recipe names to their image URLs
        """
        # First check cache for all recipes
        results = {}
        recipes_to_fetch = []
        
        for recipe_name in recipe_names:
            cache_key = recipe_name.lower()
            cache_entry = self.cache.get(cache_key)
            
            if cache_entry and self._is_cache_valid(cache_entry):
                results[recipe_name] = self._clean_url(cache_entry['url'])
            else:
                recipes_to_fetch.append(recipe_name)

        # If there are recipes that need fetching, do it in parallel
        if recipes_to_fetch:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_recipe = {
                    executor.submit(self._fetch_bing_image, recipe_name): recipe_name 
                    for recipe_name in recipes_to_fetch
                }
                
                for future in concurrent.futures.as_completed(future_to_recipe):
                    recipe_name = future_to_recipe[future]
                    try:
                        image_url = future.result()
                        if image_url:
                            cleaned_url = self._clean_url(image_url)
                            # Update cache
                            self.cache[recipe_name.lower()] = {
                                'url': cleaned_url,
                                'timestamp': datetime.now().isoformat()
                            }
                            results[recipe_name] = cleaned_url
                    except Exception as e:
                        self.logger.error(f"Error fetching image for {recipe_name}: {e}")

            # Save updated cache
            self._save_cache(self.cache)

        return results

    def _fetch_bing_image(self, query: str) -> Optional[str]:
        """
        Fetch a single image URL from Bing
        
        Args:
            query: Search query
            
        Returns:
            Image URL or None if not found
        """
        try:
            query = query.replace(" ", "+") + "+recipe+food"
            url = f"https://www.bing.com/images/search?q={query}&form=HDRSC2"
            
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, "html.parser")
            img_tags = soup.find_all("a", {"class": "iusc"})
            
            for img_tag in img_tags:
                try:
                    m = eval(img_tag["m"])
                    img_url = self._clean_url(m["murl"])
                    # Basic validation of image URL
                    if img_url.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                        return img_url
                except (KeyError, SyntaxError):
                    continue
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error in Bing image search: {e}")
            return None

    # Maintain backwards compatibility
    def get_recipe_image(self, recipe_name: str) -> Optional[str]:
        """
        Get a single image URL for a recipe, using cache if available
        
        Args:
            recipe_name: Name of the recipe to search for
            
        Returns:
            Single image URL or None if not found
        """
        results = self.get_recipe_images([recipe_name])
        return results.get(recipe_name)