import spacy
import re
from typing import List, Dict
import logging
from pathlib import Path

class EnhancedRecipeParser:
    def __init__(self):
        """Initialize the Enhanced Recipe Parser with spaCy"""
        self.logger = logging.getLogger(__name__)
        
        try:
            # Try to load the model
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            # If model is not found, download and load it
            self.logger.info("Downloading spaCy model...")
            import subprocess
            import sys
            
            try:
                # Run python -m spacy download en_core_web_sm
                subprocess.check_call([
                    sys.executable, 
                    "-m", 
                    "spacy", 
                    "download", 
                    "en_core_web_sm"
                ])
                self.nlp = spacy.load("en_core_web_sm")
            except Exception as e:
                # Fallback to simple parsing if spaCy fails
                self.logger.warning(f"Failed to download spaCy model: {str(e)}")
                self.nlp = None
                
        # Custom sentence boundaries for recipe-specific cases if spaCy is available
        if self.nlp is not None:
            self.nlp.add_pipe("sentencizer")
            
    def _clean_instruction(self, text: str) -> str:
        """Clean and normalize instruction text"""
        # Remove multiple spaces and normalize whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        # Remove numbered prefixes (e.g., "1.", "2)", "[3]")
        text = re.sub(r'^\s*(?:\d+\.|\[\d+\]|\d+\))\s*', '', text)
        return text.strip()

    def _is_valid_instruction(self, text: str) -> bool:
        """Check if the instruction is valid and meaningful"""
        if self.nlp is None:
            # Simple validation if spaCy is not available
            return len(text.split()) >= 3
            
        # Process text with spaCy
        doc = self.nlp(text)
        
        # Check if instruction has at least one verb
        has_verb = any(token.pos_ == "VERB" for token in doc)
        
        # Check minimum word count (excluding punctuation)
        min_words = len([token for token in doc if not token.is_punct]) >= 3
        
        return has_verb and min_words

    def _merge_related_instructions(self, instructions: List[str]) -> List[str]:
        """Merge related instructions based on linguistic analysis"""
        if self.nlp is None:
            # Simple merging if spaCy is not available
            return instructions
            
        merged = []
        current = ""
        
        for instr in instructions:
            doc = self.nlp(instr)
            
            # Check if instruction starts with a connecting word
            starts_with_connector = any(
                token.dep_ in ["cc", "mark"] or 
                token.lower_ in ["and", "then", "also", "while", "until"]
                for token in doc[:2]
            )
            
            # Check if previous instruction ends with a comma or connecting word
            ends_with_connector = (
                current and 
                (current.endswith(",") or 
                 any(current.lower().endswith(conn) for conn in [" and", " then", " while"]))
            )
            
            if starts_with_connector or ends_with_connector:
                current = (current + " " + instr).strip()
            else:
                if current:
                    merged.append(current)
                current = instr
                
        if current:
            merged.append(current)
            
        return merged

    def _extract_timing_info(self, instruction: str) -> Dict:
        """Extract timing information from instruction"""
        # Pattern for time expressions
        time_pattern = re.compile(
            r'(\d+)(?:\s*-\s*\d+)?\s*'
            r'(seconds?|secs?|minutes?|mins?|hours?|hrs?)'
        )
        
        timing_info = {}
        
        # Find all time mentions
        time_matches = time_pattern.finditer(instruction)
        for match in time_matches:
            amount = int(match.group(1))
            unit = match.group(2).lower()
            
            # Normalize time units
            if unit.startswith(('sec', 's')):
                timing_info['seconds'] = amount
            elif unit.startswith(('min', 'm')):
                timing_info['minutes'] = amount
            elif unit.startswith(('hour', 'hr', 'h')):
                timing_info['hours'] = amount
                
        return timing_info

    def parse_instructions(self, raw_instructions: str) -> List[Dict]:
        """Parse and structure recipe instructions with enhanced NLP processing"""
        try:
            # Handle different input formats
            if isinstance(raw_instructions, (list, tuple)):
                instructions = [str(i) for i in raw_instructions]
            elif isinstance(raw_instructions, str):
                # Split on common instruction separators
                instructions = re.split(r'(?:\r?\n)|(?:\.(?!\d))|(?:\d+\.\s*)', raw_instructions)
            else:
                return []
            
            # Clean and validate individual instructions
            valid_instructions = []
            for instr in instructions:
                cleaned = self._clean_instruction(instr)
                if cleaned and self._is_valid_instruction(cleaned):
                    valid_instructions.append(cleaned)
            
            # Merge related instructions
            merged_instructions = self._merge_related_instructions(valid_instructions)
            
            # Process each instruction
            structured_instructions = []
            for idx, instr in enumerate(merged_instructions, 1):
                timing = self._extract_timing_info(instr)
                
                if self.nlp is not None:
                    doc = self.nlp(instr)
                    # Extract key verbs (actions)
                    actions = [token.lemma_ for token in doc if token.pos_ == "VERB"]
                    # Extract ingredients mentioned
                    ingredients = [
                        token.text for token in doc 
                        if token.pos_ in ["NOUN", "PROPN"] and 
                        not token.is_stop
                    ]
                else:
                    # Simple extraction if spaCy is not available
                    actions = []
                    ingredients = []
                
                # Create structured instruction
                structured_instruction = {
                    'step_number': idx,
                    'text': instr,
                    'actions': actions,
                    'ingredients_mentioned': ingredients,
                    'timing': timing,
                    'requires_attention': any(
                        word in instr.lower() 
                        for word in ['careful', 'watch', 'monitor', 'check']
                    )
                }
                structured_instructions.append(structured_instruction)
            
            return structured_instructions
            
        except Exception as e:
            self.logger.error(f"Error parsing instructions: {str(e)}")
            return []
            
    def get_equipment_needed(self, instructions: List[Dict]) -> List[str]:
        """Extract required equipment from instructions"""
        common_equipment = {
            'oven', 'stove', 'pan', 'pot', 'bowl', 'knife', 'spoon', 'spatula',
            'whisk', 'grater', 'blender', 'mixer', 'sheet', 'dish', 'skillet',
            'cutting board', 'colander', 'strainer', 'thermometer'
        }
        
        equipment = set()
        
        for instruction in instructions:
            text = instruction['text'].lower()
            
            if self.nlp is not None:
                doc = self.nlp(text)
                # Look for equipment mentions using spaCy
                for chunk in doc.noun_chunks:
                    if any(equip in chunk.text.lower() for equip in common_equipment):
                        equipment.add(chunk.text.strip())
            else:
                # Simple word matching if spaCy is not available
                for equip in common_equipment:
                    if equip in text:
                        equipment.add(equip)
                    
        return sorted(list(equipment))