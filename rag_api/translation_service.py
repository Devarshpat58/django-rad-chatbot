"""
Translation Service for Django RAG Chatbot
Provides lightweight language detection and translation to English
Ensures all queries are processed in English for consistent RAG performance
Uses basic language detection with manual translation patterns - no API keys required
"""

import logging
import re
from typing import Dict, Tuple, Optional
from functools import lru_cache

logger = logging.getLogger(__name__)


class TranslationService:
    """
    Lightweight translation service using basic language detection
    Handles language detection and translation to English
    No API keys required - completely free and reliable
    """
    
    def __init__(self):
        self._supported_languages = {
            'en': 'English',
            'es': 'Spanish', 
            'fr': 'French',
            'de': 'German',
            'it': 'Italian',
            'pt': 'Portuguese'
        }
        
        # Basic language detection patterns
        self._language_patterns = {
            'es': [
                r'\b(apartamentos?|dormitorios?|habitaciones?|encuentra|busca|precio)\b',
                r'\b(de|con|para|en|por|menos|más)\b',
                r'\b(casa|casas|alquiler|venta)\b'
            ],
            'fr': [
                r'\b(appartements?|chambres?|trouvez|cherchez|prix)\b',
                r'\b(de|avec|pour|dans|par|moins|plus)\b',
                r'\b(maison|maisons|location|vente)\b'
            ],
            'de': [
                r'\b(wohnungen?|zimmer|finde|suche|preis|günstige?|guenstige?|billige?|apartment|apartments)\b',
                r'\b(mit|für|in|von|unter|über|dollar|euro|new\s*york|nyc)\b',
                r'\b(haus|häuser|miete|verkauf|sie|finden)\b'
            ],
            'it': [
                r'\b(appartamenti?|camere?|trova|cerca|prezzo)\b',
                r'\b(con|per|in|da|sotto|sopra)\b',
                r'\b(casa|case|affitto|vendita)\b'
            ],
            'pt': [
                r'\b(apartamentos?|quartos?|encontre|procure|preço)\b',
                r'\b(com|para|em|de|abaixo|acima)\b',
                r'\b(casa|casas|aluguel|venda)\b'
            ]
        }
        
        # Basic translation patterns for common real estate terms
        self._translation_patterns = {
            'es': {
                r'\bapartamentos?\b': 'apartments',
                r'\bdormitorios?\b': 'bedrooms',
                r'\bhabitaciones?\b': 'rooms',
                r'\bencuentra\b': 'find',
                r'\bbusca\b': 'search',
                r'\bcasa\b': 'house',
                r'\bcasas\b': 'houses',
                r'\balquiler\b': 'rental',
                r'\bprecio\b': 'price',
                r'\bmenos de\b': 'less than',
                r'\bmás de\b': 'more than'
            },
            'fr': {
                r'\bappartements?\b': 'apartments',
                r'\bchambres?\b': 'bedrooms',
                r'\btrouvez\b': 'find',
                r'\bcherchez\b': 'search',
                r'\bmaison\b': 'house',
                r'\bmaisons\b': 'houses',
                r'\blocation\b': 'rental',
                r'\bprix\b': 'price',
                r'\bmoins de\b': 'less than',
                r'\bplus de\b': 'more than'
            },
            'de': {
                r'\bwohnungen?\b': 'apartments',
                r'\bzimmer\b': 'rooms',
                r'\bfinde\b': 'find',
                r'\bfinden\b': 'find',
                r'\bsuche\b': 'search',
                r'\bhaus\b': 'house',
                r'\bhäuser\b': 'houses',
                r'\bmiete\b': 'rental',
                r'\bpreis\b': 'price',
                r'\bunter\b': 'under',
                r'\büber\b': 'over',
                r'\bgünstige?\b': 'cheap',
                r'\bguenstige?\b': 'cheap',  # ASCII version
                r'\bbillige?\b': 'cheap',
                r'\bapartment\b': 'apartment',
                r'\bapartments\b': 'apartments',
                r'\bdollar\b': 'dollar',
                r'\beuro\b': 'euro',
                r'\bnew\s*york\b': 'new york',
                r'\bnyc\b': 'new york',
                r'\bsie\b': ''  # Remove formal "you" as it's not needed in English
            },
            'it': {
                r'\bappartamenti?\b': 'apartments',
                r'\bcamere?\b': 'rooms',
                r'\btrova\b': 'find',
                r'\bcerca\b': 'search',
                r'\bcasa\b': 'house',
                r'\bcase\b': 'houses',
                r'\baffitto\b': 'rental',
                r'\bprezzo\b': 'price',
                r'\bsotto\b': 'under',
                r'\bsopra\b': 'over'
            },
            'pt': {
                r'\bapartamentos?\b': 'apartments',
                r'\bquartos?\b': 'rooms',
                r'\bencontre\b': 'find',
                r'\bprocure\b': 'search',
                r'\bcasa\b': 'house',
                r'\bcasas\b': 'houses',
                r'\baluguel\b': 'rental',
                r'\bpreço\b': 'price',
                r'\babaixo de\b': 'under',
                r'\bacima de\b': 'over'
            }
        }
        
        logger.info("Translation service initialized successfully with basic language detection (no API key required)")
    
    def is_available(self) -> bool:
        """Check if translation service is available"""
        return True  # Always available since it's built-in
    
    @lru_cache(maxsize=1000)
    def detect_language(self, text: str) -> Tuple[str, float]:
        """
        Detect the language of the input text using pattern matching
        
        Args:
            text: Input text to analyze
            
        Returns:
            Tuple of (language_code, confidence)
        """
        if not text or len(text.strip()) < 3:
            return 'en', 1.0  # Default to English for very short text
        
        try:
            # Clean text for better detection
            cleaned_text = self._clean_text_for_detection(text).lower()
            
            # Count matches for each language
            language_scores = {}
            
            for lang, patterns in self._language_patterns.items():
                score = 0
                for pattern in patterns:
                    matches = len(re.findall(pattern, cleaned_text, re.IGNORECASE))
                    score += matches
                
                if score > 0:
                    language_scores[lang] = score
            
            # Determine the most likely language
            if language_scores:
                detected_lang = max(language_scores, key=language_scores.get)
                max_score = language_scores[detected_lang]
                confidence = min(0.9, max_score * 0.3)  # Scale confidence
                
                logger.debug(f"Detected language: {detected_lang} (confidence: {confidence:.2f}) for text: {text[:50]}...")
                return detected_lang, confidence
            else:
                # Default to English if no patterns match
                return 'en', 0.8
                
        except Exception as e:
            logger.warning(f"Language detection failed: {e}. Defaulting to English.")
            return 'en', 0.5
    
    def _clean_text_for_detection(self, text: str) -> str:
        """Clean text for better language detection"""
        if not text:
            return ""
        
        # Remove URLs, emails, and numbers that might confuse detection
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'\$\d+', '', text)  # Remove prices
        text = re.sub(r'\d+', '', text)    # Remove numbers
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text.strip()
    
    @lru_cache(maxsize=500)
    def translate_to_english(self, text: str, source_language: Optional[str] = None) -> Dict[str, str]:
        """
        Translate text to English using pattern-based translation
        
        Args:
            text: Text to translate
            source_language: Source language code (optional, will auto-detect if not provided)
            
        Returns:
            Dictionary containing original text, translated text, detected language, and confidence
        """
        if not text or not text.strip():
            return {
                'original_text': text,
                'translated_text': text,
                'detected_language': 'en',
                'confidence': 1.0,
                'translation_needed': False
            }
        
        # Detect language if not provided
        if not source_language:
            detected_lang, confidence = self.detect_language(text)
        else:
            detected_lang = source_language
            confidence = 0.9
        
        # If already English, return original
        if detected_lang == 'en':
            return {
                'original_text': text,
                'translated_text': text,
                'detected_language': detected_lang,
                'confidence': confidence,
                'translation_needed': False
            }
        
        try:
            # Apply pattern-based translation
            translated_text = text
            
            if detected_lang in self._translation_patterns:
                patterns = self._translation_patterns[detected_lang]
                for pattern, replacement in patterns.items():
                    translated_text = re.sub(pattern, replacement, translated_text, flags=re.IGNORECASE)
            
            # If translation was applied, log it
            if translated_text != text:
                logger.info(f"Translated from {detected_lang} to English: '{text[:50]}...' -> '{translated_text[:50]}...'")
                translation_needed = True
            else:
                # No translation patterns matched, but we detected a foreign language
                # Keep original text but mark as foreign
                translation_needed = True
            
            return {
                'original_text': text,
                'translated_text': translated_text,
                'detected_language': detected_lang,
                'confidence': confidence,
                'translation_needed': translation_needed
            }
            
        except Exception as e:
            logger.error(f"Translation failed: {e}. Using original text.")
            return {
                'original_text': text,
                'translated_text': text,
                'detected_language': detected_lang,
                'confidence': confidence,
                'translation_needed': False,
                'translation_error': str(e)
            }
    
    def process_query(self, query_text: str) -> Dict[str, str]:
        """
        Process a user query by detecting language and translating to English if needed
        Enhanced to preserve geographic entities and improve location recognition
        
        Args:
            query_text: User's query text
            
        Returns:
            Dictionary with translation results and metadata
        """
        if not query_text or not query_text.strip():
            return {
                'original_query': query_text,
                'english_query': query_text,
                'detected_language': 'en',
                'confidence': 1.0,
                'translation_needed': False
            }
        
        # Detect and translate
        translation_result = self.translate_to_english(query_text)
        
        # Enhanced post-processing to ensure geographic entities are preserved
        english_query = translation_result['translated_text']
        
        # Ensure location entities are properly formatted
        english_query = self._enhance_location_entities(english_query)
        
        return {
            'original_query': translation_result['original_text'],
            'english_query': english_query,
            'detected_language': translation_result['detected_language'],
            'confidence': translation_result['confidence'],
            'translation_needed': translation_result['translation_needed'],
            'translation_error': translation_result.get('translation_error')
        }
    
    def _enhance_location_entities(self, text: str) -> str:
        """
        Enhance location entity recognition and formatting
        
        Args:
            text: Translated text to enhance
            
        Returns:
            Enhanced text with better location formatting
        """
        # Normalize New York variations
        text = re.sub(r'\bnew\s*york\s*city\b', 'New York', text, flags=re.IGNORECASE)
        text = re.sub(r'\bnyc\b', 'New York', text, flags=re.IGNORECASE)
        text = re.sub(r'\bnew\s*york\s*ny\b', 'New York', text, flags=re.IGNORECASE)
        
        # Ensure proper capitalization for major cities
        text = re.sub(r'\bnew\s*york\b', 'New York', text, flags=re.IGNORECASE)
        text = re.sub(r'\bmanhattan\b', 'Manhattan', text, flags=re.IGNORECASE)
        text = re.sub(r'\bbrooklyn\b', 'Brooklyn', text, flags=re.IGNORECASE)
        
        return text
    
    def get_supported_languages(self) -> Dict[str, str]:
        """
        Get list of supported languages
        
        Returns:
            Dictionary of language codes and names
        """
        return self._supported_languages
    
    def get_service_status(self) -> Dict[str, any]:
        """
        Get translation service status
        
        Returns:
            Service status information
        """
        return {
            'available': self.is_available(),
            'service': 'Basic Pattern Translation (no API key required)',
            'supported_languages_count': len(self.get_supported_languages()),
            'cache_size': self.translate_to_english.cache_info().currsize,
            'api_key_required': False
        }


# Global instance for singleton pattern
_translation_service = None


def get_translation_service() -> TranslationService:
    """Get the global translation service instance"""
    global _translation_service
    if _translation_service is None:
        _translation_service = TranslationService()
    return _translation_service


def translate_query_to_english(query_text: str) -> Dict[str, str]:
    """
    Convenience function to translate a query to English
    
    Args:
        query_text: User's query text
        
    Returns:
        Dictionary with translation results
    """
    service = get_translation_service()
    return service.process_query(query_text)


def is_translation_available() -> bool:
    """Check if translation service is available"""
    service = get_translation_service()
    return service.is_available()