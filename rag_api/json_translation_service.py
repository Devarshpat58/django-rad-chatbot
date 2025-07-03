"""
Enhanced JSON Translation Service for Django RAG Chatbot
Translates full source JSON data to reduce character limits and model failures
Handles query-related fields extraction and bidirectional translation
"""

import logging
import json
import time
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

# Import existing translation service
from .translation_service import TranslationService, get_translation_service

logger = logging.getLogger(__name__)


class JSONTranslationService:
    """
    Enhanced translation service that handles full JSON data translation
    Reduces character limits by translating complete source documents
    Extracts query-related fields for better user experience
    """
    
    def __init__(self):
        self.translation_service = get_translation_service()
        self.max_json_size = 50000  # Maximum JSON size to translate (characters)
        self.priority_fields = [
            'name', 'summary', 'description', 'space', 'neighborhood_overview',
            'host_about', 'notes', 'transit', 'access', 'interaction',
            'house_rules', 'amenities', 'property_type', 'room_type'
        ]
    
    def translate_full_response_data(self, response_data: Dict[str, Any], 
                                   user_language: str) -> Dict[str, Any]:
        """
        Translate the complete response data including all source JSON documents
        
        Args:
            response_data: Complete response data with results and metadata
            user_language: Target language for translation
            
        Returns:
            Translated response data with full JSON documents in user's language
        """
        logger.info(f"Starting full JSON translation to {user_language}")
        start_time = time.time()
        
        if user_language == 'en' or not user_language:
            logger.info("Target language is English, no translation needed")
            return response_data
        
        try:
            translated_data = response_data.copy()
            
            # Translate main response text if present
            if 'response' in translated_data and translated_data['response']:
                translated_response = self._translate_text_guaranteed(
                    translated_data['response'], 'en', user_language
                )
                translated_data['response'] = translated_response
            
            # Translate all source JSON documents in results
            if 'results' in translated_data and translated_data['results']:
                translated_results = []
                
                for result in translated_data['results']:
                    translated_result = self._translate_result_item(result, user_language)
                    translated_results.append(translated_result)
                
                translated_data['results'] = translated_results
            
            # Add translation metadata
            translated_data['translation_info'] = {
                'target_language': user_language,
                'translation_time': time.time() - start_time,
                'full_json_translated': True,
                'method': 'json_translation_service'
            }
            
            logger.info(f"Full JSON translation completed in {time.time() - start_time:.2f}s")
            return translated_data
            
        except Exception as e:
            logger.error(f"Full JSON translation failed: {e}")
            # Return original data as fallback
            response_data['translation_info'] = {
                'target_language': user_language,
                'translation_failed': True,
                'error': str(e),
                'fallback_used': True
            }
            return response_data
    
    def _translate_result_item(self, result: Dict[str, Any], 
                              target_language: str) -> Dict[str, Any]:
        """
        Translate a single result item including its source JSON data
        
        Args:
            result: Single result item with source_json
            target_language: Target language
            
        Returns:
            Translated result item
        """
        translated_result = result.copy()
        
        try:
            # Translate source JSON document
            if 'source_json' in result and result['source_json']:
                translated_json = self._translate_json_document(
                    result['source_json'], target_language
                )
                translated_result['source_json'] = translated_json
            
            # Translate AI summary
            if 'ai_summary' in result and result['ai_summary']:
                translated_summary = self._translate_text_guaranteed(
                    result['ai_summary'], 'en', target_language
                )
                translated_result['ai_summary'] = translated_summary
            
            # Update query-relevant fields with translated versions
            if 'query_relevant_fields' in result and result['query_relevant_fields']:
                translated_fields = self._translate_query_relevant_fields(
                    result['query_relevant_fields'], target_language
                )
                translated_result['query_relevant_fields'] = translated_fields
            
            # Re-extract query relevant fields from translated JSON
            if 'source_json' in translated_result:
                enhanced_fields = self._extract_enhanced_query_fields(
                    translated_result['source_json'], target_language
                )
                translated_result['enhanced_query_fields'] = enhanced_fields
            
        except Exception as e:
            logger.warning(f"Failed to translate result item: {e}")
            # Keep original result as fallback
            translated_result['translation_error'] = str(e)
        
        return translated_result
    
    def _translate_json_document(self, json_doc: Dict[str, Any], 
                                target_language: str) -> Dict[str, Any]:
        """
        Translate a complete JSON document by translating text fields
        
        Args:
            json_doc: Source JSON document
            target_language: Target language
            
        Returns:
            Translated JSON document
        """
        if not json_doc:
            return json_doc
        
        # Create a copy to avoid modifying original
        translated_doc = json_doc.copy()
        
        # Translate priority text fields first
        for field in self.priority_fields:
            if field in translated_doc and translated_doc[field]:
                original_value = translated_doc[field]
                
                if isinstance(original_value, str) and len(original_value.strip()) > 2:
                    translated_value = self._translate_text_guaranteed(
                        original_value, 'en', target_language
                    )
                    translated_doc[field] = translated_value
                elif isinstance(original_value, list):
                    # Handle list fields like amenities
                    translated_list = []
                    for item in original_value:
                        if isinstance(item, str) and len(item.strip()) > 1:
                            translated_item = self._translate_text_guaranteed(
                                item, 'en', target_language
                            )
                            translated_list.append(translated_item)
                        else:
                            translated_list.append(item)
                    translated_doc[field] = translated_list
        
        # Translate other text fields that might be relevant
        additional_fields = [
            'street', 'neighbourhood', 'neighbourhood_cleansed', 'city',
            'state', 'country', 'market', 'smart_location', 'country_code',
            'license', 'jurisdiction_names', 'cancellation_policy',
            'require_guest_profile_picture', 'require_guest_phone_verification'
        ]
        
        for field in additional_fields:
            if (field in translated_doc and 
                isinstance(translated_doc[field], str) and 
                len(translated_doc[field].strip()) > 2 and
                not translated_doc[field].isdigit()):
                
                translated_value = self._translate_text_guaranteed(
                    translated_doc[field], 'en', target_language
                )
                translated_doc[field] = translated_value
        
        return translated_doc
    
    def _translate_query_relevant_fields(self, fields: Dict[str, str], 
                                       target_language: str) -> Dict[str, str]:
        """
        Translate query-relevant fields dictionary
        
        Args:
            fields: Dictionary of field names to values
            target_language: Target language
            
        Returns:
            Translated fields dictionary
        """
        translated_fields = {}
        
        for field_name, field_value in fields.items():
            if isinstance(field_value, str) and len(field_value.strip()) > 1:
                translated_value = self._translate_text_guaranteed(
                    field_value, 'en', target_language
                )
                translated_fields[field_name] = translated_value
            else:
                translated_fields[field_name] = field_value
        
        return translated_fields
    
    def _extract_enhanced_query_fields(self, translated_json: Dict[str, Any], 
                                     target_language: str) -> Dict[str, str]:
        """
        Extract enhanced query-relevant fields from translated JSON
        
        Args:
            translated_json: Translated JSON document
            target_language: Target language
            
        Returns:
            Enhanced query-relevant fields in target language
        """
        enhanced_fields = {}
        
        # Extract key information that users typically ask about
        field_mappings = {
            'property_name': 'name',
            'property_type': 'property_type',
            'room_type': 'room_type',
            'location': ['neighbourhood_cleansed', 'city', 'country'],
            'price': 'price',
            'accommodates': 'accommodates',
            'bedrooms': 'bedrooms',
            'bathrooms': 'bathrooms',
            'beds': 'beds',
            'description': ['summary', 'description'],
            'amenities': 'amenities',
            'host': 'host_name',
            'rating': 'review_scores_rating',
            'reviews': 'number_of_reviews'
        }
        
        for display_name, json_fields in field_mappings.items():
            if isinstance(json_fields, list):
                # Try multiple fields, use first available
                for field in json_fields:
                    if field in translated_json and translated_json[field]:
                        enhanced_fields[display_name] = str(translated_json[field])
                        break
            else:
                # Single field
                if json_fields in translated_json and translated_json[json_fields]:
                    value = translated_json[json_fields]
                    if isinstance(value, list):
                        # For lists like amenities, show first few items
                        if len(value) > 3:
                            enhanced_fields[display_name] = ', '.join(str(v) for v in value[:3]) + f' (+{len(value)-3} more)'
                        else:
                            enhanced_fields[display_name] = ', '.join(str(v) for v in value)
                    else:
                        enhanced_fields[display_name] = str(value)
        
        return enhanced_fields
    
    def _translate_text_guaranteed(self, text: str, source_lang: str, 
                                  target_lang: str) -> str:
        """
        Guaranteed text translation with fallback
        
        Args:
            text: Text to translate
            source_lang: Source language
            target_lang: Target language
            
        Returns:
            Translated text (original if translation fails)
        """
        if not text or not text.strip():
            return text
        
        if source_lang == target_lang:
            return text
        
        try:
            # Use existing translation service
            if target_lang == 'en':
                result = self.translation_service.translate_text(text, 'en', source_lang)
            else:
                # For reverse translation (en -> other language)
                result = self.translation_service.translate_response_to_user_language(
                    text, target_lang
                )
            
            if result and result.get('translated_text' if target_lang == 'en' else 'translated_response'):
                translated = result.get('translated_text' if target_lang == 'en' else 'translated_response')
                if translated and translated.strip() and len(translated.strip()) > 1:
                    return translated
            
            # Fallback to original text
            return text
            
        except Exception as e:
            logger.warning(f"Text translation failed: {e}")
            return text
    
    def translate_query_to_english(self, query: str) -> Dict[str, Any]:
        """
        Translate user query to English for processing
        
        Args:
            query: User query in any language
            
        Returns:
            Dictionary with translation results
        """
        try:
            # Use existing guaranteed translation
            from .translation_service import translate_to_english_guaranteed
            return translate_to_english_guaranteed(query)
        except Exception as e:
            logger.error(f"Query translation failed: {e}")
            return {
                'english_query': query,
                'detected_language': 'en',
                'translation_needed': False,
                'ui_safe': True,
                'fallback_used': True,
                'error': str(e)
            }
    
    def get_supported_languages(self) -> Dict[str, str]:
        """Get list of supported languages"""
        return {
            'en': 'English',
            'es': 'Spanish',
            'fr': 'French',
            'de': 'German',
            'it': 'Italian',
            'pt': 'Portuguese',
            'ru': 'Russian',
            'zh': 'Chinese',
            'ja': 'Japanese',
            'ko': 'Korean',
            'ar': 'Arabic',
            'hi': 'Hindi'
        }


# Global instance
_json_translation_service = None

def get_json_translation_service() -> JSONTranslationService:
    """Get singleton instance of JSON translation service"""
    global _json_translation_service
    if _json_translation_service is None:
        _json_translation_service = JSONTranslationService()
    return _json_translation_service


# Convenience functions for easy integration
def translate_full_response_guaranteed(response_data: Dict[str, Any], 
                                     user_language: str) -> Dict[str, Any]:
    """
    Guaranteed full response translation with fallback
    
    Args:
        response_data: Complete response data
        user_language: Target language
        
    Returns:
        Translated response data
    """
    try:
        service = get_json_translation_service()
        return service.translate_full_response_data(response_data, user_language)
    except Exception as e:
        logger.error(f"Full response translation failed: {e}")
        # Return original data with error info
        response_data['translation_info'] = {
            'target_language': user_language,
            'translation_failed': True,
            'error': str(e),
            'fallback_used': True
        }
        return response_data


def extract_query_relevant_data(source_json: Dict[str, Any], 
                               query: str, user_language: str = 'en') -> Dict[str, Any]:
    """
    Extract and translate query-relevant data from source JSON
    
    Args:
        source_json: Source JSON document
        query: User query for context
        user_language: Target language
        
    Returns:
        Extracted and translated relevant data
    """
    try:
        service = get_json_translation_service()
        
        # First translate the JSON if needed
        if user_language != 'en':
            translated_json = service._translate_json_document(source_json, user_language)
        else:
            translated_json = source_json
        
        # Extract enhanced fields
        enhanced_fields = service._extract_enhanced_query_fields(translated_json, user_language)
        
        return {
            'translated_json': translated_json,
            'enhanced_fields': enhanced_fields,
            'query_context': query,
            'target_language': user_language
        }
        
    except Exception as e:
        logger.error(f"Query relevant data extraction failed: {e}")
        return {
            'translated_json': source_json,
            'enhanced_fields': {},
            'error': str(e)
        }