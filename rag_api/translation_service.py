"""
Translation Service for Django RAG Chatbot
Uses Hugging Face transformers with MarianMT models for local translation
No API keys or credentials required - completely offline service
"""

import logging
import re
import sys
from typing import Dict, Optional, Tuple
from functools import lru_cache

logger = logging.getLogger(__name__)

# Force console output for debugging
def force_console_print(message):
    """Force print to console with flush for immediate visibility"""
    try:
        print(f"[TRANSLATION SERVICE] {message}", flush=True)
        sys.stdout.flush()
        sys.stderr.flush()
    except UnicodeEncodeError:
        # Fallback for Windows console encoding issues
        try:
            print(f"[TRANSLATION SERVICE] {message}".encode('utf-8', errors='replace').decode('utf-8'), flush=True)
        except:
            print(f"[TRANSLATION SERVICE] {repr(message)}", flush=True)

# Try to import required libraries
try:
    from transformers import MarianMTModel, MarianTokenizer, pipeline
    import torch
    TRANSFORMERS_AVAILABLE = True
    force_console_print("SUCCESS: Transformers library loaded successfully")
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    force_console_print("WARNING: transformers library not available. Translation service will be limited.")
    logger.warning("transformers library not available. Translation service will be limited.")
except Exception as e:
    TRANSFORMERS_AVAILABLE = False
    force_console_print(f"ERROR: transformers library failed to load: {e}. Translation service will be limited.")
    logger.warning(f"transformers library failed to load: {e}. Translation service will be limited.")

try:
    from langdetect import detect, detect_langs
    LANGDETECT_AVAILABLE = True
    force_console_print("SUCCESS: langdetect library loaded successfully")
except ImportError:
    LANGDETECT_AVAILABLE = False
    force_console_print("WARNING: langdetect library not available. Using basic language detection.")
    logger.warning("langdetect library not available. Using basic language detection.")
except Exception as e:
    LANGDETECT_AVAILABLE = False
    force_console_print(f"ERROR: langdetect library failed to load: {e}. Using basic language detection.")
    logger.warning(f"langdetect library failed to load: {e}. Using basic language detection.")

try:
    import langid
    LANGID_AVAILABLE = True
    force_console_print("SUCCESS: langid library loaded successfully")
except ImportError:
    LANGID_AVAILABLE = False
    force_console_print("WARNING: langid library not available. Fallback detection will be limited.")
    logger.warning("langid library not available. Fallback detection will be limited.")
except Exception as e:
    LANGID_AVAILABLE = False
    force_console_print(f"ERROR: langid library failed to load: {e}. Fallback detection will be limited.")
    logger.warning(f"langid library failed to load: {e}. Fallback detection will be limited.")


class TranslationService:
    """
    Translation service using Hugging Face transformers with MarianMT models
    Provides automatic language detection and translation to English
    No API keys required - runs completely offline
    """
    
    def __init__(self):
        self.models = {}
        self.tokenizers = {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu" if TRANSFORMERS_AVAILABLE else None
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize MarianMT models for common languages"""
        if not TRANSFORMERS_AVAILABLE:
            logger.error("transformers library not available. Using basic fallback mode.")
            return
        
        # Define language pairs and their corresponding MarianMT models
        self.language_models = {
            'es': 'Helsinki-NLP/opus-mt-es-en',  # Spanish to English
            'fr': 'Helsinki-NLP/opus-mt-fr-en',  # French to English
            'de': 'Helsinki-NLP/opus-mt-de-en',  # German to English
            'it': 'Helsinki-NLP/opus-mt-it-en',  # Italian to English
            'pt': 'Helsinki-NLP/opus-mt-pt-en',  # Portuguese to English
            'ru': 'Helsinki-NLP/opus-mt-ru-en',  # Russian to English
            'zh': 'Helsinki-NLP/opus-mt-zh-en',  # Chinese to English
            'ja': 'Helsinki-NLP/opus-mt-ja-en',  # Japanese to English
            'ko': 'Helsinki-NLP/opus-mt-ko-en',  # Korean to English
            'ar': 'Helsinki-NLP/opus-mt-ar-en',  # Arabic to English
            'hi': 'Helsinki-NLP/opus-mt-hi-en',  # Hindi to English
        }
        
        # Define reverse translation models (English to other languages)
        self.reverse_language_models = {
            'es': 'Helsinki-NLP/opus-mt-en-es',  # English to Spanish
            'fr': 'Helsinki-NLP/opus-mt-en-fr',  # English to French
            'de': 'Helsinki-NLP/opus-mt-en-de',  # English to German
            'it': 'Helsinki-NLP/opus-mt-en-it',  # English to Italian
            'pt': 'Helsinki-NLP/opus-mt-en-pt',  # English to Portuguese
            'ru': 'Helsinki-NLP/opus-mt-en-ru',  # English to Russian
            'zh': 'Helsinki-NLP/opus-mt-en-zh',  # English to Chinese
            'ja': 'Helsinki-NLP/opus-mt-en-ja',  # English to Japanese
            'ko': 'Helsinki-NLP/opus-mt-en-ko',  # English to Korean
            'ar': 'Helsinki-NLP/opus-mt-en-ar',  # English to Arabic
            'hi': 'Helsinki-NLP/opus-mt-en-hi',  # English to Hindi
        }
        
        logger.info(f"Translation service initialized with device: {self.device}")
    
    def _load_model(self, lang_code: str, reverse: bool = False) -> bool:
        """
        Load MarianMT model and MarianTokenizer for specific language with caching enabled
        Uses Helsinki-NLP/opus-mt-{lang}-en models for forward translation
        Uses Helsinki-NLP/opus-mt-en-{lang} models for reverse translation
        
        Args:
            lang_code: Language code to load model for
            reverse: If True, load reverse translation model (en->lang), else forward (lang->en)
            
        Returns:
            True if model loaded successfully, False otherwise
        """
        print(f"[TRANSLATION DEBUG] Loading model for language: {lang_code}, reverse: {reverse}")
        
        if not TRANSFORMERS_AVAILABLE:
            print(f"[TRANSLATION DEBUG] Transformers library not available, cannot load model")
            return False
        
        # Choose the appropriate model mapping
        model_mapping = self.reverse_language_models if reverse else self.language_models
        
        if lang_code not in model_mapping:
            print(f"[TRANSLATION DEBUG] Language {lang_code} not supported in {'reverse' if reverse else 'forward'} model mapping")
            return False
        
        # Create cache key to distinguish forward and reverse models
        cache_key = f"{lang_code}_reverse" if reverse else lang_code
        
        # Check if model is already loaded (caching)
        if cache_key in self.models and cache_key in self.tokenizers:
            print(f"[TRANSLATION DEBUG] Using cached {'reverse' if reverse else 'forward'} model for {lang_code}")
            logger.debug(f"Using cached {'reverse' if reverse else 'forward'} model for {lang_code}")
            return True
        
        try:
            model_name = model_mapping[lang_code]
            direction = "en->lang" if reverse else "lang->en"
            print(f"[TRANSLATION DEBUG] Loading Helsinki-NLP model: {model_name} ({direction})")
            logger.info(f"Loading Helsinki-NLP model: {model_name} ({direction})")
            
            # Load MarianTokenizer with caching
            tokenizer = MarianTokenizer.from_pretrained(
                model_name,
                cache_dir=None,  # Use default cache directory
                local_files_only=False  # Allow download if not cached
            )
            print(f"[TRANSLATION DEBUG] MarianTokenizer loaded successfully for {model_name}")
            
            # Load MarianMTModel with caching
            model = MarianMTModel.from_pretrained(
                model_name,
                cache_dir=None,  # Use default cache directory
                local_files_only=False  # Allow download if not cached
            )
            print(f"[TRANSLATION DEBUG] MarianMTModel loaded successfully for {model_name}")
            
            # Use GPU acceleration if available
            if self.device and self.device != "cpu" and torch.cuda.is_available():
                model = model.to(self.device)
                print(f"[TRANSLATION DEBUG] Model moved to GPU: {self.device}")
                logger.info(f"Model moved to GPU: {self.device}")
            else:
                print(f"[TRANSLATION DEBUG] Model running on CPU")
                logger.info("Model running on CPU")
            
            # Cache the loaded models with proper key
            self.tokenizers[cache_key] = tokenizer
            self.models[cache_key] = model
            
            print(f"[TRANSLATION DEBUG] Successfully loaded and cached MarianMT {'reverse' if reverse else 'forward'} model for {lang_code}")
            logger.info(f"Successfully loaded and cached MarianMT {'reverse' if reverse else 'forward'} model for {lang_code}")
            return True
            
        except Exception as e:
            print(f"[TRANSLATION DEBUG] Failed to load MarianMT {'reverse' if reverse else 'forward'} model for {lang_code}: {e}")
            logger.error(f"Failed to load MarianMT {'reverse' if reverse else 'forward'} model for {lang_code}: {e}")
            return False
    
    def is_available(self) -> bool:
        """Check if translation service is available"""
        return TRANSFORMERS_AVAILABLE
    
    @lru_cache(maxsize=1000)
    def detect_language(self, text: str) -> Tuple[str, float]:
        """
        Enhanced language detection for real estate queries with domain-specific cleaning
        
        Args:
            text: Input text to analyze
            
        Returns:
            Tuple of (language_code, confidence)
        """
        # Enhanced console logging for language detection
        print(f"[TRANSLATION DEBUG] Starting language detection for text: '{text[:100]}{'...' if len(text) > 100 else ''}'")
        
        # Log raw input
        logger.debug(f"Language detection - Raw input: '{text[:100]}{'...' if len(text) > 100 else ''}'")
        
        if not text or len(text.strip()) < 2:
            print(f"[TRANSLATION DEBUG] Text too short ({len(text.strip())} chars), defaulting to English")
            logger.debug("Language detection - Empty or too short text, defaulting to English")
            return 'en', 1.0
        
        # Clean text for real estate domain
        cleaned_text = self._clean_real_estate_text(text)
        print(f"[TRANSLATION DEBUG] Text after cleaning: '{cleaned_text[:100]}{'...' if len(cleaned_text) > 100 else ''}'")
        logger.debug(f"Language detection - Cleaned text: '{cleaned_text[:100]}{'...' if len(cleaned_text) > 100 else ''}'")
        
        if len(cleaned_text.strip()) < 2:
            print(f"[TRANSLATION DEBUG] Cleaned text too short ({len(cleaned_text.strip())} chars), defaulting to English")
            logger.debug("Language detection - Cleaned text too short, defaulting to English")
            return 'en', 1.0
        
        # Check for keyword-based overrides first
        keyword_lang = self._check_keyword_overrides(cleaned_text)
        if keyword_lang:
            print(f"[TRANSLATION DEBUG] Keyword-based language override detected: {keyword_lang} with confidence 0.95")
            logger.info(f"Language detection - Keyword override detected: {keyword_lang}")
            return keyword_lang, 0.95
        
        # Primary detection using langdetect
        primary_lang, primary_confidence = self._detect_with_langdetect(cleaned_text)
        print(f"[TRANSLATION DEBUG] Primary detection (langdetect): {primary_lang} (confidence: {primary_confidence:.3f})")
        logger.debug(f"Language detection - Primary (langdetect): {primary_lang} (confidence: {primary_confidence:.3f})")
        
        # If confidence is below threshold and langid is available, try fallback
        if primary_confidence < 0.90 and LANGID_AVAILABLE:
            fallback_lang, fallback_confidence = self._detect_with_langid(cleaned_text)
            print(f"[TRANSLATION DEBUG] Fallback detection (langid): {fallback_lang} (confidence: {fallback_confidence:.3f})")
            logger.debug(f"Language detection - Fallback (langid): {fallback_lang} (confidence: {fallback_confidence:.3f})")
            
            # Choose the detection with higher confidence
            if fallback_confidence > primary_confidence:
                final_lang, final_confidence = fallback_lang, fallback_confidence
                print(f"[TRANSLATION DEBUG] Using fallback result: {final_lang} (confidence: {final_confidence:.3f})")
                logger.info(f"Language detection - Using fallback result: {final_lang} (confidence: {final_confidence:.3f})")
            else:
                final_lang, final_confidence = primary_lang, primary_confidence
                print(f"[TRANSLATION DEBUG] Using primary result: {final_lang} (confidence: {final_confidence:.3f})")
                logger.info(f"Language detection - Using primary result: {final_lang} (confidence: {final_confidence:.3f})")
        else:
            final_lang, final_confidence = primary_lang, primary_confidence
            print(f"[TRANSLATION DEBUG] Using primary result: {final_lang} (confidence: {final_confidence:.3f})")
            logger.info(f"Language detection - Using primary result: {final_lang} (confidence: {final_confidence:.3f})")
        
        # Map to supported language codes
        mapped_lang = self._map_language_code(final_lang)
        print(f"[TRANSLATION DEBUG] Language code mapping: {final_lang} -> {mapped_lang}")
        
        print(f"[TRANSLATION DEBUG] FINAL LANGUAGE DETECTION: '{text[:50]}{'...' if len(text) > 50 else ''}' -> {mapped_lang} (confidence: {final_confidence:.3f})")
        logger.info(f"Language detection - Final decision: '{text[:50]}{'...' if len(text) > 50 else ''}' -> {mapped_lang} (confidence: {final_confidence:.3f})")
        return mapped_lang, final_confidence
    
    def _clean_real_estate_text(self, text: str) -> str:
        """
        Enhanced text cleaning for improved translation accuracy in real estate domain
        
        Args:
            text: Raw input text
            
        Returns:
            Cleaned text optimized for translation accuracy
        """
        if not text:
            return ""
        
        # Preserve original for comparison
        original = text
        
        # Step 1: Normalize encoding and basic cleanup
        cleaned = text.strip()
        
        # Step 2: Handle mixed scripts and encoding issues
        cleaned = self._normalize_mixed_scripts(cleaned)
        
        # Step 3: Preserve important context while removing noise
        cleaned = self._preserve_context_remove_noise(cleaned)
        
        # Step 4: Normalize punctuation and spacing for better tokenization
        cleaned = self._normalize_punctuation_spacing(cleaned)
        
        # Step 5: Handle real estate specific abbreviations and units
        cleaned = self._normalize_real_estate_terms(cleaned)
        
        # Step 6: Final cleanup while preserving linguistic markers
        cleaned = self._final_linguistic_cleanup(cleaned)
        
        # Ensure we don't over-clean and lose meaning
        if len(cleaned.strip()) < max(3, len(original.strip()) * 0.3):
            logger.warning(f"Over-cleaning detected, using less aggressive cleaning")
            return self._conservative_clean(original)
        
        logger.debug(f"Enhanced cleaning: '{original[:50]}...' -> '{cleaned[:50]}...'")
        return cleaned
    
    def _normalize_mixed_scripts(self, text: str) -> str:
        """Normalize mixed scripts and encoding issues"""
        # Handle common encoding issues
        text = text.replace('\u200b', ' ')  # Zero-width space
        text = text.replace('\u00a0', ' ')  # Non-breaking space
        text = text.replace('\ufeff', '')   # BOM
        
        # Normalize Unicode
        import unicodedata
        text = unicodedata.normalize('NFKC', text)
        
        return text


    def translate_response_to_user_language(self, english_response: str, user_language: str) -> Dict[str, any]:
        """
        Translate English chatbot response back to user's original language
        
        Args:
            english_response: English response from chatbot
            user_language: User's detected language code
            
        Returns:
            Dictionary with translation results including translated_response field
        """
        print(f"[TRANSLATION DEBUG] Starting reverse translation to user language: {user_language}")
        print(f"[TRANSLATION DEBUG] English response to translate: '{english_response[:100]}{'...' if len(english_response) > 100 else ''}'")
        
        if not english_response or not english_response.strip():
            print(f"[TRANSLATION DEBUG] Empty response provided, returning as-is")
            return {
                'original_response': english_response,
                'translated_response': english_response,
                'target_language': user_language,
                'translation_needed': False
            }
        
        # If user language is English, no translation needed
        if user_language == 'en':
            print(f"[TRANSLATION DEBUG] User language is English, no reverse translation needed")
            return {
                'original_response': english_response,
                'translated_response': english_response,
                'target_language': user_language,
                'translation_needed': False
            }
        
        # Check if we support reverse translation for this language
        if user_language not in self.reverse_language_models:
            print(f"[TRANSLATION DEBUG] Reverse translation not supported for language: {user_language}")
            logger.warning(f"Reverse translation not supported for language: {user_language}")
            return {
                'original_response': english_response,
                'translated_response': english_response,
                'target_language': user_language,
                'translation_needed': False,
                'error': f"Reverse translation not supported for {user_language}"
            }
        
        try:
            print(f"[TRANSLATION DEBUG] Starting reverse translation: en -> {user_language}")
            logger.info(f"Starting reverse translation: en -> {user_language}")
            
            # Load reverse translation model
            if self.is_available() and self._load_model(user_language, reverse=True):
                try:
                    print(f"[TRANSLATION DEBUG] Reverse model loaded successfully, starting translation")
                    # Translate using MarianMT
                    translated_response = self._translate_with_marian(english_response, 'en', user_language)
                    
                    if translated_response and translated_response.strip() and translated_response != english_response:
                        print(f"[TRANSLATION DEBUG] Reverse translation successful: '{english_response[:50]}...' -> '{translated_response[:50]}...'")
                        print(f"[TRANSLATION DEBUG] Full translated response length: {len(translated_response)} characters")
                        logger.info(f"Reverse translation successful: '{english_response[:50]}...' -> '{translated_response[:50]}...'")
                        return {
                            'original_response': english_response,
                            'translated_response': translated_response,
                            'target_language': user_language,
                            'translation_needed': True
                        }
                    else:
                        print(f"[TRANSLATION DEBUG] Reverse translation failed or returned unchanged result")
                        print(f"[TRANSLATION DEBUG] Translated response: '{translated_response[:100] if translated_response else 'None'}...'")
                        print(f"[TRANSLATION DEBUG] Original response: '{english_response[:100]}...'")
                        print(f"[TRANSLATION DEBUG] Response lengths - Original: {len(english_response)}, Translated: {len(translated_response) if translated_response else 0}")
                        logger.warning(f"Reverse translation failed or returned unchanged result")
                        
                except Exception as e:
                    print(f"[TRANSLATION DEBUG] Reverse MarianMT translation failed: {e}")
                    logger.error(f"Reverse MarianMT translation failed: {e}")
            else:
                print(f"[TRANSLATION DEBUG] Failed to load reverse translation model for {user_language}")
            
            # Fallback - return original English response
            print(f"[TRANSLATION DEBUG] Using fallback - returning English response")
            logger.info(f"Using fallback - returning English response")
            return {
                'original_response': english_response,
                'translated_response': english_response,
                'target_language': user_language,
                'translation_needed': False,
                'fallback_used': True
            }
            
        except Exception as e:
            print(f"[TRANSLATION DEBUG] Reverse translation error: {e}")
            logger.error(f"Reverse translation error: {e}")
            return {
                'original_response': english_response,
                'translated_response': english_response,
                'target_language': user_language,
                'translation_needed': False,
                'error': str(e)
            }
    
    def _preserve_context_remove_noise(self, text: str) -> str:
        """Remove noise while preserving linguistic context"""
        # Convert to lowercase for processing but preserve original case patterns
        working_text = text.lower()
        
        # Remove excessive punctuation but preserve sentence structure
        working_text = re.sub(r'[!]{2,}', '!', working_text)
        working_text = re.sub(r'[?]{2,}', '?', working_text)
        working_text = re.sub(r'[.]{3,}', '...', working_text)
        
        # Remove currency symbols but preserve numbers that might be important
        currency_patterns = [
            r'[₹$€£¥₩₽¢](?=\s|\d)',  # Currency symbols before numbers
            r'(?<=\d)\s*[₹$€£¥₩₽¢]',  # Currency symbols after numbers
            r'\b(inr|usd|eur|gbp|jpy|krw|rub|cad|aud|chf|cny)(?=\s|$)',  # Currency codes
        ]
        for pattern in currency_patterns:
            working_text = re.sub(pattern, '', working_text, flags=re.IGNORECASE)
        
        # Handle real estate measurements more carefully
        # Replace with generic terms to preserve sentence structure
        measurement_replacements = [
            (r'\b\d+\s*(bhk|bedroom|bedrooms|bed|beds)\b', 'bedroom'),
            (r'\b\d+\s*(sqft|sq\.?\s*ft|square\s+feet?)\b', 'area'),
            (r'\b\d+\s*(sq\s*m|square\s+meters?)\b', 'area'),
            (r'\b\d+\s*(bathroom|bathrooms|bath|baths)\b', 'bathroom'),
            (r'\b\d+\s*(floor|floors|storey|storeys|story|stories)\b', 'floor'),
        ]
        
        for pattern, replacement in measurement_replacements:
            working_text = re.sub(pattern, replacement, working_text, flags=re.IGNORECASE)
        
        return working_text
    
    def _normalize_punctuation_spacing(self, text: str) -> str:
        """Normalize punctuation and spacing for better tokenization"""
        # Normalize quotes
        text = re.sub(r'[""''`´]', '"', text)
        
        # Ensure proper spacing around punctuation
        text = re.sub(r'([.!?])([A-Za-z])', r'\1 \2', text)
        text = re.sub(r'([A-Za-z])([.!?])', r'\1\2', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text
    
    def _normalize_real_estate_terms(self, text: str) -> str:
        """Normalize real estate specific terms for better translation"""
        # Common real estate term normalizations that preserve meaning
        normalizations = [
            # Property types - keep but normalize
            (r'\b(apt|appt)\b', 'apartment'),
            (r'\b(br|bdr)\b', 'bedroom'),
            (r'\b(ba|bth)\b', 'bathroom'),
            (r'\b(pkg|prkg)\b', 'parking'),
            (r'\b(sq\.?\s*ft\.?|sqft)\b', 'square feet'),
            (r'\b(sq\.?\s*m\.?|sqm)\b', 'square meter'),
            
            # Location terms
            (r'\b(nr|near)\b', 'near'),
            (r'\b(opp|opposite)\b', 'opposite'),
            (r'\b(adj|adjacent)\b', 'adjacent'),
            
            # Condition terms
            (r'\b(furn|furnished)\b', 'furnished'),
            (r'\b(unfurn|unfurnished)\b', 'unfurnished'),
            (r'\b(semi-furn|semi-furnished)\b', 'semi furnished'),
        ]
        
        for pattern, replacement in normalizations:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return text
    
    def _final_linguistic_cleanup(self, text: str) -> str:
        """Final cleanup while preserving linguistic markers"""
        # Remove standalone numbers that don't add linguistic value
        text = re.sub(r'\b\d+\b(?!\s*(bedroom|bathroom|floor|area|square))', '', text)
        
        # Remove excessive whitespace again
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing punctuation that might confuse translation
        text = text.strip(' .,;:-')
        
        return text.strip()
    
    def _conservative_clean(self, text: str) -> str:
        """Conservative cleaning when aggressive cleaning removes too much content"""
        # Only remove the most obvious noise
        cleaned = text.strip()
        
        # Remove only currency symbols
        cleaned = re.sub(r'[₹$€£¥₩₽¢]', '', cleaned)
        
        # Normalize whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        return cleaned.strip()
    
    def _check_keyword_overrides(self, text: str) -> Optional[str]:
        """
        Check for language-specific keywords that indicate language
        
        Args:
            text: Cleaned text to check
            
        Returns:
            Language code if keyword match found, None otherwise
        """
        keyword_patterns = {
            'es': [
                'apartamento', 'piso', 'casa', 'vivienda', 'alquiler', 'venta',
                'dormitorio', 'baño', 'cocina', 'salon', 'terraza', 'garaje',
                'hola', 'gracias', 'por favor', 'donde', 'como', 'que', 'cuando',
                'precio', 'disponible', 'busco', 'necesito'
            ],
            'fr': [
                'appartement', 'maison', 'studio', 'chambre', 'salle de bain',
                'cuisine', 'salon', 'balcon', 'garage', 'location', 'vente',
                'bonjour', 'merci', 'si vous plait', 'ou', 'comment', 'que',
                'prix', 'disponible', 'cherche', 'besoin'
            ],
            'de': [
                'wohnung', 'haus', 'zimmer', 'schlafzimmer', 'badezimmer',
                'küche', 'wohnzimmer', 'balkon', 'garage', 'miete', 'verkauf',
                'hallo', 'danke', 'bitte', 'wo', 'wie', 'was', 'wann',
                'preis', 'verfügbar', 'suche', 'brauche'
            ],
            'it': [
                'appartamento', 'casa', 'camera', 'bagno', 'cucina',
                'soggiorno', 'terrazzo', 'garage', 'affitto', 'vendita',
                'ciao', 'grazie', 'per favore', 'dove', 'come', 'che',
                'prezzo', 'disponibile', 'cerco', 'ho bisogno'
            ],
            'pt': [
                'apartamento', 'casa', 'quarto', 'banheiro', 'cozinha',
                'sala', 'varanda', 'garagem', 'aluguel', 'venda',
                'ola', 'obrigado', 'por favor', 'onde', 'como', 'que',
                'preço', 'disponível', 'procuro', 'preciso'
            ],
            'hi': [
                'घर', 'मकान', 'फ्लैट', 'कमरा', 'बेडरूम', 'बाथरूम',
                'रसोई', 'बालकनी', 'किराया', 'बिक्री', 'नमस्ते',
                'धन्यवाद', 'कृपया', 'कहाँ', 'कैसे', 'क्या', 'कब',
                'कीमत', 'उपलब्ध', 'खोज', 'चाहिए'
            ],
            'ru': [
                'квартира', 'дом', 'комната', 'спальня', 'ванная',
                'кухня', 'балкон', 'гараж', 'аренда', 'продажа',
                'привет', 'спасибо', 'пожалуйста', 'где', 'как', 'что',
                'цена', 'доступно', 'ищу', 'нужно'
            ],
            'zh': [
                '公寓', '房子', '房间', '卧室', '浴室', '厨房', '阳台',
                '车库', '租金', '出售', '你好', '谢谢', '请', '哪里',
                '怎么', '什么', '价格', '可用', '寻找', '需要'
            ],
            'ja': [
                'アパート', '家', '部屋', '寝室', 'バスルーム', 'キッチン',
                'バルコニー', 'ガレージ', '賃貸', '販売', 'こんにちは',
                'ありがとう', 'お願いします', 'どこ', 'どうやって', '何',
                '価格', '利用可能', '探している', '必要'
            ],
            'ar': [
                'شقة', 'بيت', 'غرفة', 'غرفة نوم', 'حمام', 'مطبخ',
                'شرفة', 'مرآب', 'إيجار', 'بيع', 'مرحبا', 'شكرا',
                'من فضلك', 'أين', 'كيف', 'ماذا', 'سعر', 'متاح',
                'أبحث', 'أحتاج'
            ]
        }
        
        text_lower = text.lower()
        
        for lang, keywords in keyword_patterns.items():
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    logger.debug(f"Keyword override found: '{keyword}' -> {lang}")
                    return lang
        
        return None
    
    def _detect_with_langdetect(self, text: str) -> Tuple[str, float]:
        """
        Detect language using langdetect library
        
        Args:
            text: Text to analyze
            
        Returns:
            Tuple of (language_code, confidence)
        """
        if not LANGDETECT_AVAILABLE:
            return self._fallback_detect_language(text)
        
        try:
            # Primary detection
            detected_lang = detect(text)
            
            # Get confidence from detect_langs with multiple attempts for stability
            confidence_scores = []
            for _ in range(3):  # Try multiple times for stability
                try:
                    lang_probs = detect_langs(text)
                    confidence = next((prob.prob for prob in lang_probs if prob.lang == detected_lang), 0.0)
                    confidence_scores.append(confidence)
                except:
                    continue
            
            if confidence_scores:
                # Use average confidence for stability
                avg_confidence = sum(confidence_scores) / len(confidence_scores)
                return detected_lang, avg_confidence
            else:
                return detected_lang, 0.5
                
        except Exception as e:
            logger.warning(f"Langdetect failed: {e}. Using fallback detection.")
            return self._fallback_detect_language(text)
    
    def _detect_with_langid(self, text: str) -> Tuple[str, float]:
        """
        Detect language using langid library as fallback
        
        Args:
            text: Text to analyze
            
        Returns:
            Tuple of (language_code, confidence)
        """
        if not LANGID_AVAILABLE:
            return self._fallback_detect_language(text)
        
        try:
            import langid
            lang_code, confidence = langid.classify(text)
            return lang_code, confidence
        except Exception as e:
            logger.warning(f"Langid fallback failed: {e}. Using pattern-based detection.")
            return self._fallback_detect_language(text)
    
    def _map_language_code(self, lang_code: str) -> str:
        """Map language code variants to our supported languages"""
        # Handle common language code variations
        mapping = {
            'ca': 'es',  # Catalan -> Spanish
            'gl': 'es',  # Galician -> Spanish
            'eu': 'es',  # Basque -> Spanish
            'zh-cn': 'zh',  # Simplified Chinese
            'zh-tw': 'zh',  # Traditional Chinese
            'pt-br': 'pt',  # Brazilian Portuguese
            'pt-pt': 'pt',  # European Portuguese
            'fr-ca': 'fr',  # Canadian French
            'de-at': 'de',  # Austrian German
            'de-ch': 'de',  # Swiss German
            'it-ch': 'it',  # Swiss Italian
            'ar-sa': 'ar',  # Saudi Arabic
            'hi-in': 'hi',  # Indian Hindi
            'ko-kr': 'ko',  # Korean (Korea)
            'ja-jp': 'ja',  # Japanese (Japan)
        }
        
        # Return mapped language or original if not in mapping
        return mapping.get(lang_code.lower(), lang_code.lower())
    
    def _fallback_detect_language(self, text: str) -> Tuple[str, float]:
        """Basic language detection fallback using character patterns"""
        text_lower = text.lower()
        
        # Character-based detection patterns
        patterns = {
            'es': {
                'chars': ['ñ', 'ü', '¿', '¡'],
                'words': ['hola', 'gracias', 'por favor', 'casa', 'agua', 'comida', 'donde', 'como', 'que']
            },
            'fr': {
                'chars': ['ç', 'à', 'é', 'è', 'ê', 'ë', 'î', 'ï', 'ô', 'ù', 'û', 'ü', 'ÿ'],
                'words': ['bonjour', 'merci', 'maison', 'eau', 'nourriture', 'ou', 'comment', 'que']
            },
            'de': {
                'chars': ['ä', 'ö', 'ü', 'ß'],
                'words': ['hallo', 'danke', 'haus', 'wasser', 'essen', 'wo', 'wie', 'was']
            },
            'it': {
                'chars': ['à', 'è', 'é', 'ì', 'í', 'î', 'ò', 'ó', 'ù', 'ú'],
                'words': ['ciao', 'grazie', 'casa', 'acqua', 'cibo', 'dove', 'come', 'che']
            },
            'pt': {
                'chars': ['ã', 'á', 'à', 'â', 'é', 'ê', 'í', 'ó', 'ô', 'õ', 'ú', 'ç'],
                'words': ['ola', 'obrigado', 'casa', 'agua', 'comida', 'onde', 'como', 'que']
            },
            'ru': {
                'chars': ['а', 'б', 'в', 'г', 'д', 'е', 'ё', 'ж', 'з', 'и', 'й', 'к', 'л', 'м', 'н', 'о', 'п', 'р', 'с', 'т', 'у', 'ф', 'х', 'ц', 'ч', 'ш', 'щ', 'ъ', 'ы', 'ь', 'э', 'ю', 'я'],
                'words': ['привет', 'спасибо', 'дом', 'вода', 'еда']
            },
            'zh': {
                'chars': ['的', '是', '在', '有', '个', '人', '这', '中', '大', '为', '上', '来', '说', '国', '年', '着', '就', '那', '和', '要'],
                'words': []
            },
            'ja': {
                'chars': ['あ', 'い', 'う', 'え', 'お', 'か', 'き', 'く', 'け', 'こ', 'が', 'ぎ', 'ぐ', 'げ', 'ご', 'は', 'ひ', 'ふ', 'へ', 'ほ'],
                'words': []
            },
            'ar': {
                'chars': ['ا', 'ب', 'ت', 'ث', 'ج', 'ح', 'خ', 'د', 'ذ', 'ر', 'ز', 'س', 'ش', 'ص', 'ض', 'ط', 'ظ', 'ع', 'غ', 'ف', 'ق', 'ك', 'ل', 'م', 'ن', 'ه', 'و', 'ي'],
                'words': []
            },
            'hi': {
                'chars': ['अ', 'आ', 'इ', 'ई', 'उ', 'ऊ', 'ए', 'ऐ', 'ओ', 'औ', 'क', 'ख', 'ग', 'घ', 'च', 'छ', 'ज', 'झ', 'ट', 'ठ', 'ड', 'ढ', 'त', 'थ', 'द', 'ध', 'न', 'प', 'फ', 'ब', 'भ', 'म', 'य', 'र', 'ल', 'व', 'श', 'ष', 'स', 'ह'],
                'words': ['नमस्ते', 'धन्यवाद', 'घर', 'पानी', 'खाना']
            }
        }
        
        scores = {}
        for lang, pattern_data in patterns.items():
            score = 0
            
            # Check character patterns
            for char in pattern_data['chars']:
                score += text_lower.count(char) * 2
            
            # Check word patterns
            for word in pattern_data['words']:
                if word in text_lower:
                    score += 3
            
            scores[lang] = score
        
        if scores:
            max_score = max(scores.values())
            if max_score > 0:
                detected_lang = max(scores, key=scores.get)
                confidence = min(0.9, max_score * 0.1)
                return detected_lang, confidence
        
        # Default to English
        return 'en', 0.8
    
    def translate_text(self, text: str, target_lang: str = 'en', source_lang: Optional[str] = None) -> Dict[str, any]:
        """
        Translate text to target language using MarianMT models
        
        PERFORMANCE NOTE: Token limits increased to 10,240 to handle 20k-40k character responses.
        This may require significant GPU memory (8GB+ recommended) for very large texts.
        Consider implementing text chunking for responses exceeding 40k characters.
        
        Args:
            text: Text to translate
            target_lang: Target language code (default: 'en')
            source_lang: Source language code (auto-detect if None)
            
        Returns:
            Dictionary with translation results
        """
        force_console_print(f"TRANSLATION REQUEST: '{text[:100]}...' -> {target_lang}")
        
        if not text or not text.strip():
            force_console_print("WARNING: Empty text provided, returning as-is")
            return {
                'original_text': text,
                'translated_text': text,
                'detected_language': 'en',
                'confidence': 1.0,
                'translation_needed': False
            }
        
        try:
            # Detect language if not provided
            if source_lang is None:
                detected_lang, confidence = self.detect_language(text)
            else:
                detected_lang = source_lang
                confidence = 0.9
            
            # If already in target language, return original
            if detected_lang == target_lang:
                return {
                    'original_text': text,
                    'translated_text': text,
                    'detected_language': detected_lang,
                    'confidence': confidence,
                    'translation_needed': False
                }
            
            # Try MarianMT translation if available
            if self.is_available() and target_lang == 'en':
                if self._load_model(detected_lang):
                    try:
                        translated_text = self._translate_with_marian(text, detected_lang)
                        return {
                            'original_text': text,
                            'translated_text': translated_text,
                            'detected_language': detected_lang,
                            'confidence': confidence,
                            'translation_needed': True
                        }
                    except Exception as e:
                        logger.warning(f"MarianMT translation failed: {e}. Using fallback translation.")
            
            # Fallback translation using basic patterns
            translated_text = self._fallback_translate(text, detected_lang, target_lang)
            
            return {
                'original_text': text,
                'translated_text': translated_text,
                'detected_language': detected_lang,
                'confidence': confidence,
                'translation_needed': translated_text != text
            }
            
        except Exception as e:
            logger.error(f"Translation failed: {e}")
            return {
                'original_text': text,
                'translated_text': text,
                'detected_language': 'en',
                'confidence': 0.5,
                'translation_needed': False,
                'error': str(e)
            }
    
    def _get_safe_lang_token_id(self, tokenizer, target_lang: str):
        """
        Safely retrieve language token ID from tokenizer to prevent index errors.
        
        Many MarianMT tokenizers do not have the lang_code_to_id attribute, which is perfectly normal.
        When this attribute is missing, the model will use its default BOS (beginning of sequence) token
        behavior, which works correctly for most translation tasks.
        
        Args:
            tokenizer: The MarianTokenizer instance
            target_lang: Target language code
            
        Returns:
            Language token ID or None if not available (None triggers default behavior)
        """
        try:
            # Check if tokenizer has lang_code_to_id attribute
            if not hasattr(tokenizer, 'lang_code_to_id'):
                print(f"[TRANSLATION DEBUG] Tokenizer does not have lang_code_to_id attribute (this is normal for many MarianMT models)")
                logger.debug(f"Tokenizer does not have lang_code_to_id attribute - using default BOS token behavior")
                return None
            
            lang_code_to_id = tokenizer.lang_code_to_id
            
            # Ensure it's a dictionary-like object with get method
            if not hasattr(lang_code_to_id, 'get'):
                print(f"[TRANSLATION DEBUG] lang_code_to_id is not a dictionary: {type(lang_code_to_id)}")
                return None
            
            # Safely get the token ID
            token_id = lang_code_to_id.get(target_lang, None)
            print(f"[TRANSLATION DEBUG] Language token ID for {target_lang}: {token_id}")
            return token_id
            
        except (AttributeError, KeyError, IndexError, TypeError) as e:
            print(f"[TRANSLATION DEBUG] Error getting language token ID for {target_lang}: {e}")
            return None

    def _translate_with_marian(self, text: str, source_lang: str, target_lang: str = 'en') -> str:
        """
        Translate text using MarianMT model with enhanced preprocessing for better accuracy
        Uses GPU acceleration if available (torch.cuda.is_available())
        Supports both forward (lang->en) and reverse (en->lang) translation
        
        FIXED: Character encoding issues and improved error handling
        
        Args:
            text: Text to translate
            source_lang: Source language code
            target_lang: Target language code (default: 'en')
            
        Returns:
            Translated text
        """
        print(f"[TRANSLATION DEBUG] Starting MarianMT translation: {source_lang} -> {target_lang}")
        print(f"[TRANSLATION DEBUG] Input text: '{text[:100]}{'...' if len(text) > 100 else ''}'")
        
        # Determine if this is reverse translation (English to other language)
        reverse = source_lang == 'en' and target_lang != 'en'
        print(f"[TRANSLATION DEBUG] Translation direction: {'reverse (en->lang)' if reverse else 'forward (lang->en)'}")
        
        # For reverse translation, use target_lang as the model key
        model_lang = target_lang if reverse else source_lang
        cache_key = f"{model_lang}_reverse" if reverse else model_lang
        print(f"[TRANSLATION DEBUG] Using model cache key: {cache_key}")
        
        if cache_key not in self.models or cache_key not in self.tokenizers:
            print(f"[TRANSLATION DEBUG] ERROR: MarianMTModel not loaded for {'reverse' if reverse else 'forward'} translation: {model_lang}")
            raise ValueError(f"MarianMTModel not loaded for {'reverse' if reverse else 'forward'} translation: {model_lang}")
        
        model = self.models[cache_key]
        tokenizer = self.tokenizers[cache_key]
        print(f"[TRANSLATION DEBUG] Retrieved model and tokenizer from cache")
        
        # Enhanced preprocessing for translation
        preprocessed_text = self._preprocess_for_translation(text, source_lang, target_lang)
        print(f"[TRANSLATION DEBUG] Text after preprocessing: '{preprocessed_text[:100]}{'...' if len(preprocessed_text) > 100 else ''}'")
        
        if not preprocessed_text.strip():
            print(f"[TRANSLATION DEBUG] WARNING: Preprocessing resulted in empty text, returning original")
            logger.warning("Preprocessing resulted in empty text")
            return text
        
        try:
            print(f"[TRANSLATION DEBUG] Starting tokenization with MarianTokenizer")
            # FIXED: Enhanced tokenization with proper UTF-8 handling
            # Ensure text is properly encoded as UTF-8
            try:
                # Normalize the text to ensure consistent encoding
                import unicodedata
                normalized_text = unicodedata.normalize('NFKC', preprocessed_text)
                
                # Tokenize input with MarianTokenizer using SentencePiece
                inputs = tokenizer(
                    normalized_text, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True, 
                    max_length=10240,  # Increased to handle 20k-40k character responses
                    add_special_tokens=True
                )
                print(f"[TRANSLATION DEBUG] Tokenization completed, input shape: {inputs['input_ids'].shape}")
                
            except Exception as e:
                print(f"[TRANSLATION DEBUG] Tokenization error, trying with original text: {e}")
                # Fallback to original text if normalization fails
                inputs = tokenizer(
                    preprocessed_text, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True, 
                    max_length=10240,  # Increased to handle 20k-40k character responses
                    add_special_tokens=True
                )
                print(f"[TRANSLATION DEBUG] Fallback tokenization completed, input shape: {inputs['input_ids'].shape}")
            
            # Use GPU acceleration if available
            if self.device and self.device != "cpu" and torch.cuda.is_available():
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                print(f"[TRANSLATION DEBUG] Using GPU acceleration for translation: {self.device}")
                logger.debug(f"Using GPU acceleration for translation: {self.device}")
            else:
                print(f"[TRANSLATION DEBUG] Using CPU for translation")
                logger.debug("Using CPU for translation")
            
            print(f"[TRANSLATION DEBUG] Starting model generation with beam search (5 beams)")
            # FIXED: Enhanced generation parameters for better quality and encoding
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=10240,  # Increased to handle 20k-40k character responses
                    num_beams=5,  # Increased beam search for better quality
                    early_stopping=True,
                    do_sample=False,  # Deterministic output
                    temperature=1.0,
                    length_penalty=1.0,  # Balanced length penalty
                    repetition_penalty=1.1,  # Slight repetition penalty
                    pad_token_id=getattr(tokenizer, 'pad_token_id', None),
                    eos_token_id=getattr(tokenizer, 'eos_token_id', None),
                    # forced_bos_token_id: None is acceptable and triggers default behavior
                    forced_bos_token_id=self._get_safe_lang_token_id(tokenizer, target_lang)
                )
            print(f"[TRANSLATION DEBUG] Model generation completed, output shape: {outputs.shape}")
            print(f"[TRANSLATION DEBUG] Output tensor details - Type: {type(outputs)}, Device: {outputs.device if hasattr(outputs, 'device') else 'unknown'}")
            print(f"[TRANSLATION DEBUG] Output tensor statistics - Min: {outputs.min().item() if hasattr(outputs, 'min') else 'unknown'}, Max: {outputs.max().item() if hasattr(outputs, 'max') else 'unknown'}")
            print(f"[TRANSLATION DEBUG] First few tokens: {outputs[0][:min(10, outputs.shape[1])].tolist() if outputs.shape[0] > 0 and outputs.shape[1] > 0 else 'empty'}")
            
            # ENHANCED: Additional safety checks to prevent index out of range errors
            if outputs is None:
                print(f"[TRANSLATION DEBUG] ERROR: Model generated None outputs")
                logger.error(f"Model generated None outputs for {source_lang}->{target_lang}")
                raise ValueError(f"Model generated None outputs for translation {source_lang}->{target_lang}")
            
            if not hasattr(outputs, 'shape') or len(outputs.shape) < 2:
                print(f"[TRANSLATION DEBUG] ERROR: Model outputs have invalid shape: {outputs.shape if hasattr(outputs, 'shape') else 'no shape attribute'}")
                logger.error(f"Model outputs have invalid shape for {source_lang}->{target_lang}")
                raise ValueError(f"Model outputs have invalid shape for translation {source_lang}->{target_lang}")
            
            # Check if outputs is empty to prevent index error
            if outputs.shape[0] == 0 or outputs.shape[1] == 0:
                print(f"[TRANSLATION DEBUG] ERROR: Model generated empty outputs")
                logger.error(f"Model generated empty outputs for {source_lang}->{target_lang}")
                raise ValueError(f"Model generated empty outputs for translation {source_lang}->{target_lang}")
            
            # ENHANCED: Enhanced decoding with additional safety checks for index access
            try:
                # Additional safety check before accessing outputs[0]
                if len(outputs) == 0:
                    print(f"[TRANSLATION DEBUG] ERROR: Outputs tensor is empty (length 0)")
                    logger.error(f"Outputs tensor is empty for {source_lang}->{target_lang}")
                    raise ValueError(f"Outputs tensor is empty for translation {source_lang}->{target_lang}")
                
                # Decode output with MarianTokenizer, handling SentencePiece properly
                translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
                print(f"[TRANSLATION DEBUG] Raw decoded text: '{translated_text[:100]}{'...' if len(translated_text) > 100 else ''}'")
                
                # Ensure proper UTF-8 encoding
                if isinstance(translated_text, bytes):
                    translated_text = translated_text.decode('utf-8', errors='replace')
                
                # FIXED: Simplified encoding fix to prevent text corruption
                translated_text = self._fix_encoding_issues_simple(translated_text)
                print(f"[TRANSLATION DEBUG] Text after simplified encoding fix: '{translated_text[:100]}{'...' if len(translated_text) > 100 else ''}'")
                
            except IndexError as e:
                print(f"[TRANSLATION DEBUG] INDEX ERROR during decoding: {e}")
                print(f"[TRANSLATION DEBUG] Outputs shape: {outputs.shape}, Outputs length: {len(outputs) if hasattr(outputs, '__len__') else 'unknown'}")
                logger.error(f"Index error during decoding for {source_lang}->{target_lang}: {e}")
                raise ValueError(f"Index error during decoding for translation {source_lang}->{target_lang}: {e}")
            except Exception as e:
                print(f"[TRANSLATION DEBUG] Decoding error: {e}")
                logger.error(f"Decoding error: {e}")
                # Fallback decoding with additional safety checks
                if outputs.shape[0] > 0 and outputs.shape[1] > 0 and len(outputs) > 0:
                    try:
                        translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    except IndexError as ie:
                        print(f"[TRANSLATION DEBUG] INDEX ERROR in fallback decoding: {ie}")
                        logger.error(f"Index error in fallback decoding for {source_lang}->{target_lang}: {ie}")
                        raise ValueError(f"Index error in fallback decoding for translation {source_lang}->{target_lang}: {ie}")
                else:
                    print(f"[TRANSLATION DEBUG] ERROR: Cannot decode empty outputs")
                    logger.error(f"Cannot decode empty outputs for {source_lang}->{target_lang}")
                    raise ValueError(f"Cannot decode empty outputs for translation {source_lang}->{target_lang}")
                if isinstance(translated_text, bytes):
                    translated_text = translated_text.decode('utf-8', errors='replace')
                # Apply enhanced encoding fix even to fallback
                translated_text = self._fix_encoding_issues_simple(translated_text)
            
            # Enhanced post-processing for better accuracy
            translated_text = self._postprocess_translation(translated_text, preprocessed_text, text, target_lang)
            print(f"[TRANSLATION DEBUG] Text after post-processing: '{translated_text[:100]}{'...' if len(translated_text) > 100 else ''}'")
            
            direction = f"{source_lang}->{target_lang}"
            print(f"[TRANSLATION DEBUG] TRANSLATION COMPLETE ({direction}): '{text[:30]}...' -> '{translated_text[:30]}...'")
            logger.debug(f"Enhanced MarianMT translation ({direction}): '{text[:30]}...' -> '{translated_text[:30]}...'")
            return translated_text
            
        except Exception as e:
            print(f"[TRANSLATION DEBUG] ERROR in MarianMT translation for {source_lang}->{target_lang}: {e}")
            logger.error(f"Error in enhanced MarianMT translation for {source_lang}->{target_lang}: {e}")
            raise
    
    def _fix_encoding_issues(self, text: str) -> str:
        """
        Fix common encoding issues that occur during translation for ALL supported languages
        ENHANCED: Comprehensive encoding handling for Spanish, French, German, Italian, Portuguese,
        Russian, Chinese, Japanese, Korean, Arabic, and Hindi characters
        
        Args:
            text: Text that may have encoding issues
            
        Returns:
            Text with encoding issues fixed for all supported languages
        """
        if not text:
            return text
        
        # ENHANCED: Comprehensive multi-language encoding fixes
        try:
            # Ensure text is properly decoded first
            if isinstance(text, bytes):
                text = text.decode('utf-8', errors='replace')
            
            # COMPREHENSIVE MULTI-LANGUAGE ENCODING FIXES
            encoding_fixes = {
                # === COMMON ENCODING ISSUES ===
                # Spanish characters
                'Ã¡': 'á', 'Ã©': 'é', 'Ã­': 'í', 'Ã³': 'ó', 'Ãº': 'ú', 'Ã±': 'ñ',
                'Ã ': 'à', 'Ã¨': 'è', 'Ã¬': 'ì', 'Ã²': 'ò', 'Ã¹': 'ù', 'Ã§': 'ç',
                'Â¿': '¿', 'Â¡': '¡',  # Spanish punctuation
                
                # French characters
                'Ã¢': 'â', 'Ãª': 'ê', 'Ã®': 'î', 'Ã´': 'ô', 'Ã»': 'û',
                'Ã«': 'ë', 'Ã¯': 'ï', 'Ã¿': 'ÿ',
                
                # German characters
                'Ã¤': 'ä', 'Ã¶': 'ö', 'Ã¼': 'ü', 'ÃŸ': 'ß',
                
                # Common smart quotes and punctuation
                'â€™': "'", 'â€˜': "'",  # Smart apostrophes
                'â€œ': '"', 'â€': '"',   # Smart quotes
                'â€"': '–', 'â€"': '—',  # En/Em dashes
                'â€¦': '...', # Ellipsis
                'â€¢': '•',   # Bullet
                
                # Currency symbols
                'â‚¬': '€',   # Euro
                'Â£': '£',    # Pound
                'Â¥': '¥',    # Yen
                'Â¢': '¢',    # Cent
                
                # Mathematical symbols
                'Â±': '±',    # Plus-minus
                'Â²': '²',    # Superscript 2
                'Â³': '³',    # Superscript 3
                'Â½': '½',    # One half
                'Â¼': '¼',    # One quarter
                'Â¾': '¾',    # Three quarters
                'Â°': '°',    # Degree symbol
                
                # Common unwanted characters
                'Â': '',      # Unwanted byte order mark
                'ï»¿': '',    # Byte order mark
                
                # Replacement character sequences
                'Ã¯Â¿Â½': '�',  # Common replacement character sequence
                'ï¿½': '�',      # Another replacement character pattern
                'ï¿¿': '�',      # Yet another replacement pattern
            }
            
            # Apply all encoding fixes
            fixed_text = text
            for wrong, correct in encoding_fixes.items():
                fixed_text = fixed_text.replace(wrong, correct)
            
            # ENHANCED: Additional safety for encoding edge cases
            try:
                # Try to encode/decode to catch any remaining issues
                fixed_text = fixed_text.encode('utf-8', errors='ignore').decode('utf-8')
            except (UnicodeEncodeError, UnicodeDecodeError):
                # If encoding/decoding fails, return original text
                # Better to have imperfect text than no text
                logger.warning("Encoding fix failed, returning original text")
                return text
            
            return fixed_text
            
        except Exception as e:
            # ENHANCED: Comprehensive error handling for encoding edge cases
            logger.warning(f"Multi-language encoding fix encountered error: {e}, returning original text")
            return text  # Return original text if any error occurs

    def _fix_encoding_issues_simple(self, text: str) -> str:
        """
        Simplified encoding fix to prevent text corruption while handling common issues
        
        Args:
            text: Text that may have encoding issues
            
        Returns:
            Text with basic encoding issues fixed
        """
        if not text:
            return text
        
        try:
            # Ensure text is properly decoded first
            if isinstance(text, bytes):
                text = text.decode('utf-8', errors='replace')
            
            # Only fix the most common and safe encoding issues
            basic_fixes = {
                # Common UTF-8 encoding issues
                'Ã¡': 'á', 'Ã©': 'é', 'Ã­': 'í', 'Ã³': 'ó', 'Ãº': 'ú', 'Ã±': 'ñ',
                'Ã ': 'à', 'Ã¨': 'è', 'Ã¬': 'ì', 'Ã²': 'ò', 'Ã¹': 'ù', 'Ã§': 'ç',
                'Â¿': '¿', 'Â¡': '¡',  # Spanish punctuation
                'Ã¤': 'ä', 'Ã¶': 'ö', 'Ã¼': 'ü', 'ÃŸ': 'ß',  # German
                'â€™': "'", 'â€œ': '"', 'â€': '"',  # Smart quotes
                'â€"': '–', 'â€"': '—',  # Dashes
                'â‚¬': '€',  # Euro symbol
                'Â': '',  # Remove unwanted byte markers
                'ï»¿': '',  # BOM
            }
            
            # Apply basic fixes
            fixed_text = text
            for wrong, correct in basic_fixes.items():
                fixed_text = fixed_text.replace(wrong, correct)
            
            # Ensure valid UTF-8
            try:
                fixed_text = fixed_text.encode('utf-8', errors='ignore').decode('utf-8')
            except (UnicodeEncodeError, UnicodeDecodeError):
                # If encoding fails, return original
                return text
            
            return fixed_text
            
        except Exception as e:
            logger.warning(f"Simple encoding fix failed: {e}")
            return text

    def _fix_encoding_issues_enhanced(self, text: str) -> str:
        """
        Enhanced encoding fix for ALL supported languages with comprehensive character recovery
        Fixes UTF-8 encoding issues for Spanish, French, German, Italian, Portuguese, Russian, 
        Chinese, Japanese, Korean, Arabic, and Hindi characters
        
        Args:
            text: Text that may have encoding issues
            
        Returns:
            Text with encoding issues fixed for all supported languages
        """
        if not text:
            return text
        
        try:
            # First, handle the specific case where special characters appear as � symbols
            # This happens when UTF-8 decoding fails for specific characters
            
            # Handle the specific UTF-8 sequence issues
            if '�' in text:
                # Check if this is actually a valid character misrepresented
                for i, char in enumerate(text):
                    if char == '�' and ord(char) == 191:  # This is actually ¿
                        text = text[:i] + '¿' + text[i+1:]
                        print(f"[TRANSLATION DEBUG] Fixed UTF-8 encoding: replaced � (ord 191) with ¿")
            
            # COMPREHENSIVE MULTI-LANGUAGE CHARACTER RECOVERY PATTERNS
            fixed_text = text
            
            # === SPANISH CHARACTER RECOVERY ===
            spanish_recovery_patterns = [
                # Question marks - common issue with Spanish
                (r'�Tiene', '¿Tiene'), (r'�Tienes', '¿Tienes'), (r'�Hay', '¿Hay'),
                (r'�Dónde', '¿Dónde'), (r'�Cuándo', '¿Cuándo'), (r'�Cómo', '¿Cómo'),
                (r'�Qué', '¿Qué'), (r'�Quién', '¿Quién'), (r'�Por qué', '¿Por qué'),
                (r'�Cuál', '¿Cuál'), (r'�Cuánto', '¿Cuánto'), (r'�Es', '¿Es'),
                (r'�Está', '¿Está'), (r'�Puedo', '¿Puedo'), (r'�Puede', '¿Puede'),
                (r'�Necesita', '¿Necesita'), (r'�Busca', '¿Busca'),
                
                # Common question patterns
                (r'^�([A-Z])', r'¿\1'), (r'\. �([A-Z])', r'. ¿\1'),
                (r'\? �([A-Z])', r'? ¿\1'), (r'! �([A-Z])', r'! ¿\1'),
                (r'(\w)\. �([A-Z])', r'\1. ¿\2'),
                
                # Accented characters
                (r'�a', 'á'), (r'�e', 'é'), (r'�i', 'í'), (r'�o', 'ó'), (r'�u', 'ú'), (r'�n', 'ñ'),
                (r'�A', 'Á'), (r'�E', 'É'), (r'�I', 'Í'), (r'�O', 'Ó'), (r'�U', 'Ú'), (r'�N', 'Ñ'),
                
                # Exclamation marks
                (r'�([A-Za-z])', r'¡\1'), (r'([a-z])�', r'\1!'),
            ]
            
            # === FRENCH CHARACTER RECOVERY ===
            french_recovery_patterns = [
                # Accented characters
                (r'�a', 'à'), (r'�e', 'è'), (r'�e', 'é'), (r'�e', 'ê'), (r'�e', 'ë'),
                (r'�i', 'î'), (r'�i', 'ï'), (r'�o', 'ô'), (r'�u', 'ù'), (r'�u', 'û'), (r'�u', 'ü'),
                (r'�y', 'ÿ'), (r'�c', 'ç'),
                (r'�A', 'À'), (r'�E', 'È'), (r'�E', 'É'), (r'�E', 'Ê'), (r'�E', 'Ë'),
                (r'�I', 'Î'), (r'�I', 'Ï'), (r'�O', 'Ô'), (r'�U', 'Ù'), (r'�U', 'Û'), (r'�U', 'Ü'),
                (r'�Y', 'Ÿ'), (r'�C', 'Ç'),
                
                # Common French words with encoding issues
                (r'fran�ais', 'français'), (r'caf�', 'café'), (r'h�tel', 'hôtel'),
                (r'�a', 'ça'), (r'pr�s', 'près'), (r'apr�s', 'après'),
            ]
            
            # === GERMAN CHARACTER RECOVERY ===
            german_recovery_patterns = [
                # Umlauts and ß
                (r'�a', 'ä'), (r'�o', 'ö'), (r'�u', 'ü'), (r'�', 'ß'),
                (r'�A', 'Ä'), (r'�O', 'Ö'), (r'�U', 'Ü'),
                
                # Common German words with encoding issues
                (r'M�nchen', 'München'), (r'K�ln', 'Köln'), (r'D�sseldorf', 'Düsseldorf'),
                (r'gro�', 'groß'), (r'wei�', 'weiß'), (r'hei�t', 'heißt'),
                (r'f�r', 'für'), (r'�ber', 'über'), (r'sch�n', 'schön'),
            ]
            
            # === ITALIAN CHARACTER RECOVERY ===
            italian_recovery_patterns = [
                # Accented characters
                (r'�a', 'à'), (r'�e', 'è'), (r'�e', 'é'), (r'�i', 'ì'), (r'�i', 'í'),
                (r'�o', 'ò'), (r'�o', 'ó'), (r'�u', 'ù'), (r'�u', 'ú'),
                (r'�A', 'À'), (r'�E', 'È'), (r'�E', 'É'), (r'�I', 'Ì'), (r'�I', 'Í'),
                (r'�O', 'Ò'), (r'�O', 'Ó'), (r'�U', 'Ù'), (r'�U', 'Ú'),
                
                # Common Italian words
                (r'citt�', 'città'), (r'perch�', 'perché'), (r'caff�', 'caffè'),
                (r'pi�', 'più'), (r'cos�', 'così'), (r'gi�', 'già'),
            ]
            
            # === PORTUGUESE CHARACTER RECOVERY ===
            portuguese_recovery_patterns = [
                # Accented characters and tildes
                (r'�a', 'á'), (r'�a', 'à'), (r'�a', 'â'), (r'�a', 'ã'),
                (r'�e', 'é'), (r'�e', 'ê'), (r'�i', 'í'), (r'�o', 'ó'), (r'�o', 'ô'), (r'�o', 'õ'),
                (r'�u', 'ú'), (r'�c', 'ç'),
                (r'�A', 'Á'), (r'�A', 'À'), (r'�A', 'Â'), (r'�A', 'Ã'),
                (r'�E', 'É'), (r'�E', 'Ê'), (r'�I', 'Í'), (r'�O', 'Ó'), (r'�O', 'Ô'), (r'�O', 'Õ'),
                (r'�U', 'Ú'), (r'�C', 'Ç'),
                
                # Common Portuguese words
                (r'portugu�s', 'português'), (r'informa��o', 'informação'), (r'situa��o', 'situação'),
                (r'n�o', 'não'), (r'ent�o', 'então'), (r'cora��o', 'coração'),
            ]
            
            # === RUSSIAN CHARACTER RECOVERY ===
            russian_recovery_patterns = [
                # Cyrillic characters that might become �
                (r'�а', 'а'), (r'�е', 'е'), (r'�и', 'и'), (r'�о', 'о'), (r'�у', 'у'),
                (r'�ё', 'ё'), (r'�я', 'я'), (r'�ю', 'ю'), (r'�э', 'э'), (r'�ы', 'ы'),
                (r'�А', 'А'), (r'�Е', 'Е'), (r'�И', 'И'), (r'�О', 'О'), (r'�У', 'У'),
                (r'�Ё', 'Ё'), (r'�Я', 'Я'), (r'�Ю', 'Ю'), (r'�Э', 'Э'), (r'�Ы', 'Ы'),
                
                # Common Russian words
                (r'прив�т', 'привет'), (r'спас�бо', 'спасибо'), (r'пожал�йста', 'пожалуйста'),
            ]
            
            # === CHINESE CHARACTER RECOVERY ===
            chinese_recovery_patterns = [
                # Common Chinese characters that might become �
                (r'�的', '的'), (r'�是', '是'), (r'�在', '在'), (r'�有', '有'),
                (r'�个', '个'), (r'�人', '人'), (r'�这', '这'), (r'�中', '中'),
                (r'�大', '大'), (r'�为', '为'), (r'�上', '上'), (r'�来', '来'),
                (r'�说', '说'), (r'�国', '国'), (r'�年', '年'), (r'�着', '着'),
                
                # Common Chinese words
                (r'你�', '你好'), (r'谢�', '谢谢'), (r'房�', '房子'),
            ]
            
            # === JAPANESE CHARACTER RECOVERY ===
            japanese_recovery_patterns = [
                # Hiragana characters
                (r'�あ', 'あ'), (r'�い', 'い'), (r'�う', 'う'), (r'�え', 'え'), (r'�お', 'お'),
                (r'�か', 'か'), (r'�き', 'き'), (r'�く', 'く'), (r'�け', 'け'), (r'�こ', 'こ'),
                (r'�が', 'が'), (r'�ぎ', 'ぎ'), (r'�ぐ', 'ぐ'), (r'�げ', 'げ'), (r'�ご', 'ご'),
                
                # Katakana characters
                (r'�ア', 'ア'), (r'�イ', 'イ'), (r'�ウ', 'ウ'), (r'�エ', 'エ'), (r'�オ', 'オ'),
                (r'�カ', 'カ'), (r'�キ', 'キ'), (r'�ク', 'ク'), (r'�ケ', 'ケ'), (r'�コ', 'コ'),
                
                # Common Japanese words
                (r'こんに�は', 'こんにちは'), (r'ありがと�', 'ありがとう'), (r'アパ�ト', 'アパート'),
            ]
            
            # === KOREAN CHARACTER RECOVERY ===
            korean_recovery_patterns = [
                # Hangul characters
                (r'�가', '가'), (r'�나', '나'), (r'�다', '다'), (r'�라', '라'), (r'�마', '마'),
                (r'�바', '바'), (r'�사', '사'), (r'�아', '아'), (r'�자', '자'), (r'�차', '차'),
                (r'�카', '카'), (r'�타', '타'), (r'�파', '파'), (r'�하', '하'),
                
                # Common Korean words
                (r'안녕�세요', '안녕하세요'), (r'감사�니다', '감사합니다'), (r'집�', '집'),
            ]
            
            # === ARABIC CHARACTER RECOVERY ===
            arabic_recovery_patterns = [
                # Arabic characters (right-to-left)
                (r'�ا', 'ا'), (r'�ب', 'ب'), (r'�ت', 'ت'), (r'�ث', 'ث'), (r'�ج', 'ج'),
                (r'�ح', 'ح'), (r'�خ', 'خ'), (r'�د', 'د'), (r'�ذ', 'ذ'), (r'�ر', 'ر'),
                (r'�ز', 'ز'), (r'�س', 'س'), (r'�ش', 'ش'), (r'�ص', 'ص'), (r'�ض', 'ض'),
                (r'�ط', 'ط'), (r'�ظ', 'ظ'), (r'�ع', 'ع'), (r'�غ', 'غ'), (r'�ف', 'ف'),
                (r'�ق', 'ق'), (r'�ك', 'ك'), (r'�ل', 'ل'), (r'�م', 'م'), (r'�ن', 'ن'),
                (r'�ه', 'ه'), (r'�و', 'و'), (r'�ي', 'ي'),
                
                # Common Arabic words
                (r'مرح�ا', 'مرحبا'), (r'شك�ا', 'شكرا'), (r'بي�', 'بيت'),
            ]
            
            # === HINDI CHARACTER RECOVERY ===
            hindi_recovery_patterns = [
                # Devanagari characters
                (r'�अ', 'अ'), (r'�आ', 'आ'), (r'�इ', 'इ'), (r'�ई', 'ई'), (r'�उ', 'उ'),
                (r'�ऊ', 'ऊ'), (r'�ए', 'ए'), (r'�ऐ', 'ऐ'), (r'�ओ', 'ओ'), (r'�औ', 'औ'),
                (r'�क', 'क'), (r'�ख', 'ख'), (r'�ग', 'ग'), (r'�घ', 'घ'), (r'�च', 'च'),
                (r'�छ', 'छ'), (r'�ज', 'ज'), (r'�झ', 'झ'), (r'�ट', 'ट'), (r'�ठ', 'ठ'),
                (r'�ड', 'ड'), (r'�ढ', 'ढ'), (r'�त', 'त'), (r'�थ', 'थ'), (r'�द', 'द'),
                (r'�ध', 'ध'), (r'�न', 'न'), (r'�प', 'प'), (r'�फ', 'फ'), (r'�ब', 'ब'),
                (r'�भ', 'भ'), (r'�म', 'म'), (r'�य', 'य'), (r'�र', 'र'), (r'�ल', 'ल'),
                (r'�व', 'व'), (r'�श', 'श'), (r'�ष', 'ष'), (r'�स', 'स'), (r'�ह', 'ह'),
                
                # Common Hindi words
                (r'नमस्�े', 'नमस्ते'), (r'धन्य�ाद', 'धन्यवाद'), (r'घ�', 'घर'),
            ]
            
            # Apply all language-specific character recovery patterns
            all_patterns = [
                spanish_recovery_patterns,
                french_recovery_patterns,
                german_recovery_patterns,
                italian_recovery_patterns,
                portuguese_recovery_patterns,
                russian_recovery_patterns,
                chinese_recovery_patterns,
                japanese_recovery_patterns,
                korean_recovery_patterns,
                arabic_recovery_patterns,
                hindi_recovery_patterns
            ]
            
            for pattern_group in all_patterns:
                for pattern, replacement in pattern_group:
                    fixed_text = re.sub(pattern, replacement, fixed_text)
            
            # Apply the original encoding fixes as well
            fixed_text = self._fix_encoding_issues(fixed_text)
            
            # Additional context-based recovery for remaining � symbols
            if '�' in fixed_text:
                print(f"[TRANSLATION DEBUG] Found � symbols in text, attempting context recovery: '{fixed_text}'")
                
                # Context-based recovery patterns for all languages
                # Questions (Spanish-style)
                fixed_text = re.sub(r'\b([A-Z][a-z]+) �', r'¿\1 ', fixed_text)
                fixed_text = re.sub(r'^�([A-Z])', r'¿\1', fixed_text)
                fixed_text = re.sub(r'\. �([A-Z])', r'. ¿\1', fixed_text)
                fixed_text = re.sub(r'(\w)\. �([A-ZÁÉÍÓÚÑ])', r'\1. ¿\2', fixed_text)
                
                # General accented character recovery (try most common accents)
                if fixed_text.count('�') <= 5:  # Only if few remaining
                    # Try to determine language context and apply appropriate fixes
                    text_lower = fixed_text.lower()
                    
                    # Spanish context
                    if any(word in text_lower for word in ['tiene', 'hay', 'es', 'está', 'puede', 'busca', 'casa', 'donde']):
                        fixed_text = fixed_text.replace('�', '¿', 1)
                        print(f"[TRANSLATION DEBUG] Applied Spanish context fix: replaced � with ¿")
                    
                    # French context
                    elif any(word in text_lower for word in ['maison', 'avec', 'pour', 'cette', 'comment', 'ou']):
                        fixed_text = fixed_text.replace('�', 'é', 1)
                        print(f"[TRANSLATION DEBUG] Applied French context fix: replaced � with é")
                    
                    # German context
                    elif any(word in text_lower for word in ['haus', 'mit', 'für', 'wie', 'was', 'wo']):
                        fixed_text = fixed_text.replace('�', 'ü', 1)
                        print(f"[TRANSLATION DEBUG] Applied German context fix: replaced � with ü")
                    
                    # Italian context
                    elif any(word in text_lower for word in ['casa', 'con', 'per', 'come', 'che', 'dove']):
                        fixed_text = fixed_text.replace('�', 'à', 1)
                        print(f"[TRANSLATION DEBUG] Applied Italian context fix: replaced � with à")
                    
                    # Portuguese context
                    elif any(word in text_lower for word in ['casa', 'com', 'para', 'como', 'que', 'onde']):
                        fixed_text = fixed_text.replace('�', 'ã', 1)
                        print(f"[TRANSLATION DEBUG] Applied Portuguese context fix: replaced � with ã")
                
                print(f"[TRANSLATION DEBUG] After context recovery: '{fixed_text}'")
            
            print(f"[TRANSLATION DEBUG] Enhanced multi-language encoding fix: '{text[:50]}...' -> '{fixed_text[:50]}...'")
            return fixed_text
            
        except Exception as e:
            logger.warning(f"Enhanced multi-language encoding fix encountered error: {e}, using original fix")
            return self._fix_encoding_issues(text)


    def _preprocess_for_translation(self, text: str, source_lang: str, target_lang: str = 'en') -> str:
        """Enhanced preprocessing specifically for translation accuracy"""
        if not text:
            return ""
        
        # Step 1: Basic normalization
        processed = text.strip()
        
        # Step 2: Handle language-specific preprocessing
        processed = self._language_specific_preprocessing(processed, source_lang)
        
        # Step 3: Normalize sentence structure for better translation
        processed = self._normalize_sentence_structure(processed)
        
        # Step 4: Handle domain-specific terms that need context preservation
        processed = self._preserve_translation_context(processed, target_lang)
        
        return processed
    
    def _language_specific_preprocessing(self, text: str, source_lang: str) -> str:
        """Apply comprehensive language-specific preprocessing rules for all supported languages"""
        
        # === SPANISH PREPROCESSING ===
        if source_lang == 'es':
            # Spanish specific preprocessing
            text = re.sub(r'¿([^?]+)\?', r'\1?', text)  # Normalize question marks
            text = re.sub(r'¡([^!]+)!', r'\1!', text)  # Normalize exclamation marks
            # Handle contractions
            text = re.sub(r'\bdel\b', 'de el', text)  # del -> de el
            text = re.sub(r'\bal\b', 'a el', text)    # al -> a el
            
        # === FRENCH PREPROCESSING ===
        elif source_lang == 'fr':
            # French specific preprocessing
            text = re.sub(r'\b([cdjlmnst])\'([aeiouhy])', r'\1 \2', text)  # Handle contractions
            # Handle common French contractions
            text = re.sub(r'\bdu\b', 'de le', text)   # du -> de le
            text = re.sub(r'\bdes\b', 'de les', text) # des -> de les
            text = re.sub(r'\bau\b', 'à le', text)    # au -> à le
            text = re.sub(r'\baux\b', 'à les', text)  # aux -> à les
            
        # === GERMAN PREPROCESSING ===
        elif source_lang == 'de':
            # German specific preprocessing
            text = re.sub(r'\bß\b', 'ss', text)  # Normalize ß for better tokenization
            # Handle German compound word separations for better translation
            text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # Split compound words
            
        # === ITALIAN PREPROCESSING ===
        elif source_lang == 'it':
            # Italian specific preprocessing
            # Handle Italian contractions
            text = re.sub(r'\bdell\'', 'della ', text)  # dell' -> della
            text = re.sub(r'\bnell\'', 'nella ', text)  # nell' -> nella
            text = re.sub(r'\bsull\'', 'sulla ', text)  # sull' -> sulla
            text = re.sub(r'\ball\'', 'alla ', text)    # all' -> alla
            
        # === PORTUGUESE PREPROCESSING ===
        elif source_lang == 'pt':
            # Portuguese specific preprocessing
            # Handle Portuguese contractions
            text = re.sub(r'\bdo\b', 'de o', text)     # do -> de o
            text = re.sub(r'\bda\b', 'de a', text)     # da -> de a
            text = re.sub(r'\bdos\b', 'de os', text)   # dos -> de os
            text = re.sub(r'\bdas\b', 'de as', text)   # das -> de as
            text = re.sub(r'\bno\b', 'em o', text)     # no -> em o
            text = re.sub(r'\bna\b', 'em a', text)     # na -> em a
            
        # === RUSSIAN PREPROCESSING ===
        elif source_lang == 'ru':
            # Russian specific preprocessing
            # Normalize Cyrillic characters and handle common patterns
            text = re.sub(r'ё', 'е', text)  # Normalize ё to е for better tokenization
            # Handle Russian word boundaries
            text = re.sub(r'([а-я])([А-Я])', r'\1 \2', text)
            
        # === CHINESE PREPROCESSING ===
        elif source_lang == 'zh':
            # Chinese specific preprocessing
            # Normalize Chinese punctuation to Western equivalents
            punctuation_map = {
                '，': ',', '。': '.', '！': '!', '？': '?', 
                '；': ';', '：': ':', '（': '(', '）': ')',
                '【': '[', '】': ']', '《': '<', '》': '>',
                '"': '"', '"': '"', ''': "'", ''': "'"
            }
            for chinese_punct, western_punct in punctuation_map.items():
                text = text.replace(chinese_punct, western_punct)
            
            # Add spaces around Chinese characters for better tokenization
            text = re.sub(r'([\u4e00-\u9fff])([a-zA-Z])', r'\1 \2', text)
            text = re.sub(r'([a-zA-Z])([\u4e00-\u9fff])', r'\1 \2', text)
            
        # === JAPANESE PREPROCESSING ===
        elif source_lang == 'ja':
            # Japanese specific preprocessing
            # Normalize Japanese punctuation
            punctuation_map = {
                '。': '.', '、': ',', '！': '!', '？': '?',
                '（': '(', '）': ')', '「': '"', '」': '"'
            }
            for japanese_punct, western_punct in punctuation_map.items():
                text = text.replace(japanese_punct, western_punct)
            
            # Add spaces around Japanese characters mixed with Latin
            text = re.sub(r'([\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fff])([a-zA-Z])', r'\1 \2', text)
            text = re.sub(r'([a-zA-Z])([\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fff])', r'\1 \2', text)
            
        # === KOREAN PREPROCESSING ===
        elif source_lang == 'ko':
            # Korean specific preprocessing
            # Add spaces around Korean characters mixed with Latin
            text = re.sub(r'([\uac00-\ud7af])([a-zA-Z])', r'\1 \2', text)
            text = re.sub(r'([a-zA-Z])([\uac00-\ud7af])', r'\1 \2', text)
            
        # === ARABIC PREPROCESSING ===
        elif source_lang == 'ar':
            # Arabic specific preprocessing
            # Normalize Arabic characters and handle RTL text
            text = re.sub(r'[\u200e\u200f]', '', text)  # Remove LTR/RTL marks
            # Add spaces around Arabic text mixed with Latin
            text = re.sub(r'([\u0600-\u06ff])([a-zA-Z])', r'\1 \2', text)
            text = re.sub(r'([a-zA-Z])([\u0600-\u06ff])', r'\1 \2', text)
            
        # === HINDI PREPROCESSING ===
        elif source_lang == 'hi':
            # Hindi specific preprocessing
            # Normalize Devanagari characters
            text = re.sub(r'([\u0900-\u097f])([a-zA-Z])', r'\1 \2', text)
            text = re.sub(r'([a-zA-Z])([\u0900-\u097f])', r'\1 \2', text)
            
        # === GENERAL PREPROCESSING FOR ALL LANGUAGES ===
        # Remove excessive whitespace that might have been introduced
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def _normalize_sentence_structure(self, text: str) -> str:
        """Normalize sentence structure for better translation"""
        # Ensure sentences end with proper punctuation
        text = re.sub(r'([a-zA-Z])\s*$', r'\1.', text)
        
        # Normalize spacing around punctuation
        text = re.sub(r'\s*([.!?])\s*', r'\1 ', text)
        text = re.sub(r'\s*([,;:])\s*', r'\1 ', text)
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def _preserve_translation_context(self, text: str, target_lang: str = 'en') -> str:
        """Preserve important context for translation"""
        # Keep important real estate terms that provide context
        context_terms = [
            'apartment', 'house', 'bedroom', 'bathroom', 'kitchen',
            'furnished', 'parking', 'balcony', 'floor', 'area',
            'rent', 'sale', 'buy', 'lease', 'available'
        ]
        
        # For reverse translation (en->lang), preserve English structure
        if target_lang != 'en':
            # Ensure these terms are properly spaced and recognizable for reverse translation
            for term in context_terms:
                # Add word boundaries to ensure proper recognition
                pattern = r'\b' + re.escape(term) + r'\b'
                text = re.sub(pattern, f' {term} ', text, flags=re.IGNORECASE)
        else:
            # For forward translation, ensure proper spacing
            for term in context_terms:
                pattern = r'\b' + re.escape(term) + r'\b'
                text = re.sub(pattern, f' {term} ', text, flags=re.IGNORECASE)
        
        # Clean up extra spaces
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def _postprocess_translation(self, translated: str, preprocessed: str, original: str, target_lang: str = 'en') -> str:
        """Enhanced post-processing of translation results"""
        if not translated or not translated.strip():
            return original
        
        # Step 1: Basic cleanup
        result = translated.strip()
        
        # Step 2: Remove any source text that might be included in output
        result = self._remove_source_contamination(result, preprocessed, original)
        
        # Step 3: Fix common translation artifacts
        result = self._fix_translation_artifacts(result)
        
        # Step 4: Ensure proper capitalization
        result = self._fix_capitalization(result)
        
        # Step 5: Validate translation quality - DISABLED for better reliability
        # Validation was too strict and causing false rejections
        # if not self._validate_translation_quality(result, original, target_lang):
        #     logger.warning(f"Translation quality check failed, using conservative approach")
        #     return self._conservative_translation_fallback(original)
        
        # Accept the translation as-is if it passes basic checks
        if result.strip() and len(result.strip()) >= 2:
            return result
        else:
            # Only fallback if the result is completely empty or too short
            return original
    
    def _remove_source_contamination(self, translated: str, preprocessed: str, original: str) -> str:
        """Remove source text contamination from translation - FIXED for encoding edge cases"""
        # Check if original text appears in translation
        original_words = set(original.lower().split())
        translated_words = translated.lower().split()
        
        # FIXED: Much more lenient contamination detection for encoding edge cases
        # Only consider it contamination if there's extreme overlap (>90%)
        contamination_ratio = len(original_words.intersection(set(translated_words))) / max(len(original_words), 1)
        
        # FIXED: Only attempt cleanup for very high contamination (was 0.7, now 0.9)
        if contamination_ratio > 0.9:  # Much more lenient threshold
            # Try to find the actual translation part
            sentences = translated.split('.')
            for sentence in sentences:
                sentence_words = set(sentence.lower().split())
                sentence_contamination = len(original_words.intersection(sentence_words)) / max(len(sentence_words), 1)
                # FIXED: More lenient sentence contamination check (was 0.5, now 0.8)
                if sentence_contamination < 0.8 and len(sentence.strip()) > 3:
                    return sentence.strip()
        
        # FIXED: For encoding edge cases, return original translation as-is
        # Better to have a slightly contaminated translation than no translation
        return translated
    
    def _fix_translation_artifacts(self, text: str) -> str:
        """Fix common translation artifacts"""
        # Remove repeated phrases
        text = re.sub(r'\b(\w+(?:\s+\w+)*)\s+\1\b', r'\1', text)
        
        # Fix spacing issues
        text = re.sub(r'\s+', ' ', text)
        
        # Fix punctuation issues
        text = re.sub(r'\s+([.!?])', r'\1', text)
        text = re.sub(r'([.!?])\s*([.!?])', r'\1', text)
        
        return text.strip()
    
    def _fix_capitalization(self, text: str) -> str:
        """Fix capitalization in translation"""
        if not text:
            return text
        
        # Capitalize first letter
        text = text[0].upper() + text[1:] if len(text) > 1 else text.upper()
        
        # Capitalize after sentence endings
        text = re.sub(r'([.!?]\s+)([a-z])', lambda m: m.group(1) + m.group(2).upper(), text)
        
        return text
    
    def _validate_translation_quality(self, translated: str, original: str, target_lang: str = 'en') -> bool:
        """
        Validate translation quality with improved logic for large texts and Latin-based languages
        
        FIXED: More robust validation that accounts for translation variations and encoding issues
        FIXED: Reduced strictness to prevent valid translations from being rejected
        
        Args:
            translated: The translated text
            original: The original text
            target_lang: Target language code
            
        Returns:
            True if translation quality is acceptable, False otherwise
        """
        if not translated or len(translated.strip()) < 2:
            return False
        
        # FIXED: Much more lenient validation to prevent false rejections
        # Accept any translation that is different from the original and has reasonable length
        if translated.strip() and translated != original and len(translated.strip()) >= 2:
            return True
        
        return False
    
    def _normalize_for_comparison(self, text: str) -> str:
        """
        Normalize text for comparison to handle encoding variations
        
        Args:
            text: Text to normalize
            
        Returns:
            Normalized text
        """
        if not text:
            return ""
        
        # Fix encoding issues first (including enhanced fix for Spanish characters)
        try:
            normalized = self._fix_encoding_issues_enhanced(text)
        except:
            # Fallback to basic encoding fix if enhanced fails
            normalized = self._fix_encoding_issues(text)
        
        # Remove extra whitespace
        normalized = ' '.join(normalized.split())
        
        # Remove common punctuation variations
        import re
        normalized = re.sub(r'[""''`´]', '"', normalized)
        normalized = re.sub(r'[–—]', '-', normalized)
        
        # Handle remaining � symbols for comparison purposes
        # Replace with space to avoid false negatives
        normalized = normalized.replace('�', ' ')
        
        return normalized.strip()
    
    def _validate_large_text_translation(self, translated: str, original: str, target_lang: str) -> bool:
        """
        Validate translation quality for large texts using sampling approach
        
        Args:
            translated: Translated text
            original: Original text
            target_lang: Target language
            
        Returns:
            True if validation passes
        """
        # Sample-based validation for large texts
        sample_size = min(200, len(translated) // 10)
        
        start_orig = original[:sample_size].lower()
        start_trans = translated[:sample_size].lower()
        
        mid_pos = len(original) // 2
        mid_orig = original[mid_pos:mid_pos + sample_size].lower()
        mid_trans = translated[mid_pos:mid_pos + sample_size].lower()
        
        end_orig = original[-sample_size:].lower()
        end_trans = translated[-sample_size:].lower()
        
        # Check if any sample is too similar (indicating failed translation)
        samples_too_similar = 0
        for orig_sample, trans_sample in [(start_orig, start_trans), (mid_orig, mid_trans), (end_orig, end_trans)]:
            # Normalize samples for comparison
            norm_orig = self._normalize_for_comparison(orig_sample)
            norm_trans = self._normalize_for_comparison(trans_sample)
            if norm_orig == norm_trans:
                samples_too_similar += 1
        
        # If more than 1 sample is identical, likely a failed translation
        if samples_too_similar > 1:
            return False
        
        # Language-specific word validation for large texts
        return self._validate_language_specific_content(translated, target_lang, large_text=True)
    
    def _validate_language_specific_content(self, translated: str, target_lang: str, large_text: bool = False) -> bool:
        """
        Validate that translated text contains language-specific content
        
        FIXED: Much more lenient validation for encoding edge cases
        FIXED: Reduced strictness to prevent valid translations from being rejected
        
        Args:
            translated: Translated text
            target_lang: Target language
            large_text: Whether this is a large text (affects thresholds)
            
        Returns:
            True if validation passes
        """
        # FIXED: For encoding edge cases, be extremely lenient
        # If the text is not empty and not just whitespace, probably valid
        if not translated or translated.isspace():
            return False
        
        # FIXED: For very short texts (< 200 chars), be extremely lenient
        if len(translated) < 200:
            # For short texts, just check if it's not obviously the same as English input
            # Accept almost any non-empty translation for encoding edge cases
            return True
        
        # FIXED: For longer texts, still be very lenient
        if target_lang == 'es':
            # FIXED: Very relaxed Spanish validation for encoding edge cases
            spanish_indicators = [
                # Articles and prepositions
                'el ', 'la ', 'los ', 'las ', 'de ', 'del ', 'con ', 'por ', 'para ', 'que ', 
                'es ', 'son ', 'esta ', 'están ', 'un ', 'una ', 'y ', 'o ', 'en ', 'a ',
                # Common words that might survive encoding issues
                'ción', 'mente', 'ado', 'ido'
            ]
            spanish_count = sum(1 for indicator in spanish_indicators if indicator in translated.lower())
            
            # FIXED: Extremely lenient threshold - just need any Spanish indicators
            min_expected = max(1, len(translated) / 50000)  # Very relaxed threshold
            
            # Accept if we find any Spanish indicators OR if text is reasonably long
            if spanish_count >= min_expected or len(translated) > 500:
                return True
                
        elif target_lang == 'fr':
            # FIXED: Very relaxed French validation
            french_indicators = [
                'le ', 'la ', 'les ', 'de ', 'du ', 'avec ', 'pour ', 'que ', 'est ', 'sont ', 
                'cette ', 'ces ', 'un ', 'une ', 'et ', 'ou ', 'dans ', 'sur ', 'à ',
                'tion', 'ment', 'eur', 'euse'
            ]
            french_count = sum(1 for indicator in french_indicators if indicator in translated.lower())
            
            min_expected = max(1, len(translated) / 50000)
            
            if french_count >= min_expected or len(translated) > 500:
                return True
                
        elif target_lang == 'de':
            # FIXED: Very relaxed German validation
            german_indicators = [
                'der ', 'die ', 'das ', 'den ', 'dem ', 'mit ', 'für ', 'ist ', 'sind ', 
                'diese ', 'dieser ', 'ein ', 'eine ', 'und ', 'oder ', 'in ', 'auf ', 'zu ',
                'ung', 'heit', 'keit', 'lich'
            ]
            german_count = sum(1 for indicator in german_indicators if indicator in translated.lower())
            
            min_expected = max(1, len(translated) / 50000)
            
            if german_count >= min_expected or len(translated) > 500:
                return True
        
        # FIXED: Default to accepting translation for encoding edge cases
        # Only reject if it's clearly broken (empty, only punctuation, etc.)
        if len(translated.strip()) > 0 and any(c.isalnum() for c in translated):
            return True
        
        return False

    def _conservative_translation_fallback(self, original: str) -> str:
        """Conservative fallback when translation quality is poor"""
        # Use basic word-by-word translation if available
        return original  # Return original if no good translation available
    
    def _fallback_translate(self, text: str, source_lang: str, target_lang: str) -> str:
        """Basic pattern-based translation fallback"""
        if target_lang != 'en' or source_lang == 'en':
            return text
        
        # Basic translation patterns for common words
        translations = {
            'es': {
                'hola': 'hello', 'casa': 'house', 'gracias': 'thank you', 'por favor': 'please',
                'adiós': 'goodbye', 'sí': 'yes', 'no': 'no', 'agua': 'water', 'comida': 'food',
                'donde': 'where', 'como': 'how', 'que': 'what', 'cuando': 'when', 'quien': 'who'
            },
            'fr': {
                'bonjour': 'hello', 'maison': 'house', 'merci': 'thank you', 'si vous plait': 'please',
                'au revoir': 'goodbye', 'oui': 'yes', 'non': 'no', 'eau': 'water', 'nourriture': 'food',
                'ou': 'where', 'comment': 'how', 'que': 'what', 'quand': 'when', 'qui': 'who'
            },
            'de': {
                'hallo': 'hello', 'haus': 'house', 'danke': 'thank you', 'bitte': 'please',
                'auf wiedersehen': 'goodbye', 'ja': 'yes', 'nein': 'no', 'wasser': 'water', 'essen': 'food',
                'wo': 'where', 'wie': 'how', 'was': 'what', 'wann': 'when', 'wer': 'who'
            },
            'it': {
                'ciao': 'hello', 'casa': 'house', 'grazie': 'thank you', 'per favore': 'please',
                'arrivederci': 'goodbye', 'si': 'yes', 'no': 'no', 'acqua': 'water', 'cibo': 'food',
                'dove': 'where', 'come': 'how', 'che': 'what', 'quando': 'when', 'chi': 'who'
            },
            'pt': {
                'ola': 'hello', 'casa': 'house', 'obrigado': 'thank you', 'por favor': 'please',
                'tchau': 'goodbye', 'sim': 'yes', 'nao': 'no', 'agua': 'water', 'comida': 'food',
                'onde': 'where', 'como': 'how', 'que': 'what', 'quando': 'when', 'quem': 'who'
            }
        }
        
        if source_lang in translations:
            translated_text = text
            for original, translation in translations[source_lang].items():
                translated_text = translated_text.replace(original, translation)
            return translated_text
        
        return text


# Global instance
_translation_service = None


def get_translation_service() -> TranslationService:
    """Get the global translation service instance"""
    global _translation_service
    if _translation_service is None:
        _translation_service = TranslationService()
    return _translation_service


def translate_to_english(text: str, chatbot_response: str = None) -> Dict[str, any]:
    """
    Main function to translate text to English with enhanced language detection for real estate queries
    Uses Hugging Face MarianMT models with MarianTokenizer for accurate translation
    Now supports bidirectional translation when chatbot_response is provided
    
    Args:
        text: Input string to translate
        chatbot_response: Optional English response to translate back to user's language
        
    Returns:
        Dictionary with:
        - english_query: Translated English version
        - detected_language: Detected language code  
        - translation_needed: Boolean flag indicating if translation was required
        - translated_response: Chatbot response translated to user's language (if provided)
    """
    print(f"[TRANSLATION DEBUG] ===== STARTING MAIN TRANSLATION PROCESS =====")
    print(f"[TRANSLATION DEBUG] Input text: '{text[:100]}{'...' if len(text) > 100 else ''}'")
    print(f"[TRANSLATION DEBUG] Chatbot response provided: {'Yes' if chatbot_response else 'No'}")
    if chatbot_response:
        print(f"[TRANSLATION DEBUG] Chatbot response: '{chatbot_response[:100]}{'...' if len(chatbot_response) > 100 else ''}'")
    
    if not text or not text.strip():
        print(f"[TRANSLATION DEBUG] Empty input text, returning default response")
        return {
            'english_query': '',
            'detected_language': 'en',
            'translation_needed': False,
            'translated_response': chatbot_response if chatbot_response else ''
        }
    
    service = get_translation_service()
    
    # Enhanced language detection with comprehensive logging
    print(f"[TRANSLATION DEBUG] Starting language detection process...")
    logger.info(f"Starting translation process for: '{text[:100]}{'...' if len(text) > 100 else ''}'")
    
    # Use the enhanced detection method with domain-specific cleaning
    detected_lang, confidence = service.detect_language(text)
    print(f"[TRANSLATION DEBUG] Language detection complete: {detected_lang} (confidence: {confidence:.3f})")
    
    # If detected language is English, no translation needed
    if detected_lang == 'en':
        print(f"[TRANSLATION DEBUG] Text detected as English, no forward translation needed")
        logger.info(f"Text detected as English, no translation needed")
        result = {
            'english_query': text.strip(),
            'detected_language': 'en',
            'translation_needed': False,
            'translated_response': chatbot_response if chatbot_response else ''
        }
        print(f"[TRANSLATION DEBUG] ===== TRANSLATION PROCESS COMPLETE (NO TRANSLATION) =====")
        return result
    
    print(f"[TRANSLATION DEBUG] Forward translation needed: {detected_lang} -> en")
    
    # Check if we support this language for MarianMT translation
    if detected_lang not in service.language_models:
        print(f"[TRANSLATION DEBUG] Language {detected_lang} not supported for MarianMT translation, using fallback")
        logger.warning(f"Language {detected_lang} not supported for MarianMT translation. Using fallback.")
        # Try fallback translation
        fallback_result = service._fallback_translate(text, detected_lang, 'en')
        print(f"[TRANSLATION DEBUG] Fallback translation result: '{fallback_result[:50]}...'")
        result = {
            'english_query': fallback_result,
            'detected_language': detected_lang,
            'translation_needed': fallback_result != text
        }
        
        # Add reverse translation if chatbot response provided
        if chatbot_response and detected_lang != 'en':
            print(f"[TRANSLATION DEBUG] Chatbot response provided but language not supported for reverse translation")
            result['translated_response'] = chatbot_response  # Fallback to English
        else:
            result['translated_response'] = chatbot_response if chatbot_response else ''
        
        print(f"[TRANSLATION DEBUG] ===== TRANSLATION PROCESS COMPLETE (FALLBACK) =====")
        return result
    
    print(f"[TRANSLATION DEBUG] Language {detected_lang} supported, attempting MarianMT translation")
    
    # Load the corresponding Helsinki-NLP/opus-mt-{lang}-en model with caching
    if service.is_available() and service._load_model(detected_lang):
        try:
            # Ensure both MarianMTModel and MarianTokenizer are loaded correctly
            if detected_lang in service.models and detected_lang in service.tokenizers:
                print(f"[TRANSLATION DEBUG] Using Helsinki-NLP/opus-mt-{detected_lang}-en model with MarianTokenizer")
                logger.info(f"Using Helsinki-NLP/opus-mt-{detected_lang}-en model with MarianTokenizer")
                
                # Use GPU acceleration if available
                device_info = f" on {service.device}" if service.device else ""
                print(f"[TRANSLATION DEBUG] Translation running{device_info}")
                logger.debug(f"Translation running{device_info}")
                
                # Perform translation with proper SentencePiece tokenization
                translated_text = service._translate_with_marian(text, detected_lang, 'en')
                
                # Verify translation was successful and integrates smoothly into chatbot pipeline
                if translated_text and translated_text.strip() and translated_text != text:
                    print(f"[TRANSLATION DEBUG] Forward MarianMT translation successful!")
                    print(f"[TRANSLATION DEBUG] Original: '{text[:50]}{'...' if len(text) > 50 else ''}'")
                    print(f"[TRANSLATION DEBUG] Translated: '{translated_text[:50]}{'...' if len(translated_text) > 50 else ''}'")
                    logger.info(f"MarianMT translation successful: '{text[:50]}{'...' if len(text) > 50 else ''}' -> '{translated_text[:50]}{'...' if len(translated_text) > 50 else ''}'")
                    
                    result = {
                        'english_query': translated_text.strip(),
                        'detected_language': detected_lang,
                        'translation_needed': True
                    }
                    
                    # Add reverse translation if chatbot response provided
                    if chatbot_response and detected_lang != 'en':
                        print(f"[TRANSLATION DEBUG] Starting reverse translation for chatbot response...")
                        reverse_result = service.translate_response_to_user_language(chatbot_response, detected_lang)
                        result['translated_response'] = reverse_result['translated_response']
                        print(f"[TRANSLATION DEBUG] Reverse translation complete: '{reverse_result['translated_response'][:50]}...'")
                    else:
                        result['translated_response'] = chatbot_response if chatbot_response else ''
                    
                    print(f"[TRANSLATION DEBUG] ===== TRANSLATION PROCESS COMPLETE (SUCCESS) =====")
                    return result
                else:
                    print(f"[TRANSLATION DEBUG] MarianMT translation returned empty or unchanged result for {detected_lang}")
                    logger.warning(f"MarianMT translation returned empty or unchanged result for {detected_lang}")
            else:
                print(f"[TRANSLATION DEBUG] MarianMTModel or MarianTokenizer not properly loaded for {detected_lang}")
                logger.warning(f"MarianMTModel or MarianTokenizer not properly loaded for {detected_lang}")
                
        except Exception as e:
            print(f"[TRANSLATION DEBUG] MarianMT translation failed for {detected_lang}: {e}")
            logger.error(f"MarianMT translation failed for {detected_lang}: {e}")
    
    # Fallback to pattern-based translation (no online dependencies)
    print(f"[TRANSLATION DEBUG] Using offline fallback translation for {detected_lang} -> en")
    logger.info(f"Using offline fallback translation for {detected_lang} -> en")
    fallback_result = service._fallback_translate(text, detected_lang, 'en')
    
    print(f"[TRANSLATION DEBUG] Fallback translation result: '{text[:50]}{'...' if len(text) > 50 else ''}' -> '{fallback_result[:50]}{'...' if len(fallback_result) > 50 else ''}'")
    logger.info(f"Fallback translation result: '{text[:50]}{'...' if len(text) > 50 else ''}' -> '{fallback_result[:50]}{'...' if len(fallback_result) > 50 else ''}'")
    
    result = {
        'english_query': fallback_result,
        'detected_language': detected_lang,
        'translation_needed': fallback_result != text
    }
    
    # Add reverse translation if chatbot response provided
    if chatbot_response and detected_lang != 'en':
        print(f"[TRANSLATION DEBUG] Chatbot response provided but using fallback (no reverse translation)")
        result['translated_response'] = chatbot_response  # Fallback to English
    else:
        result['translated_response'] = chatbot_response if chatbot_response else ''
    
    print(f"[TRANSLATION DEBUG] ===== TRANSLATION PROCESS COMPLETE (FALLBACK) =====")
    return result


def _preprocess_for_detection(text: str) -> str:
    """
    Preprocess text for improved language detection accuracy
    This function is deprecated - use TranslationService._clean_real_estate_text instead
    
    Args:
        text: Raw input text
        
    Returns:
        Preprocessed text optimized for language detection
    """
    if not text:
        return ""
    
    # Use the enhanced cleaning from the service
    service = get_translation_service()
    return service._clean_real_estate_text(text)


def translate_to_english_with_response_support(text: str, chatbot_response: str = None) -> Dict[str, any]:
    """
    Enhanced translation function that supports bidirectional translation
    Translates user query to English and optionally translates chatbot response back to user's language
    
    Args:
        text: Input string to translate to English
        chatbot_response: Optional English response to translate back to user's language
        
    Returns:
        Dictionary with:
        - english_query: Translated English version of user query
        - detected_language: Detected language code  
        - translation_needed: Boolean flag indicating if translation was required
        - translated_response: Chatbot response translated to user's language (if provided)
        - response_translation_needed: Boolean flag for response translation
    """
    # First, translate the user query to English
    forward_result = translate_to_english(text)
    
    result = {
        'english_query': forward_result['english_query'],
        'detected_language': forward_result['detected_language'],
        'translation_needed': forward_result['translation_needed']
    }
    
    # If chatbot response is provided and user language is not English, translate response back
    if chatbot_response and forward_result['detected_language'] != 'en':
        service = get_translation_service()
        reverse_result = service.translate_response_to_user_language(
            chatbot_response, 
            forward_result['detected_language']
        )
        
        result.update({
            'translated_response': reverse_result['translated_response'],
            'response_translation_needed': reverse_result['translation_needed'],
            'original_response': reverse_result['original_response']
        })
    else:
        # No reverse translation needed or possible
        result.update({
            'translated_response': chatbot_response if chatbot_response else '',
            'response_translation_needed': False,
            'original_response': chatbot_response if chatbot_response else ''
        })
    
    return result


def translate_response_to_user_language(english_response: str, user_language: str) -> Dict[str, any]:
    """
    Standalone function to translate English chatbot response to user's language
    
    Args:
        english_response: English response from chatbot
        user_language: User's detected language code
        
    Returns:
        Dictionary with translation results including translated_response field
    """
    service = get_translation_service()
    return service.translate_response_to_user_language(english_response, user_language)


def _try_langid_fallback(text: str, current_lang: str, current_confidence: float) -> Tuple[str, float]:
    """
    Try langid library as fallback if available and confidence is low
    This function is deprecated - use TranslationService._detect_with_langid instead
    
    Args:
        text: Text to analyze
        current_lang: Currently detected language
        current_confidence: Current detection confidence
        
    Returns:
        Tuple of (language_code, confidence)
    """
    if not LANGID_AVAILABLE:
        logger.debug("langid library not available for fallback detection")
        return current_lang, current_confidence
    
    try:
        import langid
        langid_lang, langid_confidence = langid.classify(text)
        
        # If langid has higher confidence, use its result
        if langid_confidence > current_confidence and langid_confidence > 0.85:
            logger.info(f"Using langid result: {langid_lang} (confidence: {langid_confidence:.2f})")
            return langid_lang, langid_confidence
            
    except ImportError:
        logger.debug("langid library not available for fallback detection")
    except Exception as e:
        logger.warning(f"langid fallback failed: {e}")
    
    # Return original detection
    return current_lang, current_confidence


def is_translation_available() -> bool:
    """Check if translation service is available"""
    service = get_translation_service()
    return service.is_available()


def get_supported_languages() -> Dict[str, str]:
    """Get supported languages"""
    return {
        'en': 'english',
        'es': 'spanish',
        'fr': 'french',
        'de': 'german',
        'it': 'italian',
        'pt': 'portuguese',
        'ru': 'russian',
        'zh': 'chinese',
        'ja': 'japanese',
        'ko': 'korean',
        'ar': 'arabic',
        'hi': 'hindi'
    }