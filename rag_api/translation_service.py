"""
Translation Service for Django RAG Chatbot
Uses Hugging Face transformers with MarianMT models for local translation
No API keys or credentials required - completely offline service
"""

import logging
import re
from typing import Dict, Optional, Tuple
from functools import lru_cache

logger = logging.getLogger(__name__)

# Try to import required libraries
try:
    from transformers import MarianMTModel, MarianTokenizer, pipeline
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("transformers library not available. Translation service will be limited.")
except Exception as e:
    TRANSFORMERS_AVAILABLE = False
    logger.warning(f"transformers library failed to load: {e}. Translation service will be limited.")

try:
    from langdetect import detect, detect_langs
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False
    logger.warning("langdetect library not available. Using basic language detection.")
except Exception as e:
    LANGDETECT_AVAILABLE = False
    logger.warning(f"langdetect library failed to load: {e}. Using basic language detection.")

try:
    import langid
    LANGID_AVAILABLE = True
except ImportError:
    LANGID_AVAILABLE = False
    logger.warning("langid library not available. Fallback detection will be limited.")
except Exception as e:
    LANGID_AVAILABLE = False
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
                        logger.info(f"Reverse translation successful: '{english_response[:50]}...' -> '{translated_response[:50]}...'")
                        return {
                            'original_response': english_response,
                            'translated_response': translated_response,
                            'target_language': user_language,
                            'translation_needed': True
                        }
                    else:
                        print(f"[TRANSLATION DEBUG] Reverse translation failed or returned unchanged result")
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
        
        Args:
            text: Text to translate
            target_lang: Target language code (default: 'en')
            source_lang: Source language code (auto-detect if None)
            
        Returns:
            Dictionary with translation results
        """
        if not text or not text.strip():
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
    
    def _translate_with_marian(self, text: str, source_lang: str, target_lang: str = 'en') -> str:
        """
        Translate text using MarianMT model with enhanced preprocessing for better accuracy
        Uses GPU acceleration if available (torch.cuda.is_available())
        Supports both forward (lang->en) and reverse (en->lang) translation
        
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
            # Tokenize input with MarianTokenizer using SentencePiece
            # Enhanced tokenization with better parameters
            inputs = tokenizer(
                preprocessed_text, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=512,
                add_special_tokens=True
            )
            print(f"[TRANSLATION DEBUG] Tokenization completed, input shape: {inputs['input_ids'].shape}")
            
            # Use GPU acceleration if available
            if self.device and self.device != "cpu" and torch.cuda.is_available():
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                print(f"[TRANSLATION DEBUG] Using GPU acceleration for translation: {self.device}")
                logger.debug(f"Using GPU acceleration for translation: {self.device}")
            else:
                print(f"[TRANSLATION DEBUG] Using CPU for translation")
                logger.debug("Using CPU for translation")
            
            print(f"[TRANSLATION DEBUG] Starting model generation with beam search (5 beams)")
            # Generate translation with enhanced parameters for higher accuracy
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=512,
                    num_beams=5,  # Increased beam search for better quality
                    early_stopping=True,
                    do_sample=False,  # Deterministic output
                    temperature=1.0,
                    length_penalty=1.0,  # Balanced length penalty
                    repetition_penalty=1.1,  # Slight repetition penalty
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    forced_bos_token_id=tokenizer.lang_code_to_id.get(target_lang, None) if hasattr(tokenizer, 'lang_code_to_id') else None
                )
            print(f"[TRANSLATION DEBUG] Model generation completed, output shape: {outputs.shape}")
            
            # Decode output with MarianTokenizer, handling SentencePiece properly
            translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"[TRANSLATION DEBUG] Raw decoded text: '{translated_text[:100]}{'...' if len(translated_text) > 100 else ''}'")
            
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
        """Apply language-specific preprocessing rules"""
        if source_lang == 'es':
            # Spanish specific preprocessing
            text = re.sub(r'¿([^?]+)\?', r'\1?', text)  # Normalize question marks
            text = re.sub(r'¡([^!]+)!', r'\1!', text)  # Normalize exclamation marks
        elif source_lang == 'de':
            # German specific preprocessing
            text = re.sub(r'\bß\b', 'ss', text)  # Normalize ß for better tokenization
        elif source_lang == 'fr':
            # French specific preprocessing
            text = re.sub(r'\b([cdjlmnst])\'([aeiouhy])', r'\1 \2', text)  # Handle contractions
        elif source_lang == 'zh':
            # Chinese specific preprocessing
            text = re.sub(r'[，。！？；：]', lambda m: {'，': ',', '。': '.', '！': '!', '？': '?', '；': ';', '：': ':'}[m.group()], text)
        
        return text
    
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
        
        # Step 5: Validate translation quality
        if not self._validate_translation_quality(result, original, target_lang):
            logger.warning(f"Translation quality check failed, using conservative approach")
            return self._conservative_translation_fallback(original)
        
        return result
    
    def _remove_source_contamination(self, translated: str, preprocessed: str, original: str) -> str:
        """Remove source text contamination from translation"""
        # Check if original text appears in translation
        original_words = set(original.lower().split())
        translated_words = translated.lower().split()
        
        # If too many original words appear, try to extract only the translation
        contamination_ratio = len(original_words.intersection(set(translated_words))) / max(len(original_words), 1)
        
        if contamination_ratio > 0.7:  # High contamination
            # Try to find the actual translation part
            sentences = translated.split('.')
            for sentence in sentences:
                sentence_words = set(sentence.lower().split())
                sentence_contamination = len(original_words.intersection(sentence_words)) / max(len(sentence_words), 1)
                if sentence_contamination < 0.5 and len(sentence.strip()) > 3:
                    return sentence.strip()
        
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
        """Validate translation quality"""
        if not translated or len(translated.strip()) < 2:
            return False
        
        # Check if translation is too similar to original (possible failure)
        if translated.lower() == original.lower():
            return False
        
        # Check if translation has reasonable length
        length_ratio = len(translated) / max(len(original), 1)
        if length_ratio < 0.3 or length_ratio > 3.0:
            return False
        
        # Language-specific quality checks
        if target_lang == 'en':
            # Check if translation contains reasonable English words
            english_words = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'])
            translated_words = set(translated.lower().split())
            
            if len(english_words.intersection(translated_words)) == 0 and len(translated_words) > 3:
                return False
        else:
            # For reverse translation, check if it contains non-English characters or patterns
            # This is a basic check - could be enhanced with language-specific validation
            if target_lang in ['es', 'fr', 'de', 'it', 'pt']:
                # Check for Latin script languages
                if all(ord(char) < 128 for char in translated if char.isalpha()):
                    # All ASCII - might be English contamination for these languages
                    english_indicators = ['the', 'and', 'or', 'in', 'on', 'at', 'to', 'for', 'of', 'with']
                    translated_lower = translated.lower()
                    if any(word in translated_lower for word in english_indicators):
                        return False
        
        return True
    
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