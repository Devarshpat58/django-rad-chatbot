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
        
        logger.info(f"Translation service initialized with device: {self.device}")
    
    def _load_model(self, lang_code: str) -> bool:
        """
        Load MarianMT model and MarianTokenizer for specific language with caching enabled
        Uses Helsinki-NLP/opus-mt-{lang}-en models
        
        Args:
            lang_code: Language code to load model for
            
        Returns:
            True if model loaded successfully, False otherwise
        """
        if not TRANSFORMERS_AVAILABLE or lang_code not in self.language_models:
            return False
        
        # Check if model is already loaded (caching)
        if lang_code in self.models and lang_code in self.tokenizers:
            logger.debug(f"Using cached model for {lang_code}")
            return True
        
        try:
            model_name = self.language_models[lang_code]
            logger.info(f"Loading Helsinki-NLP model: {model_name}")
            
            # Load MarianTokenizer with caching
            tokenizer = MarianTokenizer.from_pretrained(
                model_name,
                cache_dir=None,  # Use default cache directory
                local_files_only=False  # Allow download if not cached
            )
            
            # Load MarianMTModel with caching
            model = MarianMTModel.from_pretrained(
                model_name,
                cache_dir=None,  # Use default cache directory
                local_files_only=False  # Allow download if not cached
            )
            
            # Use GPU acceleration if available
            if self.device and self.device != "cpu" and torch.cuda.is_available():
                model = model.to(self.device)
                logger.info(f"Model moved to GPU: {self.device}")
            else:
                logger.info("Model running on CPU")
            
            # Cache the loaded models
            self.tokenizers[lang_code] = tokenizer
            self.models[lang_code] = model
            
            logger.info(f"Successfully loaded and cached MarianMT model for {lang_code}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load MarianMT model for {lang_code}: {e}")
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
        # Log raw input
        logger.debug(f"Language detection - Raw input: '{text[:100]}{'...' if len(text) > 100 else ''}'")
        
        if not text or len(text.strip()) < 2:
            logger.debug("Language detection - Empty or too short text, defaulting to English")
            return 'en', 1.0
        
        # Clean text for real estate domain
        cleaned_text = self._clean_real_estate_text(text)
        logger.debug(f"Language detection - Cleaned text: '{cleaned_text[:100]}{'...' if len(cleaned_text) > 100 else ''}'")
        
        if len(cleaned_text.strip()) < 2:
            logger.debug("Language detection - Cleaned text too short, defaulting to English")
            return 'en', 1.0
        
        # Check for keyword-based overrides first
        keyword_lang = self._check_keyword_overrides(cleaned_text)
        if keyword_lang:
            logger.info(f"Language detection - Keyword override detected: {keyword_lang}")
            return keyword_lang, 0.95
        
        # Primary detection using langdetect
        primary_lang, primary_confidence = self._detect_with_langdetect(cleaned_text)
        logger.debug(f"Language detection - Primary (langdetect): {primary_lang} (confidence: {primary_confidence:.3f})")
        
        # If confidence is below threshold and langid is available, try fallback
        if primary_confidence < 0.90 and LANGID_AVAILABLE:
            fallback_lang, fallback_confidence = self._detect_with_langid(cleaned_text)
            logger.debug(f"Language detection - Fallback (langid): {fallback_lang} (confidence: {fallback_confidence:.3f})")
            
            # Choose the detection with higher confidence
            if fallback_confidence > primary_confidence:
                final_lang, final_confidence = fallback_lang, fallback_confidence
                logger.info(f"Language detection - Using fallback result: {final_lang} (confidence: {final_confidence:.3f})")
            else:
                final_lang, final_confidence = primary_lang, primary_confidence
                logger.info(f"Language detection - Using primary result: {final_lang} (confidence: {final_confidence:.3f})")
        else:
            final_lang, final_confidence = primary_lang, primary_confidence
            logger.info(f"Language detection - Using primary result: {final_lang} (confidence: {final_confidence:.3f})")
        
        # Map to supported language codes
        mapped_lang = self._map_language_code(final_lang)
        
        logger.info(f"Language detection - Final decision: '{text[:50]}{'...' if len(text) > 50 else ''}' -> {mapped_lang} (confidence: {final_confidence:.3f})")
        return mapped_lang, final_confidence
    
    def _clean_real_estate_text(self, text: str) -> str:
        """
        Clean text by removing real estate domain-specific noise
        
        Args:
            text: Raw input text
            
        Returns:
            Cleaned text optimized for language detection
        """
        if not text:
            return ""
        
        # Convert to lowercase for consistency
        cleaned = text.lower().strip()
        
        # Remove currency symbols and codes
        currency_patterns = [
            r'[₹$€£¥₩₽¢]',  # Currency symbols
            r'\b(inr|usd|eur|gbp|jpy|krw|rub|cad|aud|chf|cny)\b',  # Currency codes
            r'\b(rupees?|dollars?|euros?|pounds?|yen|won)\b',  # Currency words
        ]
        for pattern in currency_patterns:
            cleaned = re.sub(pattern, ' ', cleaned, flags=re.IGNORECASE)
        
        # Remove real estate units and measurements
        unit_patterns = [
            r'\b\d+\s*(bhk|bedroom|bedrooms|bed|beds)\b',  # Room counts
            r'\b\d+\s*(sqft|sq\.?\s*ft|square\s+feet?|sq\s*m|square\s+meters?)\b',  # Area units
            r'\b\d+\s*(bathroom|bathrooms|bath|baths|toilet|toilets)\b',  # Bathroom counts
            r'\b(studio|apartment|flat|villa|house|bungalow|penthouse|duplex)\b',  # Property types
            r'\b\d+\s*(floor|floors|storey|storeys|story|stories)\b',  # Floor counts
            r'\b(furnished|unfurnished|semi-furnished|fully-furnished)\b',  # Furnishing
            r'\b(parking|garage|balcony|terrace|garden|pool|gym|elevator|lift)\b',  # Amenities
        ]
        for pattern in unit_patterns:
            cleaned = re.sub(pattern, ' ', cleaned, flags=re.IGNORECASE)
        
        # Remove common location indicators that might interfere with detection
        location_patterns = [
            r'\b(near|close\s+to|next\s+to|opposite|behind|front\s+of)\b',
            r'\b(metro|station|airport|mall|market|school|hospital|park)\b',
            r'\b(north|south|east|west|central|downtown|uptown)\b',
            r'\b(road|street|avenue|lane|colony|sector|phase|block)\b',
        ]
        for pattern in location_patterns:
            cleaned = re.sub(pattern, ' ', cleaned, flags=re.IGNORECASE)
        
        # Remove numbers and special characters but preserve language-specific characters
        cleaned = re.sub(r'\b\d+([.,]\d+)?\b', ' ', cleaned)  # Numbers
        cleaned = re.sub(r'[^\w\s\u00C0-\u017F\u0100-\u024F\u1E00-\u1EFF\u0400-\u04FF\u0590-\u05FF\u0600-\u06FF\u0900-\u097F\u4E00-\u9FFF\u3040-\u309F\u30A0-\u30FF]', ' ', cleaned)  # Keep Unicode letters
        
        # Normalize whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        return cleaned
    
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
    
    def _translate_with_marian(self, text: str, source_lang: str) -> str:
        """
        Translate text using MarianMT model with MarianTokenizer and proper SentencePiece handling
        Uses GPU acceleration if available (torch.cuda.is_available())
        
        Args:
            text: Text to translate
            source_lang: Source language code
            
        Returns:
            Translated text in English
        """
        if source_lang not in self.models or source_lang not in self.tokenizers:
            raise ValueError(f"MarianMTModel not loaded for language: {source_lang}")
        
        model = self.models[source_lang]
        tokenizer = self.tokenizers[source_lang]
        
        # Clean and prepare text
        clean_text = text.strip()
        if not clean_text:
            return ""
        
        try:
            # Tokenize input with MarianTokenizer using SentencePiece
            # Proper input/output tokenization handling
            inputs = tokenizer(
                clean_text, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=512
            )
            
            # Use GPU acceleration if available
            if self.device and self.device != "cpu" and torch.cuda.is_available():
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                logger.debug(f"Using GPU acceleration for translation: {self.device}")
            else:
                logger.debug("Using CPU for translation")
            
            # Generate translation with optimized parameters for high-quality results
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=512,
                    num_beams=4,  # Beam search for better quality
                    early_stopping=True,
                    do_sample=False,  # Deterministic output
                    temperature=1.0,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    forced_bos_token_id=tokenizer.lang_code_to_id.get('en', None) if hasattr(tokenizer, 'lang_code_to_id') else None
                )
            
            # Decode output with MarianTokenizer, handling SentencePiece properly
            translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Post-process translation for chatbot pipeline integration
            translated_text = translated_text.strip()
            
            # Remove any source text that might be included in output
            if clean_text.lower() in translated_text.lower():
                # Try to extract only the translation part
                parts = translated_text.split()
                source_parts = clean_text.split()
                if len(parts) > len(source_parts):
                    translated_text = ' '.join(parts[len(source_parts):]).strip()
            
            # Ensure translation is different from input and not empty
            if not translated_text or translated_text.lower() == clean_text.lower():
                logger.warning(f"MarianMT produced no translation or identical output for: {clean_text}")
                return clean_text
            
            logger.debug(f"MarianMT translation ({source_lang}->en): '{clean_text}' -> '{translated_text}'")
            return translated_text
            
        except Exception as e:
            logger.error(f"Error in MarianMT translation for {source_lang}: {e}")
            raise
    
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


def translate_to_english(text: str) -> Dict[str, any]:
    """
    Main function to translate text to English with enhanced language detection for real estate queries
    Uses Hugging Face MarianMT models with MarianTokenizer for accurate translation
    
    Args:
        text: Input string to translate
        
    Returns:
        Dictionary with:
        - english_query: Translated English version
        - detected_language: Detected language code  
        - translation_needed: Boolean flag indicating if translation was required
    """
    if not text or not text.strip():
        return {
            'english_query': '',
            'detected_language': 'en',
            'translation_needed': False
        }
    
    service = get_translation_service()
    
    # Enhanced language detection with comprehensive logging
    logger.info(f"Starting translation process for: '{text[:100]}{'...' if len(text) > 100 else ''}'")
    
    # Use the enhanced detection method with domain-specific cleaning
    detected_lang, confidence = service.detect_language(text)
    
    # If detected language is English, no translation needed
    if detected_lang == 'en':
        logger.info(f"Text detected as English, no translation needed")
        return {
            'english_query': text.strip(),
            'detected_language': 'en',
            'translation_needed': False
        }
    
    # Check if we support this language for MarianMT translation
    if detected_lang not in service.language_models:
        logger.warning(f"Language {detected_lang} not supported for MarianMT translation. Using fallback.")
        # Try fallback translation
        fallback_result = service._fallback_translate(text, detected_lang, 'en')
        return {
            'english_query': fallback_result,
            'detected_language': detected_lang,
            'translation_needed': fallback_result != text
        }
    
    # Load the corresponding Helsinki-NLP/opus-mt-{lang}-en model with caching
    if service.is_available() and service._load_model(detected_lang):
        try:
            # Ensure both MarianMTModel and MarianTokenizer are loaded correctly
            if detected_lang in service.models and detected_lang in service.tokenizers:
                logger.info(f"Using Helsinki-NLP/opus-mt-{detected_lang}-en model with MarianTokenizer")
                
                # Use GPU acceleration if available
                device_info = f" on {service.device}" if service.device else ""
                logger.debug(f"Translation running{device_info}")
                
                # Perform translation with proper SentencePiece tokenization
                translated_text = service._translate_with_marian(text, detected_lang)
                
                # Verify translation was successful and integrates smoothly into chatbot pipeline
                if translated_text and translated_text.strip() and translated_text != text:
                    logger.info(f"MarianMT translation successful: '{text[:50]}{'...' if len(text) > 50 else ''}' -> '{translated_text[:50]}{'...' if len(translated_text) > 50 else ''}'")
                    return {
                        'english_query': translated_text.strip(),
                        'detected_language': detected_lang,
                        'translation_needed': True
                    }
                else:
                    logger.warning(f"MarianMT translation returned empty or unchanged result for {detected_lang}")
            else:
                logger.warning(f"MarianMTModel or MarianTokenizer not properly loaded for {detected_lang}")
                
        except Exception as e:
            logger.error(f"MarianMT translation failed for {detected_lang}: {e}")
    
    # Fallback to pattern-based translation (no online dependencies)
    logger.info(f"Using offline fallback translation for {detected_lang} -> en")
    fallback_result = service._fallback_translate(text, detected_lang, 'en')
    
    logger.info(f"Fallback translation result: '{text[:50]}{'...' if len(text) > 50 else ''}' -> '{fallback_result[:50]}{'...' if len(fallback_result) > 50 else ''}'")
    
    return {
        'english_query': fallback_result,
        'detected_language': detected_lang,
        'translation_needed': fallback_result != text
    }


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