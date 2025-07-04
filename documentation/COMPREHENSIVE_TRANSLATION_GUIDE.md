# Comprehensive Translation Service Guide

## Table of Contents
1. [Overview](#overview)
2. [Features](#features)
3. [Architecture](#architecture)
4. [Implementation](#implementation)
5. [Language Support](#language-support)
6. [API Reference](#api-reference)
7. [Analytics & Monitoring](#analytics--monitoring)
8. [Advanced Features](#advanced-features)
9. [Installation & Setup](#installation--setup)
10. [Troubleshooting](#troubleshooting)

---

# Overview

The Django RAG Chatbot features a comprehensive translation service that provides high-quality multilingual support for real estate property search. The system combines MarianMT neural machine translation, enhanced language detection, bidirectional translation, full JSON document translation, and comprehensive analytics.

## Key Capabilities

- **🌍 12 Language Support**: English, Spanish, French, German, Italian, Portuguese, Russian, Chinese, Japanese, Korean, Arabic, Hindi
- **🔄 Bidirectional Translation**: Query translation (Any → English) and response translation (English → User Language)
- **📄 Full JSON Translation**: Complete document translation instead of fragment-based approach
- **🎯 Enhanced Detection**: Domain-specific language detection with real estate vocabulary
- **📊 Analytics & Monitoring**: Performance tracking, health monitoring, and usage analytics
- **🔒 Guaranteed Output**: Multiple fallback mechanisms ensure UI always receives displayable content
- **⚡ GPU Acceleration**: Automatic CUDA support for optimal performance
- **🆓 Zero Cost**: No API keys required, completely offline operation

---

# Features

## Core Translation Features

### Neural Machine Translation
- **MarianMT Models**: Helsinki-NLP opus-mt models for professional translation quality
- **Model Caching**: Efficient memory management with automatic model loading/unloading
- **Beam Search**: 4-beam search for higher quality translations
- **GPU Acceleration**: Automatic CUDA utilization when available

### Enhanced Language Detection
- **Domain-Specific Cleaning**: Real estate vocabulary preprocessing
- **Keyword Override System**: Language-specific keyword dictionaries
- **Multi-Library Detection**: Primary (langdetect) + Fallback (langid) detection
- **Confidence Scoring**: Confidence-based detection with threshold management

### Character Encoding Support
- **Multi-Language Characters**: Full support for accented characters, special symbols
- **Encoding Recovery**: Handles UTF-8 corruption and character replacement issues
- **Context-Aware Fixes**: Language-specific character pattern recovery
- **Unicode Normalization**: Proper Unicode handling for all supported languages

### Bidirectional Translation
- **Forward Translation**: User queries (Any language → English)
- **Reverse Translation**: System responses (English → User language)
- **Unified API**: Single function calls for complete translation workflow
- **Translation Metadata**: Comprehensive translation information included

### Full JSON Document Translation
- **Complete Data Translation**: Translates entire source JSON documents
- **Structure Preservation**: Maintains data structure while translating text fields
- **Priority Field Translation**: Focuses on user-relevant fields
- **Enhanced Field Mappings**: Provides enhanced field mappings in target language
- **Reduced Character Limits**: More efficient use of model token limits

### Guaranteed Translation
- **UI Safety**: Always returns displayable content
- **Multiple Fallbacks**: Neural → Pattern-based → English → Safe default
- **Error Handling**: Graceful degradation without silent failures
- **Reliability**: Conservative approach maintains system stability

---

# Architecture

## System Design

```
┌─────────────────────────────────────────────────────────────────┐
│                        Translation Service Architecture          │
├─────────────────────────────────────────────────────────────────┤
│ ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐    │
│ │   User Input    │  │   Web Interface │  │   API Endpoints │    │
│ │                 │  │                 │  │                 │    │
│ └─────────────────┘  └─────────────────┘  └─────────────────┘    │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                        Translation Pipeline                      │
├─────────────────────────────────────────────────────────────────┤
│ ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐    │
│ │ Language        │  │   Text          │  │   Character     │    │
│ │ Detection       │  │   Cleaning      │  │   Encoding      │    │
│ └─────────────────┘  └─────────────────┘  └─────────────────┘    │
│ ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐    │
│ │ Neural          │  │   Fallback      │  │   Quality       │    │
│ │ Translation     │  │   Translation   │  │   Validation    │    │
│ └─────────────────┘  └─────────────────┘  └─────────────────┘    │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                        Analytics & Monitoring                    │
├─────────────────────────────────────────────────────────────────┤
│ ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐    │
│ │ Performance     │  │   Health        │  │   Usage         │    │
│ │ Tracking        │  │   Monitoring    │  │   Analytics     │    │
│ └─────────────────┘  └─────────────────┘  └─────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

## Component Integration

### Translation Service Classes
- **TranslationService**: Core neural translation with MarianMT
- **JSONTranslationService**: Full document translation service
- **TranslationAnalytics**: Performance monitoring and analytics

### Service Functions
- **translate_to_english()**: Basic query translation
- **translate_to_english_guaranteed()**: Guaranteed safe translation
- **translate_response_guaranteed()**: Response translation with fallbacks
- **translate_full_response_guaranteed()**: Complete JSON document translation

---

# Implementation

## Core Translation Implementation

### MarianMT Translation Process

```python
from transformers import MarianMTModel, MarianTokenizer
import torch

class TranslationService:
    def __init__(self):
        self.models = {}
        self.tokenizers = {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def _load_model(self, source_lang: str) -> bool:
        """Load MarianMT model for specific language pair"""
        try:
            model_name = f"Helsinki-NLP/opus-mt-{source_lang}-en"
            self.models[source_lang] = MarianMTModel.from_pretrained(model_name)
            self.tokenizers[source_lang] = MarianTokenizer.from_pretrained(model_name)
            
            if self.device == "cuda":
                self.models[source_lang].to(self.device)
            
            return True
        except Exception as e:
            logger.error(f"Failed to load model for {source_lang}: {e}")
            return False
    
    def _translate_with_marian(self, text: str, source_lang: str) -> str:
        """Translate text using MarianMT model"""
        model = self.models[source_lang]
        tokenizer = self.tokenizers[source_lang]
        
        # Tokenize input
        inputs = tokenizer(text, return_tensors="pt", padding=True, 
                          truncation=True, max_length=512)
        
        if self.device == "cuda":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate translation
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=512,
                num_beams=4,
                early_stopping=True,
                pad_token_id=tokenizer.pad_token_id
            )
        
        # Decode output
        translated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return translated.strip()
```

### Enhanced Language Detection

```python
def detect_language_enhanced(self, text: str) -> Tuple[str, float]:
    """Enhanced language detection with domain-specific processing"""
    
    # Step 1: Clean real estate noise
    cleaned_text = self._clean_real_estate_text(text)
    
    # Step 2: Check for keyword overrides
    keyword_lang = self._check_keyword_overrides(cleaned_text)
    if keyword_lang:
        return keyword_lang, 0.95
    
    # Step 3: Primary detection with langdetect
    try:
        detected_lang = detect(cleaned_text)
        lang_probs = detect_langs(cleaned_text)
        confidence = next((prob.prob for prob in lang_probs 
                          if prob.lang == detected_lang), 0.8)
        
        # Step 4: Fallback detection if confidence low
        if confidence < 0.90:
            fallback_lang, fallback_conf = self._fallback_detection(cleaned_text)
            if fallback_conf > confidence:
                return fallback_lang, fallback_conf
        
        return detected_lang, confidence
        
    except Exception as e:
        logger.warning(f"Language detection failed: {e}")
        return self._pattern_based_detection(text)
```

### Character Encoding Enhancement

```python
def _fix_encoding_issues_enhanced(self, text: str, source_lang: str = None) -> str:
    """Enhanced encoding fixes for all supported languages"""
    
    # Language-specific character recovery patterns
    if source_lang == 'es' or any(word in text.lower() for word in ['tiene', 'hay', 'casa']):
        # Spanish character recovery
        spanish_patterns = [
            (r'Tiene', '¿Tiene'), (r'([A-Z])', r'¿\1'),
            (r'a', 'á'), (r'e', 'é'), (r'i', 'í'), (r'o', 'ó'), (r'u', 'ú'), (r'n', 'ñ'),
            # ... comprehensive pattern set
        ]
        for pattern, replacement in spanish_patterns:
            text = re.sub(pattern, replacement, text)
    
    elif source_lang == 'fr' or any(word in text.lower() for word in ['avec', 'dans', 'pour']):
        # French character recovery
        french_patterns = [
            (r'a', 'à'), (r'e', 'é'), (r'e', 'è'), (r'e', 'ê'), (r'c', 'ç'),
            # ... comprehensive pattern set
        ]
        for pattern, replacement in french_patterns:
            text = re.sub(pattern, replacement, text)
    
    # ... additional languages
    
    return text
```

---

# Language Support

## Supported Languages

### Romance Languages
- **Spanish (es)**: Helsinki-NLP/opus-mt-es-en
  - Characters: ¿¡áéíóúñ
  - Real estate terms: apartamento, dormitorio, baño, cocina, alquiler
- **French (fr)**: Helsinki-NLP/opus-mt-fr-en
  - Characters: àâäéèêëîïôöùûüÿç
  - Real estate terms: appartement, chambre, salle de bain, location
- **Italian (it)**: Helsinki-NLP/opus-mt-it-en
  - Characters: àèéìíîòóùú
  - Real estate terms: appartamento, camera, bagno, affitto
- **Portuguese (pt)**: Helsinki-NLP/opus-mt-pt-en
  - Characters: ãâêôõç
  - Real estate terms: apartamento, quarto, banheiro, aluguel

### Germanic Languages
- **German (de)**: Helsinki-NLP/opus-mt-de-en
  - Characters: äöüß
  - Real estate terms: wohnung, schlafzimmer, badezimmer, miete

### Cyrillic Languages
- **Russian (ru)**: Helsinki-NLP/opus-mt-ru-en
  - Characters: Full Cyrillic alphabet
  - Real estate terms: квартира, спальня, ванная, аренда

### Asian Languages
- **Chinese (zh)**: Helsinki-NLP/opus-mt-zh-en
  - Characters: Common Chinese characters + punctuation
  - Real estate terms: 公寓, 卧室, 浴室, 租金
- **Japanese (ja)**: Helsinki-NLP/opus-mt-ja-en
  - Characters: Hiragana, Katakana, common Kanji
  - Real estate terms: アパート, 寝室, バスルーム, 賃貸
- **Korean (ko)**: Helsinki-NLP/opus-mt-ko-en
  - Characters: Hangul character support
  - Real estate terms: 아파트, 침실, 화장실, 임대

### Semitic Languages
- **Arabic (ar)**: Helsinki-NLP/opus-mt-ar-en
  - Characters: Arabic script + RTL text handling
  - Real estate terms: شقة, غرفة نوم, حمام, إيجار
- **Hindi (hi)**: Helsinki-NLP/opus-mt-hi-en
  - Characters: Devanagari script
  - Real estate terms: घर, बेडरूम, बाथरूम, किराया

### Default Language
- **English (en)**: No translation required (default)

---

# API Reference

## Core Functions

### translate_to_english(text: str, chatbot_response: str = None) -> Dict

**Main translation function with bidirectional support**

```python
result = translate_to_english("Busco apartamento de 2 dormitorios")
# Returns:
{
    'english_query': 'I am looking for a 2-bedroom apartment',
    'detected_language': 'es',
    'translation_needed': True,
    'translated_response': ''  # Empty if no chatbot_response provided
}
```

### translate_to_english_guaranteed(text: str) -> Dict

**Guaranteed safe translation with fallback mechanisms**

```python
result = translate_to_english_guaranteed("Hola mundo")
# Returns:
{
    'english_query': 'Hello world',
    'detected_language': 'es',
    'translation_needed': True,
    'ui_safe': True,
    'performance_metrics': {
        'response_time': 0.123,
        'method_used': 'normal_translation'
    }
}
```

### translate_response_guaranteed(english_response: str, target_language: str) -> Dict

**Safe response translation to user's language**

```python
result = translate_response_guaranteed(
    "I found 3 apartments with 2 bedrooms", 
    "es"
)
# Returns:
{
    'translated_response': 'Encontré 3 apartamentos con 2 dormitorios',
    'original_response': 'I found 3 apartments with 2 bedrooms',
    'target_language': 'es',
    'translation_needed': True,
    'ui_safe': True
}
```

### translate_full_response_guaranteed(response_data: Dict, user_language: str) -> Dict

**Complete JSON document translation**

```python
response_data = {
    'response': 'I found several apartments...',
    'results': [{'name': 'Cozy Apartment', 'summary': 'Beautiful 2BR apt'}],
    'metadata': {'num_results': 1}
}

result = translate_full_response_guaranteed(response_data, 'es')
# Returns complete translated response with all JSON fields translated
```

## Analytics Functions

### get_system_health() -> Dict

**Translation system health monitoring**

```python
health = get_system_health()
# Returns:
{
    'status': 'excellent',  # excellent/good/fair/poor
    'health_score': 95.2,
    'success_rate': 98.5,
    'fallback_rate': 12.3,
    'avg_response_time': 1.245,
    'recommendations': ['System performing well']
}
```

### get_language_performance() -> Dict

**Per-language performance statistics**

```python
performance = get_language_performance()
# Returns:
{
    'es': {
        'total_requests': 45,
        'success_rate': 97.8,
        'fallback_rate': 15.6,
        'avg_response_time': 1.123
    },
    # ... other languages
}
```

---

# Analytics & Monitoring

## Performance Tracking

The translation service includes comprehensive analytics to monitor translation performance, system health, and usage patterns.

### TranslationAnalytics Class

```python
class TranslationAnalytics:
    def __init__(self):
        self.cache = caches['default']
        self._init_analytics_data()
    
    def record_translation_request(
        self, 
        source_lang: str, 
        success: bool, 
        response_time: float, 
        method: str
    ):
        """Record translation request for analytics"""
        request_data = {
            'timestamp': timezone.now().isoformat(),
            'source_language': source_lang,
            'success': success,
            'response_time': response_time,
            'method_used': method,
            'date': timezone.now().date().isoformat()
        }
        
        # Update recent requests
        recent_requests = self._get_recent_requests()
        recent_requests.append(request_data)
        if len(recent_requests) > 100:  # Keep only last 100
            recent_requests = recent_requests[-100:]
        self.cache.set('recent_translation_requests', recent_requests, None)
        
        # Update daily stats
        self._update_daily_stats(request_data)
        
        # Update language performance
        self._update_language_performance(source_lang, success, response_time, method)
```

### Health Scoring Algorithm

```python
def get_system_health(self) -> Dict[str, any]:
    """Calculate overall system health score"""
    recent_requests = self._get_recent_requests()
    
    if not recent_requests:
        return self._default_health_response()
    
    # Calculate metrics
    total_requests = len(recent_requests)
    successful_requests = sum(1 for req in recent_requests if req['success'])
    success_rate = (successful_requests / total_requests) * 100
    
    fallback_requests = sum(1 for req in recent_requests 
                           if req['method_used'] in ['guaranteed_fallback', 'safe_default'])
    fallback_rate = (fallback_requests / total_requests) * 100
    
    avg_response_time = sum(req['response_time'] for req in recent_requests) / total_requests
    
    # Health score calculation (0-100)
    health_score = (
        (success_rate * 0.5) +  # 50% weight for success rate
        ((100 - fallback_rate) * 0.3) +  # 30% weight for low fallback rate
        (max(0, 100 - (avg_response_time * 20)) * 0.2)  # 20% weight for response time
    )
    
    # Determine status
    if health_score >= 90:
        status = 'excellent'
    elif health_score >= 75:
        status = 'good'
    elif health_score >= 50:
        status = 'fair'
    else:
        status = 'poor'
    
    return {
        'status': status,
        'health_score': round(health_score, 1),
        'success_rate': round(success_rate, 1),
        'fallback_rate': round(fallback_rate, 1),
        'avg_response_time': round(avg_response_time, 3),
        'recommendations': self._generate_recommendations(health_score, success_rate, fallback_rate)
    }
```

### API Endpoints

- **GET /ajax/translation-health/**: Complete health dashboard data
- **POST /ajax/translation-reset/**: Admin endpoint to reset analytics (requires admin permissions)

---

# Advanced Features

## Bidirectional Translation Workflow

```python
# Complete bidirectional translation example
def process_multilingual_query(user_query: str, session_id: str):
    """Complete workflow for multilingual query processing"""
    
    # Step 1: Translate user query to English
    translation_result = translate_to_english_guaranteed(user_query)
    english_query = translation_result['english_query']
    user_language = translation_result['detected_language']
    
    # Step 2: Process query with RAG system
    rag_result = rag_service.process_query(
        query_text=english_query,
        session_id=session_id
    )
    
    # Step 3: Prepare complete response data
    complete_response_data = {
        'response': rag_result['response'],
        'results': rag_result['metadata'].get('results', []),
        'metadata': rag_result['metadata']
    }
    
    # Step 4: Translate complete response to user's language
    if user_language != 'en':
        final_response = translate_full_response_guaranteed(
            complete_response_data, 
            user_language
        )
    else:
        final_response = complete_response_data
    
    # Step 5: Add translation metadata
    final_response['translation_info'] = {
        'original_language': user_language,
        'query_translation': translation_result,
        'response_translated': user_language != 'en'
    }
    
    return final_response
```

## JSON Document Translation

### Priority Field Translation

```python
def _translate_json_document(self, doc: Dict, target_lang: str) -> Dict:
    """Translate complete JSON document with priority field handling"""
    
    if not doc or target_lang == 'en':
        return doc
    
    translated_doc = doc.copy()
    
    # Priority fields for real estate properties
    priority_fields = [
        'name', 'summary', 'description', 'space',
        'neighborhood_overview', 'amenities', 'property_type',
        'room_type', 'street', 'neighbourhood', 'city', 'country'
    ]
    
    # Translate text fields
    for field in priority_fields:
        if field in translated_doc:
            if isinstance(translated_doc[field], str) and translated_doc[field].strip():
                translated_text = self._safe_translate_text(
                    translated_doc[field], 
                    'en', 
                    target_lang
                )
                translated_doc[field] = translated_text
            elif isinstance(translated_doc[field], list):
                # Handle arrays (e.g., amenities)
                translated_array = []
                for item in translated_doc[field]:
                    if isinstance(item, str) and item.strip():
                        translated_item = self._safe_translate_text(
                            item, 'en', target_lang
                        )
                        translated_array.append(translated_item)
                    else:
                        translated_array.append(item)
                translated_doc[field] = translated_array
    
    return translated_doc
```

### Enhanced Field Mappings

```python
def _extract_enhanced_query_fields(self, doc: Dict, user_query: str, target_lang: str) -> Dict:
    """Extract query-relevant fields with enhanced mappings"""
    
    enhanced_fields = {}
    
    # Property identification
    if 'name' in doc:
        enhanced_fields['property_name'] = doc['name']
    
    # Location information
    location_parts = []
    for field in ['street', 'neighbourhood_cleansed', 'city', 'country']:
        if field in doc and doc[field]:
            location_parts.append(str(doc[field]))
    if location_parts:
        enhanced_fields['location'] = ', '.join(location_parts)
    
    # Property description
    description_parts = []
    for field in ['summary', 'description', 'space']:
        if field in doc and doc[field]:
            description_parts.append(str(doc[field]))
    if description_parts:
        enhanced_fields['description'] = ' '.join(description_parts)
    
    # Property features
    if 'amenities' in doc and isinstance(doc['amenities'], list):
        enhanced_fields['amenities'] = doc['amenities']
    
    # Property type and room information
    for field in ['property_type', 'room_type', 'bedrooms', 'bathrooms', 'accommodates']:
        if field in doc:
            enhanced_fields[field] = doc[field]
    
    return enhanced_fields
```

---

# Installation & Setup

## Prerequisites

### System Requirements
- Python 3.8+
- PyTorch 1.9+
- 4GB RAM minimum (8GB recommended)
- 2GB disk space for models
- GPU optional (CUDA 11.0+ for acceleration)

### Required Libraries

```bash
# Core translation libraries
pip install transformers torch
pip install sentencepiece  # Required for MarianMT tokenizers

# Language detection
pip install langdetect
pip install langid  # Optional fallback

# Django and web framework
pip install django djangorestframework

# Additional utilities
pip install numpy scipy
pip install requests urllib3
```

## Installation Steps

### 1. Install Dependencies

```bash
# Install from requirements file
pip install -r requirements.txt

# Or install individual packages
pip install transformers torch sentencepiece langdetect
```

### 2. Model Setup

Models are downloaded automatically on first use:

```python
# Test translation service setup
from rag_api.translation_service import translate_to_english

# This will download Spanish model on first use
result = translate_to_english("Hola mundo")
print(result)  # {'english_query': 'Hello world', ...}
```

### 3. GPU Configuration (Optional)

```python
# Check GPU availability
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")

# Environment variables for GPU control
# Force CPU-only mode
export TRANSFORMERS_DEVICE=cpu

# Custom model cache directory
export TRANSFORMERS_CACHE=/path/to/cache
```

### 4. Django Integration

```python
# Add to Django settings.py
INSTALLED_APPS = [
    'rag_api',
    # ... other apps
]

# Add translation service configuration
TRANSLATION_CONFIG = {
    'model_cache_dir': './models/',
    'max_length': 512,
    'num_beams': 4,
    'gpu_acceleration': True,
    'confidence_threshold': 0.85
}
```

### 5. Verify Installation

```python
# Test all components
python manage.py shell

>>> from rag_api.translation_service import *
>>> 
>>> # Test basic translation
>>> result = translate_to_english("Bonjour le monde")
>>> print(result)
>>> 
>>> # Test guaranteed translation
>>> result = translate_to_english_guaranteed("Hola")
>>> print(result)
>>> 
>>> # Test analytics
>>> from rag_api.translation_analytics import get_system_health
>>> health = get_system_health()
>>> print(health)
```

---

# Troubleshooting

## Common Issues and Solutions

### Translation Issues

**Problem**: Models not downloading
```bash
# Check internet connection and disk space
df -h  # Check disk space
python -c "import transformers; print(transformers.__version__)"

# Clear cache and retry
rm -rf ~/.cache/huggingface/
python -c "from transformers import MarianMTModel; MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-es-en')"
```

**Problem**: Poor translation quality
```python
# Check language detection accuracy
from rag_api.translation_service import get_translation_service
service = get_translation_service()
lang, conf = service.detect_language("Your text here")
print(f"Detected: {lang} (confidence: {conf:.2f})")

# Enable debug logging
import logging
logging.getLogger('rag_api.translation_service').setLevel(logging.DEBUG)
```

**Problem**: GPU memory errors
```bash
# Force CPU-only mode
export TRANSFORMERS_DEVICE=cpu

# Or reduce batch size in code
# Edit translation_service.py:
# max_length=256  # instead of 512
# num_beams=2     # instead of 4
```

### Performance Issues

**Problem**: Slow translation responses
```python
# Check model caching
from rag_api.translation_service import get_translation_service
service = get_translation_service()
print(f"Cached models: {list(service.models.keys())}")

# Pre-load frequently used models
service._load_model('es')  # Pre-load Spanish
service._load_model('fr')  # Pre-load French
```

**Problem**: High memory usage
```python
# Monitor memory usage
import psutil
process = psutil.Process()
print(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")

# Clear unused models
service._clear_unused_models()  # If implemented
```

### Analytics Issues

**Problem**: Analytics not working
```python
# Check cache configuration
from django.core.cache import caches
print(f"Available caches: {list(caches.all())}")

# Test cache storage
from django.core.cache import cache
cache.set('test_key', 'test_value', 300)
print(f"Cache test: {cache.get('test_key')}")
```

**Problem**: Health endpoint returning errors
```python
# Test analytics functions directly
from rag_api.translation_analytics import get_system_health
try:
    health = get_system_health()
    print("Analytics working:", health)
except Exception as e:
    print(f"Analytics error: {e}")
```

### Character Encoding Issues

**Problem**: Special characters not displaying correctly
```python
# Test character encoding
test_text = "Café naïve résumé"
from rag_api.translation_service import get_translation_service
service = get_translation_service()
fixed_text = service._fix_encoding_issues(test_text)
print(f"Original: {test_text}")
print(f"Fixed: {fixed_text}")
```

**Problem**: Language detection failing
```python
# Test with cleaned text
from rag_api.translation_service import get_translation_service
service = get_translation_service()
original = "Busco apartamento 2BHK $1500 near metro"
cleaned = service._clean_real_estate_text(original)
print(f"Original: {original}")
print(f"Cleaned: {cleaned}")
lang, conf = service.detect_language(cleaned)
print(f"Detected: {lang} ({conf:.2f})")
```

## Debug Mode

### Enable Comprehensive Logging

```python
# Add to Django settings.py
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
        },
        'file': {
            'class': 'logging.FileHandler',
            'filename': 'translation_debug.log',
        },
    },
    'loggers': {
        'rag_api.translation_service': {
            'level': 'DEBUG',
            'handlers': ['console', 'file'],
        },
        'rag_api.translation_analytics': {
            'level': 'DEBUG', 
            'handlers': ['console', 'file'],
        },
    },
}
```

### Test Functions

```python
# Create test_translation.py
def test_all_languages():
    """Test translation for all supported languages"""
    test_phrases = {
        'es': 'Busco apartamento de dos dormitorios',
        'fr': 'Je cherche un appartement de deux chambres',
        'de': 'Ich suche eine Wohnung mit zwei Schlafzimmern',
        'it': 'Cerco un appartamento con due camere da letto',
        'pt': 'Procuro apartamento com dois quartos',
        'ru': 'Ищу квартиру с двумя спальнями',
        'zh': '我在寻找两居室公寓',
        'ja': '2ベッドルームのアパートを探しています',
        'ko': '침실 2개짜리 아파트를 찾고 있습니다',
        'ar': 'أبحث عن شقة بغرفتي نوم',
        'hi': 'मैं दो बेडरूम का अपार्टमेंट ढूंढ रहा हूं'
    }
    
    for lang, phrase in test_phrases.items():
        print(f"\nTesting {lang}: {phrase}")
        result = translate_to_english_guaranteed(phrase)
        print(f"Result: {result['english_query']}")
        print(f"Detected: {result['detected_language']}")
        print(f"Success: {result.get('ui_safe', False)}")

if __name__ == '__main__':
    test_all_languages()
```

## Performance Monitoring

### System Health Monitoring

```python
# Monitor translation system health
def monitor_translation_health():
    """Monitor and report translation system health"""
    from rag_api.translation_analytics import (
        get_system_health,
        get_language_performance,
        get_recent_performance
    )
    
    # Overall health
    health = get_system_health()
    print(f"System Status: {health['status']}")
    print(f"Health Score: {health['health_score']}/100")
    print(f"Success Rate: {health['success_rate']}%")
    
    # Language performance
    lang_perf = get_language_performance()
    print("\nLanguage Performance:")
    for lang, stats in lang_perf.items():
        print(f"  {lang}: {stats['success_rate']:.1f}% success, "
              f"{stats['avg_response_time']:.3f}s avg")
    
    # Recent activity
    recent = get_recent_performance()
    print(f"\nRecent Activity: {len(recent)} requests")
    if recent:
        latest = recent[-1]
        print(f"Latest: {latest['source_language']} -> "
              f"{'✓' if latest['success'] else '✗'} "
              f"({latest['response_time']:.3f}s)")
```

---

*This comprehensive guide covers all aspects of the Django RAG Chatbot translation service. For additional support or feature requests, please refer to the project documentation or contact the development team.*

**Last Updated**: December 2024  
**Version**: 3.0 - Complete Translation Service

---