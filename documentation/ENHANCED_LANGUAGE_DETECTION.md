# Enhanced Language Detection for Real Estate Queries

## Overview

The translation service has been significantly enhanced to better handle real estate-related queries in the Airbnb chatbot context. The improvements focus on domain-specific text cleaning, robust language detection with fallback mechanisms, keyword-based overrides, and comprehensive logging.

## Key Improvements

### 1. Domain-Specific Text Cleaning

The `_clean_real_estate_text()` method removes noise that commonly appears in real estate queries:

#### Currency Handling
- **Symbols**: ₹, $, €, £, ¥, ₩, ₽, ¢
- **Codes**: INR, USD, EUR, GBP, JPY, KRW, RUB, CAD, AUD, CHF, CNY
- **Words**: rupees, dollars, euros, pounds, yen, won

#### Real Estate Units
- **Room counts**: 2 BHK, 3 bedroom, 4 beds
- **Area measurements**: 1200 sqft, 100 sq.ft, 80 square meters
- **Bathroom counts**: 2 bathroom, 1 bath, 3 toilets
- **Property types**: studio, apartment, flat, villa, house, bungalow, penthouse, duplex
- **Floor information**: 5th floor, 2 storey, 3 stories
- **Furnishing**: furnished, unfurnished, semi-furnished, fully-furnished
- **Amenities**: parking, garage, balcony, terrace, garden, pool, gym, elevator, lift

#### Location Noise
- **Proximity indicators**: near, close to, next to, opposite, behind, front of
- **Landmarks**: metro, station, airport, mall, market, school, hospital, park
- **Directions**: north, south, east, west, central, downtown, uptown
- **Address components**: road, street, avenue, lane, colony, sector, phase, block

### 2. Enhanced Language Detection Pipeline

The detection process follows a multi-layered approach:

1. **Input Logging**: Raw input is logged for debugging
2. **Domain Cleaning**: Real estate noise is removed
3. **Keyword Override Check**: Language-specific keywords are checked first
4. **Primary Detection**: Uses langdetect library
5. **Fallback Detection**: Uses langid library if confidence < 0.90
6. **Confidence Comparison**: Chooses the detection with higher confidence
7. **Final Decision Logging**: Complete detection process is logged

### 3. Keyword-Based Override System

The system includes comprehensive keyword dictionaries for major languages:

#### Spanish (es)
- **Real estate**: apartamento, piso, casa, vivienda, alquiler, venta, dormitorio, baño, cocina, salon, terraza, garaje
- **Common words**: hola, gracias, por favor, donde, como, que, cuando, precio, disponible, busco, necesito

#### French (fr)
- **Real estate**: appartement, maison, studio, chambre, salle de bain, cuisine, salon, balcon, garage, location, vente
- **Common words**: bonjour, merci, si vous plait, ou, comment, que, prix, disponible, cherche, besoin

#### German (de)
- **Real estate**: wohnung, haus, zimmer, schlafzimmer, badezimmer, küche, wohnzimmer, balkon, garage, miete, verkauf
- **Common words**: hallo, danke, bitte, wo, wie, was, wann, preis, verfügbar, suche, brauche

#### Italian (it)
- **Real estate**: appartamento, casa, camera, bagno, cucina, soggiorno, terrazzo, garage, affitto, vendita
- **Common words**: ciao, grazie, per favore, dove, come, che, prezzo, disponibile, cerco, ho bisogno

#### Portuguese (pt)
- **Real estate**: apartamento, casa, quarto, banheiro, cozinha, sala, varanda, garagem, aluguel, venda
- **Common words**: ola, obrigado, por favor, onde, como, que, preço, disponível, procuro, preciso

#### Hindi (hi)
- **Real estate**: घर, मकान, फ्लैट, कमरा, बेडरूम, बाथरूम, रसोई, बालकनी, किराया, बिक्री
- **Common words**: नमस्ते, धन्यवाद, कृपया, कहाँ, कैसे, क्या, कब, कीमत, उपलब्ध, खोज, चाहिए

#### Russian (ru)
- **Real estate**: квартира, дом, комната, спальня, ванная, кухня, балкон, гараж, аренда, продажа
- **Common words**: привет, спасибо, пожалуйста, где, как, что, цена, доступно, ищу, нужно

#### Chinese (zh)
- **Real estate**: 公寓, 房子, 房间, 卧室, 浴室, 厨房, 阳台, 车库, 租金, 出售
- **Common words**: 你好, 谢谢, 请, 哪里, 怎么, 什么, 价格, 可用, 寻找, 需要

#### Japanese (ja)
- **Real estate**: アパート, 家, 部屋, 寝室, バスルーム, キッチン, バルコニー, ガレージ, 賃貸, 販売
- **Common words**: こんにちは, ありがとう, お願いします, どこ, どうやって, 何, 価格, 利用可能, 探している, 必要

#### Arabic (ar)
- **Real estate**: شقة, بيت, غرفة, غرفة نوم, حمام, مطبخ, شرفة, مرآب, إيجار, بيع
- **Common words**: مرحبا, شكرا, من فضلك, أين, كيف, ماذا, سعر, متاح, أبحث, أحتاج

### 4. Comprehensive Logging

The system now logs every step of the detection process:

- **Raw Input**: Original text as received
- **Cleaned Text**: Text after domain-specific cleaning
- **Keyword Overrides**: Any language-specific keywords found
- **Primary Detection**: Langdetect results with confidence
- **Fallback Detection**: Langid results when confidence is low
- **Final Decision**: Chosen language with confidence score
- **Translation Results**: Success/failure of translation attempts

### 5. Robust Fallback Mechanisms

The system includes multiple fallback layers:

1. **Langid Fallback**: When langdetect confidence < 0.90
2. **Pattern-based Detection**: Character and word pattern matching
3. **Graceful Error Handling**: Continues operation even if libraries fail
4. **Default to English**: Safe fallback when all detection methods fail

## Configuration

### Dependencies
- **langdetect**: Primary language detection (required)
- **langid**: Fallback language detection (optional but recommended)
- **transformers**: For MarianMT translation models (optional)

### Confidence Thresholds
- **Keyword Override**: 0.95 confidence
- **Fallback Trigger**: < 0.90 confidence from primary detection
- **Minimum Acceptable**: 0.85 confidence for fallback acceptance

## Usage Examples

### Basic Usage
```python
from rag_api.translation_service import translate_to_english

# Real estate query with noise
result = translate_to_english("Busco apartamento 2 BHK ₹25000 near metro")
# Result: {'english_query': 'Looking for apartment near', 'detected_language': 'es', 'translation_needed': True}
```

### Direct Detection
```python
from rag_api.translation_service import get_translation_service

service = get_translation_service()
lang, confidence = service.detect_language("Bonjour, je cherche un appartement")
# Result: ('fr', 0.95)
```

## Testing

The implementation includes comprehensive test cases covering:
- English queries with real estate noise
- Spanish queries with currency and units
- French queries with measurements
- German queries with property types
- Mixed language queries common in international markets
- Short queries that benefit from keyword overrides
- Edge cases with heavy noise patterns

## Performance Considerations

- **Caching**: Language detection results are cached using `@lru_cache`
- **Lazy Loading**: Translation models are loaded only when needed
- **Efficient Cleaning**: Regex patterns are optimized for performance
- **Graceful Degradation**: Fallback methods ensure continued operation

## Logging Configuration

To enable detailed logging, configure Django logging:

```python
LOGGING = {
    'loggers': {
        'rag_api.translation_service': {
            'level': 'DEBUG',  # or 'INFO' for less verbose output
            'handlers': ['console'],
        },
    },
}
```

The enhanced language detection system provides significantly improved accuracy for real estate queries while maintaining robust fallback mechanisms and comprehensive logging for debugging and monitoring.