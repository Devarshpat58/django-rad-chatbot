# Django RAG Chatbot - Complete Project Documentation

**Version**: 2.0  
**Date**: June 2024  
**System**: Advanced AI-Powered Multilingual Real Estate Chatbot with MarianMT Translation

---

## TABLE OF CONTENTS

1. [EXECUTIVE SUMMARY](#executive-summary)
2. [PROJECT OVERVIEW](#project-overview)
3. [SYSTEM ARCHITECTURE](#system-architecture)
4. [CORE COMPONENTS](#core-components)
5. [TRANSLATION SERVICE](#translation-service)
6. [WEB INTERFACE](#web-interface)
7. [API REFERENCE](#api-reference)
8. [INSTALLATION AND SETUP](#installation-and-setup)
9. [CONFIGURATION GUIDE](#configuration-guide)
10. [TROUBLESHOOTING](#troubleshooting)
11. [MAINTENANCE](#maintenance)

---

# EXECUTIVE SUMMARY

The Django RAG Chatbot is a sophisticated AI-powered multilingual real estate chatbot system that combines Retrieval-Augmented Generation (RAG) with advanced translation capabilities. The system uses MarianMT models for high-quality translation, enhanced language detection, and provides intelligent property search through a modern web interface.

## Key Features
- **Multilingual Support**: MarianMT translation for 11 languages with offline operation
- **Enhanced Language Detection**: Domain-specific text cleaning with confidence thresholds
- **RAG-Powered Search**: Semantic search with AI-generated responses
- **Modern Web Interface**: Django-based responsive UI with real-time chat
- **GPU Acceleration**: Automatic GPU detection for optimal performance
- **Structured Responses**: Consistent API format for seamless integration

## Business Value
- **Global Accessibility**: Support for multiple languages without external dependencies
- **High-Quality Translation**: Professional-grade MarianMT models for accurate translation
- **Intelligent Search**: AI-powered property discovery with natural language queries
- **Scalable Architecture**: Django framework with production-ready deployment
- **Cost-Effective**: No API keys or external translation services required

---

# PROJECT OVERVIEW

The Django RAG Chatbot represents a comprehensive integration of modern AI/ML technologies for multilingual real estate property search. By combining MarianMT translation, enhanced language detection, RAG-powered search, and a modern Django web interface, it provides users with a powerful yet intuitive platform for finding properties across language barriers.

## System Purpose
A comprehensive Django-based RAG chatbot system designed for multilingual real estate property search. The system provides intelligent translation capabilities with MarianMT models, semantic search with AI responses, and a modern web interface for seamless user interaction.

## Target Users
- **International Property Seekers**: Users searching in multiple languages
- **Real Estate Professionals**: Agents serving diverse clientele
- **Property Managers**: Managing multilingual customer inquiries
- **Developers**: Building multilingual property applications

## Key Success Metrics
- **Translation Quality**: Accuracy of multilingual queries
- **Search Relevance**: Quality of property recommendations
- **System Performance**: Response time under 2 seconds
- **User Experience**: Intuitive interface across languages

---

# SYSTEM ARCHITECTURE

## High-Level Design

The Django RAG Chatbot implements a modern web architecture optimized for multilingual real estate search:

- **Presentation Layer**: Django web interface with responsive design
- **Application Layer**: Django REST framework with API endpoints
- **Translation Layer**: MarianMT models with enhanced language detection
- **Intelligence Layer**: RAG system with semantic search
- **Data Layer**: MongoDB with vector indexes
- **Infrastructure Layer**: Configuration and deployment

## Component Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              PRESENTATION LAYER                            │
├─────────────────────────────────────────────────────────────────────────────┤
│ ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐          │
│ │   Django Web    │    │   REST API      │    │   Admin Panel   │          │
│ │   Interface     │    │   Endpoints     │    │                 │          │
│ └─────────────────┘    └─────────────────┘    └─────────────────┘          │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              APPLICATION LAYER                             │
├─────────────────────────────────────────────────────────────────────────────┤
│ ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐          │
│ │   Django Views  │    │  URL Routing    │    │   Serializers   │          │
│ │                 │    │                 │    │                 │          │
│ └─────────────────┘    └─────────────────┘    └─────────────────┘          │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              TRANSLATION LAYER                             │
├─────────────────────────────────────────────────────────────────────────────┤
│ ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐          │
│ │   MarianMT      │    │Enhanced Language│    │   GPU Support   │          │
│ │   Models        │    │   Detection     │    │                 │          │
│ └─────────────────┘    └─────────────────┘    └─────────────────┘          │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              INTELLIGENCE LAYER                            │
├─────────────────────────────────────────────────────────────────────────────┤
│ ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐          │
│ │   RAG System    │    │Semantic Search  │    │   AI Response   │          │
│ │                 │    │   Engine        │    │   Generation    │          │
│ └─────────────────┘    └─────────────────┘    └─────────────────┘          │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                                DATA LAYER                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│ ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐          │
│ │   MongoDB       │    │  Vector Index   │    │  Model Cache    │          │
│ │   Database      │    │  (Embeddings)   │    │                 │          │
│ └─────────────────┘    └─────────────────┘    └─────────────────┘          │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

# CORE COMPONENTS

## 1. Translation Service (MarianMT Integration)
**High-quality neural machine translation with offline operation**

### Key Features:
- **Helsinki-NLP Models**: Using Helsinki-NLP/opus-mt-{lang}-en models for 11 languages
- **MarianTokenizer**: Proper SentencePiece tokenization for accurate translation
- **Model Caching**: Efficient caching system to avoid reloading models
- **GPU Acceleration**: Automatic GPU detection and utilization when available
- **Structured Response**: Dictionary format with english_query, detected_language, translation_needed

### Supported Languages:
- Spanish (es), French (fr), German (de), Italian (it)
- Portuguese (pt), Russian (ru), Chinese (zh), Japanese (ja)
- Korean (ko), Arabic (ar), Hindi (hi)

### Translation Pipeline:
```python
{
    'english_query': 'Translated English text',
    'detected_language': 'es',
    'translation_needed': True
}
```

## 2. Enhanced Language Detection
**Domain-specific language detection with confidence thresholds**

### Features:
- **Real Estate Domain Cleaning**: Specialized text preprocessing for property queries
- **Confidence Thresholds**: Multiple detection attempts with confidence averaging
- **Unicode Handling**: Proper Unicode processing for international text
- **Fallback Mechanisms**: Multiple detection libraries for reliability

## 3. Django Web Interface
**Modern responsive web interface for chatbot interaction**

### Components:
- **Chat Interface**: Real-time conversation with property search
- **Dashboard**: System overview and statistics
- **Search Interface**: Advanced property search capabilities
- **Documentation**: Built-in help and API documentation

## 4. RAG System Integration
**Retrieval-Augmented Generation for intelligent property search**

### Capabilities:
- **Semantic Search**: Vector-based similarity search
- **Context-Aware Responses**: AI-generated summaries based on search results
- **Property Matching**: Intelligent property recommendation
- **Natural Language Processing**: Understanding complex property queries

---

# TRANSLATION SERVICE

## MarianMT Implementation

### Model Architecture
The translation service uses Helsinki-NLP MarianMT models, which are specifically designed for high-quality machine translation:

```python
# Model Loading
model_name = f"Helsinki-NLP/opus-mt-{source_lang}-en"
model = MarianMTModel.from_pretrained(model_name)
tokenizer = MarianTokenizer.from_pretrained(model_name)
```

### Translation Process
1. **Language Detection**: Enhanced detection with domain-specific cleaning
2. **Model Loading**: Dynamic loading of language-specific models
3. **Tokenization**: SentencePiece tokenization for input processing
4. **Translation**: Neural translation with beam search
5. **Post-processing**: Output cleaning and formatting

### Performance Optimization
- **Model Caching**: Models cached in memory for subsequent requests
- **GPU Acceleration**: Automatic GPU utilization when available
- **Batch Processing**: Efficient handling of multiple queries
- **Memory Management**: Optimized model loading and unloading

### Quality Assurance
- **Beam Search**: num_beams=4 for high-quality translations
- **Deterministic Output**: Consistent translation results
- **Error Handling**: Graceful fallback to pattern-based translation
- **Validation**: Translation quality verification

---

# WEB INTERFACE

## Django Application Structure

### URL Configuration
```python
# Main URLs
urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/', include('rag_api.urls')),
    path('', include('web_interface.urls')),
]

# Web Interface URLs
urlpatterns = [
    path('', views.index, name='index'),
    path('chat/', views.chat, name='chat'),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('search/', views.search, name='search'),
    path('documentation/', views.documentation, name='documentation'),
]
```

### Views and Templates
- **Base Template**: Responsive layout with navigation
- **Chat Interface**: Real-time messaging with property search
- **Dashboard**: System statistics and performance metrics
- **Search**: Advanced property search with filters
- **Documentation**: API reference and usage guides

### Static Assets
- **CSS**: Modern styling with responsive design
- **JavaScript**: Interactive features and AJAX communication
- **Images**: Icons and branding assets
- **Fonts**: Typography optimization

---

# API REFERENCE

## Translation API

### Translate to English
**Endpoint**: `POST /api/translate/`

**Request**:
```json
{
    "query": "Hola, necesito un apartamento en Madrid"
}
```

**Response**:
```json
{
    "english_query": "Hi, I need an apartment in Madrid",
    "detected_language": "es",
    "translation_needed": true
}
```

## Chat API

### Process Query
**Endpoint**: `POST /api/chat/`

**Request**:
```json
{
    "query": "Find 2 bedroom apartments under $2000",
    "session_id": "user123"
}
```

**Response**:
```json
{
    "success": true,
    "response": "I found several 2-bedroom apartments under $2000...",
    "results": [...],
    "metadata": {
        "num_results": 5,
        "processing_time": 1.2
    }
}
```

---

# INSTALLATION AND SETUP

## Prerequisites
- Python 3.8+
- Django 4.2+
- MongoDB 4.0+
- 8GB RAM (16GB recommended)
- GPU optional (for translation acceleration)

## Installation Steps

### 1. Clone Repository
```bash
git clone <repository-url>
cd django-rad-chatbot
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Environment
```bash
# Create .env file
MONGODB_URI=mongodb://localhost:27017
DEBUG=True
SECRET_KEY=your-secret-key
```

### 4. Database Setup
```bash
python manage.py migrate
python manage.py initialize_rag
```

### 5. Start Development Server
```bash
python manage.py runserver
```

### 6. Access Application
- **Web Interface**: http://localhost:8000/
- **Chat Interface**: http://localhost:8000/chat/
- **Admin Panel**: http://localhost:8000/admin/

## Production Deployment

### Docker Deployment
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["gunicorn", "django_rag_project.wsgi:application"]
```

### Environment Variables
```bash
export DJANGO_SETTINGS_MODULE=django_rag_project.settings
export MONGODB_URI=mongodb://production-server:27017
export DEBUG=False
export ALLOWED_HOSTS=yourdomain.com
```

---

# CONFIGURATION GUIDE

## Translation Service Configuration

### Model Settings
```python
# MarianMT Configuration
TRANSLATION_CONFIG = {
    'model_cache_dir': './models/',
    'max_length': 512,
    'num_beams': 4,
    'early_stopping': True,
    'gpu_acceleration': True
}
```

### Language Detection Settings
```python
# Language Detection Configuration
LANGUAGE_DETECTION_CONFIG = {
    'confidence_threshold': 0.85,
    'max_attempts': 3,
    'domain_cleaning': True,
    'fallback_enabled': True
}
```

## RAG System Configuration

### Search Settings
```python
# RAG Configuration
RAG_CONFIG = {
    'top_k_results': 5,
    'similarity_threshold': 0.3,
    'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2',
    'response_max_length': 500
}
```

---

# TROUBLESHOOTING

## Common Issues

### Translation Service Issues

**Problem**: Models not loading
**Solution**: 
```bash
# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"

# Clear model cache
rm -rf ./models/
python manage.py initialize_rag
```

**Problem**: Poor translation quality
**Solution**:
- Verify language detection accuracy
- Check model cache integrity
- Ensure proper tokenization

### Web Interface Issues

**Problem**: Chat not responding
**Solution**:
```bash
# Check Django logs
python manage.py runserver --verbosity=2

# Verify database connection
python manage.py shell
>>> from rag_api.services import *
>>> # Test connection
```

**Problem**: Static files not loading
**Solution**:
```bash
python manage.py collectstatic
python manage.py runserver --insecure
```

## Performance Optimization

### Memory Usage
- Monitor model cache size
- Implement model unloading for unused languages
- Use CPU-only mode if GPU memory limited

### Response Time
- Enable model caching
- Use GPU acceleration when available
- Optimize database queries

---

# MAINTENANCE

## Regular Tasks

### Model Updates
```bash
# Update MarianMT models
pip install --upgrade transformers
python manage.py initialize_rag --update-models
```

### Database Maintenance
```bash
# Rebuild indexes
python manage.py initialize_rag --rebuild-indexes

# Clean old sessions
python manage.py cleanup_sessions
```

### Performance Monitoring
```bash
# Check system stats
python manage.py shell
>>> from rag_api.services import RAGService
>>> service = RAGService()
>>> print(service.get_system_stats())
```

## Backup Procedures

### Model Backup
```bash
# Backup model cache
tar -czf models_backup.tar.gz ./models/
```

### Database Backup
```bash
# MongoDB backup
mongodump --db airbnb_database --out ./backup/
```

---

*Django RAG Chatbot v2.0 - Complete Documentation*
*Last updated: June 2024*