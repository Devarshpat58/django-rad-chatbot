# Django RAG Chatbot

A sophisticated AI-powered multilingual real estate chatbot system that combines Retrieval-Augmented Generation (RAG) with advanced MarianMT translation capabilities.

## 🌟 Key Features

- **🌍 Multilingual Support**: MarianMT translation for 11 languages with offline operation
- **🧠 Enhanced Language Detection**: Domain-specific text cleaning with confidence thresholds  
- **🔍 RAG-Powered Search**: Semantic search with AI-generated responses
- **💻 Modern Web Interface**: Django-based responsive UI with real-time chat
- **⚡ GPU Acceleration**: Automatic GPU detection for optimal performance
- **📊 Structured Responses**: Consistent API format for seamless integration

## 🏗️ System Architecture

The Django RAG Chatbot implements a modern web architecture optimized for multilingual real estate search:

- **Presentation Layer**: Django web interface with responsive design
- **Application Layer**: Django REST framework with API endpoints
- **Translation Layer**: MarianMT models with enhanced language detection
- **Intelligence Layer**: RAG system with semantic search
- **Data Layer**: MongoDB with vector indexes

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Django 4.2+
- MongoDB 4.0+
- 8GB RAM (16GB recommended)
- GPU optional (for translation acceleration)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Devarshpat58/django-rad-chatbot
   cd django_rag_api
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure MongoDB:**
   - Ensure MongoDB is running on localhost:27017
   - Import your Airbnb property data into a collection
   - Update database settings in `config.py` if needed

5. **Initialize the system:**
   ```bash
   python setup.py --full-setup
   ```
   This will:
   - Build vocabulary from your MongoDB data
   - Generate embeddings and create search indexes
   - Download and setup AI models
   - Optimize the system for property search

6. **Run Django migrations:**
   ```bash
   python manage.py migrate
   ```

7. **Create superuser (optional):**
   ```bash
   python manage.py createsuperuser
   ```

8. **Start the development server:**
   ```bash
   python manage.py runserver
   ```

## 🌐 Access Points

### Web Interface
- **Chat Interface**: http://localhost:8000/chat/
- **Search Interface**: http://localhost:8000/search/
- **Dashboard**: http://localhost:8000/dashboard/
- **Documentation**: http://localhost:8000/docs/

### API Endpoints
- **Chat API**: `POST /api/v1/chat/`
- **Search API**: `POST /api/v1/search/`
- **System Status**: `GET /api/v1/status/`
- **Health Check**: `GET /api/v1/health/`

## 🌍 Multi-Language Translation Service

### Supported Languages
The system includes a self-contained translation service that supports:
- **Spanish** (Español)
- **French** (Français) 
- **German** (Deutsch)
- **Italian** (Italiano)
- **Portuguese** (Português)
- **English** (default)

### Translation Features
- **🔑 No API Keys Required**: Completely self-contained using built-in Python libraries
- **⚡ Fast Processing**: Instant pattern-based translation with LRU caching
- **🎯 Real Estate Focused**: Specialized patterns for property search terminology
- **🔍 Smart Detection**: Regex-based language detection with confidence scoring
- **📊 Translation Metadata**: API responses include translation information

### Example Translations
```

2. **Install Dependencies**
```bash
pip install -r requirements.txt
```

3. **Configure Environment**
```bash
# Create .env file
MONGODB_URI=mongodb://localhost:27017
DEBUG=True
SECRET_KEY=your-secret-key
```

4. **Database Setup**
```bash
python manage.py migrate
python manage.py initialize_rag
```

5. **Start Development Server**
```bash
python manage.py runserver
```

6. **Access Application**
- **Web Interface**: http://localhost:8000/
- **Chat Interface**: http://localhost:8000/chat/
- **Admin Panel**: http://localhost:8000/admin/

## 🌐 Translation Service

### MarianMT Integration

The system uses Helsinki-NLP MarianMT models for high-quality neural machine translation:

#### Supported Languages
- Spanish (es), French (fr), German (de), Italian (it)
- Portuguese (pt), Russian (ru), Chinese (zh), Japanese (ja)
- Korean (ko), Arabic (ar), Hindi (hi)

#### Translation Pipeline
```python
{
    'english_query': 'Translated English text',
    'detected_language': 'es', 
    'translation_needed': True
}
```

#### Key Features
- **Helsinki-NLP Models**: Using Helsinki-NLP/opus-mt-{lang}-en models
- **MarianTokenizer**: Proper SentencePiece tokenization for accurate translation
- **Model Caching**: Efficient caching system to avoid reloading models
- **GPU Acceleration**: Automatic GPU detection and utilization when available
- **Structured Response**: Dictionary format for seamless integration

## 📚 API Reference

### Translation API

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

### Chat API

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

## 🔧 Configuration

### Translation Service Configuration

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

## 📁 Project Structure

```
django-rad-chatbot/
├── django_rag_project/          # Django project settings
├── rag_api/                     # RAG API application
│   ├── services.py              # Core RAG services
│   ├── translation_service.py   # MarianMT translation service
│   └── management/commands/     # Django management commands
├── web_interface/               # Web interface application
├── templates/                   # Django templates
├── static/                      # Static files (CSS, JS, images)
├── documentation/               # Project documentation
└── requirements.txt             # Python dependencies
```

## 🛠️ Development

### Running Tests
```bash
python manage.py test
```

### Code Quality
```bash
# Format code
black .

# Lint code
flake8 .

# Type checking
mypy .
```

## 🚀 Production Deployment

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

## 🔍 Troubleshooting

### Common Issues

**Translation Service Issues**:
- Check GPU availability: `python -c "import torch; print(torch.cuda.is_available())"`
- Clear model cache: `rm -rf ./models/`
- Verify language detection accuracy

**Web Interface Issues**:
- Check Django logs: `python manage.py runserver --verbosity=2`
- Verify database connection
- Collect static files: `python manage.py collectstatic`

## 📖 Documentation

For complete documentation, see:
- [Complete Project Documentation](documentation/COMPLETE_PROJECT_DOCUMENTATION.md)
- [Translation Service Documentation](documentation/TRANSLATION_SERVICE.md)
- [Enhanced Language Detection](documentation/ENHANCED_LANGUAGE_DETECTION.md)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Helsinki-NLP for MarianMT models
- Hugging Face for transformers library
- Django community for the web framework
- MongoDB for the database solution

---

**Happy Searching! 🏡**

