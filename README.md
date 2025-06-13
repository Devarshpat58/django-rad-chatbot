# Django RAG API - Advanced Property Search System

## 🏠 Overview

A sophisticated Retrieval-Augmented Generation (RAG) system built with Django for intelligent property search. The system combines semantic search, AI-powered analysis, and natural language processing to provide comprehensive property recommendations with detailed insights.

## ✨ Key Features

- **🔍 Advanced Semantic Search**: AI-powered property search using sentence transformers
- **📊 Intelligent Query Processing**: Natural language understanding with numeric constraints
- **🤖 AI-Generated Summaries**: 200+ word comprehensive property analysis
- **📋 Query-Relevant Fields**: Smart field extraction based on user intent
- **📄 Complete JSON Data**: Full source documents from MongoDB
- **💬 Interactive Chat Interface**: Modern, responsive web UI
- **🔗 REST API**: Full API access for integrations
- **📱 Mobile Responsive**: Works on all device sizes

## 🛠️ Technology Stack

- **Backend**: Django 4.x, Python 3.8+
- **Database**: MongoDB
- **AI/ML**: Sentence Transformers, FAISS, spaCy
- **Search**: Hybrid semantic + keyword search
- **Frontend**: Modern HTML/CSS/JS with responsive design
- **API**: Django REST Framework

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- MongoDB installed and running
- Git
- 8GB+ RAM recommended for AI models

### Installation

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
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

## 💬 Using the Chat Interface

### Example Queries:
```
Find 2 bedroom apartments under $150 per night
Show me luxury properties with pools near downtown
Pet-friendly houses for families with parking
Studio apartments between $50-100 with WiFi
Properties accommodating 6 guests with kitchen access
```

### Response Features:
- **AI Summary**: 200+ word intelligent analysis
- **Query Fields**: Key information relevant to your search
- **Source JSON**: Complete MongoDB document data
- **Performance Stats**: Search time, relevance scores
- **Interactive UI**: Collapsible sections, mobile-friendly

## 🔧 Configuration

### Database Settings (`config.py`):
```python
MONGO_URI = "mongodb://localhost:27017/"
DATABASE_NAME = "airbnb_data"
COLLECTION_NAME = "listings"
```

### Search Parameters:
```python
TOP_K_RESULTS = 10
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
MAX_SUMMARY_LENGTH = 400
```

## 📁 Project Structure

```
django_rag_api/
├── core_system.py          # Main RAG system logic
├── query_processor.py      # Query understanding & processing
├── utils.py               # Utility functions & managers
├── config.py              # Configuration settings
├── setup.py               # System initialization
├── rag_api/               # Django REST API
│   ├── views.py           # API endpoints
│   ├── services.py        # RAG service integration
│   └── models.py          # Django models
├── web_interface/         # Web UI
│   ├── views.py           # Web views
│   └── templates/         # HTML templates
├── templates/             # Base templates
├── static/                # CSS/JS assets
└── documentation/         # Additional docs
```

## 🧪 API Usage Examples

### Chat API
```bash
curl -X POST http://localhost:8000/api/v1/chat/ \
     -H "Content-Type: application/json" \
     -d '{"message": "Find 2 bedroom apartments under $200"}'
```

### Search API
```bash
curl -X POST http://localhost:8000/api/v1/search/ \
     -H "Content-Type: application/json" \
     -d '{"query": "luxury properties with pools", "max_results": 5}'
```

## 🔄 System Requirements

### Minimum:
- 4GB RAM
- 2GB storage
- Python 3.8+
- MongoDB 4.4+

### Recommended:
- 8GB+ RAM
- 5GB storage
- SSD storage
- GPU support (optional, for faster processing)

## 📊 Performance Optimization

### First-Time Setup:
- Initial setup may take 10-30 minutes depending on data size
- Models are downloaded and cached locally
- Indexes are pre-built for fast search

### Production Tips:
- Use Redis for caching in production
- Consider GPU acceleration for large datasets
- Monitor memory usage with large vocabularies
- Use gunicorn for production deployment

## 🐛 Troubleshooting

### Common Issues:

**"System not initialized" error:**
```bash
python setup.py --full-setup
```

**MongoDB connection issues:**
- Verify MongoDB is running: `mongosh`
- Check connection string in `config.py`

**Memory issues during setup:**
- Reduce batch size in setup configuration
- Ensure adequate RAM available

**Missing dependencies:**
```bash
pip install --upgrade -r requirements.txt
```

### Getting Help:
1. Check the logs in `logs/` directory
2. Visit the system status page: `/api/v1/status/`
3. Review the documentation: `/docs/`

## 🚀 Deployment

### Development:
```bash
python manage.py runserver 0.0.0.0:8000
```

### Production (example with gunicorn):
```bash
gunicorn django_rag_project.wsgi:application --bind 0.0.0.0:8000 --workers 4
```

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📞 Support

For questions or issues:
- Create an issue on GitHub
- Check the documentation at `/docs/`
- Review the troubleshooting section above

## 🎯 Next Steps

After successful setup:
1. Import your property data into MongoDB
2. Run the full setup to build indexes
3. Test with the chat interface
4. Explore the API endpoints
5. Customize for your specific use case

---

**Happy Searching! 🏡**