import os
from pathlib import Path

# System Configuration
class Config:
    # Database Configuration
    MONGODB_URI = os.getenv('MONGODB_URI', 'mongodb://localhost:27017/')
    DATABASE_NAME = 'local'
    COLLECTION_NAME = 'documents'
    
    # AI/ML Model Configuration
    EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
    EMBEDDING_DIMENSION = 384
    MAX_SEQUENCE_LENGTH = 512
    
    # File Paths
    BASE_DIR = Path(__file__).parent  # Current project directory
    DATA_DIR = BASE_DIR / 'data'
    LOGS_DIR = BASE_DIR / 'logs'
    CACHE_DIR = BASE_DIR / 'cache'
    INDEXES_DIR = BASE_DIR / 'indexes'
    
    # Search Configuration
    TOP_K_RESULTS = 5
    SIMILARITY_THRESHOLD = 0.3
    FUZZY_THRESHOLD = 80
    KEYWORD_BOOST = 1.5
    
    # FAISS Configuration
    FAISS_INDEX_PATH = INDEXES_DIR / 'faiss_index.bin'
    PROCESSED_DOCS_PATH = INDEXES_DIR / 'processed_documents.pkl'
    EMBEDDINGS_CACHE_PATH = CACHE_DIR / 'embeddings_cache.pkl'
    
    # Gradio Configuration
    GRADIO_HOST = '0.0.0.0'
    GRADIO_PORT = 7860
    GRADIO_THEME = 'dark'
    
    # Search Weights
    SEMANTIC_WEIGHT = 0.8
    FUZZY_WEIGHT = 0.2
    KEYWORD_WEIGHT = 0.5
    
    # Session Configuration
    SESSION_TIMEOUT = 3600  # 1 hour
    MAX_CONVERSATION_HISTORY = 50
    
    # Performance Settings
    BATCH_SIZE = 100
    MAX_WORKERS = 4
    CACHE_SIZE = 10000
    
    @classmethod
    def ensure_directories(cls):
        """Create necessary directories if they don't exist"""
        directories = [
            cls.DATA_DIR,
            cls.LOGS_DIR,
            cls.CACHE_DIR,
            cls.INDEXES_DIR
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            
    @classmethod
    def get_log_file_path(cls, log_name):
        """Get path for a specific log file"""
        return cls.LOGS_DIR / f'{log_name}.log'
    
    @classmethod
    def get_cache_file_path(cls, cache_name):
        """Get path for a specific cache file"""
        return cls.CACHE_DIR / f'{cache_name}.pkl'

# Initialize directories on import
Config.ensure_directories()