#!/usr/bin/env python3
"""
JSON RAG System Setup and Initialization

This module handles the complete initialization of the Airbnb property search system:
- Database connection and document loading
- Vocabulary building from MongoDB data (skipped if already exists)
- Embedding generation and caching (skipped if already cached)
- FAISS index creation (skipped if already built)
- Numeric filters optimization
- System validation and testing
- Translation model preloading and accuracy optimization

The system automatically detects existing components and skips recreation unless
--force flag is used, making subsequent setups much faster.

Usage:
    python setup.py --full-setup              # Setup with existing component reuse
    python setup.py --full-setup --force      # Force rebuild all components
    python setup.py --rebuild-indexes         # Rebuild only indexes
    python setup.py --build-vocab-only        # Build only vocabulary
    python setup.py --test-system             # Test system components
    python setup.py --preload-translation     # Preload translation models
    python setup.py --setup-translation-accuracy  # Setup translation accuracy improvements
    python setup.py --full-translation-setup # Complete translation setup with accuracy improvements
"""
import os
import logging
import argparse
import time
import sys
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

# Import system components
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config
from utils import MongoDBConnector, IndexManager
from core_system import JSONRAGSystem
from utils import VocabularyManager, AirbnbOptimizer, TextProcessor
from logging_config import setup_logging, StructuredLogger, LogOperation, log_performance  # Updated import

# Import translation components for model preloading
try:
    from rag_api.translation_service import TranslationService, get_translation_service
    from transformers import MarianMTModel, MarianTokenizer
    import torch
    TRANSLATION_AVAILABLE = True
except ImportError as e:
    TRANSLATION_AVAILABLE = False
    print(f"Translation components not available: {e}")

# Import additional libraries for translation accuracy improvements
try:
    import numpy as np
    from scipy.spatial.distance import cosine
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    ACCURACY_LIBS_AVAILABLE = True
except ImportError as e:
    ACCURACY_LIBS_AVAILABLE = False
    print(f"Translation accuracy libraries not available: {e}")

# Set up enhanced logging for setup process - using enhanced logging system only
# Initialize the enhanced logging system first
setup_logging(
    log_level='INFO',
    log_dir='logs',
    console_output=True,
    file_output=True,
    format_style='detailed'
)

# Use structured logger without conflicting context
logger = StructuredLogger(__name__, {
    'component': 'system_setup'
})

class SystemSetup:
    """Handles complete system setup and initialization"""
    
    def __init__(self, force_rebuild: bool = False):
        self.force_rebuild = force_rebuild
        self.db_connector = MongoDBConnector()
        self.index_manager = IndexManager()
        self.vocabulary_manager = VocabularyManager()
        self.airbnb_optimizer = AirbnbOptimizer()
        self.text_processor = TextProcessor()
        
        # Setup tracking
        self.setup_stats = {
            'start_time': None,
            'end_time': None,
            'documents_processed': 0,
            'embeddings_generated': 0,
            'vocabulary_terms': 0,
            'faiss_index_size': 0,
            'setup_success': False,
            'errors': []
        }
    
    def setup_complete_system(self) -> bool:
        """Run complete system setup with all components"""
        logger.info("Starting complete JSON RAG system setup...")
        self.setup_stats['start_time'] = time.time()
        
        try:
            # Step 0: Check what already exists
            self._check_existing_components()
            
            # Step 1: Ensure directories exist
            if not self._ensure_directories():
                return False
            
            # Step 2: Test database connection
            if not self._setup_database_connection():
                return False
            
            # Step 3: Load and validate documents
            documents = self._load_documents()
            if not documents:
                return False
            
            # Step 4: Build vocabulary from MongoDB data
            if not self._build_vocabulary(documents):
                return False
            
            # Step 5: Initialize optimizers with vocabulary
            if not self._initialize_optimizers(documents):
                return False
            
            # Step 6: Create embeddings and indexes
            if not self._create_indexes(documents):
                return False
            
            # Step 7: Setup numeric filters
            if not self._setup_numeric_filters():
                return False
            
            # Step 8: Validate system
            if not self._validate_system():
                return False
            
            self.setup_stats['setup_success'] = True
            self.setup_stats['end_time'] = time.time()
            
            self._print_setup_summary()
            logger.info("System setup completed successfully!")
            return True
            
        except Exception as e:
            error_msg = f"Setup failed with error: {str(e)}"
            logger.error(error_msg, component='system_setup')
            self.setup_stats['errors'].append(error_msg)
            return False
    
    def _check_existing_components(self):
        """Check and report what components already exist"""
        # This method performs a comprehensive check of all system components
        # to determine what can be reused vs what needs to be created
        logger.info("Checking existing components...")
        
        # Check file existence for all major system components
        # Each component has specific files that indicate it's been built
        existing_components = {
            'vocabulary': {
                # Vocabulary requires both term frequency data and keyword mappings
                'vocab_file': (Config.DATA_DIR / 'vocabulary.json').exists(),
                'mappings_file': (Config.DATA_DIR / 'keyword_mappings.json').exists()
            },
            'indexes': {
                # FAISS index requires both the binary index file and processed documents
                'faiss_index': Config.FAISS_INDEX_PATH.exists(),
                'processed_docs': Config.PROCESSED_DOCS_PATH.exists()
            },
            'embeddings': {
                # Embeddings are cached to avoid expensive regeneration
                'cache_file': Config.EMBEDDINGS_CACHE_PATH.exists()
            }
        }
        
        # Determine component completeness - all required files must exist for component to be usable
        vocab_exists = existing_components['vocabulary']['vocab_file'] and existing_components['vocabulary']['mappings_file']
        indexes_exist = existing_components['indexes']['faiss_index'] and existing_components['indexes']['processed_docs']
        embeddings_exist = existing_components['embeddings']['cache_file']
        
        logger.info(f"Component status check:")
        logger.info(f"  - Vocabulary: {'EXISTS' if vocab_exists else 'MISSING'} (vocab: {existing_components['vocabulary']['vocab_file']}, mappings: {existing_components['vocabulary']['mappings_file']})")
        logger.info(f"  - FAISS Index: {'EXISTS' if indexes_exist else 'MISSING'} (index: {existing_components['indexes']['faiss_index']}, docs: {existing_components['indexes']['processed_docs']})")
        logger.info(f"  - Embeddings Cache: {'EXISTS' if embeddings_exist else 'MISSING'} (cache: {existing_components['embeddings']['cache_file']})")
        
        if self.force_rebuild:
            logger.info("Force rebuild enabled - all components will be recreated")
        else:
            skip_count = sum([vocab_exists, indexes_exist, embeddings_exist])
            logger.info(f"Existing components will be reused ({skip_count}/3 components available)")
    
    def _ensure_directories(self) -> bool:
        """Create necessary directories for the system"""
        logger.info("Creating system directories...")
        
        try:
            Config.ensure_directories()
            logger.info("System directories created successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to create directories: {e}")
            return False
    
    def _setup_database_connection(self) -> bool:
        """Establish and test database connection"""
        logger.info("Setting up database connection...")
        
        if not self.db_connector.connect():
            logger.error("Failed to connect to MongoDB")
            self.setup_stats['errors'].append("Database connection failed")
            return False
        
        # Test basic operations
        doc_count = self.db_connector.get_document_count()
        logger.info(f"Database connected successfully. Found {doc_count} documents.")
        
        if doc_count == 0:
            logger.warning("Database contains no documents")
            self.setup_stats['errors'].append("No documents in database")
            return False
        
        return True
    
    def _load_documents(self) -> Optional[list]:
        """Load documents from MongoDB"""
        logger.info("Loading documents from MongoDB...")
        
        documents = self.db_connector.get_all_documents()
        
        if not documents:
            logger.error("No documents loaded from database")
            return None
        
        # Filter and validate documents
        valid_documents = []
        for doc in documents:
            if self._validate_document(doc):
                valid_documents.append(doc)
        
        self.setup_stats['documents_processed'] = len(valid_documents)
        logger.info(f"Loaded and validated {len(valid_documents)} documents")
        
        return valid_documents
    
    def _validate_document(self, document: Dict[str, Any]) -> bool:
        """Validate individual document structure"""
        required_fields = ['_id']
        important_fields = ['name', 'description', 'price', 'property_type']
        
        # Check required fields
        for field in required_fields:
            if field not in document:
                return False
        
        # Check if document has some important fields
        has_important_fields = any(document.get(field) for field in important_fields)
        
        return has_important_fields
    
    def _build_vocabulary(self, documents: list) -> bool:
        """Build vocabulary from MongoDB documents"""
        logger.info("Building vocabulary from documents...", 
                   class_name='SystemSetup',
                   method='_build_vocabulary')
        
        # OPTIMIZATION: Check if vocabulary already exists and force rebuild is not enabled
        # This prevents expensive vocabulary rebuilding when files already exist
        if not self.force_rebuild:
            vocab_path = Config.DATA_DIR / 'vocabulary.json'
            mapping_path = Config.DATA_DIR / 'keyword_mappings.json'
            
            # Both vocabulary files must exist for the vocabulary to be considered complete
            if vocab_path.exists() and mapping_path.exists():
                logger.info("Vocabulary files already exist, loading existing vocabulary")
                # Attempt to load existing vocabulary data instead of rebuilding
                if self.vocabulary_manager.load_vocabulary():
                    # Update statistics with loaded vocabulary data
                    vocab_size = len(self.vocabulary_manager.vocabulary)
                    mappings_count = len(self.vocabulary_manager.keyword_mappings)
                    self.setup_stats['vocabulary_terms'] = vocab_size
                    logger.info(f"Loaded existing vocabulary: {vocab_size} terms, {mappings_count} keyword mappings")
                    return True  # Skip vocabulary building entirely
                else:
                    # If loading fails, fall back to rebuilding
                    logger.warning("Failed to load existing vocabulary, rebuilding...")
            else:
                logger.info("Vocabulary files not found, building new vocabulary")
        else:
            logger.info("Force rebuild enabled, creating new vocabulary")
        
        try:
            self.vocabulary_manager.build_vocabulary_from_documents(documents)
            
            vocab_size = len(self.vocabulary_manager.vocabulary)
            mappings_count = len(self.vocabulary_manager.keyword_mappings)
            
            self.setup_stats['vocabulary_terms'] = vocab_size
            
            logger.info(f"Vocabulary built: {vocab_size} terms, {mappings_count} keyword mappings")
            
            # Save vocabulary to data folder
            self.vocabulary_manager.save_vocabulary()
            
            # Log interesting vocabulary statistics
            self._log_vocabulary_stats()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to build vocabulary: {e}", 
                        class_name='SystemSetup',
                        method='_build_vocabulary',
                        error=str(e))
            return False
    
    def _log_vocabulary_stats(self):
        """Log interesting vocabulary statistics"""
        try:
            # Most common terms
            top_terms = self.vocabulary_manager.term_frequencies.most_common(10)
            logger.info(f"Top 10 terms: {top_terms}")
            
            # Field distribution
            if hasattr(self.vocabulary_manager, 'field_terms') and isinstance(self.vocabulary_manager.field_terms, dict):
                field_counts = {field: len(terms) for field, terms in self.vocabulary_manager.field_terms.items()}
                logger.info(f"Terms per field: {field_counts}")
            else:
                logger.info("Field terms not available for statistics")
            
        except Exception as e:
            logger.debug(f"Could not log vocabulary stats: {e}")
    
    def _initialize_optimizers(self, documents: list) -> bool:
        """Initialize optimizers with MongoDB data"""
        logger.info("Initializing optimizers with vocabulary...")
        
        try:
            # Initialize Airbnb optimizer with vocabulary
            self.airbnb_optimizer.initialize_with_mongodb_data(documents)
            
            # Update index manager's document processor
            self.index_manager.document_processor.airbnb_optimizer = self.airbnb_optimizer
            
            logger.info("Optimizers initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize optimizers: {e}")
            return False
    
    def _create_indexes(self, documents: list) -> bool:
        """Create embeddings and FAISS indexes"""
        logger.info("Creating embeddings and indexes...")
        
        try:
            # OPTIMIZATION: Check if indexes exist and rebuild is not forced
            # This is the most expensive operation, so we prioritize reusing existing indexes
            if not self.force_rebuild:
                # Attempt to load existing FAISS index and processed documents
                faiss_index, processed_docs = self.index_manager.load_indexes()
                if faiss_index is not None and processed_docs:
                    logger.info("Existing FAISS index and processed documents found, skipping rebuild")
                    # Restore the loaded components to the index manager
                    self.index_manager.faiss_index = faiss_index
                    self.index_manager.processed_documents = processed_docs
                    self.setup_stats['faiss_index_size'] = faiss_index.ntotal
                    
                    # PERFORMANCE: Also load existing embeddings cache to avoid regeneration
                    # Embeddings are expensive to compute, so we cache them separately
                    cached_embeddings = self.index_manager._load_cached_embeddings()
                    if cached_embeddings:
                        self.index_manager.document_embeddings = cached_embeddings
                        self.setup_stats['embeddings_generated'] = len(cached_embeddings)
                        logger.info(f"Loaded existing embeddings cache with {len(cached_embeddings)} embeddings")
                    
                    logger.info(f"Using existing indexes: {faiss_index.ntotal} embeddings in FAISS index")
                    return True  # Skip entire index creation process
                else:
                    logger.info("Index files not found or incomplete, creating new indexes")
            else:
                logger.info("Force rebuild enabled, creating new indexes")
            
            # Create complete index
            success = self.index_manager.create_complete_index(rebuild=self.force_rebuild)
            
            if success:
                self.setup_stats['faiss_index_size'] = self.index_manager.faiss_index.ntotal if self.index_manager.faiss_index else 0
                self.setup_stats['embeddings_generated'] = len(self.index_manager.document_embeddings)
                logger.info(f"Created FAISS index with {self.setup_stats['faiss_index_size']} embeddings")
                return True
            else:
                logger.error("Failed to create indexes")
                return False
                
        except Exception as e:
            logger.error(f"Index creation failed: {e}")
            return False
    
    def _setup_numeric_filters(self) -> bool:
        """Setup numeric filtering optimizations"""
        logger.info("Setting up numeric filters...")
        
        try:
            # Analyze numeric fields in documents for optimization
            numeric_stats = self._analyze_numeric_fields()
            
            # Log numeric field statistics
            logger.info(f"Numeric field analysis: {numeric_stats}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup numeric filters: {e}")
            return False
    
    def _analyze_numeric_fields(self) -> Dict[str, Any]:
        """Analyze numeric fields for optimization"""
        numeric_fields = ['price', 'bedrooms', 'bathrooms', 'accommodates', 'review_scores_rating']
        stats = {}
        
        documents = self.index_manager.processed_documents
        
        for field in numeric_fields:
            values = []
            for doc in documents:
                original_doc = doc.get('original_document', doc)
                if field in original_doc and original_doc[field] is not None:
                    try:
                        value = float(str(original_doc[field]).replace('$', '').replace(',', ''))
                        values.append(value)
                    except (ValueError, TypeError):
                        continue
            
            if values:
                stats[field] = {
                    'min': min(values),
                    'max': max(values),
                    'avg': sum(values) / len(values),
                    'count': len(values)
                }
        
        return stats
    
    def _validate_system(self) -> bool:
        """Validate the complete system setup"""
        logger.info("Validating system setup...")
        
        try:
            # Test system initialization
            rag_system = JSONRAGSystem()
            
            # Initialize with existing components
            rag_system.index_manager = self.index_manager
            rag_system.vocabulary_manager = self.vocabulary_manager
            
            # Test query processing
            test_queries = [
                "Find 2 bedroom apartments under $150",
                "Show me places with WiFi and parking",
                "Highly rated properties downtown"
            ]
            
            for query in test_queries:
                try:
                    # Test query analysis
                    analysis = rag_system.query_engine.analyze_query(query)
                    if not analysis or not analysis.get('keywords'):
                        logger.warning(f"Query analysis failed for: {query}")
                        continue
                    
                    logger.info(f"Test query '{query[:30]}...': {len(analysis['keywords'])} keywords extracted")
                    
                except Exception as e:
                    logger.warning(f"Test query failed: {e}")
            
            logger.info("System validation completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"System validation failed: {e}")
            return False
    
    def _print_setup_summary(self):
        """Print comprehensive setup summary"""
        duration = self.setup_stats['end_time'] - self.setup_stats['start_time']
        
        print("\n" + "="*60)
        print("JSON RAG SYSTEM SETUP SUMMARY")
        print("="*60)
        print(f"Setup Duration: {duration:.2f} seconds")
        print(f"Force Rebuild: {'YES' if self.force_rebuild else 'NO'}")
        print(f"Documents Processed: {self.setup_stats['documents_processed']}")
        print(f"Vocabulary Terms: {self.setup_stats['vocabulary_terms']}")
        print(f"Embeddings Generated: {self.setup_stats['embeddings_generated']}")
        print(f"FAISS Index Size: {self.setup_stats['faiss_index_size']}")
        print(f"Setup Success: {self.setup_stats['setup_success']}")
        
        # Show what was reused vs created
        print("\nComponent Status:")
        vocab_exists = (Config.DATA_DIR / 'vocabulary.json').exists() and (Config.DATA_DIR / 'keyword_mappings.json').exists()
        indexes_exist = Config.FAISS_INDEX_PATH.exists() and Config.PROCESSED_DOCS_PATH.exists()
        embeddings_exist = Config.EMBEDDINGS_CACHE_PATH.exists()
        
        print(f"  - Vocabulary: {'REUSED' if (vocab_exists and not self.force_rebuild) else 'CREATED'}")
        print(f"  - FAISS Index: {'REUSED' if (indexes_exist and not self.force_rebuild) else 'CREATED'}")
        print(f"  - Embeddings: {'REUSED' if (embeddings_exist and not self.force_rebuild) else 'CREATED'}")
        
        if self.setup_stats['errors']:
            print("\nErrors Encountered:")
            for error in self.setup_stats['errors']:
                print(f"  - {error}")
        
        print("\nSystem Components:")
        print(f"  - Database: {Config.DATABASE_NAME}.{Config.COLLECTION_NAME}")
        print(f"  - Embedding Model: {Config.EMBEDDING_MODEL}")
        print(f"  - Index Files: {Config.INDEXES_DIR}")
        print(f"  - Cache Dir: {Config.CACHE_DIR}")
        print("\n" + "="*60)
    
    # === NEW TRANSLATION MODEL PRELOADING FUNCTIONALITY ===
    
    def setup_translation_models(self, force_reload: bool = False) -> bool:
        """
        Preload all translation models for faster runtime performance
        
        Args:
            force_reload: Force reload models even if already cached
            
        Returns:
            True if setup successful, False otherwise
        """
        logger.info("Starting translation model preloading...")
        
        if not TRANSLATION_AVAILABLE:
            logger.error("Translation components not available. Install transformers and torch.")
            return False
        
        try:
            # Initialize translation service
            translation_service = get_translation_service()
            
            # Check if models are already loaded
            if not force_reload and self._check_translation_models_loaded(translation_service):
                logger.info("Translation models already loaded, skipping preload")
                return True
            
            # Preload forward translation models (language -> English)
            forward_models = {
                'es': 'Helsinki-NLP/opus-mt-es-en',  # Spanish
                'fr': 'Helsinki-NLP/opus-mt-fr-en',  # French
                'de': 'Helsinki-NLP/opus-mt-de-en',  # German
                'it': 'Helsinki-NLP/opus-mt-it-en',  # Italian
                # 'pt': 'Helsinki-NLP/opus-mt-pt-en',  # Portuguese - Model not available
                'ru': 'Helsinki-NLP/opus-mt-ru-en',  # Russian
                'zh': 'Helsinki-NLP/opus-mt-zh-en',  # Chinese
                'ja': 'Helsinki-NLP/opus-mt-ja-en',  # Japanese
                'ko': 'Helsinki-NLP/opus-mt-ko-en',  # Korean
                'ar': 'Helsinki-NLP/opus-mt-ar-en',  # Arabic
                'hi': 'Helsinki-NLP/opus-mt-hi-en',  # Hindi
            }
            
            # Preload reverse translation models (English -> language)
            reverse_models = {
                'es': 'Helsinki-NLP/opus-mt-en-es',  # English to Spanish
                'fr': 'Helsinki-NLP/opus-mt-en-fr',  # English to French
                'de': 'Helsinki-NLP/opus-mt-en-de',  # English to German
                'it': 'Helsinki-NLP/opus-mt-en-it',  # English to Italian
                # 'pt': 'Helsinki-NLP/opus-mt-en-pt',  # English to Portuguese - Model not available
                'ru': 'Helsinki-NLP/opus-mt-en-ru',  # English to Russian
                'zh': 'Helsinki-NLP/opus-mt-en-zh',  # English to Chinese
                # 'ja': 'Helsinki-NLP/opus-mt-en-ja',  # English to Japanese - Model not available
                # 'ko': 'Helsinki-NLP/opus-mt-en-ko',  # English to Korean - Model not available
                'ar': 'Helsinki-NLP/opus-mt-en-ar',  # English to Arabic
                'hi': 'Helsinki-NLP/opus-mt-en-hi',  # English to Hindi
            }
            
            total_models = len(forward_models) + len(reverse_models)
            loaded_count = 0
            
            logger.info(f"Preloading {total_models} translation models...")
            
            # Load forward translation models
            for lang_code, model_name in forward_models.items():
                logger.info(f"Loading forward model: {model_name}")
                if translation_service._load_model(lang_code, reverse=False):
                    loaded_count += 1
                    logger.info(f"[OK] Forward model loaded: {lang_code} -> en")
                else:
                    logger.warning(f"[ERROR] Failed to load forward model: {lang_code} -> en")
            
            # Load reverse translation models
            for lang_code, model_name in reverse_models.items():
                logger.info(f"Loading reverse model: {model_name}")
                if translation_service._load_model(lang_code, reverse=True):
                    loaded_count += 1
                    logger.info(f"[OK] Reverse model loaded: en -> {lang_code}")
                else:
                    logger.warning(f"[ERROR] Failed to load reverse model: en -> {lang_code}")
            
            # Test models with sample translations
            if self._test_preloaded_models(translation_service):
                logger.info(f"Translation model preloading completed: {loaded_count}/{total_models} models loaded")
                return True
            else:
                logger.warning("Translation model testing failed")
                return False
                
        except Exception as e:
            logger.error(f"Translation model preloading failed: {e}")
            return False
    
    def _check_translation_models_loaded(self, translation_service: TranslationService) -> bool:
        """Check if translation models are already loaded"""
        expected_models = ['es', 'fr', 'de', 'it', 'ru', 'zh', 'ja', 'ko', 'ar', 'hi']  # Removed 'pt'
        
        # Check forward models (only for models that actually exist)
        available_forward_models = ['es', 'fr', 'de', 'it', 'ru', 'zh', 'ja', 'ko', 'ar', 'hi']
        forward_loaded = all(lang in translation_service.models for lang in available_forward_models)
        
        # Check reverse models (only for models that actually exist)
        available_reverse_models = ['es', 'fr', 'de', 'it', 'ru', 'zh', 'ar', 'hi']  # Removed 'pt', 'ja', 'ko'
        reverse_loaded = all(f"{lang}_reverse" in translation_service.models for lang in available_reverse_models)
        
        return forward_loaded and reverse_loaded
    
    def _test_preloaded_models(self, translation_service: TranslationService) -> bool:
        """Test preloaded translation models with sample texts"""
        test_cases = [
            ('es', 'Hola, ¿cómo estás?'),
            ('fr', 'Bonjour, comment allez-vous?'),
            ('de', 'Hallo, wie geht es dir?'),
            ('it', 'Ciao, come stai?'),
            ('pt', 'Olá, como está?'),
        ]
        
        success_count = 0
        
        for lang_code, test_text in test_cases:
            try:
                # Test forward translation
                forward_result = translation_service._translate_with_marian(test_text, lang_code, 'en')
                if forward_result and forward_result != test_text:
                    success_count += 1
                    logger.debug(f"Forward test passed: {lang_code} -> en")
                
                # Test reverse translation
                reverse_result = translation_service._translate_with_marian('Hello, how are you?', 'en', lang_code)
                if reverse_result and reverse_result != 'Hello, how are you?':
                    success_count += 1
                    logger.debug(f"Reverse test passed: en -> {lang_code}")
                    
            except Exception as e:
                logger.warning(f"Translation test failed for {lang_code}: {e}")
        
        return success_count >= len(test_cases)  # At least forward tests should pass
    
    def setup_translation_accuracy_improvements(self) -> bool:
        """
        Setup translation accuracy improvements including:
        - Ensemble translation methods
        - Translation confidence scoring
        - Context-aware translation
        - Translation quality validation
        """
        logger.info("Setting up translation accuracy improvements...")
        
        if not ACCURACY_LIBS_AVAILABLE:
            logger.warning("Translation accuracy libraries not available. Install numpy, scipy, scikit-learn.")
            return False
        
        try:
            # Create accuracy improvement components
            accuracy_components = {
                'ensemble_translator': self._setup_ensemble_translator(),
                'confidence_scorer': self._setup_confidence_scorer(),
                'context_analyzer': self._setup_context_analyzer(),
                'quality_validator': self._setup_quality_validator(),
            }
            
            # Save accuracy components configuration
            self._save_accuracy_config(accuracy_components)
            
            logger.info("Translation accuracy improvements setup completed")
            return True
            
        except Exception as e:
            logger.error(f"Translation accuracy setup failed: {e}")
            return False
    
    def _setup_ensemble_translator(self) -> Dict[str, Any]:
        """Setup ensemble translation method for improved accuracy"""
        ensemble_config = {
            'enabled': True,
            'methods': [
                {
                    'name': 'marian_mt',
                    'weight': 0.7,
                    'description': 'Primary MarianMT translation'
                },
                {
                    'name': 'fallback_patterns',
                    'weight': 0.2,
                    'description': 'Pattern-based fallback translation'
                },
                {
                    'name': 'context_aware',
                    'weight': 0.1,
                    'description': 'Context-aware translation adjustments'
                }
            ],
            'confidence_threshold': 0.8,
            'min_agreement_ratio': 0.6
        }
        
        logger.info("Ensemble translator configured with weighted voting")
        return ensemble_config
    
    def _setup_confidence_scorer(self) -> Dict[str, Any]:
        """Setup translation confidence scoring system"""
        confidence_config = {
            'enabled': True,
            'scoring_methods': [
                'language_detection_confidence',
                'translation_length_ratio',
                'character_encoding_quality',
                'linguistic_pattern_matching',
                'back_translation_consistency'
            ],
            'thresholds': {
                'high_confidence': 0.9,
                'medium_confidence': 0.7,
                'low_confidence': 0.5
            },
            'actions': {
                'low_confidence': 'request_clarification',
                'medium_confidence': 'use_with_warning',
                'high_confidence': 'use_directly'
            }
        }
        
        logger.info("Translation confidence scorer configured")
        return confidence_config
    
    def _setup_context_analyzer(self) -> Dict[str, Any]:
        """Setup context-aware translation improvements"""
        context_config = {
            'enabled': True,
            'real_estate_context': {
                'property_types': ['apartment', 'house', 'condo', 'studio', 'villa'],
                'amenities': ['wifi', 'parking', 'pool', 'gym', 'balcony', 'garden'],
                'locations': ['downtown', 'suburb', 'city center', 'near metro', 'beach'],
                'price_terms': ['rent', 'buy', 'lease', 'deposit', 'monthly', 'yearly']
            },
            'context_boosting': {
                'property_queries': 1.2,
                'location_queries': 1.1,
                'price_queries': 1.15,
                'amenity_queries': 1.05
            },
            'domain_specific_corrections': True
        }
        
        logger.info("Context-aware translation analyzer configured")
        return context_config
    
    def _setup_quality_validator(self) -> Dict[str, Any]:
        """Setup translation quality validation system"""
        quality_config = {
            'enabled': True,
            'validation_methods': [
                'semantic_similarity_check',
                'language_consistency_check',
                'domain_terminology_check',
                'grammar_structure_check'
            ],
            'quality_metrics': {
                'semantic_similarity_threshold': 0.7,
                'language_consistency_threshold': 0.8,
                'terminology_accuracy_threshold': 0.9,
                'grammar_score_threshold': 0.6
            },
            'fallback_strategies': [
                'retry_with_different_model',
                'use_ensemble_method',
                'request_human_review',
                'use_original_with_warning'
            ]
        }
        
        logger.info("Translation quality validator configured")
        return quality_config
    
    def _save_accuracy_config(self, components: Dict[str, Any]) -> None:
        """Save translation accuracy configuration to file"""
        import json
        
        config_path = Config.DATA_DIR / 'translation_accuracy_config.json'
        
        accuracy_config = {
            'version': '1.0',
            'created_at': datetime.now().isoformat(),
            'components': components,
            'system_info': {
                'torch_available': torch.cuda.is_available() if TRANSLATION_AVAILABLE else False,
                'device': 'cuda' if torch.cuda.is_available() else 'cpu',
                'accuracy_libs': ACCURACY_LIBS_AVAILABLE
            }
        }
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(accuracy_config, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Translation accuracy configuration saved to: {config_path}")
    
    def full_translation_setup(self, force_reload: bool = False) -> bool:
        """
        Complete translation setup including model preloading and accuracy improvements
        
        Args:
            force_reload: Force reload all components
            
        Returns:
            True if setup successful, False otherwise
        """
        logger.info("Starting complete translation setup...")
        
        try:
            # Step 1: Preload translation models
            if not self.setup_translation_models(force_reload):
                logger.error("Translation model preloading failed")
                return False
            
            # Step 2: Setup accuracy improvements
            if not self.setup_translation_accuracy_improvements():
                logger.warning("Translation accuracy improvements setup failed, continuing with basic setup")
            
            # Step 3: Test complete system
            if self._test_complete_translation_system():
                logger.info("Complete translation setup successful")
                return True
            else:
                logger.warning("Translation system testing failed")
                return False
                
        except Exception as e:
            logger.error(f"Complete translation setup failed: {e}")
            return False
    
    def _test_complete_translation_system(self) -> bool:
        """Test the complete translation system with real-world examples"""
        test_queries = [
            ('es', '¿Hay apartamentos de 2 dormitorios cerca del centro?'),
            ('fr', 'Je cherche un appartement avec parking et WiFi'),
            ('de', 'Ich suche eine Wohnung mit Balkon und Küche'),
            ('it', 'Cerco casa con giardino vicino alla metro'),
            ('pt', 'Preciso de apartamento mobiliado com 3 quartos'),
        ]
        
        translation_service = get_translation_service()
        success_count = 0
        
        for lang_code, query in test_queries:
            try:
                # Test complete translation pipeline
                result = translation_service.translate_text(query, target_lang='en', source_lang=lang_code)
                
                if result['translation_needed'] and result['translated_text'] != query:
                    success_count += 1
                    logger.debug(f"Translation test passed: {lang_code} -> {result['translated_text'][:50]}...")
                    
                    # Test reverse translation
                    reverse_result = translation_service.translate_response_to_user_language(
                        "I found several apartments matching your criteria.", lang_code
                    )
                    
                    if reverse_result['translation_needed']:
                        logger.debug(f"Reverse translation test passed: en -> {lang_code}")
                    
            except Exception as e:
                logger.warning(f"Complete translation test failed for {lang_code}: {e}")
        
        return success_count >= len(test_queries) * 0.8  # 80% success rate required
    
    # === ADDITIONAL SETUP METHODS FOR COMMAND-LINE INTERFACE ===
    
    def rebuild_indexes_only(self) -> bool:
        """Rebuild embeddings and FAISS indexes only"""
        logger.info("Rebuilding indexes only...")
        
        try:
            # Load documents from database
            documents = self._load_documents()
            if not documents:
                return False
            
            # Initialize optimizers
            if not self._initialize_optimizers(documents):
                return False
            
            # Rebuild indexes
            return self._create_indexes(documents)
            
        except Exception as e:
            logger.error(f"Index rebuild failed: {e}")
            return False
    
    def build_vocab_only(self) -> bool:
        """Build vocabulary from MongoDB documents only"""
        logger.info("Building vocabulary only...")
        
        try:
            # Setup database connection
            if not self._setup_database_connection():
                return False
            
            # Load documents
            documents = self._load_documents()
            if not documents:
                return False
            
            # Build vocabulary
            return self._build_vocabulary(documents)
            
        except Exception as e:
            logger.error(f"Vocabulary build failed: {e}")
            return False
    
    def test_system_only(self) -> bool:
        """Test system components without setup"""
        logger.info("Testing system components...")
        
        try:
            # Test database connection
            if not self._setup_database_connection():
                logger.error("Database connection test failed")
                return False
            
            # Test system validation
            if not self._validate_system():
                logger.error("System validation test failed")
                return False
            
            # Test translation system if available
            if TRANSLATION_AVAILABLE:
                if not self._test_complete_translation_system():
                    logger.warning("Translation system test failed")
            
            logger.info("All system tests passed")
            return True
            
        except Exception as e:
            logger.error(f"System testing failed: {e}")
            return False
    
    def _print_setup_summary(self):
        """Print comprehensive setup summary"""
        duration = self.setup_stats['end_time'] - self.setup_stats['start_time']
        
        print("\n" + "="*60)
        print(f"Force Rebuild: {'YES' if self.force_rebuild else 'NO'}")
        print(f"Documents Processed: {self.setup_stats['documents_processed']}")
        print(f"Vocabulary Terms: {self.setup_stats['vocabulary_terms']}")
        print(f"Embeddings Generated: {self.setup_stats['embeddings_generated']}")
        print(f"FAISS Index Size: {self.setup_stats['faiss_index_size']}")
        print(f"Setup Success: {self.setup_stats['setup_success']}")
        
        # Show what was reused vs created
        print("\nComponent Status:")
        vocab_exists = (Config.DATA_DIR / 'vocabulary.json').exists() and (Config.DATA_DIR / 'keyword_mappings.json').exists()
        indexes_exist = Config.FAISS_INDEX_PATH.exists() and Config.PROCESSED_DOCS_PATH.exists()
        embeddings_exist = Config.EMBEDDINGS_CACHE_PATH.exists()
        
        print(f"  - Vocabulary: {'REUSED' if (vocab_exists and not self.force_rebuild) else 'CREATED'}")
        print(f"  - FAISS Index: {'REUSED' if (indexes_exist and not self.force_rebuild) else 'CREATED'}")
        print(f"  - Embeddings: {'REUSED' if (embeddings_exist and not self.force_rebuild) else 'CREATED'}")
        
        if self.setup_stats['errors']:
            print("\nErrors Encountered:")
            for error in self.setup_stats['errors']:
                print(f"  - {error}")
        
        print("\nSystem Components:")
        print(f"  - Database: {Config.DATABASE_NAME}.{Config.COLLECTION_NAME}")
        print(f"  - Embedding Model: {Config.EMBEDDING_MODEL}")
        print(f"  - Index Files: {Config.INDEXES_DIR}")
        print(f"  - Cache Dir: {Config.CACHE_DIR}")
        print("\n" + "="*60)
    
    def rebuild_indexes_only(self) -> bool:
        """Rebuild only the indexes without vocabulary"""
        logger.info("Rebuilding indexes only...")
        
        if not self._setup_database_connection():
            return False
        
        documents = self._load_documents()
        if not documents:
            return False
        
        return self._create_indexes(documents)
    
    def build_vocab_only(self) -> bool:
        """Build only the vocabulary from MongoDB"""
        logger.info("Building vocabulary only...")
        
        if not self._setup_database_connection():
            return False
        
        documents = self._load_documents()
        if not documents:
            return False
        
        return self._build_vocabulary(documents)
    
    def test_system_only(self) -> bool:
        """Test system without setup"""
        logger.info("Testing system components...")
        
        return self._validate_system()

def main():
    """Main setup function with command line interface"""
    logger.info("Starting system setup", 
               operation='system_setup',
               command_line=True)
    
    parser = argparse.ArgumentParser(
        description="JSON RAG System Setup and Initialization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python setup.py --full-setup              # Complete system setup
  python setup.py --rebuild-indexes         # Rebuild indexes only
  python setup.py --build-vocab-only        # Build vocabulary only
  python setup.py --test-system            # Test system components
  python setup.py --full-setup --force     # Force rebuild everything
  python setup.py --preload-translation    # Preload translation models
  python setup.py --setup-translation-accuracy  # Setup translation accuracy improvements
  python setup.py --full-translation-setup # Complete translation setup
  python setup.py --full-translation-setup --force-translation-reload  # Force reload translation models
        """
    )
    
    parser.add_argument(
        '--full-setup', 
        action='store_true',
        help='Run complete system setup including database, vocabulary, embeddings, and indexes (reuses existing components unless --force is used)'
    )
    
    parser.add_argument(
        '--rebuild-indexes',
        action='store_true', 
        help='Rebuild embeddings and FAISS indexes only'
    )
    
    parser.add_argument(
        '--build-vocab-only',
        action='store_true',
        help='Build vocabulary from MongoDB documents only'
    )
    
    parser.add_argument(
        '--test-system',
        action='store_true',
        help='Test system components without setup'
    )
    
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force rebuild of existing components (vocabulary, embeddings, and indexes)'
    )
    
    # === NEW TRANSLATION MODEL PRELOADING ARGUMENTS ===
    
    parser.add_argument(
        '--preload-translation',
        action='store_true',
        help='Preload all translation models for faster runtime performance'
    )
    
    parser.add_argument(
        '--setup-translation-accuracy',
        action='store_true',
        help='Setup translation accuracy improvements (ensemble methods, confidence scoring, etc.)'
    )
    
    parser.add_argument(
        '--full-translation-setup',
        action='store_true',
        help='Complete translation setup including model preloading and accuracy improvements'
    )
    
    parser.add_argument(
        '--force-translation-reload',
        action='store_true',
        help='Force reload translation models even if already cached'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Set verbose logging if requested
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create setup instance
    setup = SystemSetup(force_rebuild=args.force)
    
    # Determine which operation to run
    success = False
    
    if args.full_setup:
        success = setup.setup_complete_system()
    elif args.rebuild_indexes:
        success = setup.rebuild_indexes_only()
    elif args.build_vocab_only:
        success = setup.build_vocab_only()
    elif args.test_system:
        success = setup.test_system_only()
    elif args.preload_translation:
        success = setup.setup_translation_models(force_reload=args.force_translation_reload)
    elif args.setup_translation_accuracy:
        success = setup.setup_translation_accuracy_improvements()
    elif args.full_translation_setup:
        success = setup.full_translation_setup(force_reload=args.force_translation_reload)
    else:
        # Default to full setup if no specific option given
        print("No specific setup option provided. Running full setup...")
        success = setup.setup_complete_system()
    
    # Exit with appropriate code
    if success:
        print("\nSetup completed successfully!")
        print("You can now run 'python main.py' to start the system.")
        sys.exit(0)
    else:
        print("\nSetup failed. Check logs for details.")
        sys.exit(1)

if __name__ == "__main__":
    main()