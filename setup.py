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

The system automatically detects existing components and skips recreation unless
--force flag is used, making subsequent setups much faster.

Usage:
    python setup.py --full-setup              # Setup with existing component reuse
    python setup.py --full-setup --force      # Force rebuild all components
    python setup.py --rebuild-indexes         # Rebuild only indexes
    python setup.py --build-vocab-only        # Build only vocabulary
    python setup.py --test-system             # Test system components
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