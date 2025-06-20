#!/usr/bin/env python
"""
Management Commands for Django RAG System
Provides preprocessing and maintenance commands
"""

import os
import sys
import django
from pathlib import Path

# Add the project directory to Python path
project_dir = Path(__file__).parent
sys.path.insert(0, str(project_dir))

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'django_rag_project.settings')
django.setup()

import logging
from typing import Dict, Any
from datetime import datetime
import json

logger = logging.getLogger(__name__)

def preprocess_data():
    """Preprocess all data for faster response times"""
    print("Starting data preprocessing...")
    
    try:
        from preprocessor import preprocess_from_mongodb
        
        start_time = datetime.now()
        success = preprocess_from_mongodb()
        end_time = datetime.now()
        
        duration = (end_time - start_time).total_seconds()
        
        if success:
            print(f"Data preprocessing completed successfully in {duration:.2f} seconds")
            
            # Load and display stats
            from preprocessor import DataPreprocessor
            preprocessor = DataPreprocessor()
            stats = preprocessor.get_stats()
            
            if stats:
                print("\n📊 Preprocessing Statistics:")
                print(f"  Total Properties: {stats.get('total_properties', 'N/A')}")
                print(f"  Properties with Price: {stats.get('properties_with_price', 'N/A')}")
                print(f"  Properties with Location: {stats.get('properties_with_location', 'N/A')}")
                if stats.get('avg_price'):
                    print(f"  Average Price: ${stats['avg_price']:.2f}")
                    
        else:
            print("❌ Data preprocessing failed")
            return False
            
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        logger.error(f"Preprocessing error: {e}", exc_info=True)
        return False
    
    return True

def check_preprocessing_status():
    """Check the status of preprocessed data"""
    print("Checking preprocessing status...")
    
    try:
        from preprocessor import DataPreprocessor
        
        preprocessor = DataPreprocessor()
        
        # Check if data exists
        if preprocessor.load_preprocessed_data():
            print("✅ Preprocessed data is available")
            
            stats = preprocessor.get_stats()
            if stats:
                print("\n📊 Current Statistics:")
                for key, value in stats.items():
                    if isinstance(value, float):
                        print(f"  {key.replace('_', ' ').title()}: {value:.2f}")
                    else:
                        print(f"  {key.replace('_', ' ').title()}: {value}")
                        
            # Test a sample query
            sample_ids = list(preprocessor.store.metadata.keys())[:3]
            if sample_ids:
                print("\n🔍 Sample Properties:")
                for doc_id in sample_ids:
                    metadata = preprocessor.get_property_metadata(doc_id)
                    if metadata:
                        print(f"  ID: {metadata.id}")
                        print(f"    Name: {metadata.name or 'N/A'}")
                        print(f"    Price: {metadata.price_formatted or 'N/A'}")
                        print(f"    Location: {metadata.location or 'N/A'}")
                        print()
        else:
            print("❌ No preprocessed data found")
            print("   Run preprocessing with: python management_commands.py preprocess")
            return False
            
    except Exception as e:
        print(f"❌ Error checking status: {e}")
        return False
    
    return True

def test_performance():
    """Test response time performance with and without preprocessing"""
    print("Testing performance...")
    
    try:
        import time
        from rag_api.services_enhanced import EnhancedRAGService
        
        # Initialize service
        service = EnhancedRAGService.get_instance()
        
        test_queries = [
            "Find apartments under $200",
            "Properties in Manhattan", 
            "2 bedroom places with wifi",
            "Cheap places near the beach"
        ]
        
        print("\n⏱️  Performance Test Results:")
        print("-" * 50)
        
        total_time = 0
        for i, query in enumerate(test_queries, 1):
            start_time = time.time()
            
            try:
                result = service.enhanced_query(
                    query_text=query,
                    session_id=f"test_session_{i}",
                    enable_comparison=False
                )
                
                end_time = time.time()
                query_time = end_time - start_time
                total_time += query_time
                
                num_results = result.get('metadata', {}).get('num_results', 0)
                
                print(f"Query {i}: {query}")
                print(f"  Time: {query_time:.3f}s")
                print(f"  Results: {num_results}")
                print(f"  Status: {'✅ Success' if result.get('success') else '❌ Failed'}")
                print()
                
            except Exception as e:
                print(f"Query {i}: {query}")
                print(f"  Status: ❌ Error - {e}")
                print()
        
        avg_time = total_time / len(test_queries)
        print(f"Average Response Time: {avg_time:.3f}s")
        print(f"Total Test Time: {total_time:.3f}s")
        
        # Performance recommendations
        if avg_time < 1.0:
            print("🚀 Excellent performance!")
        elif avg_time < 2.0:
            print("✅ Good performance")
        elif avg_time < 5.0:
            print("⚠️  Acceptable performance - consider optimization")
        else:
            print("🐌 Slow performance - preprocessing recommended")
            
    except Exception as e:
        print(f"❌ Error during performance test: {e}")
        return False
    
    return True

def clear_preprocessing_cache():
    """Clear all preprocessed data"""
    print("Clearing preprocessing cache...")
    
    try:
        from preprocessor import DataPreprocessor
        import shutil
        
        preprocessor = DataPreprocessor()
        cache_dir = preprocessor.store.storage_dir
        
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
            print(f"✅ Cleared cache directory: {cache_dir}")
        else:
            print("ℹ️  No cache directory found")
            
        print("   Run preprocessing again with: python management_commands.py preprocess")
        
    except Exception as e:
        print(f"❌ Error clearing cache: {e}")
        return False
    
    return True
    
def rebuild_system():
    """Rebuild entire system including preprocessing"""
    print("Rebuilding entire system...")
    print("=" * 50)
    
    # Step 1: Clear cache
    print("Step 1: Clearing existing cache...")
    if not clear_preprocessing_cache():
        return False
    
    # Step 2: Reinitialize system
    print("\nStep 2: Reinitializing system components...")
    try:
        from rag_api.services_enhanced import EnhancedRAGService
        # Force reinitialization
        EnhancedRAGService._initialized = False
        EnhancedRAGService._instance = None
        service = EnhancedRAGService.get_instance()
        print("✅ System reinitialized")
    except Exception as e:
        print(f"❌ System reinitialization failed: {e}")
        return False
    
    # Step 3: Preprocess data
    print("\nStep 3: Preprocessing data...")
    if not preprocess_data():
        return False
    
    # Step 4: Test system
    print("\nStep 4: Testing system performance...")
    if not test_performance():
        print("⚠️  System rebuilt but performance test failed")
    else:
        print("✅ System successfully rebuilt and tested")
    
    return True

def show_help():
    """Show available commands"""
    print("Django RAG System Management Commands")
    print("=" * 40)
    print()
    print("Available commands:")
    print("  preprocess        - Preprocess data for faster responses")
    print("  status           - Check preprocessing status")
    print("  test            - Test system performance")
    print("  clear           - Clear preprocessing cache")
    print("  rebuild         - Rebuild entire system")
    print("  help            - Show this help message")
    print()
    print("Usage:")
    print("  python management_commands.py <command>")
    print()
    print("Examples:")
    print("  python management_commands.py preprocess")
    print("  python management_commands.py status")
    print("  python management_commands.py test")

def main():
    """Main command dispatcher"""
    if len(sys.argv) < 2:
        show_help()
        return
    
    command = sys.argv[1].lower()
    
    commands = {
        'preprocess': preprocess_data,
        'status': check_preprocessing_status,
        'test': test_performance,
        'clear': clear_preprocessing_cache,
        'rebuild': rebuild_system,
        'help': show_help
    }
    
    if command in commands:
        print(f"Executing: {command}")
        print("=" * 30)
        commands[command]()
    else:
        print(f"Unknown command: {command}")
        print()
        show_help()

if __name__ == "__main__":
    main()
