#!/usr/bin/env python3
"""
Demonstration Script for Enhanced RAG Features

This script demonstrates the new enhanced features:
1. Mandatory property fields (always shown)
2. Table comparison UI for multiple properties
3. Improved JSON source data formatting
4. Enhanced AI summaries with minimum word count
"""

import os
import sys
import django
from datetime import datetime

# Setup Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'django_rag_project.settings')
django.setup()

from rag_api.services_enhanced import EnhancedRAGService
from rag_api.models import SearchQuery, SearchSession
import json


def demonstrate_enhanced_features():
    """
    Demonstrate all enhanced features
    """
    print("🚀 Enhanced RAG Chatbot Features Demonstration")
    print("=" * 60)
    
    try:
        # Initialize Enhanced RAG Service
        print("\n📋 Initializing Enhanced RAG Service...")
        enhanced_service = EnhancedRAGService.get_instance()
        
        if not enhanced_service.is_ready():
            print("❌ Enhanced RAG service is not ready. Please check initialization.")
            return
        
        print("✅ Enhanced RAG service is ready!")
        
        # Test queries that will demonstrate different features
        demo_queries = [
            {
                'query': 'Find 2 bedroom apartments under $2000 with parking',
                'description': 'Basic property search with filters'
            },
            {
                'query': 'Show me luxury properties with pools and gyms near downtown',
                'description': 'Complex search with multiple amenities'
            },
            {
                'query': 'Compare family-friendly houses with 3+ bedrooms',
                'description': 'Search that will trigger comparison table'
            }
        ]
        
        # Process each demo query
        for i, demo in enumerate(demo_queries, 1):
            print(f"\n{'='*40}")
            print(f"Demo {i}: {demo['description']}")
            print(f"Query: '{demo['query']}'")
            print(f"{'='*40}")
            
            try:
                # Process with enhanced features enabled
                result = enhanced_service.process_query(
                    query_text=demo['query'],
                    session_id=f"demo_session_{i}",
                    max_results=5,
                    enable_comparison=True
                )
                
                if result['success']:
                    print("\n✅ Query processed successfully!")
                    
                    metadata = result['metadata']
                    
                    # Show execution metrics
                    print(f"⚡ Execution time: {metadata.get('execution_time', 0):.2f}s")
                    print(f"📊 Results found: {metadata.get('num_results', 0)}")
                    print(f"🎯 Max relevance: {metadata.get('max_similarity_score', 0)*100:.1f}%")
                    
                    # Demonstrate mandatory fields feature
                    print("\n🔑 MANDATORY FIELDS FEATURE:")
                    results = metadata.get('results', [])
                    if results:
                        for idx, result_item in enumerate(results[:2], 1):
                            print(f"\n  Property {idx} - Always Shown Fields:")
                            mandatory_fields = result_item.get('mandatory_fields', {})
                            if mandatory_fields:
                                for field, value in mandatory_fields.items():
                                    print(f"    • {field.title()}: {value}")
                            else:
                                print("    • No mandatory fields extracted")
                    
                    # Demonstrate enhanced AI summaries
                    print("\n🤖 ENHANCED AI SUMMARIES (200+ words):")
                    if results:
                        for idx, result_item in enumerate(results[:1], 1):
                            ai_summary = result_item.get('ai_summary', 'No summary available')
                            word_count = len(ai_summary.split())
                            print(f"\n  Property {idx} Summary ({word_count} words):")
                            print(f"  {ai_summary[:300]}..." if len(ai_summary) > 300 else f"  {ai_summary}")
                    
                    # Demonstrate improved JSON formatting
                    print("\n📄 IMPROVED JSON FORMATTING:")
                    if results:
                        result_item = results[0]
                        source_json = result_item.get('source_json', {})
                        if isinstance(source_json, dict):
                            # Show organized structure
                            print("  JSON is now organized into logical groups:")
                            for group_name in source_json.keys():
                                print(f"    📁 {group_name}")
                        else:
                            print("  Standard JSON structure maintained")
                    
                    # Demonstrate table comparison feature
                    print("\n📊 TABLE COMPARISON FEATURE:")
                    if metadata.get('comparison_enabled') and len(results) > 1:
                        print(f"  ✅ Comparison table ready for {len(results)} properties")
                        comparison_summary = metadata.get('comparison_summary', {})
                        if comparison_summary:
                            print(f"  📈 Price range: ${comparison_summary.get('price_range', {}).get('min', 'N/A')} - ${comparison_summary.get('price_range', {}).get('max', 'N/A')}")
                            print(f"  ⭐ Average rating: {comparison_summary.get('average_rating', 'N/A'):.1f}/100" if comparison_summary.get('average_rating') else "  ⭐ Average rating: N/A")
                            print(f"  🏠 Property types: {', '.join(comparison_summary.get('property_types', []))}")
                            print(f"  📍 Locations: {', '.join(comparison_summary.get('locations', []))}")
                    else:
                        print("  ℹ️ Comparison table available when multiple results are found")
                    
                    # Show response preview
                    response = result['response']
                    print(f"\n💬 AI Response Preview:")
                    print(f"  {response[:200]}..." if len(response) > 200 else f"  {response}")
                    
                else:
                    print(f"❌ Query failed: {result.get('metadata', {}).get('error', 'Unknown error')}")
                    
            except Exception as e:
                print(f"❌ Error processing query: {str(e)}")
        
        # Demonstrate system status
        print(f"\n{'='*40}")
        print("Enhanced System Status")
        print(f"{'='*40}")
        
        status = enhanced_service.get_system_status()
        print(f"🟢 Status: {status.get('status', 'unknown')}")
        print(f"🔧 Enhanced features: {status.get('enhanced_features_enabled', False)}")
        print(f"📋 Mandatory fields support: {status.get('mandatory_fields_support', False)}")
        print(f"📊 Comparison support: {status.get('comparison_support', False)}")
        
        # Health check
        print(f"\n{'='*40}")
        print("Health Check")
        print(f"{'='*40}")
        
        health = enhanced_service.health_check()
        print(f"❤️ System healthy: {health.get('healthy', False)}")
        
        enhanced_features = health.get('enhanced_features', {})
        print(f"🔑 Mandatory fields: {enhanced_features.get('mandatory_fields', False)}")
        print(f"📊 Comparison support: {enhanced_features.get('comparison_support', False)}")
        print(f"📄 Enhanced JSON: {enhanced_features.get('improved_json_formatting', False)}")
        
        print("\n🎉 Enhanced features demonstration complete!")
        print("\n📝 Summary of Enhancements:")
        print("   1. ✅ Mandatory fields (name, price, location, bedrooms) always shown")
        print("   2. ✅ Table comparison UI for multiple properties")
        print("   3. ✅ Improved JSON formatting with logical grouping")
        print("   4. ✅ Enhanced AI summaries with 200+ word minimum")
        print("   5. ✅ Better field extraction and formatting")
        print("   6. ✅ Comparison statistics and summaries")
        
    except Exception as e:
        print(f"❌ Failed to demonstrate enhanced features: {str(e)}")
        import traceback
        print("\n🔍 Detailed error:")
        traceback.print_exc()


def test_web_interface_integration():
    """
    Test web interface integration with enhanced features
    """
    print("\n🌐 Testing Web Interface Integration")
    print("=" * 50)
    
    try:
        from web_interface.views import ajax_chat
        from django.test import RequestFactory
        from django.contrib.sessions.middleware import SessionMiddleware
        import json
        
        # Create a test request
        factory = RequestFactory()
        request = factory.post(
            '/ajax/chat/',
            data=json.dumps({
                'message': 'Find 2 bedroom apartments with parking under $1800',
                'session_id': 'test_enhanced_session'
            }),
            content_type='application/json'
        )
        
        # Add session
        middleware = SessionMiddleware()
        middleware.process_request(request)
        request.session.save()
        
        print("📤 Test request created")
        print("🔄 Processing with enhanced service...")
        
        # This should now use EnhancedRAGService
        response = ajax_chat(request)
        
        if response.status_code == 200:
            print("✅ Web interface integration successful!")
            response_data = json.loads(response.content)
            print(f"📊 Results: {response_data.get('num_results', 0)}")
            print(f"⚡ Time: {response_data.get('execution_time', 0):.2f}s")
            print(f"📋 Enhanced data available: {bool(response_data.get('results'))}")
        else:
            print(f"❌ Web interface test failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Web interface test error: {str(e)}")


if __name__ == "__main__":
    print("🏠 Django RAG Chatbot - Enhanced Features Demo")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Run demonstrations
    demonstrate_enhanced_features()
    test_web_interface_integration()
    
    print(f"\n✨ Demo completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n🎯 Next Steps:")
    print("   1. Start the Django server: python manage.py runserver")
    print("   2. Visit: http://localhost:8000/chat/")
    print("   3. Try queries like 'Find 2BR apartments under $2000'")
    print("   4. Look for the table comparison button when multiple results appear")
    print("   5. Check the enhanced JSON formatting in source data sections")
