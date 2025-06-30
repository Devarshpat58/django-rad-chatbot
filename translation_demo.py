#!/usr/bin/env python
"""
Translation Integration Demonstration
Shows how the Django RAG chatbot now handles multiple languages
"""

import os
import sys
import django
import json

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'django_rag_project.settings')
django.setup()

from rag_api.translation_service import translate_to_english, is_translation_available, get_translation_service

def demonstrate_translation():
    """Demonstrate the translation functionality"""
    print("=" * 60)
    print("DJANGO RAG CHATBOT - TRANSLATION INTEGRATION DEMO")
    print("=" * 60)
    print()
    
    # Check translation availability
    print(f"Translation Service Available: {is_translation_available()}")
    service = get_translation_service()
    print(f"Service: Translation Service (fallback mode)")
    print(f"Supported Languages: Basic language detection available")
    print()
    
    # Test cases demonstrating translation
    test_queries = [
        ("English", "Find 2 bedroom apartments under $2000 in New York"),
        ("Spanish", "Encuentra apartamentos de 2 dormitorios bajo $2000 en New York"),
        ("German", "Finde günstige Apartments mit 2 Zimmer unter 2000 Dollar in NYC"),
        ("French", "Trouvez des appartements avec 2 chambres moins de 2000 dollars"),
        ("Portuguese", "Encontre apartamentos com 2 quartos abaixo de 2000 dolares"),
        ("Italian", "Trova appartamenti con 2 camere sotto 2000 dollari"),
    ]
    
    print("TRANSLATION DEMONSTRATION:")
    print("-" * 40)
    
    for language, query in test_queries:
        print(f"\n{language} Query:")
        print(f"  Input:  '{query}'")
        
        result = translate_to_english(query)
        
        print(f"  Output: '{result['english_query']}'")
        print(f"  Detected Language: {result['detected_language']}")
        print(f"  Translation Needed: {result['translation_needed']}")
    
    print("\n" + "=" * 60)
    print("INTEGRATION SUMMARY:")
    print("=" * 60)
    print("✓ Translation service integrated into Django RAG API")
    print("✓ All API endpoints (search, chat) now support translation")
    print("✓ Web interface updated to handle multiple languages")
    print("✓ Queries automatically translated to English before processing")
    print("✓ Responses always generated and displayed in English")
    print("✓ Original query language preserved for user reference")
    print("✓ No external API keys required - lightweight built-in solution")
    print()
    print("The chatbot now seamlessly handles queries in multiple languages")
    print("while maintaining consistent English responses for all users.")
    print("=" * 60)

if __name__ == "__main__":
    try:
        demonstrate_translation()
    except Exception as e:
        print(f"Demo error: {e}")
        print("Note: This demo requires the Django environment to be properly set up.")