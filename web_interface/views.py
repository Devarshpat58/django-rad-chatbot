"""
Web interface views for Django RAG API
Provides HTML templates and web-based interaction with translation support
"""

from django.shortcuts import render
from django.http import JsonResponse
from django.views.generic import TemplateView
from django.contrib import messages
from django.utils import timezone
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
import json
import logging

from rag_api.services import RAGService
from console_logger import ConsoleLogger
from rag_api.services_enhanced import EnhancedRAGService
from rag_api.models import SearchSession, SearchQuery
from rag_api.translation_service import translate_to_english

logger = logging.getLogger(__name__)


class HomeView(TemplateView):
    """
    Home page with search interface
    """
    template_name = 'web_interface/index.html'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        
        try:
            # Get system status
            rag_service = EnhancedRAGService.get_instance()
            system_status = rag_service.get_system_status()
            context['system_ready'] = system_status.get('status') == 'ready'
            
            # Get recent searches for this session
            session_key = self.request.session.session_key
            if session_key:
                recent_searches = SearchQuery.objects.filter(
                    session__session_key=session_key
                ).order_by('-created_at')[:5]
                context['recent_searches'] = recent_searches
            
        except Exception as e:
            logger.error(f"Error loading home page context: {e}")
            context['system_ready'] = False
            context['error'] = str(e)
        
        return context


class SearchInterfaceView(TemplateView):
    """
    Advanced search interface
    """
    template_name = 'web_interface/search.html'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        
        try:
            # Get system status
            rag_service = EnhancedRAGService.get_instance()
            system_status = rag_service.get_system_status()
            context['system_status'] = system_status
            
        except Exception as e:
            logger.error(f"Error loading search interface: {e}")
            context['error'] = str(e)
        
        return context


class ChatInterfaceView(TemplateView):
    """
    Chat-style interface
    """
    template_name = 'web_interface/chat.html'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        
        # Get conversation history
        session_key = self.request.session.session_key
        if session_key:
            try:
                session = SearchSession.objects.get(session_key=session_key)
                conversation_history = session.queries.order_by('created_at')[:20]
                context['conversation_history'] = conversation_history
            except SearchSession.DoesNotExist:
                context['conversation_history'] = []
        
        return context


class DocumentationView(TemplateView):
    """
    API documentation and usage guide
    """
    template_name = 'web_interface/documentation.html'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        
        # API endpoint information
        context['api_endpoints'] = {
            'search': {
                'url': '/api/v1/search/',
                'method': 'POST',
                'description': 'Process search queries using RAG system'
            },
            'chat': {
                'url': '/api/v1/chat/',
                'method': 'POST',
                'description': 'Chat-style interaction'
            },
            'properties': {
                'url': '/api/v1/properties/',
                'method': 'GET',
                'description': 'Browse property documents'
            },
            'status': {
                'url': '/api/v1/status/',
                'method': 'GET',
                'description': 'Get system status'
            }
        }
        
        return context


class DashboardView(TemplateView):
    """
    Admin dashboard for monitoring
    """
    template_name = 'web_interface/dashboard.html'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        
        try:
            # System metrics
            total_searches = SearchQuery.objects.count()
            total_sessions = SearchSession.objects.count()
            active_sessions = SearchSession.objects.filter(is_active=True).count()
            
            # Recent activity
            from datetime import timedelta
            recent_cutoff = timezone.now() - timedelta(hours=24)
            recent_searches = SearchQuery.objects.filter(created_at__gte=recent_cutoff)
            
            context.update({
                'total_searches': total_searches,
                'total_sessions': total_sessions,
                'active_sessions': active_sessions,
                'recent_searches_count': recent_searches.count(),
                'recent_searches': recent_searches[:10]
            })
            
            # System status
            rag_service = EnhancedRAGService.get_instance()
            system_status = rag_service.get_system_status()
            context['system_status'] = system_status
            
        except Exception as e:
            logger.error(f"Error loading dashboard: {e}")
            context['error'] = str(e)
        
        return context


@csrf_exempt
def ajax_search(request):
    """
    AJAX search endpoint for web interface with translation support
    """
    if request.method != 'POST':
        return JsonResponse({'error': 'Method not allowed'}, status=405)
    
    try:
        data = json.loads(request.body)
        query_text = data.get('query', '').strip()
        
        if not query_text:
            return JsonResponse({'error': 'Query text is required'}, status=400)
        
        # Get or create session
        if not request.session.session_key:
            request.session.create()
        session_key = request.session.session_key or f'anonymous_{timezone.now().timestamp()}'
        
        search_session, created = SearchSession.objects.get_or_create(
            session_key=session_key,
            defaults={'user': request.user if request.user.is_authenticated else None}
        )
        
        # Translate query to English if needed
        translation_result = translate_to_english(query_text)
        english_query = translation_result['english_query']
        original_language = translation_result['detected_language']
        translation_needed = translation_result['translation_needed']
        
        # Log translation if it occurred
        if translation_needed:
            logger.info(f"Translated query from {original_language}: '{query_text}' -> '{english_query}'")
        
        # Process query using translated text
        rag_service = EnhancedRAGService.get_instance()
        result = rag_service.process_query(
            query_text=english_query,  # Use translated query for processing
            session_id=session_key
        )

        # Add this line to log the response
        ConsoleLogger.log_django_web_response(
            query_text, result, session_key,
            user=request.user if hasattr(request, "user") else None,
            endpoint=request.path
        )
        
        # Store query (store original query text for user reference)
        search_query = SearchQuery.objects.create(
            session=search_session,
            query_text=query_text,  # Store original query
            response=result['response'],
            response_metadata=result['metadata'],
            execution_time=result['metadata'].get('execution_time', 0)
        )
        
        # Clean metadata of datetime objects
        def clean_metadata(obj):
            if hasattr(obj, 'isoformat'):  # datetime object
                return obj.isoformat()
            elif isinstance(obj, dict):
                return {k: clean_metadata(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [clean_metadata(item) for item in obj]
            elif isinstance(obj, (str, int, float, bool)) or obj is None:
                return obj
            else:
                return str(obj)
        
        cleaned_metadata = clean_metadata(result['metadata'])
        
        return JsonResponse({
            'success': True,
            'query_id': str(search_query.id),
            'response': result['response'],  # Response is always in English
            'metadata': cleaned_metadata,
            'timestamp': search_query.created_at.isoformat(),
            'english_query': english_query,  # Include translated query
            'translation': {
                'original_language': original_language,
                'translation_needed': translation_needed,
                'original_query': query_text
            }
        })
        
    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON in request body'}, status=400)
    except Exception as e:
        logger.error(f"AJAX search error: {str(e)}")
        return JsonResponse({
            'error': f'Search failed: {str(e)}',
            'success': False
        }, status=500)


@csrf_exempt
def ajax_chat(request):
    """
    AJAX chat endpoint for web interface with translation support
    """
    if request.method != 'POST':
        return JsonResponse({'error': 'Method not allowed'}, status=405)
    
    try:
        data = json.loads(request.body)
        message = data.get('message', '').strip()
        session_id = data.get('session_id', '')
        
        if not message:
            return JsonResponse({'error': 'Message is required'}, status=400)
        
        # Get or create session
        if not request.session.session_key:
            request.session.create()
        django_session_key = request.session.session_key or f'anonymous_{timezone.now().timestamp()}'
        
        # Use provided session_id or fall back to Django session
        effective_session_id = session_id or django_session_key
        
        search_session, created = SearchSession.objects.get_or_create(
            session_key=effective_session_id,
            defaults={'user': request.user if request.user.is_authenticated else None}
        )
        
        # Translate message to English if needed
        translation_result = translate_to_english(message)
        english_message = translation_result['english_query']
        original_language = translation_result['detected_language']
        translation_needed = translation_result['translation_needed']
        
        # Log translation if it occurred
        if translation_needed:
            logger.info(f"Translated chat message from {original_language}: '{message}' -> '{english_message}'")
        
        # Process query using chat API endpoint logic
        rag_service = EnhancedRAGService.get_instance()
        result = rag_service.process_query(
            query_text=english_message,  # Use translated message
            session_id=effective_session_id
        )

        # Add this line to log the chat response
        ConsoleLogger.log_django_web_response(
            message, result, effective_session_id,
            user=request.user if hasattr(request, "user") else None,
            endpoint=request.path
        )
        
        # Store query (store original message for user reference)
        search_query = SearchQuery.objects.create(
            session=search_session,
            query_text=message,  # Store original message
            response=result['response'],
            response_metadata=result['metadata'],
            execution_time=result['metadata'].get('execution_time', 0)
        )
        
        # Clean metadata of datetime objects
        def clean_metadata(obj):
            if hasattr(obj, 'isoformat'):  # datetime object
                return obj.isoformat()
            elif isinstance(obj, dict):
                return {k: clean_metadata(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [clean_metadata(item) for item in obj]
            elif isinstance(obj, (str, int, float, bool)) or obj is None:
                return obj
            else:
                return str(obj)
        
        cleaned_metadata = clean_metadata(result['metadata'])
        
        return JsonResponse({
            'success': True,
            'query_id': str(search_query.id),
            'response': result['response'],  # Response is always in English
            'metadata': cleaned_metadata,
            'timestamp': search_query.created_at.isoformat(),
            'session_id': effective_session_id,
            # Enhanced response data matching main.py functionality
            'results': cleaned_metadata.get('results', []),
            'execution_time': cleaned_metadata.get('execution_time', 0),
            'num_results': cleaned_metadata.get('num_results', 0),
            'max_similarity_score': cleaned_metadata.get('max_similarity_score', 0.0),
            'avg_similarity_score': cleaned_metadata.get('avg_similarity_score', 0.0),
            'english_message': english_message,  # Include translated message
            'translation': {
                'original_language': original_language,
                'translation_needed': translation_needed,
                'original_message': message
            }
        })
        
    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON in request body'}, status=400)
    except Exception as e:
        logger.error(f"AJAX chat error: {str(e)}")
        return JsonResponse({
            'error': 'The AI system is still starting up. Please wait a moment and try again.' if 'not initialized' in str(e) else f'Sorry, there was an error processing your request: {str(e)}',
            'success': False
        }, status=500)


def ajax_test(request):
    """Simple test endpoint"""
    return JsonResponse({'test': 'working', 'method': request.method})


def ajax_system_status(request):
    """
    AJAX endpoint for system status
    """
    try:
        rag_service = EnhancedRAGService.get_instance()
        system_status = rag_service.get_system_status()
        
        # Clean any datetime objects in system status
        def clean_status_data(obj):
            if hasattr(obj, 'isoformat'):  # datetime object
                return obj.isoformat()
            elif isinstance(obj, dict):
                return {k: clean_status_data(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [clean_status_data(item) for item in obj]
            elif isinstance(obj, (str, int, float, bool)) or obj is None:
                return obj
            else:
                return str(obj)
        
        cleaned_status = clean_status_data(system_status)
        return JsonResponse(cleaned_status)
        
    except Exception as e:
        logger.error(f"AJAX status error: {str(e)}")
        return JsonResponse({
            'error': str(e),
            'status': 'error'
        }, status=500)