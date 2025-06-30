"""
Django REST Framework Views for RAG API
Implements RESTful endpoints for the RAG system
"""

import logging
import uuid
from datetime import datetime
from typing import Dict, Any

from django.shortcuts import get_object_or_404
from django.utils import timezone
from django.contrib.sessions.models import Session
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator

from rest_framework import status, viewsets, permissions
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.decorators import action, api_view, permission_classes
from rest_framework.permissions import IsAuthenticated, AllowAny
from rest_framework.pagination import PageNumberPagination
from rest_framework.filters import SearchFilter, OrderingFilter
from django_filters.rest_framework import DjangoFilterBackend

from drf_spectacular.utils import extend_schema, OpenApiParameter
from drf_spectacular.types import OpenApiTypes

from .models import SearchSession, SearchQuery, SystemMetrics, PropertyDocument
from .serializers import (
    SearchRequestSerializer, SearchResponseSerializer, SearchQuerySerializer,
    PropertyDocumentSerializer, PropertySearchFiltersSerializer, SearchSessionSerializer,
    SystemStatusSerializer, SystemMetricsSerializer, ChatMessageSerializer,
    ChatResponseSerializer, BulkSearchSerializer, BulkSearchResponseSerializer
)
from .services import RAGService
from .translation_service import translate_to_english, is_translation_available

logger = logging.getLogger(__name__)


class StandardResultsSetPagination(PageNumberPagination):
    page_size = 20
    page_size_query_param = 'page_size'
    max_page_size = 100


class SearchAPIView(APIView):
    """
    Main search endpoint for RAG queries
    """
    permission_classes = [AllowAny]
    
    @extend_schema(
        request=SearchRequestSerializer,
        responses=SearchResponseSerializer,
        description="Process a search query using the RAG system",
        examples=[
            {
                "query": "Find 2 bedroom apartments under $2000",
                "max_results": 5,
                "include_metadata": True
            }
        ]
    )
    def post(self, request):
        serializer = SearchRequestSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        
        data = serializer.validated_data
        query_text = data['query']
        session_id = data.get('session_id') or getattr(request.session, 'session_key', None) or str(uuid.uuid4())
        max_results = data.get('max_results', 5)
        include_metadata = data.get('include_metadata', True)
        filters = data.get('filters', {})
        
        try:
            # Get or create search session  
            if not session_id:
                if not request.session.session_key:
                    request.session.create()
                session_id = request.session.session_key or f'anonymous_{timezone.now().timestamp()}'
            
            search_session, created = SearchSession.objects.get_or_create(
                session_key=session_id,
                defaults={'user': request.user if request.user.is_authenticated else None}
            )
            
            # Process query using RAG service
            rag_service = RAGService.get_instance()
            start_time = timezone.now()
            
            # Translate query to English if needed
            translation_result = translate_to_english(query_text)
            english_query = translation_result['english_query']
            original_language = translation_result['detected_language']
            translation_needed = translation_result['translation_needed']
            
            # Log translation if it occurred
            if translation_needed:
                logger.info(f"Translated query from {original_language}: '{query_text}' -> '{english_query}'")
            
            result = rag_service.process_query(
                query_text=english_query,  # Use translated query for processing
                session_id=session_id,
                max_results=max_results,
                filters=filters
            )
            
            # Store query in database
            search_query = SearchQuery.objects.create(
                session=search_session,
                query_text=query_text,
                processed_query={'filters': filters, 'max_results': max_results},
                response=result['response'],
                response_metadata=result['metadata'] if include_metadata else {},
                execution_time=result['metadata'].get('execution_time', 0),
                num_results=result['metadata'].get('num_results', 0),
                max_similarity_score=result['metadata'].get('max_similarity_score'),
                avg_similarity_score=result['metadata'].get('avg_similarity_score')
            )
            
            # Prepare response
            response_data = {
                'query_id': search_query.id,
                'session_id': search_session.session_key,
                'query_text': query_text,
                'response': result['response'],
                'results': result['metadata'].get('results', []),
                'execution_time': result['metadata'].get('execution_time', 0),
                'timestamp': search_query.created_at.isoformat(),
                'english_query': english_query,  # Include translated query
                'translation': {
                    'original_language': original_language,
                    'translation_needed': translation_needed,
                    'translation_available': is_translation_available()
                }
            }
            
            if include_metadata:
                response_data['metadata'] = result['metadata']
            
            return Response(response_data, status=status.HTTP_200_OK)
            
        except Exception as e:
            logger.error(f"Search API error: {str(e)}")
            return Response(
                {'error': f'Search failed: {str(e)}'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class ChatAPIView(APIView):
    """
    Chat-style interaction endpoint
    """
    permission_classes = [AllowAny]
    
    @extend_schema(
        request=ChatMessageSerializer,
        responses=ChatResponseSerializer,
        description="Chat-style interaction with the RAG system"
    )
    def post(self, request):
        serializer = ChatMessageSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        
        data = serializer.validated_data
        message = data['message']
        conversation_id = data.get('conversation_id') or request.session.session_key
        
        try:
            # Translate message to English if needed
            translation_result = translate_to_english(message)
            english_message = translation_result['english_query']
            original_language = translation_result['detected_language']
            translation_needed = translation_result['translation_needed']
            
            # Log translation if it occurred
            if translation_needed:
                logger.info(f"Translated chat message from {original_language}: '{message}' -> '{english_message}'")
            
            # Process message using RAG service
            rag_service = RAGService.get_instance()
            result = rag_service.process_query(
                query_text=english_message,  # Use translated message
                session_id=conversation_id
            )
            
            # Store in search system
            if not conversation_id:
                if not request.session.session_key:
                    request.session.create()
                conversation_id = request.session.session_key or f'chat_{timezone.now().timestamp()}'
            
            search_session, created = SearchSession.objects.get_or_create(
                session_key=conversation_id,
                defaults={'user': request.user if request.user.is_authenticated else None}
            )
            
            search_query = SearchQuery.objects.create(
                session=search_session,
                query_text=message,
                response=result['response'],
                response_metadata=result['metadata'],
                execution_time=result['metadata'].get('execution_time', 0)
            )
            
            response_data = {
                'response': result['response'],
                'conversation_id': conversation_id,
                'message_id': search_query.id,
                'timestamp': search_query.created_at.isoformat(),
                'sources': result['metadata'].get('sources', []),
                'confidence': result['metadata'].get('confidence', 0.0),
                # Enhanced data from main.py functionality
                'results': result['metadata'].get('results', []),
                'execution_time': result['metadata'].get('execution_time', 0),
                'num_results': result['metadata'].get('num_results', 0),
                'max_similarity_score': result['metadata'].get('max_similarity_score', 0.0),
                'avg_similarity_score': result['metadata'].get('avg_similarity_score', 0.0),
                'english_message': english_message,  # Include translated message
                'translation': {
                    'original_language': original_language,
                    'translation_needed': translation_needed,
                    'translation_available': is_translation_available()
                }
            }
            
            return Response(response_data, status=status.HTTP_200_OK)
            
        except Exception as e:
            logger.error(f"Chat API error: {str(e)}")
            return Response(
                {'error': f'Chat failed: {str(e)}'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class SystemStatusAPIView(APIView):
    """
    System status and health check endpoint
    """
    permission_classes = [AllowAny]
    
    @extend_schema(
        responses=SystemStatusSerializer,
        description="Get current system status and health information"
    )
    def get(self, request):
        try:
            rag_service = RAGService.get_instance()
            system_status = rag_service.get_system_status()
            
            return Response(system_status, status=status.HTTP_200_OK)
            
        except Exception as e:
            logger.error(f"System status error: {str(e)}")
            return Response(
                {
                    'status': 'error',
                    'error': str(e),
                    'last_updated': timezone.now()
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class HealthCheckAPIView(APIView):
    """
    Health check endpoint for monitoring
    """
    permission_classes = [AllowAny]
    
    @extend_schema(
        description="Perform system health check",
        responses={200: {'type': 'object', 'properties': {'healthy': {'type': 'boolean'}}}}
    )
    def get(self, request):
        try:
            rag_service = RAGService.get_instance()
            health = rag_service.health_check()
            
            status_code = status.HTTP_200_OK if health['healthy'] else status.HTTP_503_SERVICE_UNAVAILABLE
            return Response(health, status=status_code)
            
        except Exception as e:
            logger.error(f"Health check error: {str(e)}")
            return Response(
                {
                    'healthy': False,
                    'error': str(e),
                    'timestamp': timezone.now()
                },
                status=status.HTTP_503_SERVICE_UNAVAILABLE
            )


class TranslationStatusAPIView(APIView):
    """
    Translation service status endpoint
    """
    permission_classes = [AllowAny]
    
    @extend_schema(
        description="Get translation service status and supported languages",
        responses={200: {'type': 'object', 'properties': {'available': {'type': 'boolean'}}}}
    )
    def get(self, request):
        try:
            from .translation_service import get_translation_service
            
            service = get_translation_service()
            status_info = service.get_service_status()
            
            return Response({
                'translation_available': status_info['available'],
                'service_name': status_info['service'],
                'supported_languages_count': status_info['supported_languages_count'],
                'cache_size': status_info['cache_size'],
                'supported_languages': service.get_supported_languages() if status_info['available'] else {},
                'timestamp': timezone.now()
            }, status=status.HTTP_200_OK)
            
        except Exception as e:
            logger.error(f"Translation status error: {str(e)}")
            return Response(
                {
                    'translation_available': False,
                    'error': str(e),
                    'timestamp': timezone.now()
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

class PropertyDocumentViewSet(viewsets.ModelViewSet):
    """
    ViewSet for managing property documents
    """
    queryset = PropertyDocument.objects.all()
    serializer_class = PropertyDocumentSerializer
    permission_classes = [AllowAny]
    pagination_class = StandardResultsSetPagination
    filter_backends = [DjangoFilterBackend, SearchFilter, OrderingFilter]
    filterset_fields = ['property_type', 'bedrooms', 'accommodates', 'is_active']
    search_fields = ['title', 'location', 'search_vector']
    ordering_fields = ['price', 'bedrooms', 'indexed_at']
    ordering = ['-indexed_at']
    
    @extend_schema(
        description="Search properties with advanced filters",
        request=PropertySearchFiltersSerializer
    )
    @action(detail=False, methods=['post'])
    def search(self, request):
        filter_serializer = PropertySearchFiltersSerializer(data=request.data)
        if not filter_serializer.is_valid():
            return Response(filter_serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        
        filters = filter_serializer.validated_data
        
        try:
            rag_service = RAGService.get_instance()
            results = rag_service.search_properties(filters)
            
            return Response({
                'results': results,
                'filters_applied': filters,
                'timestamp': timezone.now()
            }, status=status.HTTP_200_OK)
            
        except Exception as e:
            logger.error(f"Property search error: {str(e)}")
            return Response(
                {'error': f'Property search failed: {str(e)}'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class SearchSessionViewSet(viewsets.ModelViewSet):
    """
    ViewSet for managing search sessions
    """
    queryset = SearchSession.objects.all()
    serializer_class = SearchSessionSerializer
    permission_classes = [AllowAny]
    pagination_class = StandardResultsSetPagination
    filter_backends = [DjangoFilterBackend, OrderingFilter]
    filterset_fields = ['is_active', 'user']
    ordering = ['-created_at']
    
    def get_queryset(self):
        queryset = super().get_queryset()
        if self.request.user.is_authenticated:
            return queryset.filter(user=self.request.user)
        return queryset.filter(session_key=self.request.session.session_key)
    
    @extend_schema(
        description="Get conversation history for a session",
        responses=SearchQuerySerializer(many=True)
    )
    @action(detail=True, methods=['get'])
    def history(self, request, pk=None):
        session = self.get_object()
        queries = session.queries.all()[:50]  # Limit to last 50 queries
        serializer = SearchQuerySerializer(queries, many=True)
        return Response(serializer.data)


class SearchQueryViewSet(viewsets.ReadOnlyModelViewSet):
    """
    ViewSet for viewing search query history
    """
    queryset = SearchQuery.objects.all()
    serializer_class = SearchQuerySerializer
    permission_classes = [AllowAny]
    pagination_class = StandardResultsSetPagination
    filter_backends = [DjangoFilterBackend, SearchFilter, OrderingFilter]
    filterset_fields = ['session', 'created_at']
    search_fields = ['query_text', 'response']
    ordering = ['-created_at']
    
    def get_queryset(self):
        queryset = super().get_queryset()
        if self.request.user.is_authenticated:
            return queryset.filter(session__user=self.request.user)
        session_key = self.request.session.session_key
        if session_key:
            return queryset.filter(session__session_key=session_key)
        return queryset.none()


class BulkSearchAPIView(APIView):
    """
    Bulk search endpoint for processing multiple queries
    """
    permission_classes = [AllowAny]
    
    @extend_schema(
        request=BulkSearchSerializer,
        responses=BulkSearchResponseSerializer,
        description="Process multiple search queries in bulk"
    )
    def post(self, request):
        serializer = BulkSearchSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        
        data = serializer.validated_data
        queries = data['queries']
        session_id = data.get('session_id') or request.session.session_key
        include_metadata = data.get('include_metadata', False)
        
        try:
            rag_service = RAGService.get_instance()
            
            if not session_id:
                if not request.session.session_key:
                    request.session.create()
                session_id = request.session.session_key or f'bulk_{timezone.now().timestamp()}'
            
            search_session, created = SearchSession.objects.get_or_create(
                session_key=session_id,
                defaults={'user': request.user if request.user.is_authenticated else None}
            )
            
            results = []
            successful_queries = 0
            failed_queries = 0
            start_time = timezone.now()
            
            for i, query_text in enumerate(queries):
                try:
                    # Translate query to English if needed
                    translation_result = translate_to_english(query_text)
                    english_query = translation_result['english_query']
                    original_language = translation_result['detected_language']
                    translation_needed = translation_result['translation_needed']
                    
                    # Log translation if it occurred
                    if translation_needed:
                        logger.info(f"Translated bulk query {i+1} from {original_language}: '{query_text}' -> '{english_query}'")
                    
                    result = rag_service.process_query(
                        query_text=english_query,
                        session_id=session_id
                    )
                    
                    search_query = SearchQuery.objects.create(
                        session=search_session,
                        query_text=query_text,
                        response=result['response'],
                        response_metadata=result['metadata'] if include_metadata else {},
                        execution_time=result['metadata'].get('execution_time', 0)
                    )
                    
                    query_result = {
                        'query_id': search_query.id,
                        'query_text': query_text,
                        'response': result['response'],
                        'success': True
                    }
                    
                    if include_metadata:
                        query_result['metadata'] = result['metadata']
                    
                    results.append(query_result)
                    successful_queries += 1
                    
                except Exception as e:
                    results.append({
                        'query_text': query_text,
                        'error': str(e),
                        'success': False
                    })
                    failed_queries += 1
            
            total_execution_time = (timezone.now() - start_time).total_seconds()
            
            response_data = {
                'results': results,
                'session_id': session_id,
                'total_queries': len(queries),
                'successful_queries': successful_queries,
                'failed_queries': failed_queries,
                'total_execution_time': total_execution_time,
                'timestamp': timezone.now()
            }
            
            return Response(response_data, status=status.HTTP_200_OK)
            
        except Exception as e:
            logger.error(f"Bulk search error: {str(e)}")
            return Response(
                {'error': f'Bulk search failed: {str(e)}'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class SystemMetricsViewSet(viewsets.ReadOnlyModelViewSet):
    """
    ViewSet for system metrics
    """
    queryset = SystemMetrics.objects.all()
    serializer_class = SystemMetricsSerializer
    permission_classes = [IsAuthenticated]
    pagination_class = StandardResultsSetPagination
    filter_backends = [DjangoFilterBackend, OrderingFilter]
    filterset_fields = ['metric_name', 'timestamp']
    ordering = ['-timestamp']


# Simple function-based views for quick endpoints
@extend_schema(
    responses={200: OpenApiTypes.OBJECT},
    description="API root endpoint with available endpoints and service information"
)
@api_view(['GET'])
@permission_classes([AllowAny])
def api_root(request):
    """
    API root endpoint with available endpoints
    """
    return Response({
        'message': 'Django RAG API v1.0',
        'endpoints': {
            'search': '/api/v1/search/',
            'chat': '/api/v1/chat/',
            'bulk_search': '/api/v1/bulk-search/',
            'properties': '/api/v1/properties/',
            'sessions': '/api/v1/sessions/',
            'queries': '/api/v1/queries/',
            'system_status': '/api/v1/status/',
            'health_check': '/api/v1/health/',
            'docs': '/api/docs/',
            'redoc': '/api/redoc/',
            'schema': '/api/schema/'
        },
        'timestamp': timezone.now()
    })


@api_view(['GET'])
@permission_classes([AllowAny])
def api_stats(request):
    """
    API usage statistics
    """
    try:
        total_searches = SearchQuery.objects.count()
        total_sessions = SearchSession.objects.count()
        active_sessions = SearchSession.objects.filter(is_active=True).count()
        
        # Recent activity (last 24 hours)
        from django.utils import timezone
        from datetime import timedelta
        recent_cutoff = timezone.now() - timedelta(hours=24)
        recent_searches = SearchQuery.objects.filter(created_at__gte=recent_cutoff).count()
        
        return Response({
            'total_searches': total_searches,
            'total_sessions': total_sessions,
            'active_sessions': active_sessions,
            'recent_searches_24h': recent_searches,
            'timestamp': timezone.now()
        })
        
    except Exception as e:
        logger.error(f"Stats error: {str(e)}")
        return Response(
            {'error': f'Stats unavailable: {str(e)}'},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
