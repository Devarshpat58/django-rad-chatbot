"""
URL configuration for RAG API endpoints
Defines all REST API routes for the Django RAG system
"""

from django.urls import path, include
from rest_framework.routers import DefaultRouter
from rest_framework.urlpatterns import format_suffix_patterns

from . import views

# Create a router for ViewSets
router = DefaultRouter()
router.register(r'properties', views.PropertyDocumentViewSet, basename='property')
router.register(r'sessions', views.SearchSessionViewSet, basename='session')
router.register(r'queries', views.SearchQueryViewSet, basename='query')
router.register(r'metrics', views.SystemMetricsViewSet, basename='metrics')

app_name = 'rag_api'

urlpatterns = [
    # API Root and Info
    path('', views.api_root, name='api_root'),
    path('stats/', views.api_stats, name='api_stats'),
    
    # Core Search Endpoints
    path('search/', views.SearchAPIView.as_view(), name='search'),
    path('chat/', views.ChatAPIView.as_view(), name='chat'),
    path('bulk-search/', views.BulkSearchAPIView.as_view(), name='bulk_search'),
    
    # System Endpoints
    path('status/', views.SystemStatusAPIView.as_view(), name='system_status'),
    path('health/', views.HealthCheckAPIView.as_view(), name='health_check'),
    
    # Include router URLs for ViewSets
    path('', include(router.urls)),
]

# Note: format_suffix_patterns can cause conflicts with router URLs
# urlpatterns = format_suffix_patterns(urlpatterns)
