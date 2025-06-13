"""
URL configuration for Django RAG project.
Includes both API endpoints and web interface routes.
"""

from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
# Removed drf_spectacular imports to eliminate warnings

urlpatterns = [
    # Admin interface
    path('admin/', admin.site.urls),
    
    # API Documentation removed to eliminate warnings
    
    # API routes
    path('api/v1/', include('rag_api.urls')),
    
    # Web interface routes
    path('', include('web_interface.urls')),
]

# Serve static and media files during development
if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
    
    # Add debug toolbar if available
    if 'debug_toolbar' in settings.INSTALLED_APPS:
        import debug_toolbar
        urlpatterns = [path('__debug__/', include(debug_toolbar.urls))] + urlpatterns
