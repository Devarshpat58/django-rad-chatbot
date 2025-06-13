"""
URL configuration for web interface
Defines routes for HTML templates and AJAX endpoints
"""

from django.urls import path
from . import views

app_name = 'web_interface'

urlpatterns = [
    # Main pages
    path('', views.HomeView.as_view(), name='home'),
    path('search/', views.SearchInterfaceView.as_view(), name='search'),
    path('chat/', views.ChatInterfaceView.as_view(), name='chat'),
    path('docs/', views.DocumentationView.as_view(), name='documentation'),
    path('dashboard/', views.DashboardView.as_view(), name='dashboard'),
    
    # AJAX endpoints
    path('ajax/search/', views.ajax_search, name='ajax_search'),
    path('ajax/chat/', views.ajax_chat, name='ajax_chat'),
    path('ajax/test/', views.ajax_test, name='ajax_test'),
    path('ajax/status/', views.ajax_system_status, name='ajax_status'),
]
