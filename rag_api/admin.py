"""
Django admin configuration for RAG API models
"""

from django.contrib import admin
from django.utils.html import format_html
from django.urls import reverse
from django.utils.safestring import mark_safe
from .models import SearchSession, SearchQuery, SystemMetrics, PropertyDocument


@admin.register(SearchSession)
class SearchSessionAdmin(admin.ModelAdmin):
    list_display = ['session_key_short', 'user', 'queries_count', 'created_at', 'is_active']
    list_filter = ['is_active', 'created_at', 'user']
    search_fields = ['session_key', 'user__username']
    readonly_fields = ['id', 'session_key', 'created_at', 'updated_at']
    ordering = ['-created_at']
    
    def session_key_short(self, obj):
        return f"{obj.session_key[:8]}..."
    session_key_short.short_description = "Session"
    
    def queries_count(self, obj):
        count = obj.queries.count()
        url = reverse('admin:rag_api_searchquery_changelist') + f'?session__id__exact={obj.id}'
        return format_html('<a href="{}">{} queries</a>', url, count)
    queries_count.short_description = "Queries"
    
    def get_queryset(self, request):
        return super().get_queryset(request).select_related('user')


@admin.register(SearchQuery)
class SearchQueryAdmin(admin.ModelAdmin):
    list_display = ['query_text_short', 'session_link', 'execution_time', 'num_results', 'created_at']
    list_filter = ['created_at', 'num_results', 'session__is_active']
    search_fields = ['query_text', 'response', 'session__session_key']
    readonly_fields = ['id', 'created_at', 'execution_time', 'response_preview']
    ordering = ['-created_at']
    
    fieldsets = (
        ('Query Information', {
            'fields': ('session', 'query_text', 'processed_query')
        }),
        ('Response', {
            'fields': ('response', 'response_preview', 'response_metadata')
        }),
        ('Metrics', {
            'fields': ('execution_time', 'num_results', 'max_similarity_score', 'avg_similarity_score')
        }),
        ('Timestamps', {
            'fields': ('id', 'created_at')
        })
    )
    
    def query_text_short(self, obj):
        return obj.query_text[:50] + "..." if len(obj.query_text) > 50 else obj.query_text
    query_text_short.short_description = "Query"
    
    def session_link(self, obj):
        url = reverse('admin:rag_api_searchsession_change', args=[obj.session.pk])
        return format_html('<a href="{}">{}</a>', url, obj.session.session_key[:8])
    session_link.short_description = "Session"
    
    def response_preview(self, obj):
        if obj.response:
            preview = obj.response[:200] + "..." if len(obj.response) > 200 else obj.response
            return mark_safe(f'<div style="max-width: 400px; word-wrap: break-word;">{preview}</div>')
        return "No response"
    response_preview.short_description = "Response Preview"
    
    def get_queryset(self, request):
        return super().get_queryset(request).select_related('session')


@admin.register(PropertyDocument)
class PropertyDocumentAdmin(admin.ModelAdmin):
    list_display = ['title_short', 'property_type', 'price', 'bedrooms', 'accommodates', 'location', 'is_active']
    list_filter = ['property_type', 'bedrooms', 'accommodates', 'is_active', 'indexed_at']
    search_fields = ['title', 'document_id', 'location', 'search_vector']
    readonly_fields = ['document_id', 'indexed_at', 'last_updated']
    ordering = ['-indexed_at']
    
    fieldsets = (
        ('Document Information', {
            'fields': ('document_id', 'title', 'is_active')
        }),
        ('Property Details', {
            'fields': ('property_type', 'price', 'bedrooms', 'accommodates', 'location')
        }),
        ('Amenities and Search', {
            'fields': ('amenities', 'search_vector')
        }),
        ('Timestamps', {
            'fields': ('indexed_at', 'last_updated')
        })
    )
    
    def title_short(self, obj):
        return obj.title[:30] + "..." if obj.title and len(obj.title) > 30 else (obj.title or obj.document_id)
    title_short.short_description = "Title"
    
    def get_queryset(self, request):
        return super().get_queryset(request)


@admin.register(SystemMetrics)
class SystemMetricsAdmin(admin.ModelAdmin):
    list_display = ['metric_name', 'metric_value', 'timestamp']
    list_filter = ['metric_name', 'timestamp']
    search_fields = ['metric_name']
    readonly_fields = ['timestamp']
    ordering = ['-timestamp']
    
    fieldsets = (
        ('Metric Information', {
            'fields': ('metric_name', 'metric_value', 'metadata')
        }),
        ('Timestamp', {
            'fields': ('timestamp',)
        })
    )
    
    def has_add_permission(self, request):
        # Metrics are usually added programmatically
        return False


# Custom admin site configuration
admin.site.site_header = "Django RAG API Administration"
admin.site.site_title = "RAG API Admin"
admin.site.index_title = "Welcome to RAG API Administration"
