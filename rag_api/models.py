from django.db import models
from django.contrib.auth.models import User
import uuid


class SearchSession(models.Model):
    """Track user search sessions"""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    session_key = models.CharField(max_length=40, db_index=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    is_active = models.BooleanField(default=True)
    
    class Meta:
        ordering = ['-created_at']
        verbose_name = 'Search Session'
        verbose_name_plural = 'Search Sessions'
    
    def __str__(self):
        return f"Session {self.session_key[:8]} - {self.created_at}"


class SearchQuery(models.Model):
    """Store search queries and their results"""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    session = models.ForeignKey(SearchSession, on_delete=models.CASCADE, related_name='queries')
    query_text = models.TextField()
    processed_query = models.JSONField(default=dict, blank=True)
    response = models.TextField(blank=True)
    response_metadata = models.JSONField(default=dict, blank=True)
    execution_time = models.FloatField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    # Search result metrics
    num_results = models.IntegerField(default=0)
    max_similarity_score = models.FloatField(null=True, blank=True)
    avg_similarity_score = models.FloatField(null=True, blank=True)
    
    class Meta:
        ordering = ['-created_at']
        verbose_name = 'Search Query'
        verbose_name_plural = 'Search Queries'
    
    def __str__(self):
        return f"{self.query_text[:50]}... - {self.created_at}"


class SystemMetrics(models.Model):
    """Store system performance metrics"""
    timestamp = models.DateTimeField(auto_now_add=True)
    metric_name = models.CharField(max_length=100)
    metric_value = models.FloatField()
    metadata = models.JSONField(default=dict, blank=True)
    
    class Meta:
        ordering = ['-timestamp']
        indexes = [
            models.Index(fields=['metric_name', 'timestamp']),
        ]
        verbose_name = 'System Metric'
        verbose_name_plural = 'System Metrics'
    
    def __str__(self):
        return f"{self.metric_name}: {self.metric_value} at {self.timestamp}"


class PropertyDocument(models.Model):
    """Reference to documents in MongoDB for Django integration"""
    document_id = models.CharField(max_length=100, unique=True)
    title = models.CharField(max_length=500, blank=True)
    property_type = models.CharField(max_length=100, blank=True)
    price = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    bedrooms = models.IntegerField(null=True, blank=True)
    accommodates = models.IntegerField(null=True, blank=True)
    location = models.CharField(max_length=200, blank=True)
    amenities = models.JSONField(default=list, blank=True)
    
    # Indexing and search metadata
    indexed_at = models.DateTimeField(auto_now_add=True)
    last_updated = models.DateTimeField(auto_now=True)
    is_active = models.BooleanField(default=True)
    search_vector = models.TextField(blank=True)  # Store search keywords
    
    class Meta:
        ordering = ['-indexed_at']
        indexes = [
            models.Index(fields=['property_type', 'price']),
            models.Index(fields=['bedrooms', 'accommodates']),
            models.Index(fields=['location']),
        ]
        verbose_name = 'Property Document'
        verbose_name_plural = 'Property Documents'
    
    def __str__(self):
        return f"{self.title or self.document_id} - {self.property_type}"
