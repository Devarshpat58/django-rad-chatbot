from rest_framework import serializers
from .models import SearchSession, SearchQuery, SystemMetrics, PropertyDocument


class SearchQuerySerializer(serializers.ModelSerializer):
    """Serializer for search query requests"""
    
    class Meta:
        model = SearchQuery
        fields = ['id', 'query_text', 'processed_query', 'response', 'response_metadata', 
                 'execution_time', 'num_results', 'max_similarity_score', 
                 'avg_similarity_score', 'created_at']
        read_only_fields = ['id', 'processed_query', 'response', 'response_metadata', 
                           'execution_time', 'num_results', 'max_similarity_score', 
                           'avg_similarity_score', 'created_at']


class SearchRequestSerializer(serializers.Serializer):
    """Serializer for incoming search requests"""
    query = serializers.CharField(max_length=1000, help_text="The search query text")
    session_id = serializers.CharField(max_length=40, required=False, 
                                      help_text="Optional session ID for conversation continuity")
    max_results = serializers.IntegerField(default=5, min_value=1, max_value=20,
                                          help_text="Maximum number of results to return")
    include_metadata = serializers.BooleanField(default=True, 
                                               help_text="Include search metadata in response")
    filters = serializers.DictField(required=False, default=dict,
                                   help_text="Additional filters to apply")


class SearchResponseSerializer(serializers.Serializer):
    """Serializer for search response"""
    query_id = serializers.UUIDField(read_only=True)
    session_id = serializers.CharField(read_only=True)
    query_text = serializers.CharField(read_only=True)
    response = serializers.CharField(read_only=True)
    results = serializers.ListField(read_only=True)
    metadata = serializers.DictField(read_only=True)
    execution_time = serializers.FloatField(read_only=True)
    timestamp = serializers.DateTimeField(read_only=True)


class PropertyDocumentSerializer(serializers.ModelSerializer):
    """Serializer for property documents"""
    
    class Meta:
        model = PropertyDocument
        fields = ['document_id', 'title', 'property_type', 'price', 'bedrooms', 
                 'accommodates', 'location', 'amenities', 'indexed_at', 
                 'last_updated', 'is_active']
        read_only_fields = ['indexed_at', 'last_updated']


class PropertySearchFiltersSerializer(serializers.Serializer):
    """Serializer for property search filters"""
    property_type = serializers.CharField(required=False)
    min_price = serializers.DecimalField(max_digits=10, decimal_places=2, required=False)
    max_price = serializers.DecimalField(max_digits=10, decimal_places=2, required=False)
    min_bedrooms = serializers.IntegerField(required=False)
    max_bedrooms = serializers.IntegerField(required=False)
    min_accommodates = serializers.IntegerField(required=False)
    max_accommodates = serializers.IntegerField(required=False)
    location = serializers.CharField(required=False)
    amenities = serializers.ListField(child=serializers.CharField(), required=False)


class SearchSessionSerializer(serializers.ModelSerializer):
    """Serializer for search sessions"""
    queries_count = serializers.SerializerMethodField()
    last_query = serializers.SerializerMethodField()
    
    class Meta:
        model = SearchSession
        fields = ['id', 'session_key', 'created_at', 'updated_at', 'is_active', 
                 'queries_count', 'last_query']
        read_only_fields = ['id', 'created_at', 'updated_at']
    
    def get_queries_count(self, obj):
        return obj.queries.count()
    
    def get_last_query(self, obj):
        last_query = obj.queries.first()
        if last_query:
            return {
                'query_text': last_query.query_text,
                'created_at': last_query.created_at
            }
        return None


class SystemStatusSerializer(serializers.Serializer):
    """Serializer for system status information"""
    status = serializers.CharField(read_only=True)
    database_connected = serializers.BooleanField(read_only=True)
    rag_system_loaded = serializers.BooleanField(read_only=True)
    index_stats = serializers.DictField(read_only=True)
    session_stats = serializers.DictField(read_only=True)
    performance_metrics = serializers.DictField(read_only=True)
    last_updated = serializers.DateTimeField(read_only=True)


class SystemMetricsSerializer(serializers.ModelSerializer):
    """Serializer for system metrics"""
    
    class Meta:
        model = SystemMetrics
        fields = ['timestamp', 'metric_name', 'metric_value', 'metadata']
        read_only_fields = ['timestamp']


class ChatMessageSerializer(serializers.Serializer):
    """Serializer for chat-style interactions"""
    message = serializers.CharField(max_length=2000, help_text="The user's message")
    conversation_id = serializers.CharField(max_length=40, required=False,
                                          help_text="Conversation ID for context")


class ChatResponseSerializer(serializers.Serializer):
    """Serializer for chat response"""
    response = serializers.CharField(read_only=True)
    conversation_id = serializers.CharField(read_only=True)
    message_id = serializers.UUIDField(read_only=True)
    timestamp = serializers.DateTimeField(read_only=True)
    sources = serializers.ListField(read_only=True)
    confidence = serializers.FloatField(read_only=True)


class BulkSearchSerializer(serializers.Serializer):
    """Serializer for bulk search operations"""
    queries = serializers.ListField(
        child=serializers.CharField(max_length=1000),
        max_length=10,
        help_text="List of queries to process (max 10)"
    )
    session_id = serializers.CharField(max_length=40, required=False)
    include_metadata = serializers.BooleanField(default=False)


class BulkSearchResponseSerializer(serializers.Serializer):
    """Serializer for bulk search response"""
    results = serializers.ListField(read_only=True)
    session_id = serializers.CharField(read_only=True)
    total_queries = serializers.IntegerField(read_only=True)
    successful_queries = serializers.IntegerField(read_only=True)
    failed_queries = serializers.IntegerField(read_only=True)
    total_execution_time = serializers.FloatField(read_only=True)
    timestamp = serializers.DateTimeField(read_only=True)
