from django.apps import AppConfig


class RagApiConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'rag_api'
    verbose_name = 'RAG API'
    
    def ready(self):
        """Initialize RAG system when Django starts"""
        from .services import RAGService
        # Initialize the RAG service singleton
        RAGService.get_instance()
