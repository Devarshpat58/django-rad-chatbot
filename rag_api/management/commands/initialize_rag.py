"""
Django management command to initialize the RAG system
"""

from django.core.management.base import BaseCommand, CommandError
from django.conf import settings
import logging
import sys
import os

from rag_api.services import RAGService
from rag_api.models import SystemMetrics

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = 'Initialize the RAG system and check all components'
    
    def add_arguments(self, parser):
        parser.add_argument(
            '--force',
            action='store_true',
            help='Force reinitialization even if system is already ready',
        )
        parser.add_argument(
            '--check-only',
            action='store_true',
            help='Only check system status without initialization',
        )
        parser.add_argument(
            '--verbose',
            action='store_true',
            help='Verbose output',
        )
    
    def handle(self, *args, **options):
        verbosity = options.get('verbosity', 1)
        verbose = options.get('verbose', False)
        check_only = options.get('check_only', False)
        force = options.get('force', False)
        
        if verbose or verbosity > 1:
            logging.getLogger().setLevel(logging.DEBUG)
        
        self.stdout.write(
            self.style.SUCCESS('Django RAG API System Initialization')
        )
        self.stdout.write('=' * 50)
        
        try:
            # Check Django settings
            self._check_django_settings()
            
            # Get RAG service instance
            self.stdout.write('\n1. Initializing RAG Service...')
            rag_service = RAGService.get_instance()
            
            if check_only:
                self._check_system_status(rag_service)
                return
            
            # Check if system is already ready
            if rag_service.is_ready() and not force:
                self.stdout.write(
                    self.style.SUCCESS('RAG system is already initialized and ready')
                )
                self._display_system_status(rag_service)
                return
            
            # Initialize system
            if force:
                self.stdout.write(
                    self.style.WARNING('Force reinitialization requested')
                )
            
            self._initialize_system(rag_service)
            self._run_system_tests(rag_service)
            self._display_system_status(rag_service)
            
            self.stdout.write(
                self.style.SUCCESS('\nRAG system initialization completed successfully!')
            )
            
        except Exception as e:
            logger.error(f"Initialization failed: {str(e)}")
            self.stdout.write(
                self.style.ERROR(f'Initialization failed: {str(e)}')
            )
            raise CommandError(f'RAG system initialization failed: {str(e)}')
    
    def _check_django_settings(self):
        """Check Django configuration"""
        self.stdout.write('\nChecking Django configuration...')
        
        # Check if RAG_SYSTEM_CONFIG is present
        if not hasattr(settings, 'RAG_SYSTEM_CONFIG'):
            raise CommandError('RAG_SYSTEM_CONFIG not found in Django settings')
        
        self.stdout.write(self.style.SUCCESS('Django settings OK'))
        
        # Check database connection
        from django.db import connection
        try:
            connection.ensure_connection()
            self.stdout.write(self.style.SUCCESS('Database connection OK'))
        except Exception as e:
            raise CommandError(f'Database connection failed: {str(e)}')
    
    def _check_system_status(self, rag_service):
        """Check and display system status only"""
        self.stdout.write('\nChecking system status...')
        
        try:
            status = rag_service.get_system_status()
            health = rag_service.health_check()
            
            self.stdout.write(f"\nSystem Status: {status.get('status', 'unknown')}")
            self.stdout.write(f"Database Connected: {status.get('database_connected', False)}")
            self.stdout.write(f"RAG System Loaded: {status.get('rag_system_loaded', False)}")
            
            if health.get('healthy'):
                self.stdout.write(self.style.SUCCESS('System is healthy'))
            else:
                self.stdout.write(self.style.ERROR('System has issues'))
                for check, result in health.get('checks', {}).items():
                    status_icon = 'OK' if result else 'FAIL'
                    self.stdout.write(f"  {status_icon} {check}: {result}")
            
        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f'Status check failed: {str(e)}')
            )
    
    def _initialize_system(self, rag_service):
        """Initialize the RAG system"""
        self.stdout.write('\nInitializing RAG system components...')
        
        # The RAG service should initialize itself
        if not rag_service.is_ready():
            # Try to reinitialize
            rag_service._initialize_system()
        
        if rag_service.is_ready():
            self.stdout.write(self.style.SUCCESS('RAG system initialized successfully'))
        else:
            raise CommandError('RAG system failed to initialize')
    
    def _run_system_tests(self, rag_service):
        """Run basic system tests"""
        self.stdout.write('\nRunning system tests...')
        
        try:
            # Test basic query processing
            test_query = "test system functionality"
            result = rag_service.process_query(test_query, session_id="test_session")
            
            if result.get('success'):
                execution_time = result.get('metadata', {}).get('execution_time', 0)
                self.stdout.write(
                    self.style.SUCCESS(
                        f'Query processing test passed (took {execution_time:.2f}s)'
                    )
                )
                
                # Store test metric
                SystemMetrics.objects.create(
                    metric_name='initialization_test_query_time',
                    metric_value=execution_time,
                    metadata={'test_query': test_query, 'result': 'success'}
                )
            else:
                self.stdout.write(
                    self.style.WARNING('Query processing test had issues')
                )
            
            # Test health check
            health = rag_service.health_check()
            if health.get('healthy'):
                self.stdout.write(self.style.SUCCESS('Health check passed'))
            else:
                self.stdout.write(self.style.WARNING('Health check found issues'))
            
        except Exception as e:
            self.stdout.write(
                self.style.WARNING(f'System tests failed: {str(e)}')
            )
    
    def _display_system_status(self, rag_service):
        """Display detailed system status"""
        self.stdout.write('\nSystem Status Summary:')
        self.stdout.write('-' * 30)
        
        try:
            status = rag_service.get_system_status()
            
            self.stdout.write(f"Status: {status.get('status', 'unknown')}")
            self.stdout.write(f"Database: {'Connected' if status.get('database_connected') else 'Disconnected'}")
            self.stdout.write(f"RAG System: {'Loaded' if status.get('rag_system_loaded') else 'Not Loaded'}")
            
            # Performance metrics
            perf_metrics = status.get('performance_metrics', {})
            if perf_metrics:
                self.stdout.write(f"Initialization Time: {perf_metrics.get('initialization_time', 'N/A')}s")
                self.stdout.write(f"Uptime: {perf_metrics.get('uptime', 'N/A')}s")
            
            # Index stats
            index_stats = status.get('index_stats', {})
            if index_stats:
                self.stdout.write(f"Index Stats: {index_stats}")
            
        except Exception as e:
            self.stdout.write(
                self.style.WARNING(f'Could not retrieve detailed status: {str(e)}')
            )
