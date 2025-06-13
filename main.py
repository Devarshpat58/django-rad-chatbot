
"""
JSON RAG System - Main Application with Windows Stdio Fix
Consolidated version with panic prevention and full functionality
"""

import os
import sys
import gradio as gr
import socket

def find_free_port(start_port=7860, max_attempts=50):
    """Find a free port starting from start_port."""
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', port))
                return port
        except OSError:
            continue
    return None
import logging
import threading
import time
from datetime import datetime
from typing import List, Tuple, Optional

# === ENCODING AND ENVIRONMENT SETUP ===
# Set environment for stable operation without Unicode issues
os.environ['PYTHONIOENCODING'] = 'ascii'
os.environ['PYTHONUTF8'] = '0'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['LANG'] = 'C'

# Configure stdio for ASCII output to prevent encoding issues
if sys.platform.startswith('win'):
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='ascii', errors='replace')
        sys.stderr.reconfigure(encoding='ascii', errors='replace')
    else:
        import codecs
        sys.stdout = codecs.getwriter('ascii')(sys.stdout.buffer, errors='replace')
        sys.stderr = codecs.getwriter('ascii')(sys.stderr.buffer, errors='replace')

from config import Config
from core_system import JSONRAGSystem
from query_processor import QueryProcessor
from consolidated_config import NumericConfig

# Set up logging (commented out - using enhanced logging below)
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.FileHandler(Config.get_log_file_path('main')),
#         logging.StreamHandler()
#     ]
# )

# Enhanced logging setup with comprehensive debugging capabilities
from logging_config import setup_logging, StructuredLogger, LogOperation, log_performance

# Setup comprehensive logging system
logging_config = setup_logging(
    log_level='INFO',  # Use INFO for production, DEBUG for development
    log_dir='logs',
    console_output=True,
    file_output=True,
    format_style='detailed',
    max_file_size=50 * 1024 * 1024,  # 50MB log files
    backup_count=10  # Keep 10 backup files
)

# Create structured logger for main module
logger = StructuredLogger(__name__, {'component': 'main_app', 'source_module': 'gradio_interface'})

# Log system startup
logger.info("JSON RAG System starting up", 
           log_config=str(logging_config), 
           python_version=sys.version.split()[0],
           platform=sys.platform)

# Global system instance - loaded from pre-built setup
rag_system = None
system_status = {
    'initialized': False,
    'initialization_progress': 'Loading pre-built system...',
    'error_message': '',
    'start_time': None
}

def load_initialized_system():
    """Load pre-initialized RAG system from setup"""
    global rag_system, system_status
    
    try:
        system_status['initialization_progress'] = 'Loading pre-built system components...'
        system_status['start_time'] = datetime.now()
        
        logger.info("Loading pre-initialized RAG system", 
                   extra={'operation': 'rag_system_init', 'timestamp': datetime.now().isoformat()})
        
        # Create system instance and load existing components
        logger.info("Creating JSONRAGSystem instance", 
                   extra={'operation': 'create_rag_system'})
        rag_system = JSONRAGSystem()
        logger.debug("JSONRAGSystem instance created successfully")
        
        # Quick check if system was properly set up
        if not Config.FAISS_INDEX_PATH.exists() or not Config.PROCESSED_DOCS_PATH.exists():
            system_status['error_message'] = ('System not initialized. Please run: python setup.py --full-setup')
            logger.error("System components not found. Run setup.py first.")
            return
        
        system_status['initialization_progress'] = 'Loading indexes and saved vocabulary...'
        
        # Load pre-built components quickly
        success = rag_system.initialize_system()
        
        if success:
            system_status['initialized'] = True
            system_status['initialization_progress'] = 'System loaded successfully'
            logger.info("Pre-built RAG system loaded successfully")
        else:
            system_status['error_message'] = 'Failed to load system components'
            logger.error("Failed to load pre-built RAG system")
            
    except Exception as e:
        system_status['error_message'] = f'Loading error: {str(e)}'
        logger.error(f"Error loading system: {e}", 
                    extra={'error_type': type(e).__name__, 'error_details': str(e)})
        
def chat_interface(message: str, history: List[Tuple[str, str]]) -> Tuple[str, List[Tuple[str, str]]]:
    """Main chat interface function"""
    global rag_system, system_status
    
    if not system_status['initialized']:
        if system_status['error_message']:
            error_response = f"System initialization failed: {system_status['error_message']}. Please check the logs and restart."
        else:
            error_response = f"System is still initializing... Status: {system_status['initialization_progress']}"
        
        history.append((message, error_response))
        return "", history
    
    try:
        # Process the query
        response = rag_system.process_query(message, session_id="gradio_session")
        
        # Add to history
        history.append((message, response))
        
        return "", history
        
    except Exception as e:
        error_response = f"Error processing your request: {str(e)}"
        logger.error(f"Chat interface error: {e}", 
                    extra={'message': message, 'error_type': type(e).__name__})
        history.append((message, error_response))
        return "", history

def clear_chat() -> List[Tuple[str, str]]:
    """Clear chat history"""
    return []

def get_system_info() -> str:
    """Get current system status information"""
    global rag_system, system_status
    
    info_parts = []
    
    # Basic system status with setup guidance
    status_text = 'Loaded' if system_status['initialized'] else 'Not Loaded'
    info_parts.append(f"**System Status:** {status_text}")
    info_parts.append(f"**Load Progress:** {system_status['initialization_progress']}")
    
    if system_status['start_time']:
        elapsed = datetime.now() - system_status['start_time']
        info_parts.append(f"**Load Time:** {elapsed.total_seconds():.2f} seconds")
    
    if system_status['error_message']:
        info_parts.append(f"**Error:** {system_status['error_message']}")
        info_parts.append("**Solution:** Run 'python setup.py --full-setup' to initialize the system")
    
    # Detailed system status if initialized
    if system_status['initialized'] and rag_system:
        try:
            status = rag_system.get_system_status()
            
            info_parts.append("\n**Detailed System Information:**")
            
            # Database status
            db_status = "Connected" if status['database_connected'] else "Disconnected"
            info_parts.append(f"• **Database:** {db_status}")
            
            # Index statistics
            index_stats = status['index_stats']
            info_parts.append(f"• **FAISS Index Size:** {index_stats['faiss_index_size']} documents")
            info_parts.append(f"• **Processed Documents:** {index_stats['processed_documents_count']}")
            info_parts.append(f"• **Embedding Cache Size:** {index_stats['embedding_cache_size']}")
            
            # Enhanced vocabulary and NLP statistics
            if hasattr(rag_system, 'vocabulary_manager') and rag_system.vocabulary_manager:
                vocab_size = len(rag_system.vocabulary_manager.vocabulary)
                mappings_size = len(rag_system.vocabulary_manager.keyword_mappings)
                info_parts.append(f"• **Vocabulary Size:** {vocab_size} terms")
                info_parts.append(f"• **Keyword Mappings:** {mappings_size} mappings")
            
            # Query processor capabilities
            try:
                test_processor = QueryProcessor()
                info_parts.append(f"• **Query Processor:** Available")
                if hasattr(NumericConfig, 'NUMERIC_KEYWORDS'):
                    info_parts.append(f"• **Numeric Categories:** {len(NumericConfig.NUMERIC_KEYWORDS)}")
                if hasattr(NumericConfig, 'RANGE_OPERATORS'):
                    info_parts.append(f"• **Range Patterns:** {len(NumericConfig.RANGE_OPERATORS)}")
                    
                # Test advanced processing
                test_result = test_processor.process_query("find 2 bedroom apartments under 100")
                if test_result and hasattr(test_result, 'numeric_constraints') and test_result.numeric_constraints:
                    info_parts.append(f"• **Advanced Processing:** Enabled")
                else:
                    info_parts.append(f"• **Advanced Processing:** Basic Mode")
            except Exception:
                info_parts.append(f"• **Query Processor:** Basic Fallback")
            
            # Session statistics
            session_stats = status['session_stats']
            info_parts.append(f"• **Active Sessions:** {session_stats['active_sessions']}")
            info_parts.append(f"• **Total Conversations:** {session_stats['total_conversations']}")
            info_parts.append(f"• **Avg Turns per Session:** {session_stats['average_turns_per_session']:.1f}")
            
        except Exception as e:
            info_parts.append(f"\n**Error getting detailed status:** {str(e)}")
    
    # Configuration information
    info_parts.append("\n**Configuration:**")
    info_parts.append(f"• **Database:** {Config.DATABASE_NAME}.{Config.COLLECTION_NAME}")
    info_parts.append(f"• **Embedding Model:** {Config.EMBEDDING_MODEL}")
    info_parts.append(f"• **Max Results:** {Config.TOP_K_RESULTS}")
    
    return "\n".join(info_parts)

def load_system_interface() -> str:
    """Manual system loading trigger"""
    global system_status
    
    if system_status['initialized']:
        return "System is already loaded."
    
    if 'Loading' in system_status['initialization_progress']:
        return "System loading is already in progress."
    
    # Start loading in background
    load_thread = threading.Thread(target=load_initialized_system)
    load_thread.daemon = True
    load_thread.start()
    
    return "System loading started. If this fails, run 'python setup.py --full-setup' first."

def create_gradio_interface():
    """Create and configure the Gradio interface"""
    
    # Custom CSS for better appearance
    css = """
    .gradio-container {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        max-width: 1200px;
        margin: 0 auto;
    }
    
    .status-info {
        background-color: #e9ecef;
        padding: 15px;
        border-radius: 5px;
        font-family: monospace;
    }
    
    .dark .status-info {
        background-color: #2d3748;
        color: #e2e8f0;
    }
    """
    
    # Example queries for user guidance showcasing advanced capabilities
    example_queries = [
        "Find two bedroom apartments under 150 dollars",
        "Show me luxury houses with WiFi and parking",
        "Properties with good ratings in downtown for families",
        "Budget accommodation that accommodates four people",
        "Studio apartments between 50 and 100 with kitchen access",
        "Family-friendly houses with pools near city center",
        "Premium properties under 300 for business travelers",
        "Pet-friendly apartments with accessible features"
    ]
    
    with gr.Blocks(css=css, theme=gr.themes.Soft(), title="JSON RAG System") as interface:
        gr.Markdown(
            """
            # JSON RAG System - Advanced Property Search
            
            Welcome to the intelligent property search system with advanced AI capabilities! Ask questions about Airbnb properties using natural language with sophisticated numeric processing and contextual understanding.
            
            **Advanced Query Examples:**
            - "Find two bedroom apartments under 150 dollars with WiFi"
            - "Show me luxury places that accommodate four people with parking"
            - "Budget-friendly properties between 50 and 100 for families"
            - "Studio apartments with premium amenities near downtown"
            """
        )
        
        with gr.Tab("Chat Interface"):
            chatbot = gr.Chatbot(
                height=500,
                label="Property Search Assistant"
            )
            
            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Ask about properties using natural language... (e.g., 'Find two bedroom apartments under 150 dollars')",
                    label="Your Question",
                    lines=2,
                    scale=4
                )
                submit_btn = gr.Button("Search", variant="primary", scale=1)
            
            with gr.Row():
                clear_btn = gr.Button("Clear Chat", variant="secondary")
                
            # Example queries
            gr.Markdown("**Quick Examples:**")
            with gr.Row():
                for i in range(0, len(example_queries), 2):
                    with gr.Column():
                        for j in range(i, min(i+2, len(example_queries))):
                            example_btn = gr.Button(
                                example_queries[j], 
                                variant="outline",
                                size="sm"
                            )
                            example_btn.click(
                                lambda x=example_queries[j]: x,
                                outputs=msg
                            )
        
        with gr.Tab("System Status"):
            with gr.Row():
                status_btn = gr.Button("Refresh Status", variant="primary")
                load_btn = gr.Button("Load System", variant="secondary")
            
            status_output = gr.Markdown(
                label="System Information",
                value="Click 'Refresh Status' to see current system information. If system is not loaded, run setup.py first."
            )
            
            load_output = gr.Textbox(
                label="Load Output",
                interactive=False
            )
        
        with gr.Tab("Help & Info"):
            gr.Markdown(
                """
                ## How to Use the JSON RAG System
                
                ### Search Capabilities
                - **Semantic Search**: AI-powered understanding of meaning using MiniLM embeddings
                - **Advanced Query Processing**: Intent detection and entity extraction
                - **Intelligent Numeric Filtering**: Natural language to constraint conversion
                - **Contextual Information**: Price categories, property type implications
                - **Vocabulary-Based Search**: MongoDB-derived keyword mappings
                - **Fuzzy Matching**: Intelligent handling of typos and variations
                - **Hybrid Search Strategy**: Multiple search methods with weighted scoring
                
                ### Query Examples
                
                **Basic Search:**
                - "Find apartments in downtown"
                - "Show me houses with pools"
                
                **Advanced Numeric Queries:**
                - "Find three bedroom apartments under 200 dollars"
                - "Properties between 100 and 150 for a family of five"
                - "Studio apartments under one hundred for two people"
                
                **Context-Aware Queries:**
                - "Budget accommodation for a couple with parking"
                - "Luxury properties with premium amenities downtown"
                - "Family-friendly houses with multiple bedrooms"
                
                **Complex Requirements:**
                - "2 bedroom places under $100 per night with WiFi"
                - "Properties accommodating 4 guests with parking and kitchen"
                - "Highly rated homes with pools near city center"
                
                ### System Requirements
                - MongoDB database with Airbnb property data
                - Run 'python setup.py --full-setup' before first use
                - Internet connection for AI model downloads (during setup)
                
                ### Troubleshooting
                - If system shows "Not Loaded", run 'python setup.py --full-setup' first
                - Then click "Load System" to load pre-built components
                - Check System Status tab for detailed information
                """
            )
        
        # Event handlers
        submit_btn.click(
            chat_interface,
            inputs=[msg, chatbot],
            outputs=[msg, chatbot]
        )
        
        msg.submit(
            chat_interface,
            inputs=[msg, chatbot],
            outputs=[msg, chatbot]
        )
        
        clear_btn.click(
            clear_chat,
            outputs=chatbot
        )
        
        status_btn.click(
            get_system_info,
            outputs=status_output
        )
        
        load_btn.click(
            load_system_interface,
            outputs=load_output
        )
    
    return interface

def main():
    """Main function to start the application"""
    logger.info("Starting JSON RAG System Web Interface", 
               extra={'component': 'main', 'operation': 'startup'})
    
    # Ensure configuration directories exist
    Config.ensure_directories()
    
    # Quick check if system was set up
    if not Config.FAISS_INDEX_PATH.exists() or not Config.PROCESSED_DOCS_PATH.exists():
        print("\n" + "="*60)
        print("SYSTEM NOT SET UP")
        print("="*60)
        print("The system has not been initialized yet.")
        print("Please run the setup first:\n")
        print("  python setup.py --full-setup")
        print("\nThis will:")
        print("  - Build and save vocabulary from your MongoDB data")
        print("  - Generate embeddings and create indexes")
        print("  - Optimize search for Airbnb properties")
        print("\nAfter setup completes, run this again.")
        print("="*60)
        return
    
    # Start system loading in background
    logger.info("Starting background system loading", 
               extra={'component': 'main', 'operation': 'background_loading'})
    load_thread = threading.Thread(target=load_initialized_system)
    load_thread.daemon = True
    load_thread.start()
    
    # Create and launch interface
    interface = create_gradio_interface()
    
    # Use environment port if set, otherwise default
    port = int(os.environ.get('GRADIO_SERVER_PORT', Config.GRADIO_PORT))
    logger.info(f"Launching web interface on {Config.GRADIO_HOST}:{port}", 
               extra={'host': Config.GRADIO_HOST, 'port': port})
    print(f"\nStarting web interface at: http://{Config.GRADIO_HOST}:{port}")
    print("Loading system components in background...")
    
    try:
        # Try the preferred port first, then find any available port
        try:
            interface.launch(
                server_name=Config.GRADIO_HOST,
                server_port=port,
                share=False,
                show_error=True,
                quiet=False
            )
        except Exception as port_error:
            logger.warning(f"Port {port} not available: {port_error}", 
                          extra={'original_port': port, 'error': str(port_error)})
            print(f"Port {port} is in use, trying to find an available port...")
            # Find an available port automatically
            available_port = find_free_port(port + 1)
            if available_port:
                print(f"Found available port: {available_port}")
                interface.launch(
                    server_name=Config.GRADIO_HOST,
                    server_port=available_port,
                    share=False,
                    show_error=True,
                    quiet=False
                )
            else:
                raise Exception("No available ports found in range")
    except Exception as e:
        logger.error(f"Error launching interface: {e}", 
                    extra={'error_type': type(e).__name__, 'error_details': str(e)})
        print(f"Failed to start web interface: {e}")
        print("Please check the configuration and try again.")

if __name__ == "__main__":
    main()