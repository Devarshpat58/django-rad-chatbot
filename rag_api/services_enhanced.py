"""
Enhanced RAG Service Integration for Django REST Framework
Includes mandatory property fields, improved formatting, and table comparison support
"""

import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Any

# Import context manager for multi-turn conversations
from context_manager import ContextManager, create_context_manager
from utils import SessionManager
from django.conf import settings
from django.core.cache import cache
from django.utils import timezone

# Import existing RAG system components
try:
    from core_system import JSONRAGSystem
    from query_processor import QueryProcessor
    from config import Config
except ImportError as e:
    logging.error(f"Failed to import RAG system components: {e}")
    JSONRAGSystem = None
    QueryProcessor = None
    Config = None

logger = logging.getLogger(__name__)


class EnhancedRAGService:
    """
    Enhanced service layer with mandatory fields and improved formatting
    """
    
    _instance = None
    _initialized = False
    _context_manager = None
    _preprocessor = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            try:
                self.rag_system = None
                self.query_processor = None
                self._status = {
                    'initialized': False,
                    'database_connected': False,
                    'last_error': None,
                    'initialization_time': None
                }
                self._initialize_system()
                self._initialize_preprocessor()
                
                # Initialize context manager for multi-turn conversations
                try:
                    session_manager = SessionManager() if 'SessionManager' in globals() else None
                    self._context_manager = create_context_manager(session_manager)
                    logger.info("Context manager initialized successfully")
                except Exception as e:
                    logger.warning(f"Context manager initialization failed: {e}")
                    self._context_manager = None
                
                EnhancedRAGService._initialized = True
                logger.info("Enhanced RAG Service initialized successfully")
                
            except Exception as e:
                logger.error(f"Failed to initialize Enhanced RAG Service: {e}")
                self._status['last_error'] = str(e)
    
    def _initialize_preprocessor(self):
        """Initialize the data preprocessor for fast field extraction"""
        try:
            from preprocessor import DataPreprocessor
            self.__class__._preprocessor = DataPreprocessor()
            
            # Try to load existing preprocessed data
            if self._preprocessor.load_preprocessed_data():
                logger.info("Loaded preprocessed metadata for fast field extraction")
            else:
                logger.info("No preprocessed data found - preprocessing will be triggered on first query with missing data")
                # Don't trigger automatic background preprocessing to avoid threading issues
                # Users can manually run: python management_commands.py preprocess
                
        except Exception as e:
            logger.warning(f"Could not initialize preprocessor: {e}")
            self.__class__._preprocessor = None
    
    def _preprocess_data_async(self):
        """Preprocess data in background thread with error handling"""
        try:
            import threading
            
            def preprocess():
                try:
                    logger.info("Starting background data preprocessing")
                    from preprocessor import preprocess_from_mongodb
                    success = preprocess_from_mongodb()
                    
                    if success and self._preprocessor:
                        # Safely reload preprocessed data
                        try:
                            self._preprocessor.load_preprocessed_data()
                            logger.info("Background preprocessing completed successfully")
                        except Exception as reload_error:
                            logger.warning(f"Could not reload preprocessed data: {reload_error}")
                    else:
                        logger.warning("Background preprocessing failed")
                        
                except Exception as e:
                    logger.error(f"Error in background preprocessing: {e}", exc_info=True)
            
            thread = threading.Thread(target=preprocess, daemon=True, name="preprocessor-thread")
            thread.start()
            logger.info("Background preprocessing thread started")
            
        except Exception as e:
            logger.error(f"Could not start background preprocessing thread: {e}")
            # If threading fails, skip background processing
            logger.info("Skipping background preprocessing due to threading error")
    
    @classmethod
    def get_instance(cls):
        """Get singleton instance"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def _initialize_system(self):
        """Initialize the RAG system components"""
        try:
            start_time = time.time()
            logger.info("Initializing Enhanced RAG system...")
            
            if not JSONRAGSystem:
                raise ImportError("RAG system components not available")
            
            # Initialize RAG system
            self.rag_system = JSONRAGSystem()
            success = self.rag_system.initialize_system()
            
            if success:
                self.query_processor = QueryProcessor()
                self._status.update({
                    'initialized': True,
                    'database_connected': True,
                    'initialization_time': time.time() - start_time
                })
                logger.info(f"Enhanced RAG system initialized successfully in {self._status['initialization_time']:.2f}s")
            else:
                raise Exception("RAG system initialization failed")
                
        except Exception as e:
            error_msg = f"Failed to initialize Enhanced RAG system: {str(e)}"
            logger.error(error_msg)
            self._status.update({
                'initialized': False,
                'database_connected': False,
                'last_error': error_msg
            })
    
    def is_ready(self) -> bool:
        """Check if the RAG system is ready for queries"""
        return self._status['initialized'] and self.rag_system is not None
    
    def process_query(self, query_text: str, session_id: str = None, 
                     max_results: int = 5, filters: Dict = None, 
                     enable_comparison: bool = False) -> Dict[str, Any]:
        """
        Enhanced query processing with mandatory fields and comparison support
        
        Args:
            query_text: The search query
            session_id: Optional session identifier
            max_results: Maximum number of results
            filters: Additional filters to apply
            enable_comparison: Whether to format results for comparison
        
        Returns:
            Dictionary containing response and enhanced metadata
        """
        if not self.is_ready():
            raise Exception("Enhanced RAG system not initialized")
        
        start_time = time.time()
        
        # Check if context manager is available and process with context awareness
        if self._context_manager and session_id:
            try:
                # First process with existing RAG system
                response = self.rag_system.process_query(
                    query_text, 
                    session_id=session_id or "enhanced_api_session"
                )
                
                # Extract results from response metadata
                raw_results = []
                if 'metadata' in response and 'detailed_results' in response['metadata']:
                    raw_results = response['metadata']['detailed_results']
                elif 'metadata' in response and 'results' in response['metadata']:
                    raw_results = response['metadata']['results']
                
                # Validate and ensure all results are properly formatted
                if not isinstance(raw_results, list):
                    logger.warning(f"raw_results is not a list: {type(raw_results)}, converting to list")
                    raw_results = [raw_results] if raw_results else []
                
                # Ensure each result has proper source_data field
                for result in raw_results:
                    if isinstance(result, dict) and 'source_data' not in result:
                        result['source_data'] = result.get('document', result.get('original_document', result))
                
                # Process with context manager for multi-turn awareness
                context_response = self._context_manager.process_query(
                    session_id=session_id,
                    query=query_text,
                    results=raw_results,
                    metadata=response.get('metadata', {})
                )
                
                # If context manager provides enhanced response, use it
                if context_response and isinstance(context_response, dict) and 'results' in context_response:
                    return self._format_context_aware_response(
                        query_text, context_response, response, start_time
                    )
                
            except Exception as e:
                logger.warning(f"Context-aware processing failed, falling back to standard processing: {e}")
        
        try:
            # Use existing RAG system to process query
            response = self.rag_system.process_query(
                query_text, 
                session_id=session_id or "enhanced_api_session"
            )
            
            # Extract enhanced search results with full JSON data and AI summaries
            search_results = []
            query_analysis = None
            
            try:
                if hasattr(self.rag_system, 'semantic_search') and self.rag_system.semantic_search:
                    # Get query analysis for enhanced processing
                    from query_processor import QueryProcessor
                    query_processor = QueryProcessor()
                    query_analysis = query_processor.process_query(query_text)
                    
                    # Handle ProcessedQuery object
                    cleaned_query = query_text
                    if hasattr(query_analysis, 'cleaned_query'):
                        cleaned_query = query_analysis.cleaned_query
                    elif isinstance(query_analysis, dict) and 'cleaned_query' in query_analysis:
                        cleaned_query = query_analysis['cleaned_query']
                    
                    raw_results = self.rag_system.semantic_search.hybrid_search(
                        query=cleaned_query,
                        k=max_results or 10
                    )
                    
                    # Convert to enhanced API format with full JSON and AI summaries
                    for i, (doc, score) in enumerate(raw_results[:max_results] if max_results else raw_results):
                        if isinstance(doc, dict):
                            # Get the original source document from MongoDB
                            original_doc = doc.get('original_document', doc)
                            
                            # Create a clean copy of the full document
                            clean_doc = self._clean_document_for_json(original_doc)
                            
                            # Ensure source_data is properly populated
                            source_data = clean_doc if clean_doc else original_doc
                            
                            # Extract mandatory and query-relevant fields with preprocessor optimization
                            # Get document ID for fast extraction (use _id, id, or generate from index)
                            doc_id = original_doc.get('_id') or original_doc.get('id') or f'doc_{i}'
                            mandatory_fields = self._extract_mandatory_fields_fast(original_doc, str(doc_id))
                            query_relevant_fields = self._extract_query_relevant_fields(
                                original_doc, query_text, query_analysis
                            )
                            
                            # Merge mandatory and query-relevant fields
                            all_relevant_fields = {**mandatory_fields, **query_relevant_fields}
                            
                            # Generate lightweight summary (faster processing)
                            ai_summary = self._generate_fast_summary(
                                original_doc, query_text
                            )
                            
                            result_item = {
                                'id': i + 1,
                                'score': float(score) if score else 0.0,
                                'relevance': 'high' if score > 0.8 else 'medium' if score > 0.6 else 'low',
                                # Complete source JSON data - enhanced formatting
                                'source_data': source_data,
                                'source_json': self._format_json_fast(clean_doc),
                                # Mandatory fields (always shown)
                                'mandatory_fields': mandatory_fields,
                                # Query-relevant fields
                                'query_relevant_fields': all_relevant_fields,
                                # AI-generated summary (200+ words)
                                'ai_summary': ai_summary,
                                # Comparison-ready format (fast version)
                                'comparison_data': self._prepare_comparison_data_fast(original_doc),
                                # Legacy support
                                'document': clean_doc
                            }
                            search_results.append(result_item)
            except Exception as e:
                logger.warning(f"Could not extract enhanced search results: {e}")
            
            execution_time = time.time() - start_time
            
            # Enhanced metadata
            metadata = {
                'execution_time': execution_time,
                'query_processed': True,
                'session_id': session_id,
                'timestamp': timezone.now().isoformat(),
                'results': search_results,
                'num_results': len(search_results),
                'max_similarity_score': max([r['score'] for r in search_results], default=0.0),
                'avg_similarity_score': sum([r['score'] for r in search_results]) / len(search_results) if search_results else 0.0,
                'comparison_enabled': enable_comparison,
                'has_mandatory_fields': len([r for r in search_results if r.get('mandatory_fields')]) > 0
            }
            
            # Add comparison metadata if enabled
            if enable_comparison and len(search_results) > 1:
                metadata['comparison_summary'] = self._generate_comparison_summary(search_results)
            
            return {
                'response': response,
                'metadata': metadata,
                'success': True
            }
            
        except Exception as e:
            error_msg = f"Enhanced query processing error: {str(e)}"
            logger.error(error_msg)
            
            return {
                'response': f"Sorry, I encountered an error processing your query: {str(e)}",
                'metadata': {
                    'execution_time': time.time() - start_time,
                    'query_processed': False,
                    'error': error_msg,
                    'session_id': session_id,
                    'timestamp': timezone.now().isoformat()
                },
                'success': False
            }
    
    def _extract_mandatory_fields(self, document: Dict[str, Any]) -> Dict[str, str]:
        """Extract mandatory fields that should always be displayed"""
        mandatory_fields = {}
        
        # Enhanced mandatory field mappings with more property-specific fields
        field_mappings = {
            # Core Property Info (Always shown)
            'name': ['name', 'title', 'listing_name', 'property_name', 'property_title'],
            'price': ['price', 'nightly_rate', 'rate', 'cost', 'daily_rate'],
            'location': ['neighbourhood_cleansed', 'neighbourhood', 'neighborhood', 'city', 'location', 'address', 'area', 'district'],
            'neighbourhood': ['neighbourhood_cleansed', 'neighbourhood', 'neighborhood', 'area', 'district', 'region'],
            
            # Accommodation Details
            'bedrooms': ['bedrooms', 'bedroom_count', 'bed_rooms', 'num_bedrooms'],
            'beds': ['beds', 'bed_count', 'total_beds', 'num_beds'],
            'bathrooms': ['bathrooms', 'bathroom_count', 'bath_rooms', 'num_bathrooms'],
            'accommodates': ['accommodates', 'guests', 'max_guests', 'capacity', 'sleeps'],
            
            # Property Classification
            'property_type': ['property_type', 'type', 'category', 'listing_type'],
            'room_type': ['room_type', 'room_category', 'space_type'],
            
            # Host Information
            'host_name': ['host_name', 'host', 'owner_name', 'landlord'],
            'host_response_time': ['host_response_time', 'response_time'],
            'host_is_superhost': ['host_is_superhost', 'superhost', 'is_superhost'],
            
            # Booking Details
            'instant_bookable': ['instant_bookable', 'instant_book', 'quick_book'],
            'minimum_nights': ['minimum_nights', 'min_nights', 'min_stay'],
            'maximum_nights': ['maximum_nights', 'max_nights', 'max_stay'],
            
            # Reviews & Ratings
            'review_score': ['review_scores_rating', 'rating', 'overall_rating', 'score'],
            'number_of_reviews': ['number_of_reviews', 'review_count', 'total_reviews'],
            'reviews_per_month': ['reviews_per_month', 'monthly_reviews'],
            
            # Availability
            'availability_365': ['availability_365', 'yearly_availability'],
            'availability_30': ['availability_30', 'monthly_availability'],
            
            # Additional Features
            'cancellation_policy': ['cancellation_policy', 'cancel_policy'],
            'cleaning_fee': ['cleaning_fee', 'cleaning_cost'],
            'security_deposit': ['security_deposit', 'deposit']
        }
        
        # Extract each mandatory field with enhanced formatting
        for display_name, field_options in field_mappings.items():
            for field in field_options:
                if field in document and document[field] is not None:
                    value = document[field]
                    
                    # Enhanced formatting using specialized helper methods
                    if display_name == 'price':
                        mandatory_fields[display_name] = self._format_price_field(document)
                        break  # Use specialized method instead of continuing loop
                    elif display_name == 'location':
                        mandatory_fields[display_name] = self._format_location_field(document)
                        break
                    elif display_name == 'review_score':
                        mandatory_fields[display_name] = self._format_rating_field(document)
                        break
                    elif display_name == 'number_of_reviews':
                        mandatory_fields[display_name] = self._format_reviews_field(document)
                        break
                    elif display_name == 'host_name':
                        mandatory_fields[display_name] = self._format_host_status(document)
                        break
                    elif isinstance(value, (int, float)):
                        if display_name in ['bedrooms', 'beds', 'bathrooms', 'accommodates', 'availability_365', 'availability_30']:
                            mandatory_fields[display_name] = str(int(value)) if value == int(value) else f'{value:.1f}'
                        elif display_name in ['minimum_nights', 'maximum_nights']:
                            nights = int(value) if value == int(value) else value
                            mandatory_fields[display_name] = f'{nights} night{"s" if nights != 1 else ""}'
                        elif display_name in ['cleaning_fee', 'security_deposit']:
                            mandatory_fields[display_name] = f'${value:,.0f}' if value >= 1 else f'${value:.2f}'
                        elif display_name == 'reviews_per_month':
                            mandatory_fields[display_name] = f'{value:.1f}/month'
                        else:
                            mandatory_fields[display_name] = str(value)
                    elif isinstance(value, bool):
                        mandatory_fields[display_name] = 'Yes' if value else 'No'
                    elif isinstance(value, str):
                        # Handle boolean-like strings and special cases
                        if display_name in ['instant_bookable', 'host_is_superhost']:
                            if value.lower() in ['t', 'true', 'yes', '1']:
                                mandatory_fields[display_name] = 'Yes'
                            elif value.lower() in ['f', 'false', 'no', '0']:
                                mandatory_fields[display_name] = 'No'
                            else:
                                mandatory_fields[display_name] = value
                        elif display_name == 'cancellation_policy':
                            mandatory_fields[display_name] = value.replace('_', ' ').title()
                        elif display_name == 'host_response_time':
                            mandatory_fields[display_name] = value.replace('_', ' ').title()
                        else:
                            str_value = str(value).strip()
                            # Truncate long text for specific fields
                            if display_name in ['neighbourhood'] and len(str_value) > 40:
                                mandatory_fields[display_name] = str_value  # Show FULL text
                            elif display_name not in ['name', 'location', 'neighbourhood'] and len(str_value) > 100:
                                mandatory_fields[display_name] = str_value  # Show FULL text
                            else:
                                mandatory_fields[display_name] = str_value
                    else:
                        mandatory_fields[display_name] = str(value)
                    break  # Use first available field
        
        # Ensure we have at least some basic info even if fields are missing
        if 'name' not in mandatory_fields:
            mandatory_fields['name'] = 'Property Listing'
        if 'property_type' not in mandatory_fields:
            mandatory_fields['property_type'] = 'Property'
        # Ensure price field is always present, even if not found in document
        if 'price' not in mandatory_fields:
            mandatory_fields['price'] = self._format_price_field(document)
        # Don't add default 'Location not specified' - let format method handle it
        
        return mandatory_fields
    
    def _extract_mandatory_fields_fast(self, document: Dict[str, Any], doc_id: str) -> Dict[str, str]:
        """Fast extraction using preprocessed metadata when available"""
        # Try to get preprocessed metadata first
        if self._preprocessor:
            try:
                metadata = self._preprocessor.get_property_metadata(doc_id)
                if metadata:
                    # Convert metadata to mandatory fields format
                    fast_fields = {}
                    
                    if metadata.name:
                        fast_fields['name'] = metadata.name
                    if metadata.price_formatted:
                        fast_fields['price'] = metadata.price_formatted
                    if metadata.location:
                        fast_fields['location'] = metadata.location
                    if metadata.property_type:
                        fast_fields['property_type'] = metadata.property_type
                    if metadata.room_type:
                        fast_fields['room_type'] = metadata.room_type
                    if metadata.bedrooms is not None:
                        fast_fields['bedrooms'] = str(metadata.bedrooms)
                    if metadata.bathrooms is not None:
                        fast_fields['bathrooms'] = str(metadata.bathrooms)
                    if metadata.accommodates is not None:
                        fast_fields['accommodates'] = str(metadata.accommodates)
                    if metadata.host_name:
                        fast_fields['host_name'] = metadata.host_name
                    if metadata.review_score is not None:
                        if metadata.review_score > 10:
                            fast_fields['review_score'] = f'{metadata.review_score:.1f}/100'
                        else:
                            fast_fields['review_score'] = f'{metadata.review_score:.1f}/5'
                    if metadata.number_of_reviews is not None:
                        count = metadata.number_of_reviews
                        if count == 0:
                            fast_fields['number_of_reviews'] = 'No reviews'
                        elif count == 1:
                            fast_fields['number_of_reviews'] = '1 review'
                        else:
                            fast_fields['number_of_reviews'] = f'{count:,} reviews'
                    if metadata.instant_bookable is not None:
                        fast_fields['instant_bookable'] = 'Yes' if metadata.instant_bookable else 'No'
                    
                    # Add amenity flags if available
                    if metadata.has_wifi:
                        fast_fields['wifi'] = 'Yes'
                    if metadata.has_kitchen:
                        fast_fields['kitchen'] = 'Yes'
                    if metadata.has_parking:
                        fast_fields['parking'] = 'Yes'
                    
                    logger.debug(f"Used fast extraction for document {doc_id}")
                    return fast_fields
                    
            except Exception as e:
                logger.warning(f"Fast extraction failed for {doc_id}: {e}")
        
        # Fallback to original method
        return self._extract_mandatory_fields(document)
    
    def _format_price_field(self, document: Dict[str, Any]) -> str:
        """Enhanced price field formatting with extended field detection"""
        # Extended list of possible price field names in various data sources
        price_fields = [
            'price', 'nightly_rate', 'rate', 'cost', 'daily_rate', 'listing_price',
            'price_per_night', 'night_rate', 'base_price', 'rental_price',
            'accommodation_price', 'stay_price', 'booking_price', 'charges',
            'fee', 'amount', 'pricing', 'tariff', 'rent'  
        ]
        
        # First try exact field matches
        for field in price_fields:
            if field in document and document[field] is not None:
                value = document[field]
                formatted_price = self._parse_price_value(value)
                if formatted_price:
                    return formatted_price
        
        # Then try case-insensitive search through all document fields
        for key, value in document.items():
            if value is not None:
                key_lower = str(key).lower()
                # Check if the field name contains price-related terms
                if any(price_term in key_lower for price_term in ['price', 'cost', 'rate', 'fee', 'rent', 'charge']):
                    formatted_price = self._parse_price_value(value)
                    if formatted_price:
                        return formatted_price
        
        # Last resort: look for any numeric field that could be a price
        potential_prices = []
        for key, value in document.items():
            if isinstance(value, (int, float)) and 10 <= value <= 10000:  # Reasonable price range
                potential_prices.append((key, value))
        
        if potential_prices:
            # Sort by likelihood of being a price (prefer shorter field names that might be 'price')
            potential_prices.sort(key=lambda x: (len(str(x[0])), x[0]))
            best_candidate = potential_prices[0][1]
            return f'${best_candidate:,.0f}/night (estimated)'
        
        return 'Price information not found'
    
    def _parse_price_value(self, value) -> str:
        """Parse and format a potential price value"""
        try:
            if isinstance(value, (int, float)):
                if value >= 1:
                    return f'${value:,.0f}/night'
                else:
                    return f'${value:.2f}/night'
            elif isinstance(value, str):
                # Handle string prices like "$150.00", "150", "$1,500"
                import re
                # Extract numeric value from string
                price_match = re.search(r'[\d,]+\.?\d*', str(value).replace('$', '').replace(',', ''))
                if price_match:
                    clean_price = float(price_match.group().replace(',', ''))
                    if clean_price >= 1:
                        return f'${clean_price:,.0f}/night'
                    else:
                        return f'${clean_price:.2f}/night'
        except (ValueError, AttributeError):
            pass
        return None
    
    def _format_location_field(self, document: Dict[str, Any]) -> str:
        """Enhanced location field formatting with comprehensive field scanning"""
        location_parts = []
        
        # Check if address field exists and is structured (based on actual data structure)
        address = document.get('address')
        if isinstance(address, dict):
            # Extract suburb/neighbourhood from address structure
            suburb = address.get('suburb')
            if suburb and str(suburb).strip():
                location_parts.append(str(suburb).strip())
            
            # Extract market/city from address structure
            market = address.get('market')
            if market and str(market).strip():
                market_val = str(market).strip()
                if not location_parts or market_val.lower() not in [p.lower() for p in location_parts]:
                    location_parts.append(market_val)
            
            # Add country if available
            country = address.get('country')
            if country and str(country).strip():
                country_val = str(country).strip()
                if country_val not in location_parts:
                    location_parts.append(country_val)
            
            # If we have location from address, return it
            if location_parts:
                return ', '.join(location_parts)  # Show ALL location parts
        
        # Extended list of location field names to search
        location_fields = [
            'neighbourhood_cleansed', 'neighbourhood', 'neighborhood', 'area', 'district',
            'city', 'town', 'municipality', 'locality', 'place', 'region',
            'location', 'address', 'street', 'suburb', 'zone', 'sector',
            'country', 'state', 'province', 'county', 'administrative_area'
        ]
        
        # First pass: Get neighbourhood/area
        neighbourhood = None
        for field in ['neighbourhood_cleansed', 'neighbourhood', 'neighborhood', 'area', 'district', 'suburb', 'zone']:
            if field in document and document[field] and str(document[field]).strip():
                neighbourhood = str(document[field]).strip()
                break
        
        # Second pass: Get city/town
        city = None
        for field in ['city', 'town', 'municipality', 'locality', 'place']:
            if field in document and document[field] and str(document[field]).strip():
                city_val = str(document[field]).strip()
                if not neighbourhood or city_val.lower() != neighbourhood.lower():
                    city = city_val
                    break
        
        # Third pass: Get region/state if we still don't have enough info
        region = None
        if not neighbourhood and not city:
            for field in ['region', 'state', 'province', 'county', 'administrative_area']:
                if field in document and document[field] and str(document[field]).strip():
                    region = str(document[field]).strip()
                    break
        
        # Fourth pass: Scan all document fields for location-like terms
        if not neighbourhood and not city and not region:
            for key, value in document.items():
                if value and isinstance(value, str) and len(str(value).strip()) > 0:
                    key_lower = str(key).lower()
                    value_clean = str(value).strip()
                    
                    # Check if field name suggests location
                    location_terms = ['location', 'address', 'place', 'area', 'district', 'zone', 'locale']
                    if any(term in key_lower for term in location_terms) and len(value_clean) < 100:
                        # Avoid very long strings that might be descriptions
                        location_parts.append(value_clean)
                        break
        
        # Build location string with priority
        if neighbourhood:
            location_parts.append(neighbourhood)
        if city:
            location_parts.append(city)
        if region and not neighbourhood and not city:
            location_parts.append(region)
        
        # Clean up and return
        if location_parts:
            return ', '.join(location_parts)  # Show ALL location parts
        
        # Final fallback: check for any address-like field
        for key, value in document.items():
            if value and isinstance(value, str):
                value_clean = str(value).strip()
                key_lower = str(key).lower()
                if 'address' in key_lower and 5 <= len(value_clean) <= 50:
                    return value_clean
        
        return None  # Return None instead of default message
    
    def _format_rating_field(self, document: Dict[str, Any]) -> str:
        """Enhanced rating field formatting"""
        rating_fields = ['review_scores_rating', 'rating', 'overall_rating', 'score']
        for field in rating_fields:
            if field in document and document[field] is not None:
                value = document[field]
                if isinstance(value, (int, float)):
                    if value > 10:  # Rating out of 100
                        return f'{value:.1f}/100'
                    else:  # Rating out of 5
                        return f'{value:.1f}/5'
                return str(value)
        return 'No rating'
    
    def _format_reviews_field(self, document: Dict[str, Any]) -> str:
        """Enhanced reviews field formatting"""
        reviews_fields = ['number_of_reviews', 'review_count', 'total_reviews']
        for field in reviews_fields:
            if field in document and document[field] is not None:
                value = document[field]
                if isinstance(value, (int, float)):
                    count = int(value)
                    if count == 0:
                        return 'No reviews'
                    elif count == 1:
                        return '1 review'
                    else:
                        return f'{count:,} reviews'
                return str(value)
        return 'No reviews'
    
    def _format_host_status(self, document: Dict[str, Any]) -> str:
        """Format host status information"""
        host_name = document.get('host_name', 'Host')
        is_superhost = document.get('host_is_superhost')
        
        if isinstance(is_superhost, str):
            is_superhost = is_superhost.lower() in ['t', 'true', 'yes', '1']
        
        if is_superhost:
            return f'{host_name} (Superhost)'
        else:
            return host_name
    
    def _format_amenities_count(self, document: Dict[str, Any]) -> str:
        """Format amenities count"""
        amenities = document.get('amenities', [])
        if isinstance(amenities, list):
            count = len(amenities)
            if count == 0:
                return 'No amenities listed'
            elif count == 1:
                return '1 amenity'
            else:
                return f'{count} amenities'
        elif isinstance(amenities, str):
            # Handle comma-separated amenities string
            try:
                count = len([a.strip() for a in amenities.split(',') if a.strip()])
                return f'{count} amenities' if count > 0 else 'No amenities listed'
            except Exception:
                return 'Amenities available'
        return 'No amenities listed'
    
    def _extract_query_relevant_fields(self, document: Dict[str, Any], query: str, 
                                     query_analysis: Any = None) -> Dict[str, str]:
        """Extract fields from document that are most relevant to the user's query"""
        relevant_fields = {}
        query_lower = query.lower()
        
        # Define field categories and their keywords
        field_categories = {
            'pricing': {
                'keywords': ['price', 'cost', 'cheap', 'expensive', 'budget', 'affordable', '$', 'dollar', 'fee'],
                'fields': ['cleaning_fee', 'security_deposit', 'extra_people', 'weekly_price', 'monthly_price']
            },
            'location': {
                'keywords': ['location', 'area', 'neighbourhood', 'neighborhood', 'city', 'near', 'close', 'downtown'],
                'fields': ['country', 'zipcode', 'street', 'latitude', 'longitude']
            },
            'specifications': {
                'keywords': ['bathroom', 'size', 'square', 'feet', 'room'],
                'fields': ['bathrooms', 'room_type', 'bed_type', 'square_feet']
            },
            'amenities': {
                'keywords': ['wifi', 'kitchen', 'parking', 'pool', 'gym', 'amenity', 'feature', 'facility'],
                'fields': ['amenities', 'host_amenities']
            },
            'ratings': {
                'keywords': ['rating', 'review', 'score', 'quality', 'feedback', 'star'],
                'fields': ['review_scores_rating', 'number_of_reviews', 'reviews_per_month', 
                          'review_scores_accuracy', 'review_scores_cleanliness', 'review_scores_checkin']
            },
            'host': {
                'keywords': ['host', 'owner', 'superhost'],
                'fields': ['host_name', 'host_is_superhost', 'host_response_rate', 'host_response_time',
                          'host_listings_count', 'host_total_listings_count']
            },
            'policies': {
                'keywords': ['cancel', 'policy', 'instant', 'book', 'minimum', 'maximum'],
                'fields': ['cancellation_policy', 'instant_bookable', 'minimum_nights', 'maximum_nights']
            },
            'description': {
                'keywords': ['describe', 'about', 'detail', 'space', 'summary'],
                'fields': ['summary', 'description', 'space', 'neighborhood_overview', 'notes']
            }
        }
        
        # Check which categories are relevant to the query
        relevant_categories = []
        for category, info in field_categories.items():
            if any(keyword in query_lower for keyword in info['keywords']):
                relevant_categories.append(category)
        
        # If no specific categories found, include common useful info
        if not relevant_categories:
            relevant_categories = ['ratings', 'amenities']
        
        # Extract relevant fields
        for category in relevant_categories:
            if category in field_categories:
                for field in field_categories[category]['fields']:
                    if field in document and document[field] is not None:
                        value = document[field]
                        
                        # Format the value for display
                        if isinstance(value, list):
                            if len(value) > 5:  # Truncate long lists
                                formatted_value = ', '.join(str(v) for v in value)  # Show ALL values
                            else:
                                formatted_value = ', '.join(str(v) for v in value)
                        elif isinstance(value, (int, float)):
                            if 'price' in field or 'fee' in field:
                                formatted_value = f'${value}'
                            elif 'rate' in field or 'rating' in field:
                                formatted_value = f'{value}' + ('/100' if field == 'review_scores_rating' else '')
                            else:
                                formatted_value = str(value)
                        else:
                            # Truncate long text fields
                            str_value = str(value)
                            # Show FULL text regardless of length for comprehensive display
                            formatted_value = str_value  # Show FULL text without truncation
                        
                        relevant_fields[field] = formatted_value
        
        return relevant_fields
    
    def _clean_document_for_json(self, obj):
        """Clean document for JSON serialization with better formatting"""
        if hasattr(obj, 'isoformat'):  # datetime object
            return obj.isoformat()
        elif isinstance(obj, dict):
            return {k: self._clean_document_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._clean_document_for_json(item) for item in obj]
        elif isinstance(obj, (str, int, float, bool)) or obj is None:
            return obj
        else:
            return str(obj)
    
    def _format_json_for_display(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """Format JSON document for better display with enhanced organization"""
        formatted_doc = {}
        
        # Enhanced field groups with more comprehensive coverage
        field_groups = {
            '🏠 Basic Information': [
                'name', 'title', 'property_type', 'room_type', 'accommodates', 
                'bedrooms', 'bathrooms', 'beds', 'bed_type', 'square_feet'
            ],
            '📍 Location Details': [
                'city', 'neighbourhood_cleansed', 'neighborhood', 'country', 
                'zipcode', 'street', 'address', 'latitude', 'longitude', 
                'transit', 'neighborhood_overview'
            ],
            '💰 Pricing Information': [
                'price', 'cleaning_fee', 'security_deposit', 'extra_people', 
                'weekly_price', 'monthly_price', 'guests_included'
            ],
            '⭐ Reviews & Ratings': [
                'review_scores_rating', 'number_of_reviews', 'reviews_per_month',
                'review_scores_accuracy', 'review_scores_cleanliness', 
                'review_scores_checkin', 'review_scores_communication',
                'review_scores_location', 'review_scores_value'
            ],
            '🏆 Host Information': [
                'host_id', 'host_name', 'host_since', 'host_is_superhost',
                'host_response_time', 'host_response_rate', 'host_acceptance_rate',
                'host_listings_count', 'host_total_listings_count', 'host_about'
            ],
            '📋 Booking Policies': [
                'cancellation_policy', 'instant_bookable', 'minimum_nights', 
                'maximum_nights', 'calendar_updated', 'availability_365',
                'require_guest_profile_picture', 'require_guest_phone_verification'
            ],
            '🎯 Amenities & Features': [
                'amenities', 'host_amenities', 'has_availability',
                'license', 'jurisdiction_names'
            ],
            '📝 Descriptions': [
                'summary', 'description', 'space', 'access', 
                'interaction', 'house_rules', 'notes'
            ]
        }
        
        # Group fields with enhanced formatting and value processing
        for group_name, fields in field_groups.items():
            group_data = {}
            for field in fields:
                if field in document and document[field] is not None:
                    value = document[field]
                    
                    # Apply enhanced formatting based on field type
                    formatted_value = self._format_display_value(field, value)
                    group_data[self._format_field_name(field)] = formatted_value
            
            if group_data:
                formatted_doc[group_name] = group_data
        
        # Add any remaining fields to "📋 Additional Information"
        remaining_fields = {}
        grouped_fields = set()
        for fields in field_groups.values():
            grouped_fields.update(fields)
        
        for key, value in document.items():
            if key not in grouped_fields and value is not None:
                formatted_value = self._format_display_value(key, value)
                formatted_name = self._format_field_name(key)
                remaining_fields[formatted_name] = formatted_value
        
        if remaining_fields:
            formatted_doc['📋 Additional Information'] = remaining_fields
        
        return formatted_doc
    
    def _format_field_name(self, field_name: str) -> str:
        """Format field names for better display"""
        # Convert snake_case to Title Case
        formatted = field_name.replace('_', ' ').title()
        
        # Handle special cases
        special_cases = {
            'Id': 'ID',
            'Url': 'URL',
            'Wifi': 'Wi-Fi',
            'Tv': 'TV',
            'Ac': 'A/C',
            'Gps': 'GPS',
            'Dvd': 'DVD'
        }
        
        for old, new in special_cases.items():
            formatted = formatted.replace(old, new)
        
        return formatted
    
    def _format_display_value(self, field_name: str, value: Any) -> Any:
        """Format values for enhanced display based on field type"""
        if value is None:
            return 'N/A'
        
        # Handle different field types with specific formatting
        if 'price' in field_name.lower() or 'fee' in field_name.lower():
            if isinstance(value, (int, float)) and value > 0:
                return f"${value:,.2f}" if value < 1 else f"${value:,.0f}"
            return str(value)
        
        elif 'rating' in field_name.lower() or 'score' in field_name.lower():
            if isinstance(value, (int, float)):
                if value > 10:
                    return f"{value:.1f}/100"
                else:
                    return f"{value:.1f}/5"
            return str(value)
        
        elif 'response_rate' in field_name.lower() or 'acceptance_rate' in field_name.lower():
            if isinstance(value, (int, float)):
                return f"{value}%"
            elif isinstance(value, str) and '%' not in value:
                return f"{value}%"
            return str(value)
        
        elif field_name.lower() in ['instant_bookable', 'host_is_superhost', 'has_availability']:
            if isinstance(value, bool):
                return 'Yes' if value else 'No'
            elif isinstance(value, str):
                if value.lower() in ['t', 'true', 'yes', '1']:
                    return 'Yes'
                elif value.lower() in ['f', 'false', 'no', '0']:
                    return 'No'
            return str(value)
        
        elif 'date' in field_name.lower() or 'since' in field_name.lower():
            if hasattr(value, 'strftime'):
                return value.strftime('%Y-%m-%d')
            return str(value)
        
        elif isinstance(value, list):
            if len(value) == 0:
                return 'None'
            elif len(value) <= 10:
                return ', '.join(str(item) for item in value)
            else:
                return f"{', '.join(str(item) for item in value)}"  # Show ALL items
        
        elif isinstance(value, str):
            return value  # Show FULL text without truncation
        
        elif isinstance(value, (int, float)):
            # Format large numbers with commas
            if isinstance(value, int) and value > 1000:
                return f"{value:,}"
            return str(value)
        
        return str(value)
    
    def _prepare_comparison_data(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare comprehensive document data for table comparison with enhanced formatting"""
        comparison_fields = {}
        
        # === CORE PROPERTY INFO (Always shown) ===
        comparison_fields['Property Name'] = self._format_field_value(document, ['name', 'title', 'listing_name'], 'Property Listing')
        comparison_fields['Property Type'] = self._format_field_value(document, ['property_type', 'type'], 'Property')
        comparison_fields['Room Type'] = self._format_field_value(document, ['room_type', 'space_type'], 'N/A')
        
        # === LOCATION (Enhanced) ===
        # Primary location
        neighbourhood = self._format_field_value(document, ['neighbourhood_cleansed', 'neighbourhood'], '')
        city = self._format_field_value(document, ['city'], '')
        
        if neighbourhood and city:
            location = f"{neighbourhood}, {city}"
        elif neighbourhood:
            location = neighbourhood
        elif city:
            location = city
        else:
            location = 'Location not specified'
            
        if len(location) > 35:
            # Show FULL location without truncation
            pass  # Keep full location
        comparison_fields['Location'] = location
        comparison_fields['Neighbourhood'] = self._format_field_value(document, ['neighbourhood_cleansed', 'neighbourhood'], 'N/A')
        
        # === PRICING (Comprehensive) ===
        price = document.get('price')
        if price is not None and isinstance(price, (int, float)) and price > 0:
            comparison_fields['Price/Night'] = f"${price:,.0f}"
        else:
            comparison_fields['Price/Night'] = 'N/A'
        
        # Additional fees
        cleaning_fee = document.get('cleaning_fee')
        if cleaning_fee is not None and isinstance(cleaning_fee, (int, float)) and cleaning_fee > 0:
            comparison_fields['Cleaning Fee'] = f"${cleaning_fee:,.0f}"
        else:
            comparison_fields['Cleaning Fee'] = 'None'
            
        security_deposit = document.get('security_deposit')
        if security_deposit is not None and isinstance(security_deposit, (int, float)) and security_deposit > 0:
            comparison_fields['Security Deposit'] = f"${security_deposit:,.0f}"
        else:
            comparison_fields['Security Deposit'] = 'None'
        
        # === ACCOMMODATION DETAILS (Enhanced) ===
        comparison_fields['Bedrooms'] = self._format_numeric_field(document, ['bedrooms', 'bedroom_count'])
        comparison_fields['Beds'] = self._format_numeric_field(document, ['beds', 'bed_count', 'total_beds'])
        comparison_fields['Bathrooms'] = self._format_numeric_field(document, ['bathrooms', 'bathroom_count'])
        comparison_fields['Max Guests'] = self._format_numeric_field(document, ['accommodates', 'guests', 'capacity'])
        
        # === REVIEWS & RATINGS (Enhanced) ===
        # Overall rating
        rating = document.get('review_scores_rating')
        if rating is not None and isinstance(rating, (int, float)):
            if rating > 10:
                comparison_fields['Overall Rating'] = f"{rating:.1f}/100"
            else:
                comparison_fields['Overall Rating'] = f"{rating:.1f}/5"
        else:
            comparison_fields['Overall Rating'] = 'No Rating'
        
        # Review count
        review_count = document.get('number_of_reviews')
        if review_count is not None and isinstance(review_count, (int, float)):
            count = int(review_count)
            if count == 0:
                comparison_fields['Total Reviews'] = 'No reviews'
            elif count == 1:
                comparison_fields['Total Reviews'] = '1 review'
            else:
                comparison_fields['Total Reviews'] = f'{count:,} reviews'
        else:
            comparison_fields['Total Reviews'] = 'No reviews'
            
        # Reviews per month
        reviews_per_month = document.get('reviews_per_month')
        if reviews_per_month is not None and isinstance(reviews_per_month, (int, float)) and reviews_per_month > 0:
            comparison_fields['Reviews/Month'] = f'{reviews_per_month:.1f}'
        else:
            comparison_fields['Reviews/Month'] = 'N/A'
        
        # === HOST INFORMATION (Enhanced) ===
        host_name = self._format_field_value(document, ['host_name', 'host'], 'Host')
        is_superhost = document.get('host_is_superhost')
        if isinstance(is_superhost, str) and is_superhost.lower() in ['t', 'true', 'yes', '1']:
            comparison_fields['Host'] = f'{host_name} ⭐'
        elif isinstance(is_superhost, bool) and is_superhost:
            comparison_fields['Host'] = f'{host_name} ⭐'
        else:
            comparison_fields['Host'] = host_name
            
        # Host response time
        response_time = document.get('host_response_time')
        if response_time:
            comparison_fields['Response Time'] = response_time.replace('_', ' ').title()
        else:
            comparison_fields['Response Time'] = 'N/A'
        
        # === BOOKING POLICIES (Enhanced) ===
        comparison_fields['Instant Book'] = self._format_boolean_field(document, ['instant_bookable', 'instant_book'])
        
        # Minimum nights
        min_nights = document.get('minimum_nights')
        if min_nights is not None and isinstance(min_nights, (int, float)):
            nights = int(min_nights) if min_nights == int(min_nights) else min_nights
            comparison_fields['Min Nights'] = f"{nights} night{'s' if nights != 1 else ''}"
        else:
            comparison_fields['Min Nights'] = 'N/A'
            
        # Maximum nights
        max_nights = document.get('maximum_nights')
        if max_nights is not None and isinstance(max_nights, (int, float)) and max_nights < 1000:
            nights = int(max_nights) if max_nights == int(max_nights) else max_nights
            comparison_fields['Max Nights'] = f"{nights} night{'s' if nights != 1 else ''}"
        else:
            comparison_fields['Max Nights'] = 'No limit'
        
        # Cancellation policy
        cancellation = document.get('cancellation_policy')
        if cancellation:
            comparison_fields['Cancellation'] = cancellation.replace('_', ' ').title()
        else:
            comparison_fields['Cancellation'] = 'N/A'
        
        # === AVAILABILITY ===
        availability_365 = document.get('availability_365')
        if availability_365 is not None and isinstance(availability_365, (int, float)):
            comparison_fields['Yearly Availability'] = f'{int(availability_365)} days'
        else:
            comparison_fields['Yearly Availability'] = 'N/A'
        
        # === AMENITIES (Enhanced) ===
        amenities = document.get('amenities', [])
        if isinstance(amenities, list):
            amenity_count = len(amenities)
            if amenity_count > 0:
                comparison_fields['Total Amenities'] = f'{amenity_count} amenities'
                # Top 5 amenities for comparison
                top_amenities = amenities  # Show ALL amenities
                top_amenities_str = ', '.join(top_amenities)  # Show ALL amenities
                comparison_fields['Key Amenities'] = top_amenities_str
            else:
                comparison_fields['Total Amenities'] = 'No amenities'
                comparison_fields['Key Amenities'] = 'None listed'
        else:
            comparison_fields['Total Amenities'] = 'N/A'
            comparison_fields['Key Amenities'] = 'N/A'
        
        return comparison_fields
    
    def _format_field_value(self, document: Dict[str, Any], field_options: List[str], default: str = 'N/A') -> str:
        """Helper method to format field values with fallbacks"""
        for field in field_options:
            if field in document and document[field] is not None:
                value = str(document[field]).strip()
                return value if value else default
        return default
    
    def _format_numeric_field(self, document: Dict[str, Any], field_options: List[str]) -> str:
        """Helper method to format numeric fields"""
        for field in field_options:
            if field in document and document[field] is not None:
                value = document[field]
                if isinstance(value, (int, float)):
                    return str(int(value)) if value == int(value) else f'{value:.1f}'
                else:
                    return str(value)
        return 'N/A'
    
    def _format_boolean_field(self, document: Dict[str, Any], field_options: List[str]) -> str:
        """Helper method to format boolean fields"""
        for field in field_options:
            if field in document and document[field] is not None:
                value = document[field]
                if isinstance(value, bool):
                    return 'Yes' if value else 'No'
                elif isinstance(value, str):
                    if value.lower() in ['t', 'true', 'yes', '1']:
                        return 'Yes'
                    elif value.lower() in ['f', 'false', 'no', '0']:
                        return 'No'
                return str(value)
        return 'No'
    
    def _generate_comparison_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary information for property comparison"""
        if len(results) < 2:
            return {}
        
        summary = {
            'total_properties': len(results),
            'price_range': {'min': float('inf'), 'max': 0},
            'bedroom_range': {'min': float('inf'), 'max': 0},
            'average_rating': 0,
            'locations': set(),
            'property_types': set()
        }
        
        valid_prices = []
        valid_bedrooms = []
        valid_ratings = []
        
        for result in results:
            doc = result.get('document', {})
            
            # Price analysis
            price = doc.get('price')
            if price and isinstance(price, (int, float)):
                valid_prices.append(price)
                summary['price_range']['min'] = min(summary['price_range']['min'], price)
                summary['price_range']['max'] = max(summary['price_range']['max'], price)
            
            # Bedroom analysis
            bedrooms = doc.get('bedrooms')
            if bedrooms and isinstance(bedrooms, (int, float)):
                valid_bedrooms.append(bedrooms)
                summary['bedroom_range']['min'] = min(summary['bedroom_range']['min'], bedrooms)
                summary['bedroom_range']['max'] = max(summary['bedroom_range']['max'], bedrooms)
            
            # Rating analysis
            rating = doc.get('review_scores_rating')
            if rating and isinstance(rating, (int, float)):
                valid_ratings.append(rating)
            
            # Location analysis
            location = doc.get('neighbourhood_cleansed') or doc.get('city')
            if location:
                summary['locations'].add(location)
            
            # Property type analysis
            prop_type = doc.get('property_type')
            if prop_type:
                summary['property_types'].add(prop_type)
        
        # Calculate averages and finalize
        if valid_ratings:
            summary['average_rating'] = sum(valid_ratings) / len(valid_ratings)
        
        if summary['price_range']['min'] == float('inf'):
            summary['price_range'] = None
        else:
            summary['average_price'] = sum(valid_prices) / len(valid_prices) if valid_prices else 0
        
        if summary['bedroom_range']['min'] == float('inf'):
            summary['bedroom_range'] = None
        
        summary['locations'] = list(summary['locations'])
        summary['property_types'] = list(summary['property_types'])
        
        return summary
    
    def _generate_fast_summary(self, document: Dict[str, Any], query: str) -> str:
        """Generate fast summary without slow AI calls"""
        try:
            return self._generate_comprehensive_fallback_summary(document, query, min_words=100)
        except Exception as e:
            logger.warning(f"Fast summary generation failed: {e}")
            return self._generate_basic_summary(document)
    
    def _generate_basic_summary(self, document: Dict[str, Any]) -> str:
        """Generate basic fast summary from document fields"""
        summary_parts = []
        
        # Property basics
        name = document.get('name', 'Property')
        property_type = document.get('property_type', 'accommodation')
        summary_parts.append(f"This {property_type.lower()} called '{name}'")
        
        # Key details
        if document.get('accommodates'):
            summary_parts.append(f"accommodates {document['accommodates']} guests")
        if document.get('bedrooms'):
            summary_parts.append(f"has {document['bedrooms']} bedroom(s)")
        if document.get('price'):
            summary_parts.append(f"costs ${document['price']}/night")
        
        # Location
        location = document.get('neighbourhood_cleansed') or document.get('city')
        if location:
            summary_parts.append(f"located in {location}")
        
        return '. '.join(summary_parts) + '.'  # Show ALL summary parts
    
    def _generate_enhanced_ai_summary(self, document: Dict[str, Any], query: str, min_words: int = 200) -> str:
        """Generate comprehensive AI summary from source JSON data with minimum word count (slower)"""
        try:
            # Only use AI summarization in specific cases to avoid performance issues
            # For now, use fast fallback to maintain performance
            return self._generate_comprehensive_fallback_summary(document, query, min_words)
            
        except Exception as e:
            logger.warning(f"AI summary generation failed: {e}")
            return self._generate_comprehensive_fallback_summary(document, query, min_words)
    
    def _generate_comprehensive_fallback_summary(self, document: Dict[str, Any], query: str, min_words: int = 100) -> str:
        """Generate a fast comprehensive summary with reduced word count"""
        summary_parts = []
        
        # Property overview
        property_type = document.get('property_type', 'Property')
        name = document.get('name', 'property')
        summary_parts.append(f"This {property_type.lower()} '{name}'")
        
        # Key specs
        specs = []
        if document.get('accommodates'):
            specs.append(f"accommodates {document['accommodates']} guests")
        if document.get('bedrooms'):
            specs.append(f"{document['bedrooms']} bedroom(s)")
        if document.get('price'):
            specs.append(f"${document['price']}/night")
        
        if specs:
            summary_parts.append(', '.join(specs))
        
        # Location
        location = document.get('neighbourhood_cleansed') or document.get('city')
        if location:
            summary_parts.append(f"located in {location}")
        
        # Rating if available
        if document.get('review_scores_rating') and document.get('number_of_reviews'):
            summary_parts.append(f"rated {document['review_scores_rating']}/100 from {document['number_of_reviews']} reviews")
        
        # Key amenities (limit to 3 for speed)
        if document.get('amenities'):
            amenities = document['amenities']
            if isinstance(amenities, list) and len(amenities) > 0:
                top_amenities = amenities  # Show ALL amenities
                summary_parts.append(f"amenities include {', '.join(top_amenities)}")
        
        # Join with proper formatting
        base_summary = '. '.join([part.strip() for part in summary_parts if part.strip()]) + '.'
        
        # Only add minimal context if needed to reach min_words
        if len(base_summary.split()) < min_words:
            if document.get('summary'):
                desc = str(document['summary'])  # Show FULL summary without truncation
                base_summary += f" {desc}"
        
        return base_summary
    
    def _format_json_fast(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """Fast JSON formatting with minimal processing"""
        if not document:
            return {}
        
        # Only format essential fields to improve speed
        essential_fields = {
            '🏠 Basic Info': ['name', 'property_type', 'room_type', 'accommodates', 'bedrooms', 'bathrooms'],
            '📍 Location': ['neighbourhood_cleansed', 'city', 'country'],
            '💰 Pricing': ['price', 'cleaning_fee'],
            '⭐ Reviews': ['review_scores_rating', 'number_of_reviews'],
            '🎯 Amenities': ['amenities']
        }
        
        formatted_doc = {}
        for group_name, fields in essential_fields.items():
            group_data = {}
            for field in fields:
                if field in document and document[field] is not None:
                    value = document[field]
                    # Minimal formatting for speed
                    if isinstance(value, list) and len(value) > 5:
                        group_data[field.replace('_', ' ').title()] = value  # Show FULL list
                    elif isinstance(value, str) and len(value) > 100:
                        group_data[field.replace('_', ' ').title()] = value  # Show FULL text
                    else:
                        group_data[field.replace('_', ' ').title()] = value
            
            if group_data:
                formatted_doc[group_name] = group_data
        
        return formatted_doc
    
    def _prepare_comparison_data_fast(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """Fast comparison data preparation with essential fields only"""
        comparison_fields = {}
        
        # Only extract the most critical comparison fields for speed
        essential_comparisons = {
            'name': ['name', 'title'],
            'type': ['property_type'],
            'price': ['price'],
            'location': ['neighbourhood_cleansed', 'city'],
            'bedrooms': ['bedrooms'],
            'bathrooms': ['bathrooms'],
            'guests': ['accommodates'],
            'rating': ['review_scores_rating'],
            'reviews': ['number_of_reviews']
        }
        
        for key, field_options in essential_comparisons.items():
            for field in field_options:
                if field in document and document[field] is not None:
                    value = document[field]
                    if isinstance(value, (int, float)):
                        if key == 'price':
                            comparison_fields[key.title()] = f"${value:,.0f}"
                        elif key == 'rating':
                            comparison_fields[key.title()] = f"{value}/100" if value > 10 else f"{value}/5"
                        else:
                            comparison_fields[key.title()] = str(int(value)) if value == int(value) else f"{value:.1f}"
                    else:
                        str_val = str(value)
                        comparison_fields[key.title()] = str_val  # Show FULL text
                    break
        
        return comparison_fields
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current enhanced system status"""
        status = {
            'status': 'ready' if self.is_ready() else 'not_ready',
            'database_connected': self._status['database_connected'],
            'enhanced_features_enabled': True,
            'mandatory_fields_support': True,
            'comparison_support': True,
            'last_updated': timezone.now().isoformat()
        }
        
        if self.is_ready() and hasattr(self.rag_system, 'get_system_status'):
            try:
                rag_status = self.rag_system.get_system_status()
                status.update({
                    'index_stats': rag_status.get('index_stats', {}),
                    'session_stats': rag_status.get('session_stats', {}),
                    'performance_metrics': {
                        'initialization_time': self._status.get('initialization_time', 0),
                        'uptime': time.time() - (self._status.get('initialization_time', time.time()))
                    }
                })
            except Exception as e:
                logger.warning(f"Failed to get RAG system status: {e}")
        
        return status
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get comprehensive system health information"""
        health = {
            'status': 'healthy',
            'timestamp': timezone.now().isoformat(),
            'checks': {
                'rag_system_init': self.is_ready(),
                'database_connection': self._status['database_connected'],
                'query_processing': False
            }
        }
        
        # Test query processing
        if self.is_ready():
            try:
                test_result = self.process_query("test query")
                health['checks']['query_processing'] = test_result['success']
            except Exception as e:
                health['checks']['query_processing'] = False
                health['checks']['query_processing_error'] = str(e)
        else:
            health['checks']['query_processing'] = False
        
        # Overall health status
        health['healthy'] = all([
            health['checks']['rag_system_init'],
            health['checks']['database_connection'],
            health['checks']['query_processing']
        ])
        
        return health
    
    def _format_context_aware_response(self, query_text: str, context_response: Dict[str, Any], 
                                       original_response: Dict[str, Any], start_time: float) -> Dict[str, Any]:
        """
        Format context-aware response with enhanced metadata
        """
        execution_time = time.time() - start_time
        
        # Ensure context_response is a dictionary
        if not isinstance(context_response, dict):
            logger.warning(f"context_response is not a dictionary: {type(context_response)}")
            context_response = {'results': [], 'metadata': {}}
        
        # Get results from context response
        results = context_response.get('results', [])
        metadata = context_response.get('metadata', {})
        
        # Process results with mandatory field extraction and enhanced formatting
        processed_results = []
        for result in results:
            processed_result = {
                'id': result.get('id', len(processed_results)) if isinstance(result, dict) else len(processed_results),
                'score': result.get('score', 0.0) if isinstance(result, dict) else 0.0,
                'mandatory_fields': self._extract_mandatory_fields_fast(
                    result if isinstance(result, dict) else {'content': result}, 
                    str(result.get('id', result.get('_id', f'doc_{len(processed_results)}')) if isinstance(result, dict) else f'doc_{len(processed_results)}')
                ),
                'query_relevant_fields': self._extract_query_relevant_fields(result if isinstance(result, dict) else {'content': result}, query_text),
                'ai_summary': self._generate_enhanced_ai_summary(
                    result if isinstance(result, dict) else {'content': result}, query_text, min_words=200
                ),
                'source_json': self._format_json_for_display(result if isinstance(result, dict) else {'content': result})
            }
            processed_results.append(processed_result)
        
        # Enhanced metadata with context information
        enhanced_metadata = {
            'execution_time': execution_time,
            'num_results': len(processed_results),
            'results': processed_results,
            'context_aware': True,
            'intent_detected': context_response.get('context_info', {}).get('intent', 'search'),
            'operation_type': metadata.get('operation_type', 'standard_search'),
            'used_cached_results': metadata.get('used_cached_results', False),
            'has_mandatory_fields': True,
            'comparison_support': True
        }
        
        # Add comparison data if available
        if 'comparison_table' in context_response:
            enhanced_metadata['comparison_table'] = context_response.get('comparison_table', {})
            enhanced_metadata['comparison_enabled'] = True
            enhanced_metadata['comparison_summary'] = context_response.get('comparison_summary', {})
        
        # Add context-specific information
        if 'context_info' in context_response:
            enhanced_metadata['context_info'] = context_response.get('context_info', {})
        
        # Calculate similarity scores if available
        if processed_results:
            scores = [r.get('score', 0.0) for r in processed_results if r.get('score') is not None]
            if scores:
                enhanced_metadata['max_similarity_score'] = max(scores)
                enhanced_metadata['avg_similarity_score'] = sum(scores) / len(scores)
        
        # Generate AI response based on context
        ai_response = self._generate_context_aware_response(
            query_text, processed_results, context_response
        )
        
        return {
            'success': True,
            'response': ai_response,
            'metadata': enhanced_metadata
        }
    
    def _generate_context_aware_response(self, query_text: str, results: List[Dict], 
                                        context_response: Dict[str, Any]) -> str:
        """
        Generate AI response that's aware of conversation context
        """
        # Safety check to ensure context_response is a dictionary
        if not isinstance(context_response, dict):
            logger.warning(f"context_response is not a dictionary in _generate_context_aware_response: {type(context_response)}")
            return self._generate_standard_response(query_text, results)
        
        intent = context_response.get('context_info', {}).get('intent', 'search')
        operation_type = context_response.get('metadata', {}).get('operation_type', 'standard_search')
        
        if intent == 'comparison' and 'comparison_table' in context_response:
            return self._generate_comparison_response(query_text, results, context_response)
        elif intent == 'filtering':
            return self._generate_filtering_response(query_text, results, context_response)
        elif intent == 'summarization':
            return self._generate_summarization_response(query_text, results, context_response)
        elif intent == 'reasoning':
            return self._generate_reasoning_response(query_text, results, context_response)
        else:
            return self._generate_standard_response(query_text, results)
    
    def _generate_comparison_response(self, query_text: str, results: List[Dict], 
                                    context_response: Dict[str, Any]) -> str:
        """Generate response for comparison operations"""
        # Safety check to ensure context_response is a dictionary
        if not isinstance(context_response, dict):
            logger.warning(f"context_response is not a dictionary in _generate_comparison_response: {type(context_response)}")
            return self._generate_standard_response(query_text, results)
        
        comparison_table = context_response.get('comparison_table', {})
        items_compared = comparison_table.get('items_compared', 0)
        similarities = comparison_table.get('similarities', [])
        differences = comparison_table.get('key_differences', [])
        
        response_parts = [
            f"I've compared {items_compared} properties from your previous search results."
        ]
        
        if similarities:
            similar_attrs = [s['attribute'] for s in similarities]  # Show ALL similarities
            response_parts.append(
                f"These properties share common features: {', '.join(similar_attrs)}."
            )
        
        if differences:
            diff_attrs = [d['attribute'] for d in differences]  # Show ALL differences
            response_parts.append(
                f"Key differences include: {', '.join(diff_attrs)}."
            )
        
        response_parts.append(
            "Please review the detailed comparison table below to see all attributes side-by-side."
        )
        
        return " ".join(response_parts)
    
    def _generate_filtering_response(self, query_text: str, results: List[Dict], 
                                   context_response: Dict[str, Any]) -> str:
        """Generate response for filtering operations"""
        metadata = context_response.get('metadata', {})
        original_count = metadata.get('original_count', 0)
        filtered_count = len(results)
        filters_applied = metadata.get('filters_applied', {})
        
        response_parts = [
            f"I've filtered your previous {original_count} results down to {filtered_count} properties"
        ]
        
        if filters_applied:
            filter_desc = []
            for key, value in filters_applied.items():
                if isinstance(value, list):
                    filter_desc.append(f"{key}: {', '.join(map(str, value))}")
                else:
                    filter_desc.append(f"{key}: {value}")
            
            if filter_desc:
                response_parts.append(f"based on: {'; '.join(filter_desc)}.")
        
        if filtered_count > 0:
            response_parts.append(
                f"Here are the {filtered_count} properties that meet your criteria:"
            )
        else:
            response_parts.append(
                "Unfortunately, no properties match your specific criteria. Try broadening your filters."
            )
        
        return " ".join(response_parts)
    
    def _generate_summarization_response(self, query_text: str, results: List[Dict], 
                                       context_response: Dict[str, Any]) -> str:
        """Generate response for summarization operations"""
        summary_data = context_response.get('summary', {})
        stats = summary_data.get('stats', {})
        
        response_parts = [
            f"Here's a summary of your {len(results)} search results:"
        ]
        
        if 'avg_price' in stats and stats['avg_price'] > 0:
            response_parts.append(
                f"Average price: ${stats['avg_price']:.0f}."
            )
        
        if 'price_range' in stats:
            price_range = stats['price_range']
            if price_range['max'] > 0:
                response_parts.append(
                    f"Price range: ${price_range['min']:.0f} - ${price_range['max']:.0f}."
                )
        
        if 'property_types' in stats:
            types = list(stats['property_types'].keys())  # Show ALL property types
            if types:
                response_parts.append(
                    f"Property types include: {', '.join(types)}."
                )
        
        overview = summary_data.get('overview', '')
        if overview:
            response_parts.append(overview)
        
        return " ".join(response_parts)
    
    def _generate_reasoning_response(self, query_text: str, results: List[Dict], 
                                   context_response: Dict[str, Any]) -> str:
        """Generate response for reasoning/explanation operations"""
        explanation_data = context_response.get('explanation', {})
        explanations = explanation_data.get('explanations', [])
        
        response_parts = [
            "Let me explain why these properties were recommended:"
        ]
        
        for exp in explanations:  # Show ALL explanations
            response_parts.append(f"• {exp.get('explanation', '')}")
        
        summary = explanation_data.get('summary', '')
        if summary:
            response_parts.append(summary)
        
        return "\n".join(response_parts)
    
    def _generate_standard_response(self, query_text: str, results: List[Dict]) -> str:
        """Generate standard response for regular search operations"""
        if not results:
            return "I couldn't find any properties matching your criteria. Please try a different search."
        
        response_parts = [
            f"I found {len(results)} properties that match your search for '{query_text}'."
        ]
        
        # Add brief summary of top results
        if results:
            top_result = results[0]
            mandatory_fields = top_result.get('mandatory_fields', {})
            
            if mandatory_fields:
                name = mandatory_fields.get('name', 'a property')
                price = mandatory_fields.get('price', '')
                location = mandatory_fields.get('location', '')
                bedrooms = mandatory_fields.get('bedrooms', '')
                
                description_parts = [name]
                if price:
                    description_parts.append(f"priced at {price}")
                if location:
                    description_parts.append(f"located in {location}")
                if bedrooms:
                    description_parts.append(f"with {bedrooms}")
                
                response_parts.append(
                    f"The top result is {' '.join(description_parts)}."
                )
        
        response_parts.append("Please review the detailed results below.")
        
        return " ".join(response_parts)
