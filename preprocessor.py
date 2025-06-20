"""
Data Preprocessing Module for Django RAG System
Extracts and stores key fields from JSON data during setup for faster response times
"""

import logging
import json
import time
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path
import pickle
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

@dataclass
class PropertyMetadata:
    """Structured metadata for property records"""
    id: str
    name: str = None
    price: float = None
    price_formatted: str = None
    location: str = None
    neighbourhood: str = None
    city: str = None
    property_type: str = None
    room_type: str = None
    bedrooms: int = None
    bathrooms: float = None
    accommodates: int = None
    amenities: List[str] = None
    host_name: str = None
    review_score: float = None
    number_of_reviews: int = None
    instant_bookable: bool = None
    minimum_nights: int = None
    availability_365: int = None
    
    # Additional extracted fields
    has_wifi: bool = False
    has_kitchen: bool = False
    has_parking: bool = False
    has_pool: bool = False
    has_pet: bool = False
    
    # Computed fields
    price_per_person: float = None
    value_score: float = None  # Based on price, reviews, amenities
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class FieldExtractor:
    """Extracts key fields from JSON documents using intelligent parsing"""
    
    def __init__(self):
        self.price_fields = [
            'price', 'nightly_rate', 'rate', 'cost', 'daily_rate', 'listing_price',
            'price_per_night', 'night_rate', 'base_price', 'rental_price',
            'accommodation_price', 'stay_price', 'booking_price', 'charges',
            'fee', 'amount', 'pricing', 'tariff', 'rent'
        ]
        
        self.location_fields = [
            'neighbourhood_cleansed', 'neighbourhood', 'neighborhood', 'area', 'district',
            'city', 'town', 'municipality', 'locality', 'place', 'region',
            'location', 'address', 'street', 'suburb', 'zone', 'sector',
            'country', 'state', 'province', 'county', 'neighborhood_overview',
            'government_area', 'market'
        ]
        
        self.amenity_keywords = {
            'wifi': ['wifi', 'internet', 'wireless'],
            'kitchen': ['kitchen', 'cooking', 'stove', 'oven', 'microwave'],
            'parking': ['parking', 'garage', 'car park'],
            'pool': ['pool', 'swimming', 'hot tub', 'jacuzzi'],
            'pet': ['pet', 'dog', 'cat', 'animal']
        }
    
    def extract_price(self, document: Dict[str, Any]) -> tuple[float, str]:
        """Extract price as float and formatted string"""
        # Try exact field matches
        for field in self.price_fields:
            if field in document and document[field] is not None:
                price = self._parse_price_value(document[field])
                if price is not None:
                    return price, f"${price:.0f}/night"
        
        # Try case-insensitive search
        for key, value in document.items():
            if value is not None:
                key_lower = str(key).lower()
                if any(price_term in key_lower for price_term in ['price', 'cost', 'rate', 'fee']):
                    price = self._parse_price_value(value)
                    if price is not None:
                        return price, f"${price:.0f}/night"
        
        # Look for reasonable numeric values
        for key, value in document.items():
            if isinstance(value, (int, float)) and 10 <= value <= 10000:
                return float(value), f"${value:.0f}/night (estimated)"
        
        return None, None
    
    def _parse_price_value(self, value) -> Optional[float]:
        """Parse various price formats"""
        try:
            if isinstance(value, (int, float)):
                return float(value) if value > 0 else None
            elif isinstance(value, str):
                import re
                price_match = re.search(r'[\d,]+\.?\d*', str(value).replace('$', '').replace(',', ''))
                if price_match:
                    return float(price_match.group().replace(',', ''))
        except (ValueError, AttributeError):
            pass
        return None
    
    def extract_location(self, document: Dict[str, Any]) -> tuple[str, str, str]:
        """Extract location as (full_location, neighbourhood, city)"""
        location_parts = []
        neighbourhood = None
        city = None
        
        # Check if address field exists and is structured
        address = document.get('address')
        if isinstance(address, dict):
            # Extract suburb/neighbourhood from address structure
            suburb = address.get('suburb')
            if suburb and str(suburb).strip():
                neighbourhood = str(suburb).strip()
                location_parts.append(neighbourhood)
            
            # Extract market/city from address structure
            market = address.get('market')
            if market and str(market).strip():
                city_val = str(market).strip()
                if not neighbourhood or city_val.lower() != neighbourhood.lower():
                    city = city_val
                    location_parts.append(city)
            
            # Also check government_area if we don't have neighbourhood
            if not neighbourhood:
                gov_area = address.get('government_area')
                if gov_area and str(gov_area).strip():
                    neighbourhood = str(gov_area).strip()
                    location_parts.append(neighbourhood)
            
            # Add country if available
            country = address.get('country')
            if country and str(country).strip():
                country_val = str(country).strip()
                if country_val not in location_parts:
                    location_parts.append(country_val)
        
        # Fallback: Extract neighbourhood from top-level fields
        if not neighbourhood:
            neighbourhood_fields = ['neighbourhood_cleansed', 'neighbourhood', 'neighborhood', 'area', 'district']
            for field in neighbourhood_fields:
                if field in document and document[field] and str(document[field]).strip():
                    neighbourhood = str(document[field]).strip()
                    if neighbourhood not in location_parts:
                        location_parts.append(neighbourhood)
                    break
        
        # Fallback: Extract city from top-level fields
        if not city:
            city_fields = ['city', 'town', 'municipality', 'locality']
            for field in city_fields:
                if field in document and document[field] and str(document[field]).strip():
                    city_val = str(document[field]).strip()
                    if city_val not in location_parts:
                        city = city_val
                        location_parts.append(city)
                    break
        
        # Additional fallback: Check neighborhood_overview for location info
        if not location_parts:
            neighborhood_overview = document.get('neighborhood_overview')
            if neighborhood_overview and str(neighborhood_overview).strip():
                # Extract first sentence or first 50 characters as location hint
                overview_text = str(neighborhood_overview).strip()
                if len(overview_text) < 200:
                    # Extract location mentions from overview
                    import re
                    location_match = re.search(r'([A-Z][a-zA-Z\s]+(?:,\s*[A-Z][a-zA-Z\s]*)*)', overview_text)
                    if location_match:
                        location_parts.append(location_match.group(1).strip())
        
        # Final fallback to any location field
        if not location_parts:
            for field in self.location_fields:
                if field in document and document[field] and str(document[field]).strip():
                    location_val = str(document[field]).strip()
                    if len(location_val) < 100:  # Avoid long descriptions
                        location_parts.append(location_val)
                        break
        
        full_location = ', '.join(location_parts[:3])  # Limit to 3 parts
        return full_location if full_location else None, neighbourhood, city
    
    def extract_amenities(self, document: Dict[str, Any]) -> tuple[List[str], Dict[str, bool]]:
        """Extract amenities list and key amenity flags"""
        amenities = []
        flags = {}
        
        # Check amenities field
        if 'amenities' in document and document['amenities']:
            amenities_raw = document['amenities']
            if isinstance(amenities_raw, str):
                try:
                    amenities_raw = json.loads(amenities_raw)
                except json.JSONDecodeError:
                    amenities_raw = [a.strip() for a in amenities_raw.split(',')]
            
            if isinstance(amenities_raw, list):
                amenities = [str(a).strip() for a in amenities_raw if a]
        
        # Extract key amenity flags
        amenities_text = ' '.join(amenities).lower()
        for amenity, keywords in self.amenity_keywords.items():
            flags[f'has_{amenity}'] = any(keyword in amenities_text for keyword in keywords)
        
        return amenities[:20], flags  # Limit amenities list
    
    def extract_metadata(self, document: Dict[str, Any]) -> PropertyMetadata:
        """Extract comprehensive metadata from document"""
        doc_id = str(document.get('id', document.get('_id', 'unknown')))
        
        # Extract basic fields
        name = document.get('name') or document.get('title') or document.get('property_name')
        property_type = document.get('property_type')
        room_type = document.get('room_type')
        
        # Extract numeric fields safely
        bedrooms = self._safe_int(document.get('bedrooms'))
        bathrooms = self._safe_float(document.get('bathrooms'))
        accommodates = self._safe_int(document.get('accommodates'))
        minimum_nights = self._safe_int(document.get('minimum_nights'))
        availability_365 = self._safe_int(document.get('availability_365'))
        number_of_reviews = self._safe_int(document.get('number_of_reviews'))
        
        # Extract review score
        review_score = self._safe_float(document.get('review_scores_rating')) or \
                      self._safe_float(document.get('rating'))
        
        # Extract boolean fields
        instant_bookable = self._safe_bool(document.get('instant_bookable'))
        
        # Extract host info
        host_name = document.get('host_name')
        
        # Extract price and location
        price, price_formatted = self.extract_price(document)
        full_location, neighbourhood, city = self.extract_location(document)
        
        # Extract amenities
        amenities, amenity_flags = self.extract_amenities(document)
        
        # Calculate derived fields
        price_per_person = None
        if price and accommodates and accommodates > 0:
            price_per_person = price / accommodates
        
        # Calculate value score (simple heuristic)
        value_score = self._calculate_value_score(price, review_score, len(amenities))
        
        return PropertyMetadata(
            id=doc_id,
            name=name,
            price=price,
            price_formatted=price_formatted,
            location=full_location,
            neighbourhood=neighbourhood,
            city=city,
            property_type=property_type,
            room_type=room_type,
            bedrooms=bedrooms,
            bathrooms=bathrooms,
            accommodates=accommodates,
            amenities=amenities,
            host_name=host_name,
            review_score=review_score,
            number_of_reviews=number_of_reviews,
            instant_bookable=instant_bookable,
            minimum_nights=minimum_nights,
            availability_365=availability_365,
            price_per_person=price_per_person,
            value_score=value_score,
            **amenity_flags
        )
    
    def _safe_int(self, value) -> Optional[int]:
        """Safely convert to int"""
        try:
            if value is not None and str(value).replace('.', '').isdigit():
                return int(float(value))
        except (ValueError, TypeError):
            pass
        return None
    
    def _safe_float(self, value) -> Optional[float]:
        """Safely convert to float"""
        try:
            if value is not None:
                return float(value)
        except (ValueError, TypeError):
            pass
        return None
    
    def _safe_bool(self, value) -> Optional[bool]:
        """Safely convert to bool"""
        if isinstance(value, bool):
            return value
        elif isinstance(value, str):
            return value.lower() in ['true', 't', 'yes', '1']
        elif isinstance(value, int):
            return bool(value)
        return None
    
    def _calculate_value_score(self, price: Optional[float], review_score: Optional[float], 
                              amenity_count: int) -> Optional[float]:
        """Calculate a simple value score"""
        if not price:
            return None
        
        score = 0.0
        
        # Price component (inverse - lower price = higher score)
        if price > 0:
            price_score = max(0, 100 - (price / 10))  # Rough heuristic
            score += price_score * 0.4
        
        # Review score component
        if review_score:
            if review_score > 10:  # Scale from 100
                score += (review_score / 100) * 40
            else:  # Scale from 5
                score += (review_score / 5) * 40
        
        # Amenity count component
        amenity_score = min(20, amenity_count)  # Cap at 20
        score += amenity_score
        
        return round(score, 2)

class MetadataStore:
    """Stores and retrieves preprocessed metadata"""
    
    def __init__(self, storage_dir: str = "preprocessed_data"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        
        self.metadata_file = self.storage_dir / "property_metadata.pkl"
        self.index_file = self.storage_dir / "metadata_index.pkl"
        self.stats_file = self.storage_dir / "preprocessing_stats.json"
        
        self.metadata: Dict[str, PropertyMetadata] = {}
        self.price_index: Dict[str, List[str]] = {}  # price_range -> [doc_ids]
        self.location_index: Dict[str, List[str]] = {}  # location -> [doc_ids]
        self.type_index: Dict[str, List[str]] = {}  # property_type -> [doc_ids]
    
    def store_metadata(self, metadata_list: List[PropertyMetadata]):
        """Store metadata and build indexes"""
        logger.info(f"Storing metadata for {len(metadata_list)} properties")
        
        self.metadata = {meta.id: meta for meta in metadata_list}
        self._build_indexes()
        self._save_to_disk()
        
        stats = {
            'total_properties': len(metadata_list),
            'properties_with_price': len([m for m in metadata_list if m.price]),
            'properties_with_location': len([m for m in metadata_list if m.location]),
            'avg_price': self._calculate_avg_price(metadata_list),
            'preprocessing_time': datetime.now().isoformat()
        }
        
        with open(self.stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Metadata stored with stats: {stats}")
    
    def _build_indexes(self):
        """Build search indexes for fast retrieval"""
        self.price_index.clear()
        self.location_index.clear()
        self.type_index.clear()
        
        for doc_id, meta in self.metadata.items():
            # Price index
            if meta.price:
                price_range = self._get_price_range(meta.price)
                self.price_index.setdefault(price_range, []).append(doc_id)
            
            # Location index
            if meta.location:
                location_key = meta.location.lower()
                self.location_index.setdefault(location_key, []).append(doc_id)
                
                # Also index by city and neighbourhood
                if meta.city:
                    city_key = meta.city.lower()
                    self.location_index.setdefault(city_key, []).append(doc_id)
                if meta.neighbourhood:
                    neighbourhood_key = meta.neighbourhood.lower()
                    self.location_index.setdefault(neighbourhood_key, []).append(doc_id)
            
            # Property type index
            if meta.property_type:
                type_key = meta.property_type.lower()
                self.type_index.setdefault(type_key, []).append(doc_id)
    
    def _get_price_range(self, price: float) -> str:
        """Get price range string for indexing"""
        if price < 50:
            return "0-50"
        elif price < 100:
            return "50-100"
        elif price < 200:
            return "100-200"
        elif price < 300:
            return "200-300"
        elif price < 500:
            return "300-500"
        else:
            return "500+"
    
    def _calculate_avg_price(self, metadata_list: List[PropertyMetadata]) -> Optional[float]:
        """Calculate average price"""
        prices = [m.price for m in metadata_list if m.price]
        return sum(prices) / len(prices) if prices else None
    
    def _save_to_disk(self):
        """Save metadata and indexes to disk"""
        with open(self.metadata_file, 'wb') as f:
            pickle.dump(self.metadata, f)
        
        indexes = {
            'price_index': self.price_index,
            'location_index': self.location_index,
            'type_index': self.type_index
        }
        
        with open(self.index_file, 'wb') as f:
            pickle.dump(indexes, f)
    
    def load_metadata(self) -> bool:
        """Load metadata and indexes from disk"""
        try:
            if self.metadata_file.exists():
                with open(self.metadata_file, 'rb') as f:
                    self.metadata = pickle.load(f)
            
            if self.index_file.exists():
                with open(self.index_file, 'rb') as f:
                    indexes = pickle.load(f)
                    self.price_index = indexes.get('price_index', {})
                    self.location_index = indexes.get('location_index', {})
                    self.type_index = indexes.get('type_index', {})
            
            logger.info(f"Loaded metadata for {len(self.metadata)} properties")
            return len(self.metadata) > 0
            
        except Exception as e:
            logger.error(f"Error loading metadata: {e}")
            return False
    
    def get_metadata(self, doc_id: str) -> Optional[PropertyMetadata]:
        """Get metadata for a specific document"""
        return self.metadata.get(doc_id)
    
    def search_by_price(self, min_price: float, max_price: float) -> List[str]:
        """Get document IDs within price range"""
        doc_ids = set()
        for price_range, ids in self.price_index.items():
            range_parts = price_range.split('-')
            if len(range_parts) == 2:
                range_min = float(range_parts[0])
                range_max = float(range_parts[1]) if range_parts[1] != '+' else float('inf')
                if not (max_price < range_min or min_price > range_max):
                    doc_ids.update(ids)
        return list(doc_ids)
    
    def search_by_location(self, location: str) -> List[str]:
        """Get document IDs for location"""
        location_key = location.lower()
        return self.location_index.get(location_key, [])
    
    def get_stats(self) -> Dict[str, Any]:
        """Get preprocessing statistics"""
        try:
            if self.stats_file.exists():
                with open(self.stats_file, 'r') as f:
                    return json.load(f)
        except Exception:
            pass
        return {}

class DataPreprocessor:
    """Main preprocessor class"""
    
    def __init__(self, storage_dir: str = "preprocessed_data"):
        self.extractor = FieldExtractor()
        self.store = MetadataStore(storage_dir)
        
    def preprocess_documents(self, documents: List[Dict[str, Any]], 
                           batch_size: int = 100, max_workers: int = 4) -> bool:
        """Preprocess documents in parallel batches"""
        logger.info(f"Starting preprocessing of {len(documents)} documents")
        start_time = time.time()
        
        def process_batch(batch):
            return [self.extractor.extract_metadata(doc) for doc in batch]
        
        all_metadata = []
        
        # Process in parallel batches
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            batches = [documents[i:i+batch_size] for i in range(0, len(documents), batch_size)]
            
            futures = {executor.submit(process_batch, batch): i for i, batch in enumerate(batches)}
            
            for future in as_completed(futures):
                batch_idx = futures[future]
                try:
                    batch_metadata = future.result()
                    all_metadata.extend(batch_metadata)
                    logger.info(f"Completed batch {batch_idx + 1}/{len(batches)}")
                except Exception as e:
                    logger.error(f"Error processing batch {batch_idx}: {e}")
        
        # Store results
        self.store.store_metadata(all_metadata)
        
        processing_time = time.time() - start_time
        logger.info(f"Preprocessing completed in {processing_time:.2f}s")
        
        return True
    
    def load_preprocessed_data(self) -> bool:
        """Load preprocessed data from disk"""
        return self.store.load_metadata()
    
    def get_property_metadata(self, doc_id: str) -> Optional[PropertyMetadata]:
        """Get metadata for a property"""
        return self.store.get_metadata(doc_id)
    
    def search_properties(self, **kwargs) -> List[str]:
        """Search properties by metadata"""
        doc_ids = set()
        
        if 'min_price' in kwargs or 'max_price' in kwargs:
            min_price = kwargs.get('min_price', 0)
            max_price = kwargs.get('max_price', float('inf'))
            price_ids = self.store.search_by_price(min_price, max_price)
            if not doc_ids:
                doc_ids = set(price_ids)
            else:
                doc_ids &= set(price_ids)
        
        if 'location' in kwargs:
            location_ids = self.store.search_by_location(kwargs['location'])
            if not doc_ids:
                doc_ids = set(location_ids)
            else:
                doc_ids &= set(location_ids)
        
        return list(doc_ids)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get preprocessing statistics"""
        return self.store.get_stats()

def preprocess_from_mongodb():
    """Standalone function to preprocess data from MongoDB"""
    try:
        from utils import IndexManager
        
        # Get documents from MongoDB
        index_manager = IndexManager()
        documents = index_manager.db_connector.get_all_documents()
        
        if not documents:
            logger.error("No documents found in MongoDB")
            return False
        
        # Preprocess
        preprocessor = DataPreprocessor()
        return preprocessor.preprocess_documents(documents)
        
    except Exception as e:
        logger.error(f"Error in MongoDB preprocessing: {e}")
        return False

if __name__ == "__main__":
    # Run preprocessing if called directly
    preprocess_from_mongodb()
