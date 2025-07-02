#!/usr/bin/env python3
"""
Context Manager for RAG-based Chatbot System
Enables multi-turn, context-aware conversation handling with intelligent
follow-up intent recognition, comparison, filtering, and reasoning capabilities.
"""

import json
import logging
import uuid
import re
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
from copy import deepcopy

# Core system imports
from logging_config import StructuredLogger
from utils import SessionManager, ConversationTurn

# Import Django components for enhanced integration
try:
    import django
    from django.conf import settings
    DJANGO_AVAILABLE = True
except ImportError:
    DJANGO_AVAILABLE = False

# Set up logging
logger = StructuredLogger(__name__, {'component': 'context_manager'})


@dataclass
class ContextualAttribute:
    """Represents a contextual attribute extracted from queries or responses"""
    name: str
    value: Any
    source: str  # 'query', 'response', 'inferred'
    timestamp: datetime
    confidence: float = 1.0
    data_type: str = 'string'  # 'string', 'numeric', 'list', 'boolean'
    field_mappings: List[str] = field(default_factory=list)


@dataclass
class CachedResult:
    """Represents cached results from previous queries"""
    query_id: str
    original_query: str
    results: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    timestamp: datetime
    result_count: int
    similarity_scores: List[float] = field(default_factory=list)
    extracted_attributes: Dict[str, ContextualAttribute] = field(default_factory=dict)


@dataclass
class FollowUpIntent:
    """Represents detected follow-up intent"""
    intent_type: str  # 'comparison', 'filtering', 'summarization', 'reasoning'
    confidence: float
    target_entities: List[str]
    operation: str  # 'compare_A_B', 'filter_by_X', 'summarize_all', 'explain_why'
    context_references: List[str]  # References to previous results
    original_query: str = ""  # Store the original query for processing


class IntentClassifier:
    """Classifies follow-up intents in multi-turn conversations"""
    
    def __init__(self):
        self.logger = StructuredLogger(__name__, {'class_name': self.__class__.__name__})
        self.intent_patterns = self._build_intent_patterns()
        
    def _build_intent_patterns(self) -> Dict[str, List[str]]:
        """Build patterns for intent classification"""
        return {
            'comparison': [
                r'\bcompare\b.*\band\b',
                r'\bdifference\s+between\b',
                r'\bversus\b|\bvs\.?\b',
                r'\bwhich\s+is\s+better\b',
                r'\bhow\s+does\s+.+\s+compare\b',
                r'\b[a-z]+\s+vs\.?\s+[a-z]+\b',
                r'\bA\s+and\s+C\b|\blisting\s+[a-z]\s+and\s+[a-z]\b'
            ],
            'filtering': [
                r'\bonly\s+(with|having|that\s+have)\b',
                r'\bfilter\s+by\b|\bfiltered\s+by\b',
                r'\bmust\s+have\b|\brequired\b',
                r'\bexclude\b|\bwithout\b',
                r'\bthat\s+(include|have|contain)\b',
                r'\brefine\s+(by|to|with)\b',
                r'\bnarrow\s+down\b'
            ],
            'summarization': [
                r'\bsummarize\b|\bsummary\b',
                r'\boverview\s+of\b',
                r'\bshow\s+me\s+all\b',
                r'\blist\s+all\b',
                r'\btell\s+me\s+about\s+all\b',
                r'\bbreakdown\s+of\b'
            ],
            'reasoning': [
                r'\bwhy\s+(is|are|should)\b',
                r'\bexplain\b|\bexplanation\b',
                r'\bhow\s+come\b',
                r'\bwhat\s+makes\b.*\bbetter\b',
                r'\breason\s+(for|why)\b',
                r'\bjustify\b|\bjustification\b'
            ],
            'reference': [
                r'\b(it|them|those|these|that)\b',
                r'\bthe\s+(previous|last|above)\b',
                r'\bfrom\s+before\b',
                r'\bearlier\s+(results|listings)\b',
                r'\bthose\s+(properties|listings)\b'
            ]
        }
    
    def classify_intent(self, query: str, conversation_history: List[ConversationTurn]) -> FollowUpIntent:
        """Classify the intent of a follow-up query"""
        query_lower = query.lower()
        intent_scores = {}
        
        # Check for intent patterns
        for intent_type, patterns in self.intent_patterns.items():
            score = 0
            matched_patterns = 0
            
            for pattern in patterns:
                if re.search(pattern, query_lower, re.IGNORECASE):
                    score += 1
                    matched_patterns += 1
            
            if matched_patterns > 0:
                intent_scores[intent_type] = score / len(patterns)
        
        # Determine primary intent
        if not intent_scores:
            return FollowUpIntent('search', 0.5, [], 'new_search', [], query)
        
        primary_intent = max(intent_scores, key=intent_scores.get)
        confidence = intent_scores[primary_intent]
        
        # Extract target entities and operations
        target_entities = self._extract_target_entities(query, conversation_history)
        operation = self._determine_operation(query, primary_intent, target_entities)
        context_refs = self._find_context_references(query, conversation_history)
        
        return FollowUpIntent(
            intent_type=primary_intent,
            confidence=confidence,
            target_entities=target_entities,
            operation=operation,
            context_references=context_refs,
            original_query=query
        )
    
    def _extract_target_entities(self, query: str, history: List[ConversationTurn]) -> List[str]:
        """Extract entities that are targets of the follow-up query"""
        entities = []
        
        # Look for explicit references like "A and B", "listing C", etc.
        entity_patterns = [
            r'\b([A-Z])\s+and\s+([A-Z])\b',  # "A and B"
            r'\blisting\s+([A-Z])\b',        # "listing A"
            r'\bproperty\s+([A-Z])\b',       # "property B"
            r'\boption\s+([A-Z])\b',        # "option C"
            r'\b([A-Z]),\s+([A-Z])\b'       # "A, B"
        ]
        
        for pattern in entity_patterns:
            matches = re.finditer(pattern, query, re.IGNORECASE)
            for match in matches:
                entities.extend([group.upper() for group in match.groups() if group])
        
        # Look for numbered references
        number_patterns = [
            r'\b(first|second|third|fourth|fifth)\b',
            r'\b(\d+)(?:st|nd|rd|th)?\s+(?:option|property|listing)\b',
            r'\bnumber\s+(\d+)\b'
        ]
        
        for pattern in number_patterns:
            matches = re.finditer(pattern, query, re.IGNORECASE)
            for match in matches:
                entities.append(match.group(1))
        
        return list(set(entities))
    
    def _determine_operation(self, query: str, intent: str, entities: List[str]) -> str:
        """Determine the specific operation to perform"""
        query_lower = query.lower()
        
        if intent == 'comparison':
            if len(entities) >= 2:
                return f"compare_{entities[0]}_{entities[1]}"
            else:
                return "compare_all"
        elif intent == 'filtering':
            # Extract what to filter by
            filter_terms = []
            filter_patterns = [
                r'\bwith\s+([\w\s]+?)(?:\s+and|$)',
                r'\bhaving\s+([\w\s]+?)(?:\s+and|$)',
                r'\bthat\s+have\s+([\w\s]+?)(?:\s+and|$)'
            ]
            
            for pattern in filter_patterns:
                matches = re.finditer(pattern, query_lower)
                for match in matches:
                    filter_terms.append(match.group(1).strip())
            
            if filter_terms:
                return f"filter_by_{'_'.join(filter_terms)}"
            else:
                return "filter_generic"
        elif intent == 'summarization':
            return "summarize_all"
        elif intent == 'reasoning':
            return "explain_why"
        else:
            return "process_request"
    
    def _find_context_references(self, query: str, history: List[ConversationTurn]) -> List[str]:
        """Find references to previous conversation context"""
        references = []
        
        # Look for temporal references
        temporal_patterns = [
            r'\bprevious\s+(results?|search|query)\b',
            r'\blast\s+(search|query|results?)\b',
            r'\bbefore\b',
            r'\bearlier\b'
        ]
        
        for pattern in temporal_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                references.append('previous_results')
                break
        
        # Look for pronoun references
        pronoun_patterns = ['them', 'those', 'these', 'it']
        for pronoun in pronoun_patterns:
            if re.search(r'\b' + pronoun + r'\b', query, re.IGNORECASE):
                references.append('pronoun_reference')
                break
        
        return references


class ComparisonProcessor:
    """
    Processes comparison requests for properties/items from cached results
    """
    
    def __init__(self):
        self.comparison_keywords = [
            'compare', 'comparison', 'vs', 'versus', 'difference', 'differences',
            'similar', 'similarities', 'contrast', 'against', 'between'
        ]
        
    def extract_item_references(self, query: str, cached_results: List[Dict]) -> List[Dict]:
        """
        Extract specific items/properties referenced in comparison query
        """
        referenced_items = []
        query_lower = query.lower()
        
        # Pattern matching for property references
        import re
        
        # Look for patterns like "property 1", "apartment A", "listing 3", "first one", etc.
        patterns = [
            r'property\s+(\d+|[a-z])',
            r'apartment\s+(\d+|[a-z])',
            r'listing\s+(\d+|[a-z])',
            r'option\s+(\d+|[a-z])',
            r'item\s+(\d+|[a-z])',
            r'(first|second|third|fourth|fifth|last)\s*(one|property|apartment|listing)?',
            r'(\d+)(st|nd|rd|th)\s*(one|property|apartment|listing)?',
            r'([a-z])\s*and\s*([a-z])'
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, query_lower, re.IGNORECASE)
            for match in matches:
                ref = match.group(1) if match.groups() else match.group(0)
                
                # Convert reference to index
                index = self._convert_reference_to_index(ref, len(cached_results))
                if 0 <= index < len(cached_results):
                    if cached_results[index] not in referenced_items:
                        referenced_items.append(cached_results[index])
        
        # If no specific references found, try to extract by name mentions
        if not referenced_items:
            for result in cached_results:
                name = result.get('name', '').lower()
                if name and name in query_lower:
                    referenced_items.append(result)
        
        # Default to first two if no specific references and it's clearly a comparison
        if not referenced_items and any(keyword in query_lower for keyword in self.comparison_keywords):
            referenced_items = cached_results[:2]
            
        return referenced_items[:5]  # Limit to 5 items for comparison
    
    def _convert_reference_to_index(self, ref: str, total_count: int) -> int:
        """
        Convert various reference formats to array index
        """
        ref = ref.strip().lower()
        
        # Handle numeric references
        if ref.isdigit():
            return int(ref) - 1  # Convert 1-based to 0-based
            
        # Handle letter references (a, b, c, etc.)
        if len(ref) == 1 and ref.isalpha():
            return ord(ref) - ord('a')
            
        # Handle ordinal words
        ordinal_map = {
            'first': 0, 'second': 1, 'third': 2, 'fourth': 3, 'fifth': 4,
            'last': total_count - 1
        }
        
        return ordinal_map.get(ref, -1)
    
    def generate_comparison_table(self, items: List[Dict], comparison_attributes: List[str] = None) -> Dict:
        """
        Generate a tabular comparison of items with key similarities and differences
        """
        if not items or len(items) < 2:
            return {
                'error': 'Need at least 2 items for comparison',
                'items_provided': len(items) if items else 0
            }
        
        # Auto-detect comparison attributes if not provided
        if not comparison_attributes:
            comparison_attributes = self._detect_comparison_attributes(items)
        
        # Build comparison table with enhanced formatting
        comparison_table = {
            'items_compared': len(items),
            'comparison_attributes': comparison_attributes,
            'table_format': 'enhanced',
            'ui_support': True,
            'table_data': [],
            'similarities': [],
            'key_differences': []
        }
        
        # Create enhanced table headers
        headers = ['Attribute'] + [self._generate_item_label(item, i) for i, item in enumerate(items)]
        comparison_table['headers'] = headers
        
        # Build enhanced comparison rows
        for attr in comparison_attributes:
            row_data = {
                'attribute': attr,
                'attribute_display': attr.replace('_', ' ').title(),
                'values': {},
                'raw_values': [],
                'formatted_values': [],
                'has_differences': False
            }
            
            values = []
            for i, item in enumerate(items):
                raw_value = self._extract_attribute_value(item, attr)
                formatted_value = self._format_comparison_value(attr, raw_value)
                
                item_label = self._generate_item_label(item, i)
                row_data['values'][item_label] = {
                    'raw': raw_value,
                    'formatted': formatted_value,
                    'display_type': self._get_display_type(attr, raw_value)
                }
                
                row_data['raw_values'].append(raw_value)
                row_data['formatted_values'].append(formatted_value)
                values.append(formatted_value)
            
            # Traditional row format for backward compatibility
            traditional_row = [row_data['attribute_display']] + values
            row_data['traditional_format'] = traditional_row
            
            comparison_table['table_data'].append(row_data)
            
            # Check for similarities and differences
            unique_values = list(set(str(v) for v in row_data['raw_values'] if v is not None))
            
            if len(unique_values) == 1 and unique_values[0] != 'N/A':
                row_data['has_differences'] = False
            elif len(unique_values) > 1:
                row_data['has_differences'] = True
            
            # Use formatted values for display
            unique_values = list(set(str(v) for v in values if v is not None))
            if len(unique_values) == 1 and unique_values[0] != 'N/A':
                comparison_table['similarities'].append({
                    'attribute': attr,
                    'common_value': unique_values[0]
                })
            elif len(unique_values) > 1:
                comparison_table['key_differences'].append({
                    'attribute': attr,
                    'values': {item.get('name', f"Item {i+1}"): values[i] for i, item in enumerate(items)}
                })
        
        # Generate summary insights
        comparison_table['summary'] = self._generate_comparison_summary(
            items, comparison_table['similarities'], comparison_table['key_differences']
        )
        
        return comparison_table
    
    def _detect_comparison_attributes(self, items: List[Dict]) -> List[str]:
        """
        Automatically detect which attributes are relevant for comparison
        """
        # Core fields that should always be shown regardless of query
        core_fields = [
            'name', 'title', 'property_name',
            'price', 'cost', 'rent', 'monthly_rent', 'nightly_rate',
            'location', 'address', 'neighborhood', 'area', 'city',
            'bedrooms', 'beds', 'bathrooms'
        ]
        
        # Extended attributes for detailed comparison
        extended_attributes = [
            'rating', 'reviews', 'review_count', 'stars',
            'accommodates', 'guests', 'capacity',
            'size', 'sqft', 'square_feet',
            'pet_friendly', 'pets_allowed', 'amenities',
            'parking', 'garage', 'balcony', 'pool',
            'wifi', 'internet', 'utilities_included',
            'lease_term', 'availability', 'move_in_date',
            'host_name', 'instant_book', 'minimum_nights'
        ]
        
        # Get all available attributes from items
        all_attributes = set()
        if items:  # Check if items is not None
            for item in items:
                if isinstance(item, dict):
                    all_attributes.update(item.keys())
        
        # Filter to relevant comparison attributes
        comparison_attributes = []
        
        # Always add core fields that exist (in order)
        for attr in core_fields:
            if attr in all_attributes:
                comparison_attributes.append(attr)
        
        # Add extended attributes that exist
        for attr in extended_attributes:
            if attr in all_attributes and attr not in comparison_attributes:
                comparison_attributes.append(attr)
        
        # Add other relevant attributes (non-system fields)
        for attr in sorted(all_attributes):
            if (attr not in comparison_attributes and 
                not attr.startswith('_') and 
                attr not in ['id', 'created_at', 'updated_at', 'timestamp', 'metadata']):
                comparison_attributes.append(attr)
        
        return comparison_attributes[:10]  # Limit to top 10 attributes
    
    def _extract_attribute_value(self, item: Dict, attribute: str) -> str:
        """
        Extract and format attribute value from item
        """
        value = item.get(attribute)
        
        # If not found at top level, check mandatory_fields
        if value is None and 'mandatory_fields' in item:
            value = item['mandatory_fields'].get(attribute)
        
        # If still not found, check source_data or document
        if value is None:
            source_data = item.get('source_data') or item.get('document') or item.get('source_json', {})
            if isinstance(source_data, dict):
                value = source_data.get(attribute)
        
        if value is None:
            return 'N/A'
        
        # Format different types of values
        if isinstance(value, bool):
            return 'Yes' if value else 'No'
        elif isinstance(value, (int, float)):
            if attribute in ['price', 'rent', 'cost', 'monthly_rent']:
                return f"${value:,.2f}" if isinstance(value, float) else f"${value:,}"
            elif attribute in ['rating', 'stars']:
                return f"{value}⭐" if isinstance(value, (int, float)) else str(value)
            else:
                return str(value)
        elif isinstance(value, list):
            return ', '.join(str(v) for v in value[:3])  # Limit list items
        else:
            return str(value)

    
    def _generate_comparison_summary(self, items: List[Dict], similarities: List[Dict], differences: List[Dict]) -> Dict:
        """
        Generate a summary of the comparison highlighting key insights
        """
        summary = {
            'total_similarities': len(similarities),
            'total_differences': len(differences),
            'recommendation': None,
            'key_insights': []
        }
        
        # Generate insights based on differences
        if differences:
            # Price comparison insight
            price_diff = next((d for d in differences if 'price' in d['attribute'].lower()), None)
            if price_diff:
                prices = [v for v in price_diff['values'].values() if v != 'N/A']
                if len(prices) >= 2:
                    summary['key_insights'].append(f"Price range varies from {min(prices)} to {max(prices)}")
            
            # Rating comparison insight
            rating_diff = next((d for d in differences if 'rating' in d['attribute'].lower()), None)
            if rating_diff:
                summary['key_insights'].append("Items have different user ratings - consider reviews when choosing")
            
            # Location insight
            location_diff = next((d for d in differences if 'location' in d['attribute'].lower()), None)
            if location_diff:
                summary['key_insights'].append("Different locations - consider proximity to work/amenities")
        
        # Commonalities insight
        if similarities:
            common_features = [s['attribute'] for s in similarities[:3]]
            summary['key_insights'].append(f"All options share: {', '.join(common_features)}")
        
        return summary
    
    def _generate_item_label(self, item: Dict, index: int) -> str:
        """
        Generate a descriptive label for an item in comparisons
        """
        # Try to find a meaningful name
        name_fields = ['name', 'title', 'property_name', 'listing_title']
        for field in name_fields:
            if field in item and item[field]:
                name = str(item[field])[:30]  # Truncate long names
                if len(str(item[field])) > 30:
                    name += "..."
                return name
        
        # Fall back to location + type info
        location = item.get('neighborhood', item.get('location', item.get('city', '')))
        if location:
            return f"{chr(65 + index)}: {location[:20]}{'...' if len(location) > 20 else ''}"
        
        # Final fallback
        return f"Property {chr(65 + index)}"
    
    def _format_comparison_value(self, attribute: str, value: Any) -> str:
        """
        Format values for better display in comparisons
        """
        if value is None or (isinstance(value, str) and value.lower() in ['none', 'null', '']):
            return 'N/A'
        
        # Price formatting
        if any(price_term in attribute.lower() for price_term in ['price', 'cost', 'rent', 'rate']):
            try:
                num_value = float(str(value).replace('$', '').replace(',', ''))
                return f"${num_value:,.0f}"
            except (ValueError, TypeError):
                return str(value)
        
        # Rating formatting
        if any(rating_term in attribute.lower() for rating_term in ['rating', 'star', 'review']):
            try:
                num_value = float(value)
                if 'count' in attribute.lower():
                    return f"{int(num_value)} reviews"
                else:
                    return f"{num_value:.1f}/5.0 ★"
            except (ValueError, TypeError):
                return str(value)
        
        # Boolean formatting
        if isinstance(value, bool):
            return "✓ Yes" if value else "✗ No"
        
        # List formatting (like amenities)
        if isinstance(value, list):
            if len(value) == 0:
                return "None"
            elif len(value) <= 3:
                return ", ".join(str(v) for v in value)
            else:
                return f"{', '.join(str(v) for v in value[:3])} +{len(value)-3} more"
        
        # Capacity/bedroom formatting
        if any(capacity_term in attribute.lower() for capacity_term in ['bedroom', 'bed', 'bath', 'guest', 'accommodate']):
            try:
                num_value = int(float(str(value)))
                if 'bedroom' in attribute.lower() or 'bed' in attribute.lower():
                    return f"{num_value} bed{'s' if num_value != 1 else ''}"
                elif 'bath' in attribute.lower():
                    return f"{num_value} bath{'s' if num_value != 1 else ''}"
                else:
                    return f"{num_value} guests"
            except (ValueError, TypeError):
                return str(value)
        
        # Default string formatting
        return str(value)[:50] + ('...' if len(str(value)) > 50 else '')
    
    def _get_display_type(self, attribute: str, value: Any) -> str:
        """
        Determine the display type for UI rendering
        """
        if value is None:
            return 'text'
    
    def _format_results_for_display(self, results: List[Dict], intent: 'FollowUpIntent', 
                                   attributes: Dict[str, 'ContextualAttribute']) -> Dict[str, Any]:
        """
        Format results with enhanced display information for UI
        """
        if not results:
            return {'formatted_items': [], 'display_type': 'list'}
        
        formatted_items = []
        
        for i, item in enumerate(results):
            formatted_item = {
                'index': i,
                'item_label': self._generate_item_label(item, i),
                'core_info': self._extract_core_info(item),
                'detailed_info': self._extract_detailed_info(item),
                'formatted_source': self._format_source_json(item),
                'display_priority': self._calculate_display_priority(item, attributes)
            }
            formatted_items.append(formatted_item)
        
        return {
            'formatted_items': formatted_items,
            'display_type': 'comparison_table' if intent.intent_type == 'comparison' else 'list',
            'total_items': len(results),
            'ui_enhancements': {
                'sortable': True,
                'filterable': True,
                'expandable_details': True
            }
        }
    
    def _extract_core_info(self, item: Dict) -> Dict[str, str]:
        """
        Extract core information that should always be displayed
        """
        core_info = {}
        
        # Name/Title
        name_fields = ['name', 'title', 'property_name', 'listing_title']
        for field in name_fields:
            if field in item and item[field]:
                core_info['name'] = str(item[field])
                break
        
        # Price
        price_fields = ['price', 'nightly_rate', 'monthly_rent', 'cost']
        for field in price_fields:
            if field in item and item[field] is not None:
                try:
                    price = float(str(item[field]).replace('$', '').replace(',', ''))
                    core_info['price'] = f"${price:,.0f}"
                except (ValueError, TypeError):
                    core_info['price'] = str(item[field])
                break
        
        # Location
        location_fields = ['neighborhood', 'city', 'location', 'address']
        for field in location_fields:
            if field in item and item[field]:
                core_info['location'] = str(item[field])
                break
        
        # Capacity
        capacity_fields = ['bedrooms', 'beds', 'accommodates']
        for field in capacity_fields:
            if field in item and item[field] is not None:
                try:
                    num = int(float(str(item[field])))
                    if field in ['bedrooms', 'beds']:
                        core_info['bedrooms'] = f"{num} bed{'s' if num != 1 else ''}"
                    else:
                        core_info['accommodates'] = f"{num} guests"
                except (ValueError, TypeError):
                    pass
        
        # Bathrooms
        if 'bathrooms' in item and item['bathrooms'] is not None:
            try:
                num = int(float(str(item['bathrooms'])))
                core_info['bathrooms'] = f"{num} bath{'s' if num != 1 else ''}"
            except (ValueError, TypeError):
                pass
        
        return core_info
    
    def _extract_detailed_info(self, item: Dict) -> Dict[str, Any]:
        """
        Extract detailed information for expandable views
        """
        detailed_info = {}
        
        # Rating information
        if 'rating' in item or 'review_scores_rating' in item:
            rating = item.get('rating', item.get('review_scores_rating'))
            review_count = item.get('number_of_reviews', item.get('review_count', 0))
            
            if rating:
                try:
                    rating_val = float(rating)
                    detailed_info['rating'] = {
                        'score': f"{rating_val:.1f}/5.0",
                        'stars': '★' * int(rating_val),
                        'review_count': f"{review_count} reviews" if review_count else "No reviews"
                    }
                except (ValueError, TypeError):
                    pass
        
        # Amenities
        if 'amenities' in item and isinstance(item['amenities'], list):
            detailed_info['amenities'] = {
                'count': len(item['amenities']),
                'top_5': item['amenities'][:5],
                'has_wifi': any('wifi' in str(amenity).lower() for amenity in item['amenities']),
                'has_parking': any('parking' in str(amenity).lower() for amenity in item['amenities']),
                'pet_friendly': any('pet' in str(amenity).lower() for amenity in item['amenities'])
            }
        
        # Host information
        if 'host_name' in item:
            detailed_info['host'] = {
                'name': item['host_name'],
                'is_superhost': item.get('host_is_superhost', False),
                'response_rate': item.get('host_response_rate', 'N/A')
            }
        
        return detailed_info
    
    def _format_source_json(self, item: Dict) -> Dict[str, Any]:
        """
        Format the source JSON data for better display
        """
        # Create a cleaned version of the source data
        formatted_source = {}
        
        # Organize fields by category
        categories = {
            'Basic Info': ['id', 'name', 'title', 'description'],
            'Location': ['neighborhood', 'city', 'country', 'latitude', 'longitude'],
            'Pricing': ['price', 'nightly_rate', 'monthly_rent', 'cleaning_fee'],
            'Capacity': ['accommodates', 'bedrooms', 'beds', 'bathrooms'],
            'Property Details': ['property_type', 'room_type', 'minimum_nights', 'maximum_nights'],
            'Host Info': ['host_name', 'host_since', 'host_response_rate', 'host_is_superhost'],
            'Reviews': ['rating', 'review_scores_rating', 'number_of_reviews'],
            'Amenities': ['amenities'],
            'Other': []
        }
        
        # Categorize fields
        categorized_data = {cat: {} for cat in categories}
        
        for key, value in item.items():
            placed = False
            for category, fields in categories.items():
                if key in fields:
                    categorized_data[category][key] = self._format_json_value(value)
                    placed = True
                    break
            
            if not placed and not key.startswith('_'):
                categorized_data['Other'][key] = self._format_json_value(value)
        
        # Remove empty categories
        formatted_source = {k: v for k, v in categorized_data.items() if v}
        
        return formatted_source
    
    def _format_json_value(self, value: Any) -> Any:
        """
        Format individual JSON values for better display
        """
        if isinstance(value, str) and len(value) > 100:
            return value[:100] + "..."
        elif isinstance(value, list) and len(value) > 10:
            return value[:10] + [f"... {len(value) - 10} more items"]
        elif isinstance(value, bool):
            return "Yes" if value else "No"
        elif value is None:
            return "N/A"
        else:
            return value
    
    def _calculate_display_priority(self, item: Dict, 
                                  attributes: Dict[str, 'ContextualAttribute']) -> int:
        """
        Calculate display priority based on query relevance
        """
        priority = 0
        
        # Higher priority for items matching query attributes
        for attr_name, attr in attributes.items():
            if attr_name in item and item[attr_name] is not None:
                priority += 10
        
        # Higher priority for items with complete core info
        core_fields = ['name', 'price', 'location', 'bedrooms']
        for field in core_fields:
            if any(field in item for field in [field, f'{field}s', f'nightly_{field}']):
                priority += 5
        
        return priority
        
        if any(price_term in attribute.lower() for price_term in ['price', 'cost', 'rent', 'rate']):
            return 'currency'
        
        if any(rating_term in attribute.lower() for rating_term in ['rating', 'star']):
            return 'rating'
        
        if isinstance(value, bool):
            return 'boolean'
        
        if isinstance(value, list):
            return 'list'
        
        if isinstance(value, (int, float)):
            return 'numeric'
        
        return 'text'


class ContextualAttributeExtractor:
    """Extracts and manages contextual attributes from queries and responses"""
    
    def __init__(self):
        self.logger = StructuredLogger(__name__, {'class_name': self.__class__.__name__})
        self.attribute_patterns = self._build_attribute_patterns()
        
    def _build_attribute_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Build patterns for extracting contextual attributes"""
        return {
            'location': {
                'patterns': [
                    r'\bin\s+(\w+(?:\s+\w+)*)\b',
                    r'\bnear\s+(\w+(?:\s+\w+)*)\b',
                    r'\baround\s+(\w+(?:\s+\w+)*)\b'
                ],
                'data_type': 'string',
                'field_mappings': ['city', 'neighborhood', 'location', 'address']
            },
            'price': {
                'patterns': [
                    r'\$([\d,]+(?:\.\d{2})?)\s*(?:per\s+night)?',
                    r'\bunder\s+\$([\d,]+)\b',
                    r'\bover\s+\$([\d,]+)\b',
                    r'\bbetween\s+\$([\d,]+)\s+and\s+\$([\d,]+)\b'
                ],
                'data_type': 'numeric',
                'field_mappings': ['price', 'nightly_rate', 'cost']
            },
            'capacity': {
                'patterns': [
                    r'\b(\d+)\s+(?:guests?|people|persons?)\b',
                    r'\baccommodates?\s+(\d+)\b',
                    r'\bsleeps?\s+(\d+)\b'
                ],
                'data_type': 'numeric',
                'field_mappings': ['accommodates', 'guests', 'capacity', 'sleeps']
            },
            'bedrooms': {
                'patterns': [
                    r'\b(\d+)\s+bedrooms?\b',
                    r'\b(\d+)\s+br\b',
                    r'\b(\d+)\s+bed\b'
                ],
                'data_type': 'numeric',
                'field_mappings': ['bedrooms', 'beds']
            },
            'bathrooms': {
                'patterns': [
                    r'\b(\d+(?:\.\d+)?)\s+bathrooms?\b',
                    r'\b(\d+(?:\.\d+)?)\s+ba\b',
                    r'\b(\d+(?:\.\d+)?)\s+bath\b'
                ],
                'data_type': 'numeric',
                'field_mappings': ['bathrooms', 'baths']
            },
            'amenities': {
                'patterns': [
                    r'\bwifi\b|\binternet\b',
                    r'\bparking\b|\bgarage\b',
                    r'\bpool\b|\bswimming\b',
                    r'\bkitchen\b|\bcooking\b',
                    r'\bgym\b|\bfitness\b',
                    r'\bbalcony\b|\bterrace\b',
                    r'\bpet.?friendly\b',
                    r'\bair.?conditioning\b|\bac\b'
                ],
                'data_type': 'list',
                'field_mappings': ['amenities', 'features']
            },
            'rating': {
                'patterns': [
                    r'\b(\d+(?:\.\d+)?)\s+stars?\b',
                    r'\brating\s+(?:of\s+)?(\d+(?:\.\d+)?)\b',
                    r'\b(\d+(?:\.\d+)?)\s+out\s+of\s+\d+\b'
                ],
                'data_type': 'numeric',
                'field_mappings': ['rating', 'review_scores_rating', 'stars']
            }
        }
    
    def extract_from_query(self, query: str) -> Dict[str, ContextualAttribute]:
        """Extract contextual attributes from a user query"""
        attributes = {}
        
        for attr_name, config in self.attribute_patterns.items():
            values = self._extract_attribute_values(query, config)
            if values:
                attributes[attr_name] = ContextualAttribute(
                    name=attr_name,
                    value=values[0] if len(values) == 1 else values,
                    source='query',
                    timestamp=datetime.now(),
                    confidence=0.8,
                    data_type=config['data_type'],
                    field_mappings=config['field_mappings']
                )
        
        return attributes
    
    def extract_from_response(self, response_data: Dict[str, Any]) -> Dict[str, ContextualAttribute]:
        """Extract contextual attributes from system response data"""
        attributes = {}
        
        # Extract from results if available
        if 'results' in response_data and isinstance(response_data['results'], list):
            for result in response_data['results']:
                result_attrs = self._extract_from_result_item(result)
                for attr_name, attr in result_attrs.items():
                    if attr_name not in attributes:
                        attributes[attr_name] = attr
                    else:
                        # Merge values if attribute already exists
                        self._merge_attribute_values(attributes[attr_name], attr)
        
        # Extract from metadata
        if 'metadata' in response_data:
            metadata_attrs = self._extract_from_metadata(response_data['metadata'])
            attributes.update(metadata_attrs)
        
        return attributes
    
    def _extract_attribute_values(self, text: str, config: Dict[str, Any]) -> List[Any]:
        """Extract attribute values using patterns"""
        values = []
        
        for pattern in config['patterns']:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                if config['data_type'] == 'numeric':
                    try:
                        # Handle multiple capture groups for ranges
                        if len(match.groups()) > 1:
                            values.extend([float(g.replace(',', '')) for g in match.groups() if g])
                        else:
                            values.append(float(match.group(1).replace(',', '')))
                    except (ValueError, IndexError):
                        continue
                elif config['data_type'] == 'string':
                    try:
                        values.append(match.group(1).strip())
                    except IndexError:
                        values.append(match.group(0).strip())
                elif config['data_type'] == 'list':
                    values.append(match.group(0).strip())
        
        return list(set(values))  # Remove duplicates
    
    def _extract_from_result_item(self, result: Dict[str, Any]) -> Dict[str, ContextualAttribute]:
        """Extract attributes from a single result item"""
        attributes = {}
        
        # Define key fields to extract
        field_mappings = {
            'price': ['price', 'nightly_rate'],
            'location': ['city', 'neighborhood', 'location'],
            'capacity': ['accommodates', 'guests'],
            'bedrooms': ['bedrooms', 'beds'],
            'bathrooms': ['bathrooms'],
            'rating': ['rating', 'review_scores_rating'],
            'amenities': ['amenities', 'features']
        }
        
        for attr_name, field_names in field_mappings.items():
            for field_name in field_names:
                if field_name in result and result[field_name] is not None:
                    config = self.attribute_patterns.get(attr_name, {})
                    attributes[attr_name] = ContextualAttribute(
                        name=attr_name,
                        value=result[field_name],
                        source='response',
                        timestamp=datetime.now(),
                        confidence=0.9,
                        data_type=config.get('data_type', 'string'),
                        field_mappings=[field_name]
                    )
                    break
        
        return attributes
    
    def _extract_from_metadata(self, metadata: Dict[str, Any]) -> Dict[str, ContextualAttribute]:
        """Extract attributes from response metadata"""
        attributes = {}
        
        # Extract summary statistics
        if 'summary_stats' in metadata:
            stats = metadata['summary_stats']
            for stat_name, value in stats.items():
                if stat_name in ['avg_price', 'min_price', 'max_price']:
                    attributes[f"summary_{stat_name}"] = ContextualAttribute(
                        name=f"summary_{stat_name}",
                        value=value,
                        source='metadata',
                        timestamp=datetime.now(),
                        confidence=1.0,
                        data_type='numeric'
                    )
        
        return attributes
    
    def _merge_attribute_values(self, existing: ContextualAttribute, new: ContextualAttribute):
        """Merge values from multiple attribute instances"""
        if existing.data_type == 'list':
            if isinstance(existing.value, list):
                if isinstance(new.value, list):
                    existing.value.extend(new.value)
                else:
                    existing.value.append(new.value)
            else:
                existing.value = [existing.value, new.value]
        elif existing.data_type == 'numeric':
            # For numeric values, keep the most confident one
            if new.confidence > existing.confidence:
                existing.value = new.value
                existing.confidence = new.confidence


class ContextCache:
    """Manages caching of conversation context and results"""
    
    def __init__(self, max_size: int = 100, ttl_hours: int = 24):
        self.logger = StructuredLogger(__name__, {'class_name': self.__class__.__name__})
        self.max_size = max_size
        self.ttl_hours = ttl_hours
        self.cache: Dict[str, CachedResult] = {}
        self.access_order: deque = deque(maxlen=max_size)
        
    def store_result(self, session_id: str, query_id: str, query: str, 
                    results: List[Dict[str, Any]], metadata: Dict[str, Any]) -> str:
        """Store query results in cache"""
        cache_key = f"{session_id}_{query_id}"
        
        # Extract attributes from results
        extractor = ContextualAttributeExtractor()
        extracted_attrs = {}
        
        for i, result in enumerate(results):
            result_attrs = extractor._extract_from_result_item(result)
            for attr_name, attr in result_attrs.items():
                key = f"{attr_name}_{i}"
                extracted_attrs[key] = attr
        
        # Create cached result
        cached_result = CachedResult(
            query_id=query_id,
            original_query=query,
            results=results,
            metadata=metadata,
            timestamp=datetime.now(),
            result_count=len(results),
            similarity_scores=metadata.get('similarity_scores', []),
            extracted_attributes=extracted_attrs
        )
        
        # Store in cache
        self.cache[cache_key] = cached_result
        self.access_order.append(cache_key)
        
        # Cleanup if necessary
        self._cleanup_cache()
        
        self.logger.info("Stored result in cache", extra={
            'cache_key': cache_key,
            'result_count': len(results),
            'attributes_extracted': len(extracted_attrs)
        })
        
        return cache_key
    
    def get_result(self, cache_key: str) -> Optional[CachedResult]:
        """Retrieve cached result"""
        if cache_key in self.cache:
            result = self.cache[cache_key]
            
            # Check if result has expired
            if self._is_expired(result):
                del self.cache[cache_key]
                return None
            
            # Move to end (most recently accessed)
            if cache_key in self.access_order:
                self.access_order.remove(cache_key)
            self.access_order.append(cache_key)
            
            return result
        
        return None
    
    def get_recent_results(self, session_id: str, limit: int = 5) -> List[CachedResult]:
        """Get recent results for a session"""
        session_results = []
        
        for cache_key in reversed(list(self.access_order)):
            if cache_key.startswith(session_id + "_"):
                result = self.get_result(cache_key)
                if result:
                    session_results.append(result)
                    if len(session_results) >= limit:
                        break
        
        return session_results
    
    def search_by_attributes(self, session_id: str, 
                           attributes: Dict[str, Any]) -> List[CachedResult]:
        """Search cached results by attributes"""
        matching_results = []
        
        for cache_key, cached_result in self.cache.items():
            if cache_key.startswith(session_id + "_") and not self._is_expired(cached_result):
                if self._matches_attributes(cached_result, attributes):
                    matching_results.append(cached_result)
        
        return matching_results
    
    def _is_expired(self, result: CachedResult) -> bool:
        """Check if cached result has expired"""
        expiry_time = result.timestamp + timedelta(hours=self.ttl_hours)
        return datetime.now() > expiry_time
    
    def _cleanup_cache(self):
        """Remove expired and excess entries"""
        # Remove expired entries
        expired_keys = [
            key for key, result in self.cache.items() 
            if self._is_expired(result)
        ]
        
        for key in expired_keys:
            del self.cache[key]
            if key in self.access_order:
                self.access_order.remove(key)
        
        # Remove excess entries (LRU)
        while len(self.cache) > self.max_size:
            oldest_key = self.access_order.popleft()
            if oldest_key in self.cache:
                del self.cache[oldest_key]
    
    def _matches_attributes(self, cached_result: CachedResult, 
                          target_attributes: Dict[str, Any]) -> bool:
        """Check if cached result matches target attributes"""
        for attr_name, target_value in target_attributes.items():
            found_match = False
            
            for cached_attr in cached_result.extracted_attributes.values():
                if cached_attr.name == attr_name:
                    if self._values_match(cached_attr.value, target_value):
                        found_match = True
                        break
            
            if not found_match:
                return False
        
        return True
    
    def _values_match(self, cached_value: Any, target_value: Any) -> bool:
        """Check if two attribute values match"""
        if isinstance(cached_value, str) and isinstance(target_value, str):
            return cached_value.lower() == target_value.lower()
        elif isinstance(cached_value, (int, float)) and isinstance(target_value, (int, float)):
            return abs(cached_value - target_value) < 0.01
        else:
            return cached_value == target_value
    
    def clear_session_cache(self, session_id: str):
        """Clear all cached results for a session"""
        keys_to_remove = [
            key for key in self.cache.keys() 
            if key.startswith(session_id + "_")
        ]
        
        for key in keys_to_remove:
            del self.cache[key]
            if key in self.access_order:
                self.access_order.remove(key)
        
        self.logger.info("Cleared session cache", extra={
            'session_id': session_id,
            'keys_removed': len(keys_to_remove)
        })


class ConversationContext:
    """Manages conversation context for a session"""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.conversation_history: List[ConversationTurn] = []
        self.accumulated_attributes: Dict[str, ContextualAttribute] = {}
        self.active_filters: Dict[str, Any] = {}
        self.current_result_set: Optional[str] = None  # Cache key for current results
        self.comparison_targets: List[str] = []  # For comparison operations
        
    def add_turn(self, turn: ConversationTurn, attributes: Dict[str, ContextualAttribute]):
        """Add a conversation turn with extracted attributes"""
        self.conversation_history.append(turn)
        
        # Update accumulated attributes
        for attr_name, attr in attributes.items():
            if attr_name in self.accumulated_attributes:
                # Update existing attribute if new one is more confident
                if attr.confidence > self.accumulated_attributes[attr_name].confidence:
                    self.accumulated_attributes[attr_name] = attr
            else:
                self.accumulated_attributes[attr_name] = attr
        
        # Limit history size
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]
    
    def update_active_filters(self, new_filters: Dict[str, Any]):
        """Update active filters"""
        self.active_filters.update(new_filters)
    
    def clear_active_filters(self):
        """Clear all active filters"""
        self.active_filters.clear()
    
    def set_current_result_set(self, cache_key: str):
        """Set the current result set reference"""
        self.current_result_set = cache_key
    
    def add_comparison_targets(self, targets: List[str]):
        """Add targets for comparison operations"""
        self.comparison_targets.extend(targets)
        self.comparison_targets = list(set(self.comparison_targets))  # Remove duplicates


class ContextManager:
    """Main context manager coordinating all context-aware operations"""
    
    def __init__(self, session_manager: Optional[SessionManager] = None):
        self.logger = StructuredLogger(__name__, {'class_name': self.__class__.__name__})
        
        # Core components
        self.session_manager = session_manager or SessionManager()
        self.intent_classifier = IntentClassifier()
        self.attribute_extractor = ContextualAttributeExtractor()
        self.context_cache = ContextCache()
        self.comparison_processor = ComparisonProcessor()
        
        # Session contexts
        self.session_contexts: Dict[str, ConversationContext] = {}
        
        self.logger.info("ContextManager initialized", extra={
            'components_loaded': 4
        })
    
    def process_query(self, session_id: str, query: str, 
                     results: List[Dict[str, Any]], metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Process a query within conversation context"""
        try:
            # Check if this is a reverse translation result that shouldn't be processed
            if (isinstance(results, list) and len(results) == 1 and 
                isinstance(results[0], dict) and 
                'translated_response' in results[0] and 'original_response' in results[0]):
                # This is a reverse translation result, return it as-is without context processing
                logger.debug("Detected reverse translation result, skipping context processing")
                return {
                    'results': results,
                    'metadata': metadata,
                    'context_info': {
                        'session_id': session_id,
                        'intent': 'translation',
                        'confidence': 1.0,
                        'operation': 'reverse_translation',
                        'follow_up_capable': False,
                        'translation_result': True
                    }
                }
            
            # Get or create conversation context
            context = self._get_conversation_context(session_id)
            
            # Extract attributes from query
            query_attributes = self.attribute_extractor.extract_from_query(query)
            
            # Extract attributes from response
            response_attributes = self.attribute_extractor.extract_from_response({
                'results': results or [],
                'metadata': metadata
            })
            
            # Combine attributes
            all_attributes = {**query_attributes, **response_attributes}
            
            # Classify intent for follow-up handling
            intent = self.intent_classifier.classify_intent(query, context.conversation_history)
            
            # Store results in cache
            query_id = str(uuid.uuid4())
            cache_key = self.context_cache.store_result(
                session_id, query_id, query, results, metadata
            )
            
            # Create conversation turn
            turn = ConversationTurn(
                user_query=query,
                system_response=metadata.get('response', ''),
                timestamp=datetime.now(),
                entities_extracted=list(all_attributes.keys()),
                intent_classified=intent.intent_type,
                search_query_used=query,
                documents_retrieved=[str(r.get('_id', i)) for i, r in enumerate(results)],
                response_time=metadata.get('execution_time', 0.0)
            )
            
            # Process based on intent
            processed_response = self._process_by_intent(
                intent, context, cache_key, results, metadata, all_attributes
            )
            
            # Update context
            context.add_turn(turn, all_attributes)
            context.set_current_result_set(cache_key)
            
            # Add enhanced context and formatting information to response
            processed_response.update({
                'context_info': {
                    'session_id': session_id,
                    'intent': intent.intent_type,
                    'confidence': intent.confidence,
                    'operation': intent.operation,
                    'attributes_extracted': len(all_attributes),
                    'cache_key': cache_key,
                    'follow_up_capable': True
                },
                'display_format': {
                    'enhanced_formatting': True,
                    'ui_table_support': intent.intent_type == 'comparison',
                    'source_data_formatted': True
                },
                'formatted_results': self._format_results_for_display(results, intent, all_attributes)
            })
            
            return processed_response
            
        except Exception as e:
            self.logger.error("Error processing query in context", extra={
                'session_id': session_id,
                'error': str(e),
                'query_length': len(query)
            })
            
            # Return original response on error
            return {
                'results': results,
                'metadata': metadata,
                'context_info': {
                    'error': str(e),
                    'session_id': session_id,
                    'follow_up_capable': False
                }
            }
    
    def _get_conversation_context(self, session_id: str) -> ConversationContext:
        """Get or create conversation context for session"""
        if session_id not in self.session_contexts:
            self.session_contexts[session_id] = ConversationContext(session_id)
        
        return self.session_contexts[session_id]
    
    def _process_by_intent(self, intent: FollowUpIntent, context: ConversationContext,
                          cache_key: str, results: List[Dict[str, Any]], 
                          metadata: Dict[str, Any], attributes: Dict[str, ContextualAttribute]) -> Dict[str, Any]:
        """Process query based on detected intent"""
        if intent.intent_type == 'comparison':
            return self._handle_comparison(intent, context, results, metadata)
        elif intent.intent_type == 'filtering':
            return self._handle_filtering(intent, context, results, metadata, attributes)
        elif intent.intent_type == 'summarization':
            return self._handle_summarization(intent, context, results, metadata)
        elif intent.intent_type == 'reasoning':
            return self._handle_reasoning(intent, context, results, metadata)
        else:
            return {'results': results, 'metadata': metadata}
    
    def _handle_comparison(self, intent: FollowUpIntent, context: ConversationContext,
                          results: List[Dict[str, Any]], metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Handle comparison operations using cached results"""
        if context.current_result_set:
            cached_result = self.context_cache.get_result(context.current_result_set)
            if cached_result:
                # Use the original query to extract item references
                original_query = intent.original_query if hasattr(intent, 'original_query') else ''
                
                # Extract specific items referenced in the comparison query
                referenced_items = self.comparison_processor.extract_item_references(
                    original_query, cached_result.results
                )
                
                if referenced_items:
                    # Generate comprehensive comparison table
                    comparison_table = self.comparison_processor.generate_comparison_table(
                        referenced_items
                    )
                    
                    return {
                        'results': referenced_items,  # Return only the compared items
                        'metadata': {
                            **metadata,
                            'operation_type': 'comparison',
                            'items_compared': len(referenced_items),
                            'source_results_total': len(cached_result.results),
                            'comparison_query': original_query,
                            'used_cached_results': True
                        },
                        'comparison_table': comparison_table,
                        'comparison_summary': {
                            'type': 'tabular_comparison',
                            'similarities_found': len(comparison_table.get('similarities', [])),
                            'differences_found': len(comparison_table.get('key_differences', [])),
                            'recommendation': 'Review the table below for detailed comparison'
                        }
                    }
                else:
                    # No specific items referenced, compare first two by default
                    default_items = cached_result.results[:2]
                    if len(default_items) >= 2:
                        comparison_table = self.comparison_processor.generate_comparison_table(
                            default_items
                        )
                        
                        return {
                            'results': default_items,
                            'metadata': {
                                **metadata,
                                'operation_type': 'comparison',
                                'items_compared': len(default_items),
                                'comparison_note': 'Compared first two items from previous results',
                                'used_cached_results': True
                            },
                            'comparison_table': comparison_table
                        }
        
        # Fallback to original results if no cached data available
        return {
            'results': results, 
            'metadata': {
                **metadata,
                'operation_type': 'comparison_fallback',
                'note': 'No previous results available for comparison'
            }
        }
    
    def _handle_filtering(self, intent: FollowUpIntent, context: ConversationContext,
                         results: List[Dict[str, Any]], metadata: Dict[str, Any],
                         attributes: Dict[str, ContextualAttribute]) -> Dict[str, Any]:
        """Handle filtering operations"""
        if context.current_result_set:
            cached_result = self.context_cache.get_result(context.current_result_set)
            if cached_result:
                # Extract filter criteria from attributes
                filter_criteria = self._extract_filter_criteria(attributes, intent)
                
                # Apply filters to cached results
                filtered_results = self._apply_filters(cached_result.results, filter_criteria)
                
                # Update context with active filters
                context.update_active_filters(filter_criteria)
                
                return {
                    'results': filtered_results,
                    'metadata': {
                        **metadata,
                        'operation_type': 'filtering',
                        'filters_applied': filter_criteria,
                        'original_count': len(cached_result.results),
                        'filtered_count': len(filtered_results)
                    }
                }
        
        return {'results': results, 'metadata': metadata}
    
    def _handle_summarization(self, intent: FollowUpIntent, context: ConversationContext,
                             results: List[Dict[str, Any]], metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Handle summarization operations"""
        if context.current_result_set:
            cached_result = self.context_cache.get_result(context.current_result_set)
            if cached_result:
                summary = self._create_summary(cached_result.results)
                
                return {
                    'results': cached_result.results,
                    'metadata': {
                        **metadata,
                        'operation_type': 'summarization',
                        'summary_stats': summary['stats'],
                        'total_results': len(cached_result.results)
                    },
                    'summary': summary
                }
        
        return {'results': results, 'metadata': metadata}
    
    def _handle_reasoning(self, intent: FollowUpIntent, context: ConversationContext,
                         results: List[Dict[str, Any]], metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Handle reasoning and explanation operations"""
        if context.current_result_set:
            cached_result = self.context_cache.get_result(context.current_result_set)
            if cached_result:
                explanation = self._generate_explanation(cached_result.results, intent)
                
                return {
                    'results': cached_result.results[:3],  # Show top 3 for explanation
                    'metadata': {
                        **metadata,
                        'operation_type': 'reasoning',
                        'explanation_type': intent.operation
                    },
                    'explanation': explanation
                }
        
        return {'results': results, 'metadata': metadata}
    
    def _perform_comparison(self, entities: List[str], results: List[Dict[str, Any]], 
                           operation: str) -> List[Dict[str, Any]]:
        """Perform comparison between specified entities"""
        if len(entities) >= 2:
            # Try to find entities by letter references (A, B, C)
            entity_indices = []
            for entity in entities:
                if entity.upper() in 'ABCDEFGHIJ':
                    idx = ord(entity.upper()) - ord('A')
                    if 0 <= idx < len(results):
                        entity_indices.append(idx)
                elif entity.isdigit():
                    idx = int(entity) - 1
                    if 0 <= idx < len(results):
                        entity_indices.append(idx)
            
            if len(entity_indices) >= 2:
                return [results[i] for i in entity_indices[:2]]
        
        # Fallback to first two results
        return results[:2]
    
    def _create_comparison_table(self, entities: List[str], 
                                results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create a comparison table for results"""
        if len(results) < 2:
            return {}
        
        # Select two results to compare
        result_a = results[0]
        result_b = results[1]
        
        # Define fields to compare
        compare_fields = ['name', 'price', 'property_type', 'bedrooms', 'bathrooms', 
                         'accommodates', 'rating', 'neighborhood']
        
        comparison = {
            'property_a': {},
            'property_b': {},
            'differences': {},
            'similarities': []
        }
        
        for field in compare_fields:
            if field in result_a:
                comparison['property_a'][field] = result_a[field]
            if field in result_b:
                comparison['property_b'][field] = result_b[field]
            
            # Note differences and similarities
            if field in result_a and field in result_b:
                val_a, val_b = result_a[field], result_b[field]
                
                if isinstance(val_a, (int, float)) and isinstance(val_b, (int, float)):
                    diff = abs(val_a - val_b)
                    comparison['differences'][field] = {
                        'value_a': val_a,
                        'value_b': val_b,
                        'difference': diff
                    }
                    
                    if diff < 0.01:  # Essentially equal
                        comparison['similarities'].append(field)
                elif str(val_a).lower() == str(val_b).lower():
                    comparison['similarities'].append(field)
                else:
                    comparison['differences'][field] = {
                        'value_a': val_a,
                        'value_b': val_b,
                        'difference': 'different_values'
                    }
        
        return comparison
    
    def _extract_filter_criteria(self, attributes: Dict[str, ContextualAttribute], 
                                intent: FollowUpIntent) -> Dict[str, Any]:
        """Extract filter criteria from attributes and intent"""
        criteria = {}
        
        # Extract from attributes
        for attr_name, attr in attributes.items():
            if attr.data_type == 'list' and attr_name == 'amenities':
                criteria['amenities'] = attr.value if isinstance(attr.value, list) else [attr.value]
            elif attr.data_type == 'numeric':
                criteria[attr_name] = attr.value
            elif attr.data_type == 'string':
                criteria[attr_name] = attr.value
        
        # Extract from intent operation
        if 'filter_by_' in intent.operation:
            filter_terms = intent.operation.replace('filter_by_', '').split('_')
            for term in filter_terms:
                if term in ['balcony', 'wifi', 'pool', 'parking', 'kitchen']:
                    if 'amenities' not in criteria:
                        criteria['amenities'] = []
                    criteria['amenities'].append(term)
        
        return criteria
    
    def _apply_filters(self, results: List[Dict[str, Any]], 
                      criteria: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Apply filter criteria to results"""
        filtered_results = []
        
        for result in results:
            matches = True
            
            for criterion, value in criteria.items():
                if criterion == 'amenities':
                    result_amenities = result.get('amenities', [])
                    if isinstance(result_amenities, str):
                        result_amenities = result_amenities.lower().split(',')
                    elif isinstance(result_amenities, list):
                        result_amenities = [str(a).lower() for a in result_amenities]
                    else:
                        result_amenities = []
                    
                    required_amenities = value if isinstance(value, list) else [value]
                    
                    for required in required_amenities:
                        required_lower = str(required).lower()
                        if not any(required_lower in amenity for amenity in result_amenities):
                            matches = False
                            break
                
                elif criterion in result:
                    result_value = result[criterion]
                    
                    if isinstance(result_value, (int, float)) and isinstance(value, (int, float)):
                        # For numeric values, allow some tolerance
                        if abs(result_value - value) > 0.01:
                            matches = False
                    elif str(result_value).lower() != str(value).lower():
                        matches = False
                
                if not matches:
                    break
            
            if matches:
                filtered_results.append(result)
        
        return filtered_results
    
    def _create_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create a summary of results"""
        if not results:
            return {'stats': {}, 'overview': 'No results to summarize'}
        
        summary = {
            'stats': {
                'total_properties': len(results),
                'avg_price': 0,
                'price_range': {'min': float('inf'), 'max': 0},
                'property_types': {},
                'bedroom_distribution': {},
                'avg_rating': 0
            },
            'overview': ''
        }
        
        # Calculate statistics
        prices = []
        ratings = []
        
        for result in results:
            # Price statistics
            price = result.get('price')
            if price is not None:
                try:
                    price_val = float(str(price).replace('$', '').replace(',', ''))
                    prices.append(price_val)
                    summary['stats']['price_range']['min'] = min(summary['stats']['price_range']['min'], price_val)
                    summary['stats']['price_range']['max'] = max(summary['stats']['price_range']['max'], price_val)
                except ValueError:
                    pass
            
            # Property type distribution
            prop_type = result.get('property_type')
            if prop_type:
                summary['stats']['property_types'][prop_type] = summary['stats']['property_types'].get(prop_type, 0) + 1
            
            # Bedroom distribution
            bedrooms = result.get('bedrooms')
            if bedrooms is not None:
                summary['stats']['bedroom_distribution'][str(bedrooms)] = summary['stats']['bedroom_distribution'].get(str(bedrooms), 0) + 1
            
            # Rating statistics
            rating = result.get('rating') or result.get('review_scores_rating')
            if rating is not None:
                try:
                    rating_val = float(rating)
                    ratings.append(rating_val)
                except ValueError:
                    pass
        
        # Calculate averages
        if prices:
            summary['stats']['avg_price'] = sum(prices) / len(prices)
        if ratings:
            summary['stats']['avg_rating'] = sum(ratings) / len(ratings)
        
        # Handle edge case for min price
        if summary['stats']['price_range']['min'] == float('inf'):
            summary['stats']['price_range']['min'] = 0
        
        # Create overview text
        overview_parts = [
            f"Found {len(results)} properties total.",
            f"Price range: ${summary['stats']['price_range']['min']:.0f} - ${summary['stats']['price_range']['max']:.0f}",
            f"Average price: ${summary['stats']['avg_price']:.0f}" if summary['stats']['avg_price'] > 0 else "",
            f"Average rating: {summary['stats']['avg_rating']:.1f}" if summary['stats']['avg_rating'] > 0 else ""
        ]
        
        summary['overview'] = ' '.join(part for part in overview_parts if part)
        
        return summary
    
    def _generate_explanation(self, results: List[Dict[str, Any]], 
                             intent: FollowUpIntent) -> Dict[str, Any]:
        """Generate explanations for why certain results are recommended"""
        if not results:
            return {'explanation': 'No results available to explain'}
        
        explanations = []
        
        for i, result in enumerate(results[:3]):  # Explain top 3
            factors = []
            
            # Price competitiveness
            price = result.get('price')
            if price:
                try:
                    price_val = float(str(price).replace('$', '').replace(',', ''))
                    if price_val < 100:
                        factors.append("competitively priced")
                    elif price_val > 300:
                        factors.append("premium pricing for luxury features")
                except ValueError:
                    pass
            
            # High rating
            rating = result.get('rating') or result.get('review_scores_rating')
            if rating:
                try:
                    rating_val = float(rating)
                    if rating_val >= 4.5:
                        factors.append("highly rated by guests")
                    elif rating_val >= 4.0:
                        factors.append("well-reviewed")
                except ValueError:
                    pass
            
            # Property features
            prop_type = result.get('property_type', '')
            if 'entire' in prop_type.lower():
                factors.append("provides privacy as an entire property")
            
            bedrooms = result.get('bedrooms')
            if bedrooms and int(bedrooms) >= 2:
                factors.append(f"spacious with {bedrooms} bedrooms")
            
            # Amenities
            amenities = result.get('amenities', [])
            if isinstance(amenities, str):
                amenities = amenities.lower()
                if 'wifi' in amenities:
                    factors.append("includes WiFi")
                if 'parking' in amenities:
                    factors.append("offers parking")
                if 'pool' in amenities:
                    factors.append("has pool access")
            
            explanation_text = f"Property {chr(65+i)} is recommended because it is " + ", ".join(factors) + "."
            
            explanations.append({
                'property': chr(65+i),
                'explanation': explanation_text,
                'factors': factors
            })
        
        return {
            'explanations': explanations,
            'summary': f"These {len(explanations)} properties were selected based on their combination of pricing, ratings, and amenities."
        }
    
    def get_context_summary(self, session_id: str) -> Dict[str, Any]:
        """Get a summary of the current conversation context"""
        context = self._get_conversation_context(session_id)
        recent_results = self.context_cache.get_recent_results(session_id)
        
        return {
            'session_id': session_id,
            'conversation_turns': len(context.conversation_history),
            'accumulated_attributes': len(context.accumulated_attributes),
            'active_filters': context.active_filters,
            'current_result_set': context.current_result_set,
            'comparison_targets': context.comparison_targets,
            'cached_results_count': len(recent_results),
            'last_activity': recent_results[0].timestamp.isoformat() if recent_results else None
        }
    
    def clear_context(self, session_id: str):
        """Clear all context for a session"""
        if session_id in self.session_contexts:
            del self.session_contexts[session_id]
        
        self.context_cache.clear_session_cache(session_id)
        
        self.logger.info("Cleared context for session", extra={
            'session_id': session_id
        })


# Convenience function for easy integration
def create_context_manager(session_manager: Optional[SessionManager] = None) -> ContextManager:
    """Create and return a configured ContextManager instance"""
    return ContextManager(session_manager)


# Export key classes
__all__ = [
    'ContextManager',
    'ConversationContext', 
    'ContextualAttribute',
    'CachedResult',
    'FollowUpIntent',
    'IntentClassifier',
    'ContextualAttributeExtractor',
    'ContextCache',
    'create_context_manager'
]
