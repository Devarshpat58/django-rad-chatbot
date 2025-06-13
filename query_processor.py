#!/usr/bin/env python3
"""
Query Processor for JSON RAG System
Advanced query processing with intent detection and contextual information extraction
"""

import re
import os
import logging
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime

# Import structured logger
from logging_config import StructuredLogger

# Set encoding environment
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['PYTHONUTF8'] = '1'

from consolidated_config import NumericConfig
from consolidated_config import (
    PROPERTY_TYPE_SYNONYMS, AMENITY_SYNONYMS, LOCATION_SYNONYMS
)

@dataclass
class ProcessedQuery:
    """Structured representation of a processed query"""
    original_query: str
    cleaned_query: str
    intent: str
    confidence: float
    entities: Dict[str, List[str]]
    numeric_constraints: Dict[str, Any]
    contextual_info: Dict[str, Any]
    enhanced_query: str
    database_filters: Dict[str, Any]
    search_strategy: Dict[str, Any]
    timestamp: datetime

class NumericProcessor:
    """Handles numeric constraint extraction and processing"""
    
    def __init__(self):
        self.logger = StructuredLogger(__name__, {'class_name': self.__class__.__name__})
        self.logger.info("Initializing NumericProcessor", extra={
                        'module': 'query_processor',
                        'class_name': 'NumericProcessor',
                        'method': '__init__'
                    })
        self.numeric_patterns = self._compile_patterns()
        self.logger.info("NumericProcessor initialized successfully", extra={
                        'module': 'query_processor',
                        'class_name': 'NumericProcessor',
                        'method': '__init__',
                        'patterns_compiled': len(self.numeric_patterns)
                    })
    
    def _compile_patterns(self) -> Dict[str, Any]:
        """Compile regex patterns for numeric extraction"""
        compiled = {}
        
        for constraint_type, config in NumericConfig.NUMERIC_KEYWORDS.items():
            compiled[constraint_type] = {
                'patterns': [re.compile(pattern, re.IGNORECASE) for pattern in config['patterns']],
                'keywords': config['keywords'],
                'field_names': config['field_names']
            }
        
        # Compile range operator patterns
        compiled['operators'] = {}
        for op_type, config in NumericConfig.RANGE_OPERATORS.items():
            compiled['operators'][op_type] = {
                'patterns': [re.compile(pattern, re.IGNORECASE) for pattern in config['patterns']],
                'operator': config['operator']
            }
        
        return compiled
    
    def extract_numeric_constraints(self, query: str) -> Dict[str, Any]:
        """Extract all numeric constraints from query"""
        self.logger.info("Starting numeric constraint extraction", extra={
                        'module': 'query_processor',
                        'class_name': 'NumericProcessor',
                        'method': 'extract_numeric_constraints',
                        'query_length': len(query)
                    })
        constraints = {}
        
        # Convert number words to digits
        processed_query = self._convert_number_words(query)
        
        # Extract each type of numeric constraint
        for constraint_type in ['bedrooms', 'bathrooms', 'guests', 'price']:
            constraint = self._extract_constraint_type(processed_query, constraint_type)
            if constraint:
                constraints[constraint_type] = constraint
        
        # Add context-based implications
        context_constraints = self._extract_contextual_constraints(query)
        for key, value in context_constraints.items():
            if key not in constraints:
                constraints[key] = value
        
        self.logger.info("Numeric constraint extraction completed", extra={
                        'module': 'query_processor',
                        'class_name': 'NumericProcessor',
                        'method': 'extract_numeric_constraints',
                        'constraints_found': len(constraints),
                        'constraint_types': list(constraints.keys())
                    })
        return constraints
    
    def _convert_number_words(self, query: str) -> str:
        """Convert number words to digits"""
        processed = query.lower()
        
        for word, number in NumericConfig.NUMBER_WORDS.items():
            # Replace whole words only
            pattern = r'\b' + re.escape(word) + r'\b'
            processed = re.sub(pattern, str(number), processed)
        
        return processed
    
    def _extract_constraint_type(self, query: str, constraint_type: str) -> Optional[Dict[str, Any]]:
        """Extract specific constraint type from query"""
        if constraint_type not in self.numeric_patterns:
            return None
        
        patterns = self.numeric_patterns[constraint_type]['patterns']
        
        # Try each pattern
        for pattern in patterns:
            matches = pattern.findall(query)
            if matches:
                # Determine operator type
                operator_info = self._determine_operator(query, matches[0])
                
                if constraint_type == 'price':
                    return self._process_price_constraint(matches[0], operator_info, query)
                else:
                    return self._process_numeric_constraint(matches[0], operator_info, constraint_type)
        
        return None
    
    def _determine_operator(self, query: str, matched_value: str) -> Dict[str, str]:
        """Determine the operator type for the constraint"""
        for op_type, config in self.numeric_patterns['operators'].items():
            for pattern in config['patterns']:
                if pattern.search(query):
                    return {'type': op_type, 'operator': config['operator']}
        
        return {'type': 'exact', 'operator': '='}
    
    def _process_price_constraint(self, value: str, operator_info: Dict, query: str) -> Dict[str, Any]:
        """Process price-specific constraints"""
        # Clean price value
        price_value = re.sub(r'[,$]', '', str(value))
        try:
            price = float(price_value)
        except ValueError:
            return None
        
        constraint = {
            'value': price,
            'operator': operator_info['operator'],
            'type': operator_info['type'],
            'field_names': NumericConfig.NUMERIC_KEYWORDS['price']['field_names']
        }
        
        # Handle range patterns specifically for price
        if 'under' in query.lower() or 'below' in query.lower():
            constraint['operator'] = '<='
            constraint['type'] = 'maximum'
        elif 'over' in query.lower() or 'above' in query.lower():
            constraint['operator'] = '>='
            constraint['type'] = 'minimum'
        
        return constraint
    
    def _process_numeric_constraint(self, value: str, operator_info: Dict, constraint_type: str) -> Dict[str, Any]:
        """Process general numeric constraints"""
        try:
            numeric_value = int(float(str(value)))
        except ValueError:
            return None
        
        return {
            'value': numeric_value,
            'operator': operator_info['operator'],
            'type': operator_info['type'],
            'field_names': NumericConfig.NUMERIC_KEYWORDS[constraint_type]['field_names']
        }
    
    def _extract_contextual_constraints(self, query: str) -> Dict[str, Any]:
        """Extract constraints based on contextual patterns"""
        constraints = {}
        
        for context_type, config in NumericConfig.CONTEXT_PATTERNS.items():
            for keyword in config['keywords']:
                if re.search(r'\b' + re.escape(keyword) + r'\b', query.lower()):
                    implications = config['implications']
                    for constraint_type, value in implications.items():
                        if isinstance(value, list) and len(value) == 2:
                            # Range constraint
                            constraints[constraint_type] = {
                                'min_value': value[0],
                                'max_value': value[1],
                                'operator': 'between',
                                'type': 'range',
                                'source': f'context_{context_type}',
                                'field_names': NumericConfig.NUMERIC_KEYWORDS.get(constraint_type, {}).get('field_names', [constraint_type])
                            }
                        else:
                            # Exact constraint
                            constraints[constraint_type] = {
                                'value': value,
                                'operator': '=',
                                'type': 'exact',
                                'source': f'context_{context_type}',
                                'field_names': NumericConfig.NUMERIC_KEYWORDS.get(constraint_type, {}).get('field_names', [constraint_type])
                            }
                    break
        
        return constraints

class IntentClassifier:
    """Classifies query intent and confidence"""
    
    def __init__(self):
        self.logger = StructuredLogger(__name__, {'class_name': self.__class__.__name__})
        self.logger.info("Initializing IntentClassifier", extra={
                        'module': 'query_processor',
                        'class_name': 'IntentClassifier',
                        'method': '__init__'
                    })
        self.intent_patterns = self._compile_intent_patterns()
        self.logger.info("IntentClassifier initialized successfully", extra={
                        'module': 'query_processor',
                        'class_name': 'IntentClassifier',
                        'method': '__init__',
                        'intent_patterns_count': len(self.intent_patterns)
                    })
    
    def _compile_intent_patterns(self) -> Dict[str, List]:
        """Compile intent classification patterns"""
        patterns = {}
        
        # General search intents
        patterns['search'] = [
            re.compile(r'\b(?:find|search|look|show|display)\b', re.IGNORECASE),
            re.compile(r'\b(?:properties|places|accommodation)\b', re.IGNORECASE),
            re.compile(r'\b(?:available|vacancy|rent)\b', re.IGNORECASE)
        ]
        
        # Filter intents
        patterns['filter'] = [
            re.compile(r'\b(?:with|having|that|include|excluding)\b', re.IGNORECASE),
            re.compile(r'\b(?:filter|narrow|refine)\b', re.IGNORECASE)
        ]
        
        # Information intents
        patterns['info'] = [
            re.compile(r'\b(?:what|how|why|when|where|tell|explain)\b', re.IGNORECASE),
            re.compile(r'\b(?:details|information|about|describe)\b', re.IGNORECASE)
        ]
        
        # Comparison intents
        patterns['compare'] = [
            re.compile(r'\b(?:compare|versus|vs|difference|better|best)\b', re.IGNORECASE),
            re.compile(r'\b(?:similar|alternative|option)\b', re.IGNORECASE)
        ]
        
        # Add numeric-specific patterns
        for intent_type, config in NumericConfig.NUMERIC_INTENT_PATTERNS.items():
            patterns[intent_type] = [re.compile(pattern, re.IGNORECASE) for pattern in config['patterns']]
        
        return patterns
    
    def classify_intent(self, query: str) -> Tuple[str, float]:
        """Classify query intent with confidence score"""
        intent_scores = {}
        
        for intent_type, patterns in self.intent_patterns.items():
            score = 0
            matches = 0
            
            for pattern in patterns:
                if pattern.search(query):
                    matches += 1
                    score += 1
            
            if matches > 0:
                intent_scores[intent_type] = score / len(patterns)
        
        if not intent_scores:
            return 'search', 0.5  # Default intent
        
        # Get highest scoring intent
        best_intent = max(intent_scores.items(), key=lambda x: x[1])
        return best_intent[0], best_intent[1]

class EntityExtractor:
    """Extracts entities from queries"""
    
    def __init__(self):
        self.logger = StructuredLogger(__name__, {'class_name': self.__class__.__name__})
        self.logger.info("Initializing EntityExtractor", extra={
                        'module': 'query_processor',
                        'class_name': 'EntityExtractor',
                        'method': '__init__'
                    })
        self.synonym_maps = {
            'property_types': PROPERTY_TYPE_SYNONYMS,
            'amenities': AMENITY_SYNONYMS,
            'locations': LOCATION_SYNONYMS
        }
    
    def extract_entities(self, query: str) -> Dict[str, List[str]]:
        """Extract all entities from query"""
        entities = {
            'property_types': [],
            'amenities': [],
            'locations': [],
            'numeric_values': [],
            'dates': [],
            'special_requirements': []
        }
        
        # Extract property types
        entities['property_types'] = self._extract_property_types(query)
        
        # Extract amenities
        entities['amenities'] = self._extract_amenities(query)
        
        # Extract locations
        entities['locations'] = self._extract_locations(query)
        
        # Extract numeric values
        entities['numeric_values'] = self._extract_numeric_values(query)
        
        # Extract special requirements
        entities['special_requirements'] = self._extract_special_requirements(query)
        
        return entities
    
    def _extract_property_types(self, query: str) -> List[str]:
        """Extract property type mentions"""
        found_types = []
        query_lower = query.lower()
        
        for property_type, synonyms in self.synonym_maps['property_types'].items():
            for synonym in synonyms:
                if re.search(r'\b' + re.escape(synonym.lower()) + r'\b', query_lower):
                    if property_type not in found_types:
                        found_types.append(property_type)
                    break
        
        return found_types
    
    def _extract_amenities(self, query: str) -> List[str]:
        """Extract amenity mentions"""
        found_amenities = []
        query_lower = query.lower()
        
        for amenity, synonyms in self.synonym_maps['amenities'].items():
            for synonym in synonyms:
                if re.search(r'\b' + re.escape(synonym.lower()) + r'\b', query_lower):
                    if amenity not in found_amenities:
                        found_amenities.append(amenity)
                    break
        
        return found_amenities
    
    def _extract_locations(self, query: str) -> List[str]:
        """Extract location mentions"""
        found_locations = []
        query_lower = query.lower()
        
        for location, synonyms in self.synonym_maps['locations'].items():
            for synonym in synonyms:
                if re.search(r'\b' + re.escape(synonym.lower()) + r'\b', query_lower):
                    if location not in found_locations:
                        found_locations.append(location)
                    break
        
        return found_locations
    
    def _extract_numeric_values(self, query: str) -> List[Dict[str, Any]]:
        """Extract numeric values with context"""
        numeric_entities = []
        
        # Extract price values
        price_pattern = re.compile(r'\$\s*(\d+(?:,\d{3})*(?:\.\d{2})?)', re.IGNORECASE)
        for match in price_pattern.finditer(query):
            numeric_entities.append({
                'type': 'price',
                'value': float(match.group(1).replace(',', '')),
                'raw_text': match.group(0),
                'position': match.span()
            })
        
        # Extract general numbers
        number_pattern = re.compile(r'\b(\d+)\b')
        for match in number_pattern.finditer(query):
            numeric_entities.append({
                'type': 'number',
                'value': int(match.group(1)),
                'raw_text': match.group(0),
                'position': match.span()
            })
        
        return numeric_entities
    
    def _extract_special_requirements(self, query: str) -> List[str]:
        """Extract special requirement mentions"""
        requirements = []
        query_lower = query.lower()
        
        special_patterns = {
            'pet_friendly': [r'\bpet\b', r'\bdog\b', r'\bcat\b', r'\bpet.friendly\b'],
            'accessible': [r'\baccessible\b', r'\bwheelchair\b', r'\bdisability\b'],
            'parking': [r'\bparking\b', r'\bgarage\b', r'\bcar\b'],
            'quiet': [r'\bquiet\b', r'\bpeaceful\b', r'\bsilent\b'],
            'family_friendly': [r'\bfamily\b', r'\bkids\b', r'\bchildren\b'],
            'luxury': [r'\bluxury\b', r'\bluxurious\b', r'\bupscale\b', r'\bpremium\b']
        }
        
        for req_type, patterns in special_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    requirements.append(req_type)
                    break
        
        return requirements

class QueryProcessor:
    """Main query processor that coordinates all query analysis"""
    
    def __init__(self):
        self.logger = StructuredLogger(__name__, {'class_name': self.__class__.__name__})
        self.logger.info("Initializing QueryProcessor", extra={
                        'module': 'query_processor',
                        'class_name': 'QueryProcessor',
                        'method': '__init__'
                    })
        self.numeric_processor = NumericProcessor()
        self.intent_classifier = IntentClassifier()
        self.entity_extractor = EntityExtractor()
        self.logger.info("QueryProcessor initialized successfully", extra={
                        'module': 'query_processor',
                        'class_name': 'QueryProcessor',
                        'method': '__init__',
                        'components_initialized': 3
                    })
    
    def process_query(self, query: str, session_context: Optional[Dict] = None) -> ProcessedQuery:
        """Process query and return structured information"""
        try:
            # Clean input query
            cleaned_query = self._clean_query(query)
            
            # Classify intent
            intent, confidence = self.intent_classifier.classify_intent(cleaned_query)
            
            # Extract entities
            entities = self.entity_extractor.extract_entities(cleaned_query)
            
            # Extract numeric constraints
            numeric_constraints = self.numeric_processor.extract_numeric_constraints(cleaned_query)
            
            # Generate contextual information
            contextual_info = self._generate_contextual_info(
                entities, numeric_constraints, session_context
            )
            
            # Enhance query with context
            enhanced_query = self._enhance_query_with_context(
                cleaned_query, entities, numeric_constraints, contextual_info
            )
            
            # Generate database filters
            database_filters = self._generate_database_filters(
                numeric_constraints, entities
            )
            
            # Determine search strategy
            search_strategy = self._determine_search_strategy(
                intent, entities, numeric_constraints
            )
            
            self.logger.info("Query processing completed successfully", 
                            source_module="query_processor", 
                            method="process_query",
                            intent=intent,
                            confidence=confidence,
                            entities_found=sum(len(entity_list) for entity_list in entities.values()),
                            numeric_constraints_found=len(numeric_constraints))
            return ProcessedQuery(
                original_query=query,
                cleaned_query=cleaned_query,
                intent=intent,
                confidence=confidence,
                entities=entities,
                numeric_constraints=numeric_constraints,
                contextual_info=contextual_info,
                enhanced_query=enhanced_query,
                database_filters=database_filters,
                search_strategy=search_strategy,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            self.logger.error("Error processing query", 
                            source_module="query_processor", 
                            method="process_query",
                            error=str(e),
                            query_length=len(query),
                            exc_info=True)
            # Return basic processed query on error
            return ProcessedQuery(
                original_query=query,
                cleaned_query=query,
                intent='search',
                confidence=0.5,
                entities={},
                numeric_constraints={},
                contextual_info={},
                enhanced_query=query,
                database_filters={},
                search_strategy={'method': 'hybrid', 'weights': {'semantic': 0.8, 'fuzzy': 0.2}},
                timestamp=datetime.now()
            )
    
    def _clean_query(self, query: str) -> str:
        """Clean and normalize query"""
        # Remove extra whitespace
        cleaned = re.sub(r'\s+', ' ', query.strip())
        
        # Convert to ASCII-safe format
        # Preserve unicode characters instead of removing them
        cleaned = cleaned
        
        return cleaned
    
    def _generate_contextual_info(self, entities: Dict, numeric_constraints: Dict, 
                                session_context: Optional[Dict] = None) -> Dict[str, Any]:
        """Generate contextual information for enhanced search"""
        contextual_info = {
            'search_complexity': self._calculate_search_complexity(entities, numeric_constraints),
            'property_type_context': self._get_property_type_context(entities.get('property_types', [])),
            'guest_context': self._get_guest_context(numeric_constraints),
            'amenity_importance': self._calculate_amenity_importance(entities.get('amenities', [])),
            'price_context': self._get_price_context(numeric_constraints),
            'session_enhancement': self._get_session_enhancement(session_context)
        }
        
        return contextual_info
    
    def _calculate_search_complexity(self, entities: Dict, numeric_constraints: Dict) -> str:
        """Calculate complexity level of search"""
        complexity_score = 0
        
        # Count constraints
        complexity_score += len(numeric_constraints)
        complexity_score += sum(len(entity_list) for entity_list in entities.values())
        
        if complexity_score <= 2:
            return 'simple'
        elif complexity_score <= 5:
            return 'moderate'
        else:
            return 'complex'
    
    def _get_property_type_context(self, property_types: List[str]) -> Dict[str, Any]:
        """Get context based on property types"""
        context = {
            'default_constraints': {},
            'search_weight_adjustments': {}
        }
        
        for prop_type in property_types:
            if prop_type in NumericConfig.PROPERTY_NUMERIC_DEFAULTS:
                defaults = NumericConfig.PROPERTY_NUMERIC_DEFAULTS[prop_type]
                for constraint_type, default_value in defaults.items():
                    if constraint_type not in context['default_constraints']:
                        context['default_constraints'][constraint_type] = default_value
        
        return context
    
    def _get_guest_context(self, numeric_constraints: Dict) -> Dict[str, Any]:
        """Get context based on guest requirements"""
        guest_constraint = numeric_constraints.get('guests')
        if not guest_constraint:
            return {}
        
        guest_count = guest_constraint.get('value', 1)
        
        context = {
            'group_size': 'single' if guest_count == 1 else 'couple' if guest_count == 2 
                         else 'small_group' if guest_count <= 4 else 'large_group',
            'recommended_bedrooms': max(1, (guest_count + 1) // 2),
            'recommended_bathrooms': max(1, guest_count // 3)
        }
        
        return context
    
    def _calculate_amenity_importance(self, amenities: List[str]) -> Dict[str, float]:
        """Calculate importance weights for amenities"""
        # Define amenity importance weights
        importance_weights = {
            'wifi': 0.9,
            'parking': 0.8,
            'kitchen': 0.7,
            'pool': 0.6,
            'gym': 0.5,
            'pet_friendly': 0.8,
            'accessibility': 0.9
        }
        
        amenity_scores = {}
        for amenity in amenities:
            amenity_scores[amenity] = importance_weights.get(amenity, 0.5)
        
        return amenity_scores
    
    def _get_price_context(self, numeric_constraints: Dict) -> Dict[str, Any]:
        """Get context based on price constraints"""
        price_constraint = numeric_constraints.get('price')
        if not price_constraint:
            return {}
        
        price_value = price_constraint.get('value', 0)
        
        if price_value < 50:
            category = 'budget'
        elif price_value < 150:
            category = 'mid_range'
        elif price_value < 300:
            category = 'premium'
        else:
            category = 'luxury'
        
        return {
            'price_category': category,
            'price_sensitivity': 'high' if price_constraint.get('operator') in ['<=', '<'] else 'low'
        }
    
    def _get_session_enhancement(self, session_context: Optional[Dict]) -> Dict[str, Any]:
        """Get enhancements from session context"""
        if not session_context:
            return {}
        
        return {
            'previous_searches': session_context.get('search_history', []),
            'accumulated_preferences': session_context.get('preferences', {}),
            'refinement_pattern': session_context.get('refinement_pattern', 'exploration')
        }
    
    def _enhance_query_with_context(self, query: str, entities: Dict, 
                                  numeric_constraints: Dict, contextual_info: Dict) -> str:
        """Enhance query with contextual information"""
        enhanced_parts = [query]
        
        # Add property type context
        prop_context = contextual_info.get('property_type_context', {})
        if prop_context.get('default_constraints'):
            for constraint_type, default_value in prop_context['default_constraints'].items():
                if constraint_type not in numeric_constraints:
                    if isinstance(default_value, list):
                        enhanced_parts.append(f"{constraint_type} between {default_value[0]} and {default_value[1]}")
                    else:
                        enhanced_parts.append(f"{constraint_type} {default_value}")
        
        # Add guest context recommendations
        guest_context = contextual_info.get('guest_context', {})
        if guest_context.get('recommended_bedrooms') and 'bedrooms' not in numeric_constraints:
            enhanced_parts.append(f"bedrooms {guest_context['recommended_bedrooms']}")
        
        return ' '.join(enhanced_parts)
    
    def _generate_database_filters(self, numeric_constraints: Dict, entities: Dict) -> Dict[str, Any]:
        """Generate MongoDB-compatible filters"""
        filters = {}
        
        # Process numeric constraints
        for constraint_type, constraint in numeric_constraints.items():
            field_names = constraint.get('field_names', [constraint_type])
            
            mongo_filter = self._create_mongo_constraint(constraint)
            if mongo_filter:
                # Use OR logic for multiple field names
                if len(field_names) > 1:
                    filters[f"{constraint_type}_constraint"] = {
                        '$or': [{field: mongo_filter} for field in field_names]
                    }
                else:
                    filters[field_names[0]] = mongo_filter
        
        # Process entity constraints
        if entities.get('property_types'):
            filters['property_type'] = {'$in': entities['property_types']}
        
        if entities.get('locations'):
            # Create location filter (city, neighborhood, etc.)
            location_filters = []
            for location in entities['locations']:
                location_filters.extend([
                    {'city': {'$regex': location, '$options': 'i'}},
                    {'neighborhood': {'$regex': location, '$options': 'i'}},
                    {'address': {'$regex': location, '$options': 'i'}}
                ])
            if location_filters:
                filters['location_constraint'] = {'$or': location_filters}
        
        return filters
    
    def _create_mongo_constraint(self, constraint: Dict) -> Optional[Dict]:
        """Create MongoDB constraint from processed constraint"""
        operator = constraint.get('operator')
        value = constraint.get('value')
        
        if operator == '=':
            return value
        elif operator == '>=':
            return {'$gte': value}
        elif operator == '<=':
            return {'$lte': value}
        elif operator == 'between':
            min_val = constraint.get('min_value', value)
            max_val = constraint.get('max_value', value)
            return {'$gte': min_val, '$lte': max_val}
        
        return None
    
    def _determine_search_strategy(self, intent: str, entities: Dict, 
                                 numeric_constraints: Dict) -> Dict[str, Any]:
        """Determine optimal search strategy based on query analysis"""
        strategy = {
            'method': 'hybrid',
            'weights': {'semantic': 0.6, 'fuzzy': 0.2, 'keyword': 0.2},
            'filters_first': False,
            'boost_factors': {}
        }
        
        # Adjust based on intent
        if intent in ['specific_search', 'capacity_search']:
            strategy['filters_first'] = True
            strategy['weights'].update({'semantic': 0.4, 'keyword': 0.6})
        
        # Adjust based on constraint complexity
        constraint_count = len(numeric_constraints)
        if constraint_count > 2:
            strategy['filters_first'] = True
            strategy['weights']['keyword'] = min(0.8, 0.2 + 0.2 * constraint_count)
        
        # Adjust based on entities
        entity_count = sum(len(entity_list) for entity_list in entities.values())
        if entity_count > 3:
            strategy['weights']['semantic'] = min(0.9, 0.6 + 0.1 * entity_count)
        
        # Set boost factors for important fields
        if numeric_constraints.get('price'):
            strategy['boost_factors']['price_match'] = 1.5
        
        if entities.get('property_types'):
            strategy['boost_factors']['property_type_match'] = 1.3
        
        return strategy
