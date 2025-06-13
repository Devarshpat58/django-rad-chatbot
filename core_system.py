import logging
import re
import json
import time
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from collections import defaultdict

try:
    from fuzzywuzzy import fuzz
except ImportError:
    fuzz = None

try:
    from sentence_transformers import SentenceTransformer
    import torch
except ImportError:
    SentenceTransformer = None
    torch = None

try:
    import spacy
except ImportError:
    spacy = None

from config import Config
from logging_config import StructuredLogger
from utils import SessionManager, ConversationTurn
from consolidated_config import FIELD_CATEGORIES
from utils import AirbnbOptimizer, TextProcessor, KeywordExtractor
from utils import IndexManager
from consolidated_config import RAG_SYSTEM_CONFIG as AIRBNB_CONFIG
# from utils import AirbnbDataOptimizer  # Class defined locally in this file

# Set up logging
# Enhanced logging setup  
from logging_config import setup_logging, StructuredLogger, LogOperation, log_performance

logger = StructuredLogger(__name__, {'component': 'core_system', 'source_module': 'rag_engine'})

# Log core system module initialization
logger.info("Core system module initialized", 
           source_module="core_system", 
           classes=["QueryUnderstandingEngine", "NumericSearchEngine", "SemanticSearchEngine", 
                    "AIJSONSummarizer", "SummaryGenerator", "ResponseGenerator", "JSONRAGSystem"])

class QueryUnderstandingEngine:
    """Advanced NLP-powered query understanding and intent analysis"""
    
    def __init__(self):
        self.logger = StructuredLogger(__name__, {'class_name': self.__class__.__name__})
        self.logger.info("Initializing QueryUnderstandingEngine", 
                        module="core_system",
                        class_name="QueryUnderstandingEngine", component="QueryUnderstandingEngine")
        self.text_processor = TextProcessor()
        self.keyword_extractor = KeywordExtractor()
        self.airbnb_optimizer = AirbnbOptimizer()
        logger.debug("Core processors initialized", 
                    component="QueryUnderstandingEngine",
                    modules=["TextProcessor", "KeywordExtractor", "AirbnbOptimizer"])
        
        # NLP models
        self.sentence_model = None
        self.nlp_model = None
        self.model_loaded = False
        
        # Intent classification patterns enhanced with NLP
        self.intent_embeddings = {}
        self.entity_patterns = self._build_entity_patterns()
        
        # Load NLP models
        self._load_nlp_models()
    
    def _load_nlp_models(self):
        """Load NLP models for advanced query understanding"""
        logger.info("Loading NLP models", component="QueryUnderstandingEngine", method="_load_nlp_models")
        try:
            # Load sentence transformer for semantic similarity
            logger.debug("Attempting to load SentenceTransformer model", component="QueryUnderstandingEngine")
            if SentenceTransformer is not None:
                # Force CPU to avoid meta tensor issues with device transfers
                import torch
                device = 'cpu'
                self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
                
                # Ensure model parameters are properly transferred from meta tensors
                if hasattr(self.sentence_model, '_modules'):
                    for module in self.sentence_model.modules():
                        if hasattr(module, 'weight') and hasattr(module.weight, 'is_meta') and module.weight.is_meta:
                            # Use to_empty() for meta tensor handling
                            try:
                                module.to_empty(device=device)
                            except:
                                pass  # Skip to_empty to avoid meta tensor issues
                
                if hasattr(self.sentence_model, 'eval'):
                    self.sentence_model.eval()
                logger.info("Sentence transformer model loaded successfully", 
                           component="QueryUnderstandingEngine", 
                           model="all-MiniLM-L6-v2")

                
                # Pre-compute intent embeddings
                self._compute_intent_embeddings()
            
            # Load spacy model if available
            if spacy is not None:
                try:
                    self.nlp_model = spacy.load('en_core_web_sm')
                    logger.info("SpaCy model loaded for NLP processing")
                except OSError:
                    logger.warning("SpaCy English model not found, using pattern-based NLP")
                    self.nlp_model = None
            
            self.model_loaded = True
            
        except Exception as e:
            logger.warning(f"Could not load NLP models: {e}. Using pattern-based analysis.")
            self.model_loaded = False
    
    def _compute_intent_embeddings(self):
        """Pre-compute embeddings for intent classification"""
        if self.sentence_model is None:
            return
        
        intent_examples = {
            'search': [
                'find properties', 'search for places', 'show me listings', 
                'get accommodations', 'look for rentals', 'list available options'
            ],
            'filter': [
                'with amenities', 'having features', 'that includes', 
                'containing facilities', 'must have'
            ],
            'compare': [
                'compare properties', 'which is better', 'difference between', 
                'versus options', 'best choice'
            ],
            'recommend': [
                'recommend places', 'suggest properties', 'good for', 
                'best options', 'advice on'
            ],
            'price': [
                'cost information', 'pricing details', 'budget options', 
                'expensive properties', 'affordable places'
            ],
            'location': [
                'where is located', 'area information', 'neighborhood details', 
                'near landmarks', 'distance from'
            ]
        }
        
        try:
            for intent, examples in intent_examples.items():
                # Compute average embedding for each intent
                embeddings = self.sentence_model.encode(examples)
                avg_embedding = np.mean(embeddings, axis=0)
                self.intent_embeddings[intent] = avg_embedding
            
            logger.info("Intent embeddings computed successfully")
            
        except Exception as e:
            self.logger.error("Error computing intent embeddings",
                             module="core_system",
                             class_name="QueryUnderstandingEngine",
                             method="_compute_intent_embeddings",
                             error=str(e),
                             exc_info=True)
    
    def _build_entity_patterns(self):
        """Build comprehensive entity recognition patterns"""
        return {
            'price': [
                r'\$?\d+(?:\.\d{2})?',
                r'\d+\s*dollars?',
                r'\d+\s*bucks?',
                r'under\s+\$?\d+',
                r'over\s+\$?\d+',
                r'between\s+\$?\d+\s+and\s+\$?\d+'
            ],
            'capacity': [
                r'\d+\s*(?:people|person|guests?|pax)',
                r'accommodate\s+\d+',
                r'sleep\s+\d+',
                r'fits?\s+\d+'
            ],
            'bedrooms': [
                r'\d+\s*(?:bedroom|bed|br)\b',
                r'\d+\s*bed\s*room',
                r'studio\s*apartment'
            ],
            'bathrooms': [
                r'\d+\s*(?:bathroom|bath|ba)\b',
                r'\d+\s*bath\s*room',
                r'half\s*bath'
            ],
            'location': [
                r'in\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*',
                r'near\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*',
                r'downtown',
                r'city\s*center',
                r'beach\s*front',
                r'airport'
            ],
            'amenities': [
                r'\bwifi\b|\binternet\b',
                r'\bparking\b|\bgarage\b',
                r'\bpool\b|\bswimming\b',
                r'\bkitchen\b|\bcooking\b',
                r'\bgym\b|\bfitness\b',
                r'\bac\b|\bair\s*conditioning\b'
            ],
            'rating': [
                r'\d+\s*stars?',
                r'highly\s*rated',
                r'good\s*reviews?',
                r'rating\s*above\s*\d+',
                r'\d+\s*out\s*of\s*\d+'
            ]
        }
    
    def analyze_query(self, query: str, session_context: str = "") -> Dict[str, Any]:
        """Comprehensive NLP-powered query analysis with advanced numeric processing"""
        # Try advanced query processing first
        try:
            from query_processor import QueryProcessor
            if not hasattr(self, 'query_processor'):
                self.query_processor = QueryProcessor()
            
            # Convert session_context to proper format
            context_dict = None
            if session_context:
                context_dict = {
                    'search_history': [session_context],
                    'preferences': {},
                    'refinement_pattern': 'exploration'
                }
            
            # Use advanced processing
            processed_query = self.query_processor.process_query(query, context_dict)
            
            analysis = {
                'original_query': query,
                'cleaned_query': processed_query.cleaned_query,
                'intent': processed_query.intent,
                'entities': self._convert_entities_format(processed_query.entities),
                'keywords': (list(processed_query.entities.get('property_types', [])) + 
                           list(processed_query.entities.get('amenities', []))) if processed_query.entities else [],
                'semantic_features': processed_query.contextual_info,
                'numeric_constraints': processed_query.numeric_constraints,
                'context_enhanced_query': processed_query.enhanced_query,
                'query_expansion': self._expand_query_nlp(processed_query.entities.get('keywords', []), query),
                'confidence_score': processed_query.confidence,
                'nlp_entities': processed_query.entities,
                'semantic_intent_score': processed_query.confidence,
                'database_filters': processed_query.database_filters,
                'search_strategy': processed_query.search_strategy,
                'advanced_processing': True
            }
            
            logger.info(f"Advanced query analysis completed - Intent: {analysis['intent']}, Numeric constraints: {len(analysis['numeric_constraints'])}")
            return analysis
            
        except ImportError:
            logger.warning("Advanced query processor not available, using fallback")
        except Exception as e:
            self.logger.error("Error in advanced query processing, falling back to basic processing",
                             module="core_system",
                             class_name="QueryUnderstandingEngine", 
                             method="understand_query",
                             error=str(e),
                             query_length=len(query) if query else 0,
                             exc_info=True)
        
        # Fallback to original processing
        analysis = {
            'original_query': query,
            'cleaned_query': '',
            'intent': '',
            'entities': [],
            'keywords': [],
            'semantic_features': {},
            'numeric_constraints': {},
            'context_enhanced_query': '',
            'query_expansion': [],
            'confidence_score': 0.0,
            'nlp_entities': [],
            'semantic_intent_score': 0.0,
            'advanced_processing': False
        }
        
        # Basic preprocessing
        cleaned_query = self.text_processor.clean_text(query)
        analysis['cleaned_query'] = cleaned_query
        
        # NLP-based intent classification
        if self.model_loaded:
            analysis['intent'], analysis['semantic_intent_score'] = self._classify_intent_nlp(cleaned_query)
        else:
            analysis['intent'] = self._classify_intent_pattern(cleaned_query)
            analysis['semantic_intent_score'] = 0.5
        
        # Enhanced entity extraction with NLP
        if self.nlp_model is not None:
            analysis['nlp_entities'] = self._extract_entities_nlp(cleaned_query)
        
        analysis['entities'] = self._extract_entities_enhanced(cleaned_query)
        
        # Keyword extraction
        analysis['keywords'] = self.keyword_extractor.get_query_keywords(cleaned_query)
        
        # Enhanced semantic feature analysis
        analysis['semantic_features'] = self._analyze_semantic_features_enhanced(cleaned_query)
        
        # Basic numeric constraint extraction
        analysis['numeric_constraints'] = self._extract_basic_numeric_constraints(cleaned_query)
        
        # Context enhancement with NLP understanding
        if session_context:
            analysis['context_enhanced_query'] = self._enhance_with_context_nlp(cleaned_query, session_context)
        else:
            analysis['context_enhanced_query'] = cleaned_query
        
        # Query expansion using NLP similarity
        analysis['query_expansion'] = self._expand_query_nlp(analysis['keywords'], query)
        
        # Calculate enhanced confidence score
        analysis['confidence_score'] = self._calculate_confidence_enhanced(analysis)
        
        return analysis
    
    def _convert_entities_format(self, entities_dict: Dict[str, List[str]]) -> List[str]:
        """Convert new entity format to legacy format for compatibility"""
        entity_list = []
        if entities_dict is None:
            return entity_list
        for entity_type, entity_list_items in entities_dict.items():
            entity_list.extend(entity_list_items)
        return entity_list
    
    def _extract_query_expansion(self, processed_query) -> List[str]:
        """Extract query expansion terms from processed query"""
        expansion_terms = []
        
        # Add synonyms from entities
        if hasattr(processed_query, 'entities') and processed_query.entities is not None:
            for entity_type, entities in processed_query.entities.items():
                expansion_terms.extend(entities)
        
        # Add contextual terms
        if hasattr(processed_query, 'contextual_info') and processed_query.contextual_info is not None:
            context = processed_query.contextual_info
            if 'guest_context' in context and context['guest_context'] is not None:
                expansion_terms.append(context['guest_context'].get('group_size', ''))
        
        return [term for term in expansion_terms if term]
    
    def _extract_basic_numeric_constraints(self, query: str) -> Dict[str, Any]:
        """Extract basic numeric constraints as fallback"""
        constraints = {}
        
        # Basic price extraction
        price_pattern = r'\$\s*(\d+(?:,\d{3})*(?:\.\d{2})?)|(?:under|below)\s*\$?\s*(\d+)|(?:over|above)\s*\$?\s*(\d+)'
        price_matches = re.findall(price_pattern, query, re.IGNORECASE)
        if price_matches:
            for match in price_matches:
                price_val = next((val for val in match if val), None)
                if price_val:
                    try:
                        constraints['price'] = {
                            'value': float(price_val.replace(',', '')),
                            'operator': '<=',
                            'field_names': ['price', 'cost']
                        }
                        break
                    except ValueError:
                        continue
        
        # Basic bedroom extraction
        bedroom_pattern = r'(\d+)\s*(?:bed|bedroom|bedrooms|br)s?\b'
        bedroom_match = re.search(bedroom_pattern, query, re.IGNORECASE)
        if bedroom_match:
            try:
                constraints['bedrooms'] = {
                    'value': int(bedroom_match.group(1)),
                    'operator': '=',
                    'field_names': ['bedrooms', 'beds']
                }
            except ValueError:
                pass
        
        # Basic guest extraction
        guest_pattern = r'(?:for|accommodate|sleeps)\s*(\d+)\s*(?:guest|guests|people|person)s?'
        guest_match = re.search(guest_pattern, query, re.IGNORECASE)
        if guest_match:
            try:
                constraints['guests'] = {
                    'value': int(guest_match.group(1)),
                    'operator': '>=',
                    'field_names': ['guests', 'accommodates']
                }
            except ValueError:
                pass
        
        return constraints
    
    def _classify_intent_nlp(self, query: str) -> Tuple[str, float]:
        """NLP-based intent classification using semantic similarity"""
        if not self.sentence_model or not self.intent_embeddings:
            return self._classify_intent_pattern(query), 0.5
        
        try:
            # Get query embedding
            query_embedding = self.sentence_model.encode([query])[0]
            
            # Calculate similarity with each intent
            intent_scores = {}
            for intent, intent_embedding in self.intent_embeddings.items():
                # Cosine similarity
                similarity = np.dot(query_embedding, intent_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(intent_embedding)
                )
                intent_scores[intent] = similarity
            
            # Get best matching intent
            best_intent = max(intent_scores, key=intent_scores.get)
            best_score = intent_scores[best_intent]
            
            # Fall back to pattern matching if confidence is low
            if best_score < 0.3:
                return self._classify_intent_pattern(query), best_score
            
            return best_intent, best_score
            
        except Exception as e:
            self.logger.error("Error in NLP intent classification",
                             module="core_system",
                             class_name="QueryUnderstandingEngine",
                             method="_classify_intent_nlp",
                             error=str(e),
                             query_length=len(query) if query else 0,
                             exc_info=True)
            return self._classify_intent_pattern(query), 0.5
    
    def _extract_entities_nlp(self, query: str) -> List[Dict[str, Any]]:
        """Extract entities using SpaCy NLP model"""
        if not self.nlp_model:
            return []
        
        try:
            doc = self.nlp_model(query)
            entities = []
            
            for ent in doc.ents:
                entities.append({
                    'text': ent.text,
                    'label': ent.label_,
                    'start': ent.start_char,
                    'end': ent.end_char,
                    'description': spacy.explain(ent.label_) if hasattr(spacy, 'explain') else ent.label_
                })
            
            return entities
            
        except Exception as e:
            self.logger.error("Error in NLP entity extraction",
                             module="core_system",
                             class_name="QueryUnderstandingEngine",
                             method="_extract_entities_nlp",
                             error=str(e),
                             query_length=len(query) if query else 0,
                             exc_info=True)
            return []
    
    def _extract_entities_enhanced(self, query: str) -> List[str]:
        """Enhanced entity extraction combining patterns and NLP"""
        entities = []
        query_lower = query.lower()
        
        # Pattern-based extraction
        for entity_type, patterns in self.entity_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, query_lower, re.IGNORECASE)
                if matches:
                    for match in matches:
                        if isinstance(match, tuple):
                            match = match[0] if match[0] else match[1]
                        entities.append(f"{entity_type}:{match}")
        
        # Add property types from vocabulary
        for prop_type, variants in AIRBNB_CONFIG['property_mappings'].items():
            for variant in variants:
                if variant in query_lower:
                    entities.append(f"property_type:{prop_type}")
                    break
        
        # Add amenities from vocabulary
        for amenity_cat, terms in AIRBNB_CONFIG['amenity_categories'].items():
            for term in terms:
                if term in query_lower:
                    entities.append(f"amenity:{amenity_cat}")
                    break
        
        return list(set(entities))  # Remove duplicates
    
    def _analyze_semantic_features_enhanced(self, query: str) -> Dict[str, Any]:
        """Enhanced semantic feature analysis with NLP insights"""
        features = {
            'length': len(query.split()),
            'specificity': 0.0,
            'urgency': 0.0,
            'sentiment': 'neutral',
            'complexity': 0.0,
            'semantic_density': 0.0,
            'entity_density': 0.0
        }
        
        query_lower = query.lower()
        words = query_lower.split()
        
        # Enhanced specificity calculation
        specific_indicators = [
            'exactly', 'specifically', 'must have', 'required', 'need', 'want',
            'looking for', 'prefer', 'only', 'just', 'precisely'
        ]
        specificity_count = sum(1 for ind in specific_indicators if ind in query_lower)
        features['specificity'] = min(specificity_count / max(len(words), 1), 1.0)
        
        # Enhanced urgency detection
        urgent_indicators = [
            'asap', 'urgent', 'immediately', 'quick', 'fast', 'soon', 'now',
            'today', 'tonight', 'weekend', 'emergency'
        ]
        urgency_count = sum(1 for ind in urgent_indicators if ind in query_lower)
        features['urgency'] = min(urgency_count / max(len(words), 1), 1.0)
        
        # Enhanced sentiment analysis
        positive_indicators = [
            'good', 'great', 'excellent', 'amazing', 'perfect', 'love', 'beautiful',
            'nice', 'wonderful', 'fantastic', 'awesome', 'best'
        ]
        negative_indicators = [
            'bad', 'terrible', 'awful', 'hate', 'avoid', 'worst', 'horrible',
            'disgusting', 'dirty', 'noisy', 'uncomfortable'
        ]
        
        pos_score = sum(1 for ind in positive_indicators if ind in query_lower)
        neg_score = sum(1 for ind in negative_indicators if ind in query_lower)
        
        if pos_score > neg_score:
            features['sentiment'] = 'positive'
        elif neg_score > pos_score:
            features['sentiment'] = 'negative'
        
        # Complexity based on conjunctions and constraints
        complexity_indicators = ['and', 'with', 'but', 'or', 'also', 'plus', 'near', 'close to']
        features['complexity'] = sum(1 for ind in complexity_indicators if ind in query_lower)
        
        # Semantic density (meaningful words ratio)
        stop_words = {'a', 'an', 'the', 'is', 'are', 'in', 'on', 'at', 'for', 'to', 'of'}
        meaningful_words = [w for w in words if w not in stop_words and len(w) > 2]
        features['semantic_density'] = len(meaningful_words) / max(len(words), 1)
        
        # Entity density (entities per word)
        entity_count = len(re.findall(r'\d+|\$\d+|(?:wifi|parking|pool|kitchen|downtown|beach)\b', query_lower))
        features['entity_density'] = entity_count / max(len(words), 1)
        
        return features
    
    def _enhance_with_context_nlp(self, query: str, context: str) -> str:
        """Enhanced context integration using NLP understanding"""
        if not context:
            return query
        
        # Extract important context using NLP if available
        if self.nlp_model:
            try:
                context_doc = self.nlp_model(context)
                important_context = []
                
                # Extract named entities and important nouns
                for token in context_doc:
                    if (token.pos_ in ['NOUN', 'PROPN'] and 
                        token.text.lower() not in ['user', 'system', 'property', 'place'] and
                        len(token.text) > 2):
                        important_context.append(token.text.lower())
                
                # Add entities
                for ent in context_doc.ents:
                    if ent.label_ in ['GPE', 'ORG', 'MONEY', 'CARDINAL']:
                        important_context.append(ent.text.lower())
                
                # Combine with current query
                if important_context:
                    context_terms = list(set(important_context))[:3]  # Limit context
                    # Ensure all context terms are strings
                    context_terms = [str(term) for term in context_terms if term is not None]
                    return f"{query} {' '.join(context_terms)}"
                
            except Exception as e:
                self.logger.error("Error in NLP context enhancement",
                                 module="core_system",
                                 class_name="QueryUnderstandingEngine",
                                 method="_enhance_query_with_context",
                                 error=str(e),
                                 query_length=len(query) if query else 0,
                                 exc_info=True)
        
        # Fallback to keyword-based context enhancement
        return self._enhance_with_context(query, context)
    
    def _expand_query_nlp(self, keywords: List[str], query: str = "") -> List[str]:
        """Enhanced query expansion using NLP similarity"""
        expanded = keywords.copy()
        
        # Use sentence transformer for semantic expansion if available
        if self.sentence_model and keywords:
            try:
                # Get embeddings for keywords
                keyword_embeddings = self.sentence_model.encode(keywords)
                
                # Expansion candidates from domain vocabulary
                candidates = list(AIRBNB_CONFIG['synonyms'].keys())
                if len(candidates) > 50:
                    candidates = candidates[:50]  # Limit for performance
                
                candidate_embeddings = self.sentence_model.encode(candidates)
                
                # Find semantically similar terms
                for i, keyword_emb in enumerate(keyword_embeddings):
                    similarities = np.dot(candidate_embeddings, keyword_emb) / (
                        np.linalg.norm(candidate_embeddings, axis=1) * np.linalg.norm(keyword_emb)
                    )
                    
                    # Add top similar terms
                    top_indices = np.argsort(similarities)[-2:]  # Top 2 similar
                    for idx in top_indices:
                        if similarities[idx] > 0.6:  # High similarity threshold
                            expanded.append(candidates[idx])
                
            except Exception as e:
                self.logger.error("Error in NLP query expansion",
                                 module="core_system",
                                 class_name="QueryUnderstandingEngine",
                                 method="_expand_query",
                                 error=str(e),
                                 query_length=len(query) if query else 0,
                                 exc_info=True)
        
        # Add predefined synonyms
        for keyword in keywords:
            if keyword in AIRBNB_CONFIG['synonyms']:
                expanded.extend(AIRBNB_CONFIG['synonyms'][keyword][:2])
        
        return list(set(expanded))
    
    def _calculate_confidence_enhanced(self, analysis: Dict[str, Any]) -> float:
        """Calculate enhanced confidence score with NLP factors"""
        confidence = 0.0
        
        # Base confidence from query structure
        if 5 <= len(analysis['keywords']) <= 15:
            confidence += 0.2
        elif len(analysis['keywords']) > 0:
            confidence += 0.1
        
        # NLP-based confidence factors
        if analysis['semantic_intent_score'] > 0.7:
            confidence += 0.3
        elif analysis['semantic_intent_score'] > 0.5:
            confidence += 0.2
        
        # Entity extraction confidence
        confidence += min(len(analysis['entities']) * 0.05, 0.2)
        
        # Numeric constraints boost confidence
        if analysis['numeric_constraints']:
            confidence += 0.2
        
        # Semantic feature contributions
        semantic_features = analysis['semantic_features']
        if semantic_features['specificity'] > 0.5:
            confidence += 0.1
        if semantic_features['semantic_density'] > 0.6:
            confidence += 0.1
        
        # NLP entities boost
        if analysis['nlp_entities']:
            confidence += min(len(analysis['nlp_entities']) * 0.03, 0.1)
        
        return min(confidence, 1.0)
    
    def _classify_intent_pattern(self, query: str) -> str:
        """Fallback pattern-based intent classification"""
        return self._classify_intent(query)
    
    def _classify_intent(self, query: str) -> str:
        """Classify query intent with enhanced patterns"""
        query_lower = query.lower()
        
        intent_patterns = {
            'search': ['find', 'search', 'look for', 'show me', 'get', 'list', 'want', 'need'],
            'filter': ['with', 'having', 'that has', 'including', 'contains'],
            'compare': ['compare', 'versus', 'vs', 'difference', 'better', 'best', 'which'],
            'recommend': ['recommend', 'suggest', 'advice', 'good', 'best for'],
            'info': ['tell me', 'what is', 'how', 'why', 'explain', 'describe'],
            'price': ['cost', 'price', 'expensive', 'cheap', 'budget', 'afford'],
            'location': ['where', 'location', 'area', 'near', 'close to', 'around']
        }
        
        intent_scores = {}
        for intent, patterns in intent_patterns.items():
            score = sum(2 if pattern in query_lower else 0 for pattern in patterns)
            if score > 0:
                intent_scores[intent] = score
        
        return max(intent_scores, key=intent_scores.get) if intent_scores else 'search'
    
    def _extract_entities(self, query: str) -> List[str]:
        """Extract named entities from query"""
        entities = []
        query_lower = query.lower()
        
        # Property types
        for prop_type, variants in AIRBNB_CONFIG['property_mappings'].items():
            for variant in variants:
                if variant in query_lower:
                    entities.append(f"property_type:{prop_type}")
                    break
        
        # Amenities
        for amenity_cat, terms in AIRBNB_CONFIG['amenity_categories'].items():
            for term in terms:
                if term in query_lower:
                    entities.append(f"amenity:{amenity_cat}")
                    break
        
        # Numeric entities
        bedroom_match = re.search(r'(\d+)\s*bedroom', query_lower)
        if bedroom_match:
            entities.append(f"bedrooms:{bedroom_match.group(1)}")
        
        bathroom_match = re.search(r'(\d+)\s*bathroom', query_lower)
        if bathroom_match:
            entities.append(f"bathrooms:{bathroom_match.group(1)}")
        
        price_match = re.search(r'\$(\d+)', query_lower)
        if price_match:
            entities.append(f"price:{price_match.group(1)}")
        
        return entities
    
    def _analyze_semantic_features(self, query: str) -> Dict[str, Any]:
        """Analyze semantic features of the query"""
        features = {
            'length': len(query.split()),
            'specificity': 0.0,
            'urgency': 0.0,
            'sentiment': 'neutral',
            'complexity': 0.0
        }
        
        query_lower = query.lower()
        
        # Specificity indicators
        specific_terms = ['exactly', 'specifically', 'must have', 'required', 'need']
        features['specificity'] = sum(1 for term in specific_terms if term in query_lower) / len(specific_terms)
        
        # Urgency indicators
        urgent_terms = ['asap', 'urgent', 'immediately', 'quick', 'fast']
        features['urgency'] = sum(1 for term in urgent_terms if term in query_lower) / len(urgent_terms)
        
        # Sentiment indicators
        positive_terms = ['good', 'great', 'excellent', 'amazing', 'perfect', 'love']
        negative_terms = ['bad', 'terrible', 'awful', 'hate', 'avoid']
        
        pos_score = sum(1 for term in positive_terms if term in query_lower)
        neg_score = sum(1 for term in negative_terms if term in query_lower)
        
        if pos_score > neg_score:
            features['sentiment'] = 'positive'
        elif neg_score > pos_score:
            features['sentiment'] = 'negative'
        
        # Complexity (number of constraints)
        complexity_indicators = ['and', 'with', 'but', 'or', 'also', 'plus']
        features['complexity'] = sum(1 for ind in complexity_indicators if ind in query_lower)
        
        return features
    
    def _enhance_with_context(self, query: str, context: str) -> str:
        """Enhance query with conversation context"""
        if not context:
            return query
        
        # Extract important terms from context
        context_keywords = self.keyword_extractor.get_query_keywords(context)
        query_keywords = self.keyword_extractor.get_query_keywords(query)
        
        # Add context keywords that don't conflict with current query
        enhanced_terms = []
        for keyword in context_keywords:
            if keyword not in query_keywords and len(keyword) > 2:
                enhanced_terms.append(keyword)
        
        if enhanced_terms:
            # Ensure all enhanced terms are strings
            enhanced_terms = [str(term) for term in enhanced_terms[:3] if term is not None]
            return f"{query} {' '.join(enhanced_terms)}"  # Limit to avoid noise
        
        return query
    
    def _expand_query(self, keywords: List[str]) -> List[str]:
        """Expand query with synonyms and related terms"""
        expanded = keywords.copy()
        
        for keyword in keywords:
            if keyword in AIRBNB_CONFIG['synonyms']:
                expanded.extend(AIRBNB_CONFIG['synonyms'][keyword][:2])  # Limit expansion
        
        return list(set(expanded))
    
    def _calculate_confidence(self, analysis: Dict[str, Any]) -> float:
        """Calculate confidence score for query analysis"""
        confidence = 0.0
        
        # Base confidence from query length and clarity
        if 5 <= len(analysis['keywords']) <= 15:
            confidence += 0.3
        elif len(analysis['keywords']) > 0:
            confidence += 0.1
        
        # Confidence from entity extraction
        confidence += min(len(analysis['entities']) * 0.1, 0.3)
        
        # Confidence from numeric constraints
        if analysis['numeric_constraints']:
            confidence += 0.2
        
        # Confidence from intent classification
        if analysis['intent'] != 'search':  # More specific intent
            confidence += 0.2
        
        return min(confidence, 1.0)

class NumericSearchEngine:
    """Enhanced numeric filtering with Airbnb-optimized constraint detection"""
    
    def __init__(self, vocabulary_manager=None):
        self.logger = StructuredLogger(__name__, {'class_name': self.__class__.__name__})
        self.logger.info("Initializing NumericSearchEngine", 
                        module="core_system",
                        class_name="NumericSearchEngine",
                        component="numeric_search")
        self.text_processor = TextProcessor()
        self.vocabulary_manager = vocabulary_manager
        self.logger.debug("NumericSearchEngine processors initialized", 
                    component="NumericSearchEngine",
                    has_vocabulary_manager=vocabulary_manager is not None)
        self.constraint_patterns = self._build_enhanced_patterns()
        # Airbnb-specific optimizations based on data understanding
        self.price_indicators = ['under', 'below', 'less', 'over', 'above', 'more', 'between', 'around']
        self.capacity_multipliers = {'couple': 2, 'family': 4, 'group': 6, 'party': 8}
        self.rating_thresholds = {'excellent': 90, 'good': 80, 'decent': 70, 'basic': 60}
    
    def _build_enhanced_patterns(self):
        """Build enhanced constraint patterns using vocabulary if available"""
        patterns = {
            'price': {
                'under': [r'under\s*\$?(\d+)', r'below\s*\$?(\d+)', r'less\s*than\s*\$?(\d+)'],
                'over': [r'over\s*\$?(\d+)', r'above\s*\$?(\d+)', r'more\s*than\s*\$?(\d+)'],
                'exact': [r'exactly\s*\$?(\d+)', r'\$?(\d+)\s*exactly'],
                'range': [r'\$?(\d+)\s*to\s*\$?(\d+)', r'between\s*\$?(\d+)\s*and\s*\$?(\d+)']
            },
            'bedrooms': {
                'exact': [r'(\d+)\s*bedroom', r'(\d+)\s*bed(?!room)', r'(\d+)\s*br\b'],
                'min': [r'at\s*least\s*(\d+)\s*bedroom', r'minimum\s*(\d+)\s*bedroom'],
                'max': [r'maximum\s*(\d+)\s*bedroom', r'up\s*to\s*(\d+)\s*bedroom']
            },
            'bathrooms': {
                'exact': [r'(\d+)\s*bathroom', r'(\d+)\s*bath(?!room)', r'(\d+)\s*ba\b'],
                'min': [r'at\s*least\s*(\d+)\s*bathroom'],
                'max': [r'maximum\s*(\d+)\s*bathroom']
            },
            'accommodates': {
                'exact': [r'(\d+)\s*guest', r'(\d+)\s*people', r'(\d+)\s*person'],
                'min': [r'accommodate\s*(\d+)', r'sleep\s*(\d+)', r'fit\s*(\d+)'],
                'max': [r'maximum\s*(\d+)\s*guest', r'up\s*to\s*(\d+)\s*people']
            },
            'rating': {
                'min': [r'rating\s*above\s*(\d+)', r'rating\s*over\s*(\d+)', r'highly\s*rated'],
                'exact': [r'rating\s*of\s*(\d+)', r'(\d+)\s*star']
            }
        }
        
        # Add vocabulary-derived patterns if available
        if (self.vocabulary_manager and 
            hasattr(self.vocabulary_manager, 'numeric_patterns') and
            isinstance(self.vocabulary_manager.numeric_patterns, dict)):
            for constraint_type, indicators in self.vocabulary_manager.numeric_patterns.items():
                field_name = constraint_type.replace('_indicators', '')
                if field_name == 'guest':
                    field_name = 'accommodates'
                elif field_name == 'bedroom':
                    field_name = 'bedrooms'
                elif field_name == 'bathroom':
                    field_name = 'bathrooms'
                
                if field_name in patterns:
                    # Add dynamic patterns based on vocabulary
                    for indicator in indicators[:3]:  # Limit to avoid pattern explosion
                        clean_indicator = indicator.lower().strip()
                        if len(clean_indicator) > 2:
                            # Add as exact match pattern
                            dynamic_pattern = f'(\\d+)\\s*{re.escape(clean_indicator)}'
                            if dynamic_pattern not in patterns[field_name]['exact']:
                                patterns[field_name]['exact'].append(dynamic_pattern)
        
        return patterns
    
    def extract_numeric_constraints(self, query: str) -> Dict[str, Any]:
        """Advanced numeric constraint extraction with intelligent parsing"""
        constraints = {}
        
        # Preprocess query for better extraction
        query_processed = self._preprocess_numeric_query(query)
        
        # Extract constraints using multiple strategies
        pattern_constraints = self._extract_using_patterns(query_processed)
        contextual_constraints = self._extract_contextual_constraints(query_processed)
        natural_constraints = self._extract_natural_language_numbers(query_processed)
        
        # Merge constraints with priority: patterns > contextual > natural
        constraints.update(natural_constraints)
        constraints.update(contextual_constraints)
        constraints.update(pattern_constraints)
        
        # Apply domain-specific intelligence for Airbnb data
        constraints = self._apply_domain_intelligence(constraints, query)
        
        return constraints
    
    def _preprocess_numeric_query(self, query: str) -> str:
        """Preprocess query for better numeric extraction"""
        query_lower = query.lower()
        
        # Normalize common phrases
        replacements = {
            'one': '1', 'two': '2', 'three': '3', 'four': '4', 'five': '5',
            'six': '6', 'seven': '7', 'eight': '8', 'nine': '9', 'ten': '10',
            'couple': 'accommodates 2', 'family': 'accommodates 4',
            'large group': 'accommodates 8', 'small group': 'accommodates 4',
            'single': '1 person', 'double': '2 people'
        }
        
        for old, new in replacements.items():
            query_lower = query_lower.replace(old, new)
        
        return query_lower
    
    def _extract_using_patterns(self, query: str) -> Dict[str, Any]:
        """Extract constraints using regex patterns"""
        constraints = {}
        
        for field, field_patterns in self.constraint_patterns.items():
            field_constraints = {}
            
            for constraint_type, patterns in field_patterns.items():
                for pattern in patterns:
                    try:
                        if constraint_type == 'range':
                            matches = re.findall(pattern, query, re.IGNORECASE)
                            if matches and len(matches[0]) >= 2:
                                if isinstance(matches[0], tuple):
                                    field_constraints['min'] = float(matches[0][0])
                                    field_constraints['max'] = float(matches[0][1])
                                break
                        else:
                            matches = re.findall(pattern, query, re.IGNORECASE)
                            if matches:
                                if pattern == r'highly\s*rated':  # Special case
                                    field_constraints['min'] = 80
                                else:
                                    value = float(matches[0])
                                    if constraint_type == 'under':
                                        field_constraints['max'] = value
                                    elif constraint_type == 'over':
                                        field_constraints['min'] = value
                                    else:
                                        field_constraints[constraint_type] = value
                                break
                    except (ValueError, IndexError) as e:
                        logger.debug(f"Pattern matching error: {e}")
                        continue
            
            if field_constraints:
                constraints[field] = field_constraints
        
        # Add context-aware constraint enhancement
        if self.vocabulary_manager:
            constraints = self._enhance_constraints_with_context(query, constraints)
        
        return constraints
    
    def _enhance_constraints_with_context(self, query: str, constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance constraints using vocabulary context"""
        enhanced_constraints = constraints.copy()
        
        # Look for contextual numeric hints in vocabulary
        query_terms = query.split()
        
        for term in query_terms:
            if term.isdigit():
                number_value = int(term)
                
                # Find context around the number
                term_index = query_terms.index(term)
                context_window = 3
                start_idx = max(0, term_index - context_window)
                end_idx = min(len(query_terms), term_index + context_window + 1)
                context_terms = query_terms[start_idx:end_idx]
                
                # Determine what the number refers to based on context
                for context_term in context_terms:
                    # Check vocabulary patterns
                    if (hasattr(self.vocabulary_manager, 'numeric_patterns') and
                        isinstance(self.vocabulary_manager.numeric_patterns, dict)):
                        for pattern_type, indicators in self.vocabulary_manager.numeric_patterns.items():
                            if any(indicator.lower() in context_term.lower() for indicator in indicators[:5]):
                                field_name = pattern_type.replace('_indicators', '')
                                if field_name == 'guest':
                                    field_name = 'accommodates'
                                elif field_name == 'bedroom':
                                    field_name = 'bedrooms'
                                elif field_name == 'bathroom':
                                    field_name = 'bathrooms'
                                
                                # Add constraint if not already present
                                if field_name not in enhanced_constraints:
                                    enhanced_constraints[field_name] = {'exact': number_value}
        
        return enhanced_constraints
    
    def _extract_contextual_constraints(self, query: str) -> Dict[str, Any]:
        """Extract constraints based on context and domain knowledge"""
        constraints = {}
        query_words = query.split()
        
        # Look for contextual clues
        for i, word in enumerate(query_words):
            if word.isdigit():
                value = int(word)
                context_words = []
                
                # Get surrounding context (2 words before and after)
                start_idx = max(0, i-2)
                end_idx = min(len(query_words), i+3)
                context_words = query_words[start_idx:end_idx]
                context_text = ' '.join(context_words)
                
                # Determine field and constraint type from context
                if any(term in context_text for term in ['bedroom', 'bed', 'br']):
                    constraints['bedrooms'] = {'exact': value}
                elif any(term in context_text for term in ['bathroom', 'bath', 'ba']):
                    constraints['bathrooms'] = {'exact': value}
                elif any(term in context_text for term in ['guest', 'people', 'person', 'accommodate']):
                    constraints['accommodates'] = {'exact': value}
                elif any(term in context_text for term in ['dollar', '$', 'price', 'cost']):
                    # Determine if it's a max or min based on context
                    if any(term in context_text for term in ['under', 'below', 'max', 'less']):
                        constraints['price'] = {'max': value}
                    elif any(term in context_text for term in ['over', 'above', 'min', 'more']):
                        constraints['price'] = {'min': value}
                    else:
                        constraints['price'] = {'exact': value}
        
        return constraints
    
    def _extract_natural_language_numbers(self, query: str) -> Dict[str, Any]:
        """Extract numbers from natural language expressions"""
        constraints = {}
        
        # Common natural language patterns
        natural_patterns = {
            'accommodates': [
                (r'for\s+(\d+)\s+people', 'exact'),
                (r'sleeps?\s+(\d+)', 'exact'),
                (r'fits?\s+(\d+)', 'exact'),
                (r'group\s+of\s+(\d+)', 'exact')
            ],
            'bedrooms': [
                (r'(\d+)\s*bed', 'exact'),
                (r'(\d+)\s*br\b', 'exact')
            ],
            'price': [
                (r'budget\s+of\s+\$?(\d+)', 'max'),
                (r'spend\s+\$?(\d+)', 'max'),
                (r'around\s+\$?(\d+)', 'exact')
            ]
        }
        
        for field, patterns in natural_patterns.items():
            for pattern, constraint_type in patterns:
                matches = re.findall(pattern, query, re.IGNORECASE)
                if matches:
                    value = int(matches[0])
                    constraints[field] = {constraint_type: value}
                    break
        
        return constraints
    
    def _apply_domain_intelligence(self, constraints: Dict[str, Any], query: str) -> Dict[str, Any]:
        """Apply Airbnb-specific domain intelligence to constraints"""
        enhanced = constraints.copy()
        query_lower = query.lower()
        
        # Apply capacity multipliers for group types
        for group_type, capacity in self.capacity_multipliers.items():
            if group_type in query_lower and 'accommodates' not in enhanced:
                enhanced['accommodates'] = {'min': capacity}
        
        # Apply rating thresholds for quality descriptors
        for quality, rating in self.rating_thresholds.items():
            if quality in query_lower and 'rating' not in enhanced:
                enhanced['rating'] = {'min': rating}
        
        # Intelligent price reasoning
        if 'luxury' in query_lower and 'price' not in enhanced:
            enhanced['price'] = {'min': 200}  # Assume luxury starts at $200
        elif 'budget' in query_lower and 'price' not in enhanced:
            enhanced['price'] = {'max': 100}  # Budget cap at $100
        
        # Smart bedroom-bathroom correlation
        if 'bedrooms' in enhanced and 'bathrooms' not in enhanced:
            bedroom_count = enhanced['bedrooms'].get('exact', enhanced['bedrooms'].get('min', 1))
            if bedroom_count >= 3:
                enhanced['bathrooms'] = {'min': 2}  # Large places likely have 2+ baths
        
        return enhanced
    
    def apply_constraints(self, documents: List[Dict[str, Any]], 
                        constraints: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Apply numeric constraints to filter documents"""
        if not constraints:
            return documents
        
        filtered_docs = []
        
        for doc in documents:
            if self._document_matches_constraints(doc, constraints):
                filtered_docs.append(doc)
        
        return filtered_docs
    
    def _document_matches_constraints(self, doc: Dict[str, Any], 
                                    constraints: Dict[str, Any]) -> bool:
        """Check if document matches all constraints"""
        original_doc = doc.get('original_document', doc)
        
        for field, field_constraints in constraints.items():
            if not self._field_matches_constraints(original_doc, field, field_constraints):
                return False
        
        return True
    
    def _field_matches_constraints(self, doc: Dict[str, Any], field: str, 
                                 field_constraints) -> bool:
        """Check if a specific field matches its constraints (lenient matching)"""
        # If field is missing or None, only fail for exact constraints
        if field not in doc or doc[field] is None:
            # For missing fields, only reject if there's an exact constraint
            if isinstance(field_constraints, dict) and 'exact' in field_constraints:
                return False
            # For min/max constraints on missing fields, assume it could match
            return True
        
        try:
            value = float(str(doc[field]).replace('$', '').replace(',', ''))
        except (ValueError, TypeError):
            # If field can't be converted to number, only fail for exact constraints
            if isinstance(field_constraints, dict) and 'exact' in field_constraints:
                return False
            # For non-numeric fields with min/max, assume they could match
            return True
        
        # Handle different constraint formats
        if isinstance(field_constraints, dict):
            # Standard constraint dictionary
            for constraint_type, constraint_value in field_constraints.items():
                if constraint_type == 'exact' and value != constraint_value:
                    return False
                elif constraint_type == 'min' and value < constraint_value:
                    return False
                elif constraint_type == 'max' and value > constraint_value:
                    return False
        else:
            # Simple numeric constraint (treat as max)
            try:
                max_value = float(field_constraints)
                if value > max_value:
                    return False
            except (ValueError, TypeError):
                return False
        
        return True

class SemanticSearchEngine:
    """Enhanced semantic search with vocabulary-based improvements"""
    
    def __init__(self, index_manager: IndexManager, vocabulary_manager=None):
        self.logger = StructuredLogger(__name__, {'class_name': self.__class__.__name__})
        self.logger.info("Initializing SemanticSearchEngine", 
                        module="core_system",
                        class_name="SemanticSearchEngine",
                        component="semantic_search",
                        index_manager=str(type(index_manager)), 
                        vocabulary_manager=str(type(vocabulary_manager)) if vocabulary_manager else "None")
        self.index_manager = index_manager
        self.vocabulary_manager = vocabulary_manager
        self.text_processor = TextProcessor()
    
    def semantic_search(self, query: str, k: int = Config.TOP_K_RESULTS) -> List[Tuple[Dict[str, Any], float]]:
        """Enhanced semantic search with query optimization and multi-stage retrieval"""
        logger.info("Starting semantic search", 
                   module="SemanticSearchEngine", 
                   query=query[:50], 
                   k=k)
        if not self.index_manager.faiss_index:
            logger.warning("FAISS index not available")
            return []
        
        # Stage 1: Query preprocessing and expansion
        enhanced_query = self._enhance_query(query)
        logger.debug(f"Enhanced query: {enhanced_query}")
        
        # Stage 2: Multi-query embedding generation
        query_embeddings = self._generate_multiple_embeddings(enhanced_query, query)
        if not query_embeddings:
            logger.warning("Could not generate query embeddings")
            return []
        
        # Stage 3: Multi-embedding search with score fusion
        all_results = []
        for i, (emb, weight) in enumerate(query_embeddings):
            stage_results = self._perform_faiss_search(emb, k * 2)  # Get more for diversity
            # Apply weight to scores
            weighted_results = [(doc, score * weight, f"stage_{i}") for doc, score in stage_results]
            all_results.extend(weighted_results)
        
        # Stage 4: Result fusion and re-ranking
        fused_results = self._fuse_and_rerank_results(all_results, query, k)
        
        logger.info("Semantic search completed", 
                   module="SemanticSearchEngine", 
                   query=query[:50], 
                   results_count=len(fused_results[:k]))
        return fused_results[:k]
    
    def _enhance_query(self, query: str) -> str:
        """Enhance query using vocabulary and domain knowledge"""
        enhanced_query = query.strip()
        
        # Vocabulary-based enhancement
        if self.vocabulary_manager and hasattr(self.vocabulary_manager, 'get_synonyms'):
            words = enhanced_query.lower().split()
            enhanced_words = []
            
            for word in words:
                enhanced_words.append(word)
                # Add synonyms for better semantic matching
                synonyms = self.vocabulary_manager.get_synonyms(word)
                if synonyms:
                    enhanced_words.extend(synonyms[:2])  # Add top 2 synonyms
            
            enhanced_query = ' '.join(enhanced_words)
        
        # Domain-specific enhancements
        domain_expansions = {
            'luxury': 'luxury premium high-end upscale exclusive',
            'budget': 'budget affordable cheap economical low-cost',
            'family': 'family child-friendly kids children spacious',
            'romantic': 'romantic couples intimate cozy private',
            'business': 'business work professional corporate wifi',
            'beach': 'beach waterfront oceanfront seaside coastal',
            'mountain': 'mountain hiking nature scenic outdoor',
            'city': 'city urban downtown central metropolitan'
        }
        
        query_lower = enhanced_query.lower()
        for keyword, expansion in domain_expansions.items():
            if keyword in query_lower:
                enhanced_query += f' {expansion}'
        
        return enhanced_query
    
    def _generate_multiple_embeddings(self, enhanced_query: str, original_query: str) -> List[Tuple[Any, float]]:
        """Generate multiple embeddings with different strategies"""
        embeddings = []
        
        # Primary embedding: enhanced query
        primary_emb = self.index_manager.embedding_generator.generate_embedding(enhanced_query)
        if primary_emb is not None:
            embeddings.append((primary_emb, 1.0))  # Highest weight
        
        # Secondary embedding: original query
        if original_query != enhanced_query:
            secondary_emb = self.index_manager.embedding_generator.generate_embedding(original_query)
            if secondary_emb is not None:
                embeddings.append((secondary_emb, 0.7))  # Lower weight
        
        # Tertiary embedding: key phrases only
        key_phrases = self._extract_key_phrases(original_query)
        if key_phrases and key_phrases != original_query:
            phrase_emb = self.index_manager.embedding_generator.generate_embedding(key_phrases)
            if phrase_emb is not None:
                embeddings.append((phrase_emb, 0.5))  # Lowest weight
        
        return embeddings
    
    def _extract_key_phrases(self, query: str) -> str:
        """Extract key phrases from query"""
        # Remove common stop words and extract meaningful terms
        stop_words = {'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'is', 'in', 'of', 'on', 'that', 'the', 'to', 'will', 'with'}
        words = [word.lower().strip('.,!?') for word in query.split() if word.lower() not in stop_words and len(word) > 2]
        return ' '.join(words[:5])  # Return top 5 key words
    
    def _perform_faiss_search(self, embedding: Any, k: int) -> List[Tuple[Dict[str, Any], float]]:
        """Perform FAISS search with error handling"""
        try:
            embedding = embedding.reshape(1, -1).astype('float32')
            scores, indices = self.index_manager.faiss_index.search(embedding, k)
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx >= len(self.index_manager.processed_documents) or idx < 0:
                    continue
                
                doc = self.index_manager.processed_documents[idx]
                results.append((doc, float(score)))
            
            return results
            
        except Exception as e:
            logger.error(f'Error in FAISS search: {e}')
            return []
    
    def _fuse_and_rerank_results(self, all_results: List[Tuple], query: str, k: int) -> List[Tuple[Dict[str, Any], float]]:
        """Fuse results from multiple embeddings and re-rank"""
        # Group results by document ID
        doc_scores = {}
        doc_data = {}
        
        for doc, score, stage in all_results:
            doc_id = doc.get('id', str(hash(str(doc))))
            
            if doc_id not in doc_scores:
                doc_scores[doc_id] = []
                doc_data[doc_id] = doc
            
            doc_scores[doc_id].append(score)
        
        # Calculate final scores using RRF (Reciprocal Rank Fusion)
        final_results = []
        for doc_id, scores in doc_scores.items():
            # Use max score with bonus for multiple matches
            max_score = max(scores)
            diversity_bonus = min(len(scores) * 0.1, 0.3)  # Cap bonus at 0.3
            final_score = max_score + diversity_bonus
            
            # Apply query relevance boost
            doc = doc_data[doc_id]
            relevance_boost = self._calculate_relevance_boost(doc, query)
            final_score *= (1.0 + relevance_boost)
            
            final_results.append((doc, final_score))
        
        # Sort by final score and return top results
        final_results.sort(key=lambda x: x[1], reverse=True)
        return final_results[:k * 2]  # Return extra for further filtering
    
    def _calculate_relevance_boost(self, doc: Dict[str, Any], query: str) -> float:
        """Calculate relevance boost based on query-document matching"""
        boost = 0.0
        query_lower = query.lower()
        
        # Check title match
        title = doc.get('title', '').lower()
        if any(word in title for word in query_lower.split()):
            boost += 0.2
        
        # Check description match
        description = doc.get('description', '').lower()
        if any(word in description for word in query_lower.split()):
            boost += 0.1
        
        # Check exact phrase match
        searchable_text = doc.get('searchable_text', '').lower()
        if query_lower in searchable_text:
            boost += 0.3
        
        return min(boost, 0.5)  # Cap boost at 0.5
    
    def fuzzy_search(self, query: str, documents: List[Dict[str, Any]], 
                    threshold: int = Config.FUZZY_THRESHOLD) -> List[Tuple[Dict[str, Any], float]]:
        """Enhanced fuzzy string matching with multi-field scoring"""
        if fuzz is None:
            logger.warning("fuzzywuzzy not available for fuzzy search")
            return []
        
        results = []
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        for doc in documents:
            # Multi-field fuzzy matching with weighted scores
            scores = []
            
            # Title matching (highest weight)
            title = doc.get('title', '').lower()
            if title:
                title_score = fuzz.partial_ratio(query_lower, title) / 100.0
                title_word_overlap = len(query_words.intersection(set(title.split()))) / max(len(query_words), 1)
                scores.append((title_score * 0.4) + (title_word_overlap * 0.3))
            
            # Description matching (medium weight)
            description = doc.get('description', '').lower()
            if description:
                desc_score = fuzz.partial_ratio(query_lower, description) / 100.0
                desc_word_overlap = len(query_words.intersection(set(description.split()))) / max(len(query_words), 1)
                scores.append((desc_score * 0.2) + (desc_word_overlap * 0.2))
            
            # Full text matching (lower weight)
            searchable_text = doc.get('searchable_text', '').lower()
            if searchable_text:
                text_score = fuzz.partial_ratio(query_lower, searchable_text) / 100.0
                text_word_overlap = len(query_words.intersection(set(searchable_text.split()))) / max(len(query_words), 1)
                scores.append((text_score * 0.1) + (text_word_overlap * 0.1))
            
            # Calculate final weighted fuzzy similarity
            if scores:
                final_score = max(scores)  # Take the best matching field
                # Boost score if multiple fields match well
                if len([s for s in scores if s > 0.3]) > 1:
                    final_score *= 1.2
                    
                # Convert to percentage for threshold comparison
                score_percentage = final_score * 100
                
                if score_percentage >= threshold:
                    results.append((doc, final_score))
        
        # Sort by similarity score (descending) and apply additional relevance scoring
        results.sort(key=lambda x: x[1], reverse=True)
        
        # Apply query-specific boosts
        enhanced_results = []
        for doc, score in results:
            boost = self._calculate_fuzzy_relevance_boost(doc, query)
            enhanced_score = min(score * (1.0 + boost), 1.0)  # Cap at 1.0
            enhanced_results.append((doc, enhanced_score))
        
        enhanced_results.sort(key=lambda x: x[1], reverse=True)
        return enhanced_results[:Config.TOP_K_RESULTS]
    
    def _calculate_fuzzy_relevance_boost(self, doc: Dict[str, Any], query: str) -> float:
        """Calculate additional relevance boost for fuzzy search results"""
        boost = 0.0
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        # Exact phrase match bonus
        searchable_text = doc.get('searchable_text', '').lower()
        if query_lower in searchable_text:
            boost += 0.3
        
        # Word frequency bonus
        text_words = set(searchable_text.split())
        word_overlap_ratio = len(query_words.intersection(text_words)) / max(len(query_words), 1)
        boost += word_overlap_ratio * 0.2
        
        # Title prominence bonus
        title = doc.get('title', '').lower()
        if any(word in title for word in query_words):
            boost += 0.2
        
        return min(boost, 0.4)  # Cap boost at 0.4
    
    def enhanced_keyword_search(self, query: str, documents: List[Dict[str, Any]]) -> List[Tuple[Dict[str, Any], float]]:
        """Advanced keyword search with TF-IDF-like scoring and field weighting"""
        keyword_extractor = KeywordExtractor()
        query_keywords = keyword_extractor.get_query_keywords(query)
        
        # Enhance keywords using vocabulary if available
        if self.vocabulary_manager:
            enhanced_keywords = self.vocabulary_manager.enhance_query_terms(query_keywords)
        else:
            enhanced_keywords = query_keywords
        
        if not enhanced_keywords:
            return []
        
        # Calculate IDF-like scores for keywords
        keyword_weights = self._calculate_keyword_weights(enhanced_keywords, documents)
        
        results = []
        
        for doc in documents:
            doc_score = self._calculate_document_keyword_score(doc, enhanced_keywords, keyword_weights, query)
            
            if doc_score > 0:
                results.append((doc, doc_score))
        
        # Sort and normalize scores
        results.sort(key=lambda x: x[1], reverse=True)
        if results:
            max_score = results[0][1]
            if max_score > 0:
                results = [(doc, score / max_score) for doc, score in results]
        
        return results[:Config.TOP_K_RESULTS]
    
    def _calculate_keyword_weights(self, keywords: List[str], documents: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate IDF-like weights for keywords based on document frequency"""
        keyword_weights = {}
        total_docs = len(documents)
        
        for keyword in keywords:
            doc_count = sum(1 for doc in documents 
                          if keyword.lower() in doc.get('searchable_text', '').lower())
            
            if doc_count > 0:
                # IDF-like calculation: log(total_docs / doc_frequency)
                import math
                weight = math.log(total_docs / doc_count)
                keyword_weights[keyword] = max(weight, 0.1)  # Minimum weight
            else:
                keyword_weights[keyword] = 1.0  # Default weight for rare terms
        
        return keyword_weights
    
    def _calculate_document_keyword_score(self, doc: Dict[str, Any], keywords: List[str], 
                                        keyword_weights: Dict[str, float], query: str) -> float:
        """Calculate comprehensive keyword-based score for a document"""
        total_score = 0.0
        
        # Field weights for different document sections
        field_configs = [
            ('title', 3.0),           # Highest weight
            ('description', 2.0),     # Medium weight  
            ('searchable_text', 1.0)  # Base weight
        ]
        
        matched_keywords = set()
        
        for field_name, field_weight in field_configs:
            field_text = doc.get(field_name, '').lower()
            if not field_text:
                continue
                
            field_score = 0.0
            field_words = set(field_text.split())
            
            for keyword in keywords:
                keyword_lower = keyword.lower()
                keyword_weight = keyword_weights.get(keyword, 1.0)
                
                # Exact keyword match
                if keyword_lower in field_words:
                    field_score += keyword_weight * 2.0
                    matched_keywords.add(keyword)
                
                # Partial keyword match (substring)
                elif keyword_lower in field_text:
                    field_score += keyword_weight * 1.0
                    matched_keywords.add(keyword)
                
                # Fuzzy matching for slight variations
                else:
                    for word in field_words:
                        if self._are_words_similar(keyword_lower, word):
                            field_score += keyword_weight * 0.5
                            matched_keywords.add(keyword)
                            break
            
            total_score += field_score * field_weight
        
        # Bonus for matching multiple keywords
        coverage_bonus = len(matched_keywords) / max(len(keywords), 1)
        total_score *= (1.0 + coverage_bonus * 0.3)
        
        # Bonus for exact query phrase match
        searchable_text = doc.get('searchable_text', '').lower()
        if query.lower() in searchable_text:
            total_score *= 1.5
        
        return total_score
    
    def _are_words_similar(self, word1: str, word2: str) -> bool:
        """Check if two words are similar (for fuzzy matching)"""
        # Simple similarity check based on common prefixes/suffixes
        if len(word1) < 3 or len(word2) < 3:
            return False
        
        # Check for common stem (first 3-4 characters)
        min_len = min(len(word1), len(word2))
        stem_len = min(4, min_len - 1)
        
        return word1[:stem_len] == word2[:stem_len] and abs(len(word1) - len(word2)) <= 2
    
    def hybrid_search(self, query: str, k: int = Config.TOP_K_RESULTS, documents: List[Dict[str, Any]] = None, weights: Dict[str, float] = None) -> List[Tuple[Dict[str, Any], float]]:
        """Advanced hybrid search with intelligent result fusion and adaptive weighting"""
        # Perform different types of searches with expanded result sets
        semantic_results = self.semantic_search(query, k * 2)
        # Use provided documents or default to processed documents
        search_documents = documents if documents is not None else self.index_manager.processed_documents
        
        fuzzy_results = self.fuzzy_search(query, search_documents, Config.FUZZY_THRESHOLD)
        keyword_results = self.enhanced_keyword_search(query, search_documents)
        
        # Use provided weights or calculate adaptive weights
        if weights is None:
            weights = self._calculate_adaptive_weights(query)
        else:
            # Ensure all required weight keys are present
            default_weights = self._calculate_adaptive_weights(query)
            for key in default_weights:
                if key not in weights:
                    weights[key] = default_weights[key]
        
        # Apply weights to different search types with safe unpacking
        weighted_results = {}
        
        # Handle semantic results
        if semantic_results:
            weighted_results['semantic'] = []
            for result in semantic_results:
                if len(result) >= 2:
                    doc, score = result[0], result[1]
                    weighted_results['semantic'].append((doc, score * weights['semantic'], 'semantic'))
        
        # Handle fuzzy results
        if fuzzy_results:
            weighted_results['fuzzy'] = []
            for result in fuzzy_results:
                if len(result) >= 2:
                    doc, score = result[0], result[1]
                    weighted_results['fuzzy'].append((doc, score * weights['fuzzy'], 'fuzzy'))
        
        # Handle keyword results
        if keyword_results:
            weighted_results['keyword'] = []
            for result in keyword_results:
                if len(result) >= 2:
                    doc, score = result[0], result[1]
                    weighted_results['keyword'].append((doc, score * weights['keyword'], 'keyword'))
        
        # Combine and deduplicate results
        combined_results = self._combine_search_results(weighted_results, query)
        
        # Apply final relevance scoring and ranking
        final_results = self._apply_final_ranking(combined_results, query)
        
        return final_results[:k]
    
    def _calculate_adaptive_weights(self, query: str) -> Dict[str, float]:
        """Calculate adaptive weights based on query characteristics"""
        query_lower = query.lower()
        query_words = query_lower.split()
        
        # Base weights
        weights = {
            'semantic': 0.5,
            'fuzzy': 0.3,
            'keyword': 0.2
        }
        
        # Adjust weights based on query characteristics
        
        # Short queries benefit more from fuzzy/keyword search
        if len(query_words) <= 2:
            weights['fuzzy'] += 0.15
            weights['keyword'] += 0.1
            weights['semantic'] -= 0.25
        
        # Long queries benefit more from semantic search
        elif len(query_words) >= 5:
            weights['semantic'] += 0.2
            weights['fuzzy'] -= 0.1
            weights['keyword'] -= 0.1
        
        # Specific domain terms favor keyword search
        domain_terms = ['bedrooms', 'bathrooms', 'price', 'guests', 'amenities', 'location']
        if any(term in query_lower for term in domain_terms):
            weights['keyword'] += 0.15
            weights['semantic'] -= 0.075
            weights['fuzzy'] -= 0.075
        
        # Quoted phrases favor fuzzy search
        if '"' in query or "'" in query:
            weights['fuzzy'] += 0.2
            weights['semantic'] -= 0.1
            weights['keyword'] -= 0.1
        
        # Normalize weights to sum to 1
        total_weight = sum(weights.values())
        for key in weights:
            weights[key] = weights[key] / total_weight
        
        return weights
    
    def _combine_search_results(self, weighted_results: Dict[str, List], query: str) -> List[Tuple[Dict[str, Any], float, List[str]]]:
        """Combine and deduplicate results from different search methods"""
        # Group results by document
        doc_results = {}
        
        for search_type, results in weighted_results.items():
            if not results:  # Skip empty results
                continue
            for result_tuple in results:
                if len(result_tuple) != 3:
                    logger.warning(f"Unexpected result tuple length in {search_type}: {len(result_tuple)}")
                    continue
                doc, score, method = result_tuple
                doc_id = doc.get('id', str(hash(str(doc))))
                
                if doc_id not in doc_results:
                    doc_results[doc_id] = {
                        'doc': doc,
                        'scores': [],
                        'methods': []
                    }
                
                doc_results[doc_id]['scores'].append(score)
                doc_results[doc_id]['methods'].append(method)
        
        # Calculate combined scores using multiple fusion strategies
        combined = []
        for doc_id, data in doc_results.items():
            doc = data['doc']
            scores = data['scores']
            methods = data['methods']
            
            # Reciprocal Rank Fusion (RRF) combined with weighted averaging
            max_score = max(scores)
            avg_score = sum(scores) / len(scores)
            
            # Diversity bonus for multiple search methods
            diversity_bonus = len(set(methods)) * 0.1
            
            # Final score combines max, average, and diversity
            final_score = (max_score * 0.6 + avg_score * 0.4) * (1.0 + diversity_bonus)
            
            combined.append((doc, final_score, methods))
        
        return combined
    
    def _apply_final_ranking(self, combined_results: List[Tuple], query: str) -> List[Tuple[Dict[str, Any], float]]:
        """Apply final relevance scoring and ranking"""
        final_results = []
        
        for result_item in combined_results:
            if not isinstance(result_item, (list, tuple)) or len(result_item) != 3:
                logger.warning(f"Invalid result item in combined_results: {type(result_item)}, length: {len(result_item) if hasattr(result_item, '__len__') else 'N/A'}")
                continue
            doc, score, methods = result_item
            # Apply query-specific relevance boosts
            relevance_boost = self._calculate_comprehensive_relevance(doc, query, methods)
            
            # Calculate confidence score based on search method agreement
            confidence_score = self._calculate_confidence_score(doc, query, methods)
            
            # Combine scores with confidence weighting
            final_score = score * (1.0 + relevance_boost) * confidence_score
            
            final_results.append((doc, final_score))
        
        # Sort by final score
        final_results.sort(key=lambda x: x[1], reverse=True)
        
        return final_results
    
    def _calculate_comprehensive_relevance(self, doc: Dict[str, Any], query: str, methods: List[str]) -> float:
        """Calculate comprehensive relevance boost"""
        boost = 0.0
        query_lower = query.lower()
        
        # Field-specific matching bonuses
        title = doc.get('title', '').lower()
        description = doc.get('description', '').lower()
        
        # Title relevance (highest impact)
        if any(word in title for word in query_lower.split()):
            boost += 0.3
            # Extra boost if it's an exact phrase match in title
            if query_lower in title:
                boost += 0.2
        
        # Description relevance
        if any(word in description for word in query_lower.split()):
            boost += 0.15
        
        # Method diversity bonus
        if len(set(methods)) >= 3:
            boost += 0.1
        elif len(set(methods)) == 2:
            boost += 0.05
        
        return min(boost, 0.6)  # Cap boost at 0.6
    
    def _calculate_confidence_score(self, doc: Dict[str, Any], query: str, methods: List[str]) -> float:
        """Calculate confidence score based on method agreement"""
        # Base confidence
        confidence = 0.8
        
        # Higher confidence if multiple methods agree
        unique_methods = len(set(methods))
        total_methods = len(methods)
        
        if unique_methods >= 2:
            # Multiple methods found this document
            agreement_ratio = total_methods / unique_methods
            confidence += min(agreement_ratio * 0.1, 0.2)
        
        # Text quality indicators
        searchable_text = doc.get('searchable_text', '')
        if len(searchable_text) > 100:  # Well-documented entries
            confidence += 0.05
        
        return min(confidence, 1.0)  # Cap at 1.0

class AIJSONSummarizer:
    """AI-powered JSON summarizer that uses transformer models to create intelligent summaries"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self._load_summarization_model()
    
    def _load_summarization_model(self):
        """Load a lightweight summarization model"""
        try:
            # Use sentence transformer for semantic understanding
            if SentenceTransformer is not None:
                # Force CPU to avoid meta tensor issues with device transfers
                import torch
                device = 'cpu'
                self.model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
                
                # Ensure model parameters are properly transferred from meta tensors
                if hasattr(self.model, '_modules'):
                    for module in self.model.modules():
                        if hasattr(module, 'weight') and hasattr(module.weight, 'is_meta') and module.weight.is_meta:
                            # Use to_empty() for meta tensor handling
                            try:
                                module.to_empty(device=device)
                            except:
                                pass  # Skip to_empty to avoid meta tensor issues
                
                if hasattr(self.model, 'eval'):
                    self.model.eval()
                logger.info("AI summarization model loaded")
            else:
                logger.warning("SentenceTransformer not available for AI summarization")
        except Exception as e:
            logger.error("Error loading AI summarization model",
                        module="core_system",
                        class_name="AIJSONSummarizer", 
                        method="__init__",
                        error=str(e),
                        exc_info=True)
    
    def generate_intelligent_summary(self, json_doc: Dict[str, Any], query: str = "", max_length: int = 400) -> str:
        """Generate an intelligent summary of JSON document using AI understanding"""
        try:
            # Extract key information from the JSON document
            key_info = self._extract_key_information(json_doc)
            
            # Analyze query to understand user intent
            query_focus = self._analyze_query_focus(query) if query else None
            
            # Generate contextual summary based on query focus
            if query_focus:
                summary = self._generate_contextual_summary(key_info, query_focus, max_length)
            else:
                summary = self._generate_general_summary(key_info, max_length)
            
            return summary
            
        except Exception as e:
            logger.error("Error in AI summarization",
                        module="core_system", 
                        class_name="AIJSONSummarizer",
                        method="generate_intelligent_summary",
                        error=str(e),
                        query_length=len(query) if query else 0,
                        exc_info=True)
            return self._fallback_summary(json_doc, query)
    
    def _extract_key_information(self, json_doc: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and categorize key information from JSON document"""
        key_info = {
            'identity': {},
            'description': {},
            'specifications': {},
            'location': {},
            'pricing': {},
            'ratings': {},
            'amenities': [],
            'host': {}
        }
        
        # Identity information
        if 'name' in json_doc and json_doc['name']:
            key_info['identity']['name'] = str(json_doc['name']).strip()
        if 'property_type' in json_doc:
            key_info['identity']['type'] = str(json_doc['property_type'])
        if 'room_type' in json_doc:
            key_info['identity']['room_type'] = str(json_doc['room_type'])
        
        # Description
        for field in ['summary', 'description', 'space']:
            if field in json_doc and json_doc[field]:
                text = str(json_doc[field]).strip()
                if len(text) > 50:  # Only meaningful descriptions
                    key_info['description'][field] = text[:800]  # Allow longer descriptions for better summaries
        
        # Specifications
        spec_fields = ['accommodates', 'bedrooms', 'bathrooms', 'beds']
        for field in spec_fields:
            if field in json_doc and json_doc[field] is not None:
                key_info['specifications'][field] = json_doc[field]
        
        # Location
        location_fields = ['neighbourhood_cleansed', 'city', 'zipcode']
        for field in location_fields:
            if field in json_doc and json_doc[field]:
                key_info['location'][field] = str(json_doc[field])
        
        # Pricing
        price_fields = ['price', 'cleaning_fee', 'minimum_nights', 'maximum_nights']
        for field in price_fields:
            if field in json_doc and json_doc[field] is not None:
                key_info['pricing'][field] = json_doc[field]
        
        # Ratings
        rating_fields = ['review_scores_rating', 'number_of_reviews']
        for field in rating_fields:
            if field in json_doc and json_doc[field] is not None:
                key_info['ratings'][field] = json_doc[field]
        
        # Amenities
        if 'amenities' in json_doc and json_doc['amenities']:
            if isinstance(json_doc['amenities'], list):
                key_info['amenities'] = [str(a) for a in json_doc['amenities'][:10]]  # Top 10 amenities
            else:
                amenities_str = str(json_doc['amenities'])
                key_info['amenities'] = [a.strip() for a in amenities_str.split(',')[:10]]
        
        # Host information
        host_fields = ['host_name', 'host_is_superhost', 'host_response_rate']
        for field in host_fields:
            if field in json_doc and json_doc[field] is not None:
                key_info['host'][field] = json_doc[field]
        
        return key_info
    
    def _analyze_query_focus(self, query: str) -> Dict[str, float]:
        """Analyze query to understand what aspects user is most interested in"""
        query_lower = query.lower()
        
        focus_keywords = {
            'pricing': ['price', 'cost', 'cheap', 'expensive', 'budget', 'affordable', '$', 'dollar', 'fee'],
            'location': ['location', 'area', 'neighbourhood', 'neighborhood', 'city', 'near', 'close', 'downtown'],
            'specifications': ['bedroom', 'bathroom', 'bed', 'accommodate', 'guest', 'people', 'size'],
            'amenities': ['wifi', 'kitchen', 'parking', 'pool', 'gym', 'balcony', 'amenity', 'feature'],
            'ratings': ['rating', 'review', 'score', 'quality', 'feedback', 'star'],
            'description': ['describe', 'about', 'detail', 'space', 'summary']
        }
        
        focus_scores = {}
        for category, keywords in focus_keywords.items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            if score > 0:
                focus_scores[category] = score
        
        # Normalize scores
        total_score = sum(focus_scores.values())
        if total_score > 0:
            focus_scores = {k: v/total_score for k, v in focus_scores.items()}
        
        return focus_scores
    
    def _generate_contextual_summary(self, key_info: Dict[str, Any], query_focus: Dict[str, float], max_length: int) -> str:
        """Generate a summary focused on query-relevant aspects"""
        summary_parts = []
        
        # Always start with identity
        if key_info['identity'].get('name'):
            summary_parts.append(f"**{key_info['identity']['name']}**")
        
        # Add type information
        type_info = []
        if key_info['identity'].get('type'):
            type_info.append(key_info['identity']['type'])
        if key_info['identity'].get('room_type'):
            type_info.append(key_info['identity']['room_type'])
        if type_info:
            summary_parts.append(f"({' - '.join(type_info)})")
        
        # Add content based on query focus (sorted by relevance)
        sorted_focus = sorted(query_focus.items(), key=lambda x: x[1], reverse=True)
        
        for category, score in sorted_focus:
            if score > 0.2:  # Only include significant focuses
                if category == 'pricing' and key_info['pricing']:
                    price_info = []
                    if 'price' in key_info['pricing']:
                        price_info.append(f"${key_info['pricing']['price']}/night")
                    if 'cleaning_fee' in key_info['pricing']:
                        price_info.append(f"${key_info['pricing']['cleaning_fee']} cleaning")
                    if price_info:
                        summary_parts.append(f"Price: {', '.join(price_info)}")
                
                elif category == 'specifications' and key_info['specifications']:
                    spec_info = []
                    for field, value in key_info['specifications'].items():
                        if field == 'accommodates':
                            spec_info.append(f"{value} guests")
                        elif field == 'bedrooms':
                            spec_info.append(f"{value} bedrooms")
                        elif field == 'bathrooms':
                            spec_info.append(f"{value} bathrooms")
                    if spec_info:
                        summary_parts.append(f"Specs: {', '.join(spec_info)}")
                
                elif category == 'location' and key_info['location']:
                    location_info = []
                    if 'neighbourhood_cleansed' in key_info['location']:
                        location_info.append(key_info['location']['neighbourhood_cleansed'])
                    if 'city' in key_info['location']:
                        location_info.append(key_info['location']['city'])
                    if location_info:
                        summary_parts.append(f"Location: {', '.join(location_info)}")
                
                elif category == 'amenities' and key_info['amenities']:
                    top_amenities = key_info['amenities'][:5]  # Top 5 amenities
                    summary_parts.append(f"Amenities: {', '.join(top_amenities)}")
                
                elif category == 'ratings' and key_info['ratings']:
                    rating_info = []
                    if 'review_scores_rating' in key_info['ratings']:
                        rating = key_info['ratings']['review_scores_rating']
                        rating_info.append(f"{rating}/100 rating")
                    if 'number_of_reviews' in key_info['ratings']:
                        reviews = key_info['ratings']['number_of_reviews']
                        rating_info.append(f"{reviews} reviews")
                    if rating_info:
                        summary_parts.append(f"Rating: {', '.join(rating_info)}")
        
        # Add key description if space allows and it's relevant
        if 'description' in query_focus and key_info['description']:
            desc_text = ""
            for field in ['summary', 'description', 'space']:
                if field in key_info['description']:
                    desc_text = key_info['description'][field]
                    break
            
            if desc_text:
                # Allow longer descriptions with increased buffer
                current_length = len(' | '.join(summary_parts))
                remaining_length = max_length - current_length - 50  # Larger buffer for longer summaries
                if remaining_length > 100:
                    truncated_desc = desc_text[:remaining_length] + "..." if len(desc_text) > remaining_length else desc_text
                    summary_parts.append(f"Description: {truncated_desc}")
        
        summary = ' | '.join(summary_parts)
        
        # Truncate if necessary
        if len(summary) > max_length:
            summary = summary[:max_length-3] + "..."
        
        return summary
    
    def _generate_general_summary(self, key_info: Dict[str, Any], max_length: int) -> str:
        """Generate a general summary when no specific query focus is identified"""
        summary_parts = []
        
        # Name and type
        if key_info['identity'].get('name'):
            summary_parts.append(f"**{key_info['identity']['name']}**")
        
        type_info = []
        if key_info['identity'].get('type'):
            type_info.append(key_info['identity']['type'])
        if key_info['identity'].get('room_type'):
            type_info.append(key_info['identity']['room_type'])
        if type_info:
            summary_parts.append(f"({' - '.join(type_info)})")
        
        # Key specifications
        if key_info['specifications']:
            spec_info = []
            for field, value in key_info['specifications'].items():
                if field == 'accommodates':
                    spec_info.append(f"{value} guests")
                elif field == 'bedrooms' and value:
                    spec_info.append(f"{value} bedrooms")
            if spec_info:
                summary_parts.append(f"Specs: {', '.join(spec_info)}")
        
        # Price if available
        if key_info['pricing'].get('price'):
            summary_parts.append(f"Price: ${key_info['pricing']['price']}/night")
        
        # Location
        if key_info['location'].get('neighbourhood_cleansed'):
            summary_parts.append(f"Location: {key_info['location']['neighbourhood_cleansed']}")
        
        # Rating if available
        if key_info['ratings'].get('review_scores_rating'):
            rating = key_info['ratings']['review_scores_rating']
            reviews = key_info['ratings'].get('number_of_reviews', 0)
            summary_parts.append(f"Rating: {rating}/100 ({reviews} reviews)")
        
        # Top amenities
        if key_info['amenities']:
            top_amenities = key_info['amenities'][:3]  # Top 3 for general summary
            summary_parts.append(f"Amenities: {', '.join(top_amenities)}")
        
        summary = ' | '.join(summary_parts)
        
        # Truncate if necessary
        if len(summary) > max_length:
            summary = summary[:max_length-3] + "..."
        
        return summary
    
    def summarize_json(self, json_doc: Dict[str, Any], query: str = "") -> str:
        """Main interface method for JSON summarization - generates intelligent summary from source JSON"""
        # Ensure we're working with the original source document from search results
        source_doc = json_doc.get('original_document', json_doc)
        
        # Use the longer, enhanced AI summarization with increased max length
        return self.generate_intelligent_summary(source_doc, query, max_length=400)
    
    def _fallback_summary(self, json_doc: Dict[str, Any], query: str) -> str:
        """Fallback summary method when AI processing fails"""
        summary_parts = []
        
        # Basic information
        if 'name' in json_doc and json_doc['name']:
            summary_parts.append(f"Property: {json_doc['name']}")
        
        if 'property_type' in json_doc:
            summary_parts.append(f"Type: {json_doc['property_type']}")
        
        if 'price' in json_doc and json_doc['price']:
            summary_parts.append(f"Price: ${json_doc['price']}")
        
        if 'accommodates' in json_doc:
            summary_parts.append(f"Guests: {json_doc['accommodates']}")
        
        return ' | '.join(summary_parts[:5])  # Limit to 5 key points


class SummaryGenerator:
    """Generates intelligent AI-powered summaries for properties using transformer models"""
    
    def __init__(self):
        self.text_processor = TextProcessor()
        self.ai_summarizer = AIJSONSummarizer()
    
    def generate_summary(self, document: Dict[str, Any], query: str = "") -> str:
        """Generate intelligent AI-powered query-aware summary for a property"""
        original_doc = document.get('original_document', document)
        
        try:
            # Use AI-powered intelligent summarization with increased max length
            ai_summary = self.ai_summarizer.generate_intelligent_summary(
                original_doc, 
                query, 
                max_length=500
            )
            
            if ai_summary and len(ai_summary.strip()) > 20:  # Require more substantial summaries
                return ai_summary
            else:
                # Fallback to basic summary if AI fails or summary too short
                return self._generate_basic_summary(original_doc, query)
                
        except Exception as e:
            logger.warning(f"AI summarization failed: {e}, using fallback")
            return self._generate_basic_summary(original_doc, query)
    
    def _generate_basic_summary(self, document: Dict[str, Any], query: str = "") -> str:
        """Generate basic fallback summary when AI processing fails"""
        summary_parts = []
        
        # Basic property information
        if 'name' in document and document['name']:
            summary_parts.append(f"**{document['name']}**")
        
        if 'property_type' in document:
            summary_parts.append(f"Type: {document['property_type']}")
        
        if 'price' in document and document['price']:
            summary_parts.append(f"${document['price']}/night")
        
        if 'accommodates' in document:
            summary_parts.append(f"{document['accommodates']} guests")
        
        if 'bedrooms' in document and document['bedrooms']:
            summary_parts.append(f"{document['bedrooms']} bedrooms")
        
        if 'neighbourhood_cleansed' in document and document['neighbourhood_cleansed']:
            summary_parts.append(f"Located in {document['neighbourhood_cleansed']}")
        
        return " | ".join(summary_parts) if summary_parts else "Property details available"
    
    def _analyze_query_topics(self, query: str) -> List[str]:
        """Identify topics the user is asking about"""
        query_lower = query.lower()
        topics = []
        
        topic_keywords = {
            'price': ['price', 'cost', 'expensive', 'cheap', 'budget', 'afford'],
            'location': ['location', 'area', 'neighborhood', 'where', 'near'],
            'amenities': ['amenity', 'wifi', 'parking', 'kitchen', 'pool', 'gym'],
            'capacity': ['bedroom', 'bed', 'bathroom', 'guest', 'people', 'accommodate'],
            'reviews': ['review', 'rating', 'score', 'feedback'],
            'host': ['host', 'owner', 'superhost']
        }
        
        for topic, keywords in topic_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                topics.append(topic)
        
        return topics
    
    def _generate_query_response(self, document: Dict[str, Any], topics: List[str]) -> str:
        """Generate Airbnb-optimized response based on query topics"""
        responses = []
        
        for topic in topics:
            if topic == 'price' and 'price' in document:
                price = str(document['price']).replace('$', '').replace(',', '')
                try:
                    price_val = float(price)
                    # Add context for Airbnb pricing
                    if price_val < 50:
                        price_context = " (Budget-friendly)"
                    elif price_val > 200:
                        price_context = " (Premium)"
                    else:
                        price_context = " (Mid-range)"
                    responses.append(f"Price: ${price}/night{price_context}")
                except ValueError:
                    responses.append(f"Price: ${price}/night")
            
            elif topic == 'location':
                location_parts = []
                if 'neighbourhood_cleansed' in document:
                    location_parts.append(document['neighbourhood_cleansed'])
                if 'neighbourhood_group_cleansed' in document:
                    location_parts.append(document['neighbourhood_group_cleansed'])
                if location_parts:
                    responses.append(f"Located in {', '.join(location_parts)}")
            
            elif topic == 'capacity':
                capacity_info = []
                if 'accommodates' in document:
                    capacity_info.append(f"accommodates {document['accommodates']} guests")
                if 'bedrooms' in document and document['bedrooms']:
                    capacity_info.append(f"{document['bedrooms']} bedrooms")
                if 'beds' in document and document['beds']:
                    capacity_info.append(f"{document['beds']} beds")
                if 'bathrooms' in document and document['bathrooms']:
                    capacity_info.append(f"{document['bathrooms']} bathrooms")
                if capacity_info:
                    responses.append(", ".join(capacity_info))
            
            elif topic == 'reviews':
                review_parts = []
                if 'review_scores_rating' in document and document['review_scores_rating']:
                    rating = document['review_scores_rating']
                    rating_desc = "Excellent" if rating >= 90 else "Good" if rating >= 80 else "Fair"
                    review_parts.append(f"Rating: {rating}/100 ({rating_desc})")
                if 'number_of_reviews' in document and document['number_of_reviews']:
                    review_parts.append(f"{document['number_of_reviews']} reviews")
                if review_parts:
                    responses.append(", ".join(review_parts))
            
            elif topic == 'amenities' and 'amenities' in document:
                amenities_str = str(document['amenities'])
                # Extract key amenities for Airbnb
                key_amenities = []
                amenities_lower = amenities_str.lower()
                for amenity in ['wifi', 'parking', 'pool', 'kitchen', 'gym', 'air conditioning']:
                    if amenity in amenities_lower:
                        key_amenities.append(amenity.title())
                
                if key_amenities:
                    responses.append(f"Key amenities: {', '.join(key_amenities)}")
                else:
                    amenities_preview = amenities_str[:80] + "..." if len(amenities_str) > 80 else amenities_str
                    responses.append(f"Amenities: {amenities_preview}")
            
            elif topic == 'host' and 'host_name' in document:
                host_info = [f"Host: {document['host_name']}"]
                if 'host_is_superhost' in document and document['host_is_superhost']:
                    host_info.append("(Superhost)")
                responses.append(" ".join(host_info))
        
        return "; ".join(responses)
    
    def _extract_key_information(self, document: Dict[str, Any]) -> str:
        """Extract key information for summary"""
        key_parts = []
        
        # Property type and room type
        if 'property_type' in document:
            key_parts.append(str(document['property_type']))
        if 'room_type' in document:
            key_parts.append(str(document['room_type']))
        
        # Capacity
        capacity_parts = []
        if 'bedrooms' in document and document['bedrooms']:
            capacity_parts.append(f"{document['bedrooms']}BR")
        if 'bathrooms' in document and document['bathrooms']:
            capacity_parts.append(f"{document['bathrooms']}BA")
        if capacity_parts:
            key_parts.append("/".join(capacity_parts))
        
        # Price
        if 'price' in document and document['price']:
            price_str = str(document['price']).replace('$', '')
            key_parts.append(f"${price_str}/night")
        
        return " • ".join(key_parts)

class AirbnbDataOptimizer:
    """Optimizes data presentation and search based on Airbnb property characteristics"""
    
    def __init__(self):
        self.field_categories = {
            'location': ['neighbourhood_cleansed', 'neighbourhood_group_cleansed', 'city', 'zipcode', 'latitude', 'longitude'],
            'pricing': ['price', 'cleaning_fee', 'extra_people', 'security_deposit', 'minimum_nights', 'maximum_nights'],
            'property': ['property_type', 'room_type', 'bed_type', 'accommodates', 'bedrooms', 'bathrooms', 'beds'],
            'host': ['host_id', 'host_name', 'host_since', 'host_response_time', 'host_response_rate', 'host_is_superhost', 'host_listings_count'],
            'booking': ['availability_365', 'cancellation_policy', 'minimum_nights', 'maximum_nights'],
            'reviews': ['number_of_reviews', 'review_scores_rating', 'review_scores_accuracy', 'review_scores_cleanliness', 'review_scores_checkin', 'review_scores_communication', 'review_scores_location', 'review_scores_value'],
            'general': ['name', 'summary', 'description', 'space', 'amenities']
        }
    
    def categorize_query(self, query: str) -> Dict[str, Any]:
        """Categorize query into primary and secondary categories"""
        query_lower = query.lower()
        
        category_keywords = {
            'location': ['location', 'area', 'neighbourhood', 'neighborhood', 'city', 'near', 'close', 'distance', 'where'],
            'pricing': ['price', 'cost', 'cheap', 'expensive', 'budget', 'afford', 'rate', 'fee', 'money'],
            'property': ['property', 'house', 'apartment', 'room', 'bedroom', 'bathroom', 'bed', 'type', 'size'],
            'host': ['host', 'owner', 'superhost', 'response', 'listings'],
            'booking': ['available', 'availability', 'book', 'reserve', 'nights', 'minimum', 'maximum', 'cancel'],
            'reviews': ['review', 'rating', 'score', 'feedback', 'guest', 'experience', 'quality']
        }
        
        scores = {}
        for category, keywords in category_keywords.items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            if score > 0:
                scores[category] = score
        
        if not scores:
            return {'primary_category': 'general', 'secondary_categories': []}
        
        # Primary category is the one with highest score
        primary_category = max(scores, key=scores.get)
        
        # Secondary categories are others with scores
        secondary_categories = [cat for cat, score in scores.items() 
                              if cat != primary_category and score > 0]
        
        return {
            'primary_category': primary_category,
            'secondary_categories': secondary_categories,
            'category_scores': scores
        }
    
    def optimize_document_summary(self, doc: Dict[str, Any], query_analysis: Dict[str, Any]) -> str:
        """Generate optimized summary based on query analysis"""
        primary_category = query_analysis.get('primary_category', 'general')
        
        # Get relevant fields for this category
        relevant_fields = self.field_categories.get(primary_category, self.field_categories['general'])
        
        summary_parts = []
        
        # Always include name if available
        if 'name' in doc and doc['name']:
            summary_parts.append(f"Property: {doc['name']}")
        
        # Add category-specific information
        for field in relevant_fields:
            if field in doc and doc[field] is not None:
                value = doc[field]
                # Format the value for display
                if isinstance(value, list):
                    if len(value) > 5:  # Truncate long lists
                        formatted_value = ', '.join(str(v) for v in value[:5]) + f' (+{len(value)-5} more)'
                    else:
                        formatted_value = ', '.join(str(v) for v in value)
                elif isinstance(value, (int, float)):
                    if field == 'price':
                        formatted_value = f'${value}'
                    elif 'rate' in field or 'rating' in field:
                        formatted_value = f'{value}' + ('/100' if field == 'review_scores_rating' else '')
                    else:
                        formatted_value = str(value)
                else:
                    # Truncate long text fields
                    str_value = str(value)
                    if len(str_value) > 150:
                        formatted_value = str_value[:150] + '...'
                    else:
                        formatted_value = str_value
                
                summary_parts.append(f"{field.replace('_', ' ').title()}: {formatted_value}")
        
        return ' | '.join(summary_parts[:8])  # Limit to 8 key points
    
    def get_response_template(self, query_analysis: Dict[str, Any]) -> str:
        """Get appropriate response template based on query analysis"""
        primary_category = query_analysis.get('primary_category', 'general')
        
        templates = {
            'location': "Based on the location data, here are the key insights:\n\n",
            'pricing': "Here's the pricing analysis for your query:\n\n",
            'property': "Property characteristics and features analysis:\n\n",
            'host': "Host and service quality information:\n\n",
            'booking': "Booking and availability insights:\n\n",
            'reviews': "Guest experience and review analysis:\n\n",
            'general': "Here are the relevant insights from the data:\n\n"
        }
        
        return templates.get(primary_category, templates['general'])
    
    def get_relevant_fields_for_json(self, query_analysis: Dict[str, Any]) -> List[str]:
        """Get relevant fields for JSON output based on query analysis"""
        primary_category = query_analysis.get('primary_category', 'general')
        secondary_categories = query_analysis.get('secondary_categories', [])
        
        # Always include basic identification fields
        relevant_fields = ['id', 'name']
        
        # Add fields based on primary category
        if primary_category in self.field_categories:
            relevant_fields.extend(self.field_categories[primary_category])
        
        # Add fields from secondary categories
        for category in secondary_categories:
            if category in self.field_categories:
                relevant_fields.extend(self.field_categories[category])
        
        # Remove duplicates while preserving order
        seen = set()
        unique_fields = []
        for field in relevant_fields:
            if field not in seen:
                seen.add(field)
                unique_fields.append(field)
        
        return unique_fields
    

class ResponseGenerator:
    """AI-powered response generator with intelligent summarization and formatting"""
    
    def __init__(self, vocabulary_manager=None):
        self.summary_generator = SummaryGenerator()  # Now AI-powered
        self.numeric_engine = NumericSearchEngine(vocabulary_manager)
        self.vocabulary_manager = vocabulary_manager
        self.data_optimizer = AirbnbDataOptimizer()
        self.ai_summarizer = AIJSONSummarizer()  # Direct AI summarization access
    
    def generate_response(self, search_results: List[Tuple[Dict[str, Any], float]], 
                        query: str, constraints: Dict[str, Any] = None, 
                        data_analysis: Dict[str, Any] = None, 
                        contextual_info: Dict[str, Any] = None) -> str:
        """Generate comprehensive response from search results with enhanced data analysis"""
        if not search_results:
            return "No matching properties found. Please try a different search."
        
        # Use provided data analysis or generate it
        query_analysis = data_analysis if data_analysis else self.data_optimizer.categorize_query(query)
        
        # Apply numeric constraints if provided (with lenient filtering)
        if constraints:
            filtered_results = []
            for doc, score in search_results:
                if self.numeric_engine._document_matches_constraints(
                    doc.get('original_document', doc), constraints
                ):
                    filtered_results.append((doc, score))
            
            # If filtering is too strict, fall back to partial matching
            if len(filtered_results) < max(1, len(search_results) * 0.1):  # Less than 10% results
                logger.info(f"Constraint filtering too strict ({len(filtered_results)} results), using lenient mode")
                filtered_results = self._apply_lenient_constraints(search_results, constraints)
            
            search_results = filtered_results
        
        if not search_results:
            return "No properties match your specific requirements. Try broadening your search."
        
        response_parts = []
        
        # Introduction
        intro = f"Found {len(search_results)} properties matching your search:"
        response_parts.append(intro)
        response_parts.append("\n")
        
        # Process each result
        for i, (doc, score) in enumerate(search_results[:5], 1):
            property_section = self._format_property_result(doc, score, query, i, query_analysis)
            response_parts.append(property_section)
            response_parts.append("\n---\n")
        
        return "\n".join(response_parts)
    
    def _format_property_result(self, document: Dict[str, Any], score: float, 
                              query: str, index: int, query_analysis: Dict[str, Any] = None) -> str:
        """Format a single property result with full source JSON, relevant fields, and AI summary"""
        original_doc = document.get('original_document', document)
        
        # Start building the response with property header
        result_parts = [
            f"**Property {index}** (Relevance: {score:.2f})",
            "",
            "**Complete Source JSON:**",
            "```json"
        ]
        
        # Always provide the complete source JSON document
        try:
            # Clean the complete document for JSON serialization
            cleaned_doc = self._clean_document_for_json(original_doc)
            json_str = json.dumps(cleaned_doc, indent=2, ensure_ascii=True)
            result_parts.append(json_str)
        except Exception as e:
            result_parts.append(f"Error formatting property data: {e}")
        
        result_parts.append("```")
        result_parts.append("")
        
        # Add query-relevant fields section
        relevant_fields = self._extract_query_relevant_fields(original_doc, query, query_analysis)
        if relevant_fields:
            result_parts.append("**Query-Relevant Fields:**")
            for field_name, field_value in relevant_fields.items():
                result_parts.append(f"• **{field_name.replace('_', ' ').title()}:** {field_value}")
            result_parts.append("")
        
        # Generate AI-powered summary from the retrieved source JSON
        try:
            # Ensure we're working with the original source document from search results
            source_document = {'original_document': original_doc} if 'original_document' not in document else document
            
            # Use AI-powered summary generator with source JSON
            summary = self.summary_generator.generate_summary(source_document, query)
            
            # Enhanced fallback: Use direct AI summarization of source JSON for longer, better summaries  
            if len(summary.split()) < 8 or summary == "Property details available":
                ai_summary = self.ai_summarizer.summarize_json(original_doc, query)
                if ai_summary and len(ai_summary.split()) > len(summary.split()):
                    summary = ai_summary
        except Exception as e:
            logger.warning(f"AI summary generation failed: {e}")
            # Ultimate fallback to data optimizer
            if query_analysis:
                summary = self.data_optimizer.optimize_document_summary(original_doc, query_analysis)
            else:
                summary = f"Property available for ${original_doc.get('price', 'N/A')}/night"
        
        # Add AI summary using retrieved source JSON
        result_parts.append("**AI Summary (from source JSON):**")
        result_parts.append(summary)
        result_parts.append("")
        
        return "\n".join(result_parts)
    
    def _extract_query_relevant_fields(self, document: Dict[str, Any], query: str, 
                                     query_analysis: Dict[str, Any] = None) -> Dict[str, str]:
        """Extract fields from document that are most relevant to the user's query"""
        relevant_fields = {}
        query_lower = query.lower()
        
        # Define field categories and their keywords
        field_categories = {
            'pricing': {
                'keywords': ['price', 'cost', 'cheap', 'expensive', 'budget', 'affordable', '$', 'dollar', 'fee'],
                'fields': ['price', 'cleaning_fee', 'security_deposit', 'extra_people']
            },
            'location': {
                'keywords': ['location', 'area', 'neighbourhood', 'neighborhood', 'city', 'near', 'close', 'downtown'],
                'fields': ['neighbourhood_cleansed', 'city', 'country', 'zipcode', 'street']
            },
            'specifications': {
                'keywords': ['bedroom', 'bathroom', 'bed', 'accommodate', 'guest', 'people', 'size'],
                'fields': ['accommodates', 'bedrooms', 'bathrooms', 'beds', 'property_type', 'room_type']
            },
            'amenities': {
                'keywords': ['wifi', 'kitchen', 'parking', 'pool', 'gym', 'amenity', 'feature', 'facility'],
                'fields': ['amenities']
            },
            'ratings': {
                'keywords': ['rating', 'review', 'score', 'quality', 'feedback', 'star'],
                'fields': ['review_scores_rating', 'number_of_reviews', 'reviews_per_month']
            },
            'host': {
                'keywords': ['host', 'owner', 'superhost'],
                'fields': ['host_name', 'host_is_superhost', 'host_response_rate', 'host_response_time']
            },
            'description': {
                'keywords': ['describe', 'about', 'detail', 'space', 'summary'],
                'fields': ['summary', 'description', 'space', 'neighborhood_overview']
            }
        }
        
        # Check which categories are relevant to the query
        relevant_categories = []
        for category, info in field_categories.items():
            if any(keyword in query_lower for keyword in info['keywords']):
                relevant_categories.append(category)
        
        # If no specific categories found, use query analysis if available
        if not relevant_categories and query_analysis:
            if 'category' in query_analysis:
                query_category = query_analysis['category'].lower()
                if query_category in field_categories:
                    relevant_categories.append(query_category)
        
        # If still no categories, include basic info
        if not relevant_categories:
            relevant_categories = ['specifications', 'pricing', 'location']
        
        # Extract relevant fields
        for category in relevant_categories:
            if category in field_categories:
                for field in field_categories[category]['fields']:
                    if field in document and document[field] is not None:
                        value = document[field]
                        # Format the value for display
                        if isinstance(value, list):
                            if len(value) > 5:  # Truncate long lists
                                formatted_value = ', '.join(str(v) for v in value[:5]) + f' (+{len(value)-5} more)'
                            else:
                                formatted_value = ', '.join(str(v) for v in value)
                        elif isinstance(value, (int, float)):
                            if field == 'price':
                                formatted_value = f'${value}'
                            elif 'rate' in field or 'rating' in field:
                                formatted_value = f'{value}' + ('/100' if field == 'review_scores_rating' else '')
                            else:
                                formatted_value = str(value)
                        else:
                            # Truncate long text fields
                            str_value = str(value)
                            if len(str_value) > 150:
                                formatted_value = str_value[:150] + '...'
                            else:
                                formatted_value = str_value
                
                        relevant_fields[field] = formatted_value
        
        # Always include name and property_type if available
        for essential_field in ['name', 'property_type']:
            if essential_field in document and document[essential_field]:
                if essential_field not in relevant_fields:
                    relevant_fields[essential_field] = str(document[essential_field])
        
        return relevant_fields
    
    def _clean_document_for_json(self, doc: Dict[str, Any], relevant_fields: List[str] = None) -> Dict[str, Any]:
        """Clean document for JSON serialization without Unicode issues"""
        cleaned = {}
        
        # If relevant fields specified, filter to those fields
        fields_to_process = relevant_fields if relevant_fields else doc.keys()
        
        for key in fields_to_process:
            if key not in doc or doc[key] is None:
                continue
            
            value = doc[key]
            
            try:
                # Handle different data types with encoding safety
                if isinstance(value, (int, float, bool)):
                    cleaned[key] = value
                elif isinstance(value, str):
                    # Preserve unicode characters for better content representation
                    # Use the original string value to preserve unicode characters
                    cleaned[key] = value


                elif isinstance(value, list):
                    clean_list = []
                    for item in value:
                        if item is not None:
                            item_str = str(item)
                            # Preserve unicode characters in list items
                            # Add the original item to preserve unicode
                            clean_list.append(item_str)


                    if clean_list:
                        cleaned[key] = clean_list
                else:
                    value_str = str(value)
                    # Remove non-ASCII characters
                    clean_str = ''.join(char for char in value_str if ord(char) < 128)
                    if clean_str.strip():
                        cleaned[key] = clean_str
            except Exception as e:
                # Skip problematic fields rather than crashing
                continue
        
        return cleaned

    def _apply_lenient_constraints(self, search_results: List[Tuple[Dict[str, Any], float]], 
                                 constraints: Dict[str, Any]) -> List[Tuple[Dict[str, Any], float]]:
        """Apply constraints more leniently, prioritizing partial matches"""
        scored_results = []
        
        for doc, score in search_results:
            original_doc = doc.get('original_document', doc)
            constraint_match_score = 0
            total_constraints = 0
            
            for field, field_constraints in constraints.items():
                total_constraints += 1
                
                # Check if field exists and is valid
                if field in original_doc and original_doc[field] is not None:
                    try:
                        value = float(str(original_doc[field]).replace('$', '').replace(',', ''))
                        
                        # Check each constraint type
                        field_matches = True
                        
                        # Handle different constraint structures
                        if isinstance(field_constraints, dict):
                            # Standard constraint dict format
                            for constraint_type, constraint_value in field_constraints.items():
                                if constraint_type == 'exact' and abs(value - constraint_value) <= constraint_value * 0.1:  # 10% tolerance
                                    constraint_match_score += 1
                                elif constraint_type == 'min' and value >= constraint_value * 0.9:  # 90% of min value
                                    constraint_match_score += 1
                                elif constraint_type == 'max' and value <= constraint_value * 1.1:  # 110% of max value
                                    constraint_match_score += 1
                                else:
                                    field_matches = False
                        else:
                            # Simple numeric constraint (treat as max)
                            try:
                                max_value = float(field_constraints)
                                if value <= max_value * 1.1:  # 110% tolerance
                                    constraint_match_score += 1
                                else:
                                    field_matches = False
                            except (ValueError, TypeError):
                                field_matches = False
                        
                        if field_matches:
                            constraint_match_score += 0.5  # Bonus for having the field
                    
                    except (ValueError, TypeError):
                        # Field exists but not numeric, give partial credit
                        constraint_match_score += 0.25
                else:
                    # Field missing, give minimal credit if no exact constraint
                    if isinstance(field_constraints, dict) and 'exact' not in field_constraints:
                        constraint_match_score += 0.1
                    elif not isinstance(field_constraints, dict):
                        constraint_match_score += 0.1
            
            # Include results that match at least some constraints
            if total_constraints == 0 or constraint_match_score > 0:
                # Adjust score based on constraint matching
                adjusted_score = score * (1 + constraint_match_score / total_constraints if total_constraints > 0 else 1)
                scored_results.append((doc, adjusted_score))
        
        # Sort by adjusted score and return top results
        scored_results.sort(key=lambda x: x[1], reverse=True)
        
        # Ensure we return at least some results
        min_results = min(len(search_results), 10)  # Return at least 10 or all available
        return scored_results[:max(min_results, len(scored_results) // 2)]
    

class JSONRAGSystem:
    """Advanced RAG system with comprehensive enhancements as standard features"""
    
    def __init__(self):
        logger.info("Initializing JSONRAGSystem", extra={
                   'module': 'core_system',
                   'class_name': 'JSONRAGSystem',
                   'method': '__init__'
               })
        self.index_manager = IndexManager()
        self.session_manager = SessionManager()
        self.query_engine = QueryUnderstandingEngine()
        self.vocabulary_manager = None
        self.semantic_search = None
        self.numeric_search = None
        self.response_generator = None
        self.data_optimizer = None
        self.system_initialized = False
        
        # Prevent multiple initialization
        self._initialized = False
        
        # Initialize the system automatically
        if not hasattr(self, '_initialized') or not self._initialized:
            self.initialize_system()
            self._initialized = True
    
    def initialize_system(self) -> bool:
        """Initialize the comprehensive RAG system with all enhanced features as standard"""
        logger.info("Initializing JSON RAG System with full enhancements", extra={
                   'module': 'core_system',
                   'class_name': 'JSONRAGSystem',
                   'method': 'initialize_system'
               })
        
        try:
            # Create indexes
            if not self.index_manager.create_complete_index():
                logger.error("Failed to create search indexes", extra={
                            'module': 'core_system',
                            'class_name': 'JSONRAGSystem',
                            'method': 'initialize_system'
                        })
                return False
            
            # Initialize vocabulary manager and load saved vocabulary
            from utils import VocabularyManager
            self.vocabulary_manager = VocabularyManager()
            
            # Load pre-built vocabulary from files
            if self.vocabulary_manager.load_vocabulary():
                logger.info("Vocabulary manager loaded from saved files", extra={
                            'module': 'core_system',
                            'class_name': 'JSONRAGSystem',
                            'method': 'initialize_system'
                        })
            else:
                logger.warning("No saved vocabulary found, vocabulary features will be limited", extra={
                            'module': 'core_system',
                            'class_name': 'JSONRAGSystem',
                            'method': 'initialize_system'
                        })
                # Initialize with empty vocabulary rather than rebuilding
                self.vocabulary_manager.vocabulary = set()
                self.vocabulary_manager.keyword_mappings = {}
                self.vocabulary_manager.numeric_patterns = {}
            
            # Initialize all enhanced search engines as standard components
            self.semantic_search = SemanticSearchEngine(self.index_manager, self.vocabulary_manager)
            self.numeric_search = NumericSearchEngine(self.vocabulary_manager)
            
            # Initialize enhanced response generator as standard
            self.response_generator = ResponseGenerator(self.vocabulary_manager)
            
            # Initialize data optimizer for intelligent query processing
            self.data_optimizer = AirbnbDataOptimizer()
            
            # Integrate AirbnbOptimizer with vocabulary manager
            if hasattr(self.query_engine, 'airbnb_optimizer'):
                # Only load documents if vocabulary wasn't loaded from files
                if not self.vocabulary_manager.vocabulary:
                    documents = self.index_manager.db_connector.get_all_documents()
                    if documents:
                        self.query_engine.airbnb_optimizer.initialize_with_mongodb_data(documents)
                        logger.info("AirbnbOptimizer initialized with MongoDB data as fallback", extra={
                                    'module': 'core_system',
                                    'class_name': 'JSONRAGSystem',
                                    'method': 'initialize_system'
                                })
                else:
                    # Use loaded vocabulary with optimizer
                    self.query_engine.airbnb_optimizer.vocabulary_manager = self.vocabulary_manager
                    logger.info("AirbnbOptimizer enhanced with loaded vocabulary", extra={
                                'module': 'core_system',
                                'class_name': 'JSONRAGSystem',
                                'method': 'initialize_system'
                            })
            
            self.system_initialized = True
            logger.info("JSON RAG System with comprehensive enhancements initialized successfully", extra={
                        'module': 'core_system',
                        'class_name': 'JSONRAGSystem',
                        'method': 'initialize_system'
                    })
            return True
            
        except Exception as e:
            logger.error("Error initializing JSON RAG system", extra={
                        'module': 'core_system',
                        'class_name': 'JSONRAGSystem',
                        'method': 'initialize_system',
                        'error': str(e)
                    }, exc_info=True)
            return False
    
    def process_query(self, query: str, session_id: str = "default") -> str:
        """Process user query and return formatted response with error handling"""
        try:
            # Input validation
            if not query or not isinstance(query, str):
                return "Please provide a valid search query."

            # Get or create session
            session = self.session_manager.get_session(session_id)
            if not session:
                logger.error("Failed to create session", extra={
                            'module': 'core_system',
                            'class_name': 'JSONRAGSystem',
                            'method': 'process_query',
                            'session_id': session_id,
                            'error': 'Session creation failed'
                        })
                return "Error: Unable to create session. Please try again."

            # Query analysis with null check
            query_analysis = self.query_engine.analyze_query(query, session.get_recent_context())
            if not query_analysis:
                logger.warning("Query analysis returned None", extra={
                            'module': 'core_system',
                            'class_name': 'JSONRAGSystem',
                            'method': 'process_query',
                            'query': query
                        })
                return "Sorry, I couldn't understand your query. Please try rephrasing it."

            # Extract numeric constraints safely
            numeric_constraints = {}
            if query_analysis.get('numeric_constraints'):
                numeric_constraints = query_analysis['numeric_constraints']

            # Perform semantic search with null checks
            # Check if semantic search is initialized
            if not self.semantic_search:
                logger.error("Semantic search not initialized", extra={
                            'module': 'core_system',
                            'class_name': 'JSONRAGSystem',
                            'method': 'process_query'
                        })
                return "Search system not fully initialized. Please try again."
                
            search_results = self.semantic_search.hybrid_search(
                query=query_analysis.get('cleaned_query', query),
                k=Config.TOP_K_RESULTS
            )
            
            if not search_results:
                logger.warning("Search returned no results", extra={
                            'module': 'core_system',
                            'class_name': 'JSONRAGSystem',
                            'method': 'process_query',
                            'query': query
                        })
                return "No matching properties found. Try adjusting your search criteria."

            # Apply numeric constraints safely
            if numeric_constraints:
                search_results = self._apply_intelligent_constraints(search_results, numeric_constraints)

            # Generate response with null checks
            if not hasattr(self, 'response_generator') or not self.response_generator:
                logger.error("Response generator not initialized", extra={
                            'module': 'core_system',
                            'class_name': 'JSONRAGSystem',
                            'method': 'process_query'
                        })
                return "System error: Response generator not available."

            response = self.response_generator.generate_response(
                search_results=search_results,
                query=query,
                constraints=numeric_constraints,
                data_analysis=query_analysis,
                contextual_info=session.get_recent_context() if session else None
            )

            # Update session with conversation turn
            if session:
                self.session_manager.add_conversation_turn(
                    session_id=session_id,
                    user_query=query,
                    system_response=response,
                    entities_extracted=query_analysis.get('entities', []),
                    numeric_constraints=numeric_constraints
                )

            return response

        except Exception as e:
            logger.error("Error processing query", extra={
                        'module': 'core_system',
                        'class_name': 'JSONRAGSystem',
                        'method': 'process_query',
                        'error': str(e),
                        'session_id': session_id,
                        'query_length': len(query) if query else 0
                    }, exc_info=True)
            return f"I apologize, but I encountered an error processing your request. Please try again or rephrase your query."
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status information"""
        status = {
            'system_initialized': self.system_initialized,
            'database_connected': False,
            'index_stats': {
                'faiss_index_size': 0,
                'processed_documents_count': 0,
                'embedding_cache_size': 0
            },
            'components_status': {
                'query_engine': bool(self.query_engine),
                'semantic_search': bool(self.semantic_search),
                'numeric_search': bool(self.numeric_search),
                'response_generator': bool(self.response_generator),
                'vocabulary_manager': bool(self.vocabulary_manager),
                'session_manager': bool(self.session_manager),
                'index_manager': bool(self.index_manager)
            },
            'vocabulary_stats': {
                'vocabulary_size': 0,
                'keyword_mappings_count': 0,
                'numeric_patterns_count': 0
            },
            'nlp_models': {
                'sentence_transformer_loaded': False,
                'spacy_model_loaded': False
            },
            'performance_metrics': {
                'avg_query_time': 0.0,
                'total_queries_processed': 0
            }
        }
        
        # Check database connection
        try:
            if self.index_manager and hasattr(self.index_manager, 'db_connector'):
                status['database_connected'] = self.index_manager.db_connector.test_connection()
        except Exception as e:
            logger.warning("Could not check database connection", extra={
                        'module': 'core_system',
                        'class_name': 'JSONRAGSystem',
                        'method': 'get_system_status',
                        'error': str(e)
                    })
        
        # Get index statistics
        try:
            if self.index_manager:
                if hasattr(self.index_manager, 'faiss_index') and self.index_manager.faiss_index:
                    status['index_stats']['faiss_index_size'] = self.index_manager.faiss_index.ntotal
                
                if hasattr(self.index_manager, 'db_connector') and self.index_manager.db_connector:
                    doc_count = self.index_manager.db_connector.get_document_count()
                    status['index_stats']['processed_documents_count'] = doc_count if doc_count else 0
                
                if hasattr(self.index_manager, 'embedding_cache'):
                    status['index_stats']['embedding_cache_size'] = len(getattr(self.index_manager, 'embedding_cache', {}))
        except Exception as e:
            logger.warning("Could not get index statistics", extra={
                        'module': 'core_system',
                        'class_name': 'JSONRAGSystem',
                        'method': 'get_system_status',
                        'error': str(e)
                    })
        
        # Get vocabulary statistics
        try:
            if self.vocabulary_manager:
                if hasattr(self.vocabulary_manager, 'vocabulary'):
                    vocab = getattr(self.vocabulary_manager, 'vocabulary', set())
                    status['vocabulary_stats']['vocabulary_size'] = len(vocab) if vocab else 0
                
                if hasattr(self.vocabulary_manager, 'keyword_mappings'):
                    mappings = getattr(self.vocabulary_manager, 'keyword_mappings', {})
                    status['vocabulary_stats']['keyword_mappings_count'] = len(mappings) if mappings else 0
                
                if hasattr(self.vocabulary_manager, 'numeric_patterns'):
                    patterns = getattr(self.vocabulary_manager, 'numeric_patterns', {})
                    status['vocabulary_stats']['numeric_patterns_count'] = len(patterns) if patterns else 0
        except Exception as e:
            logger.warning("Could not get vocabulary statistics", extra={
                        'module': 'core_system',
                        'class_name': 'JSONRAGSystem',
                        'method': 'get_system_status',
                        'error': str(e)
                    })
        
        # Check NLP models
        try:
            if self.query_engine:
                status['nlp_models']['sentence_transformer_loaded'] = bool(
                    hasattr(self.query_engine, 'sentence_model') and 
                    getattr(self.query_engine, 'sentence_model', None)
                )
                status['nlp_models']['spacy_model_loaded'] = bool(
                    hasattr(self.query_engine, 'nlp_model') and 
                    getattr(self.query_engine, 'nlp_model', None)
                )
        except Exception as e:
            logger.warning("Could not check NLP models", extra={
                        'module': 'core_system',
                        'class_name': 'JSONRAGSystem',
                        'method': 'get_system_status',
                        'error': str(e)
                    })
        
        # Get session statistics
        try:
            if self.session_manager:
                session_stats = self.session_manager.get_session_stats()
                status['session_stats'] = session_stats
                
                total_queries = session_stats.get('total_conversations', 0)
                status['performance_metrics']['total_queries_processed'] = total_queries
                
                # Simple average (could be enhanced with actual timing data)
                if total_queries > 0:
                    status['performance_metrics']['avg_query_time'] = 1.0  # Placeholder
            else:
                # Provide default session stats if session_manager is not available
                status['session_stats'] = {
                    'active_sessions': 0,
                    'total_conversations': 0,
                    'average_turns_per_session': 0.0
                }
        except Exception as e:
            logger.warning("Could not get session statistics", extra={
                        'module': 'core_system',
                        'class_name': 'JSONRAGSystem',
                        'method': 'get_system_status',
                        'error': str(e)
                    })
            # Provide default session stats on error
            status['session_stats'] = {
                'active_sessions': 0,
                'total_conversations': 0,
                'average_turns_per_session': 0.0
            }
        
        return status
    
    def _apply_intelligent_constraints(self, search_results: List[Tuple[Dict[str, Any], float]], 
                                     constraints: Dict[str, Any]) -> List[Tuple[Dict[str, Any], float]]:
        """Apply numeric constraints to search results with intelligent filtering"""
        if not constraints or not self.numeric_search:
            return search_results
        
        try:
            filtered_results = []
            for doc, score in search_results:
                original_doc = doc.get('original_document', doc)
                if self.numeric_search._document_matches_constraints(original_doc, constraints):
                    filtered_results.append((doc, score))
            
            # If filtering is too restrictive, use lenient mode
            if len(filtered_results) < max(1, len(search_results) * 0.1):
                logger.info("Applying lenient constraint filtering", extra={
                            'module': 'core_system',
                            'class_name': 'JSONRAGSystem',
                            'method': '_apply_intelligent_constraints',
                            'original_results': len(search_results),
                            'filtered_results': len(filtered_results)
                        })
                if hasattr(self.response_generator, '_apply_lenient_constraints'):
                    return self.response_generator._apply_lenient_constraints(search_results, constraints)
            
            return filtered_results
            
        except Exception as e:
            logger.warning("Error applying constraints", extra={
                        'module': 'core_system',
                        'class_name': 'JSONRAGSystem',
                        'method': '_apply_intelligent_constraints',
                        'error': str(e)
                    })
            return search_results