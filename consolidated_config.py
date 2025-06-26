#!/usr/bin/env python3
"""
Consolidated Configuration for Django RAG System
Combines all configuration settings for the Airbnb property search system
"""

# =============================================================================
# PROPERTY TYPE SYNONYMS AND MAPPINGS
# =============================================================================

PROPERTY_TYPE_SYNONYMS = {
    'apartment': [
        'apartment', 'flat', 'unit', 'condo', 'condominium', 'studio',
        'pied-a-terre', 'loft', 'duplex'
    ],
    'house': [
        'house', 'home', 'villa', 'bungalow', 'cottage', 'cabin',
        'chalet', 'mansion', 'residence', 'detached'
    ],
    'room': [
        'room', 'bedroom', 'private room', 'shared room', 'guest room',
        'master bedroom', 'dormitory', 'dorm','bhk', 'suite', 'apartment room'
    ],
    'studio': [
        'studio', 'bachelor', 'efficiency', 'micro apartment', 'bedsit',
        'studio flat', 'open-plan'
    ],
    'townhouse': [
        'townhouse', 'townhome', 'row house', 'brownstone', 'mews',
        'terraced house', 'connected home'
    ],
    'guesthouse': [
        'guesthouse', 'guest house', 'guest suite', 'casita', 'annex',
        'granny flat', 'pool house', 'garden house'
    ],
    'unique': [
        'treehouse', 'boat', 'yacht', 'houseboat', 'rv', 'camper',
        'tiny house', 'barn', 'farm stay', 'lighthouse', 'castle'
    ]
}

# =============================================================================
# AMENITY AND LOCATION SYNONYMS
# =============================================================================

AMENITY_SYNONYMS = {
    'wifi': [
        'wifi', 'wi-fi', 'wireless', 'internet', 'broadband', 'connectivity',
        'network', 'web access', 'online'
    ],
    'parking': [
        'parking', 'garage', 'carport', 'driveway', 'parking spot',
        'parking space', 'car park', 'covered parking', 'street parking'
    ],
    'pool': [
        'pool', 'swimming pool', 'swimming', 'swim', 'outdoor pool',
        'indoor pool', 'heated pool', 'lap pool'
    ],
    'kitchen': [
        'kitchen', 'kitchenette', 'cooking', 'full kitchen', 'private kitchen',
        'equipped kitchen', 'modern kitchen', 'chef kitchen'
    ],
    'ac': [
        'ac', 'air conditioning', 'air con', 'cooling', 'climate control',
        'central air', 'air conditioner', 'hvac'
    ],
    'heating': [
        'heating', 'heat', 'central heating', 'heater', 'radiator',
        'furnace', 'heated floors', 'thermostat'
    ],
    'washer': [
        'washer', 'washing machine', 'laundry', 'clothes washer',
        'washer dryer', 'laundry facilities', 'washing'
    ],
    'tv': [
        'tv', 'television', 'smart tv', 'cable tv', 'satellite tv',
        'netflix', 'streaming', 'hdtv', 'flat screen'
    ],
    'workspace': [
        'workspace', 'desk', 'work desk', 'office', 'study area',
        'laptop friendly', 'work station', 'business center'
    ],
    'gym': [
        'gym', 'fitness', 'fitness center', 'exercise room', 'workout',
        'fitness equipment', 'exercise equipment', 'weights'
    ]
}

LOCATION_SYNONYMS = {
    'downtown': [
        'downtown', 'city center', 'central', 'cbd', 'heart of city',
        'city centre', 'urban core', 'midtown'
    ],
    'new_york': [
        'new york', 'nyc', 'new york city', 'manhattan', 'brooklyn', 
        'queens', 'bronx', 'staten island', 'ny', 'new york ny'
    ],
    'beach': [
        'beach', 'beachfront', 'oceanfront', 'seaside', 'coastal',
        'waterfront', 'shore', 'beach access'
    ],
    'suburb': [
        'suburb', 'suburban', 'residential', 'residential area',
        'quiet neighborhood', 'suburbs', 'outskirts'
    ],
    'airport': [
        'airport', 'near airport', 'airport area', 'airport vicinity',
        'airport shuttle', 'airport transfer'
    ],
    'shopping': [
        'shopping', 'mall', 'shopping center', 'retail', 'shops',
        'shopping district', 'commercial', 'market'
    ],
    'historic': [
        'historic', 'old town', 'historic district', 'heritage',
        'historic center', 'cultural district', 'historic area'
    ],
    'business': [
        'business district', 'financial district', 'commercial district',
        'business center', 'corporate', 'office district'
    ],
    'entertainment': [
        'entertainment district', 'nightlife', 'theater district',
        'restaurant row', 'dining district', 'entertainment area'
    ]
}

# =============================================================================
# FIELD CATEGORIES AND WEIGHTS
# =============================================================================

FIELD_CATEGORIES = {
    'pricing': {
        'fields': ['price', 'nightly_rate', 'weekly_rate', 'monthly_rate', 'cleaning_fee', 'service_fee', 'total_price'],
        'weight': 0.9,
        'description': 'Pricing information'
    },
    'location': {
        'fields': ['address', 'city', 'state', 'country', 'neighborhood', 'zipcode', 'latitude', 'longitude'],
        'weight': 1.0,
        'description': 'Location details'
    },
    'amenities': {
        'fields': ['amenities', 'house_amenities', 'safety_items', 'accessibility'],
        'weight': 0.8,
        'description': 'Property amenities and features'
    },
    'property': {
        'fields': ['property_type', 'room_type', 'bedrooms', 'bathrooms', 'beds', 'accommodates', 'square_feet'],
        'weight': 1.0,
        'description': 'Property specifications'
    },
    'host_info': {
        'fields': ['host_name', 'host_since', 'host_response_time', 'host_response_rate', 'host_is_superhost'],
        'weight': 0.6,
        'description': 'Host information'
    },
    'reviews': {
        'fields': ['overall_rating', 'accuracy_rating', 'cleanliness_rating', 'checkin_rating', 'communication_rating', 'location_rating', 'value_rating', 'review_count'],
        'weight': 0.8,
        'description': 'Reviews and ratings'
    },
    'general': {
        'fields': ['name', 'description', 'summary', 'space', 'access', 'interaction', 'neighborhood_overview', 'notes', 'transit'],
        'weight': 0.7,
        'description': 'General property information'
    }
}

# =============================================================================
# NUMERIC CONFIGURATION
# =============================================================================

class NumericConfig:
    """Centralized configuration for numeric query processing"""

    # Numeric keyword mappings
    NUMERIC_KEYWORDS = {
        # Bedroom mappings
        'bedrooms': {
            'keywords': [
                'bedroom', 'bedrooms', 'bed', 'beds', 'br', 'bdr', 'bdrs',
                'sleeping room', 'sleeping rooms', 'sleep', 'sleeps'
            ],
            'patterns': [
                r'\b(\d+)\s*(?:bed|bedroom|bedrooms|br|bdr)s?\b',
                r'\b(?:bed|bedroom|bedrooms|br|bdr)s?\s*(\d+)\b',
                r'\b(\d+)\s*(?:-|\s)?(?:bed|bedroom|br)\b',
                r'\bstudio\b'  # Special case for studio = 0 bedrooms
            ],
            'field_names': ['bedrooms', 'beds', 'bedroom_count', 'bed_count'],
            'studio_value': 0,
            'default_range': [1, 10]
        },
        
        # Bathroom mappings
        'bathrooms': {
            'keywords': [
                'bathroom', 'bathrooms', 'bath', 'baths', 'ba', 'full bath',
                'half bath', 'powder room', 'washroom', 'restroom'
            ],
            'patterns': [
                r'\b(\d+(?:\.\d+)?)\s*(?:bath|bathroom|bathrooms|ba)s?\b',
                r'\b(?:bath|bathroom|bathrooms|ba)s?\s*(\d+(?:\.\d+)?)\b',
                r'\b(\d+(?:\.\d+)?)\s*(?:-|\s)?(?:bath|bathroom|ba)\b'
            ],
            'field_names': ['bathrooms', 'baths', 'bathroom_count', 'bath_count'],
            'default_range': [1, 8]
        },
        
        # Guest/Accommodation mappings
        'guests': {
            'keywords': [
                'guest', 'guests', 'people', 'person', 'occupant', 'occupants',
                'accommodate', 'accommodates', 'sleeps', 'capacity', 'max guest',
                'maximum guest', 'guest limit', 'occupancy'
            ],
            'patterns': [
                r'\b(?:accommodate|sleeps|guest|guests|people|person)s?\s*(\d+)\b',
                r'\b(\d+)\s*(?:guest|guests|people|person|occupant)s?\b',
                r'\bfor\s*(\d+)\s*(?:guest|guests|people|person)s?\b',
                r'\bup\s*to\s*(\d+)\s*(?:guest|guests|people)s?\b'
            ],
            'field_names': ['guests', 'accommodates', 'guest_capacity', 'max_guests', 'occupancy'],
            'default_range': [1, 16]
        },
        
        # Price mappings
        'price': {
            'keywords': [
                'price', 'cost', 'rate', 'fee', 'charge', 'amount', 'budget',
                'expensive', 'cheap', 'affordable', 'under', 'below', 'above',
                'maximum', 'minimum', 'max', 'min', 'dollar', 'usd'
            ],
            'patterns': [
                r'\$\s*(\d+(?:,\d{3})*(?:\.\d{2})?)\b',
                r'\b(\d+(?:,\d{3})*(?:\.\d{2})?)\s*(?:dollar|usd|$)s?\b',
                r'\bunder\s*\$?\s*(\d+(?:,\d{3})*)\b',
                r'\bbelow\s*\$?\s*(\d+(?:,\d{3})*)\b',
                r'\babove\s*\$?\s*(\d+(?:,\d{3})*)\b',
                r'\bover\s*\$?\s*(\d+(?:,\d{3})*)\b',
                r'\bmax\s*\$?\s*(\d+(?:,\d{3})*)\b',
                r'\bmaximum\s*\$?\s*(\d+(?:,\d{3})*)\b',
                r'\bmin\s*\$?\s*(\d+(?:,\d{3})*)\b',
                r'\bminimum\s*\$?\s*(\d+(?:,\d{3})*)\b'
            ],
            'field_names': ['price', 'cost', 'rate', 'nightly_rate', 'daily_rate'],
            'currency_symbols': ['$', 'USD', 'usd'],
            'default_range': [10, 10000]
        }
    }

    # Range operators for numeric constraints
    RANGE_OPERATORS = {
        'exact': {
            'patterns': [
                r'\bexactly\s*(\d+)\b',
                r'^(\d+)$',
                r'\b(\d+)\s*(?:bed|bedroom|bath|bathroom|guest)s?\s*only\b'
            ],
            'operator': '='
        },
        
        'minimum': {
            'patterns': [
                r'\bat\s*least\s*(\d+)\b',
                r'\bminimum\s*(?:of\s*)?(\d+)\b',
                r'\bmin\s*(\d+)\b',
                r'\b(\d+)\s*(?:or\s*)?(?:more|plus|above)\b',
                r'\b(?:more\s*than|above|over)\s*(\d+)\b'
            ],
            'operator': '>='
        },
        
        'maximum': {
            'patterns': [
                r'\bat\s*most\s*(\d+)\b',
                r'\bmaximum\s*(?:of\s*)?(\d+)\b',
                r'\bmax\s*(\d+)\b',
                r'\b(\d+)\s*(?:or\s*)?(?:less|fewer|below|under)\b',
                r'\b(?:less\s*than|below|under)\s*(\d+)\b',
                r'\bup\s*to\s*(\d+)\b'
            ],
            'operator': '<='
        },
        
        'range': {
            'patterns': [
                r'\bbetween\s*(\d+)\s*(?:and|to|-)\s*(\d+)\b',
                r'\b(\d+)\s*(?:to|-)\s*(\d+)\b',
                r'\b(\d+)\s*through\s*(\d+)\b',
                r'\bfrom\s*(\d+)\s*to\s*(\d+)\b'
            ],
            'operator': 'between'
        }
    }

    # Special number word mappings
    NUMBER_WORDS = {
        'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
        'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
        'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14, 'fifteen': 15,
        'sixteen': 16, 'seventeen': 17, 'eighteen': 18, 'nineteen': 19, 'twenty': 20,
        'studio': 0,  # Studio apartments have 0 bedrooms
        'single': 1,
        'double': 2,
        'triple': 3,
        'quad': 4,
        'multiple': 2  # Default for 'multiple'
    }

    # Property type numeric expectations
    PROPERTY_NUMERIC_DEFAULTS = {
        'studio': {'bedrooms': 0, 'bathrooms': 1, 'guests': 2},
        'apartment': {'bedrooms': [1, 3], 'bathrooms': [1, 2], 'guests': [1, 6]},
        'house': {'bedrooms': [2, 6], 'bathrooms': [1, 4], 'guests': [3, 12]},
        'villa': {'bedrooms': [3, 8], 'bathrooms': [2, 6], 'guests': [6, 16]},
        'condo': {'bedrooms': [1, 3], 'bathrooms': [1, 2], 'guests': [2, 6]},
        'loft': {'bedrooms': [1, 2], 'bathrooms': [1, 2], 'guests': [2, 4]},
        'cabin': {'bedrooms': [1, 4], 'bathrooms': [1, 3], 'guests': [2, 8]},
        'townhouse': {'bedrooms': [2, 4], 'bathrooms': [1, 3], 'guests': [4, 8]}
    }

    # Context-based constraint patterns
    CONTEXT_PATTERNS = {
        'family': {
            'keywords': ['family', 'kids', 'children', 'child-friendly'],
            'implications': {
                'bedrooms': [2, 4],
                'guests': [4, 8],
                'bathrooms': [2, 3]
            }
        },
        'couple': {
            'keywords': ['couple', 'romantic', 'honeymoon', 'two people'],
            'implications': {
                'bedrooms': 1,
                'guests': 2,
                'bathrooms': 1
            }
        },
        'business': {
            'keywords': ['business', 'work', 'corporate', 'conference'],
            'implications': {
                'bedrooms': 1,
                'guests': [1, 2]
            }
        },
        'group': {
            'keywords': ['group', 'friends', 'party', 'large group'],
            'implications': {
                'bedrooms': [3, 6],
                'guests': [6, 12],
                'bathrooms': [2, 4]
            }
        }
    }

    # Numeric intent patterns
    NUMERIC_INTENT_PATTERNS = {
        'specific_search': {
            'patterns': [
                r'\bexactly\s+\d+\b',
                r'\b\d+\s+bed(?:room)?s?\s+exactly\b',
                r'\bprecisely\s+\d+\b'
            ]
        },
        'capacity_search': {
            'patterns': [
                r'\baccommodate(?:s)?\s+\d+\b',
                r'\bsleep(?:s)?\s+\d+\b',
                r'\bfor\s+\d+\s+(?:people|guests?)\b'
            ]
        },
        'budget_search': {
            'patterns': [
                r'\bunder\s+\$?\d+\b',
                r'\bbelow\s+\$?\d+\b',
                r'\bbudget\s+\$?\d+\b',
                r'\bcheap(?:er)?\b'
            ]
        },
        'luxury_search': {
            'patterns': [
                r'\bover\s+\$?\d+\b',
                r'\babove\s+\$?\d+\b',
                r'\bluxury\b',
                r'\bpremium\b',
                r'\bhigh.?end\b'
            ]
        }
    }

    # Legacy attribute for backward compatibility
    NUMERIC_INTENT_PATTERNS = {
        'specific_search': {
            'patterns': [
                r'\bexactly\s+\d+\b',
                r'\b\d+\s+bed(?:room)?s?\s+exactly\b',
                r'\bprecisely\s+\d+\b'
            ]
        },
        'capacity_search': {
            'patterns': [
                r'\baccommodate(?:s)?\s+\d+\b',
                r'\bsleep(?:s)?\s+\d+\b',
                r'\bfor\s+\d+\s+(?:people|guests?)\b'
            ]
        },
        'budget_search': {
            'patterns': [
                r'\bunder\s+\$?\d+\b',
                r'\bbelow\s+\$?\d+\b',
                r'\bbudget\s+\$?\d+\b',
                r'\bcheap(?:er)?\b'
            ]
        },
        'luxury_search': {
            'patterns': [
                r'\bover\s+\$?\d+\b',
                r'\babove\s+\$?\d+\b',
                r'\bluxury\b',
                r'\bpremium\b',
                r'\bhigh.?end\b'
            ]
        }
    }

# =============================================================================
# MAIN CONFIGURATION
# =============================================================================

RAG_SYSTEM_CONFIG = {
    # Search field weights
    'field_weights': {
        'name': 1.0,
        'description': 0.7,
        'amenities': 0.8,
        'neighborhood': 0.9,
        'property_type': 0.8,
        'room_type': 0.7,
        'summary': 0.6
    },
    
    # Property type mappings
    'property_mappings': {
        'apartment': ['apartment', 'flat', 'condo', 'condominium', 'unit'],
        'house': ['house', 'home', 'villa', 'cottage', 'cabin', 'bungalow'],
        'room': ['room', 'bedroom', 'private room', 'shared room'],
        'studio': ['studio', 'efficiency', 'bachelor'],
        'loft': ['loft', 'penthouse', 'attic'],
        'townhouse': ['townhouse', 'townhome', 'row house'],
        'other': ['other', 'unique', 'unusual']
    },
    
    # Amenity categories
    'amenity_categories': {
        'kitchen': ['kitchen', 'cooking', 'microwave', 'refrigerator', 'stove', 'oven', 'dishwasher', 'coffee maker'],
        'internet': ['wifi', 'internet', 'wireless', 'broadband'],
        'parking': ['parking', 'garage', 'driveway', 'street parking'],
        'laundry': ['washer', 'dryer', 'laundry', 'washing machine'],
        'entertainment': ['tv', 'television', 'cable', 'netflix', 'streaming'],
        'comfort': ['air conditioning', 'heating', 'fireplace', 'fan'],
        'outdoor': ['pool', 'hot tub', 'balcony', 'patio', 'garden', 'yard'],
        'safety': ['smoke detector', 'carbon monoxide detector', 'first aid kit', 'fire extinguisher']
    },
    
    # Synonym mappings for query expansion (ASCII only)
    'synonyms': {
        'cheap': ['budget', 'affordable', 'economical', 'inexpensive', 'low cost', 'value'],
        'expensive': ['luxury', 'premium', 'high-end', 'upscale', 'costly', 'pricey'],
        'close': ['near', 'nearby', 'adjacent', 'walking distance', 'proximity', 'convenient'],
        'big': ['large', 'spacious', 'huge', 'roomy', 'expansive', 'vast'],
        'small': ['cozy', 'compact', 'tiny', 'intimate', 'snug', 'modest'],
        'clean': ['spotless', 'pristine', 'immaculate', 'tidy', 'neat', 'sanitized'],
        'nice': ['great', 'wonderful', 'excellent', 'amazing', 'fantastic', 'superb'],
        'quiet': ['peaceful', 'tranquil', 'calm', 'serene', 'silent', 'still'],
        'central': ['downtown', 'city center', 'urban', 'metropolitan', 'core', 'heart'],
        'modern': ['contemporary', 'updated', 'renovated', 'new', 'current', 'fresh'],
        'wifi': ['internet', 'wireless', 'broadband', 'connection', 'online'],
        'parking': ['garage', 'spot', 'space', 'lot', 'driveway'],
        'kitchen': ['cooking', 'culinary', 'food prep', 'kitchenette'],
        'pool': ['swimming', 'swim', 'water', 'aquatic'],
        'beach': ['ocean', 'sea', 'waterfront', 'shore', 'coastal'],
        'mountain': ['hill', 'peak', 'elevation', 'scenic', 'nature']
    },
    
    # Price range mappings
    'price_ranges': {
        'budget': {'min': 0, 'max': 75},
        'moderate': {'min': 75, 'max': 150},
        'expensive': {'min': 150, 'max': 300},
        'luxury': {'min': 300, 'max': 1000}
    },
    
    # Accommodation capacity mappings
    'accommodation_ranges': {
        'solo': {'min': 1, 'max': 1},
        'couple': {'min': 2, 'max': 2},
        'small_group': {'min': 3, 'max': 4},
        'large_group': {'min': 5, 'max': 8},
        'party': {'min': 9, 'max': 20}
    }
}

# =============================================================================
# EXPORTS
# =============================================================================

# Export all configurations for easy importing
__all__ = [
    'PROPERTY_TYPE_SYNONYMS',
    'AMENITY_SYNONYMS', 
    'LOCATION_SYNONYMS',
    'FIELD_CATEGORIES',
    'NumericConfig',
    'RAG_SYSTEM_CONFIG'
]
