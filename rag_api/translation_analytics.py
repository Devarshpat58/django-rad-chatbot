"""
Translation Analytics and Monitoring Service
Tracks translation performance, fallback usage, and system health
"""

import logging
import json
import time
from datetime import datetime, timedelta
from collections import defaultdict, deque
from typing import Dict, List, Any, Optional
from django.core.cache import cache
from django.utils import timezone

logger = logging.getLogger(__name__)

class TranslationAnalytics:
    """Tracks and analyzes translation system performance"""
    
    def __init__(self):
        self.session_stats = defaultdict(lambda: {
            'total_requests': 0,
            'successful_translations': 0,
            'fallback_usage': 0,
            'avg_response_time': 0.0,
            'language_distribution': defaultdict(int),
            'fallback_reasons': defaultdict(int),
            'error_types': defaultdict(int)
        })
        
        # Recent performance tracking (last 100 requests)
        self.recent_performance = deque(maxlen=100)
        
        # Cache keys for persistent storage
        self.CACHE_KEY_DAILY_STATS = "translation_daily_stats"
        self.CACHE_KEY_PERFORMANCE_HISTORY = "translation_performance_history"
        
    def record_translation_request(self, 
                                 language_detected: str,
                                 target_language: str,
                                 success: bool,
                                 fallback_used: bool,
                                 fallback_reason: Optional[str] = None,
                                 response_time: float = 0.0,
                                 error_type: Optional[str] = None):
        """Record a translation request for analytics"""
        
        timestamp = timezone.now()
        session_key = timestamp.strftime('%Y-%m-%d')
        
        # Update session stats
        stats = self.session_stats[session_key]
        stats['total_requests'] += 1
        
        if success:
            stats['successful_translations'] += 1
        
        if fallback_used:
            stats['fallback_usage'] += 1
            if fallback_reason:
                stats['fallback_reasons'][fallback_reason] += 1
        
        if error_type:
            stats['error_types'][error_type] += 1
            
        stats['language_distribution'][language_detected] += 1
        
        # Update average response time
        current_avg = stats['avg_response_time']
        total_requests = stats['total_requests']
        stats['avg_response_time'] = ((current_avg * (total_requests - 1)) + response_time) / total_requests
        
        # Track recent performance
        performance_record = {
            'timestamp': timestamp.isoformat(),
            'language': language_detected,
            'target': target_language,
            'success': success,
            'fallback_used': fallback_used,
            'fallback_reason': fallback_reason,
            'response_time': response_time,
            'error_type': error_type
        }
        
        self.recent_performance.append(performance_record)
        
        # Persist to cache for dashboard access
        self._persist_stats()
        
        logger.info(f"Translation analytics recorded: {language_detected}->{target_language}, "
                   f"success={success}, fallback={fallback_used}, time={response_time:.3f}s")
    
    def _persist_stats(self):
        """Persist analytics data to cache"""
        try:
            # Convert defaultdict to regular dict for JSON serialization
            serializable_stats = {}
            for date, stats in self.session_stats.items():
                serializable_stats[date] = {
                    'total_requests': stats['total_requests'],
                    'successful_translations': stats['successful_translations'],
                    'fallback_usage': stats['fallback_usage'],
                    'avg_response_time': stats['avg_response_time'],
                    'language_distribution': dict(stats['language_distribution']),
                    'fallback_reasons': dict(stats['fallback_reasons']),
                    'error_types': dict(stats['error_types'])
                }
            
            # Store daily stats (expires in 7 days)
            cache.set(self.CACHE_KEY_DAILY_STATS, serializable_stats, 7 * 24 * 3600)
            
            # Store recent performance (expires in 24 hours)
            cache.set(self.CACHE_KEY_PERFORMANCE_HISTORY, list(self.recent_performance), 24 * 3600)
            
        except Exception as e:
            logger.error(f"Failed to persist translation analytics: {e}")
    
    def get_daily_stats(self, date: Optional[str] = None) -> Dict[str, Any]:
        """Get translation statistics for a specific date"""
        if date is None:
            date = timezone.now().strftime('%Y-%m-%d')
        
        # Try to get from memory first
        if date in self.session_stats:
            stats = self.session_stats[date]
            return {
                'date': date,
                'total_requests': stats['total_requests'],
                'successful_translations': stats['successful_translations'],
                'fallback_usage': stats['fallback_usage'],
                'success_rate': (stats['successful_translations'] / max(stats['total_requests'], 1)) * 100,
                'fallback_rate': (stats['fallback_usage'] / max(stats['total_requests'], 1)) * 100,
                'avg_response_time': stats['avg_response_time'],
                'language_distribution': dict(stats['language_distribution']),
                'fallback_reasons': dict(stats['fallback_reasons']),
                'error_types': dict(stats['error_types'])
            }
        
        # Fallback to cache
        cached_stats = cache.get(self.CACHE_KEY_DAILY_STATS, {})
        if date in cached_stats:
            stats = cached_stats[date]
            return {
                'date': date,
                'success_rate': (stats['successful_translations'] / max(stats['total_requests'], 1)) * 100,
                'fallback_rate': (stats['fallback_usage'] / max(stats['total_requests'], 1)) * 100,
                **stats
            }
        
        # No data available
        return {
            'date': date,
            'total_requests': 0,
            'successful_translations': 0,
            'fallback_usage': 0,
            'success_rate': 0.0,
            'fallback_rate': 0.0,
            'avg_response_time': 0.0,
            'language_distribution': {},
            'fallback_reasons': {},
            'error_types': {}
        }
    
    def get_recent_performance(self) -> List[Dict[str, Any]]:
        """Get recent translation performance data"""
        # Try memory first
        if self.recent_performance:
            return list(self.recent_performance)
        
        # Fallback to cache
        cached_performance = cache.get(self.CACHE_KEY_PERFORMANCE_HISTORY, [])
        return cached_performance
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall translation system health metrics"""
        recent_data = self.get_recent_performance()
        
        if not recent_data:
            return {
                'status': 'no_data',
                'health_score': 0,
                'recommendations': ['No recent translation data available']
            }
        
        # Calculate health metrics from recent data
        total_recent = len(recent_data)
        successful = sum(1 for r in recent_data if r['success'])
        fallback_used = sum(1 for r in recent_data if r['fallback_used'])
        avg_response_time = sum(r['response_time'] for r in recent_data) / total_recent
        
        success_rate = (successful / total_recent) * 100
        fallback_rate = (fallback_used / total_recent) * 100
        
        # Calculate health score (0-100)
        health_score = 0
        health_score += min(success_rate, 100) * 0.5  # 50% weight for success rate
        health_score += max(0, 100 - fallback_rate) * 0.3  # 30% weight for low fallback rate
        health_score += max(0, 100 - (avg_response_time * 20)) * 0.2  # 20% weight for response time
        
        # Determine status
        if health_score >= 90:
            status = 'excellent'
        elif health_score >= 75:
            status = 'good'
        elif health_score >= 60:
            status = 'fair'
        else:
            status = 'poor'
        
        # Generate recommendations
        recommendations = []
        if success_rate < 95:
            recommendations.append(f"Success rate is {success_rate:.1f}% - consider model optimization")
        if fallback_rate > 20:
            recommendations.append(f"High fallback usage ({fallback_rate:.1f}%) - check model availability")
        if avg_response_time > 3.0:
            recommendations.append(f"Slow response time ({avg_response_time:.2f}s) - consider caching")
        
        if not recommendations:
            recommendations.append("Translation system is performing well")
        
        return {
            'status': status,
            'health_score': round(health_score, 1),
            'success_rate': round(success_rate, 1),
            'fallback_rate': round(fallback_rate, 1),
            'avg_response_time': round(avg_response_time, 3),
            'total_recent_requests': total_recent,
            'recommendations': recommendations
        }
    
    def get_language_performance(self) -> Dict[str, Dict[str, Any]]:
        """Get performance metrics broken down by language"""
        recent_data = self.get_recent_performance()
        
        language_stats = defaultdict(lambda: {
            'total_requests': 0,
            'successful_translations': 0,
            'fallback_usage': 0,
            'total_response_time': 0.0
        })
        
        for record in recent_data:
            lang = record['language']
            stats = language_stats[lang]
            stats['total_requests'] += 1
            
            if record['success']:
                stats['successful_translations'] += 1
            
            if record['fallback_used']:
                stats['fallback_usage'] += 1
            
            stats['total_response_time'] += record['response_time']
        
        # Calculate rates and averages
        result = {}
        for lang, stats in language_stats.items():
            total = stats['total_requests']
            result[lang] = {
                'total_requests': total,
                'success_rate': (stats['successful_translations'] / total) * 100 if total > 0 else 0,
                'fallback_rate': (stats['fallback_usage'] / total) * 100 if total > 0 else 0,
                'avg_response_time': stats['total_response_time'] / total if total > 0 else 0
            }
        
        return result

# Global analytics instance
translation_analytics = TranslationAnalytics()