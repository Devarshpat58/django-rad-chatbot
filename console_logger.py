import logging
import json
from datetime import datetime

from logging_config import StructuredLogger

class ConsoleLogger:
    """
    ConsoleLogger provides detailed, structured logging for Django web responses,
    including query, response, session, and metadata for debugging and analytics.
    """

    # Use a dedicated logger for console output
    logger = StructuredLogger("console_logger", {"component": "web_interface", "source_module": "console_logger"})

    @staticmethod
    def log_django_web_response(query_text, result, session_key, user=None, endpoint=None, extra_metadata=None):
        """
        Log a detailed summary of a Django web response for debugging and analytics.

        Args:
            query_text (str): The user's query.
            result (dict): The result dictionary returned by the RAG service.
            session_key (str): The session key for the request.
            user (User, optional): The Django user object, if available.
            endpoint (str, optional): The endpoint being logged (e.g., '/ajax/search/').
            extra_metadata (dict, optional): Any extra metadata to include.
        """
        # Prepare log context
        context = {
            "session_id": session_key,
            "user_id": getattr(user, "id", None) if user else "anonymous",
            "endpoint": endpoint or "unknown",
            "timestamp": datetime.utcnow().isoformat(),
        }
        if extra_metadata:
            context.update(extra_metadata)

        # Extract response summary
        response_text = result.get("response", "")
        response_preview = response_text[:300] + ("..." if len(response_text) > 300 else "")
        metadata = result.get("metadata", {})
        num_results = metadata.get("num_results") or len(metadata.get("results", [])) or 0
        exec_time = metadata.get("execution_time", None)

        # Build log message
        message = (
            f"Web response | Query: '{query_text}' | "
            f"Results: {num_results} | "
            f"ExecTime: {exec_time}s | "
            f"Session: {session_key} | "
            f"User: {context['user_id']}"
        )

        # Add preview of response for debugging
        context["response_preview"] = response_preview
        context["num_results"] = num_results
        context["execution_time"] = exec_time

        # Optionally log full metadata as JSON (truncated for safety)
        try:
            context["metadata_json"] = json.dumps(metadata)[:1000]
        except Exception:
            context["metadata_json"] = str(metadata)[:1000]

        # Log at INFO level with all context
        ConsoleLogger.logger.info(message, **context)

    @staticmethod
    def log_error(error_message, session_key=None, user=None, endpoint=None, extra=None):
        """
        Log an error with context.
        """
        context = {
            "session_id": session_key,
            "user_id": getattr(user, "id", None) if user else "anonymous",
            "endpoint": endpoint or "unknown",
            "timestamp": datetime.utcnow().isoformat(),
        }
        if extra:
            context.update(extra)
        ConsoleLogger.logger.error(error_message, **context)

    @staticmethod
    def log_event(event_message, **context):
        """
        Log a custom event with arbitrary context.
        """
        context["timestamp"] = datetime.utcnow().isoformat()
        ConsoleLogger.logger.info(event_message, **context)