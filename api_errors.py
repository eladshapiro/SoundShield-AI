"""
Standardized API Error Handling for SoundShield-AI

Provides consistent error response format across all API endpoints.
"""

from flask import jsonify
from datetime import datetime


class APIError(Exception):
    """Standardized API error with structured JSON response."""

    def __init__(self, code: str, message: str, status_code: int = 400, details: str = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.details = details

    def to_response(self):
        body = {
            'error': {
                'code': self.code,
                'message': self.message,
                'timestamp': datetime.utcnow().isoformat() + 'Z',
            }
        }
        if self.details:
            body['error']['details'] = self.details
        return jsonify(body), self.status_code


# --- Pre-defined error factories ---

def file_not_selected():
    return APIError('FILE_NOT_SELECTED', 'No file was selected for upload.', 400)

def file_type_not_allowed(filename: str):
    return APIError('FILE_TYPE_NOT_ALLOWED', f'File type not allowed: {filename}', 415)

def file_too_large(max_mb: int):
    return APIError('FILE_TOO_LARGE', f'File exceeds maximum size of {max_mb} MB.', 413)

def analysis_not_found(analysis_id):
    return APIError('ANALYSIS_NOT_FOUND', f'Analysis {analysis_id} not found.', 404)

def analysis_failed(detail: str):
    return APIError('ANALYSIS_FAILED', 'Audio analysis failed.', 500, details=detail)

def internal_error(detail: str = None):
    return APIError('INTERNAL_ERROR', 'An internal server error occurred.', 500, details=detail)

def invalid_language(lang: str):
    return APIError('INVALID_LANGUAGE', f'Unsupported language: {lang}. Use "en" or "he".', 400)


def register_error_handlers(app):
    """Register API error handlers on a Flask app."""

    @app.errorhandler(APIError)
    def handle_api_error(error):
        return error.to_response()

    @app.errorhandler(413)
    def handle_request_entity_too_large(error):
        return file_too_large(500).to_response()

    @app.errorhandler(404)
    def handle_not_found(error):
        return APIError('NOT_FOUND', 'The requested resource was not found.', 404).to_response()

    @app.errorhandler(500)
    def handle_internal_error(error):
        return internal_error().to_response()
