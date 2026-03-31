"""
Centralized Configuration for SoundShield-AI

All detection thresholds, paths, and tunable parameters in one place.
Supports environment variable overrides and .env file loading.

תצורה מרכזית למערכת SoundShield-AI
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Tuple

# Load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def _env_float(key: str, default: float) -> float:
    return float(os.getenv(key, str(default)))


def _env_int(key: str, default: int) -> int:
    return int(os.getenv(key, str(default)))


def _env_bool(key: str, default: bool) -> bool:
    val = os.getenv(key)
    if val is None:
        return default
    return val.lower() in ('1', 'true', 'yes')


def _env_str(key: str, default: str) -> str:
    return os.getenv(key, default)


# ---------------------------------------------------------------------------
# Audio Analyzer
# ---------------------------------------------------------------------------
@dataclass
class AudioAnalyzerConfig:
    sample_rate: int = _env_int('AUDIO_SAMPLE_RATE', 22050)
    calibration_seconds: float = _env_float('AUDIO_CALIBRATION_SECONDS', 30.0)
    silence_threshold_min: float = _env_float('AUDIO_SILENCE_THRESHOLD_MIN', 0.005)
    silence_threshold_multiplier: float = _env_float('AUDIO_SILENCE_THRESHOLD_MULTIPLIER', 0.5)
    loud_threshold_min: float = _env_float('AUDIO_LOUD_THRESHOLD_MIN', 0.1)
    loud_threshold_sigma: float = _env_float('AUDIO_LOUD_THRESHOLD_SIGMA', 3.0)
    default_silence_threshold: float = _env_float('AUDIO_DEFAULT_SILENCE_THRESHOLD', 0.01)
    default_loud_threshold: float = _env_float('AUDIO_DEFAULT_LOUD_THRESHOLD', 0.3)
    segment_length: float = _env_float('AUDIO_SEGMENT_LENGTH', 5.0)
    frame_length: int = _env_int('AUDIO_FRAME_LENGTH', 2048)
    hop_length: int = _env_int('AUDIO_HOP_LENGTH', 512)
    n_mfcc: int = _env_int('AUDIO_N_MFCC', 13)


# ---------------------------------------------------------------------------
# Cry Detector
# ---------------------------------------------------------------------------
@dataclass
class CryDetectorConfig:
    frequency_range: Tuple[int, int] = (
        _env_int('CRY_FREQ_MIN', 200),
        _env_int('CRY_FREQ_MAX', 800),
    )
    duration_range: Tuple[float, float] = (0.5, 10.0)
    energy_threshold: float = _env_float('CRY_ENERGY_THRESHOLD', 0.02)
    pitch_variance_threshold: float = _env_float('CRY_PITCH_VARIANCE_THRESHOLD', 0.15)
    spectral_rolloff_threshold: float = _env_float('CRY_SPECTRAL_ROLLOFF_THRESHOLD', 0.4)
    zero_crossing_rate_threshold: float = _env_float('CRY_ZCR_THRESHOLD', 0.02)
    # Cry intensity thresholds
    intensity_low_max: float = _env_float('CRY_INTENSITY_LOW_MAX', 0.12)
    intensity_medium_max: float = _env_float('CRY_INTENSITY_MEDIUM_MAX', 0.2)
    # Scoring
    min_score_ratio: float = _env_float('CRY_MIN_SCORE_RATIO', 0.50)
    voiced_ratio_threshold: float = _env_float('CRY_VOICED_RATIO_THRESHOLD', 0.3)
    am_depth_threshold: float = _env_float('CRY_AM_DEPTH_THRESHOLD', 0.15)
    spectral_flatness_threshold: float = _env_float('CRY_SPECTRAL_FLATNESS_THRESHOLD', 0.3)
    f0_std_threshold: float = _env_float('CRY_F0_STD_THRESHOLD', 20.0)
    # Response detection
    response_window: float = _env_float('CRY_RESPONSE_WINDOW', 10.0)
    min_response_duration: float = _env_float('CRY_MIN_RESPONSE_DURATION', 1.0)
    response_energy_threshold: float = _env_float('CRY_RESPONSE_ENERGY_THRESHOLD', 0.05)
    response_pitch_threshold: int = _env_int('CRY_RESPONSE_PITCH_THRESHOLD', 100)
    # Response quality thresholds
    response_quality_poor_max: float = _env_float('CRY_RESPONSE_QUALITY_POOR_MAX', 0.08)
    response_quality_adequate_max: float = _env_float('CRY_RESPONSE_QUALITY_ADEQUATE_MAX', 0.15)


# ---------------------------------------------------------------------------
# Violence Detector
# ---------------------------------------------------------------------------
@dataclass
class ViolenceDetectorConfig:
    min_energy_gate: float = _env_float('VIOLENCE_MIN_ENERGY_GATE', 0.05)
    merge_gap_seconds: float = _env_float('VIOLENCE_MERGE_GAP_SECONDS', 1.5)
    max_merged_duration_seconds: float = _env_float('VIOLENCE_MAX_MERGED_DURATION', 20.0)
    # Shouting thresholds
    shouting_energy: float = _env_float('VIOLENCE_SHOUTING_ENERGY', 0.12)
    shouting_freq_variance: float = _env_float('VIOLENCE_SHOUTING_FREQ_VARIANCE', 0.4)
    shouting_spectral_rolloff: float = _env_float('VIOLENCE_SHOUTING_SPECTRAL_ROLLOFF', 0.7)
    shouting_duration: float = _env_float('VIOLENCE_SHOUTING_DURATION', 0.3)
    # Aggressive tone thresholds
    aggressive_energy: float = _env_float('VIOLENCE_AGGRESSIVE_ENERGY', 0.10)
    aggressive_pitch_variance: float = _env_float('VIOLENCE_AGGRESSIVE_PITCH_VARIANCE', 0.35)
    aggressive_spectral_bandwidth: float = _env_float('VIOLENCE_AGGRESSIVE_SPECTRAL_BW', 0.6)
    aggressive_zcr: float = _env_float('VIOLENCE_AGGRESSIVE_ZCR', 0.08)
    # Threatening thresholds
    threatening_energy: float = _env_float('VIOLENCE_THREATENING_ENERGY', 0.08)
    threatening_freq_low: int = _env_int('VIOLENCE_THREATENING_FREQ_LOW', 80)
    threatening_spectral_contrast: float = _env_float('VIOLENCE_THREATENING_CONTRAST', 0.5)
    threatening_duration: float = _env_float('VIOLENCE_THREATENING_DURATION', 1.0)
    # Physical violence thresholds
    physical_energy_spike: float = _env_float('VIOLENCE_PHYSICAL_ENERGY_SPIKE', 0.4)
    physical_hf_content: float = _env_float('VIOLENCE_PHYSICAL_HF_CONTENT', 0.8)
    physical_rapid_changes: float = _env_float('VIOLENCE_PHYSICAL_RAPID_CHANGES', 0.6)
    # Context analysis
    before_violence_window: float = _env_float('VIOLENCE_BEFORE_WINDOW', 5.0)
    after_violence_window: float = _env_float('VIOLENCE_AFTER_WINDOW', 5.0)
    silence_after_violence: float = _env_float('VIOLENCE_SILENCE_AFTER', 2.0)
    continued_distress: float = _env_float('VIOLENCE_CONTINUED_DISTRESS', 10.0)


# ---------------------------------------------------------------------------
# Emotion Detector
# ---------------------------------------------------------------------------
@dataclass
class EmotionDetectorConfig:
    confidence_threshold: float = _env_float('EMOTION_CONFIDENCE_THRESHOLD', 0.6)
    min_segment_length: float = _env_float('EMOTION_MIN_SEGMENT_LENGTH', 0.5)
    # Anger thresholds
    anger_energy: float = _env_float('EMOTION_ANGER_ENERGY', 0.08)
    anger_pitch_variance: float = _env_float('EMOTION_ANGER_PITCH_VARIANCE', 0.3)
    anger_spectral_rolloff: float = _env_float('EMOTION_ANGER_SPECTRAL_ROLLOFF', 0.6)
    # Stress thresholds
    stress_energy_variance: float = _env_float('EMOTION_STRESS_ENERGY_VARIANCE', 0.03)
    stress_zcr: float = _env_float('EMOTION_STRESS_ZCR', 0.05)
    stress_spectral_centroid_variance: float = _env_float('EMOTION_STRESS_SC_VARIANCE', 0.2)
    # Calm thresholds
    calm_energy: float = _env_float('EMOTION_CALM_ENERGY', 0.01)
    calm_pitch_stability: float = _env_float('EMOTION_CALM_PITCH_STABILITY', 0.8)
    calm_spectral_rolloff: float = _env_float('EMOTION_CALM_SPECTRAL_ROLLOFF', 0.4)
    # Aggression thresholds
    aggression_energy: float = _env_float('EMOTION_AGGRESSION_ENERGY', 0.2)
    aggression_pitch_variance: float = _env_float('EMOTION_AGGRESSION_PITCH_VARIANCE', 0.4)
    aggression_spectral_bandwidth: float = _env_float('EMOTION_AGGRESSION_SPECTRAL_BW', 0.7)


# ---------------------------------------------------------------------------
# Neglect Detector
# ---------------------------------------------------------------------------
@dataclass
class NeglectDetectorConfig:
    unanswered_cry_duration: float = _env_float('NEGLECT_UNANSWERED_CRY_DURATION', 30.0)
    min_cry_duration: float = _env_float('NEGLECT_MIN_CRY_DURATION', 5.0)
    response_window: float = _env_float('NEGLECT_RESPONSE_WINDOW', 15.0)
    silence_after_distress: float = _env_float('NEGLECT_SILENCE_AFTER_DISTRESS', 60.0)
    repeated_unanswered_cries: int = _env_int('NEGLECT_REPEATED_UNANSWERED', 3)
    long_term_silence: float = _env_float('NEGLECT_LONG_TERM_SILENCE', 300.0)
    # Prolonged silence pattern
    prolonged_silence_duration: float = _env_float('NEGLECT_PROLONGED_SILENCE_DURATION', 120.0)
    prolonged_silence_energy: float = _env_float('NEGLECT_PROLONGED_SILENCE_ENERGY', 0.02)
    # Lack of interaction pattern
    adult_speech_threshold: float = _env_float('NEGLECT_ADULT_SPEECH_THRESHOLD', 0.1)
    interaction_window: float = _env_float('NEGLECT_INTERACTION_WINDOW', 3600.0)
    # Ignored distress pattern
    distress_duration_threshold: float = _env_float('NEGLECT_DISTRESS_DURATION', 20.0)
    response_energy_threshold: float = _env_float('NEGLECT_RESPONSE_ENERGY', 0.08)


# ---------------------------------------------------------------------------
# Inappropriate Language Detector
# ---------------------------------------------------------------------------
@dataclass
class LanguageDetectorConfig:
    words_file: str = _env_str('LANGUAGE_WORDS_FILE', 'inappropriate_words.json')
    default_language: str = _env_str('LANGUAGE_DEFAULT', 'en')


# ---------------------------------------------------------------------------
# Advanced Analyzer (ML models)
# ---------------------------------------------------------------------------
@dataclass
class AdvancedAnalyzerConfig:
    use_advanced: bool = _env_bool('USE_ADVANCED_MODELS', True)
    whisper_model: str = _env_str('WHISPER_MODEL', 'base')
    hubert_model: str = _env_str('HUBERT_MODEL', 'superb/hubert-large-superb-er')
    chunk_duration: float = _env_float('ADVANCED_CHUNK_DURATION', 7.0)


# ---------------------------------------------------------------------------
# Security
# ---------------------------------------------------------------------------
@dataclass
class SecurityConfig:
    secret_key: str = _env_str('SECRET_KEY', 'soundshield-dev-key-change-in-production')
    rate_limit_default: str = _env_str('RATE_LIMIT_DEFAULT', '60/minute')
    rate_limit_upload: str = _env_str('RATE_LIMIT_UPLOAD', '10/minute')
    rate_limit_api: str = _env_str('RATE_LIMIT_API', '120/minute')
    enable_security_headers: bool = _env_bool('ENABLE_SECURITY_HEADERS', True)
    hsts_max_age: int = _env_int('HSTS_MAX_AGE', 31536000)
    auth_enabled: bool = _env_bool('AUTH_ENABLED', False)
    jwt_expiry_hours: int = _env_int('JWT_EXPIRY_HOURS', 24)


# ---------------------------------------------------------------------------
# Web Application
# ---------------------------------------------------------------------------
@dataclass
class WebAppConfig:
    upload_folder: str = _env_str('UPLOAD_FOLDER', 'uploads')
    reports_folder: str = _env_str('REPORTS_FOLDER', 'reports')
    max_content_length: int = _env_int('MAX_UPLOAD_SIZE_MB', 500) * 1024 * 1024
    max_workers: int = _env_int('ANALYSIS_MAX_WORKERS', 2)
    allowed_extensions: tuple = ('wav', 'mp3', 'm4a', 'flac', 'aac', 'ogg')
    host: str = _env_str('WEB_HOST', '0.0.0.0')
    port: int = _env_int('WEB_PORT', 5000)
    debug: bool = _env_bool('WEB_DEBUG', True)
    cors_origins: str = _env_str('CORS_ORIGINS', '*')


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
@dataclass
class DatabaseConfig:
    db_path: str = _env_str('DATABASE_PATH', 'soundshield.db')
    retention_days: int = _env_int('DATA_RETENTION_DAYS', 90)


# ---------------------------------------------------------------------------
# Pipeline Constants
# ---------------------------------------------------------------------------
@dataclass
class PipelineConfig:
    supported_languages: tuple = ('en', 'he')
    supported_formats: tuple = ('.wav', '.mp3', '.m4a', '.flac', '.aac', '.ogg')
    max_audio_length_seconds: int = _env_int('MAX_AUDIO_LENGTH', 3600)
    min_audio_length_seconds: int = _env_int('MIN_AUDIO_LENGTH', 1)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
@dataclass
class LoggingConfig:
    log_level: str = _env_str('LOG_LEVEL', 'INFO')
    log_format: str = _env_str('LOG_FORMAT', 'json')  # 'json' or 'text'
    log_file: str = _env_str('LOG_FILE', 'soundshield.log')
    enable_correlation_id: bool = _env_bool('ENABLE_CORRELATION_ID', True)


# ---------------------------------------------------------------------------
# Application-wide config singleton
# ---------------------------------------------------------------------------
@dataclass
class SoundShieldConfig:
    audio: AudioAnalyzerConfig = field(default_factory=AudioAnalyzerConfig)
    cry: CryDetectorConfig = field(default_factory=CryDetectorConfig)
    violence: ViolenceDetectorConfig = field(default_factory=ViolenceDetectorConfig)
    emotion: EmotionDetectorConfig = field(default_factory=EmotionDetectorConfig)
    neglect: NeglectDetectorConfig = field(default_factory=NeglectDetectorConfig)
    language: LanguageDetectorConfig = field(default_factory=LanguageDetectorConfig)
    advanced: AdvancedAnalyzerConfig = field(default_factory=AdvancedAnalyzerConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    web: WebAppConfig = field(default_factory=WebAppConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    logging_config: LoggingConfig = field(default_factory=LoggingConfig)

    # Application metadata
    version: str = '2.0.0'
    app_name: str = 'SoundShield-AI'


# Global config instance — import this in other modules
config = SoundShieldConfig()
