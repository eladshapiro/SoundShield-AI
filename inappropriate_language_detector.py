"""
Inappropriate Language Detection Module
"""

import re
import os
import json
import sys
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# Setup logging for debug output
logging.basicConfig(
    level=logging.DEBUG,
    format='%(message)s',
    stream=sys.stdout,
    force=True
)
logger = logging.getLogger(__name__)

# Force stdout to be unbuffered for immediate output
try:
    sys.stdout.reconfigure(line_buffering=True)
except:
    pass

@dataclass
class InappropriateWord:
    """Class for inappropriate word detection"""
    word: str
    timestamp: float
    severity: str  # low, medium, high, critical
    context: str
    language: str  # hebrew, english

class InappropriateLanguageDetector:
    """
    Detects inappropriate language and profanity in audio transcriptions
    """
    
    def __init__(self, words_file: str = 'inappropriate_words.json'):
        """
        Initialize inappropriate language detector
        
        Args:
            words_file: Path to JSON file containing inappropriate words.
                       If file doesn't exist, will use default words.
        """
        self.words_file = words_file
        self.inappropriate_words_hebrew, self.inappropriate_words_english = self._load_words_from_file()
        self.context_indicators = self._load_context_indicators()
    
    def _load_words_from_file(self) -> Tuple[Dict[str, Dict], Dict[str, Dict]]:
        """
        Load inappropriate words from JSON file or use defaults
        Returns tuple of (hebrew_words, english_words)
        """
        # Try to load from file
        if os.path.exists(self.words_file):
            try:
                with open(self.words_file, 'r', encoding='utf-8') as f:
                    words_data = json.load(f)
                    hebrew_words = words_data.get('hebrew', {})
                    english_words = words_data.get('english', {})
                    print(f"[OK] Loaded {len(hebrew_words)} Hebrew words and {len(english_words)} English words from {self.words_file}")
                    return hebrew_words, english_words
            except Exception as e:
                print(f"[WARNING] Error loading words from {self.words_file}: {e}")
                print("   Using default words instead")
        
        # Fallback to default words
        print("[INFO] Using default inappropriate words (built-in)")
        return self._load_hebrew_words(), self._load_english_words()
        
    def _load_hebrew_words(self) -> Dict[str, Dict]:
        """
        Load Hebrew inappropriate words dictionary - comprehensive list
        """
        return {
            # Profanity and vulgar words - critical severity
            'זונה': {'severity': 'critical', 'category': 'profanity'},
            'זונות': {'severity': 'critical', 'category': 'profanity'},
            'כוסית': {'severity': 'critical', 'category': 'profanity'},
            'כוס': {'severity': 'critical', 'category': 'profanity'},
            'מזדיין': {'severity': 'critical', 'category': 'profanity'},
            'מזדיינת': {'severity': 'critical', 'category': 'profanity'},
            'מזדיינים': {'severity': 'critical', 'category': 'profanity'},
            'תזדיין': {'severity': 'critical', 'category': 'profanity'},
            'תזדייני': {'severity': 'critical', 'category': 'profanity'},
            'תזדיינו': {'severity': 'critical', 'category': 'profanity'},
            'זין': {'severity': 'critical', 'category': 'profanity'},
            'זיינים': {'severity': 'critical', 'category': 'profanity'},
            'מחורבן': {'severity': 'high', 'category': 'profanity'},
            'מחורבנת': {'severity': 'high', 'category': 'profanity'},
            'מחורבנים': {'severity': 'high', 'category': 'profanity'},
            'מחרבן': {'severity': 'high', 'category': 'profanity'},
            'מחרבנת': {'severity': 'high', 'category': 'profanity'},
            'חרא': {'severity': 'critical', 'category': 'profanity'},
            'חראת': {'severity': 'critical', 'category': 'profanity'},
            'חראים': {'severity': 'critical', 'category': 'profanity'},
            'שרמוטה': {'severity': 'critical', 'category': 'profanity'},
            'שרמוטות': {'severity': 'critical', 'category': 'profanity'},
            'שרמוט': {'severity': 'critical', 'category': 'profanity'},
            'שרמוטים': {'severity': 'critical', 'category': 'profanity'},
            'בולבול': {'severity': 'critical', 'category': 'profanity'},
            'בולבולים': {'severity': 'critical', 'category': 'profanity'},
            'כוס אמא': {'severity': 'critical', 'category': 'profanity'},
            'כוס של אמא': {'severity': 'critical', 'category': 'profanity'},
            'זין של אמא': {'severity': 'critical', 'category': 'profanity'},
            
            # Insults and offensive words - medium to critical
            'טמבל': {'severity': 'medium', 'category': 'insult'},
            'טמבלה': {'severity': 'medium', 'category': 'insult'},
            'טמבלים': {'severity': 'medium', 'category': 'insult'},
            'טמבליות': {'severity': 'medium', 'category': 'insult'},
            'אידיוט': {'severity': 'medium', 'category': 'insult'},
            'אידיוטית': {'severity': 'medium', 'category': 'insult'},
            'אידיוטים': {'severity': 'medium', 'category': 'insult'},
            'אידיוטיות': {'severity': 'medium', 'category': 'insult'},
            'מטומטם': {'severity': 'medium', 'category': 'insult'},
            'מטומטמת': {'severity': 'medium', 'category': 'insult'},
            'מטומטמים': {'severity': 'medium', 'category': 'insult'},
            'מטומטמות': {'severity': 'medium', 'category': 'insult'},
            'דפוק': {'severity': 'high', 'category': 'insult'},
            'דפוקה': {'severity': 'high', 'category': 'insult'},
            'דפוקים': {'severity': 'high', 'category': 'insult'},
            'דפוקות': {'severity': 'high', 'category': 'insult'},
            'מפגר': {'severity': 'critical', 'category': 'insult'},
            'מפגרת': {'severity': 'critical', 'category': 'insult'},
            'מפגרים': {'severity': 'critical', 'category': 'insult'},
            'מפגרות': {'severity': 'critical', 'category': 'insult'},
            'פסיכי': {'severity': 'high', 'category': 'insult'},
            'פסיכית': {'severity': 'high', 'category': 'insult'},
            'פסיכים': {'severity': 'high', 'category': 'insult'},
            'פסיכיות': {'severity': 'high', 'category': 'insult'},
            'משוגע': {'severity': 'medium', 'category': 'insult'},
            'משוגעת': {'severity': 'medium', 'category': 'insult'},
            'משוגעים': {'severity': 'medium', 'category': 'insult'},
            'משוגעות': {'severity': 'medium', 'category': 'insult'},
            'מטורף': {'severity': 'medium', 'category': 'insult'},
            'מטורפת': {'severity': 'medium', 'category': 'insult'},
            'מטורפים': {'severity': 'medium', 'category': 'insult'},
            'מטורפות': {'severity': 'medium', 'category': 'insult'},
            'מטורף עליך': {'severity': 'high', 'category': 'insult'},
            'מטורפת עליך': {'severity': 'high', 'category': 'insult'},
            'חתיכת': {'severity': 'high', 'category': 'insult'},
            'חתיכת זונה': {'severity': 'critical', 'category': 'insult'},
            'חתיכת שרמוטה': {'severity': 'critical', 'category': 'insult'},
            'חתיכת מפגר': {'severity': 'critical', 'category': 'insult'},
            'חתיכת אידיוט': {'severity': 'high', 'category': 'insult'},
            'בן זונה': {'severity': 'critical', 'category': 'insult'},
            'בת זונה': {'severity': 'critical', 'category': 'insult'},
            'בני זונות': {'severity': 'critical', 'category': 'insult'},
            'בנות זונות': {'severity': 'critical', 'category': 'insult'},
            'בן של זונה': {'severity': 'critical', 'category': 'insult'},
            'בת של זונה': {'severity': 'critical', 'category': 'insult'},
            'בן של שרמוטה': {'severity': 'critical', 'category': 'insult'},
            'בת של שרמוטה': {'severity': 'critical', 'category': 'insult'},
            
            # Threats and aggression - high to critical
            'אני אכה אותך': {'severity': 'critical', 'category': 'threat'},
            'אני אזרוק אותך': {'severity': 'critical', 'category': 'threat'},
            'אני אכה': {'severity': 'critical', 'category': 'threat'},
            'אני אזרוק': {'severity': 'critical', 'category': 'threat'},
            'אני אכה אותך חזק': {'severity': 'critical', 'category': 'threat'},
            'אני אכה אותך קשה': {'severity': 'critical', 'category': 'threat'},
            'אני אכה אותך חזק מאוד': {'severity': 'critical', 'category': 'threat'},
            'אני אזרוק אותך החוצה': {'severity': 'critical', 'category': 'threat'},
            'אני אעניש אותך': {'severity': 'high', 'category': 'threat'},
            'אני אעניש': {'severity': 'high', 'category': 'threat'},
            'תסגור את הפה': {'severity': 'high', 'category': 'aggression'},
            'תסגרי את הפה': {'severity': 'high', 'category': 'aggression'},
            'תסגור את הפה שלך': {'severity': 'high', 'category': 'aggression'},
            'תשתוק': {'severity': 'medium', 'category': 'aggression'},
            'תשתקי': {'severity': 'medium', 'category': 'aggression'},
            'תשתקו': {'severity': 'medium', 'category': 'aggression'},
            'תפסיק': {'severity': 'medium', 'category': 'aggression'},
            'תפסיקי': {'severity': 'medium', 'category': 'aggression'},
            'תפסיקו': {'severity': 'medium', 'category': 'aggression'},
            'תפסיק כבר': {'severity': 'high', 'category': 'aggression'},
            'תפסיקי כבר': {'severity': 'high', 'category': 'aggression'},
            'תפסיקו כבר': {'severity': 'high', 'category': 'aggression'},
            'די': {'severity': 'low', 'category': 'aggression'},
            'דיי': {'severity': 'low', 'category': 'aggression'},
            'מספיק': {'severity': 'low', 'category': 'aggression'},
            'מספיק כבר': {'severity': 'medium', 'category': 'aggression'},
            'תעצור': {'severity': 'medium', 'category': 'aggression'},
            'תעצרי': {'severity': 'medium', 'category': 'aggression'},
            'תעצרו': {'severity': 'medium', 'category': 'aggression'},
            'תעצור כבר': {'severity': 'high', 'category': 'aggression'},
            'תעצרי כבר': {'severity': 'high', 'category': 'aggression'},
            'תעצרו כבר': {'severity': 'high', 'category': 'aggression'},
            'תשתוק כבר': {'severity': 'high', 'category': 'aggression'},
            'תשתקי כבר': {'severity': 'high', 'category': 'aggression'},
            'תשתקו כבר': {'severity': 'high', 'category': 'aggression'},
            'תשתוק מיד': {'severity': 'high', 'category': 'aggression'},
            'תשתקי מיד': {'severity': 'high', 'category': 'aggression'},
            'תשתקו מיד': {'severity': 'high', 'category': 'aggression'},
            
            # Negative labeling and emotional abuse
            'אתה ילד רע': {'severity': 'medium', 'category': 'negative_labeling'},
            'את ילדה רעה': {'severity': 'medium', 'category': 'negative_labeling'},
            'אתה ילד רע מאוד': {'severity': 'high', 'category': 'negative_labeling'},
            'את ילדה רעה מאוד': {'severity': 'high', 'category': 'negative_labeling'},
            'אתה ילד רע ועצלן': {'severity': 'high', 'category': 'negative_labeling'},
            'את ילדה רעה ועצלנית': {'severity': 'high', 'category': 'negative_labeling'},
            'לא אוהב אותך': {'severity': 'medium', 'category': 'emotional_abuse'},
            'לא אוהבת אותך': {'severity': 'medium', 'category': 'emotional_abuse'},
            'לא אוהבים אותך': {'severity': 'medium', 'category': 'emotional_abuse'},
            'לא אוהבות אותך': {'severity': 'medium', 'category': 'emotional_abuse'},
            'אף אחד לא אוהב אותך': {'severity': 'high', 'category': 'emotional_abuse'},
            'אף אחד לא אוהב אותך': {'severity': 'high', 'category': 'emotional_abuse'},
            'כולם שונאים אותך': {'severity': 'high', 'category': 'emotional_abuse'},
            'כולם שונאות אותך': {'severity': 'high', 'category': 'emotional_abuse'},
            'אתה לא שווה כלום': {'severity': 'high', 'category': 'emotional_abuse'},
            'את לא שווה כלום': {'severity': 'high', 'category': 'emotional_abuse'},
            'אתה לא שווה': {'severity': 'medium', 'category': 'emotional_abuse'},
            'את לא שווה': {'severity': 'medium', 'category': 'emotional_abuse'},
            
            # Additional profanity and vulgar words
            'פיפי': {'severity': 'low', 'category': 'vulgar'},
            'קקי': {'severity': 'low', 'category': 'vulgar'},
            'פוץ': {'severity': 'critical', 'category': 'profanity'},
            'פוצים': {'severity': 'critical', 'category': 'profanity'},
            'תחת': {'severity': 'high', 'category': 'vulgar'},
            'ישבן': {'severity': 'low', 'category': 'vulgar'},
            'תחת שלך': {'severity': 'high', 'category': 'profanity'},
            'אמא שלך': {'severity': 'critical', 'category': 'profanity'},
            'אמא שלך זונה': {'severity': 'critical', 'category': 'profanity'},
            'אמא שלך שרמוטה': {'severity': 'critical', 'category': 'profanity'},
            'אבא שלך': {'severity': 'high', 'category': 'profanity'},
            'אבא שלך זונה': {'severity': 'critical', 'category': 'profanity'},
            'כל המשפחה שלך': {'severity': 'high', 'category': 'profanity'},
            'כל המשפחה שלך זונות': {'severity': 'critical', 'category': 'profanity'},
            
            # More insults and derogatory terms
            'חמור': {'severity': 'medium', 'category': 'insult'},
            'חמורה': {'severity': 'medium', 'category': 'insult'},
            'חמורים': {'severity': 'medium', 'category': 'insult'},
            'חמורות': {'severity': 'medium', 'category': 'insult'},
            'חזיר': {'severity': 'medium', 'category': 'insult'},
            'חזירה': {'severity': 'medium', 'category': 'insult'},
            'חזירים': {'severity': 'medium', 'category': 'insult'},
            'חזירות': {'severity': 'medium', 'category': 'insult'},
            'כלב': {'severity': 'medium', 'category': 'insult'},
            'כלבה': {'severity': 'high', 'category': 'insult'},
            'כלבים': {'severity': 'medium', 'category': 'insult'},
            'כלבות': {'severity': 'high', 'category': 'insult'},
            'חתול': {'severity': 'low', 'category': 'insult'},
            'חתולה': {'severity': 'medium', 'category': 'insult'},
            'עכבר': {'severity': 'low', 'category': 'insult'},
            'עכברוש': {'severity': 'medium', 'category': 'insult'},
            'נחש': {'severity': 'medium', 'category': 'insult'},
            'תולעת': {'severity': 'medium', 'category': 'insult'},
            'חרק': {'severity': 'low', 'category': 'insult'},
            'פרעה': {'severity': 'medium', 'category': 'insult'},
            'פרעה שלך': {'severity': 'high', 'category': 'insult'},
            'מטומטם עליך': {'severity': 'high', 'category': 'insult'},
            'דפוק עליך': {'severity': 'high', 'category': 'insult'},
            'מפגר עליך': {'severity': 'critical', 'category': 'insult'},
            'אידיוט עליך': {'severity': 'high', 'category': 'insult'},
            'טמבל עליך': {'severity': 'medium', 'category': 'insult'},
            'פסיכי עליך': {'severity': 'high', 'category': 'insult'},
            'משוגע עליך': {'severity': 'medium', 'category': 'insult'},
            'מטורף עליך': {'severity': 'medium', 'category': 'insult'},
            'חתיכת מטומטם': {'severity': 'high', 'category': 'insult'},
            'חתיכת דפוק': {'severity': 'high', 'category': 'insult'},
            'חתיכת מפגר': {'severity': 'critical', 'category': 'insult'},
            'חתיכת אידיוט': {'severity': 'high', 'category': 'insult'},
            'חתיכת טמבל': {'severity': 'medium', 'category': 'insult'},
            'חתיכת פסיכי': {'severity': 'high', 'category': 'insult'},
            'חתיכת משוגע': {'severity': 'medium', 'category': 'insult'},
            'חתיכת מטורף': {'severity': 'medium', 'category': 'insult'},
            'חתיכת חמור': {'severity': 'medium', 'category': 'insult'},
            'חתיכת חזיר': {'severity': 'medium', 'category': 'insult'},
            'חתיכת כלב': {'severity': 'high', 'category': 'insult'},
            'חתיכת כלבה': {'severity': 'critical', 'category': 'insult'},
            
            # More threats and aggressive phrases
            'אני אכה אותך חזק': {'severity': 'critical', 'category': 'threat'},
            'אני אכה אותך קשה': {'severity': 'critical', 'category': 'threat'},
            'אני אכה אותך מאוד חזק': {'severity': 'critical', 'category': 'threat'},
            'אני אזרוק אותך החוצה': {'severity': 'critical', 'category': 'threat'},
            'אני אזרוק אותך מהגן': {'severity': 'critical', 'category': 'threat'},
            'אני אעניש אותך': {'severity': 'high', 'category': 'threat'},
            'אני אעניש אותך קשה': {'severity': 'critical', 'category': 'threat'},
            'אני אעניש אותך מאוד': {'severity': 'critical', 'category': 'threat'},
            'אני אכה אותך על היד': {'severity': 'critical', 'category': 'threat'},
            'אני אכה אותך על הישבן': {'severity': 'critical', 'category': 'threat'},
            'אני אכה אותך על הראש': {'severity': 'critical', 'category': 'threat'},
            'אני אכה אותך על הפנים': {'severity': 'critical', 'category': 'threat'},
            'אני אכה אותך על הגב': {'severity': 'critical', 'category': 'threat'},
            'אני אכה אותך על הרגל': {'severity': 'critical', 'category': 'threat'},
            'אני אכה אותך על הידיים': {'severity': 'critical', 'category': 'threat'},
            'אני אכה אותך על הרגליים': {'severity': 'critical', 'category': 'threat'},
            'אני אכה אותך על כל הגוף': {'severity': 'critical', 'category': 'threat'},
            'אני אכה אותך עד שתשתוק': {'severity': 'critical', 'category': 'threat'},
            'אני אכה אותך עד שתפסיק': {'severity': 'critical', 'category': 'threat'},
            'אני אכה אותך עד שתעשה מה שאני אומר': {'severity': 'critical', 'category': 'threat'},
            'אני אכה אותך אם לא תפסיק': {'severity': 'critical', 'category': 'threat'},
            'אני אכה אותך אם לא תשתק': {'severity': 'critical', 'category': 'threat'},
            'אני אכה אותך אם לא תעשה מה שאני אומר': {'severity': 'critical', 'category': 'threat'},
            'אני אזרוק אותך אם לא תפסיק': {'severity': 'critical', 'category': 'threat'},
            'אני אזרוק אותך אם לא תשתק': {'severity': 'critical', 'category': 'threat'},
            'אני אעניש אותך אם לא תפסיק': {'severity': 'critical', 'category': 'threat'},
            'אני אעניש אותך אם לא תשתק': {'severity': 'critical', 'category': 'threat'},
            'אני אעניש אותך אם לא תעשה מה שאני אומר': {'severity': 'critical', 'category': 'threat'},
            'אני אכה אותך עכשיו': {'severity': 'critical', 'category': 'threat'},
            'אני אכה אותך מיד': {'severity': 'critical', 'category': 'threat'},
            'אני אכה אותך תיכף': {'severity': 'critical', 'category': 'threat'},
            'אני אזרוק אותך עכשיו': {'severity': 'critical', 'category': 'threat'},
            'אני אזרוק אותך מיד': {'severity': 'critical', 'category': 'threat'},
            'אני אזרוק אותך תיכף': {'severity': 'critical', 'category': 'threat'},
            'אני אעניש אותך עכשיו': {'severity': 'high', 'category': 'threat'},
            'אני אעניש אותך מיד': {'severity': 'high', 'category': 'threat'},
            'אני אעניש אותך תיכף': {'severity': 'high', 'category': 'threat'},
            
            # More aggressive commands
            'תשתוק כבר עכשיו': {'severity': 'high', 'category': 'aggression'},
            'תשתקי כבר עכשיו': {'severity': 'high', 'category': 'aggression'},
            'תשתקו כבר עכשיו': {'severity': 'high', 'category': 'aggression'},
            'תשתוק כבר מיד': {'severity': 'high', 'category': 'aggression'},
            'תשתקי כבר מיד': {'severity': 'high', 'category': 'aggression'},
            'תשתקו כבר מיד': {'severity': 'high', 'category': 'aggression'},
            'תשתוק כבר תיכף': {'severity': 'high', 'category': 'aggression'},
            'תשתקי כבר תיכף': {'severity': 'high', 'category': 'aggression'},
            'תשתקו כבר תיכף': {'severity': 'high', 'category': 'aggression'},
            'תפסיק כבר עכשיו': {'severity': 'high', 'category': 'aggression'},
            'תפסיקי כבר עכשיו': {'severity': 'high', 'category': 'aggression'},
            'תפסיקו כבר עכשיו': {'severity': 'high', 'category': 'aggression'},
            'תפסיק כבר מיד': {'severity': 'high', 'category': 'aggression'},
            'תפסיקי כבר מיד': {'severity': 'high', 'category': 'aggression'},
            'תפסיקו כבר מיד': {'severity': 'high', 'category': 'aggression'},
            'תפסיק כבר תיכף': {'severity': 'high', 'category': 'aggression'},
            'תפסיקי כבר תיכף': {'severity': 'high', 'category': 'aggression'},
            'תפסיקו כבר תיכף': {'severity': 'high', 'category': 'aggression'},
            'תעצור כבר עכשיו': {'severity': 'high', 'category': 'aggression'},
            'תעצרי כבר עכשיו': {'severity': 'high', 'category': 'aggression'},
            'תעצרו כבר עכשיו': {'severity': 'high', 'category': 'aggression'},
            'תעצור כבר מיד': {'severity': 'high', 'category': 'aggression'},
            'תעצרי כבר מיד': {'severity': 'high', 'category': 'aggression'},
            'תעצרו כבר מיד': {'severity': 'high', 'category': 'aggression'},
            'תעצור כבר תיכף': {'severity': 'high', 'category': 'aggression'},
            'תעצרי כבר תיכף': {'severity': 'high', 'category': 'aggression'},
            'תעצרו כבר תיכף': {'severity': 'high', 'category': 'aggression'},
            'תסגור את הפה כבר': {'severity': 'high', 'category': 'aggression'},
            'תסגרי את הפה כבר': {'severity': 'high', 'category': 'aggression'},
            'תסגרו את הפה כבר': {'severity': 'high', 'category': 'aggression'},
            'תסגור את הפה עכשיו': {'severity': 'high', 'category': 'aggression'},
            'תסגרי את הפה עכשיו': {'severity': 'high', 'category': 'aggression'},
            'תסגרו את הפה עכשיו': {'severity': 'high', 'category': 'aggression'},
            'תסגור את הפה מיד': {'severity': 'high', 'category': 'aggression'},
            'תסגרי את הפה מיד': {'severity': 'high', 'category': 'aggression'},
            'תסגרו את הפה מיד': {'severity': 'high', 'category': 'aggression'},
            'תסגור את הפה תיכף': {'severity': 'high', 'category': 'aggression'},
            'תסגרי את הפה תיכף': {'severity': 'high', 'category': 'aggression'},
            'תסגרו את הפה תיכף': {'severity': 'high', 'category': 'aggression'},
            
            # More negative labeling and emotional abuse
            'אתה ילד רע מאוד': {'severity': 'high', 'category': 'negative_labeling'},
            'את ילדה רעה מאוד': {'severity': 'high', 'category': 'negative_labeling'},
            'אתה ילד רע ועצלן': {'severity': 'high', 'category': 'negative_labeling'},
            'את ילדה רעה ועצלנית': {'severity': 'high', 'category': 'negative_labeling'},
            'אתה ילד רע ולא ממושמע': {'severity': 'high', 'category': 'negative_labeling'},
            'את ילדה רעה ולא ממושמעת': {'severity': 'high', 'category': 'negative_labeling'},
            'אתה ילד רע ולא טוב': {'severity': 'high', 'category': 'negative_labeling'},
            'את ילדה רעה ולא טובה': {'severity': 'high', 'category': 'negative_labeling'},
            'אתה ילד רע ולא נחמד': {'severity': 'high', 'category': 'negative_labeling'},
            'את ילדה רעה ולא נחמדה': {'severity': 'high', 'category': 'negative_labeling'},
            'אתה ילד רע ולא אוהבים אותך': {'severity': 'critical', 'category': 'emotional_abuse'},
            'את ילדה רעה ולא אוהבים אותך': {'severity': 'critical', 'category': 'emotional_abuse'},
            'אתה ילד רע ואף אחד לא אוהב אותך': {'severity': 'critical', 'category': 'emotional_abuse'},
            'את ילדה רעה ואף אחד לא אוהב אותך': {'severity': 'critical', 'category': 'emotional_abuse'},
            'אתה ילד רע וכולם שונאים אותך': {'severity': 'critical', 'category': 'emotional_abuse'},
            'את ילדה רעה וכולם שונאים אותך': {'severity': 'critical', 'category': 'emotional_abuse'},
            'אתה ילד רע וכולם שונאות אותך': {'severity': 'critical', 'category': 'emotional_abuse'},
            'את ילדה רעה וכולם שונאות אותך': {'severity': 'critical', 'category': 'emotional_abuse'},
            'אתה ילד רע ולא שווה כלום': {'severity': 'critical', 'category': 'emotional_abuse'},
            'את ילדה רעה ולא שווה כלום': {'severity': 'critical', 'category': 'emotional_abuse'},
            'אתה ילד רע ולא שווה': {'severity': 'high', 'category': 'emotional_abuse'},
            'את ילדה רעה ולא שווה': {'severity': 'high', 'category': 'emotional_abuse'},
            'אתה ילד רע ולא טוב בכלל': {'severity': 'high', 'category': 'negative_labeling'},
            'את ילדה רעה ולא טובה בכלל': {'severity': 'high', 'category': 'negative_labeling'},
            'אתה ילד רע ולא נחמד בכלל': {'severity': 'high', 'category': 'negative_labeling'},
            'את ילדה רעה ולא נחמדה בכלל': {'severity': 'high', 'category': 'negative_labeling'},
            'אתה ילד רע ולא ממושמע בכלל': {'severity': 'high', 'category': 'negative_labeling'},
            'את ילדה רעה ולא ממושמעת בכלל': {'severity': 'high', 'category': 'negative_labeling'},
            'אתה ילד רע ולא ממושמע בכלל ולא אוהבים אותך': {'severity': 'critical', 'category': 'emotional_abuse'},
            'את ילדה רעה ולא ממושמעת בכלל ולא אוהבים אותך': {'severity': 'critical', 'category': 'emotional_abuse'},
        }
    
    def _load_english_words(self) -> Dict[str, Dict]:
        """
        Load English inappropriate words dictionary - comprehensive list
        טעינת מילון מילים לא הולמות באנגלית
        """
        return {
            # Profanity - Critical
            'fuck': {'severity': 'critical', 'category': 'profanity'},
            'fucking': {'severity': 'critical', 'category': 'profanity'},
            'fucked': {'severity': 'critical', 'category': 'profanity'},
            'fucker': {'severity': 'critical', 'category': 'profanity'},
            'fuckers': {'severity': 'critical', 'category': 'profanity'},
            'shit': {'severity': 'high', 'category': 'profanity'},
            'shitting': {'severity': 'high', 'category': 'profanity'},
            'shitted': {'severity': 'high', 'category': 'profanity'},
            'damn': {'severity': 'medium', 'category': 'profanity'},
            'damned': {'severity': 'medium', 'category': 'profanity'},
            'dammit': {'severity': 'medium', 'category': 'profanity'},
            'bitch': {'severity': 'critical', 'category': 'profanity'},
            'bitches': {'severity': 'critical', 'category': 'profanity'},
            'asshole': {'severity': 'critical', 'category': 'profanity'},
            'assholes': {'severity': 'critical', 'category': 'profanity'},
            'bastard': {'severity': 'high', 'category': 'profanity'},
            'bastards': {'severity': 'high', 'category': 'profanity'},
            'hell': {'severity': 'medium', 'category': 'profanity'},
            'crap': {'severity': 'medium', 'category': 'profanity'},
            'piss': {'severity': 'high', 'category': 'profanity'},
            'pissed': {'severity': 'high', 'category': 'profanity'},
            'pissing': {'severity': 'high', 'category': 'profanity'},
            
            # Insults - Medium to Critical
            'idiot': {'severity': 'medium', 'category': 'insult'},
            'idiots': {'severity': 'medium', 'category': 'insult'},
            'stupid': {'severity': 'medium', 'category': 'insult'},
            'stupids': {'severity': 'medium', 'category': 'insult'},
            'dumb': {'severity': 'medium', 'category': 'insult'},
            'dumbass': {'severity': 'high', 'category': 'insult'},
            'moron': {'severity': 'high', 'category': 'insult'},
            'morons': {'severity': 'high', 'category': 'insult'},
            'retard': {'severity': 'critical', 'category': 'insult'},
            'retarded': {'severity': 'critical', 'category': 'insult'},
            'retards': {'severity': 'critical', 'category': 'insult'},
            'crazy': {'severity': 'medium', 'category': 'insult'},
            'crazies': {'severity': 'medium', 'category': 'insult'},
            'psycho': {'severity': 'high', 'category': 'insult'},
            'psychos': {'severity': 'high', 'category': 'insult'},
            'nuts': {'severity': 'medium', 'category': 'insult'},
            'freak': {'severity': 'medium', 'category': 'insult'},
            'freaks': {'severity': 'medium', 'category': 'insult'},
            'loser': {'severity': 'medium', 'category': 'insult'},
            'losers': {'severity': 'medium', 'category': 'insult'},
            'jerk': {'severity': 'medium', 'category': 'insult'},
            'jerks': {'severity': 'medium', 'category': 'insult'},
            'ass': {'severity': 'high', 'category': 'insult'},
            'dick': {'severity': 'critical', 'category': 'profanity'},
            'dicks': {'severity': 'critical', 'category': 'profanity'},
            'cock': {'severity': 'critical', 'category': 'profanity'},
            'cocks': {'severity': 'critical', 'category': 'profanity'},
            
            # Threats and Aggression
            'shut up': {'severity': 'high', 'category': 'aggression'},
            'shut your mouth': {'severity': 'high', 'category': 'aggression'},
            'shut your face': {'severity': 'high', 'category': 'aggression'},
            'shut it': {'severity': 'high', 'category': 'aggression'},
            'stop it': {'severity': 'medium', 'category': 'aggression'},
            'stop that': {'severity': 'medium', 'category': 'aggression'},
            'stop now': {'severity': 'high', 'category': 'aggression'},
            'enough': {'severity': 'low', 'category': 'aggression'},
            'enough already': {'severity': 'medium', 'category': 'aggression'},
            'i will hit you': {'severity': 'critical', 'category': 'threat'},
            'i will beat you': {'severity': 'critical', 'category': 'threat'},
            'i will beat': {'severity': 'critical', 'category': 'threat'},
            'i will hit': {'severity': 'critical', 'category': 'threat'},
            'i will hurt you': {'severity': 'critical', 'category': 'threat'},
            'i will hurt': {'severity': 'critical', 'category': 'threat'},
            'i will punish you': {'severity': 'high', 'category': 'threat'},
            'i will punish': {'severity': 'high', 'category': 'threat'},
            'i will smack you': {'severity': 'critical', 'category': 'threat'},
            'i will smack': {'severity': 'critical', 'category': 'threat'},
            'i will slap you': {'severity': 'critical', 'category': 'threat'},
            'i will slap': {'severity': 'critical', 'category': 'threat'},
            
            # Negative Labeling
            'you are bad': {'severity': 'medium', 'category': 'negative_labeling'},
            'you are very bad': {'severity': 'high', 'category': 'negative_labeling'},
            'you are so bad': {'severity': 'high', 'category': 'negative_labeling'},
            'bad child': {'severity': 'medium', 'category': 'negative_labeling'},
            'bad kid': {'severity': 'medium', 'category': 'negative_labeling'},
            'bad children': {'severity': 'medium', 'category': 'negative_labeling'},
            'you are stupid': {'severity': 'high', 'category': 'negative_labeling'},
            'you are dumb': {'severity': 'high', 'category': 'negative_labeling'},
            'you are an idiot': {'severity': 'high', 'category': 'negative_labeling'},
            "i don't like you": {'severity': 'medium', 'category': 'emotional_abuse'},
            "i don't like you at all": {'severity': 'high', 'category': 'emotional_abuse'},
            'nobody likes you': {'severity': 'high', 'category': 'emotional_abuse'},
            'everyone hates you': {'severity': 'high', 'category': 'emotional_abuse'},
            'you are worthless': {'severity': 'high', 'category': 'emotional_abuse'},
            'you are nothing': {'severity': 'high', 'category': 'emotional_abuse'},
            
            # Additional profanity and vulgar words
            'piss off': {'severity': 'high', 'category': 'aggression'},
            'pissed off': {'severity': 'high', 'category': 'aggression'},
            'fuck off': {'severity': 'critical', 'category': 'profanity'},
            'fuck you': {'severity': 'critical', 'category': 'profanity'},
            'fuck your': {'severity': 'critical', 'category': 'profanity'},
            'fuck your mother': {'severity': 'critical', 'category': 'profanity'},
            'fuck your mom': {'severity': 'critical', 'category': 'profanity'},
            'fuck your dad': {'severity': 'critical', 'category': 'profanity'},
            'fuck your family': {'severity': 'critical', 'category': 'profanity'},
            'son of a bitch': {'severity': 'critical', 'category': 'profanity'},
            'son of bitch': {'severity': 'critical', 'category': 'profanity'},
            'motherfucker': {'severity': 'critical', 'category': 'profanity'},
            'mother fucker': {'severity': 'critical', 'category': 'profanity'},
            'motherfuckers': {'severity': 'critical', 'category': 'profanity'},
            'mother fuckers': {'severity': 'critical', 'category': 'profanity'},
            'ass': {'severity': 'high', 'category': 'profanity'},
            'asses': {'severity': 'high', 'category': 'profanity'},
            'asswipe': {'severity': 'critical', 'category': 'profanity'},
            'ass wipe': {'severity': 'critical', 'category': 'profanity'},
            'asshole': {'severity': 'critical', 'category': 'profanity'},
            'ass holes': {'severity': 'critical', 'category': 'profanity'},
            'dickhead': {'severity': 'critical', 'category': 'profanity'},
            'dick head': {'severity': 'critical', 'category': 'profanity'},
            'dickheads': {'severity': 'critical', 'category': 'profanity'},
            'dick heads': {'severity': 'critical', 'category': 'profanity'},
            'cocksucker': {'severity': 'critical', 'category': 'profanity'},
            'cock sucker': {'severity': 'critical', 'category': 'profanity'},
            'cocksuckers': {'severity': 'critical', 'category': 'profanity'},
            'cock suckers': {'severity': 'critical', 'category': 'profanity'},
            'pussy': {'severity': 'critical', 'category': 'profanity'},
            'pussies': {'severity': 'critical', 'category': 'profanity'},
            'pussycat': {'severity': 'low', 'category': 'neutral'},  # This is OK
            'twat': {'severity': 'critical', 'category': 'profanity'},
            'twats': {'severity': 'critical', 'category': 'profanity'},
            'cunt': {'severity': 'critical', 'category': 'profanity'},
            'cunts': {'severity': 'critical', 'category': 'profanity'},
            'whore': {'severity': 'critical', 'category': 'profanity'},
            'whores': {'severity': 'critical', 'category': 'profanity'},
            'slut': {'severity': 'critical', 'category': 'profanity'},
            'sluts': {'severity': 'critical', 'category': 'profanity'},
            'slutty': {'severity': 'critical', 'category': 'profanity'},
            'douche': {'severity': 'high', 'category': 'profanity'},
            'douchebag': {'severity': 'critical', 'category': 'profanity'},
            'douche bag': {'severity': 'critical', 'category': 'profanity'},
            'douchebags': {'severity': 'critical', 'category': 'profanity'},
            'douche bags': {'severity': 'critical', 'category': 'profanity'},
            'douchebaggery': {'severity': 'critical', 'category': 'profanity'},
            'scumbag': {'severity': 'critical', 'category': 'profanity'},
            'scum bag': {'severity': 'critical', 'category': 'profanity'},
            'scumbags': {'severity': 'critical', 'category': 'profanity'},
            'scum bags': {'severity': 'critical', 'category': 'profanity'},
            'dipshit': {'severity': 'high', 'category': 'profanity'},
            'dip shit': {'severity': 'high', 'category': 'profanity'},
            'dipshits': {'severity': 'high', 'category': 'profanity'},
            'dip shits': {'severity': 'high', 'category': 'profanity'},
            'shithead': {'severity': 'critical', 'category': 'profanity'},
            'shit head': {'severity': 'critical', 'category': 'profanity'},
            'shitheads': {'severity': 'critical', 'category': 'profanity'},
            'shit heads': {'severity': 'critical', 'category': 'profanity'},
            'shitface': {'severity': 'critical', 'category': 'profanity'},
            'shit face': {'severity': 'critical', 'category': 'profanity'},
            'shitfaces': {'severity': 'critical', 'category': 'profanity'},
            'shit faces': {'severity': 'critical', 'category': 'profanity'},
            'fuckface': {'severity': 'critical', 'category': 'profanity'},
            'fuck face': {'severity': 'critical', 'category': 'profanity'},
            'fuckfaces': {'severity': 'critical', 'category': 'profanity'},
            'fuck faces': {'severity': 'critical', 'category': 'profanity'},
            'fuckhead': {'severity': 'critical', 'category': 'profanity'},
            'fuck head': {'severity': 'critical', 'category': 'profanity'},
            'fuckheads': {'severity': 'critical', 'category': 'profanity'},
            'fuck heads': {'severity': 'critical', 'category': 'profanity'},
            'fuckwit': {'severity': 'critical', 'category': 'profanity'},
            'fuck wit': {'severity': 'critical', 'category': 'profanity'},
            'fuckwits': {'severity': 'critical', 'category': 'profanity'},
            'fuck wits': {'severity': 'critical', 'category': 'profanity'},
            'fucktard': {'severity': 'critical', 'category': 'profanity'},
            'fuck tard': {'severity': 'critical', 'category': 'profanity'},
            'fucktards': {'severity': 'critical', 'category': 'profanity'},
            'fuck tards': {'severity': 'critical', 'category': 'profanity'},
            'fucknut': {'severity': 'critical', 'category': 'profanity'},
            'fuck nut': {'severity': 'critical', 'category': 'profanity'},
            'fucknuts': {'severity': 'critical', 'category': 'profanity'},
            'fuck nuts': {'severity': 'critical', 'category': 'profanity'},
            'fuckwad': {'severity': 'critical', 'category': 'profanity'},
            'fuck wad': {'severity': 'critical', 'category': 'profanity'},
            'fuckwads': {'severity': 'critical', 'category': 'profanity'},
            'fuck wads': {'severity': 'critical', 'category': 'profanity'},
            'fuckstick': {'severity': 'critical', 'category': 'profanity'},
            'fuck stick': {'severity': 'critical', 'category': 'profanity'},
            'fucksticks': {'severity': 'critical', 'category': 'profanity'},
            'fuck sticks': {'severity': 'critical', 'category': 'profanity'},
            'fuckboy': {'severity': 'critical', 'category': 'profanity'},
            'fuck boy': {'severity': 'critical', 'category': 'profanity'},
            'fuckboys': {'severity': 'critical', 'category': 'profanity'},
            'fuck boys': {'severity': 'critical', 'category': 'profanity'},
            'fuckgirl': {'severity': 'critical', 'category': 'profanity'},
            'fuck girl': {'severity': 'critical', 'category': 'profanity'},
            'fuckgirls': {'severity': 'critical', 'category': 'profanity'},
            'fuck girls': {'severity': 'critical', 'category': 'profanity'},
            'fucktoy': {'severity': 'critical', 'category': 'profanity'},
            'fuck toy': {'severity': 'critical', 'category': 'profanity'},
            'fucktoys': {'severity': 'critical', 'category': 'profanity'},
            'fuck toys': {'severity': 'critical', 'category': 'profanity'},
            'fuckup': {'severity': 'critical', 'category': 'profanity'},
            'fuck up': {'severity': 'critical', 'category': 'profanity'},
            'fuckups': {'severity': 'critical', 'category': 'profanity'},
            'fuck ups': {'severity': 'critical', 'category': 'profanity'},
            'fuckaround': {'severity': 'critical', 'category': 'profanity'},
            'fuck around': {'severity': 'critical', 'category': 'profanity'},
            'fuckaround': {'severity': 'critical', 'category': 'profanity'},
            'fuck around': {'severity': 'critical', 'category': 'profanity'},
            'fuckoff': {'severity': 'critical', 'category': 'profanity'},
            'fuck off': {'severity': 'critical', 'category': 'profanity'},
            'fuckyou': {'severity': 'critical', 'category': 'profanity'},
            'fuck you': {'severity': 'critical', 'category': 'profanity'},
            'fuckyourself': {'severity': 'critical', 'category': 'profanity'},
            'fuck yourself': {'severity': 'critical', 'category': 'profanity'},
            'fuckherself': {'severity': 'critical', 'category': 'profanity'},
            'fuck herself': {'severity': 'critical', 'category': 'profanity'},
            'fuckhimself': {'severity': 'critical', 'category': 'profanity'},
            'fuck himself': {'severity': 'critical', 'category': 'profanity'},
            'fuckthemselves': {'severity': 'critical', 'category': 'profanity'},
            'fuck themselves': {'severity': 'critical', 'category': 'profanity'},
            'fuckit': {'severity': 'critical', 'category': 'profanity'},
            'fuck it': {'severity': 'critical', 'category': 'profanity'},
            'fuckthat': {'severity': 'critical', 'category': 'profanity'},
            'fuck that': {'severity': 'critical', 'category': 'profanity'},
            'fuckthis': {'severity': 'critical', 'category': 'profanity'},
            'fuck this': {'severity': 'critical', 'category': 'profanity'},
            'fuckall': {'severity': 'critical', 'category': 'profanity'},
            'fuck all': {'severity': 'critical', 'category': 'profanity'},
            'fuckeverything': {'severity': 'critical', 'category': 'profanity'},
            'fuck everything': {'severity': 'critical', 'category': 'profanity'},
            'fuckeveryone': {'severity': 'critical', 'category': 'profanity'},
            'fuck everyone': {'severity': 'critical', 'category': 'profanity'},
            'fuckeverybody': {'severity': 'critical', 'category': 'profanity'},
            'fuck everybody': {'severity': 'critical', 'category': 'profanity'},
            'fucknobody': {'severity': 'critical', 'category': 'profanity'},
            'fuck nobody': {'severity': 'critical', 'category': 'profanity'},
            'fuckanyone': {'severity': 'critical', 'category': 'profanity'},
            'fuck anyone': {'severity': 'critical', 'category': 'profanity'},
            'fuckanybody': {'severity': 'critical', 'category': 'profanity'},
            'fuck anybody': {'severity': 'critical', 'category': 'profanity'},
            'fucksomeone': {'severity': 'critical', 'category': 'profanity'},
            'fuck someone': {'severity': 'critical', 'category': 'profanity'},
            'fucksomebody': {'severity': 'critical', 'category': 'profanity'},
            'fuck somebody': {'severity': 'critical', 'category': 'profanity'},
            'fuckme': {'severity': 'critical', 'category': 'profanity'},
            'fuck me': {'severity': 'critical', 'category': 'profanity'},
            'fuckhim': {'severity': 'critical', 'category': 'profanity'},
            'fuck him': {'severity': 'critical', 'category': 'profanity'},
            'fuckher': {'severity': 'critical', 'category': 'profanity'},
            'fuck her': {'severity': 'critical', 'category': 'profanity'},
            'fuckthem': {'severity': 'critical', 'category': 'profanity'},
            'fuck them': {'severity': 'critical', 'category': 'profanity'},
            'fuckus': {'severity': 'critical', 'category': 'profanity'},
            'fuck us': {'severity': 'critical', 'category': 'profanity'},
            'fuckyour': {'severity': 'critical', 'category': 'profanity'},
            'fuck your': {'severity': 'critical', 'category': 'profanity'},
            'fuckmy': {'severity': 'critical', 'category': 'profanity'},
            'fuck my': {'severity': 'critical', 'category': 'profanity'},
            'fuckhis': {'severity': 'critical', 'category': 'profanity'},
            'fuck his': {'severity': 'critical', 'category': 'profanity'},
            'fuckher': {'severity': 'critical', 'category': 'profanity'},
            'fuck her': {'severity': 'critical', 'category': 'profanity'},
            'fucktheir': {'severity': 'critical', 'category': 'profanity'},
            'fuck their': {'severity': 'critical', 'category': 'profanity'},
            'fuckour': {'severity': 'critical', 'category': 'profanity'},
            'fuck our': {'severity': 'critical', 'category': 'profanity'},
            'fuckyour': {'severity': 'critical', 'category': 'profanity'},
            'fuck your': {'severity': 'critical', 'category': 'profanity'},
            'fuckmy': {'severity': 'critical', 'category': 'profanity'},
            'fuck my': {'severity': 'critical', 'category': 'profanity'},
            'fuckhis': {'severity': 'critical', 'category': 'profanity'},
            'fuck his': {'severity': 'critical', 'category': 'profanity'},
            'fuckher': {'severity': 'critical', 'category': 'profanity'},
            'fuck her': {'severity': 'critical', 'category': 'profanity'},
            'fucktheir': {'severity': 'critical', 'category': 'profanity'},
            'fuck their': {'severity': 'critical', 'category': 'profanity'},
            'fuckour': {'severity': 'critical', 'category': 'profanity'},
            'fuck our': {'severity': 'critical', 'category': 'profanity'},
            
            # More insults
            'piece of shit': {'severity': 'critical', 'category': 'insult'},
            'piece of crap': {'severity': 'high', 'category': 'insult'},
            'piece of garbage': {'severity': 'high', 'category': 'insult'},
            'piece of trash': {'severity': 'high', 'category': 'insult'},
            'piece of junk': {'severity': 'medium', 'category': 'insult'},
            'piece of work': {'severity': 'medium', 'category': 'insult'},
            'piece of': {'severity': 'medium', 'category': 'insult'},
            'worthless piece of': {'severity': 'critical', 'category': 'insult'},
            'useless piece of': {'severity': 'high', 'category': 'insult'},
            'stupid piece of': {'severity': 'high', 'category': 'insult'},
            'dumb piece of': {'severity': 'high', 'category': 'insult'},
            'idiot piece of': {'severity': 'high', 'category': 'insult'},
            'moron piece of': {'severity': 'high', 'category': 'insult'},
            'retard piece of': {'severity': 'critical', 'category': 'insult'},
            'crazy piece of': {'severity': 'medium', 'category': 'insult'},
            'psycho piece of': {'severity': 'high', 'category': 'insult'},
            'nuts piece of': {'severity': 'medium', 'category': 'insult'},
            'freak piece of': {'severity': 'medium', 'category': 'insult'},
            'loser piece of': {'severity': 'medium', 'category': 'insult'},
            'jerk piece of': {'severity': 'medium', 'category': 'insult'},
            'ass piece of': {'severity': 'high', 'category': 'insult'},
            'dick piece of': {'severity': 'critical', 'category': 'insult'},
            'cock piece of': {'severity': 'critical', 'category': 'insult'},
            'bitch piece of': {'severity': 'critical', 'category': 'insult'},
            'bastard piece of': {'severity': 'high', 'category': 'insult'},
            'hell piece of': {'severity': 'medium', 'category': 'insult'},
            'damn piece of': {'severity': 'medium', 'category': 'insult'},
            'shit piece of': {'severity': 'critical', 'category': 'insult'},
            'crap piece of': {'severity': 'high', 'category': 'insult'},
            'piss piece of': {'severity': 'high', 'category': 'insult'},
            'fuck piece of': {'severity': 'critical', 'category': 'insult'},
            'fucking piece of': {'severity': 'critical', 'category': 'insult'},
            'fucked piece of': {'severity': 'critical', 'category': 'insult'},
            'fucker piece of': {'severity': 'critical', 'category': 'insult'},
            'fuckers piece of': {'severity': 'critical', 'category': 'insult'},
            
            # More threats
            'i will hit you hard': {'severity': 'critical', 'category': 'threat'},
            'i will hit you very hard': {'severity': 'critical', 'category': 'threat'},
            'i will hit you really hard': {'severity': 'critical', 'category': 'threat'},
            'i will hit you so hard': {'severity': 'critical', 'category': 'threat'},
            'i will hit you hard now': {'severity': 'critical', 'category': 'threat'},
            'i will hit you hard right now': {'severity': 'critical', 'category': 'threat'},
            'i will hit you hard immediately': {'severity': 'critical', 'category': 'threat'},
            'i will hit you hard soon': {'severity': 'critical', 'category': 'threat'},
            'i will hit you hard later': {'severity': 'critical', 'category': 'threat'},
            'i will hit you hard if': {'severity': 'critical', 'category': 'threat'},
            'i will hit you hard unless': {'severity': 'critical', 'category': 'threat'},
            'i will hit you hard until': {'severity': 'critical', 'category': 'threat'},
            'i will hit you hard when': {'severity': 'critical', 'category': 'threat'},
            'i will hit you hard where': {'severity': 'critical', 'category': 'threat'},
            'i will hit you hard what': {'severity': 'critical', 'category': 'threat'},
            'i will hit you hard why': {'severity': 'critical', 'category': 'threat'},
            'i will hit you hard how': {'severity': 'critical', 'category': 'threat'},
            'i will hit you hard who': {'severity': 'critical', 'category': 'threat'},
            'i will hit you hard which': {'severity': 'critical', 'category': 'threat'},
            'i will hit you hard whose': {'severity': 'critical', 'category': 'threat'},
            'i will hit you hard whom': {'severity': 'critical', 'category': 'threat'},
            'i will hit you hard whatever': {'severity': 'critical', 'category': 'threat'},
            'i will hit you hard whenever': {'severity': 'critical', 'category': 'threat'},
            'i will hit you hard wherever': {'severity': 'critical', 'category': 'threat'},
            'i will hit you hard however': {'severity': 'critical', 'category': 'threat'},
            'i will hit you hard whoever': {'severity': 'critical', 'category': 'threat'},
            'i will hit you hard whichever': {'severity': 'critical', 'category': 'threat'},
            'i will hit you hard whomever': {'severity': 'critical', 'category': 'threat'},
            'i will hit you hard whatsoever': {'severity': 'critical', 'category': 'threat'},
            'i will hit you hard whatsoever': {'severity': 'critical', 'category': 'threat'},
            'i will hit you hard no matter': {'severity': 'critical', 'category': 'threat'},
            'i will hit you hard regardless': {'severity': 'critical', 'category': 'threat'},
            'i will hit you hard anyway': {'severity': 'critical', 'category': 'threat'},
            'i will hit you hard anyhow': {'severity': 'critical', 'category': 'threat'},
            'i will hit you hard anyways': {'severity': 'critical', 'category': 'threat'},
            'i will hit you hard any way': {'severity': 'critical', 'category': 'threat'},
            'i will hit you hard any how': {'severity': 'critical', 'category': 'threat'},
            'i will hit you hard any ways': {'severity': 'critical', 'category': 'threat'},
            'i will hit you hard no matter what': {'severity': 'critical', 'category': 'threat'},
            'i will hit you hard no matter how': {'severity': 'critical', 'category': 'threat'},
            'i will hit you hard no matter when': {'severity': 'critical', 'category': 'threat'},
            'i will hit you hard no matter where': {'severity': 'critical', 'category': 'threat'},
            'i will hit you hard no matter who': {'severity': 'critical', 'category': 'threat'},
            'i will hit you hard no matter which': {'severity': 'critical', 'category': 'threat'},
            'i will hit you hard no matter whose': {'severity': 'critical', 'category': 'threat'},
            'i will hit you hard no matter whom': {'severity': 'critical', 'category': 'threat'},
            'i will hit you hard regardless of': {'severity': 'critical', 'category': 'threat'},
            'i will hit you hard in spite of': {'severity': 'critical', 'category': 'threat'},
            'i will hit you hard despite': {'severity': 'critical', 'category': 'threat'},
            'i will hit you hard even if': {'severity': 'critical', 'category': 'threat'},
            'i will hit you hard even though': {'severity': 'critical', 'category': 'threat'},
            'i will hit you hard even when': {'severity': 'critical', 'category': 'threat'},
            'i will hit you hard even where': {'severity': 'critical', 'category': 'threat'},
            'i will hit you hard even who': {'severity': 'critical', 'category': 'threat'},
            'i will hit you hard even which': {'severity': 'critical', 'category': 'threat'},
            'i will hit you hard even whose': {'severity': 'critical', 'category': 'threat'},
            'i will hit you hard even whom': {'severity': 'critical', 'category': 'threat'},
            'i will hit you hard as long as': {'severity': 'critical', 'category': 'threat'},
            'i will hit you hard as soon as': {'severity': 'critical', 'category': 'threat'},
            'i will hit you hard as far as': {'severity': 'critical', 'category': 'threat'},
            'i will hit you hard as much as': {'severity': 'critical', 'category': 'threat'},
            'i will hit you hard as many as': {'severity': 'critical', 'category': 'threat'},
            'i will hit you hard as few as': {'severity': 'critical', 'category': 'threat'},
            'i will hit you hard as little as': {'severity': 'critical', 'category': 'threat'},
            'i will hit you hard as well as': {'severity': 'critical', 'category': 'threat'},
            'i will hit you hard as good as': {'severity': 'critical', 'category': 'threat'},
            'i will hit you hard as bad as': {'severity': 'critical', 'category': 'threat'},
            'i will hit you hard as big as': {'severity': 'critical', 'category': 'threat'},
            'i will hit you hard as small as': {'severity': 'critical', 'category': 'threat'},
            'i will hit you hard as tall as': {'severity': 'critical', 'category': 'threat'},
            'i will hit you hard as short as': {'severity': 'critical', 'category': 'threat'},
            'i will hit you hard as wide as': {'severity': 'critical', 'category': 'threat'},
            'i will hit you hard as narrow as': {'severity': 'critical', 'category': 'threat'},
            'i will hit you hard as deep as': {'severity': 'critical', 'category': 'threat'},
            'i will hit you hard as shallow as': {'severity': 'critical', 'category': 'threat'},
            'i will hit you hard as high as': {'severity': 'critical', 'category': 'threat'},
            'i will hit you hard as low as': {'severity': 'critical', 'category': 'threat'},
            'i will hit you hard as fast as': {'severity': 'critical', 'category': 'threat'},
            'i will hit you hard as slow as': {'severity': 'critical', 'category': 'threat'},
            'i will hit you hard as quick as': {'severity': 'critical', 'category': 'threat'},
            'i will hit you hard as quick as': {'severity': 'critical', 'category': 'threat'},
        }
    
    def _load_context_indicators(self) -> List[str]:
        """Load context indicators that make words more concerning"""
        return [
            'אל', 'ת', 'אתה', 'את', 'אתה', 'אני', 'לך', 'לו',
            'you', 'i', 'don\'t', 'won\'t', 'can\'t', 'stop'
        ]
    
    def detect_inappropriate_language(self, transcription: str, 
                                     segments: List[Dict] = None, 
                                     language: str = 'en') -> List[InappropriateWord]:
        """
        Detect inappropriate language in transcription
        
        Args:
            transcription: Full transcription text
            segments: List of transcription segments with timestamps
            language: Language code ('en' for English, 'he' for Hebrew)
            
        Returns:
            List of detected inappropriate words with details
        """
        detected_words = []
        text_lower = transcription.lower()
        
        # Check words based on selected language (but check both to be safe)
        if language == 'he':
            # Check Hebrew words first (primary)
            checked_count = 0
            found_count = 0
            for word, info in self.inappropriate_words_hebrew.items():
                word_lower = word.lower()
                matches = self._find_word_occurrences(text_lower, word_lower, 'hebrew')
                checked_count += 1
                
                if matches:
                    found_count += 1
                    print(f"    [FOUND] '{word}' in transcription ({len(matches)} matches)", flush=True)
                    for i, match in enumerate(matches[:3]):  # Show first 3 matches
                        print(f"      Match {i+1}: ...{match['context']}...", flush=True)
                
                for match in matches:
                    timestamp = self._find_timestamp_for_text(match['position'], segments) if segments else 0.0
                    
                    detected_words.append(InappropriateWord(
                        word=word,
                        timestamp=timestamp,
                        severity=info['severity'],
                        context=match['context'],
                        language='hebrew'
                    ))
            
            print(f"  [DEBUG] Checked {checked_count} Hebrew words, found {found_count} matches", flush=True)
        
        # Also check English words (mixed language scenarios)
        checked_count = 0
        found_count = 0
        for word, info in self.inappropriate_words_english.items():
            word_lower = word.lower()
            matches = self._find_word_occurrences(text_lower, word_lower, 'english')
            checked_count += 1
            
            if matches:
                found_count += 1
                print(f"    [FOUND] '{word}' in transcription ({len(matches)} matches)", flush=True)
                for i, match in enumerate(matches[:3]):  # Show first 3 matches
                    print(f"      Match {i+1}: ...{match['context']}...", flush=True)
            
            for match in matches:
                timestamp = self._find_timestamp_for_text(match['position'], segments) if segments else 0.0
                
                detected_words.append(InappropriateWord(
                    word=word,
                    timestamp=timestamp,
                    severity=info['severity'],
                    context=match['context'],
                    language='english'
                ))
        
        print(f"  [DEBUG] Checked {checked_count} English words, found {found_count} matches", flush=True)
        
        # Sort by severity and timestamp
        severity_order = {'critical': 4, 'high': 3, 'medium': 2, 'low': 1}
        detected_words.sort(key=lambda x: (severity_order.get(x.severity, 0), x.timestamp), reverse=True)
        
        return detected_words
    
    def _find_word_occurrences(self, text: str, word: str, language: str) -> List[Dict]:
        """
        Find all occurrences of a word in text with context
        Uses multiple search strategies for better detection
        """
        occurrences = []
        word_lower = word.lower()
        text_lower = text.lower()
        
        # Strategy 1: Simple substring search (most aggressive - catches everything)
        # This is the primary method - we want to catch all occurrences
        start_pos = 0
        while True:
            pos = text_lower.find(word_lower, start_pos)
            if pos == -1:
                break
            
            # For Hebrew, be very lenient - just check basic boundaries
            # For English, also be lenient but check word boundaries
            is_valid = True
            if language == 'hebrew':
                # Hebrew: very lenient - just check if not in middle of another word
                # Allow if at start/end or surrounded by spaces/punctuation
                if pos > 0 and pos + len(word) < len(text_lower):
                    char_before = text_lower[pos - 1]
                    char_after = text_lower[pos + len(word)]
                    # Only invalid if both are Hebrew letters (might be part of another word)
                    if char_before.isalpha() and char_after.isalpha() and \
                       ('\u0590' <= char_before <= '\u05FF' or '\u0590' <= char_after <= '\u05FF'):
                        # Might be part of word, but still check if it's a valid match
                        # Be lenient - if word is long enough, it's probably valid
                        if len(word) < 3:
                            is_valid = False
            else:
                # English: check word boundaries more strictly
                if pos > 0 and pos + len(word) < len(text_lower):
                    char_before = text_lower[pos - 1]
                    char_after = text_lower[pos + len(word)]
                    # Invalid if both are alphanumeric (part of another word)
                    if char_before.isalnum() and char_after.isalnum():
                        is_valid = False
            
            if is_valid:
                context_start = max(0, pos - 50)
                context_end = min(len(text_lower), pos + len(word) + 50)
                context = text[context_start:context_end].strip()
                
                # Avoid duplicates (within 2 characters)
                if not any(abs(occ['position'] - pos) < 2 for occ in occurrences):
                    occurrences.append({
                        'position': pos,
                        'context': context
                    })
            
            start_pos = pos + 1
        
        # Strategy 2: Exact word match with word boundaries (for English as backup)
        if language == 'english' and len(occurrences) == 0:
            # If substring didn't find anything, try word boundaries
            pattern = r'\b' + re.escape(word) + r'\b'
            for match in re.finditer(pattern, text, re.IGNORECASE):
                start = match.start()
                end = match.end()
                context_start = max(0, start - 50)
                context_end = min(len(text), end + 50)
                context = text[context_start:context_end].strip()
                
                # Avoid duplicates
                if not any(abs(occ['position'] - start) < 2 for occ in occurrences):
                    occurrences.append({
                        'position': start,
                        'context': context
                    })
        
        # Strategy 3: Remove Hebrew diacritics (nikud) and search again
        if language == 'hebrew':
            # Remove common Hebrew diacritics for fuzzy matching
            text_no_nikud = self._remove_hebrew_nikud(text_lower)
            word_no_nikud = self._remove_hebrew_nikud(word_lower)
            
            start_pos = 0
            while True:
                pos = text_no_nikud.find(word_no_nikud, start_pos)
                if pos == -1:
                    break
                
                # Check word boundaries
                if (pos == 0 or text_no_nikud[pos-1] in ' \n\t.,!?;:') and \
                   (pos + len(word_no_nikud) >= len(text_no_nikud) or text_no_nikud[pos + len(word_no_nikud)] in ' \n\t.,!?;:'):
                    context_start = max(0, pos - 30)
                    context_end = min(len(text_no_nikud), pos + len(word_no_nikud) + 30)
                    context = text[context_start:context_end].strip()
                    
                    # Avoid duplicates
                    if not any(abs(occ['position'] - pos) < 3 for occ in occurrences):
                        occurrences.append({
                            'position': pos,
                            'context': context
                        })
                
                start_pos = pos + 1
        
        return occurrences
    
    def _remove_hebrew_nikud(self, text: str) -> str:
        """
        Remove Hebrew diacritics (nikud) from text for fuzzy matching
        """
        # Hebrew diacritics Unicode range
        nikud = '\u0591-\u05C7'
        return re.sub(f'[{nikud}]', '', text)
    
    def _find_timestamp_for_text(self, text_position: int, segments: List[Dict]) -> float:
        """
        Find timestamp for a text position using segments
        """
        if not segments:
            return 0.0
        
        # Find which segment contains this position
        # This is approximate - we'd need character-level mapping for exact match
        char_count = 0
        for segment in segments:
            segment_text = segment.get('text', '')
            segment_length = len(segment_text)
            
            if char_count <= text_position < char_count + segment_length:
                return segment.get('start', 0.0)
            
            char_count += segment_length
        
        # If not found, return last segment end time
        if segments:
            return segments[-1].get('end', 0.0)
        
        return 0.0
    
    def analyze_with_whisper(self, audio_file: str, language: str = 'en') -> Dict:
        """
        Analyze audio file using Whisper for transcription and inappropriate language detection
        
        Args:
            audio_file: Path to audio file
            language: Language code ('en' for English, 'he' for Hebrew)
        """
        try:
            import whisper
            import os
            
            print("  Loading Whisper for transcription...", flush=True)
            model = whisper.load_model("base")
            
            # Map language code to Whisper language code
            whisper_lang = 'he' if language == 'he' else 'en'
            print(f"  Transcribing audio ({'Hebrew' if language == 'he' else 'English'})...", flush=True)
            
            # Ensure absolute path
            audio_path = os.path.abspath(audio_file)
            
            # Check if file exists
            if not os.path.exists(audio_path):
                print(f"  [ERROR] Audio file not found: {audio_path}", flush=True)
                return {
                    'status': 'error',
                    'error': f'Audio file not found: {audio_path}',
                    'detected_inappropriate_words': 0
                }
            
            print(f"  [DEBUG] Audio file path: {audio_path}", flush=True)
            print(f"  [DEBUG] File exists: {os.path.exists(audio_path)}", flush=True)
            
            # Check if ffmpeg is available
            import subprocess
            ffmpeg_available = False
            try:
                result_check = subprocess.run(['ffmpeg', '-version'], 
                                            capture_output=True, 
                                            timeout=2)
                ffmpeg_available = result_check.returncode == 0
            except (FileNotFoundError, subprocess.TimeoutExpired):
                pass
            
            if not ffmpeg_available:
                print("\n" + "="*80, flush=True)
                print("  [WARNING] ⚠️ FFmpeg not found in PATH!", flush=True)
                print("  [WARNING] Whisper transcription will fail without FFmpeg.", flush=True)
                print("="*80, flush=True)
                print("  [INFO] 📖 To install FFmpeg:", flush=True)
                print("", flush=True)
                print("  Windows (Option 1 - Chocolatey):", flush=True)
                print("    1. Install Chocolatey (if not installed):", flush=True)
                print("       Run PowerShell as Administrator:", flush=True)
                print("       Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))", flush=True)
                print("    2. Install FFmpeg: choco install ffmpeg", flush=True)
                print("", flush=True)
                print("  Windows (Option 2 - Manual):", flush=True)
                print("    1. Download from: https://ffmpeg.org/download.html", flush=True)
                print("    2. Extract to C:\\ffmpeg", flush=True)
                print("    3. Add C:\\ffmpeg\\bin to PATH", flush=True)
                print("    4. Restart terminal", flush=True)
                print("", flush=True)
                print("  Windows (Option 3 - winget):", flush=True)
                print("    winget install ffmpeg", flush=True)
                print("", flush=True)
                print("  📖 For detailed instructions, see: INSTALL_FFMPEG.md", flush=True)
                print("="*80 + "\n", flush=True)
            
            try:
                result = model.transcribe(audio_path, language=whisper_lang)
            except FileNotFoundError as e:
                # This usually means ffmpeg is missing
                print(f"  [WARNING] Whisper transcription failed (likely missing ffmpeg): {e}", flush=True)
                print("  [INFO] Trying alternative method with librosa and numpy array...", flush=True)
                
                # Try alternative: load with librosa and pass numpy array directly to Whisper
                try:
                    import librosa
                    import numpy as np
                    
                    print("  [INFO] Loading audio with librosa...", flush=True)
                    audio_data, sr = librosa.load(audio_path, sr=16000)
                    
                    # Whisper can accept numpy arrays directly (in some versions)
                    # But if that doesn't work, we'll try saving to WAV with soundfile
                    print("  [INFO] Attempting transcription with numpy array...", flush=True)
                    try:
                        # Try passing numpy array directly (works in newer Whisper versions)
                        result = model.transcribe(audio_data, language=whisper_lang)
                        print("  [SUCCESS] Transcription with numpy array worked!", flush=True)
                    except (TypeError, AttributeError) as e3:
                        # If numpy array doesn't work, try saving to WAV
                        print(f"  [INFO] Numpy array method failed ({e3}), trying WAV file...", flush=True)
                        import soundfile as sf
                        import tempfile
                        
                        # Save to temporary WAV file
                        temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                        temp_wav_path = temp_wav.name
                        temp_wav.close()
                        
                        print(f"  [INFO] Saving to temporary WAV: {temp_wav_path}", flush=True)
                        sf.write(temp_wav_path, audio_data, sr)
                        
                        # Verify file was created
                        if not os.path.exists(temp_wav_path):
                            raise FileNotFoundError(f"Temporary WAV file was not created: {temp_wav_path}")
                        
                        print(f"  [INFO] Temporary WAV file created: {os.path.getsize(temp_wav_path)} bytes", flush=True)
                        print("  [INFO] Transcribing with Whisper using temporary WAV...", flush=True)
                        
                        # Try transcribing the WAV file
                        result = model.transcribe(temp_wav_path, language=whisper_lang)
                        
                        # Clean up
                        try:
                            os.unlink(temp_wav_path)
                        except:
                            pass
                        
                except ImportError as e_import:
                    print(f"  [ERROR] Required libraries not available: {e_import}", flush=True)
                    print("  [INFO] Please install: pip install librosa soundfile", flush=True)
                    return {
                        'status': 'error',
                        'error': f'Whisper transcription failed: {e}. Required libraries not available: {e_import}',
                        'detected_inappropriate_words': 0,
                        'suggestion': 'Install ffmpeg or run: pip install librosa soundfile'
                    }
                except Exception as e2:
                    print(f"  [ERROR] Alternative method also failed: {e2}", flush=True)
                    import traceback
                    print(f"  [ERROR] Traceback: {traceback.format_exc()}", flush=True)
                    
                    # Provide helpful installation instructions
                    print("\n  [SOLUTION] To fix this issue:", flush=True)
                    print("    1. Install ffmpeg:", flush=True)
                    print("       Windows: choco install ffmpeg (or download from https://ffmpeg.org/download.html)", flush=True)
                    print("       Or add ffmpeg to PATH", flush=True)
                    print("    2. Alternatively, install pydub: pip install pydub", flush=True)
                    print("    3. Make sure librosa and soundfile are installed: pip install librosa soundfile", flush=True)
                    
                    return {
                        'status': 'error',
                        'error': f'Whisper transcription failed: {e}. Alternative method also failed: {e2}',
                        'detected_inappropriate_words': 0,
                        'transcription': '',
                        'has_inappropriate_language': False,
                        'suggestion': 'Install ffmpeg: https://ffmpeg.org/download.html or use: choco install ffmpeg (Windows)',
                        'note': 'Whisper requires ffmpeg to process audio files. Please install ffmpeg and add it to your PATH.',
                        'workaround': 'The analysis will continue without transcription. Other analysis steps (emotions, crying, violence, neglect) will still work.'
                    }
            except Exception as e:
                print(f"  [ERROR] Whisper transcription error: {e}", flush=True)
                return {
                    'status': 'error',
                    'error': str(e),
                    'detected_inappropriate_words': 0
                }
            
            transcription = result["text"]
            segments = result["segments"]
            
            print(f"  Transcription completed: {len(transcription)} characters", flush=True)
            # Print to both stdout and stderr to ensure visibility
            debug_msg = "=" * 80 + "\n  FULL TRANSCRIPTION:\n  " + transcription + "\n" + "=" * 80
            print(debug_msg, flush=True)
            sys.stderr.write(debug_msg + "\n")
            sys.stderr.flush()
            
            # Detect inappropriate language
            print("\n  [DEBUG] Starting inappropriate language detection...", flush=True)
            print(f"  [DEBUG] Checking {len(self.inappropriate_words_hebrew)} Hebrew words and {len(self.inappropriate_words_english)} English words...", flush=True)
            print(f"  [DEBUG] Transcription length: {len(transcription)} characters", flush=True)
            print(f"  [DEBUG] Transcription preview (first 100 chars): {transcription[:100]}", flush=True)
            
            detected_words = self.detect_inappropriate_language(transcription, segments, language=language)
            
            print(f"\n  [RESULT] Found {len(detected_words)} inappropriate words/phrases", flush=True)
            
            # Debug: Show what was found
            if len(detected_words) > 0:
                print("  [DETECTED WORDS]:", flush=True)
                for word_obj in detected_words[:20]:  # Show first 20
                    print(f"    - '{word_obj.word}' ({word_obj.severity}, {word_obj.language}) at {word_obj.timestamp:.1f}s", flush=True)
                    print(f"      Context: ...{word_obj.context}...", flush=True)
            else:
                print("  [WARNING] No words detected!", flush=True)
                print("  [DEBUG] Checking if common words appear in transcription...", flush=True)
                # Try to find any partial matches - check more words
                transcription_lower = transcription.lower()
                found_any = False
                
                # Check Hebrew words
                print(f"  [DEBUG] Checking Hebrew words (first 50 of {len(self.inappropriate_words_hebrew)})...", flush=True)
                for word in list(self.inappropriate_words_hebrew.keys())[:50]:  # Check first 50 words
                    word_lower = word.lower()
                    if word_lower in transcription_lower:
                        print(f"    [FOUND!] '{word}' appears in transcription!", flush=True)
                        # Show where it appears
                        pos = transcription_lower.find(word_lower)
                        context_start = max(0, pos - 30)
                        context_end = min(len(transcription_lower), pos + len(word_lower) + 30)
                        print(f"      Context: ...{transcription[context_start:context_end]}...", flush=True)
                        found_any = True
                
                # Check English words
                print(f"  [DEBUG] Checking English words (first 50 of {len(self.inappropriate_words_english)})...", flush=True)
                for word in list(self.inappropriate_words_english.keys())[:50]:  # Check first 50 words
                    word_lower = word.lower()
                    if word_lower in transcription_lower:
                        print(f"    [FOUND!] '{word}' appears in transcription!", flush=True)
                        # Show where it appears
                        pos = transcription_lower.find(word_lower)
                        context_start = max(0, pos - 30)
                        context_end = min(len(transcription_lower), pos + len(word_lower) + 30)
                        print(f"      Context: ...{transcription[context_start:context_end]}...", flush=True)
                        found_any = True
                
                if not found_any:
                    print("  [DEBUG] No common inappropriate words found in transcription", flush=True)
                    print("  [DEBUG] This might mean:", flush=True)
                    print("    1. Whisper didn't transcribe the words correctly", flush=True)
                    print("    2. The words are written differently in the transcription", flush=True)
                    print("    3. The words need to be added to the word list", flush=True)
                    print(f"  [DEBUG] Transcription sample (first 200 chars): {transcription[:200]}", flush=True)
            
            # Group by severity
            by_severity = {
                'critical': [],
                'high': [],
                'medium': [],
                'low': []
            }
            
            for word_obj in detected_words:
                by_severity[word_obj.severity].append({
                    'word': word_obj.word,
                    'timestamp': word_obj.timestamp,
                    'language': word_obj.language,
                    'context': word_obj.context
                })
            
            return {
                'transcription': transcription,
                'segments': segments,
                'detected_inappropriate_words': len(detected_words),
                'words_by_severity': by_severity,
                'total_words': len(detected_words),
                'has_inappropriate_language': len(detected_words) > 0,
                'detected_words': [
                    {
                        'word': w.word,
                        'timestamp': w.timestamp,
                        'severity': w.severity,
                        'language': w.language,
                        'context': w.context
                    }
                    for w in detected_words
                ]
            }
            
        except ImportError as e:
            error_msg = f'Whisper not installed. Install with: pip install openai-whisper. Error: {e}'
            print(f"  [ERROR] {error_msg}", flush=True)
            return {
                'status': 'whisper_not_installed',
                'error': error_msg,
                'detected_inappropriate_words': 0,
                'transcription': '',
                'has_inappropriate_language': False
            }
        except Exception as e:
            import traceback
            error_msg = str(e)
            error_trace = traceback.format_exc()
            print(f"  [ERROR] Unexpected error in analyze_with_whisper: {error_msg}", flush=True)
            print(f"  [ERROR] Traceback:\n{error_trace}", flush=True)
            return {
                'status': 'error',
                'error': error_msg,
                'detected_inappropriate_words': 0,
                'transcription': '',
                'has_inappropriate_language': False
            }

if __name__ == "__main__":
    detector = InappropriateLanguageDetector()
    print("Inappropriate Language Detector initialized successfully")
    print(f"Hebrew words: {len(detector.inappropriate_words_hebrew)}")
    print(f"English words: {len(detector.inappropriate_words_english)}")
