"""
Report Generation Module for Kindergarten Recording Analyzer
מודול יצירת דוחות למערכת ניתוח הקלטות גן ילדים
"""

import json
import csv
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import os
import librosa
import soundfile as sf
import numpy as np

class ReportGenerator:
    def __init__(self, output_dir: str = "reports"):
        """
        Initialize Report Generator
        אתחול מחולל הדוחות
        
        Args:
            output_dir: Directory to save reports
        """
        self.output_dir = output_dir
        self._ensure_output_directory()
        
        # Hebrew translations for report elements
        # תרגומים לעברית עבור אלמנטי הדוח
        self.translations = {
            'emotions': {
                'anger': 'כעס',
                'stress': 'לחץ',
                'calm': 'רגוע',
                'aggression': 'אגרסיה'
            },
            'violence_types': {
                'shouting': 'צעקות',
                'aggressive_tone': 'טון אגרסיבי',
                'threatening': 'איום',
                'potential_physical_violence': 'אלימות פיזית פוטנציאלית'
            },
            'severity_levels': {
                'low': 'נמוכה',
                'medium': 'בינונית',
                'high': 'גבוהה',
                'critical': 'קריטית',
                'none': 'אין'
            },
            'cry_intensity': {
                'low': 'חלש',
                'medium': 'בינוני',
                'high': 'חזק'
            },
            'response_quality': {
                'poor': 'גרועה',
                'adequate': 'מספקת',
                'good': 'טובה',
                'none': 'אין'
            }
        }
    
    def _ensure_output_directory(self):
        """Ensure output directory exists"""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    
    def extract_incident_audio_clips(self, analysis_results: Dict, 
                                   audio_file_path: str) -> List[Dict]:
        """
        Extract audio clips for each incident so parents can listen
        חילוץ קטעי אודיו לכל אירוע כדי שהורים יוכלו לשמוע
        """
        print("יוצר קטעי אודיו לאירועים...")
        print("Creating audio clips for incidents...")
        
        clips = []
        
        try:
            # Load the original audio
            audio, sr = librosa.load(audio_file_path, sr=22050)
            
            # Extract clips for concerning emotions
            for emotion in analysis_results.get('concerning_emotions', []):
                clip_info = self._extract_audio_clip(
                    audio, sr, emotion['start_time'], emotion['end_time'],
                    'emotion', emotion['detected_emotion'], emotion['severity']
                )
                if clip_info:
                    clips.append(clip_info)
            
            # Extract clips for violence incidents
            for violence in analysis_results.get('violence_segments', []):
                clip_info = self._extract_audio_clip(
                    audio, sr, violence['start_time'], violence['end_time'],
                    'violence', ', '.join(violence.get('violence_types', [])), violence.get('adjusted_severity', 'unknown')
                )
                if clip_info:
                    clips.append(clip_info)
            
            # Extract clips for cry incidents
            for cry in analysis_results.get('cry_with_responses', []):
                clip_info = self._extract_audio_clip(
                    audio, sr, cry['start_time'], cry['end_time'],
                    'cry', f"cry_{cry.get('intensity', 'medium')}", 'medium'
                )
                if clip_info:
                    clips.append(clip_info)
            
            # Extract clips for unanswered cries (neglect)
            neglect_analysis = analysis_results.get('neglect_analysis', {})
            for unanswered_cry in neglect_analysis.get('unanswered_cries', []):
                clip_info = self._extract_audio_clip(
                    audio, sr, unanswered_cry['cry_start_time'], unanswered_cry['cry_end_time'],
                    'neglect', 'unanswered_cry', unanswered_cry.get('neglect_severity', 'medium')
                )
                if clip_info:
                    clips.append(clip_info)
            
            print(f"נוצרו {len(clips)} קטעי אודיו")
            print(f"Created {len(clips)} audio clips")
            
        except Exception as e:
            print(f"שגיאה ביצירת קטעי אודיו: {str(e)}")
            print(f"Error creating audio clips: {str(e)}")
        
        return clips
    
    def _extract_audio_clip(self, audio: np.ndarray, sr: int, 
                          start_time: float, end_time: float,
                          incident_type: str, description: str, severity: str) -> Optional[Dict]:
        """
        Extract a single audio clip for an incident
        חילוץ קטע אודיו יחיד לאירוע
        """
        try:
            # Convert time to samples
            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)
            
            # Ensure we don't go beyond audio bounds
            start_sample = max(0, start_sample)
            end_sample = min(len(audio), end_sample)
            
            if start_sample >= end_sample:
                return None
            
            # Extract the clip
            clip_audio = audio[start_sample:end_sample]
            
            # Create filename
            timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
            clip_filename = f"incident_{timestamp_str}_{incident_type}_{int(start_time)}s-{int(end_time)}s.wav"
            clip_path = os.path.join(self.output_dir, clip_filename)
            
            # Save the clip
            sf.write(clip_path, clip_audio, sr)
            
            return {
                'filename': clip_filename,
                'path': clip_path,
                'start_time': start_time,
                'end_time': end_time,
                'duration': end_time - start_time,
                'incident_type': incident_type,
                'description': description,
                'severity': severity,
                'size_kb': os.path.getsize(clip_path) // 1024
            }
            
        except Exception as e:
            print(f"שגיאה בחילוץ קטע: {str(e)}")
            return None

    def generate_comprehensive_report(self, analysis_results: Dict, 
                                    audio_file_path: str) -> Dict:
        """
        Generate comprehensive analysis report
        יצירת דוח ניתוח מקיף
        
        Args:
            analysis_results: Results from all analysis modules
            audio_file_path: Path to the analyzed audio file
            
        Returns:
            Dictionary containing report data
        """
        timestamp = datetime.now()
        
        report = {
            'metadata': {
                'file_path': audio_file_path,
                'file_name': os.path.basename(audio_file_path),
                'analysis_timestamp': timestamp.isoformat(),
                'analysis_date_hebrew': self._format_hebrew_date(timestamp),
                'audio_duration': analysis_results.get('duration', 0)
            },
            'summary': self._generate_summary(analysis_results),
            'detailed_findings': self._generate_detailed_findings(analysis_results),
            'recommendations': self._generate_recommendations(analysis_results),
            'statistics': self._generate_statistics(analysis_results)
        }
        
        # Extract audio clips for incidents
        audio_clips = self.extract_incident_audio_clips(analysis_results, audio_file_path)
        report['audio_clips'] = audio_clips
        
        # Generate different report formats
        self._save_json_report(report)
        self._save_html_report(report)
        self._save_csv_report(report)
        
        return report
    
    def _generate_summary(self, analysis_results: Dict) -> Dict:
        """
        Generate executive summary
        יצירת סיכום מנהלים
        """
        summary = {
            'overall_assessment': 'normal',
            'total_incidents': 0,
            'critical_incidents': 0,
            'key_findings': [],
            'risk_level': 'low'
        }
        
        # Count incidents
        emotion_incidents = len(analysis_results.get('concerning_emotions', []))
        violence_incidents = len(analysis_results.get('violence_segments', []))
        unanswered_cries = len(analysis_results.get('unanswered_cries', []))
        neglect_incidents = len(analysis_results.get('ignored_distress_episodes', []))
        
        # Count inappropriate language incidents
        inappropriate_lang = analysis_results.get('inappropriate_language', {})
        inappropriate_words = inappropriate_lang.get('detected_inappropriate_words', 0)
        
        summary['total_incidents'] = emotion_incidents + violence_incidents + unanswered_cries + neglect_incidents + inappropriate_words
        
        # Count critical incidents
        critical_count = 0
        
        # Critical emotions
        for emotion in analysis_results.get('concerning_emotions', []):
            if emotion.get('severity') == 'critical':
                critical_count += 1
        
        # Critical violence
        for violence in analysis_results.get('violence_segments', []):
            if violence.get('adjusted_severity') == 'critical':
                critical_count += 1
        
        # Critical neglect
        neglect_analysis = analysis_results.get('neglect_analysis', {})
        if neglect_analysis.get('neglect_severity') == 'critical':
            critical_count += 1
        
        summary['critical_incidents'] = critical_count
        
        # Generate key findings
        key_findings = []
        
        if violence_incidents > 0:
            key_findings.append(f"זוהו {violence_incidents} אירועי אלימות פוטנציאליים")
        
        if unanswered_cries > 0:
            key_findings.append(f"זוהו {unanswered_cries} אירועי בכי ללא תגובה")
        
        if emotion_incidents > 0:
            key_findings.append(f"זוהו {emotion_incidents} אירועי רגשות מדאיגים")
        
        if inappropriate_words > 0:
            # Count by severity
            words_by_severity = inappropriate_lang.get('words_by_severity', {})
            critical_words = len(words_by_severity.get('critical', []))
            high_words = len(words_by_severity.get('high', []))
            
            if critical_words > 0:
                key_findings.append(f"⚠️ זוהו {inappropriate_words} מילים לא הולמות ({critical_words} קריטיות)")
            else:
                key_findings.append(f"זוהו {inappropriate_words} מילים לא הולמות")
        
        neglect_severity = neglect_analysis.get('neglect_severity', 'none')
        if neglect_severity != 'none':
            severity_hebrew = self.translations['severity_levels'].get(neglect_severity, neglect_severity)
            key_findings.append(f"רמת הזנחה: {severity_hebrew}")
        
        summary['key_findings'] = key_findings
        
        # Count critical inappropriate words
        words_by_severity = inappropriate_lang.get('words_by_severity', {})
        critical_inappropriate = len(words_by_severity.get('critical', []))
        high_inappropriate = len(words_by_severity.get('high', []))
        
        # Adjust critical count for inappropriate language
        if critical_inappropriate > 0:
            critical_count += critical_inappropriate
        
        # Determine overall assessment
        if critical_count > 0 or critical_inappropriate > 0:
            summary['overall_assessment'] = 'critical'
            summary['risk_level'] = 'critical'
        elif summary['total_incidents'] > 5 or high_inappropriate > 2:
            summary['overall_assessment'] = 'concerning'
            summary['risk_level'] = 'high'
        elif summary['total_incidents'] > 2:
            summary['overall_assessment'] = 'moderate'
            summary['risk_level'] = 'medium'
        else:
            summary['overall_assessment'] = 'normal'
            summary['risk_level'] = 'low'
        
        return summary
    
    def _generate_detailed_findings(self, analysis_results: Dict) -> Dict:
        """
        Generate detailed findings section
        יצירת סעיף ממצאים מפורטים
        """
        findings = {
            'emotional_analysis': [],
            'violence_analysis': [],
            'cry_analysis': [],
            'neglect_analysis': [],
            'inappropriate_language': []
        }
        
        # Emotional analysis findings
        for emotion in analysis_results.get('concerning_emotions', []):
            emotion_data = {
                'timestamp': f"{emotion['start_time']:.1f}s - {emotion['end_time']:.1f}s",
                'detected_emotion': self.translations['emotions'].get(emotion['detected_emotion'], emotion['detected_emotion']),
                'confidence': f"{emotion['confidence']:.2f}",
                'severity': self.translations['severity_levels'].get(emotion['severity'], emotion['severity']),
                'description': self._get_emotion_description(emotion)
            }
            findings['emotional_analysis'].append(emotion_data)
        
        # Violence analysis findings
        for violence in analysis_results.get('violence_segments', []):
            violence_types_hebrew = [self.translations['violence_types'].get(vt, vt) 
                                   for vt in violence.get('violence_types', [])]
            
            violence_data = {
                'timestamp': f"{violence['start_time']:.1f}s - {violence['end_time']:.1f}s",
                'violence_types': ', '.join(violence_types_hebrew),
                'severity': self.translations['severity_levels'].get(violence.get('adjusted_severity', 'unknown'), 'לא ידוע'),
                'confidence': f"{violence.get('confidence', 0):.2f}",
                'context': violence.get('context', {}).get('overall_assessment', 'לא זמין'),
                'description': self._get_violence_description(violence)
            }
            findings['violence_analysis'].append(violence_data)
        
        # Cry analysis findings
        for cry in analysis_results.get('cry_with_responses', []):
            cry_data = {
                'timestamp': f"{cry['start_time']:.1f}s - {cry['end_time']:.1f}s",
                'duration': f"{cry['duration']:.1f} שניות",
                'intensity': self.translations['cry_intensity'].get(cry.get('intensity', 'medium'), 'בינוני'),
                'response_detected': 'כן' if cry.get('response_detected', False) else 'לא',
                'response_quality': self.translations['response_quality'].get(cry.get('response_quality', 'none'), 'אין'),
                'description': self._get_cry_description(cry)
            }
            findings['cry_analysis'].append(cry_data)
        
        # Advanced analysis findings
        advanced_analysis = analysis_results.get('advanced_analysis', {})
        if advanced_analysis:
            findings['advanced_analysis'] = {
                'whisper_transcription': advanced_analysis.get('whisper_analysis', {}),
                'emotion_detection': advanced_analysis.get('emotion_analysis', {}),
                'combined_insights': advanced_analysis.get('combined_insights', {})
            }
        
        # Neglect analysis findings
        neglect_analysis = analysis_results.get('neglect_analysis', {})
        
        for unanswered_cry in neglect_analysis.get('unanswered_cries', []):
            neglect_data = {
                'timestamp': f"{unanswered_cry['cry_start_time']:.1f}s - {unanswered_cry['cry_end_time']:.1f}s",
                'cry_duration': f"{unanswered_cry['cry_duration']:.1f} שניות",
                'time_without_response': f"{unanswered_cry['time_without_response']:.1f} שניות",
                'neglect_severity': self.translations['severity_levels'].get(unanswered_cry.get('neglect_severity', 'low'), 'נמוכה'),
                'description': f"בכי שנמשך {unanswered_cry['cry_duration']:.1f} שניות ללא תגובת צוות"
            }
            findings['neglect_analysis'].append(neglect_data)
        
        # Inappropriate language findings
        inappropriate_language = analysis_results.get('inappropriate_language', {})
        if inappropriate_language.get('detected_words'):
            for word_data in inappropriate_language['detected_words']:
                lang_display = 'עברית' if word_data['language'] == 'hebrew' else 'אנגלית'
                lang_display_en = 'Hebrew' if word_data['language'] == 'hebrew' else 'English'
                
                inappropriate_data = {
                    'timestamp': f"{word_data['timestamp']:.1f}s",
                    'word': word_data['word'],
                    'language': f"{lang_display} / {lang_display_en}",
                    'severity': self.translations['severity_levels'].get(word_data['severity'], word_data['severity']),
                    'context': word_data.get('context', '')[:100],  # First 100 chars
                    'description': f"מילה לא הולמת: '{word_data['word']}' ({lang_display})"
                }
                findings['inappropriate_language'].append(inappropriate_data)
        
        return findings
    
    def _generate_recommendations(self, analysis_results: Dict) -> List[str]:
        """
        Generate recommendations based on analysis
        יצירת המלצות בהתבסס על הניתוח
        """
        recommendations = []
        
        # Violence recommendations
        violence_segments = analysis_results.get('violence_segments', [])
        if violence_segments:
            recommendations.append("מומלץ להעביר הדרכה לצוות על ניהול קונפליקטים ותקשורת עם ילדים")
            recommendations.append("יש לשקול מעקב נוסף אחר התנהגות הצוות בזמן הקונפליקטים")
        
        # Inappropriate language recommendations
        inappropriate_lang = analysis_results.get('inappropriate_language', {})
        if inappropriate_lang.get('detected_inappropriate_words', 0) > 0:
            recommendations.append("⚠️ זוהו מילים לא הולמות - יש להעביר הדרכה על תקשורת מקצועית")
            recommendations.append("⚠️ Inappropriate language detected - professional communication training needed")
            
            words_by_severity = inappropriate_lang.get('words_by_severity', {})
            if len(words_by_severity.get('critical', [])) > 0:
                recommendations.append("⚠️ זוהו מילים קריטיות - יש לבצע בדיקה מיידית")
                recommendations.append("⚠️ Critical words detected - immediate investigation required")
        
        # Neglect recommendations
        neglect_analysis = analysis_results.get('neglect_analysis', {})
        neglect_severity = neglect_analysis.get('neglect_severity', 'none')
        
        if neglect_severity in ['high', 'critical']:
            recommendations.append("יש לבצע בדיקה מיידית של תנאי הטיפול בילדים")
            recommendations.append("מומלץ להעביר הדרכה לצוות על זיהוי צרכי ילדים ותגובה מהירה")
        
        # Unanswered cries recommendations
        unanswered_cries = neglect_analysis.get('unanswered_cries', [])
        if len(unanswered_cries) > 2:
            recommendations.append("יש לשפר את מערכת התראה לתגובה לבכי ילדים")
            recommendations.append("מומלץ להגדיר פרוטוקול תגובה מהירה לבכי")
        
        # Emotional recommendations
        concerning_emotions = analysis_results.get('concerning_emotions', [])
        if concerning_emotions:
            recommendations.append("יש לספק תמיכה רגשית לצוות ולשקול הפחתת לחץ")
            recommendations.append("מומלץ להעביר הדרכה על ניהול מתח ורגשות")
        
        # General recommendations
        if not recommendations:
            recommendations.append("הניתוח מראה סביבה תקינה ללא בעיות משמעותיות")
            recommendations.append("מומלץ להמשיך במעקב שוטף")
        
        return recommendations
    
    def _generate_statistics(self, analysis_results: Dict) -> Dict:
        """
        Generate statistical summary
        יצירת סיכום סטטיסטי
        """
        stats = {
            'audio_duration_minutes': analysis_results.get('duration', 0) / 60,
            'total_incidents': 0,
            'incidents_per_hour': 0,
            'severity_distribution': {'low': 0, 'medium': 0, 'high': 0, 'critical': 0},
            'time_distribution': {}
        }
        
        # Count total incidents
        emotion_incidents = len(analysis_results.get('concerning_emotions', []))
        violence_incidents = len(analysis_results.get('violence_segments', []))
        unanswered_cries = len(analysis_results.get('unanswered_cries', []))
        neglect_incidents = len(analysis_results.get('ignored_distress_episodes', []))
        
        stats['total_incidents'] = emotion_incidents + violence_incidents + unanswered_cries + neglect_incidents
        
        # Calculate incidents per hour
        duration_hours = stats['audio_duration_minutes'] / 60
        if duration_hours > 0:
            stats['incidents_per_hour'] = stats['total_incidents'] / duration_hours
        
        # Severity distribution
        for emotion in analysis_results.get('concerning_emotions', []):
            severity = emotion.get('severity', 'low')
            stats['severity_distribution'][severity] += 1
        
        for violence in analysis_results.get('violence_segments', []):
            severity = violence.get('adjusted_severity', 'low')
            stats['severity_distribution'][severity] += 1
        
        return stats
    
    def _get_emotion_description(self, emotion: Dict) -> str:
        """Generate description for emotion incident"""
        emotion_hebrew = self.translations['emotions'].get(emotion['detected_emotion'], emotion['detected_emotion'])
        severity_hebrew = self.translations['severity_levels'].get(emotion['severity'], emotion['severity'])
        
        return f"זוהה {emotion_hebrew} ברמת חומרה {severity_hebrew} (ביטחון: {emotion['confidence']:.2f})"
    
    def _get_violence_description(self, violence: Dict) -> str:
        """Generate description for violence incident"""
        violence_types_hebrew = [self.translations['violence_types'].get(vt, vt) 
                               for vt in violence.get('violence_types', [])]
        severity_hebrew = self.translations['severity_levels'].get(violence.get('adjusted_severity', 'unknown'), 'לא ידוע')
        
        return f"זוהה {', '.join(violence_types_hebrew)} ברמת חומרה {severity_hebrew}"
    
    def _get_cry_description(self, cry: Dict) -> str:
        """Generate description for cry incident"""
        intensity_hebrew = self.translations['cry_intensity'].get(cry.get('intensity', 'medium'), 'בינוני')
        response_hebrew = 'עם תגובה' if cry.get('response_detected', False) else 'ללא תגובה'
        
        return f"בכי {intensity_hebrew} {response_hebrew}"
    
    def _find_matching_clip(self, audio_clips: List[Dict], incident_type: str, timestamp: str) -> Optional[Dict]:
        """
        Find matching audio clip for an incident
        מציאת קטע אודיו תואם לאירוע
        """
        for clip in audio_clips:
            if clip['incident_type'] == incident_type:
                # Extract start time from timestamp string (e.g., "5.0s - 10.0s")
                try:
                    start_time_str = timestamp.split(' - ')[0].replace('s', '')
                    start_time = float(start_time_str)
                    if abs(clip['start_time'] - start_time) < 1.0:  # Within 1 second
                        return clip
                except:
                    continue
        return None
    
    def _generate_audio_player(self, clip: Dict, player_id: str) -> str:
        """
        Generate HTML audio player for a clip
        יצירת נגן אודיו HTML לקטע
        """
        if not clip:
            return ""
        
        return f"""
        <div class="audio-player">
            <p><strong>🎵 האזן לאירוע:</strong></p>
            <audio controls style="width: 100%; margin: 10px 0;">
                <source src="{clip['filename']}" type="audio/wav">
                הדפדפן שלך לא תומך בנגן אודיו
            </audio>
            <p><small>משך: {clip['duration']:.1f} שניות | גודל: {clip['size_kb']} KB</small></p>
        </div>
        """
    
    def _format_hebrew_date(self, timestamp: datetime) -> str:
        """Format timestamp in Hebrew"""
        months_hebrew = {
            1: 'ינואר', 2: 'פברואר', 3: 'מרץ', 4: 'אפריל',
            5: 'מאי', 6: 'יוני', 7: 'יולי', 8: 'אוגוסט',
            9: 'ספטמבר', 10: 'אוקטובר', 11: 'נובמבר', 12: 'דצמבר'
        }
        
        day = timestamp.day
        month = months_hebrew[timestamp.month]
        year = timestamp.year
        time = timestamp.strftime('%H:%M')
        
        return f"{day} {month} {year}, {time}"
    
    def _save_json_report(self, report: Dict):
        """Save report as JSON file"""
        filename = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"JSON report saved: {filepath}")
    
    def _save_html_report(self, report: Dict):
        """Save report as HTML file"""
        filename = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        filepath = os.path.join(self.output_dir, filename)
        
        html_content = self._generate_html_content(report)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"HTML report saved: {filepath}")
    
    def _generate_html_content(self, report: Dict) -> str:
        """Generate HTML content for report"""
        html = f"""
<!DOCTYPE html>
<html dir="rtl" lang="he">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>דוח ניתוח הקלטות גן ילדים</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #2c3e50; text-align: center; margin-bottom: 30px; }}
        h2 {{ color: #34495e; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
        .summary {{ background-color: #ecf0f1; padding: 15px; border-radius: 5px; margin: 20px 0; }}
        .critical {{ background-color: #e74c3c; color: white; }}
        .high {{ background-color: #e67e22; color: white; }}
        .medium {{ background-color: #f39c12; color: white; }}
        .low {{ background-color: #27ae60; color: white; }}
        .incident {{ background-color: #f8f9fa; border: 1px solid #dee2e6; padding: 10px; margin: 10px 0; border-radius: 5px; }}
        .metadata {{ background-color: #e8f4f8; padding: 10px; border-radius: 5px; margin: 10px 0; }}
        .recommendations {{ background-color: #d4edda; padding: 15px; border-radius: 5px; }}
        ul {{ padding-right: 20px; }}
        li {{ margin: 5px 0; }}
        .timestamp {{ font-weight: bold; color: #7f8c8d; }}
        .audio-player {{ 
            background-color: #f8f9fa; 
            border: 1px solid #dee2e6; 
            border-radius: 5px; 
            padding: 15px; 
            margin: 10px 0; 
        }}
        .audio-player audio {{ 
            border-radius: 5px; 
            box-shadow: 0 2px 5px rgba(0,0,0,0.1); 
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>דוח ניתוח הקלטות גן ילדים</h1>
        
        <div class="metadata">
            <h2>פרטי הקובץ</h2>
            <p><strong>שם הקובץ:</strong> {report['metadata']['file_name']}</p>
            <p><strong>תאריך ניתוח:</strong> {report['metadata']['analysis_date_hebrew']}</p>
            <p><strong>משך ההקלטה:</strong> {report['metadata']['audio_duration']:.1f} שניות ({report['metadata']['audio_duration']/60:.1f} דקות)</p>
        </div>
        
        <div class="summary {report['summary']['risk_level']}">
            <h2>סיכום כללי</h2>
            <p><strong>הערכה כללית:</strong> {report['summary']['overall_assessment']}</p>
            <p><strong>סה"כ אירועים:</strong> {report['summary']['total_incidents']}</p>
            <p><strong>אירועים קריטיים:</strong> {report['summary']['critical_incidents']}</p>
            <p><strong>ממצאים עיקריים:</strong></p>
            <ul>
        """
        
        for finding in report['summary']['key_findings']:
            html += f"<li>{finding}</li>"
        
        html += """
            </ul>
        </div>
        
        <h2>ממצאים מפורטים</h2>
        """
        
        # Emotional analysis
        if report['detailed_findings']['emotional_analysis']:
            html += "<h3>ניתוח רגשי</h3>"
            for i, emotion in enumerate(report['detailed_findings']['emotional_analysis']):
                # Find matching audio clip
                matching_clip = self._find_matching_clip(report.get('audio_clips', []), 'emotion', emotion['timestamp'])
                
                html += f"""
                <div class="incident">
                    <p class="timestamp">{emotion['timestamp']}</p>
                    <p><strong>רגש שזוהה:</strong> {emotion['detected_emotion']}</p>
                    <p><strong>חומרה:</strong> {emotion['severity']}</p>
                    <p><strong>ביטחון:</strong> {emotion['confidence']}</p>
                    <p>{emotion['description']}</p>
                    {self._generate_audio_player(matching_clip, f"emotion_{i}") if matching_clip else ""}
                </div>
                """
        
        # Violence analysis
        if report['detailed_findings']['violence_analysis']:
            html += "<h3>ניתוח אלימות</h3>"
            for i, violence in enumerate(report['detailed_findings']['violence_analysis']):
                matching_clip = self._find_matching_clip(report.get('audio_clips', []), 'violence', violence['timestamp'])
                
                html += f"""
                <div class="incident">
                    <p class="timestamp">{violence['timestamp']}</p>
                    <p><strong>סוג אלימות:</strong> {violence['violence_types']}</p>
                    <p><strong>חומרה:</strong> {violence['severity']}</p>
                    <p><strong>הקשר:</strong> {violence['context']}</p>
                    <p>{violence['description']}</p>
                    {self._generate_audio_player(matching_clip, f"violence_{i}") if matching_clip else ""}
                </div>
                """
        
        # Cry analysis
        if report['detailed_findings']['cry_analysis']:
            html += "<h3>ניתוח בכי</h3>"
            for i, cry in enumerate(report['detailed_findings']['cry_analysis']):
                matching_clip = self._find_matching_clip(report.get('audio_clips', []), 'cry', cry['timestamp'])
                
                html += f"""
                <div class="incident">
                    <p class="timestamp">{cry['timestamp']}</p>
                    <p><strong>עוצמת בכי:</strong> {cry['intensity']}</p>
                    <p><strong>תגובה:</strong> {cry['response_detected']}</p>
                    <p><strong>איכות תגובה:</strong> {cry['response_quality']}</p>
                    <p>{cry['description']}</p>
                    {self._generate_audio_player(matching_clip, f"cry_{i}") if matching_clip else ""}
                </div>
                """
        
        # Neglect analysis
        if report['detailed_findings']['neglect_analysis']:
            html += "<h3>ניתוח הזנחה</h3>"
            for i, neglect in enumerate(report['detailed_findings']['neglect_analysis']):
                matching_clip = self._find_matching_clip(report.get('audio_clips', []), 'neglect', neglect['timestamp'])
                
                html += f"""
                <div class="incident">
                    <p class="timestamp">{neglect['timestamp']}</p>
                    <p><strong>משך בכי:</strong> {neglect['cry_duration']}</p>
                    <p><strong>זמן ללא תגובה:</strong> {neglect['time_without_response']}</p>
                    <p><strong>חומרת הזנחה:</strong> {neglect['neglect_severity']}</p>
                    <p>{neglect['description']}</p>
                    {self._generate_audio_player(matching_clip, f"neglect_{i}") if matching_clip else ""}
                </div>
                """
        
        # Inappropriate language analysis
        if report['detailed_findings'].get('inappropriate_language'):
            html += "<h3>⚠️ זיהוי שפה לא הולמת וקללות</h3>"
            for i, word_data in enumerate(report['detailed_findings']['inappropriate_language']):
                severity_class = f"severity-{word_data.get('severity', 'medium')}"
                
                html += f"""
                <div class="incident {severity_class}">
                    <p class="timestamp">⏱️ {word_data['timestamp']}</p>
                    <p><strong>מילה / Word:</strong> <span style="color: #721c24; font-weight: bold;">{word_data['word']}</span></p>
                    <p><strong>שפה / Language:</strong> {word_data['language']}</p>
                    <p><strong>חומרה / Severity:</strong> <span class="severity-badge severity-{word_data.get('severity', 'medium')}">{word_data['severity']}</span></p>
                    {f"<p><strong>הקשר / Context:</strong> <em>{word_data.get('context', '')[:150]}...</em></p>" if word_data.get('context') else ""}
                    <p>{word_data['description']}</p>
                </div>
                """
        
        # Recommendations
        html += """
        <div class="recommendations">
            <h2>המלצות</h2>
            <ul>
        """
        
        for recommendation in report['recommendations']:
            html += f"<li>{recommendation}</li>"
        
        html += """
            </ul>
        </div>
        
        <h2>סטטיסטיקות</h2>
        <div class="metadata">
            <p><strong>משך הקלטה:</strong> {:.1f} דקות</p>
            <p><strong>סה"כ אירועים:</strong> {}</p>
            <p><strong>אירועים לשעה:</strong> {:.2f}</p>
            <p><strong>התפלגות חומרה:</strong></p>
            <ul>
                <li>נמוכה: {}</li>
                <li>בינונית: {}</li>
                <li>גבוהה: {}</li>
                <li>קריטית: {}</li>
            </ul>
        </div>
        
    </div>
</body>
</html>
        """.format(
            report['statistics']['audio_duration_minutes'],
            report['statistics']['total_incidents'],
            report['statistics']['incidents_per_hour'],
            report['statistics']['severity_distribution']['low'],
            report['statistics']['severity_distribution']['medium'],
            report['statistics']['severity_distribution']['high'],
            report['statistics']['severity_distribution']['critical']
        )
        
        return html
    
    def _save_csv_report(self, report: Dict):
        """Save report as CSV file"""
        filename = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Write header
            writer.writerow(['סוג אירוע', 'זמן התחלה', 'זמן סיום', 'חומרה', 'תיאור'])
            
            # Write emotional incidents
            for emotion in report['detailed_findings']['emotional_analysis']:
                writer.writerow([
                    'רגש',
                    f"{emotion['timestamp'].split(' - ')[0]}s",
                    f"{emotion['timestamp'].split(' - ')[1].replace('s', '')}s",
                    emotion['severity'],
                    emotion['description']
                ])
            
            # Write violence incidents
            for violence in report['detailed_findings']['violence_analysis']:
                writer.writerow([
                    'אלימות',
                    f"{violence['timestamp'].split(' - ')[0]}s",
                    f"{violence['timestamp'].split(' - ')[1].replace('s', '')}s",
                    violence['severity'],
                    violence['description']
                ])
            
            # Write cry incidents
            for cry in report['detailed_findings']['cry_analysis']:
                writer.writerow([
                    'בכי',
                    f"{cry['timestamp'].split(' - ')[0]}s",
                    f"{cry['timestamp'].split(' - ')[1].replace('s', '')}s",
                    cry['response_detected'],
                    cry['description']
                ])
            
            # Write neglect incidents
            for neglect in report['detailed_findings']['neglect_analysis']:
                writer.writerow([
                    'הזנחה',
                    f"{neglect['timestamp'].split(' - ')[0]}s",
                    f"{neglect['timestamp'].split(' - ')[1].replace('s', '')}s",
                    neglect['neglect_severity'],
                    neglect['description']
                ])
        
        print(f"CSV report saved: {filepath}")

if __name__ == "__main__":
    # Test the report generator
    generator = ReportGenerator()
    print("Report Generator initialized successfully")
    print("Ready to generate comprehensive reports...")
