# Quick Start Guide / מדריך התחלה מהירה

## התקנה מהירה / Quick Installation

```bash
# 1. התקן תלויות
pip install -r requirements.txt

# 2. הרץ סקריפט התקנה
python install.py

# 3. בדוק שהכל עובד
python example_usage.py
```

## שימוש מהיר / Quick Usage

### 1. ממשק ווב (הכי קל) / Web Interface (Easiest)
```bash
python web_app.py
```
פתח בדפדפן: http://localhost:5000

### 2. שורת פקודה / Command Line
```bash
python main.py your_recording.wav
```

### 3. דוגמה / Example
```bash
python example_usage.py
```

## מה המערכת עושה / What the System Does

✅ **זיהוי רגשות** - מזהה כעס, לחץ, אגרסיה בקול הצוות  
✅ **זיהוי בכי** - מזהה בכי תינוקות ובודק אם הצוות הגיב  
✅ **זיהוי אלימות** - מזהה צעקות, איומים ואלימות מילולית  
✅ **זיהוי הזנחה** - מזהה אי תגובה לבכי וחוסר אינטראקציה  
✅ **דוחות מפורטים** - יוצר דוחות בעברית עם המלצות  

## פורמטי אודיו נתמכים / Supported Audio Formats
WAV, MP3, M4A, FLAC, AAC, OGG (עד 500MB)

## דוגמת תוצאה / Example Output
```
סיכום ניתוח / Analysis Summary
==============================
הערכה כללית: concerning
סה"כ אירועים: 3
אירועים קריטיים: 1
רמת סיכון: high

ממצאים עיקריים / Key Findings:
• זוהו 2 אירועי אלימות פוטנציאליים
• זוהו 1 אירועי בכי ללא תגובה
• רמת הזנחה: בינונית

המלצות / Recommendations:
• מומלץ להעביר הדרכה לצוות על ניהול קונפליקטים
• יש לשפר את מערכת התראה לתגובה לבכי ילדים
```

## ⚠️ חשוב מאוד / Very Important
- המערכת מיועדת להגנה על ילדים בלבד
- יש להשתמש בה בהתאם לחוק המקומי
- יש להשיג הסכמה מפורשת לפני השימוש
- זה כלי עזר בלבד - לא תחליף לשיפוט אנושי מקצועי

## בעיות נפוצות / Common Issues

### שגיאה: "ModuleNotFoundError"
```bash
pip install -r requirements.txt
```

### שגיאה: "Audio file not found"
- וודא שהנתיב לקובץ נכון
- בדוק שהקובץ קיים וניתן לקריאה

### ממשק ווב לא נטען
- בדוק שהפורט 5000 פנוי
- נסה: http://127.0.0.1:5000

## תמיכה / Support
- קרא את README.md למידע מפורט
- הרץ `python example_usage.py` לבדיקה
- וודא שכל התלויות מותקנות

---
**הערה:** זה מדריך התחלה מהירה. למידע מפורט קרא את README.md
