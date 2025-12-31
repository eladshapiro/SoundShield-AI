# התקנת FFmpeg / Installing FFmpeg

FFmpeg הוא דרישה חובה עבור Whisper transcription. ללא FFmpeg, זיהוי שפה לא הולמת לא יעבוד.

FFmpeg is a required dependency for Whisper transcription. Without FFmpeg, inappropriate language detection will not work.

## Windows

### שיטה 1: Chocolatey (מומלץ) / Method 1: Chocolatey (Recommended)

1. התקן Chocolatey (אם לא מותקן) / Install Chocolatey (if not installed):
   ```powershell
   # הרץ PowerShell כמנהל / Run PowerShell as Administrator
   Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
   ```

2. התקן FFmpeg / Install FFmpeg:
   ```powershell
   choco install ffmpeg
   ```

3. הפעל מחדש את הטרמינל / Restart your terminal

### שיטה 2: הורדה ידנית / Method 2: Manual Download

1. הורד FFmpeg מ- / Download FFmpeg from: https://ffmpeg.org/download.html
2. חלץ לתיקייה (למשל / Extract to a folder (e.g., `C:\ffmpeg`)
3. הוסף ל-PATH / Add to PATH:
   - פתח System Properties → Environment Variables
   - הוסף `C:\ffmpeg\bin` ל-PATH
   - הפעל מחדש את הטרמינל

### שיטה 3: winget (Windows 10/11)

```powershell
winget install ffmpeg
```

### בדיקה / Verification:

```bash
ffmpeg -version
```

אם אתה רואה את גרסת FFmpeg, ההתקנה הצליחה!

If you see the FFmpeg version, installation was successful!

## Linux

```bash
sudo apt-get update
sudo apt-get install ffmpeg
```

## macOS

```bash
brew install ffmpeg
```

## הערות חשובות / Important Notes

- **לאחר התקנת FFmpeg, הפעל מחדש את הטרמינל / After installing FFmpeg, restart your terminal**
- **ודא ש-FFmpeg ב-PATH / Make sure FFmpeg is in PATH**
- **בלי FFmpeg, הניתוח יעבוד אבל זיהוי שפה לא הולמת לא יעבוד / Without FFmpeg, analysis will work but inappropriate language detection won't work**

## פתרון בעיות / Troubleshooting

אם FFmpeg לא נמצא אחרי ההתקנה / If FFmpeg is not found after installation:

1. ודא שהתיקייה ב-PATH / Make sure the folder is in PATH
2. הפעל מחדש את הטרמינל / Restart your terminal
3. בדוק עם / Check with: `ffmpeg -version`
4. אם עדיין לא עובד, נסה להוסיף את הנתיב המלא / If still not working, try adding the full path

