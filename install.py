"""
Installation Script for Kindergarten Recording Analyzer
סקריפט התקנה למערכת ניתוח הקלטות גן ילדים
"""

import os
import sys
import subprocess
import platform

def check_python_version():
    """Check if Python version is compatible"""
    print("בודק גרסת Python...")
    print("Checking Python version...")
    
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ נדרש. גרסה נוכחית:", f"{version.major}.{version.minor}")
        print("❌ Python 3.8+ required. Current version:", f"{version.major}.{version.minor}")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} תואם")
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} compatible")
    return True

def install_requirements():
    """Install required packages"""
    print("\nמתקין חבילות נדרשות...")
    print("Installing required packages...")
    
    try:
        # Upgrade pip first
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        
        # Install requirements
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        
        print("✅ חבילות הותקנו בהצלחה")
        print("✅ Packages installed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ שגיאה בהתקנת חבילות: {e}")
        print(f"❌ Error installing packages: {e}")
        return False

def create_directories():
    """Create required directories"""
    print("\nיוצר תיקיות נדרשות...")
    print("Creating required directories...")
    
    directories = ['uploads', 'reports', 'models', 'templates']
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"✅ נוצרה תיקייה: {directory}")
            print(f"✅ Created directory: {directory}")
        else:
            print(f"ℹ️  תיקייה כבר קיימת: {directory}")
            print(f"ℹ️  Directory already exists: {directory}")

def test_installation():
    """Test if installation was successful"""
    print("\nבודק התקנה...")
    print("Testing installation...")
    
    try:
        # Test imports
        import librosa
        import numpy as np
        import pandas as pd
        import flask
        import sklearn
        
        print("✅ כל המודולים נטענו בהצלחה")
        print("✅ All modules loaded successfully")
        
        # Test basic functionality
        from main import KindergartenRecordingAnalyzer
        analyzer = KindergartenRecordingAnalyzer()
        
        print("✅ המערכת אותחלה בהצלחה")
        print("✅ System initialized successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ שגיאה בטעינת מודול: {e}")
        print(f"❌ Error loading module: {e}")
        return False
    except Exception as e:
        print(f"❌ שגיאה באימות המערכת: {e}")
        print(f"❌ Error validating system: {e}")
        return False

def main():
    """Main installation function"""
    print("=" * 60)
    print("מתקין מערכת ניתוח הקלטות גן ילדים")
    print("Installing Kindergarten Recording Analyzer")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        print("\n❌ התקנה נכשלה - גרסת Python לא תואמת")
        print("❌ Installation failed - Python version incompatible")
        sys.exit(1)
    
    # Install requirements
    if not install_requirements():
        print("\n❌ התקנה נכשלה - שגיאה בהתקנת חבילות")
        print("❌ Installation failed - Error installing packages")
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Test installation
    if not test_installation():
        print("\n❌ התקנה נכשלה - שגיאה באימות המערכת")
        print("❌ Installation failed - Error validating system")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("🎉 התקנה הושלמה בהצלחה!")
    print("🎉 Installation completed successfully!")
    print("=" * 60)
    
    print("\n📋 הוראות שימוש:")
    print("📋 Usage Instructions:")
    print()
    print("1. הרצת דוגמה:")
    print("   python example_usage.py")
    print()
    print("2. ניתוח קובץ יחיד:")
    print("   python main.py your_recording.wav")
    print()
    print("3. ממשק ווב:")
    print("   python web_app.py")
    print("   ואז פתח: http://localhost:5000")
    print()
    print("4. קריאת מדריך:")
    print("   קרא את README.md למידע מפורט")
    print()
    
    print("⚠️  הערות חשובות:")
    print("⚠️  Important Notes:")
    print("- המערכת מיועדת להגנה על ילדים בלבד")
    print("- יש להשתמש בה בהתאם לחוק המקומי")
    print("- יש להשיג הסכמה מפורשת לפני השימוש")
    print()
    print("- System is for child protection only")
    print("- Use in accordance with local laws")
    print("- Obtain explicit consent before use")
    
    print("\n✅ המערכת מוכנה לשימוש!")
    print("✅ System is ready for use!")

if __name__ == "__main__":
    main()
