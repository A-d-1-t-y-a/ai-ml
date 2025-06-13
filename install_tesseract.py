"""
Tesseract Installation Helper
Helps verify and configure Tesseract OCR for the semester NER system
"""

import os
import subprocess
import sys
from pathlib import Path

def check_tesseract():
    """Check if Tesseract is available"""
    try:
        result = subprocess.run(['tesseract', '--version'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Tesseract is installed and available in PATH")
            print(f"Version: {result.stdout.split()[1]}")
            return True
    except FileNotFoundError:
        pass
    
    # Check common Windows installation paths
    common_paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        r"C:\tools\tesseract\tesseract.exe"
    ]
    
    for path in common_paths:
        if os.path.exists(path):
            print(f"✅ Tesseract found at: {path}")
            print("Note: Please add this to your PATH or update config.py")
            return path
    
    print("❌ Tesseract not found")
    return False

def test_pytesseract():
    """Test pytesseract import and configuration"""
    try:
        import pytesseract
        print("✅ pytesseract module imported successfully")
        
        # Try to get version
        try:
            version = pytesseract.get_tesseract_version()
            print(f"✅ Tesseract version accessible: {version}")
            return True
        except Exception as e:
            print(f"❌ Cannot access Tesseract: {e}")
            return False
            
    except ImportError:
        print("❌ pytesseract not installed. Run: pip install pytesseract")
        return False

def download_instructions():
    """Print download instructions"""
    print("\n" + "="*60)
    print("TESSERACT INSTALLATION INSTRUCTIONS")
    print("="*60)
    print("1. Download Tesseract for Windows:")
    print("   https://github.com/UB-Mannheim/tesseract/wiki")
    print("\n2. Look for the latest installer:")
    print("   tesseract-ocr-w64-setup-v5.X.X.exe")
    print("\n3. During installation:")
    print("   - Install to default location: C:\\Program Files\\Tesseract-OCR\\")
    print("   - ✅ CHECK 'Add to PATH' option")
    print("\n4. After installation, restart your terminal")
    print("\n5. Test with: tesseract --version")
    print("="*60)

def update_config():
    """Update config.py with correct Tesseract path"""
    tesseract_path = check_tesseract()
    
    if tesseract_path and isinstance(tesseract_path, str):
        # Read current config
        try:
            with open('config.py', 'r') as f:
                content = f.read()
            
            # Update the TESSERACT_CMD line
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if line.startswith('TESSERACT_CMD'):
                    lines[i] = f"TESSERACT_CMD = r'{tesseract_path}'"
                    break
            
            # Write back
            with open('config.py', 'w') as f:
                f.write('\n'.join(lines))
            
            print(f"✅ Updated config.py with Tesseract path: {tesseract_path}")
            
        except Exception as e:
            print(f"❌ Could not update config.py: {e}")

def main():
    """Main installation helper"""
    print("Tesseract OCR Installation Helper")
    print("-" * 40)
    
    # Check if Tesseract is available
    tesseract_status = check_tesseract()
    
    # Check pytesseract
    pytesseract_status = test_pytesseract()
    
    if tesseract_status and pytesseract_status:
        print("\n🎉 All good! Tesseract is ready to use.")
        
        # Update config if needed
        if isinstance(tesseract_status, str):
            update_config()
        
        return True
    
    if not tesseract_status:
        download_instructions()
    
    if not pytesseract_status:
        print("\n📦 Install pytesseract:")
        print("pip install pytesseract")
    
    print("\n⚠️  Please install missing components and run this script again.")
    return False

if __name__ == "__main__":
    main() 