"""
Setup Script for Semester Named Entity Recognition System
Handles installation and configuration
"""

import os
import sys
import subprocess
import platform

def print_step(step_num, description):
    """Print formatted step"""
    print(f"\n{'='*60}")
    print(f"STEP {step_num}: {description}")
    print('='*60)

def run_command(command, description=""):
    """Run a command and handle errors"""
    try:
        print(f"Running: {command}")
        if description:
            print(f"Purpose: {description}")
        
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Success!")
            if result.stdout:
                print(result.stdout)
            return True
        else:
            print("❌ Failed!")
            if result.stderr:
                print(f"Error: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error running command: {str(e)}")
        return False

def check_python():
    """Check Python installation"""
    print_step(1, "CHECKING PYTHON INSTALLATION")
    
    try:
        python_version = sys.version_info
        print(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
        
        if python_version.major >= 3 and python_version.minor >= 8:
            print("✅ Python version is compatible!")
            return True
        else:
            print("❌ Python 3.8+ required!")
            return False
    except Exception as e:
        print(f"❌ Error checking Python: {str(e)}")
        return False

def install_packages():
    """Install required packages"""
    print_step(2, "INSTALLING PYTHON PACKAGES")
    
    packages = [
        "pytesseract==0.3.10",
        "Pillow==10.0.1", 
        "PyPDF2==3.0.1",
        "pdf2image==1.16.3",
        "scikit-learn==1.3.2",
        "pandas==2.1.4",
        "numpy==1.24.4",
        "matplotlib==3.8.2", 
        "seaborn==0.13.0",
        "nltk==3.8.1",
        "joblib==1.3.2",
        "opencv-python==4.8.1.78"
    ]
    
    print("Installing packages one by one...")
    failed_packages = []
    
    for package in packages:
        print(f"\nInstalling {package}...")
        success = run_command(f"pip install {package}")
        if not success:
            failed_packages.append(package)
    
    if failed_packages:
        print(f"\n❌ Failed to install: {', '.join(failed_packages)}")
        print("Try installing manually:")
        for pkg in failed_packages:
            print(f"  pip install {pkg}")
        return False
    else:
        print("\n✅ All packages installed successfully!")
        return True

def check_tesseract():
    """Check Tesseract installation"""
    print_step(3, "CHECKING TESSERACT OCR")
    
    try:
        import pytesseract
        
        # Try to get tesseract version
        try:
            version = pytesseract.get_tesseract_version()
            print(f"✅ Tesseract OCR found! Version: {version}")
            return True
        except:
            print("❌ Tesseract OCR not found in PATH")
            print("\nTo install Tesseract:")
            if platform.system() == "Windows":
                print("1. Download from: https://github.com/UB-Mannheim/tesseract/wiki")
                print("2. Install to: C:\\Program Files\\Tesseract-OCR\\")
                print("3. Add to system PATH")
            elif platform.system() == "Darwin":  # macOS
                print("Run: brew install tesseract")
            else:  # Linux
                print("Run: sudo apt install tesseract-ocr")
            return False
    except ImportError:
        print("❌ pytesseract not installed")
        return False

def setup_directories():
    """Create necessary directories"""
    print_step(4, "SETTING UP DIRECTORIES")
    
    directories = ['data', 'models', 'output', 'temp']
    
    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
            print(f"✅ Created directory: {directory}")
        except Exception as e:
            print(f"❌ Failed to create {directory}: {str(e)}")
            return False
    
    return True

def download_nltk_data():
    """Download required NLTK data"""
    print_step(5, "DOWNLOADING NLTK DATA")
    
    try:
        import nltk
        
        data_sets = [
            'punkt',
            'averaged_perceptron_tagger', 
            'maxent_ne_chunker',
            'words',
            'stopwords'
        ]
        
        for dataset in data_sets:
            print(f"Downloading {dataset}...")
            try:
                nltk.download(dataset, quiet=True)
                print(f"✅ Downloaded {dataset}")
            except Exception as e:
                print(f"❌ Failed to download {dataset}: {str(e)}")
        
        return True
    except ImportError:
        print("❌ NLTK not installed")
        return False

def create_sample_data():
    """Create sample training data"""
    print_step(6, "CREATING SAMPLE DATA")
    
    sample_data = [
        {"text": "This course is offered in the first semester.", "label": "SEMESTER_1"},
        {"text": "Students must complete spring semester requirements.", "label": "SPRING_SEMESTER"},
        {"text": "The fall semester schedule includes advanced courses.", "label": "FALL_SEMESTER"},
        {"text": "Second semester curriculum focuses on practical applications.", "label": "SEMESTER_2"},
        {"text": "Summer semester offers intensive programs.", "label": "SUMMER_SEMESTER"},
        {"text": "This is a general studies course.", "label": "OTHER"}
    ]
    
    try:
        import json
        with open('data/sample_training_data.json', 'w', encoding='utf-8') as f:
            json.dump(sample_data, f, indent=2, ensure_ascii=False)
        print("✅ Sample training data created!")
        return True
    except Exception as e:
        print(f"❌ Failed to create sample data: {str(e)}")
        return False

def test_basic_functionality():
    """Test basic system functionality"""
    print_step(7, "TESTING BASIC FUNCTIONALITY")
    
    try:
        # Test imports
        print("Testing imports...")
        import pandas as pd
        import numpy as np
        import sklearn
        print("✅ Core packages imported successfully!")
        
        # Test text processing
        print("Testing text processing...")
        import nltk
        from nltk.tokenize import word_tokenize
        tokens = word_tokenize("This is a test sentence.")
        print(f"✅ Text processing works! Tokens: {tokens}")
        
        return True
    except Exception as e:
        print(f"❌ Basic functionality test failed: {str(e)}")
        return False

def main():
    """Main setup function"""
    print("🚀 SEMESTER NER SYSTEM SETUP")
    print("This script will set up everything you need to run the system.")
    
    # Check if user wants to proceed
    response = input("\nDo you want to proceed with setup? (y/n): ").lower().strip()
    if response != 'y':
        print("Setup cancelled.")
        return
    
    # Run setup steps
    steps = [
        check_python,
        install_packages,
        check_tesseract,
        setup_directories, 
        download_nltk_data,
        create_sample_data,
        test_basic_functionality
    ]
    
    success_count = 0
    for step in steps:
        if step():
            success_count += 1
        else:
            print(f"\n⚠️  Step failed, but continuing...")
    
    print(f"\n{'='*60}")
    print(f"SETUP COMPLETE: {success_count}/{len(steps)} steps successful")
    print('='*60)
    
    if success_count >= 5:  # Most steps successful
        print("✅ Setup mostly successful! You can now run:")
        print("   python simple_demo.py    # For a basic demo")
        print("   python train_model.py    # To train the model")
        print("   python main.py           # For interactive mode")
    else:
        print("❌ Setup had issues. Please check the errors above.")
        print("You may need to:")
        print("1. Install Tesseract OCR manually")
        print("2. Install packages manually: pip install -r requirements.txt")
    
    print(f"\nNext steps:")
    print("1. If Tesseract failed, install it manually")
    print("2. Run: python simple_demo.py")
    print("3. Check README.md for detailed instructions")

if __name__ == "__main__":
    main() 