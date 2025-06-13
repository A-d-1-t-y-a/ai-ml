"""
Safe Package Installation Script for Python 3.13
Handles compatibility issues and installs packages one by one
"""

import subprocess
import sys
import os

def run_pip_command(command):
    """Run pip command and return success status"""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Success!")
            return True
        else:
            print("❌ Failed!")
            print(f"Error: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Exception: {str(e)}")
        return False

def install_package(package_name, fallback_name=None):
    """Install a package with optional fallback"""
    print(f"\n📦 Installing {package_name}...")
    
    # Try main package first
    success = run_pip_command(f"pip install {package_name}")
    
    # Try fallback if main fails
    if not success and fallback_name:
        print(f"   Trying fallback: {fallback_name}")
        success = run_pip_command(f"pip install {fallback_name}")
    
    return success

def main():
    print("🚀 SAFE PACKAGE INSTALLATION FOR PYTHON 3.13")
    print("=" * 60)
    
    # Check Python version
    print(f"Python version: {sys.version}")
    
    # Core packages (install these first)
    core_packages = [
        ("numpy", None),
        ("pandas", None), 
        ("scikit-learn", None),
        ("joblib", None),
        ("matplotlib", None),
        ("nltk", None)
    ]
    
    # Image processing packages
    image_packages = [
        ("Pillow", None),
        ("opencv-python", "opencv-python-headless")
    ]
    
    # PDF and OCR packages  
    pdf_packages = [
        ("PyPDF2", "pypdf"),
        ("pdf2image", None),
        ("pytesseract", None)
    ]
    
    # Visualization packages
    viz_packages = [
        ("seaborn", None)
    ]
    
    all_package_groups = [
        ("Core Packages", core_packages),
        ("Image Processing", image_packages), 
        ("PDF Processing", pdf_packages),
        ("Visualization", viz_packages)
    ]
    
    successful_installs = 0
    total_packages = sum(len(packages) for _, packages in all_package_groups)
    
    for group_name, packages in all_package_groups:
        print(f"\n🔹 Installing {group_name}...")
        print("-" * 40)
        
        for package, fallback in packages:
            if install_package(package, fallback):
                successful_installs += 1
    
    print(f"\n{'='*60}")
    print(f"INSTALLATION COMPLETE")
    print(f"{'='*60}")
    print(f"✅ Successfully installed: {successful_installs}/{total_packages} packages")
    
    if successful_installs >= total_packages - 2:  # Allow 2 failures
        print("🎉 Installation mostly successful!")
        print("\nYou can now try:")
        print("   py simple_demo.py    # Test basic functionality")
        print("   py train_model.py    # Train the model")
        print("   py main.py           # Interactive mode")
    else:
        print("⚠️  Many packages failed to install.")
        print("This is common with Python 3.13. You can still use:")
        print("   py simple_demo.py    # This works without external packages")
    
    print(f"\nFor PDF processing, you'll still need to install Tesseract OCR manually.")

if __name__ == "__main__":
    main() 