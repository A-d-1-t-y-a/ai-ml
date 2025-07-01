@echo off
echo 🚀 Time Series Forecasting Project - Windows Setup
echo ================================================

:: Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

echo ✅ Python found
python --version

:: Check if we're in a virtual environment
if not defined VIRTUAL_ENV (
    echo.
    echo 📦 Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo ❌ Failed to create virtual environment
        pause
        exit /b 1
    )
    
    echo ✅ Virtual environment created
    echo.
    echo 🔄 Activating virtual environment...
    call venv\Scripts\activate.bat
    
    if errorlevel 1 (
        echo ❌ Failed to activate virtual environment
        pause
        exit /b 1
    )
    
    echo ✅ Virtual environment activated
) else (
    echo ✅ Virtual environment already active
)

echo.
echo 📥 Installing/upgrading pip and setuptools...
python -m pip install --upgrade pip setuptools wheel

if errorlevel 1 (
    echo ❌ Failed to upgrade pip and setuptools
    pause
    exit /b 1
)

echo.
echo 📦 Installing project dependencies...
python -m pip install -r requirements.txt

if errorlevel 1 (
    echo ❌ Failed to install requirements
    echo.
    echo 🔧 Trying alternative installation method...
    python -m pip install --use-pep517 -r requirements.txt
    
    if errorlevel 1 (
        echo ❌ Alternative installation also failed
        echo Please check your internet connection and try again
        pause
        exit /b 1
    )
)

echo ✅ All dependencies installed successfully!

echo.
echo 🎯 Running complete project setup...
python setup_project.py

echo.
echo 🎉 Setup complete! 
echo.
echo 📚 You can now run:
echo   - Dashboard: streamlit run dashboard.py
echo   - Full pipeline: python main_pipeline.py  
echo   - Quick demo: python financial_data_demo.py
echo   - Tests: python test_suite.py
echo.
pause 