@echo off
echo ===================================================
echo JCORA-MEC Implementation - Build Script (Windows)
echo ===================================================

REM Check if Maven is installed
where mvn >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo Maven is not installed or not in PATH.
    echo Please install Maven and add it to your PATH.
    exit /b 1
)

REM Create output directory if it doesn't exist
if not exist "output" mkdir output

REM Clean and build the project
echo Building the project...
call mvn clean package

if %ERRORLEVEL% neq 0 (
    echo Build failed. Please check the error messages above.
    exit /b 1
)

echo Build completed successfully.
echo The JAR file is located at: target\jcora-mec-1.0-SNAPSHOT.jar
echo.
echo To run the simulation, use: run.bat [config_file]
echo.
