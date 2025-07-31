@echo off
echo Building EEDTO Implementation...

:: Check if Maven is installed
where mvn >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo Maven is not installed or not in PATH. Please install Maven and try again.
    exit /b 1
)

:: Create logs directory if it doesn't exist
if not exist logs mkdir logs

:: Create output directory if it doesn't exist
if not exist output mkdir output

:: Build with Maven
mvn clean package

if %ERRORLEVEL% neq 0 (
    echo Build failed.
    exit /b 1
)

echo Build successful. You can run the simulation using run.bat
