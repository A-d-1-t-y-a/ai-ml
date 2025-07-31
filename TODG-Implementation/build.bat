@echo off
echo TODG Simulation - Building project

REM Check if Maven is installed
where mvn >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo Maven is not installed or not in PATH. Please install Maven and try again.
    exit /b 1
)

REM Create output directory if it doesn't exist
if not exist "output" mkdir output

REM Build the project with Maven
echo Building project with Maven...
call mvn clean package

if %ERRORLEVEL% neq 0 (
    echo Build failed! Please check the error messages above.
    exit /b 1
)

echo Build completed successfully.
echo You can now run the simulation using run.bat
exit /b 0
