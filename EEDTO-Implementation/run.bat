@echo off
echo Running EEDTO Simulation...

:: Check if the JAR file exists
if not exist target\EEDTO-Implementation-1.0-SNAPSHOT.jar (
    echo JAR file not found. Please build the project first using build.bat
    exit /b 1
)

:: Create logs directory if it doesn't exist
if not exist logs mkdir logs

:: Create output directory if it doesn't exist
if not exist output mkdir output

:: Run the simulation
java -jar target\EEDTO-Implementation-1.0-SNAPSHOT.jar

if %ERRORLEVEL% neq 0 (
    echo Simulation failed.
    exit /b 1
)

echo Simulation completed successfully. Results are available in the output directory.
echo Logs are available in the logs directory.
