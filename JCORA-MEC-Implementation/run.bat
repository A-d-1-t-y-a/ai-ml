@echo off
echo ===================================================
echo JCORA-MEC Implementation - Run Script (Windows)
echo ===================================================

REM Check if Java is installed
where java >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo Java is not installed or not in PATH.
    echo Please install Java and add it to your PATH.
    exit /b 1
)

REM Check if the JAR file exists
if not exist "target\JCORA-MEC-Implementation-1.0-SNAPSHOT.jar-with-dependencies.jar" (
    echo JAR file not found. Please build the project first using build.bat
    exit /b 1
)

REM Set default configuration file
set CONFIG_FILE=config\simulation.properties

REM Check if a configuration file was provided
if not "%~1"=="" (
    set CONFIG_FILE=%~1
)

REM Check if the configuration file exists
if not exist "%CONFIG_FILE%" (
    echo Configuration file not found: %CONFIG_FILE%
    echo Using default configuration file: config\simulation.properties
    set CONFIG_FILE=config\simulation.properties
)

REM Run the simulation
echo Running JCORA-MEC simulation with configuration: %CONFIG_FILE%
echo.
java -jar target\JCORA-MEC-Implementation-1.0-SNAPSHOT.jar-with-dependencies.jar %CONFIG_FILE%

echo.
if %ERRORLEVEL% equ 0 (
    echo Simulation completed successfully.
    echo Results are available in the output directory.
) else (
    echo Simulation failed. Please check the error messages above.
)
