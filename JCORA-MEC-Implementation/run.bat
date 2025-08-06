@echo off
setlocal enabledelayedexpansion

echo ===================================================
echo JCORA-MEC Implementation - Run Script (Windows)
echo ===================================================
echo.

REM Check if Java is installed
echo [INFO] Checking for Java installation...
where java >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Java is not installed or not in PATH.
    echo [ERROR] Please install Java JDK 8 or higher.
    exit /b 1
)
echo [INFO] Java found.

REM Check if the JAR file exists
set JAR_WITH_DEPS=target\JCORA-MEC-Implementation-1.0-SNAPSHOT-jar-with-dependencies.jar
set JAR_SIMPLE=target\JCORA-MEC-Implementation-1.0-SNAPSHOT.jar

echo [INFO] Checking for JAR files...
if exist "%JAR_WITH_DEPS%" (
    set JAR_FILE=%JAR_WITH_DEPS%
    echo [INFO] Using JAR with dependencies: %JAR_FILE%
) else if exist "%JAR_SIMPLE%" (
    set JAR_FILE=%JAR_SIMPLE%
    echo [INFO] Using simple JAR: %JAR_FILE%
    echo [WARNING] This may require additional classpath configuration.
) else (
    echo [ERROR] JAR file not found. Please build the project first using build.bat
    echo [ERROR] Expected: %JAR_WITH_DEPS%
    echo [ERROR] Or: %JAR_SIMPLE%
    exit /b 1
)

REM Set default configuration file
set CONFIG_FILE=config\simulation.properties

REM Check if a configuration file was provided
if not "%~1"=="" (
    set CONFIG_FILE=%~1
    echo [INFO] Using provided configuration: %CONFIG_FILE%
) else (
    echo [INFO] Using default configuration: %CONFIG_FILE%
)

REM Check if the configuration file exists
if not exist "%CONFIG_FILE%" (
    echo [WARNING] Configuration file not found: %CONFIG_FILE%
    if exist "config\simulation.properties" (
        set CONFIG_FILE=config\simulation.properties
        echo [INFO] Falling back to default: %CONFIG_FILE%
    ) else (
        echo [ERROR] No configuration file found. Creating default configuration...
        if not exist "config" mkdir config
        echo # Default JCORA-MEC Simulation Configuration > config\simulation.properties
        echo simulation.duration=3600 >> config\simulation.properties
        echo simulation.timestep=1.0 >> config\simulation.properties
        echo task.generation.probability=0.1 >> config\simulation.properties
        echo devices.count=10 >> config\simulation.properties
        echo servers.count=3 >> config\simulation.properties
        set CONFIG_FILE=config\simulation.properties
        echo [INFO] Created default configuration: %CONFIG_FILE%
    )
)

REM Create output and logs directories
echo [INFO] Creating output directories...
if not exist "output" mkdir output
if not exist "logs" mkdir logs

REM Run the simulation
echo.
echo [INFO] Starting JCORA-MEC simulation...
echo [INFO] Configuration: %CONFIG_FILE%
echo [INFO] JAR file: %JAR_FILE%
echo [INFO] Output directory: output\
echo [INFO] Logs directory: logs\
echo.
echo ========================================
echo           SIMULATION OUTPUT
echo ========================================

REM Run with simplified classpath approach
set CLASSPATH=target\classes;target\dependency\*

REM Run with classpath instead of JAR
java -Xmx2g -cp "%CLASSPATH%" org.jcora.mec.Main "%CONFIG_FILE%"
set SIMULATION_RESULT=%ERRORLEVEL%

echo ========================================
echo         SIMULATION COMPLETED
echo ========================================
echo.

if %SIMULATION_RESULT% equ 0 (
    echo [SUCCESS] Simulation completed successfully!
    echo.
    echo [INFO] Generated files:
    if exist "output\*.csv" (
        echo [INFO] CSV files:
        dir /b output\*.csv 2>nul
    )
    if exist "output\*.png" (
        echo [INFO] Chart files:
        dir /b output\*.png 2>nul
    )
    if exist "output\*.txt" (
        echo [INFO] Report files:
        dir /b output\*.txt 2>nul
    )
    if exist "logs\*.log" (
        echo [INFO] Log files:
        dir /b logs\*.log 2>nul
    )
    echo.
    echo [INFO] Results are available in the output directory.
    echo [INFO] Open output\ folder to view simulation results.
) else (
    echo [ERROR] Simulation failed with exit code: %SIMULATION_RESULT%
    echo [ERROR] Please check the error messages above.
    echo [ERROR] Common issues:
    echo [ERROR] - Invalid configuration file
    echo [ERROR] - Insufficient memory (try increasing -Xmx)
    echo [ERROR] - Missing dependencies
    exit /b %SIMULATION_RESULT%
)

echo.
echo [INFO] Run completed. Check output directory for results.
