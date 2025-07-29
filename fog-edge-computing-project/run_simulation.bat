@echo off
setlocal EnableDelayedExpansion

echo ========================================================================
echo [INFO] Fog and Edge Computing Simulation - Complete Test Script (Windows)
echo [INFO] Based on PureEdgeSim framework
echo ========================================================================
echo.

REM Set environment variables
set "PROJECT_DIR=%~dp0"
set "JAVA_OPTS=-Xmx2g"

REM Create timestamp using simple format
set "TIMESTAMP=sim_%date:~-4%%date:~-7,2%%date:~-10,2%_%time:~0,2%%time:~3,2%%time:~6,2%"
set "TIMESTAMP=%TIMESTAMP: =0%"
set "TIMESTAMP=%TIMESTAMP:/=-%"
set "TIMESTAMP=%TIMESTAMP::=-%"
set "TIMESTAMP=%TIMESTAMP:.=-%"

REM Create directories
set "RESULTS_DIR=%PROJECT_DIR%simulation_results\%TIMESTAMP%"
set "LOG_DIR=%PROJECT_DIR%logs"
set "LOG_FILE=%LOG_DIR%\%TIMESTAMP%_simulation.log"
set "CONFIG_BACKUP_DIR=%RESULTS_DIR%\configs"

if not exist "%RESULTS_DIR%" mkdir "%RESULTS_DIR%"
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"
if not exist "%CONFIG_BACKUP_DIR%" mkdir "%CONFIG_BACKUP_DIR%"

REM Start logging
echo [INFO] Simulation started at %date% %time% > "%LOG_FILE%"
echo [INFO] Results will be saved to: %RESULTS_DIR% >> "%LOG_FILE%"

echo [INFO] Simulation started at %date% %time%
echo [INFO] Timestamp: %TIMESTAMP%
echo [INFO] Results directory: %PROJECT_DIR%simulation_results\%TIMESTAMP%
echo.

REM Check Java installation
echo [INFO] Checking Java installation...
java -version > "%RESULTS_DIR%\java_version.txt" 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Java is not installed or not in PATH. Please install Java 11 or higher.
    exit /b 1
)
echo [INFO] Java is installed.
type "%RESULTS_DIR%\java_version.txt"
echo.

REM Check Maven installation
echo [INFO] Checking Maven installation...
mvn -v > "%RESULTS_DIR%\maven_version.txt" 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Maven is not installed or not in PATH. Please install Maven.
    exit /b 1
)
echo [INFO] Maven is installed.
echo.

REM Check for required configuration files
echo [INFO] Checking for required configuration files...
set "CONFIG_FILES=simulation_parameters.properties applications.xml edge_devices.xml edge_datacenters.xml cloud_datacenters.xml"

for %%F in (%CONFIG_FILES%) do (
    if not exist "%PROJECT_DIR%src\main\resources\%%F" (
        echo [ERROR] Missing %%F file.
        exit /b 1
    )
    echo [INFO]   Found %%F
    copy "%PROJECT_DIR%src\main\resources\%%F" "%CONFIG_BACKUP_DIR%\" >nul 2>&1
)

echo [INFO] All required configuration files found and copied to results directory.
echo.

REM Build the project
echo [INFO] Building project with Maven - this may take a few minutes...
call mvn clean package -DskipTests > "%RESULTS_DIR%\maven_build.log" 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Maven build failed. See log file for details: %RESULTS_DIR%\maven_build.log
    exit /b 1
)

REM Check if the JAR file was created
if not exist "%PROJECT_DIR%target\fog-edge-computing-project-1.0-SNAPSHOT-jar-with-dependencies.jar" (
    echo [ERROR] JAR file was not created properly.
    exit /b 1
)

for %%F in ("%PROJECT_DIR%target\fog-edge-computing-project-1.0-SNAPSHOT-jar-with-dependencies.jar") do set JAR_SIZE=%%~zF
echo [INFO] JAR file created successfully (Size: !JAR_SIZE! bytes)
echo.

REM Run the simulation
echo [INFO] Starting simulation execution...
echo ========== SIMULATION OUTPUT START ==========

java %JAVA_OPTS% -jar "%PROJECT_DIR%target\fog-edge-computing-project-1.0-SNAPSHOT-jar-with-dependencies.jar" > "%RESULTS_DIR%\simulation_output.txt" 2>&1
set SIMULATION_RESULT=%ERRORLEVEL%

echo ========== SIMULATION OUTPUT END ==========
echo.

if %SIMULATION_RESULT% NEQ 0 (
    echo [ERROR] Simulation execution failed with exit code: %SIMULATION_RESULT%
    exit /b 1
)

echo [INFO] Simulation execution completed successfully.
echo.

REM Copy simulation results
if exist "%PROJECT_DIR%output" (
    xcopy "%PROJECT_DIR%output\*.*" "%RESULTS_DIR%\" /Y /I /Q >nul 2>&1
    echo [INFO] Output files copied to results directory.
)

REM Generate summary report
echo [INFO] Generating summary report...
echo ------------------------------------------ > "%RESULTS_DIR%\summary_report.txt"
echo SIMULATION SUMMARY REPORT >> "%RESULTS_DIR%\summary_report.txt"
echo Timestamp: %TIMESTAMP% >> "%RESULTS_DIR%\summary_report.txt"
echo ------------------------------------------ >> "%RESULTS_DIR%\summary_report.txt"

REM Check for specific result files
for %%F in ("%RESULTS_DIR%\*.csv") do (
    echo File: %%~nxF >> "%RESULTS_DIR%\summary_report.txt"
)

echo ------------------------------------------ >> "%RESULTS_DIR%\summary_report.txt"
echo End of Summary Report >> "%RESULTS_DIR%\summary_report.txt"
echo ------------------------------------------ >> "%RESULTS_DIR%\summary_report.txt"

REM Copy log file to results directory
copy "%LOG_FILE%" "%RESULTS_DIR%\full_execution.log" >nul 2>&1

echo [INFO] End-to-End test completed successfully.
echo [INFO] Results are available in: %RESULTS_DIR%
echo ========================================================================

exit /b 0
