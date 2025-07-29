@echo off
echo ========================================================================
echo Fog and Edge Computing Simulation - End-to-End Test Script (Windows)
echo Based on PureEdgeSim framework
echo ========================================================================
echo.

REM Set environment variables
set PROJECT_DIR=%~dp0
set JAVA_OPTS=-Xmx2g

REM Check if Java is installed
java -version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Java is not installed or not in PATH. Please install Java 11 or higher.
    exit /b 1
)

REM Check if Maven is installed
mvn -v >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Maven is not installed or not in PATH. Please install Maven.
    exit /b 1
)

echo [INFO] Checking for required configuration files...
if not exist "%PROJECT_DIR%src\main\resources\simulation_parameters.properties" (
    echo [ERROR] Missing simulation_parameters.properties file.
    exit /b 1
)
if not exist "%PROJECT_DIR%src\main\resources\applications.xml" (
    echo [ERROR] Missing applications.xml file.
    exit /b 1
)
if not exist "%PROJECT_DIR%src\main\resources\edge_devices.xml" (
    echo [ERROR] Missing edge_devices.xml file.
    exit /b 1
)
if not exist "%PROJECT_DIR%src\main\resources\edge_datacenters.xml" (
    echo [ERROR] Missing edge_datacenters.xml file.
    exit /b 1
)
if not exist "%PROJECT_DIR%src\main\resources\cloud.xml" (
    echo [ERROR] Missing cloud.xml file.
    exit /b 1
)

echo [INFO] All configuration files found.
echo.

echo [INFO] Cleaning and building the project with Maven...
call mvn clean package -DskipTests
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Maven build failed.
    exit /b 1
)
echo [INFO] Build successful.
echo.

REM Create timestamp for this run
for /f "tokens=2 delims==" %%a in ('wmic OS Get localdatetime /value') do set dt=%%a
set TIMESTAMP=%dt:~0,4%-%dt:~4,2%-%dt:~6,2%_%dt:~8,2%-%dt:~10,2%-%dt:~12,2%
set RESULTS_DIR=%PROJECT_DIR%simulation_results\%TIMESTAMP%

echo [INFO] Creating results directory: %RESULTS_DIR%
mkdir "%RESULTS_DIR%" 2>nul

echo [INFO] Running simulation with default orchestrator...
echo [INFO] Timestamp: %TIMESTAMP%
echo [INFO] Results will be saved to: %RESULTS_DIR%
echo.

REM Run the simulation
echo [INFO] Starting simulation execution...
java %JAVA_OPTS% -jar "%PROJECT_DIR%target\fog-edge-computing-project-1.0-SNAPSHOT-jar-with-dependencies.jar"
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Simulation execution failed.
    exit /b 1
)

echo.
echo [INFO] Simulation completed successfully.
echo [INFO] Results are available in: %RESULTS_DIR%
echo.

REM Run analysis on results
echo [INFO] Generating summary report...
echo ------------------------------------------ > "%RESULTS_DIR%\summary_report.txt"
echo SIMULATION SUMMARY REPORT >> "%RESULTS_DIR%\summary_report.txt"
echo Timestamp: %TIMESTAMP% >> "%RESULTS_DIR%\summary_report.txt"
echo ------------------------------------------ >> "%RESULTS_DIR%\summary_report.txt"
echo. >> "%RESULTS_DIR%\summary_report.txt"

REM Check if result files exist and add to summary
if exist "%RESULTS_DIR%\task_execution_summary.csv" (
    echo Task Execution Results: Available >> "%RESULTS_DIR%\summary_report.txt"
    
    REM Calculate average execution time from CSV (simplified)
    echo   - Analysis will be performed by the Java application >> "%RESULTS_DIR%\summary_report.txt"
) else (
    echo Task Execution Results: Not available >> "%RESULTS_DIR%\summary_report.txt"
)

if exist "%RESULTS_DIR%\energy_consumption.csv" (
    echo Energy Consumption Results: Available >> "%RESULTS_DIR%\summary_report.txt"
    echo   - Analysis will be performed by the Java application >> "%RESULTS_DIR%\summary_report.txt"
) else (
    echo Energy Consumption Results: Not available >> "%RESULTS_DIR%\summary_report.txt"
)

if exist "%RESULTS_DIR%\network_usage.csv" (
    echo Network Usage Results: Available >> "%RESULTS_DIR%\summary_report.txt"
    echo   - Analysis will be performed by the Java application >> "%RESULTS_DIR%\summary_report.txt"
) else (
    echo Network Usage Results: Not available >> "%RESULTS_DIR%\summary_report.txt"
)

echo. >> "%RESULTS_DIR%\summary_report.txt"
echo ------------------------------------------ >> "%RESULTS_DIR%\summary_report.txt"
echo End of Summary Report >> "%RESULTS_DIR%\summary_report.txt"
echo ------------------------------------------ >> "%RESULTS_DIR%\summary_report.txt"

echo [INFO] Summary report generated: %RESULTS_DIR%\summary_report.txt
echo.
echo [INFO] End-to-End test completed successfully.
echo ========================================================================

exit /b 0
