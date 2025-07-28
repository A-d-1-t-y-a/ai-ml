@echo off
REM Build and Run Script for Fog-Edge Computing Simulation
REM National College of Ireland - H9FEC: Fog and Edge Computing
REM --------------------------------------------------------

echo ===== FOG-EDGE COMPUTING SIMULATION =====
echo Building and running simulation...

REM Set working directory
set PROJECT_DIR=%~dp0
cd %PROJECT_DIR%

REM Create output directories if they don't exist
if not exist "target" mkdir target
if not exist "logs" mkdir logs
if not exist "results" mkdir results

echo.
echo [1/3] Compiling Java files...
javac -d target -sourcepath src/main/java src/main/java/com/nci/fogedge/SimulationDemo.java

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: Compilation failed! Please check the error messages above.
    goto :end
)

echo.
echo [2/3] Creating JAR file...
cd target
jar cfe fog-edge-sim.jar com.nci.fogedge.SimulationDemo com/nci/fogedge/*.class com/nci/fogedge/*/*.class
cd ..

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: JAR creation failed! Please check the error messages above.
    goto :end
)

echo.
echo [3/3] Running simulation...
java -jar target/fog-edge-sim.jar

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: Simulation execution failed! Please check the error messages above.
    goto :end
)

echo.
echo Simulation completed successfully!
echo Results are available in the 'results' directory.
echo Logs are available in the 'logs' directory.

:end
echo.
echo Press any key to exit...
pause > nul
