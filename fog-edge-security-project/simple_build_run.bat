@echo off
echo ===================================================
echo Secure Fog Computing Framework - Simple Build and Run
echo ===================================================
echo.

REM Create output directory if it doesn't exist
if not exist "target\classes" mkdir target\classes

echo Compiling SimulationDemo.java...
javac -d target\classes src\main\java\org\nci\fogedge\SimulationDemo.java

if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Compilation failed.
    exit /b 1
)

echo.
echo Running SimulationDemo...
echo.

java -cp target\classes org.nci.fogedge.SimulationDemo

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: Simulation failed to run.
    exit /b 1
) else (
    echo.
    echo ===================================================
    echo Simulation completed successfully!
    echo ===================================================
)
