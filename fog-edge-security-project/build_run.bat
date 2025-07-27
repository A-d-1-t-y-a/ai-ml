@echo off
setlocal enabledelayedexpansion

echo ===================================================
echo Secure Fog Computing Framework - Build and Run Tool
echo ===================================================
echo.

REM Check if Java is installed
where java >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Java not found. Please install Java and add it to your PATH.
    exit /b 1
)

REM Check if Maven is installed
where mvn >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo WARNING: Maven not found. Will use direct Java compilation instead.
    set USE_MAVEN=false
) else (
    set USE_MAVEN=true
)

REM Create directories if they don't exist
if not exist "target" mkdir target
if not exist "target\classes" mkdir target\classes
if not exist "results" mkdir results

echo.
echo [1] Building project...

REM Skip Maven build and use direct Java compilation
set USE_MAVEN=false

if "%USE_MAVEN%"=="false" (
    echo Using direct Java compilation...
    dir /s /b src\main\java\*.java > sources.txt
    javac -d target\classes -cp target\classes @sources.txt
    if %ERRORLEVEL% NEQ 0 (
        echo ERROR: Compilation failed.
        del sources.txt
        exit /b 1
    )
    del sources.txt
)

echo.
echo [2] Running simulation...
echo.

REM Copy resources to target directory
if exist "src\main\resources" (
    xcopy /Y /E /I src\main\resources target\classes\resources
)

REM Run the simulation demo
java -cp target\classes org.nci.fogedge.SimulationDemo

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: Simulation failed to run.
    exit /b 1
) else (
    echo.
    echo ===================================================
    echo Simulation completed successfully!
    echo Results are available in the results directory.
    echo ===================================================
)

endlocal
