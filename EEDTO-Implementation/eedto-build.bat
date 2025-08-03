@echo off
setlocal enabledelayedexpansion

echo ===============================================
echo    EEDTO Implementation - Build and Run
echo ===============================================
echo.

:: Create a copy of this file as eedto-build.bat for compatibility
copy "%~f0" "eedto-build.bat" >nul 2>&1

echo [1/8] Checking Maven installation...
where mvn >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo Maven not found! Please install Maven and add it to your PATH.
    exit /b 1
) else (
    echo Maven found!
)

echo [2/8] Checking Java installation...
where java >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo Java not found! Please install Java 11+ and add it to your PATH.
    exit /b 1
) else (
    echo Java found!
)

echo.
echo [3/8] Creating required directories...
if not exist logs mkdir logs
if not exist output mkdir output
if not exist target mkdir target
echo Directories created successfully!

echo.
echo [4/8] Cleaning previous build...
call mvn clean

echo.
echo [5/8] Compiling project...
call mvn compile

echo.
echo [6/8] Copying dependencies...
call mvn dependency:copy-dependencies

echo.
echo [7/8] Creating JAR package...
call mvn package -DskipTests

echo.
echo [8/8] Running simulation...
echo.

:: First try to run from the assembled JAR
if exist "target\eedto-1.0-SNAPSHOT-jar-with-dependencies.jar" (
    echo Running from assembled JAR...
    java -jar "target\eedto-1.0-SNAPSHOT-jar-with-dependencies.jar"
) else (
    :: If assembled JAR doesn't exist, try running from classes with dependencies
    echo Assembled JAR not found, running from classes...
    
    :: Build the classpath with all dependencies
    set CLASSPATH=target\classes
    
    for %%i in (target\dependency\*.jar) do (
        set CLASSPATH=!CLASSPATH!;%%i
    )
    
    :: Run the main class
    java -cp "!CLASSPATH!" com.fog.eedto.SimpleMain
)

echo.
echo ===============================================
echo    Build and Run Process Completed!
echo ===============================================
echo Check the following directories for outputs:
echo - logs\     : Simulation logs and results
echo - output\   : Generated charts and reports
echo - target\   : Compiled classes and JAR files
echo.
pause
