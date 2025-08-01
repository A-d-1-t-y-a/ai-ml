@echo off
setlocal

echo ===============================================
echo    EEDTO Implementation - Build and Run
echo ===============================================
echo.

echo [1/8] Checking Maven installation...
where mvn >nul 2>nul
if %errorlevel% neq 0 (
    echo ERROR: Maven is not installed or not in PATH.
    pause
    exit /b 1
)
echo Maven found!

echo [2/8] Checking Java installation...
where java >nul 2>nul
if %errorlevel% neq 0 (
    echo ERROR: Java is not installed or not in PATH.
    pause
    exit /b 1
)
echo Java found!

echo.
echo [3/8] Creating required directories...
if not exist logs mkdir logs
if not exist output mkdir output
if not exist target mkdir target
echo Directories created successfully!

echo.
echo [4/8] Cleaning previous build...
call mvn clean
echo Clean completed!

echo.
echo [5/8] Resolving dependencies...
call mvn dependency:resolve
echo Dependencies resolved!

echo.
echo [6/8] Compiling project...
call mvn compile -DskipTests
echo Compilation completed!

echo.
echo [7/8] Packaging application...
call mvn package -DskipTests
echo Packaging completed!

echo.
echo [8/8] Running EEDTO simulation...
echo ===============================================
echo    Starting Simulation Execution
echo ===============================================

REM Try to run from JAR first
if exist "target\eedto-1.0-SNAPSHOT-jar-with-dependencies.jar" (
    echo Running from packaged JAR...
    java -jar target\eedto-1.0-SNAPSHOT-jar-with-dependencies.jar
    goto end
)

REM Try to run SimpleMain from compiled classes
if exist "target\classes\com\fog\eedto\SimpleMain.class" (
    echo Running SimpleMain from compiled classes...
    java -cp "target\classes" com.fog.eedto.SimpleMain
    goto end
)

REM Try to run Main from compiled classes
if exist "target\classes\com\fog\eedto\Main.class" (
    echo Running Main from compiled classes...
    java -cp "target\classes" com.fog.eedto.Main
    goto end
)

REM Try with Maven exec as last resort
echo Attempting to run with Maven exec...
call mvn exec:java -Dexec.mainClass="com.fog.eedto.SimpleMain"

:end
echo.
echo ===============================================
echo    Build and Run Process Completed!
echo ===============================================
echo.
echo Check the following directories for outputs:
echo - logs\     : Simulation logs and results
echo - output\   : Generated charts and reports
echo - target\   : Compiled classes and JAR files
echo.
pause
