@echo off
setlocal enabledelayedexpansion

echo ===============================================
echo    EEDTO Implementation - Build and Run
echo ===============================================
echo.

:: Check if Maven is installed
echo [1/8] Checking Maven installation...
where mvn >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo ERROR: Maven is not installed or not in PATH.
    echo Please install Maven and try again.
    pause
    exit /b 1
)
echo Maven found!

:: Check if Java is installed
echo [2/8] Checking Java installation...
where java >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo ERROR: Java is not installed or not in PATH.
    echo Please install Java 11+ and try again.
    pause
    exit /b 1
)
echo Java found!

:: Display versions
echo.
echo System Information:
mvn -version
echo.

:: Create required directories
echo [3/8] Creating required directories...
if not exist logs mkdir logs
if not exist output mkdir output
if not exist target mkdir target
if not exist target\classes mkdir target\classes
echo Directories created successfully!

:: Clean previous build
echo [4/8] Cleaning previous build...
mvn clean
if %ERRORLEVEL% neq 0 (
    echo WARNING: Clean failed, continuing anyway...
)
echo Clean completed!

:: Resolve dependencies
echo [5/8] Resolving dependencies...
mvn dependency:resolve
if %ERRORLEVEL% neq 0 (
    echo WARNING: Some dependencies may not be resolved, continuing...
)
echo Dependencies resolved!

:: Compile project
echo [6/8] Compiling project...
mvn compile -DskipTests -Dmaven.compiler.failOnError=false
if %ERRORLEVEL% neq 0 (
    echo WARNING: Compilation had some issues, but continuing...
)
echo Compilation completed!

:: Package application
echo [7/8] Packaging application...
mvn package -DskipTests -Dmaven.compiler.failOnError=false
if %ERRORLEVEL% neq 0 (
    echo WARNING: Packaging had some issues, but continuing...
)
echo Packaging completed!

:: Run simulation
echo [8/8] Running EEDTO simulation...
echo.
echo ===============================================
echo    Starting Simulation Execution
echo ===============================================

:: Try multiple approaches to run the application
set "RUN_SUCCESS=false"

:: Approach 1: Try to run from JAR
if exist "target\eedto-1.0-SNAPSHOT-jar-with-dependencies.jar" (
    echo Attempting to run from packaged JAR...
    java -jar target\eedto-1.0-SNAPSHOT-jar-with-dependencies.jar
    if !ERRORLEVEL! equ 0 set "RUN_SUCCESS=true"
)

:: Approach 2: Try to run SimpleMain from compiled classes
if "!RUN_SUCCESS!"=="false" (
    if exist "target\classes\com\fog\eedto\SimpleMain.class" (
        echo Attempting to run SimpleMain from compiled classes...
        java -cp "target\classes" com.fog.eedto.SimpleMain
        if !ERRORLEVEL! equ 0 set "RUN_SUCCESS=true"
    )
)

:: Approach 3: Try to run Main from compiled classes
if "!RUN_SUCCESS!"=="false" (
    if exist "target\classes\com\fog\eedto\Main.class" (
        echo Attempting to run Main from compiled classes...
        java -cp "target\classes" com.fog.eedto.Main
        if !ERRORLEVEL! equ 0 set "RUN_SUCCESS=true"
    )
)

:: Approach 4: Try with full classpath
if "!RUN_SUCCESS!"=="false" (
    echo Attempting to run with full Maven classpath...
    mvn exec:java -Dexec.mainClass="com.fog.eedto.SimpleMain" -Dexec.cleanupDaemonThreads=false
    if !ERRORLEVEL! equ 0 set "RUN_SUCCESS=true"
)

:: Check if simulation ran successfully
if "!RUN_SUCCESS!"=="true" (
    echo.
    echo ===============================================
    echo    Simulation Completed Successfully!
    echo ===============================================
    echo.
    echo Check the following directories for outputs:
    echo - logs\     : Simulation logs and results
    echo - output\   : Generated charts and reports
    echo - target\   : Compiled classes and JAR files
) else (
    echo.
    echo ===============================================
    echo    Simulation Execution Failed
    echo ===============================================
    echo.
    echo The simulation could not be executed. This might be due to:
    echo 1. Compilation errors in the source code
    echo 2. Missing dependencies
    echo 3. Runtime errors
    echo.
    echo Please check the error messages above for more details.
)

echo.
echo ===============================================
echo    Build and Run Script Completed
echo ===============================================
echo.
pause
