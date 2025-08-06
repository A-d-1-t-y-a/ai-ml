@echo off
setlocal enabledelayedexpansion

echo ===================================================
echo JCORA-MEC Implementation - Build Script (Windows)
echo ===================================================
echo.

REM Check if Maven is installed
echo [INFO] Checking for Maven installation...
where mvn >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Maven is not installed or not in PATH.
    echo [ERROR] Please install Maven and add it to your PATH.
    echo [ERROR] Download from: https://maven.apache.org/download.cgi
    exit /b 1
)
echo [INFO] Maven found.

REM Check if Java is installed
echo [INFO] Checking for Java installation...
where java >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Java is not installed or not in PATH.
    echo [ERROR] Please install Java JDK 8 or higher.
    exit /b 1
)
echo [INFO] Java found.

REM Create necessary directories
echo [INFO] Creating project directories...
if not exist "output" mkdir output
if not exist "logs" mkdir logs
if not exist "target" mkdir target
echo [INFO] Directories created.

REM Clean previous builds
echo [INFO] Cleaning previous builds...
if exist "target\*.jar" del /q "target\*.jar"
if exist "target\classes" rmdir /s /q "target\classes"

REM Download dependencies and compile
echo [INFO] Downloading dependencies and compiling...
call mvn clean compile dependency:copy-dependencies

if %ERRORLEVEL% neq 0 (
    echo [ERROR] Compilation failed. Please check the error messages above.
    exit /b 1
)

REM Package the application
echo [INFO] Packaging application...
call mvn package -DskipTests

if %ERRORLEVEL% neq 0 (
    echo [ERROR] Packaging failed. Please check the error messages above.
    exit /b 1
)

REM Verify JAR files were created
set JAR_WITH_DEPS=target\JCORA-MEC-Implementation-1.0-SNAPSHOT-jar-with-dependencies.jar
set JAR_SIMPLE=target\JCORA-MEC-Implementation-1.0-SNAPSHOT.jar

if exist "%JAR_WITH_DEPS%" (
    echo [SUCCESS] Build completed successfully!
    echo [INFO] JAR with dependencies: %JAR_WITH_DEPS%
    echo [INFO] Simple JAR: %JAR_SIMPLE%
) else (
    echo [ERROR] JAR file was not created. Build may have failed.
    exit /b 1
)

echo.
echo [INFO] Build artifacts:
dir /b target\*.jar
echo.
echo [INFO] To run the simulation, use: run.bat [config_file]
echo [INFO] Example: run.bat config\simulation.properties
echo.
echo [SUCCESS] Build process completed successfully!
