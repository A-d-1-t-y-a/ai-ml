@echo off
echo TODG Simulation - Build and Run

set ACTION=%1
if "%ACTION%"=="" set ACTION=both

REM Check if Maven is installed
where mvn >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo Maven is not installed or not in PATH. Please install Maven and try again.
    exit /b 1
)

REM Create output directory if it doesn't exist
if not exist "output" mkdir output

if "%ACTION%"=="build" goto build
if "%ACTION%"=="run" goto run
if "%ACTION%"=="both" goto build

echo Invalid action. Use: todg.bat [build|run|both]
exit /b 1

:build
echo Building project with Maven...
call mvn clean package

if %ERRORLEVEL% neq 0 (
    echo Build failed! Please check the error messages above.
    exit /b 1
)

echo Build completed successfully.
if "%ACTION%"=="both" goto run
exit /b 0

:run
REM Check if the JAR file exists
if not exist "target\TODG-Implementation-1.0-SNAPSHOT-jar-with-dependencies.jar" (
    echo JAR file not found. Please build the project first using todg.bat build
    exit /b 1
)

echo Running TODG simulation...
java -jar target\TODG-Implementation-1.0-SNAPSHOT-jar-with-dependencies.jar src\main\resources\simulation.properties

if %ERRORLEVEL% neq 0 (
    echo Simulation failed! Please check the error messages above.
    exit /b 1
)

echo Simulation completed successfully.
echo Results are available in the output directory.
exit /b 0
