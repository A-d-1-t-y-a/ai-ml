@echo off
setlocal enabledelayedexpansion

:: JCORA-MEC Combined Build and Run Script for Windows
echo JCORA-MEC Mobile Edge Computing Simulation
echo =========================================

:: Parse command line arguments
set ACTION=both
set CONFIG_FILE=config/simulation.properties

if "%1"=="build" (
    set ACTION=build
) else if "%1"=="run" (
    set ACTION=run
)

if not "%2"=="" (
    set CONFIG_FILE=%2
)

:: Build section
if "%ACTION%"=="build" goto build
if "%ACTION%"=="both" goto build
goto run

:build
echo Building JCORA-MEC project...
call mvn clean package -DskipTests
if %ERRORLEVEL% neq 0 (
    echo Build failed with error code %ERRORLEVEL%
    exit /b %ERRORLEVEL%
)
echo Build completed successfully.
if "%ACTION%"=="build" goto end

:run
echo Running JCORA-MEC simulation with configuration: %CONFIG_FILE%
echo.

:: Create output directory if it doesn't exist
if not exist output mkdir output

:: Run the application using the jar with dependencies
call java -cp target/JCORA-MEC-Implementation-1.0-SNAPSHOT-jar-with-dependencies.jar org.jcora.mec.Main %CONFIG_FILE%
if %ERRORLEVEL% neq 0 (
    echo Simulation failed with error code %ERRORLEVEL%
    exit /b %ERRORLEVEL%
)

:end
echo.
echo JCORA-MEC process completed.
exit /b 0
