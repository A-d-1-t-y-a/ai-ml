@echo off
echo TODG Simulation - Running simulation

REM Check if the JAR file exists
if not exist "target\TODG-Implementation-1.0-SNAPSHOT-jar-with-dependencies.jar" (
    echo JAR file not found. Please build the project first using build.bat
    exit /b 1
)

REM Create output directory if it doesn't exist
if not exist "output" mkdir output

REM Run the simulation
echo Running TODG simulation...
java -jar target\TODG-Implementation-1.0-SNAPSHOT-jar-with-dependencies.jar src\main\resources\simulation.properties

if %ERRORLEVEL% neq 0 (
    echo Simulation failed! Please check the error messages above.
    exit /b 1
)

echo Simulation completed successfully.
echo Results are available in the output directory.
exit /b 0
