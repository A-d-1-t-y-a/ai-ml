@echo off
echo ===================================================
echo     SECURE FOG COMPUTING SIMULATION RUNNER
echo     Windows Script
echo ===================================================
echo.

echo Creating directories...
if not exist "target\classes" mkdir target\classes

echo Compiling the project...
javac -d target/classes -sourcepath src/main/java src/main/java/org/nci/fogedge/SecureFogSimulation.java

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Compilation failed! Please check the error messages above.
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo Compilation successful!
echo.
echo Running the simulation...
echo ===================================================
echo.

java -cp target/classes org.nci.fogedge.SecureFogSimulation

echo.
if %ERRORLEVEL% NEQ 0 (
    echo Simulation failed! Please check the error messages above.
) else (
    echo Simulation completed successfully!
    echo Check the reports directory for output files.
)

echo.
pause
