@echo off
echo Building and running Secure Fog Computing Simulation...

REM Create directories for compiled classes
mkdir target\classes 2>nul

REM Compile the project
echo Compiling Java files...
javac -d target/classes src/main/java/org/nci/fogedge/model/*.java src/main/java/org/nci/fogedge/security/*.java src/main/java/org/nci/fogedge/simulation/*.java src/main/java/org/nci/fogedge/topology/*.java src/main/java/org/nci/fogedge/utils/*.java src/main/java/org/nci/fogedge/*.java

REM Check if compilation was successful
if %errorlevel% neq 0 (
    echo Compilation failed!
    exit /b %errorlevel%
)

REM Create reports directory
mkdir reports 2>nul

REM Run the simulation
echo Running simulation...
java -cp target/classes org.nci.fogedge.SecureFogSimulation

echo Done!
