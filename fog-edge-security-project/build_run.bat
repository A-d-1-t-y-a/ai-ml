@echo off
REM Build and run script for Fog and Edge Computing Security Simulation
REM Based on the paper: "An Overview of Fog Computing and Edge Computing Security and Privacy Issues"
REM (Sensors 2021, 21, 8226, https://doi.org/10.3390/s21248226)

echo ====================================================
echo Fog and Edge Computing Security Simulation Build Tool
echo ====================================================

REM Check if Maven is installed
where mvn >nul 2>nul
if %ERRORLEVEL% == 0 (
    echo Maven found, using Maven for build...
    
    REM Build with Maven
    call mvn clean compile
    
    REM Check if build was successful
    if %ERRORLEVEL% == 0 (
        echo Maven build successful!
        echo Running simulation...
        
        REM Run with Maven
        call mvn exec:java -Dexec.mainClass="org.nci.fogedge.SimulationDemo"
        
        exit /b %ERRORLEVEL%
    ) else (
        echo Maven build failed. Falling back to direct Java compilation...
    )
) else (
    echo Maven not found. Using direct Java compilation...
)

REM Create build directories if they don't exist
if not exist target\classes mkdir target\classes

REM Compile Java files
echo Compiling Java source files...
dir /s /b src\main\java\*.java > sources.txt
javac -d target\classes -cp "lib\*" @sources.txt

REM Check if compilation was successful
if %ERRORLEVEL% == 0 (
    echo Compilation successful!
    
    REM Copy resources
    echo Copying resources...
    if exist src\main\resources (
        xcopy /s /y src\main\resources\* target\classes\
    )
    
    REM Run the simulation
    echo Running simulation...
    java -cp "target\classes;lib\*" org.nci.fogedge.SimulationDemo
    
    exit /b %ERRORLEVEL%
) else (
    echo Compilation failed!
    exit /b 1
)
