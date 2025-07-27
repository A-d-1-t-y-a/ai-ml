#!/bin/bash

# Build and run script for Fog and Edge Computing Security Simulation
# Based on the paper: "An Overview of Fog Computing and Edge Computing Security and Privacy Issues"
# (Sensors 2021, 21, 8226, https://doi.org/10.3390/s21248226)

echo "===================================================="
echo "Fog and Edge Computing Security Simulation Build Tool"
echo "===================================================="

# Check if Maven is installed
if command -v mvn &> /dev/null; then
    echo "Maven found, using Maven for build..."
    
    # Build with Maven
    mvn clean compile
    
    # Check if build was successful
    if [ $? -eq 0 ]; then
        echo "Maven build successful!"
        echo "Running simulation..."
        
        # Run with Maven
        mvn exec:java -Dexec.mainClass="org.nci.fogedge.SimulationDemo"
        
        exit $?
    else
        echo "Maven build failed. Falling back to direct Java compilation..."
    fi
else
    echo "Maven not found. Using direct Java compilation..."
fi

# Create build directories if they don't exist
mkdir -p target/classes

# Compile Java files
echo "Compiling Java source files..."
find src/main/java -name "*.java" > sources.txt
javac -d target/classes -cp "lib/*" @sources.txt

# Check if compilation was successful
if [ $? -eq 0 ]; then
    echo "Compilation successful!"
    
    # Copy resources
    echo "Copying resources..."
    if [ -d src/main/resources ]; then
        cp -r src/main/resources/* target/classes/
    fi
    
    # Run the simulation
    echo "Running simulation..."
    java -cp "target/classes:lib/*" org.nci.fogedge.SimulationDemo
    
    exit $?
else
    echo "Compilation failed!"
    exit 1
fi
