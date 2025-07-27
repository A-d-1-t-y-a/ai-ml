#!/bin/bash

echo "==================================================="
echo "Secure Fog Computing Framework - Build and Run Tool"
echo "==================================================="
echo ""

# Check if Java is installed
if ! command -v java &> /dev/null; then
    echo "ERROR: Java not found. Please install Java and add it to your PATH."
    exit 1
fi

# Check if Maven is installed
if ! command -v mvn &> /dev/null; then
    echo "WARNING: Maven not found. Will use direct Java compilation instead."
    USE_MAVEN=false
else
    USE_MAVEN=true
fi

# Create directories if they don't exist
mkdir -p target/classes
mkdir -p results

echo ""
echo "[1] Building project..."

if [ "$USE_MAVEN" = true ]; then
    echo "Using Maven for build..."
    mvn clean compile
    if [ $? -ne 0 ]; then
        echo "ERROR: Maven build failed."
        echo "Falling back to direct Java compilation..."
        USE_MAVEN=false
    fi
fi

if [ "$USE_MAVEN" = false ]; then
    echo "Using direct Java compilation..."
    find src/main/java -name "*.java" > sources.txt
    javac -d target/classes @sources.txt
    if [ $? -ne 0 ]; then
        echo "ERROR: Compilation failed."
        rm sources.txt
        exit 1
    fi
    rm sources.txt
fi

echo ""
echo "[2] Running simulation..."
echo ""

# Copy resources to target directory
if [ -d "src/main/resources" ]; then
    cp -r src/main/resources/* target/classes/ 2>/dev/null || :
fi

# Run the simulation demo
java -cp target/classes org.nci.fogedge.SimulationDemo

if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: Simulation failed to run."
    exit 1
else
    echo ""
    echo "==================================================="
    echo "Simulation completed successfully!"
    echo "Results are available in the results directory."
    echo "==================================================="
fi
