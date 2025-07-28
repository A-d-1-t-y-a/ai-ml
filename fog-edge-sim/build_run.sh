#!/bin/bash
# Build and Run Script for Fog-Edge Computing Simulation
# National College of Ireland - H9FEC: Fog and Edge Computing
# --------------------------------------------------------

echo "===== FOG-EDGE COMPUTING SIMULATION ====="
echo "Building and running simulation..."

# Set working directory
PROJECT_DIR=$(dirname "$0")
cd "$PROJECT_DIR"

# Create output directories if they don't exist
mkdir -p target
mkdir -p logs
mkdir -p results

echo
echo "[1/3] Compiling Java files..."
javac -d target -sourcepath src/main/java src/main/java/com/nci/fogedge/SimulationDemo.java

if [ $? -ne 0 ]; then
    echo
    echo "ERROR: Compilation failed! Please check the error messages above."
    exit 1
fi

echo
echo "[2/3] Creating JAR file..."
cd target
jar cfe fog-edge-sim.jar com.nci.fogedge.SimulationDemo com/nci/fogedge/*.class com/nci/fogedge/*/*.class
cd ..

if [ $? -ne 0 ]; then
    echo
    echo "ERROR: JAR creation failed! Please check the error messages above."
    exit 1
fi

echo
echo "[3/3] Running simulation..."
java -jar target/fog-edge-sim.jar

if [ $? -ne 0 ]; then
    echo
    echo "ERROR: Simulation execution failed! Please check the error messages above."
    exit 1
fi

echo
echo "Simulation completed successfully!"
echo "Results are available in the 'results' directory."
echo "Logs are available in the 'logs' directory."
