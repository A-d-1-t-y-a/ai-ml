#!/bin/bash

echo "==================================================="
echo "JCORA-MEC Implementation - Run Script (Linux)"
echo "==================================================="

# Check if Java is installed
if ! command -v java &> /dev/null; then
    echo "Java is not installed or not in PATH."
    echo "Please install Java and add it to your PATH."
    exit 1
fi

# Check if the JAR file exists
if [ ! -f "target/jcora-mec-1.0-SNAPSHOT.jar" ]; then
    echo "JAR file not found. Please build the project first using ./build.sh"
    exit 1
fi

# Set default configuration file
CONFIG_FILE="config/simulation.properties"

# Check if a configuration file was provided
if [ ! -z "$1" ]; then
    CONFIG_FILE="$1"
fi

# Check if the configuration file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Configuration file not found: $CONFIG_FILE"
    echo "Using default configuration file: config/simulation.properties"
    CONFIG_FILE="config/simulation.properties"
fi

# Run the simulation
echo "Running JCORA-MEC simulation with configuration: $CONFIG_FILE"
echo ""
java -jar target/jcora-mec-1.0-SNAPSHOT.jar "$CONFIG_FILE"

echo ""
if [ $? -eq 0 ]; then
    echo "Simulation completed successfully."
    echo "Results are available in the output directory."
else
    echo "Simulation failed. Please check the error messages above."
fi
