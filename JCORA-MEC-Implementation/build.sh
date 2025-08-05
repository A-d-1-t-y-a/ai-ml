#!/bin/bash

echo "==================================================="
echo "JCORA-MEC Implementation - Build Script (Linux)"
echo "==================================================="

# Check if Maven is installed
if ! command -v mvn &> /dev/null; then
    echo "Maven is not installed or not in PATH."
    echo "Please install Maven and add it to your PATH."
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p output

# Clean and build the project
echo "Building the project..."
mvn clean package

if [ $? -ne 0 ]; then
    echo "Build failed. Please check the error messages above."
    exit 1
fi

echo "Build completed successfully."
echo "The JAR file is located at: target/jcora-mec-1.0-SNAPSHOT.jar"
echo ""
echo "To run the simulation, use: ./run.sh [config_file]"
echo ""

# Make the run script executable
chmod +x run.sh
