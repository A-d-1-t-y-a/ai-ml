#!/bin/bash

echo "TODG Simulation - Building project"

# Check if Maven is installed
if ! command -v mvn &> /dev/null; then
    echo "Maven is not installed or not in PATH. Please install Maven and try again."
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p output

# Build the project with Maven
echo "Building project with Maven..."
mvn clean package

if [ $? -ne 0 ]; then
    echo "Build failed! Please check the error messages above."
    exit 1
fi

echo "Build completed successfully."
echo "You can now run the simulation using ./run.sh"
exit 0
