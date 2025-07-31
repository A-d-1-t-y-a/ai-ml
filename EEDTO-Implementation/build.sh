#!/bin/bash

echo "Building EEDTO Implementation..."

# Check if Maven is installed
if ! command -v mvn &> /dev/null; then
    echo "Maven is not installed or not in PATH. Please install Maven and try again."
    exit 1
fi

# Create logs directory if it doesn't exist
mkdir -p logs

# Create output directory if it doesn't exist
mkdir -p output

# Build with Maven
mvn clean package

if [ $? -ne 0 ]; then
    echo "Build failed."
    exit 1
fi

echo "Build successful. You can run the simulation using ./run.sh"

# Make run script executable
chmod +x run.sh

exit 0
