#!/bin/bash

echo "Running EEDTO Simulation..."

# Check if the JAR file exists
if [ ! -f target/EEDTO-Implementation-1.0-SNAPSHOT.jar ]; then
    echo "JAR file not found. Please build the project first using ./build.sh"
    exit 1
fi

# Create logs directory if it doesn't exist
mkdir -p logs

# Create output directory if it doesn't exist
mkdir -p output

# Run the simulation
java -jar target/EEDTO-Implementation-1.0-SNAPSHOT.jar

if [ $? -ne 0 ]; then
    echo "Simulation failed."
    exit 1
fi

echo "Simulation completed successfully. Results are available in the output directory."
echo "Logs are available in the logs directory."

exit 0
