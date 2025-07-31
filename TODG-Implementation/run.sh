#!/bin/bash

echo "TODG Simulation - Running simulation"

# Check if the JAR file exists
if [ ! -f "target/TODG-Implementation-1.0-SNAPSHOT-jar-with-dependencies.jar" ]; then
    echo "JAR file not found. Please build the project first using ./build.sh"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p output

# Run the simulation
echo "Running TODG simulation..."
java -jar target/TODG-Implementation-1.0-SNAPSHOT-jar-with-dependencies.jar src/main/resources/simulation.properties

if [ $? -ne 0 ]; then
    echo "Simulation failed! Please check the error messages above."
    exit 1
fi

echo "Simulation completed successfully."
echo "Results are available in the output directory."
exit 0
