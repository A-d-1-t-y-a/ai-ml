#!/bin/bash

# JCORA-MEC Combined Build and Run Script for Linux/macOS
echo "JCORA-MEC Mobile Edge Computing Simulation"
echo "========================================="

# Parse command line arguments
ACTION="both"
CONFIG_FILE="config/simulation.properties"

if [ "$1" = "build" ]; then
    ACTION="build"
elif [ "$1" = "run" ]; then
    ACTION="run"
fi

if [ ! -z "$2" ]; then
    CONFIG_FILE="$2"
fi

# Build section
if [ "$ACTION" = "build" ] || [ "$ACTION" = "both" ]; then
    echo "Building JCORA-MEC project..."
    mvn clean package -DskipTests
    if [ $? -ne 0 ]; then
        echo "Build failed with error code $?"
        exit $?
    fi
    echo "Build completed successfully."
fi

# Run section
if [ "$ACTION" = "run" ] || [ "$ACTION" = "both" ]; then
    echo "Running JCORA-MEC simulation with configuration: $CONFIG_FILE"
    echo ""
    
    # Create output directory if it doesn't exist
    mkdir -p output
    
    # Run the application using the jar with dependencies
    # Check if the JAR exists, if not try with classes directory
    if [ -f "target/JCORA-MEC-Implementation-1.0-SNAPSHOT-jar-with-dependencies.jar" ]; then
        java -cp "target/JCORA-MEC-Implementation-1.0-SNAPSHOT-jar-with-dependencies.jar" org.jcora.mec.Main "$CONFIG_FILE"
    elif [ -d "target/classes" ]; then
        java -cp "target/classes:target/dependency/*" org.jcora.mec.Main "$CONFIG_FILE"
    else
        echo "Error: Could not find compiled classes or JAR file. Please build the project first."
        exit 1
    fi
    if [ $? -ne 0 ]; then
        echo "Simulation failed with error code $?"
        exit $?
    fi
fi

echo ""
echo "JCORA-MEC process completed."
exit 0
