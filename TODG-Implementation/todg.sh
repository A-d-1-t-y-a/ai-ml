#!/bin/bash

echo "TODG Simulation - Build and Run"

ACTION=$1
if [ -z "$ACTION" ]; then
    ACTION="both"
fi

# Check if Maven is installed
if ! command -v mvn &> /dev/null; then
    echo "Maven is not installed or not in PATH. Please install Maven and try again."
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p output

build() {
    echo "Building project with Maven..."
    mvn clean package

    if [ $? -ne 0 ]; then
        echo "Build failed! Please check the error messages above."
        exit 1
    fi

    echo "Build completed successfully."
}

run() {
    # Check if the JAR file exists
    if [ ! -f "target/TODG-Implementation-1.0-SNAPSHOT-jar-with-dependencies.jar" ]; then
        echo "JAR file not found. Please build the project first using ./todg.sh build"
        exit 1
    fi

    echo "Running TODG simulation..."
    java -jar target/TODG-Implementation-1.0-SNAPSHOT-jar-with-dependencies.jar src/main/resources/simulation.properties

    if [ $? -ne 0 ]; then
        echo "Simulation failed! Please check the error messages above."
        exit 1
    fi

    echo "Simulation completed successfully."
    echo "Results are available in the output directory."
}

case "$ACTION" in
    build)
        build
        ;;
    run)
        run
        ;;
    both)
        build
        run
        ;;
    *)
        echo "Invalid action. Use: ./todg.sh [build|run|both]"
        exit 1
        ;;
esac

exit 0
