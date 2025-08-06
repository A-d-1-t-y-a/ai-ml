#!/bin/bash
set -e

echo "==================================================="
echo "JCORA-MEC Implementation - Build Script (Linux)"
echo "==================================================="
echo

# Check if Maven is installed
echo "[INFO] Checking for Maven installation..."
if ! command -v mvn &> /dev/null; then
    echo "[ERROR] Maven is not installed or not in PATH."
    echo "[ERROR] Please install Maven and add it to your PATH."
    echo "[ERROR] On Ubuntu/Debian: sudo apt-get install maven"
    echo "[ERROR] On CentOS/RHEL: sudo yum install maven"
    echo "[ERROR] On macOS: brew install maven"
    exit 1
fi
echo "[INFO] Maven found."

# Check if Java is installed
echo "[INFO] Checking for Java installation..."
if ! command -v java &> /dev/null; then
    echo "[ERROR] Java is not installed or not in PATH."
    echo "[ERROR] Please install Java JDK 8 or higher."
    echo "[ERROR] On Ubuntu/Debian: sudo apt-get install openjdk-8-jdk"
    echo "[ERROR] On CentOS/RHEL: sudo yum install java-1.8.0-openjdk-devel"
    echo "[ERROR] On macOS: brew install openjdk@8"
    exit 1
fi
echo "[INFO] Java found."

# Create necessary directories
echo "[INFO] Creating project directories..."
mkdir -p output
mkdir -p logs
mkdir -p target
echo "[INFO] Directories created."

# Clean previous builds
echo "[INFO] Cleaning previous builds..."
rm -f target/*.jar
rm -rf target/classes

# Download dependencies and compile
echo "[INFO] Downloading dependencies and compiling..."
mvn clean compile dependency:copy-dependencies

if [ $? -ne 0 ]; then
    echo "[ERROR] Compilation failed. Please check the error messages above."
    exit 1
fi

# Package the application
echo "[INFO] Packaging application..."
mvn package -DskipTests

if [ $? -ne 0 ]; then
    echo "[ERROR] Packaging failed. Please check the error messages above."
    exit 1
fi

# Verify JAR files were created
JAR_WITH_DEPS="target/JCORA-MEC-Implementation-1.0-SNAPSHOT-jar-with-dependencies.jar"
JAR_SIMPLE="target/JCORA-MEC-Implementation-1.0-SNAPSHOT.jar"

if [ -f "$JAR_WITH_DEPS" ]; then
    echo "[SUCCESS] Build completed successfully!"
    echo "[INFO] JAR with dependencies: $JAR_WITH_DEPS"
    echo "[INFO] Simple JAR: $JAR_SIMPLE"
else
    echo "[ERROR] JAR file was not created. Build may have failed."
    exit 1
fi

echo
echo "[INFO] Build artifacts:"
ls -la target/*.jar
echo
echo "[INFO] To run the simulation, use: ./run.sh [config_file]"
echo "[INFO] Example: ./run.sh config/simulation.properties"
echo
echo "[SUCCESS] Build process completed successfully!"

# Make the run script executable
chmod +x run.sh
echo "[INFO] Made run.sh executable."
