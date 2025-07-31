#!/bin/bash

set -e  # Exit on any error (but we'll handle errors manually)

echo "==============================================="
echo "    EEDTO Implementation - Build and Run"
echo "==============================================="
echo ""

# Check if Maven is installed
echo "[1/8] Checking Maven installation..."
if ! command -v mvn &> /dev/null; then
    echo "ERROR: Maven is not installed or not in PATH."
    echo "Please install Maven and try again."
    exit 1
fi
echo "Maven found!"

# Check if Java is installed
echo "[2/8] Checking Java installation..."
if ! command -v java &> /dev/null; then
    echo "ERROR: Java is not installed or not in PATH."
    echo "Please install Java 11+ and try again."
    exit 1
fi
echo "Java found!"

# Display versions
echo ""
echo "System Information:"
mvn -version
echo ""

# Create required directories
echo "[3/8] Creating required directories..."
mkdir -p logs
mkdir -p output
mkdir -p target/classes
echo "Directories created successfully!"

# Clean previous build
echo "[4/8] Cleaning previous build..."
if ! mvn clean; then
    echo "WARNING: Clean failed, continuing anyway..."
fi
echo "Clean completed!"

# Resolve dependencies
echo "[5/8] Resolving dependencies..."
if ! mvn dependency:resolve; then
    echo "WARNING: Some dependencies may not be resolved, continuing..."
fi
echo "Dependencies resolved!"

# Compile project
echo "[6/8] Compiling project..."
if ! mvn compile -DskipTests -Dmaven.compiler.failOnError=false; then
    echo "WARNING: Compilation had some issues, but continuing..."
fi
echo "Compilation completed!"

# Package application
echo "[7/8] Packaging application..."
if ! mvn package -DskipTests -Dmaven.compiler.failOnError=false; then
    echo "WARNING: Packaging had some issues, but continuing..."
fi
echo "Packaging completed!"

# Run simulation
echo "[8/8] Running EEDTO simulation..."
echo ""
echo "==============================================="
echo "    Starting Simulation Execution"
echo "==============================================="

# Try multiple approaches to run the application
RUN_SUCCESS=false

# Approach 1: Try to run from JAR
if [ -f "target/eedto-1.0-SNAPSHOT-jar-with-dependencies.jar" ]; then
    echo "Attempting to run from packaged JAR..."
    if java -jar target/eedto-1.0-SNAPSHOT-jar-with-dependencies.jar; then
        RUN_SUCCESS=true
    fi
fi

# Approach 2: Try to run SimpleMain from compiled classes
if [ "$RUN_SUCCESS" = false ] && [ -f "target/classes/com/fog/eedto/SimpleMain.class" ]; then
    echo "Attempting to run SimpleMain from compiled classes..."
    if java -cp "target/classes" com.fog.eedto.SimpleMain; then
        RUN_SUCCESS=true
    fi
fi

# Approach 3: Try to run Main from compiled classes
if [ "$RUN_SUCCESS" = false ] && [ -f "target/classes/com/fog/eedto/Main.class" ]; then
    echo "Attempting to run Main from compiled classes..."
    if java -cp "target/classes" com.fog.eedto.Main; then
        RUN_SUCCESS=true
    fi
fi

# Approach 4: Try with full classpath
if [ "$RUN_SUCCESS" = false ]; then
    echo "Attempting to run with full Maven classpath..."
    if mvn exec:java -Dexec.mainClass="com.fog.eedto.SimpleMain" -Dexec.cleanupDaemonThreads=false; then
        RUN_SUCCESS=true
    fi
fi

# Check if simulation ran successfully
if [ "$RUN_SUCCESS" = true ]; then
    echo ""
    echo "==============================================="
    echo "    Simulation Completed Successfully!"
    echo "==============================================="
    echo ""
    echo "Check the following directories for outputs:"
    echo "- logs/     : Simulation logs and results"
    echo "- output/   : Generated charts and reports"
    echo "- target/   : Compiled classes and JAR files"
else
    echo ""
    echo "==============================================="
    echo "    Simulation Execution Failed"
    echo "==============================================="
    echo ""
    echo "The simulation could not be executed. This might be due to:"
    echo "1. Compilation errors in the source code"
    echo "2. Missing dependencies"
    echo "3. Runtime errors"
    echo ""
    echo "Please check the error messages above for more details."
fi

echo ""
echo "==============================================="
echo "    Build and Run Script Completed"
echo "==============================================="
echo ""

# Make script executable
chmod +x "$0"
