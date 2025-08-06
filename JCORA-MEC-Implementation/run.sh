#!/bin/bash
set -e

echo "==================================================="
echo "JCORA-MEC Implementation - Run Script (Linux)"
echo "==================================================="
echo

# Check if Java is installed
echo "[INFO] Checking for Java installation..."
if ! command -v java &> /dev/null; then
    echo "[ERROR] Java is not installed or not in PATH."
    echo "[ERROR] Please install Java JDK 8 or higher."
    exit 1
fi
echo "[INFO] Java found."

# Check if the JAR file exists
JAR_WITH_DEPS="target/JCORA-MEC-Implementation-1.0-SNAPSHOT-jar-with-dependencies.jar"
JAR_SIMPLE="target/JCORA-MEC-Implementation-1.0-SNAPSHOT.jar"

echo "[INFO] Checking for JAR files..."
if [ -f "$JAR_WITH_DEPS" ]; then
    JAR_FILE="$JAR_WITH_DEPS"
    echo "[INFO] Using JAR with dependencies: $JAR_FILE"
elif [ -f "$JAR_SIMPLE" ]; then
    JAR_FILE="$JAR_SIMPLE"
    echo "[INFO] Using simple JAR: $JAR_FILE"
    echo "[WARNING] This may require additional classpath configuration."
else
    echo "[ERROR] JAR file not found. Please build the project first using ./build.sh"
    echo "[ERROR] Expected: $JAR_WITH_DEPS"
    echo "[ERROR] Or: $JAR_SIMPLE"
    exit 1
fi

# Set default configuration file
CONFIG_FILE="config/simulation.properties"

# Check if a configuration file was provided
if [ ! -z "$1" ]; then
    CONFIG_FILE="$1"
    echo "[INFO] Using provided configuration: $CONFIG_FILE"
else
    echo "[INFO] Using default configuration: $CONFIG_FILE"
fi

# Check if the configuration file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "[WARNING] Configuration file not found: $CONFIG_FILE"
    if [ -f "config/simulation.properties" ]; then
        CONFIG_FILE="config/simulation.properties"
        echo "[INFO] Falling back to default: $CONFIG_FILE"
    else
        echo "[ERROR] No configuration file found. Creating default configuration..."
        mkdir -p config
        cat > config/simulation.properties << EOF
# Default JCORA-MEC Simulation Configuration
simulation.duration=3600
simulation.timestep=1.0
task.generation.probability=0.1
devices.count=10
servers.count=3
EOF
        CONFIG_FILE="config/simulation.properties"
        echo "[INFO] Created default configuration: $CONFIG_FILE"
    fi
fi

# Create output and logs directories
echo "[INFO] Creating output directories..."
mkdir -p output
mkdir -p logs

# Run the simulation
echo
echo "[INFO] Starting JCORA-MEC simulation..."
echo "[INFO] Configuration: $CONFIG_FILE"
echo "[INFO] JAR file: $JAR_FILE"
echo "[INFO] Output directory: output/"
echo "[INFO] Logs directory: logs/"
echo
echo "========================================"
echo "           SIMULATION OUTPUT"
echo "========================================"

set +e  # Allow simulation to fail without exiting script
java -Xmx2g -jar "$JAR_FILE" "$CONFIG_FILE"
SIMULATION_RESULT=$?
set -e

echo "========================================"
echo "         SIMULATION COMPLETED"
echo "========================================"
echo

if [ $SIMULATION_RESULT -eq 0 ]; then
    echo "[SUCCESS] Simulation completed successfully!"
    echo
    echo "[INFO] Generated files:"
    if ls output/*.csv 1> /dev/null 2>&1; then
        echo "[INFO] CSV files:"
        ls -1 output/*.csv
    fi
    if ls output/*.png 1> /dev/null 2>&1; then
        echo "[INFO] Chart files:"
        ls -1 output/*.png
    fi
    if ls output/*.txt 1> /dev/null 2>&1; then
        echo "[INFO] Report files:"
        ls -1 output/*.txt
    fi
    if ls logs/*.log 1> /dev/null 2>&1; then
        echo "[INFO] Log files:"
        ls -1 logs/*.log
    fi
    echo
    echo "[INFO] Results are available in the output directory."
    echo "[INFO] Open output/ folder to view simulation results."
else
    echo "[ERROR] Simulation failed with exit code: $SIMULATION_RESULT"
    echo "[ERROR] Please check the error messages above."
    echo "[ERROR] Common issues:"
    echo "[ERROR] - Invalid configuration file"
    echo "[ERROR] - Insufficient memory (try increasing -Xmx)"
    echo "[ERROR] - Missing dependencies"
    exit $SIMULATION_RESULT
fi

echo
echo "[INFO] Run completed. Check output directory for results."
