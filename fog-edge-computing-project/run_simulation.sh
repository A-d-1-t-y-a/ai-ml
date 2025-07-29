#!/bin/bash

echo "========================================================================"
echo "Fog and Edge Computing Simulation - End-to-End Test Script (Linux)"
echo "Based on PureEdgeSim framework"
echo "========================================================================"
echo ""

# Set environment variables
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/"
JAVA_OPTS="-Xmx2g"

# Check if Java is installed
if ! command -v java &> /dev/null; then
    echo "[ERROR] Java is not installed or not in PATH. Please install Java 11 or higher."
    exit 1
fi

# Check Java version
JAVA_VERSION=$(java -version 2>&1 | awk -F '"' '/version/ {print $2}')
echo "[INFO] Using Java version: $JAVA_VERSION"

# Check if Maven is installed
if ! command -v mvn &> /dev/null; then
    echo "[ERROR] Maven is not installed or not in PATH. Please install Maven."
    exit 1
fi

echo "[INFO] Checking for required configuration files..."
MISSING_FILES=0

for CONFIG_FILE in "simulation_parameters.properties" "applications.xml" "edge_devices.xml" "edge_datacenters.xml" "cloud.xml"; do
    if [ ! -f "${PROJECT_DIR}src/main/resources/${CONFIG_FILE}" ]; then
        echo "[ERROR] Missing ${CONFIG_FILE} file."
        MISSING_FILES=1
    fi
done

if [ $MISSING_FILES -eq 1 ]; then
    echo "[ERROR] One or more configuration files are missing. Please check the resources directory."
    exit 1
fi

echo "[INFO] All configuration files found."
echo ""

echo "[INFO] Cleaning and building the project with Maven..."
mvn clean package -DskipTests
if [ $? -ne 0 ]; then
    echo "[ERROR] Maven build failed."
    exit 1
fi
echo "[INFO] Build successful."
echo ""

# Create timestamp for this run
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
RESULTS_DIR="${PROJECT_DIR}simulation_results/${TIMESTAMP}"

echo "[INFO] Creating results directory: ${RESULTS_DIR}"
mkdir -p "${RESULTS_DIR}"

echo "[INFO] Running simulation with default orchestrator..."
echo "[INFO] Timestamp: ${TIMESTAMP}"
echo "[INFO] Results will be saved to: ${RESULTS_DIR}"
echo ""

# Run the simulation
echo "[INFO] Starting simulation execution..."
java ${JAVA_OPTS} -jar "${PROJECT_DIR}target/fog-edge-computing-project-1.0-SNAPSHOT-jar-with-dependencies.jar"
if [ $? -ne 0 ]; then
    echo "[ERROR] Simulation execution failed."
    exit 1
fi

echo ""
echo "[INFO] Simulation completed successfully."
echo "[INFO] Results are available in: ${RESULTS_DIR}"
echo ""

# Run analysis on results
echo "[INFO] Generating summary report..."
cat > "${RESULTS_DIR}/summary_report.txt" << EOL
------------------------------------------
SIMULATION SUMMARY REPORT
Timestamp: ${TIMESTAMP}
------------------------------------------

EOL

# Check if result files exist and add to summary
if [ -f "${RESULTS_DIR}/task_execution_summary.csv" ]; then
    echo "Task Execution Results: Available" >> "${RESULTS_DIR}/summary_report.txt"
    echo "  - Analysis will be performed by the Java application" >> "${RESULTS_DIR}/summary_report.txt"
else
    echo "Task Execution Results: Not available" >> "${RESULTS_DIR}/summary_report.txt"
fi

if [ -f "${RESULTS_DIR}/energy_consumption.csv" ]; then
    echo "Energy Consumption Results: Available" >> "${RESULTS_DIR}/summary_report.txt"
    echo "  - Analysis will be performed by the Java application" >> "${RESULTS_DIR}/summary_report.txt"
else
    echo "Energy Consumption Results: Not available" >> "${RESULTS_DIR}/summary_report.txt"
fi

if [ -f "${RESULTS_DIR}/network_usage.csv" ]; then
    echo "Network Usage Results: Available" >> "${RESULTS_DIR}/summary_report.txt"
    echo "  - Analysis will be performed by the Java application" >> "${RESULTS_DIR}/summary_report.txt"
else
    echo "Network Usage Results: Not available" >> "${RESULTS_DIR}/summary_report.txt"
fi

cat >> "${RESULTS_DIR}/summary_report.txt" << EOL

------------------------------------------
End of Summary Report
------------------------------------------
EOL

echo "[INFO] Summary report generated: ${RESULTS_DIR}/summary_report.txt"
echo ""
echo "[INFO] End-to-End test completed successfully."
echo "========================================================================"

exit 0
