#!/bin/bash

# Fog and Edge Computing System - Linux/Mac Run Script
# Based on IEEE INFOCOM 2022 Research Paper Implementation

# Default parameters
MODE="run"
CONFIG="default"
DURATION=300

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)
            MODE="$2"
            shift 2
            ;;
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --duration)
            DURATION="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [--mode MODE] [--config CONFIG] [--duration SECONDS]"
            echo ""
            echo "Modes:"
            echo "  build  - Build the project"
            echo "  test   - Run tests"
            echo "  run    - Run the system (default)"
            echo "  report - Generate reports and graphs"
            echo "  clean  - Clean the project"
            echo "  full   - Build, test, run, and generate reports"
            echo ""
            echo "Examples:"
            echo "  $0"
            echo "  $0 --mode full --duration 600"
            echo "  $0 --mode test"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=== FOG AND EDGE COMPUTING SYSTEM ==="
echo "Based on IEEE INFOCOM 2022 Research Paper"
echo "System Version: 1.0.0"
echo ""

# Check if Java is installed
if ! command -v java &> /dev/null; then
    echo "ERROR: Java is not installed or not in PATH"
    echo "Please install Java 11 or higher and try again"
    exit 1
fi

echo "Java version found:"
java -version 2>&1 | head -n 1

# Check if Maven is installed
if ! command -v mvn &> /dev/null; then
    echo "ERROR: Maven is not installed or not in PATH"
    echo "Please install Maven 3.6+ and try again"
    exit 1
fi

echo "Maven version found:"
mvn -version 2>&1 | head -n 1

# Create necessary directories
DIRECTORIES=("data" "logs" "graphs" "reports")
for dir in "${DIRECTORIES[@]}"; do
    if [ ! -d "$dir" ]; then
        mkdir -p "$dir"
        echo "Created directory: $dir"
    fi
done

# Function to build the project
build_project() {
    echo "Building project..."
    mvn clean compile package -DskipTests
    if [ $? -ne 0 ]; then
        echo "ERROR: Build failed"
        exit 1
    fi
    echo "Build completed successfully"
}

# Function to run the system
start_system() {
    local config_file=$1
    local run_duration=$2
    
    echo "Starting Fog and Edge Computing System..."
    echo "Configuration: $config_file"
    echo "Duration: $run_duration seconds"
    echo ""
    
    # Set system properties
    export JAVA_OPTS="-Xmx2g -Xms1g"
    
    # Run the application
    start_time=$(date +%s)
    java -jar target/fog-edge-computing-1.0.0.jar --config="$config_file" --duration="$run_duration"
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    
    echo ""
    echo "System execution completed"
    echo "Total execution time: ${duration} seconds"
}

# Function to run tests
run_tests() {
    echo "Running tests..."
    mvn test
    if [ $? -ne 0 ]; then
        echo "WARNING: Some tests failed"
    else
        echo "All tests passed"
    fi
}

# Function to generate reports
generate_reports() {
    echo "Generating reports..."
    
    # Check if Python is available for data visualization
    if command -v python3 &> /dev/null; then
        echo "Python found: $(python3 --version)"
        
        # Install required Python packages
        echo "Installing Python dependencies..."
        pip3 install matplotlib pandas numpy seaborn > /dev/null 2>&1
        
        # Generate visualizations
        echo "Generating data visualizations..."
        if [ -f "scripts/generate_graphs.py" ]; then
            python3 scripts/generate_graphs.py
            if [ $? -eq 0 ]; then
                echo "Graphs generated successfully"
            else
                echo "Warning: Graph generation failed"
            fi
        else
            echo "Warning: generate_graphs.py not found"
        fi
    elif command -v python &> /dev/null; then
        echo "Python found: $(python --version)"
        
        # Install required Python packages
        echo "Installing Python dependencies..."
        pip install matplotlib pandas numpy seaborn > /dev/null 2>&1
        
        # Generate visualizations
        echo "Generating data visualizations..."
        if [ -f "scripts/generate_graphs.py" ]; then
            python scripts/generate_graphs.py
            if [ $? -eq 0 ]; then
                echo "Graphs generated successfully"
            else
                echo "Warning: Graph generation failed"
            fi
        else
            echo "Warning: generate_graphs.py not found"
        fi
    else
        echo "Python not found - skipping graph generation"
    fi
    
    # Generate performance report
    echo "Generating performance report..."
    if [ -d "data/system" ]; then
        latest_file=$(ls -t data/system/system_metrics_*.csv 2>/dev/null | head -n 1)
        if [ -n "$latest_file" ]; then
            echo "Latest metrics file: $(basename "$latest_file")"
        fi
    fi
}

# Function to clean up
clean_project() {
    echo "Cleaning project..."
    mvn clean
    echo "Clean completed"
}

# Main execution logic
case $MODE in
    "build")
        build_project
        ;;
    "test")
        build_project
        run_tests
        ;;
    "run")
        build_project
        start_system "$CONFIG" "$DURATION"
        ;;
    "report")
        generate_reports
        ;;
    "clean")
        clean_project
        ;;
    "full")
        build_project
        run_tests
        start_system "$CONFIG" "$DURATION"
        generate_reports
        ;;
    *)
        echo "Unknown mode: $MODE"
        echo "Usage: $0 [--mode MODE] [--config CONFIG] [--duration SECONDS]"
        echo "Use --help for more information"
        exit 1
        ;;
esac

echo ""
echo "=== SCRIPT COMPLETED ===" 