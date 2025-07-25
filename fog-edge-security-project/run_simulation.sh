#!/bin/bash

echo "==================================================="
echo "    SECURE FOG COMPUTING SIMULATION RUNNER"
echo "    Linux Script"
echo "==================================================="
echo ""

echo "Creating directories..."
mkdir -p target/classes

echo "Compiling the project..."
javac -d target/classes -sourcepath src/main/java src/main/java/org/nci/fogedge/SecureFogSimulation.java

if [ $? -ne 0 ]; then
    echo ""
    echo "Compilation failed! Please check the error messages above."
    read -p "Press Enter to continue..."
    exit 1
fi

echo ""
echo "Compilation successful!"
echo ""
echo "Running the simulation..."
echo "==================================================="
echo ""

java -cp target/classes org.nci.fogedge.SecureFogSimulation

if [ $? -ne 0 ]; then
    echo ""
    echo "Simulation failed! Please check the error messages above."
else
    echo ""
    echo "Simulation completed successfully!"
    echo "Check the reports directory for output files."
fi

echo ""
read -p "Press Enter to continue..."
