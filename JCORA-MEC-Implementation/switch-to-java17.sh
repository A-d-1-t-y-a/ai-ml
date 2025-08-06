#!/bin/bash

# Script to switch Java version from 11 to 17 on macOS
# For JCORA-MEC project compatibility

echo "====================================="
echo "Java Version Switcher for JCORA-MEC"
echo "====================================="

# Check if running on macOS
if [ "$(uname)" != "Darwin" ]; then
    echo "Error: This script is intended for macOS only."
    exit 1
fi

# Check current Java version
echo "Checking current Java version..."
current_version=$(java -version 2>&1 | awk -F '"' '/version/ {print $2}' | cut -d'.' -f1)
echo "Current Java version: $current_version"

# Check if Java 17 is installed using /usr/libexec/java_home
echo "Checking for Java 17 installation..."
if /usr/libexec/java_home -v 17 &>/dev/null; then
    JAVA17_HOME=$(/usr/libexec/java_home -v 17)
    echo "Found Java 17 at: $JAVA17_HOME"
else
    echo "Java 17 not found. Please install Java 17 first."
    echo "You can download it from: https://www.oracle.com/java/technologies/downloads/#java17"
    echo "Or install via Homebrew: brew install --cask temurin17"
    exit 1
fi

# Update JAVA_HOME in current shell
echo "Setting JAVA_HOME to Java 17..."
export JAVA_HOME=$JAVA17_HOME
echo "JAVA_HOME is now set to: $JAVA_HOME"

# Update PATH to prioritize Java 17
export PATH=$JAVA_HOME/bin:$PATH

# Verify the change
echo "Verifying Java version after change..."
java -version

# Update pom.xml if it exists
if [ -f "pom.xml" ]; then
    echo "Updating pom.xml to use Java 17..."
    # Backup the original pom.xml
    cp pom.xml pom.xml.backup
    
    # Use sed to update Java version in pom.xml
    # This handles both maven.compiler.source/target and java.version properties
    sed -i '' 's/<java.version>11<\/java.version>/<java.version>17<\/java.version>/g' pom.xml
    sed -i '' 's/<maven.compiler.source>11<\/maven.compiler.source>/<maven.compiler.source>17<\/maven.compiler.source>/g' pom.xml
    sed -i '' 's/<maven.compiler.target>11<\/maven.compiler.target>/<maven.compiler.target>17<\/maven.compiler.target>/g' pom.xml
    
    echo "pom.xml updated. Original backed up as pom.xml.backup"
fi

echo ""
echo "====================================="
echo "Java 17 is now active for this terminal session."
echo ""
echo "To make this change permanent, add these lines to your ~/.zshrc or ~/.bash_profile:"
echo "export JAVA_HOME=$JAVA_HOME"
echo "export PATH=\$JAVA_HOME/bin:\$PATH"
echo ""
echo "To verify JCORA-MEC compatibility, run: ./jcora-mec.sh"
echo "====================================="
