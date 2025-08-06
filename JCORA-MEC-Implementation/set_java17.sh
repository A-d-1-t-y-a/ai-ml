#!/bin/bash
# This script must be sourced, not executed
# Usage: source ./set_java17.sh

# Check if running on macOS
if [ "$(uname)" != "Darwin" ]; then
    echo "Error: This script is intended for macOS only."
    return 1
fi

# Check if Java 17 is installed
if /usr/libexec/java_home -v 17 &>/dev/null; then
    JAVA17_HOME=$(/usr/libexec/java_home -v 17)
    echo "Found Java 17 at: $JAVA17_HOME"
else
    echo "Java 17 not found. Please install Java 17 first."
    echo "You can download it from: https://www.oracle.com/java/technologies/downloads/#java17"
    echo "Or install via Homebrew: brew install --cask temurin17"
    return 1
fi

# Set Java 17 as current Java version
export JAVA_HOME="$JAVA17_HOME"
export PATH="$JAVA_HOME/bin:$PATH"

echo "Java 17 is now active for this terminal session."
java -version

# Update pom.xml if it exists and user confirms
if [ -f "pom.xml" ]; then
    read -p "Do you want to update pom.xml to use Java 17? (y/n): " update_pom
    if [[ "$update_pom" == "y" || "$update_pom" == "Y" ]]; then
        # Backup the original pom.xml
        cp pom.xml pom.xml.backup
        
        # Use sed to update Java version in pom.xml
        sed -i '' 's/<java.version>11<\/java.version>/<java.version>17<\/java.version>/g' pom.xml
        sed -i '' 's/<maven.compiler.source>11<\/maven.compiler.source>/<maven.compiler.source>17<\/maven.compiler.source>/g' pom.xml
        sed -i '' 's/<maven.compiler.target>11<\/maven.compiler.target>/<maven.compiler.target>17<\/maven.compiler.target>/g' pom.xml
        
        echo "pom.xml updated. Original backed up as pom.xml.backup"
    fi
fi

echo ""
echo "To make this change permanent, add these lines to your ~/.zshrc or ~/.bash_profile:"
echo "export JAVA_HOME=\"$JAVA17_HOME\""
echo "export PATH=\"\$JAVA_HOME/bin:\$PATH\""
echo ""
echo "To verify JCORA-MEC compatibility, run: ./jcora-mec.sh"
