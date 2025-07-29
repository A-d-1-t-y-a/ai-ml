#!/bin/bash

# Quick Java and Maven Installation Script for Linux
# Simple and fast installation for fog-edge-computing project

echo "🚀 Quick Java and Maven Installation for Fog-Edge-Computing Project"
echo "================================================================"

# Detect distribution
if [ -f /etc/os-release ]; then
    . /etc/os-release
    DISTRO=$ID
else
    DISTRO="unknown"
fi

echo "📦 Detected: $DISTRO"

# Update packages
echo "🔄 Updating packages..."
case $DISTRO in
    ubuntu|debian)
        sudo apt update
        ;;
    centos|rhel|fedora)
        sudo yum update -y 2>/dev/null || sudo dnf update -y 2>/dev/null
        ;;
esac

# Install Java 11
echo "☕ Installing Java 11..."
case $DISTRO in
    ubuntu|debian)
        sudo apt install -y openjdk-11-jdk
        ;;
    centos|rhel|fedora)
        sudo yum install -y java-11-openjdk-devel 2>/dev/null || sudo dnf install -y java-11-openjdk-devel
        ;;
    *)
        echo "❌ Unsupported distribution. Please install Java 11 manually."
        exit 1
        ;;
esac

# Install Maven
echo "📦 Installing Maven..."
case $DISTRO in
    ubuntu|debian)
        sudo apt install -y maven
        ;;
    centos|rhel|fedora)
        sudo yum install -y maven 2>/dev/null || sudo dnf install -y maven
        ;;
    *)
        echo "❌ Unsupported distribution. Please install Maven manually."
        exit 1
        ;;
esac

# Verify installation
echo "✅ Verifying installation..."
java -version
mvn -version

echo ""
echo "🎉 Installation completed!"
echo ""
echo "📋 Next steps:"
echo "1. Clone your fog-edge-computing project"
echo "2. cd fog-edge-computing-project"
echo "3. mvn clean package"
echo "4. java -jar target/fog-edge-computing-project-1.0-SNAPSHOT-jar-with-dependencies.jar"
echo ""
echo "🚀 Ready to run your fog-edge-computing simulation!"