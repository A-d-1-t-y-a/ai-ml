#!/bin/bash

# Java and Maven Installation Script for Linux
# Supports Ubuntu, Debian, CentOS, RHEL, Fedora, and other distributions
# Author: Generated for Fog-Edge-Computing Project
# Version: 1.0

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to detect Linux distribution
detect_distro() {
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        DISTRO=$ID
        VERSION=$VERSION_ID
    elif [ -f /etc/redhat-release ]; then
        DISTRO="rhel"
        VERSION=$(cat /etc/redhat-release | grep -oE '[0-9]+\.[0-9]+' | head -1)
    else
        DISTRO="unknown"
        VERSION="unknown"
    fi
    print_status "Detected distribution: $DISTRO $VERSION"
}

# Function to check if running as root
check_root() {
    if [ "$EUID" -eq 0 ]; then
        print_warning "Running as root. This is not recommended for security reasons."
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_error "Installation aborted."
            exit 1
        fi
    fi
}

# Function to update package manager
update_packages() {
    print_status "Updating package lists..."
    case $DISTRO in
        ubuntu|debian)
            sudo apt update
            ;;
        centos|rhel|fedora)
            sudo yum update -y || sudo dnf update -y
            ;;
        *)
            print_warning "Unknown distribution. Please update packages manually."
            ;;
    esac
    print_success "Package lists updated"
}

# Function to install Java
install_java() {
    print_status "Installing Java 11..."
    
    case $DISTRO in
        ubuntu|debian)
            # Install OpenJDK 11
            sudo apt install -y openjdk-11-jdk
            JAVA_HOME="/usr/lib/jvm/java-11-openjdk-amd64"
            ;;
        centos|rhel)
            # Install OpenJDK 11
            sudo yum install -y java-11-openjdk-devel || sudo dnf install -y java-11-openjdk-devel
            JAVA_HOME="/usr/lib/jvm/java-11-openjdk"
            ;;
        fedora)
            # Install OpenJDK 11
            sudo dnf install -y java-11-openjdk-devel
            JAVA_HOME="/usr/lib/jvm/java-11-openjdk"
            ;;
        *)
            print_error "Unsupported distribution for automatic Java installation"
            print_status "Please install Java 11 manually and set JAVA_HOME"
            return 1
            ;;
    esac
    
    # Verify Java installation
    if command -v java &> /dev/null; then
        JAVA_VERSION=$(java -version 2>&1 | head -n 1 | cut -d'"' -f2)
        print_success "Java installed successfully: $JAVA_VERSION"
        return 0
    else
        print_error "Java installation failed"
        return 1
    fi
}

# Function to install Maven
install_maven() {
    print_status "Installing Maven..."
    
    # Check if Maven is already installed
    if command -v mvn &> /dev/null; then
        MAVEN_VERSION=$(mvn -version | head -n 1 | cut -d' ' -f3)
        print_success "Maven already installed: $MAVEN_VERSION"
        return 0
    fi
    
    case $DISTRO in
        ubuntu|debian)
            # Install Maven from package manager
            sudo apt install -y maven
            ;;
        centos|rhel|fedora)
            # Install Maven from package manager
            sudo yum install -y maven || sudo dnf install -y maven
            ;;
        *)
            # Manual Maven installation
            install_maven_manual
            ;;
    esac
    
    # Verify Maven installation
    if command -v mvn &> /dev/null; then
        MAVEN_VERSION=$(mvn -version | head -n 1 | cut -d' ' -f3)
        print_success "Maven installed successfully: $MAVEN_VERSION"
        return 0
    else
        print_error "Maven installation failed"
        return 1
    fi
}

# Function to install Maven manually
install_maven_manual() {
    print_status "Installing Maven manually..."
    
    # Create temporary directory
    TEMP_DIR=$(mktemp -d)
    cd "$TEMP_DIR"
    
    # Download Maven
    MAVEN_VERSION="3.9.6"
    MAVEN_URL="https://archive.apache.org/dist/maven/maven-3/$MAVEN_VERSION/binaries/apache-maven-$MAVEN_VERSION-bin.tar.gz"
    
    print_status "Downloading Maven $MAVEN_VERSION..."
    wget -q "$MAVEN_URL" -O maven.tar.gz
    
    if [ ! -f maven.tar.gz ]; then
        print_error "Failed to download Maven"
        return 1
    fi
    
    # Extract Maven
    print_status "Extracting Maven..."
    sudo tar -xzf maven.tar.gz -C /opt/
    
    # Set up environment variables
    MAVEN_HOME="/opt/apache-maven-$MAVEN_VERSION"
    
    # Add to PATH in shell profile
    SHELL_PROFILE="$HOME/.bashrc"
    if [ -f "$HOME/.zshrc" ]; then
        SHELL_PROFILE="$HOME/.zshrc"
    fi
    
    # Add environment variables to shell profile
    echo "" >> "$SHELL_PROFILE"
    echo "# Maven Environment Variables" >> "$SHELL_PROFILE"
    echo "export MAVEN_HOME=$MAVEN_HOME" >> "$SHELL_PROFILE"
    echo "export PATH=\$PATH:\$MAVEN_HOME/bin" >> "$SHELL_PROFILE"
    
    # Source the profile for current session
    export MAVEN_HOME="$MAVEN_HOME"
    export PATH="$PATH:$MAVEN_HOME/bin"
    
    # Clean up
    cd /
    rm -rf "$TEMP_DIR"
    
    print_success "Maven installed manually to $MAVEN_HOME"
}

# Function to set up environment variables
setup_environment() {
    print_status "Setting up environment variables..."
    
    # Determine JAVA_HOME if not set
    if [ -z "$JAVA_HOME" ]; then
        if [ -d "/usr/lib/jvm/java-11-openjdk-amd64" ]; then
            JAVA_HOME="/usr/lib/jvm/java-11-openjdk-amd64"
        elif [ -d "/usr/lib/jvm/java-11-openjdk" ]; then
            JAVA_HOME="/usr/lib/jvm/java-11-openjdk"
        else
            JAVA_HOME=$(dirname $(dirname $(readlink -f $(which java))))
        fi
    fi
    
    # Add to shell profile
    SHELL_PROFILE="$HOME/.bashrc"
    if [ -f "$HOME/.zshrc" ]; then
        SHELL_PROFILE="$HOME/.zshrc"
    fi
    
    # Add JAVA_HOME to shell profile
    if ! grep -q "JAVA_HOME" "$SHELL_PROFILE"; then
        echo "" >> "$SHELL_PROFILE"
        echo "# Java Environment Variables" >> "$SHELL_PROFILE"
        echo "export JAVA_HOME=$JAVA_HOME" >> "$SHELL_PROFILE"
        echo "export PATH=\$PATH:\$JAVA_HOME/bin" >> "$SHELL_PROFILE"
    fi
    
    # Export for current session
    export JAVA_HOME="$JAVA_HOME"
    export PATH="$PATH:$JAVA_HOME/bin"
    
    print_success "Environment variables configured"
}

# Function to verify installation
verify_installation() {
    print_status "Verifying installation..."
    
    # Check Java
    if command -v java &> /dev/null; then
        JAVA_VERSION=$(java -version 2>&1 | head -n 1 | cut -d'"' -f2)
        print_success "Java: $JAVA_VERSION"
    else
        print_error "Java not found"
        return 1
    fi
    
    # Check Maven
    if command -v mvn &> /dev/null; then
        MAVEN_VERSION=$(mvn -version | head -n 1 | cut -d' ' -f3)
        print_success "Maven: $MAVEN_VERSION"
    else
        print_error "Maven not found"
        return 1
    fi
    
    # Check JAVA_HOME
    if [ -n "$JAVA_HOME" ]; then
        print_success "JAVA_HOME: $JAVA_HOME"
    else
        print_warning "JAVA_HOME not set"
    fi
    
    # Check MAVEN_HOME
    if [ -n "$MAVEN_HOME" ]; then
        print_success "MAVEN_HOME: $MAVEN_HOME"
    else
        print_warning "MAVEN_HOME not set"
    fi
    
    print_success "Installation verification completed"
}

# Function to test the fog-edge-computing project
test_project() {
    print_status "Testing fog-edge-computing project..."
    
    # Check if project directory exists
    if [ ! -d "fog-edge-computing-project" ]; then
        print_warning "fog-edge-computing-project directory not found"
        print_status "Please clone or download the project first"
        return 1
    fi
    
    cd fog-edge-computing-project
    
    # Test Maven compilation
    print_status "Testing Maven compilation..."
    if mvn clean compile; then
        print_success "Maven compilation successful"
    else
        print_error "Maven compilation failed"
        return 1
    fi
    
    # Test Maven package
    print_status "Testing Maven package..."
    if mvn package; then
        print_success "Maven package successful"
    else
        print_error "Maven package failed"
        return 1
    fi
    
    # Test running the application
    print_status "Testing application execution..."
    if java -jar target/fog-edge-computing-project-1.0-SNAPSHOT-jar-with-dependencies.jar; then
        print_success "Application execution successful"
    else
        print_error "Application execution failed"
        return 1
    fi
    
    print_success "Project test completed successfully"
}

# Main installation function
main() {
    echo "=========================================="
    echo "Java and Maven Installation Script"
    echo "for Fog-Edge-Computing Project"
    echo "=========================================="
    echo
    
    # Check if running as root
    check_root
    
    # Detect distribution
    detect_distro
    
    # Update packages
    update_packages
    
    # Install Java
    if install_java; then
        print_success "Java installation completed"
    else
        print_error "Java installation failed"
        exit 1
    fi
    
    # Install Maven
    if install_maven; then
        print_success "Maven installation completed"
    else
        print_error "Maven installation failed"
        exit 1
    fi
    
    # Setup environment variables
    setup_environment
    
    # Verify installation
    if verify_installation; then
        print_success "Installation verification completed"
    else
        print_error "Installation verification failed"
        exit 1
    fi
    
    echo
    echo "=========================================="
    print_success "Installation completed successfully!"
    echo "=========================================="
    echo
    
    # Ask if user wants to test the project
    read -p "Do you want to test the fog-edge-computing project? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        test_project
    fi
    
    echo
    print_status "Next steps:"
    echo "1. Restart your terminal or run: source ~/.bashrc"
    echo "2. Clone the fog-edge-computing project"
    echo "3. Run: cd fog-edge-computing-project"
    echo "4. Run: mvn clean package"
    echo "5. Run: java -jar target/fog-edge-computing-project-1.0-SNAPSHOT-jar-with-dependencies.jar"
    echo
    print_success "Happy coding! 🚀"
}

# Run main function
main "$@"