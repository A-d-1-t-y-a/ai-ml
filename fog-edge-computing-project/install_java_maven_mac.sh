#!/bin/bash

# =============================================================================
# Java 17+ and Maven Installation Script for macOS
# =============================================================================
# This script installs Java 17 or higher and Maven on macOS
# Supports both Intel and Apple Silicon (M1/M2) Macs
# =============================================================================

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

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check macOS version
check_macos_version() {
    print_status "Checking macOS version..."
    local os_version=$(sw_vers -productVersion)
    print_status "macOS version: $os_version"
    
    # Check if macOS version is supported (10.14 or higher)
    if [[ $(echo "$os_version" | cut -d. -f1) -lt 10 ]] || \
       ([[ $(echo "$os_version" | cut -d. -f1) -eq 10 ]] && \
        [[ $(echo "$os_version" | cut -d. -f2) -lt 14 ]]); then
        print_error "This script requires macOS 10.14 (Mojave) or higher"
        exit 1
    fi
}

# Function to check architecture
check_architecture() {
    print_status "Checking system architecture..."
    local arch=$(uname -m)
    if [[ "$arch" == "arm64" ]]; then
        print_status "Detected Apple Silicon (M1/M2) Mac"
        ARCH="arm64"
    elif [[ "$arch" == "x86_64" ]]; then
        print_status "Detected Intel Mac"
        ARCH="x86_64"
    else
        print_error "Unsupported architecture: $arch"
        exit 1
    fi
}

# Function to install Homebrew if not present
install_homebrew() {
    if ! command_exists brew; then
        print_status "Installing Homebrew..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        
        # Add Homebrew to PATH for Apple Silicon Macs
        if [[ "$ARCH" == "arm64" ]]; then
            echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
            eval "$(/opt/homebrew/bin/brew shellenv)"
        fi
        
        print_success "Homebrew installed successfully"
    else
        print_status "Homebrew is already installed"
        brew update
    fi
}

# Function to install Java
install_java() {
    print_status "Installing Java 17+..."
    
    # Check if Java is already installed
    if command_exists java; then
        local java_version=$(java -version 2>&1 | head -n 1 | cut -d'"' -f2)
        print_status "Java is already installed: $java_version"
        
        # Check if it's Java 17 or higher
        local major_version=$(echo "$java_version" | cut -d'.' -f1)
        if [[ "$major_version" -ge 17 ]]; then
            print_success "Java $java_version is already installed and meets requirements"
            return 0
        else
            print_warning "Java $java_version is installed but version is below 17"
        fi
    fi
    
    # Install OpenJDK 17 using Homebrew
    print_status "Installing OpenJDK 17..."
    brew install openjdk@17
    
    # Create symbolic link for system-wide access
    print_status "Setting up Java for system-wide access..."
    sudo ln -sfn /opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk /Library/Java/JavaVirtualMachines/openjdk-17.jdk
    
    # Add Java to PATH
    echo 'export PATH="/opt/homebrew/opt/openjdk@17/bin:$PATH"' >> ~/.zshrc
    echo 'export JAVA_HOME="/opt/homebrew/opt/openjdk@17"' >> ~/.zshrc
    
    # Source the profile
    source ~/.zshrc
    
    print_success "Java 17 installed successfully"
}

# Function to install Maven
install_maven() {
    print_status "Installing Maven..."
    
    if command_exists mvn; then
        local maven_version=$(mvn -version 2>&1 | head -n 1 | cut -d' ' -f3)
        print_status "Maven is already installed: $maven_version"
        return 0
    fi
    
    # Install Maven using Homebrew
    brew install maven
    
    print_success "Maven installed successfully"
}

# Function to verify installations
verify_installations() {
    print_status "Verifying installations..."
    
    # Verify Java
    if command_exists java; then
        local java_version=$(java -version 2>&1 | head -n 1 | cut -d'"' -f2)
        local major_version=$(echo "$java_version" | cut -d'.' -f1)
        
        if [[ "$major_version" -ge 17 ]]; then
            print_success "Java verification passed: $java_version"
        else
            print_error "Java verification failed: Version $java_version is below 17"
            exit 1
        fi
    else
        print_error "Java verification failed: java command not found"
        exit 1
    fi
    
    # Verify Maven
    if command_exists mvn; then
        local maven_version=$(mvn -version 2>&1 | head -n 1 | cut -d' ' -f3)
        print_success "Maven verification passed: $maven_version"
    else
        print_error "Maven verification failed: mvn command not found"
        exit 1
    fi
    
    # Verify JAVA_HOME
    if [[ -n "$JAVA_HOME" ]]; then
        print_success "JAVA_HOME is set: $JAVA_HOME"
    else
        print_warning "JAVA_HOME is not set. You may need to restart your terminal or run:"
        echo "export JAVA_HOME=\"/opt/homebrew/opt/openjdk@17\""
    fi
}

# Function to display final information
display_final_info() {
    echo ""
    echo "=============================================================================="
    print_success "Installation completed successfully!"
    echo "=============================================================================="
    echo ""
    echo "Installed versions:"
    echo "  Java: $(java -version 2>&1 | head -n 1 | cut -d'"' -f2)"
    echo "  Maven: $(mvn -version 2>&1 | head -n 1 | cut -d' ' -f3)"
    echo ""
    echo "Important notes:"
    echo "  1. You may need to restart your terminal for all changes to take effect"
    echo "  2. If JAVA_HOME is not set, run: export JAVA_HOME=\"/opt/homebrew/opt/openjdk@17\""
    echo "  3. To test the installation, run: java -version && mvn -version"
    echo ""
    echo "For the fog-edge-computing project:"
    echo "  1. Navigate to the project directory: cd fog-edge-computing-project"
    echo "  2. Build the project: mvn clean compile"
    echo "  3. Run the simulation: mvn exec:java -Dexec.mainClass=\"org.fog.edge.computing.Main\""
    echo ""
    echo "=============================================================================="
}

# Main execution
main() {
    echo "=============================================================================="
    echo "Java 17+ and Maven Installation Script for macOS"
    echo "=============================================================================="
    echo ""
    
    # Check if running as root
    if [[ $EUID -eq 0 ]]; then
        print_error "This script should not be run as root"
        exit 1
    fi
    
    # Check macOS version
    check_macos_version
    
    # Check architecture
    check_architecture
    
    # Install Homebrew
    install_homebrew
    
    # Install Java
    install_java
    
    # Install Maven
    install_maven
    
    # Verify installations
    verify_installations
    
    # Display final information
    display_final_info
}

# Run main function
main "$@"