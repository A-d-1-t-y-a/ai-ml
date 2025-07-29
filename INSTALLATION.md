# Java and Maven Installation Guide for Fog-Edge-Computing Project

This guide provides multiple ways to install Java 11 and Maven on Linux systems to run the fog-edge-computing project.

## 🚀 Quick Installation (Recommended)

### Option 1: One-Command Installation
```bash
# Download and run the quick installation script
curl -sSL https://raw.githubusercontent.com/your-repo/install-scripts/main/quick_install.sh | bash
```

### Option 2: Manual Quick Install
```bash
# For Ubuntu/Debian
sudo apt update
sudo apt install -y openjdk-11-jdk maven

# For CentOS/RHEL/Fedora
sudo yum update -y
sudo yum install -y java-11-openjdk-devel maven
# OR for newer systems
sudo dnf install -y java-11-openjdk-devel maven
```

## 🔧 Comprehensive Installation

### Option 1: Full Installation Script
```bash
# Download the comprehensive installation script
wget https://raw.githubusercontent.com/your-repo/install-scripts/main/install_java_maven.sh
chmod +x install_java_maven.sh
./install_java_maven.sh
```

### Option 2: Manual Installation

#### Step 1: Install Java 11

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install -y openjdk-11-jdk
```

**CentOS/RHEL/Fedora:**
```bash
sudo yum update -y
sudo yum install -y java-11-openjdk-devel
# OR
sudo dnf install -y java-11-openjdk-devel
```

**Verify Java installation:**
```bash
java -version
```

#### Step 2: Install Maven

**Ubuntu/Debian:**
```bash
sudo apt install -y maven
```

**CentOS/RHEL/Fedora:**
```bash
sudo yum install -y maven
# OR
sudo dnf install -y maven
```

**Verify Maven installation:**
```bash
mvn -version
```

#### Step 3: Set Environment Variables

Add these lines to your `~/.bashrc` or `~/.zshrc`:

```bash
# Java Environment Variables
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64  # Ubuntu/Debian
# OR
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk        # CentOS/RHEL/Fedora

export PATH=$PATH:$JAVA_HOME/bin
```

**Reload your shell profile:**
```bash
source ~/.bashrc
# OR
source ~/.zshrc
```

## 🧪 Test Your Installation

After installation, test that everything works:

```bash
# Test Java
java -version

# Test Maven
mvn -version

# Test environment variables
echo $JAVA_HOME
```

## 🚀 Run the Fog-Edge-Computing Project

Once Java and Maven are installed:

```bash
# Clone the project (if not already done)
git clone <your-repo-url>
cd fog-edge-computing-project

# Build the project
mvn clean package

# Run the simulation
java -jar target/fog-edge-computing-project-1.0-SNAPSHOT-jar-with-dependencies.jar
```

## 🔧 Troubleshooting

### Common Issues

**1. Java not found:**
```bash
# Check if Java is installed
which java
# If not found, reinstall Java 11
```

**2. Maven not found:**
```bash
# Check if Maven is installed
which mvn
# If not found, reinstall Maven
```

**3. JAVA_HOME not set:**
```bash
# Find Java installation
sudo update-alternatives --config java
# Set JAVA_HOME accordingly
```

**4. Permission denied:**
```bash
# Make scripts executable
chmod +x install_java_maven.sh
chmod +x quick_install.sh
```

### Distribution-Specific Issues

**Ubuntu/Debian:**
- If `openjdk-11-jdk` is not found, try: `sudo apt install -y openjdk-11-jdk`
- If Maven is not found, try: `sudo apt install -y maven`

**CentOS/RHEL/Fedora:**
- If `java-11-openjdk-devel` is not found, enable EPEL repository first
- If Maven is not found, try manual installation from Apache website

## 📋 System Requirements

- **OS:** Linux (Ubuntu 18.04+, Debian 9+, CentOS 7+, RHEL 7+, Fedora 28+)
- **Java:** OpenJDK 11 or higher
- **Maven:** 3.6.0 or higher
- **Memory:** 2GB RAM minimum, 4GB recommended
- **Disk:** 1GB free space

## 🎯 What Gets Installed

- **Java 11 (OpenJDK):** Runtime environment for the fog-edge-computing project
- **Maven 3.9.6:** Build tool for compiling and packaging the project
- **Environment Variables:** JAVA_HOME and PATH configuration
- **Dependencies:** All required libraries and tools

## 📞 Support

If you encounter issues:

1. Check the troubleshooting section above
2. Verify your Linux distribution is supported
3. Ensure you have sudo privileges
4. Check system requirements

## 🎉 Success!

Once installation is complete, you'll see:
- Java version information
- Maven version information
- Ready to run the fog-edge-computing simulation

Happy coding! 🚀