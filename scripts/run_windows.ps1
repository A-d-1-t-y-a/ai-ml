# Fog and Edge Computing System - Windows Run Script
# Based on IEEE INFOCOM 2022 Research Paper Implementation

param(
    [string]$Mode = "run",
    [string]$Config = "default",
    [int]$Duration = 300
)

Write-Host "=== FOG AND EDGE COMPUTING SYSTEM ===" -ForegroundColor Green
Write-Host "Based on IEEE INFOCOM 2022 Research Paper" -ForegroundColor Cyan
Write-Host "System Version: 1.0.0" -ForegroundColor Yellow
Write-Host ""

# Check if Java is installed
$javaVersion = java -version 2>&1 | Select-String "version"
if (-not $javaVersion) {
    Write-Host "ERROR: Java is not installed or not in PATH" -ForegroundColor Red
    Write-Host "Please install Java 11 or higher and try again" -ForegroundColor Red
    exit 1
}

Write-Host "Java version found:" -ForegroundColor Green
Write-Host $javaVersion -ForegroundColor White

# Check if Maven is installed
$mavenVersion = mvn -version 2>&1 | Select-String "Apache Maven"
if (-not $mavenVersion) {
    Write-Host "ERROR: Maven is not installed or not in PATH" -ForegroundColor Red
    Write-Host "Please install Maven 3.6+ and try again" -ForegroundColor Red
    exit 1
}

Write-Host "Maven version found:" -ForegroundColor Green
Write-Host $mavenVersion -ForegroundColor White

# Create necessary directories
$directories = @("data", "logs", "graphs", "reports")
foreach ($dir in $directories) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        Write-Host "Created directory: $dir" -ForegroundColor Green
    }
}

# Function to build the project
function Build-Project {
    Write-Host "Building project..." -ForegroundColor Yellow
    mvn clean compile package -DskipTests
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Build failed" -ForegroundColor Red
        exit 1
    }
    Write-Host "Build completed successfully" -ForegroundColor Green
}

# Function to run the system
function Start-System {
    param([string]$ConfigFile, [int]$RunDuration)
    
    Write-Host "Starting Fog and Edge Computing System..." -ForegroundColor Yellow
    Write-Host "Configuration: $ConfigFile" -ForegroundColor Cyan
    Write-Host "Duration: $RunDuration seconds" -ForegroundColor Cyan
    Write-Host ""
    
    # Set system properties
    $env:JAVA_OPTS = "-Xmx2g -Xms1g"
    
    # Run the application
    $startTime = Get-Date
    java -jar target/fog-edge-computing-1.0.0.jar --config=$ConfigFile --duration=$RunDuration
    
    $endTime = Get-Date
    $duration = ($endTime - $startTime).TotalSeconds
    
    Write-Host ""
    Write-Host "System execution completed" -ForegroundColor Green
    Write-Host "Total execution time: $([math]::Round($duration, 2)) seconds" -ForegroundColor Cyan
}

# Function to run tests
function Invoke-Tests {
    Write-Host "Running tests..." -ForegroundColor Yellow
    mvn test
    if ($LASTEXITCODE -ne 0) {
        Write-Host "WARNING: Some tests failed" -ForegroundColor Yellow
    } else {
        Write-Host "All tests passed" -ForegroundColor Green
    }
}

# Function to generate reports
function Generate-Reports {
    Write-Host "Generating reports..." -ForegroundColor Yellow
    
    # Check if Python is available for data visualization
    $pythonVersion = python --version 2>&1
    if ($pythonVersion) {
        Write-Host "Python found: $pythonVersion" -ForegroundColor Green
        
        # Install required Python packages
        Write-Host "Installing Python dependencies..." -ForegroundColor Yellow
        pip install matplotlib pandas numpy seaborn 2>$null
        
        # Generate visualizations
        Write-Host "Generating data visualizations..." -ForegroundColor Yellow
        python scripts/generate_graphs.py 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-Host "Graphs generated successfully" -ForegroundColor Green
        } else {
            Write-Host "Warning: Graph generation failed" -ForegroundColor Yellow
        }
    } else {
        Write-Host "Python not found - skipping graph generation" -ForegroundColor Yellow
    }
    
    # Generate performance report
    Write-Host "Generating performance report..." -ForegroundColor Yellow
    if (Test-Path "data/system") {
        $latestFile = Get-ChildItem "data/system" -Filter "system_metrics_*.csv" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
        if ($latestFile) {
            Write-Host "Latest metrics file: $($latestFile.Name)" -ForegroundColor Cyan
        }
    }
}

# Function to clean up
function Clear-Project {
    Write-Host "Cleaning project..." -ForegroundColor Yellow
    mvn clean
    Write-Host "Clean completed" -ForegroundColor Green
}

# Main execution logic
switch ($Mode.ToLower()) {
    "build" {
        Build-Project
    }
    "test" {
        Build-Project
        Invoke-Tests
    }
    "run" {
        Build-Project
        Start-System -ConfigFile $Config -RunDuration $Duration
    }
    "report" {
        Generate-Reports
    }
    "clean" {
        Clear-Project
    }
    "full" {
        Build-Project
        Invoke-Tests
        Start-System -ConfigFile $Config -RunDuration $Duration
        Generate-Reports
    }
    default {
        Write-Host "Usage: .\run_windows.ps1 [-Mode <mode>] [-Config <config>] [-Duration <seconds>]" -ForegroundColor Yellow
        Write-Host ""
        Write-Host "Modes:" -ForegroundColor Cyan
        Write-Host "  build  - Build the project" -ForegroundColor White
        Write-Host "  test   - Run tests" -ForegroundColor White
        Write-Host "  run    - Run the system (default)" -ForegroundColor White
        Write-Host "  report - Generate reports and graphs" -ForegroundColor White
        Write-Host "  clean  - Clean the project" -ForegroundColor White
        Write-Host "  full   - Build, test, run, and generate reports" -ForegroundColor White
        Write-Host ""
        Write-Host "Examples:" -ForegroundColor Cyan
        Write-Host "  .\run_windows.ps1" -ForegroundColor White
        Write-Host "  .\run_windows.ps1 -Mode full -Duration 600" -ForegroundColor White
        Write-Host "  .\run_windows.ps1 -Mode test" -ForegroundColor White
    }
}

Write-Host ""
Write-Host "=== SCRIPT COMPLETED ===" -ForegroundColor Green 