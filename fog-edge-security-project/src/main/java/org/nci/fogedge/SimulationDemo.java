package org.nci.fogedge;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

/**
 * A simple demonstration of the Secure Fog Computing Framework
 * This class simulates the behavior of the framework without requiring external dependencies
 */
public class SimulationDemo {
    
    // Wireless types for IoT devices
    public enum WirelessType {
        WIFI(50.0, 0.05),
        BLE(15.0, 0.02),
        LORAWAN(1.0, 0.01);
        
        private final double dataRate;
        private final double energyConsumption;
        
        WirelessType(double dataRate, double energyConsumption) {
            this.dataRate = dataRate;
            this.energyConsumption = energyConsumption;
        }
        
        public double getDataRate() {
            return dataRate;
        }
        
        public double getEnergyConsumption() {
            return energyConsumption;
        }
    }
    
    // Security levels
    public enum SecurityLevel {
        LOW(0.01, 0.05),
        MEDIUM(0.05, 0.15),
        HIGH(0.10, 0.30);
        
        private final double encryptionOverhead;
        private final double energyOverhead;
        
        SecurityLevel(double encryptionOverhead, double energyOverhead) {
            this.encryptionOverhead = encryptionOverhead;
            this.energyOverhead = energyOverhead;
        }
        
        public double getEncryptionOverhead() {
            return encryptionOverhead;
        }
        
        public double getEnergyOverhead() {
            return energyOverhead;
        }
    }
    
    // Simulation parameters
    private boolean securityEnabled;
    private int numIoTDevices;
    private int numEdgeNodes;
    private int numFogNodes;
    private Random random;
    
    // Simulation metrics
    private double totalDataGenerated;
    private double totalDataProcessed;
    private double totalEnergyConsumption;
    private double totalProcessingTime;
    private double totalSecurityOverhead;
    private int detectedAttacks;
    private Map<WirelessType, Integer> wirelessTypeDistribution;
    
    /**
     * Creates a new SimulationDemo instance
     */
    public SimulationDemo() {
        this.random = new Random();
        this.wirelessTypeDistribution = new HashMap<>();
        
        // Initialize wireless type distribution
        for (WirelessType type : WirelessType.values()) {
            wirelessTypeDistribution.put(type, 0);
        }
    }
    
    /**
     * Main method to run the simulation
     * 
     * @param args Command line arguments (not used)
     */
    public static void main(String[] args) {
        System.out.println("Starting Secure Fog Computing Simulation Demo...");
        
        try {
            // Create and start simulation
            SimulationDemo simulation = new SimulationDemo();
            
            // Set simulation parameters
            boolean securityEnabled = true;
            int numIoTDevices = 10;
            int numEdgeNodes = 3;
            int numFogNodes = 1;
            
            // Start simulation
            simulation.startSimulation(securityEnabled, numIoTDevices, numEdgeNodes, numFogNodes);
            
            System.out.println("Simulation completed successfully!");
        } catch (Exception e) {
            System.out.println("Simulation encountered an error: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    /**
     * Starts the simulation with the specified parameters
     * 
     * @param securityEnabled Whether security features are enabled
     * @param numIoTDevices Number of IoT devices to simulate
     * @param numEdgeNodes Number of edge nodes to simulate
     * @param numFogNodes Number of fog nodes to simulate
     */
    public void startSimulation(boolean securityEnabled, int numIoTDevices, int numEdgeNodes, int numFogNodes) {
        System.out.println("Initializing simulation with " + 
                numIoTDevices + " IoT devices, " + 
                numEdgeNodes + " edge nodes, " + 
                numFogNodes + " fog nodes, and security " + 
                (securityEnabled ? "enabled" : "disabled"));
        
        // Set simulation parameters
        this.securityEnabled = securityEnabled;
        this.numIoTDevices = numIoTDevices;
        this.numEdgeNodes = numEdgeNodes;
        this.numFogNodes = numFogNodes;
        
        // Initialize metrics
        this.totalDataGenerated = 0.0;
        this.totalDataProcessed = 0.0;
        this.totalEnergyConsumption = 0.0;
        this.totalProcessingTime = 0.0;
        this.totalSecurityOverhead = 0.0;
        this.detectedAttacks = 0;
        
        // Run the simulation
        runSimulation();
        
        // Process and display results
        processResults();
    }
    
    /**
     * Runs the simulation for a specified duration
     */
    private void runSimulation() {
        // Get simulation duration
        int durationMs = 5000;
        
        System.out.println("Running simulation for " + durationMs + " ms");
        
        // Simulate data generation and processing
        for (int timeStep = 0; timeStep < durationMs; timeStep += 1000) {
            System.out.println("Simulation time: " + timeStep + " ms");
            
            // Simulate IoT devices generating data
            simulateIoTDevices();
            
            // Simulate edge nodes processing data
            simulateEdgeNodes();
            
            // Simulate fog nodes processing data
            simulateFogNodes();
            
            // Simulate security (if enabled)
            if (securityEnabled) {
                simulateSecurity();
            }
            
            // Simulate time passing
            try {
                Thread.sleep(100); // Sleep for 100ms to avoid CPU hogging
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
        
        System.out.println("Simulation completed after " + durationMs + " ms");
    }
    
    /**
     * Simulates IoT devices generating data
     */
    private void simulateIoTDevices() {
        System.out.println("Simulating " + numIoTDevices + " IoT devices generating data...");
        
        for (int i = 0; i < numIoTDevices; i++) {
            // Randomly select wireless type
            WirelessType wirelessType = WirelessType.values()[random.nextInt(WirelessType.values().length)];
            
            // Count wireless type distribution
            wirelessTypeDistribution.put(wirelessType, wirelessTypeDistribution.get(wirelessType) + 1);
            
            // Generate data based on wireless type
            double dataGenerated = wirelessType.getDataRate() * (0.8 + 0.4 * random.nextDouble());
            
            // Calculate energy consumption
            double energyConsumption = wirelessType.getEnergyConsumption() * dataGenerated;
            
            // Add security overhead if enabled
            double securityOverhead = 0.0;
            if (securityEnabled) {
                SecurityLevel securityLevel = SecurityLevel.values()[random.nextInt(SecurityLevel.values().length)];
                securityOverhead = dataGenerated * securityLevel.getEncryptionOverhead();
                energyConsumption += dataGenerated * securityLevel.getEnergyOverhead();
            }
            
            // Update metrics
            totalDataGenerated += dataGenerated;
            totalEnergyConsumption += energyConsumption;
            totalSecurityOverhead += securityOverhead;
        }
        
        System.out.println("IoT devices generated " + String.format("%.2f", totalDataGenerated) + " KB of data");
    }
    
    /**
     * Simulates edge nodes processing data
     */
    private void simulateEdgeNodes() {
        System.out.println("Simulating " + numEdgeNodes + " edge nodes processing data...");
        
        // Calculate data received by edge nodes (from IoT devices)
        double dataReceived = totalDataGenerated;
        
        // Calculate processing time based on data volume
        double processingTime = dataReceived * 0.5; // 0.5 ms per KB
        
        // Calculate data reduction (filtering)
        double dataReduction = 0.3; // 30% reduction
        double dataProcessed = dataReceived * (1.0 - dataReduction);
        
        // Calculate energy consumption
        double energyConsumption = dataReceived * 0.02; // 0.02 J per KB
        
        // Add security overhead if enabled
        double securityOverhead = 0.0;
        if (securityEnabled) {
            securityOverhead = processingTime * 0.15; // 15% overhead
            energyConsumption *= 1.1; // 10% additional energy
        }
        
        // Update metrics
        totalDataProcessed += dataProcessed;
        totalEnergyConsumption += energyConsumption;
        totalProcessingTime += processingTime;
        totalSecurityOverhead += securityOverhead;
        
        System.out.println("Edge nodes processed " + String.format("%.2f", dataProcessed) + " KB of data");
    }
    
    /**
     * Simulates fog nodes processing data
     */
    private void simulateFogNodes() {
        System.out.println("Simulating " + numFogNodes + " fog nodes processing data...");
        
        // Calculate data received by fog nodes (from edge nodes)
        double dataReceived = totalDataProcessed;
        
        // Calculate processing time based on data volume
        double processingTime = dataReceived * 0.3; // 0.3 ms per KB
        
        // Calculate data reduction (analytics)
        double dataReduction = 0.5; // 50% reduction
        double dataProcessed = dataReceived * (1.0 - dataReduction);
        
        // Calculate energy consumption
        double energyConsumption = dataReceived * 0.01; // 0.01 J per KB
        
        // Add security overhead if enabled
        double securityOverhead = 0.0;
        if (securityEnabled) {
            securityOverhead = processingTime * 0.1; // 10% overhead
            energyConsumption *= 1.05; // 5% additional energy
        }
        
        // Update metrics
        totalDataProcessed = dataProcessed; // Replace with final processed data
        totalEnergyConsumption += energyConsumption;
        totalProcessingTime += processingTime;
        totalSecurityOverhead += securityOverhead;
        
        System.out.println("Fog nodes processed " + String.format("%.2f", dataProcessed) + " KB of data");
    }
    
    /**
     * Simulates security operations (intrusion detection, etc.)
     */
    private void simulateSecurity() {
        System.out.println("Simulating security operations...");
        
        // Simulate attack detection
        double attackProbability = 0.05;
        if (random.nextDouble() < attackProbability) {
            detectedAttacks++;
            System.out.println("Security alert: Attack detected!");
        }
    }
    
    /**
     * Processes and displays simulation results
     */
    private void processResults() {
        // Perform analysis
        double energyEfficiency = analyzeEnergyEfficiency();
        double securityOverhead = analyzeSecurityOverhead();
        double dataReduction = analyzeDataReduction();
        
        // Display simulation results
        System.out.println("\nSimulation Results:");
        System.out.println("===========================================");
        System.out.println("Total Data Generated: " + String.format("%.2f", totalDataGenerated) + " KB");
        System.out.println("Total Data Processed: " + String.format("%.2f", totalDataProcessed) + " KB");
        System.out.println("Total Energy Consumption: " + String.format("%.2f", totalEnergyConsumption) + " J");
        System.out.println("Total Processing Time: " + String.format("%.2f", totalProcessingTime) + " ms");
        System.out.println("Total Security Overhead: " + String.format("%.2f", totalSecurityOverhead) + " ms");
        System.out.println("Detected Attacks: " + detectedAttacks);
        System.out.println("===========================================");
        
        // Display analysis results
        System.out.println("\nAnalysis Results:");
        System.out.println("- Energy Efficiency: " + String.format("%.2f", energyEfficiency) + " KB/J");
        System.out.println("- Security Overhead: " + String.format("%.2f", securityOverhead) + "%");
        System.out.println("- Data Reduction: " + String.format("%.2f", dataReduction) + "%");
        
        // Generate detailed report
        String report = generateDetailedReport();
        System.out.println("\nDetailed Report:\n" + report);
    }
    
    /**
     * Analyzes energy efficiency (data processed per unit of energy)
     * 
     * @return Energy efficiency in KB/J
     */
    private double analyzeEnergyEfficiency() {
        if (totalEnergyConsumption <= 0) {
            return 0.0;
        }
        
        double efficiency = totalDataProcessed / totalEnergyConsumption;
        return efficiency;
    }
    
    /**
     * Analyzes security overhead as percentage of total processing time
     * 
     * @return Security overhead percentage
     */
    private double analyzeSecurityOverhead() {
        if (totalProcessingTime <= 0) {
            return 0.0;
        }
        
        double overheadPercentage = (totalSecurityOverhead / totalProcessingTime) * 100.0;
        return overheadPercentage;
    }
    
    /**
     * Analyzes data reduction (ratio of processed data to generated data)
     * 
     * @return Data reduction percentage
     */
    private double analyzeDataReduction() {
        if (totalDataGenerated <= 0) {
            return 0.0;
        }
        
        double reductionPercentage = ((totalDataGenerated - totalDataProcessed) / totalDataGenerated) * 100.0;
        return reductionPercentage;
    }
    
    /**
     * Generates a detailed report of simulation results
     * 
     * @return Detailed report as a string
     */
    private String generateDetailedReport() {
        StringBuilder report = new StringBuilder();
        
        report.append("## Configuration\n");
        report.append("- Security Enabled: ").append(securityEnabled).append("\n");
        report.append("- Total IoT Devices: ").append(numIoTDevices).append("\n");
        report.append("- Total Edge Nodes: ").append(numEdgeNodes).append("\n");
        report.append("- Total Fog Nodes: ").append(numFogNodes).append("\n");
        
        // Add wireless technology distribution
        report.append("- Wireless Technology Distribution:\n");
        for (Map.Entry<WirelessType, Integer> entry : wirelessTypeDistribution.entrySet()) {
            report.append("  - ").append(entry.getKey()).append(": ")
                  .append(entry.getValue()).append(" devices\n");
        }
        report.append("\n");
        
        report.append("## Performance Metrics\n");
        report.append("- Total Data Generated: ").append(String.format("%.2f", totalDataGenerated)).append(" KB\n");
        report.append("- Total Data Processed: ").append(String.format("%.2f", totalDataProcessed)).append(" KB\n");
        report.append("- Total Energy Consumption: ").append(String.format("%.2f", totalEnergyConsumption)).append(" J\n");
        report.append("- Total Processing Time: ").append(String.format("%.2f", totalProcessingTime)).append(" ms\n");
        report.append("- Total Security Overhead: ").append(String.format("%.2f", totalSecurityOverhead)).append(" ms\n\n");
        
        report.append("## Security Metrics\n");
        report.append("- Security Enabled: ").append(securityEnabled).append("\n");
        report.append("- Detected Attacks: ").append(detectedAttacks).append("\n");
        report.append("- Security Overhead Percentage: ").append(String.format("%.2f", analyzeSecurityOverhead())).append("%\n\n");
        
        report.append("## Analysis Results\n");
        report.append("- Energy Efficiency: ").append(String.format("%.2f", analyzeEnergyEfficiency())).append(" KB/J\n");
        report.append("- Data Reduction: ").append(String.format("%.2f", analyzeDataReduction())).append("%\n");
        
        return report.toString();
    }
}
