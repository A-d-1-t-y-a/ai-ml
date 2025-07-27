package org.nci.fogedge;

import org.nci.fogedge.security.SecurityLevel;
import org.nci.fogedge.security.SecurityManager;
import org.nci.fogedge.topology.EdgeNode;
import org.nci.fogedge.topology.FogNode;
import org.nci.fogedge.topology.IoTDevice;
import org.nci.fogedge.util.ConfigurationManager;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * A standalone simulation runner that doesn't rely on CloudSim/iFogSim
 * This class demonstrates the functionality of our secure fog computing framework
 */
public class SimulationRunner {
    
    private ConfigurationManager config;
    private SecurityManager securityManager;
    private List<FogNode> fogNodes;
    private List<EdgeNode> edgeNodes;
    private List<IoTDevice> iotDevices;
    private Random random;
    
    // Simulation metrics
    private double totalDataGenerated;
    private double totalDataProcessed;
    private double totalEnergyConsumption;
    private double totalProcessingTime;
    private double totalSecurityOverhead;
    private int detectedAttacks;
    
    /**
     * Creates a new SimulationRunner instance
     */
    public SimulationRunner() {
        // Load configuration
        this.config = new ConfigurationManager();
        
        // Initialize lists
        this.fogNodes = new ArrayList<>();
        this.edgeNodes = new ArrayList<>();
        this.iotDevices = new ArrayList<>();
        
        // Initialize random number generator
        this.random = new Random();
        
        System.out.println("SimulationRunner initialized");
    }
    
    /**
     * Main method to run the simulation
     * 
     * @param args Command line arguments (not used)
     */
    public static void main(String[] args) {
        System.out.println("Starting Secure Fog Computing Simulation...");
        
        try {
            // Create and start simulation
            SimulationRunner simulation = new SimulationRunner();
            
            // Get configuration parameters
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
        
        // Create security manager
        this.securityManager = new SecurityManager(securityEnabled);
        
        // Initialize metrics
        this.totalDataGenerated = 0.0;
        this.totalDataProcessed = 0.0;
        this.totalEnergyConsumption = 0.0;
        this.totalProcessingTime = 0.0;
        this.totalSecurityOverhead = 0.0;
        this.detectedAttacks = 0;
        
        // Create fog computing topology
        createTopology(numIoTDevices, numEdgeNodes, numFogNodes);
        
        // Run the simulation
        runSimulation();
        
        // Process and display results
        processResults();
    }
    
    /**
     * Creates the fog computing topology with the specified number of nodes
     * 
     * @param numIoTDevices Number of IoT devices
     * @param numEdgeNodes Number of edge nodes
     * @param numFogNodes Number of fog nodes
     */
    private void createTopology(int numIoTDevices, int numEdgeNodes, int numFogNodes) {
        // Create fog nodes
        for (int i = 0; i < numFogNodes; i++) {
            FogNode fogNode = new FogNode("fog-" + i, securityManager);
            fogNodes.add(fogNode);
        }
        
        // Create edge nodes and connect to fog nodes
        for (int i = 0; i < numEdgeNodes; i++) {
            // Assign to a fog node (round-robin)
            FogNode parentFog = fogNodes.get(i % fogNodes.size());
            
            EdgeNode edgeNode = new EdgeNode("edge-" + i, parentFog, securityManager);
            edgeNodes.add(edgeNode);
        }
        
        // Create IoT devices and connect to edge nodes
        IoTDevice.WirelessType[] wirelessTypes = IoTDevice.WirelessType.values();
        
        for (int i = 0; i < numIoTDevices; i++) {
            // Assign to an edge node (round-robin)
            EdgeNode parentEdge = edgeNodes.get(i % edgeNodes.size());
            
            // Randomly select wireless type
            IoTDevice.WirelessType wirelessType = wirelessTypes[random.nextInt(wirelessTypes.length)];
            
            IoTDevice device = new IoTDevice("iot-" + i, wirelessType, parentEdge, securityManager);
            iotDevices.add(device);
        }
        
        System.out.println("Created topology with " + 
                fogNodes.size() + " fog nodes, " + 
                edgeNodes.size() + " edge nodes, and " + 
                iotDevices.size() + " IoT devices");
    }
    
    /**
     * Runs the simulation for a specified duration
     */
    private void runSimulation() {
        // Get simulation duration from config
        int durationMs = 5000;
        
        System.out.println("Running simulation for " + durationMs + " ms");
        
        // Simulate data generation and processing
        for (int timeStep = 0; timeStep < durationMs; timeStep += 1000) {
            System.out.println("Simulation time: " + timeStep + " ms");
            
            // Each IoT device generates data
            for (IoTDevice device : iotDevices) {
                // Generate random data size based on wireless type
                int dataSize = 0;
                switch (device.getWirelessType()) {
                    case WIFI:
                        dataSize = 1024 + random.nextInt(1024); // 1-2 KB
                        break;
                    case BLE:
                        dataSize = 256 + random.nextInt(256);   // 256-512 bytes
                        break;
                    case LORAWAN:
                        dataSize = 50 + random.nextInt(50);     // 50-100 bytes
                        break;
                }
                
                // Generate and send data
                device.generateAndSendData(dataSize);
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
     * Processes and displays simulation results
     */
    private void processResults() {
        // Collect metrics from all nodes
        collectIoTMetrics();
        collectEdgeMetrics();
        collectFogMetrics();
        
        // Set detected attacks (if any)
        this.detectedAttacks = securityManager.getDetectedAttacks().size();
        
        // Perform analysis
        double energyEfficiency = analyzeEnergyEfficiency();
        double securityOverhead = analyzeSecurityOverhead();
        double dataReduction = analyzeDataReduction();
        
        // Display analysis results
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
     * Collects metrics from IoT devices
     */
    private void collectIoTMetrics() {
        System.out.println("Collecting metrics from " + iotDevices.size() + " IoT devices");
        
        for (IoTDevice device : iotDevices) {
            this.totalDataGenerated += device.getDataGenerationRate();
            this.totalEnergyConsumption += device.getEnergyConsumption();
            this.totalSecurityOverhead += device.getSecurityOverhead();
        }
    }
    
    /**
     * Collects metrics from edge nodes
     */
    private void collectEdgeMetrics() {
        System.out.println("Collecting metrics from " + edgeNodes.size() + " edge nodes");
        
        for (EdgeNode node : edgeNodes) {
            this.totalDataProcessed += node.getDataProcessed();
            this.totalEnergyConsumption += node.getEnergyConsumption();
            this.totalProcessingTime += node.getProcessingTime();
            this.totalSecurityOverhead += node.getSecurityOverhead();
        }
    }
    
    /**
     * Collects metrics from fog nodes
     */
    private void collectFogMetrics() {
        System.out.println("Collecting metrics from " + fogNodes.size() + " fog nodes");
        
        for (FogNode node : fogNodes) {
            this.totalDataProcessed += node.getDataProcessed();
            this.totalEnergyConsumption += node.getEnergyConsumption();
            this.totalProcessingTime += node.getProcessingTime();
            this.totalSecurityOverhead += node.getSecurityOverhead();
        }
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
        System.out.println("Energy Efficiency: " + String.format("%.2f", efficiency) + " KB/J");
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
        System.out.println("Security Overhead: " + String.format("%.2f", overheadPercentage) + "%");
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
        System.out.println("Data Reduction: " + String.format("%.2f", reductionPercentage) + "%");
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
        report.append("- Security Enabled: ").append(securityManager.isSecurityEnabled()).append("\n");
        report.append("- Total IoT Devices: ").append(iotDevices.size()).append("\n");
        report.append("- Total Edge Nodes: ").append(edgeNodes.size()).append("\n");
        report.append("- Total Fog Nodes: ").append(fogNodes.size()).append("\n");
        
        // Add wireless technology distribution
        report.append("- Wireless Technology Distribution:\n");
        Map<IoTDevice.WirelessType, Integer> wirelessTypeDistribution = new HashMap<>();
        
        // Initialize wireless type distribution
        for (IoTDevice.WirelessType type : IoTDevice.WirelessType.values()) {
            wirelessTypeDistribution.put(type, 0);
        }
        
        // Count wireless type distribution
        for (IoTDevice device : iotDevices) {
            IoTDevice.WirelessType type = device.getWirelessType();
            wirelessTypeDistribution.put(type, wirelessTypeDistribution.get(type) + 1);
        }
        
        for (Map.Entry<IoTDevice.WirelessType, Integer> entry : wirelessTypeDistribution.entrySet()) {
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
        report.append("- Security Enabled: ").append(securityManager.isSecurityEnabled()).append("\n");
        report.append("- Detected Attacks: ").append(detectedAttacks).append("\n");
        report.append("- Security Overhead Percentage: ").append(String.format("%.2f", analyzeSecurityOverhead())).append("%\n\n");
        
        report.append("## Analysis Results\n");
        report.append("- Energy Efficiency: ").append(String.format("%.2f", analyzeEnergyEfficiency())).append(" KB/J\n");
        report.append("- Data Reduction: ").append(String.format("%.2f", analyzeDataReduction())).append("%\n");
        
        return report.toString();
    }
}
