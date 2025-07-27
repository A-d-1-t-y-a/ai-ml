package org.nci.fogedge;

import org.apache.log4j.Level;
import org.cloudbus.cloudsim.Log;
import org.cloudbus.cloudsim.core.CloudSim;
import org.nci.fogedge.model.SimulationResults;
import org.nci.fogedge.security.SecurityManager;
import org.nci.fogedge.security.SecurityLevel;
import org.nci.fogedge.topology.EdgeNode;
import org.nci.fogedge.topology.FogNode;
import org.nci.fogedge.topology.IoTDevice;
import org.nci.fogedge.util.ConfigurationManager;
import org.nci.fogedge.util.LoggingUtil;

import java.util.ArrayList;
import java.util.Calendar;
import java.util.List;
import java.util.Random;

/**
 * Main simulation class for the secure fog computing framework.
 * This class sets up and runs the simulation of a secure fog computing architecture
 * based on the 2021 paper "A Lightweight Security Framework for IoT-Fog-Cloud Architecture".
 */
public class SecureFogSimulation {
    private ConfigurationManager config;
    private SecurityManager securityManager;
    private List<FogNode> fogNodes;
    private List<EdgeNode> edgeNodes;
    private List<IoTDevice> iotDevices;
    private SimulationResults results;
    private Random random;
    
    /**
     * Creates a new SecureFogSimulation instance
     */
    public SecureFogSimulation() {
        // Initialize logging
        LoggingUtil.initializeLogging(true, Level.INFO);
        
        // Load configuration
        this.config = new ConfigurationManager();
        
        // Initialize lists
        this.fogNodes = new ArrayList<>();
        this.edgeNodes = new ArrayList<>();
        this.iotDevices = new ArrayList<>();
        
        // Initialize random number generator
        this.random = new Random();
    }
    
    /**
     * Main method to run the simulation
     * 
     * @param args Command line arguments (not used)
     */
    public static void main(String[] args) {
        Log.printLine("Starting Secure Fog Computing Simulation...");
        
        try {
            // Create and start simulation
            SecureFogSimulation simulation = new SecureFogSimulation();
            
            // Get configuration parameters
            boolean securityEnabled = simulation.config.getBoolean("security.securityEnabled", true);
            int numIoTDevices = simulation.config.getInt("topology.numIoTDevices", 50);
            int numEdgeNodes = simulation.config.getInt("topology.numEdgeNodes", 5);
            int numFogNodes = simulation.config.getInt("topology.numFogNodes", 2);
            
            // Start simulation
            simulation.startSimulation(securityEnabled, numIoTDevices, numEdgeNodes, numFogNodes);
            
            Log.printLine("Simulation completed successfully!");
        } catch (Exception e) {
            Log.printLine("Simulation encountered an error: " + e.getMessage());
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
        // Initialize CloudSim
        int numUser = 1;
        Calendar calendar = Calendar.getInstance();
        boolean traceFlag = false;
        CloudSim.init(numUser, calendar, traceFlag);
        
        Log.printLine("Initializing simulation with " + 
                numIoTDevices + " IoT devices, " + 
                numEdgeNodes + " edge nodes, " + 
                numFogNodes + " fog nodes, and security " + 
                (securityEnabled ? "enabled" : "disabled"));
        
        // Create security manager
        this.securityManager = new SecurityManager(securityEnabled);
        
        // Create results collector
        this.results = new SimulationResults(securityEnabled);
        
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
        
        Log.printLine("Created topology with " + 
                fogNodes.size() + " fog nodes, " + 
                edgeNodes.size() + " edge nodes, and " + 
                iotDevices.size() + " IoT devices");
    }
    
    /**
     * Runs the simulation for a specified duration
     */
    private void runSimulation() {
        // Get simulation duration from config
        int durationMs = config.getInt("simulation.durationMs", 10000);
        
        Log.printLine("Running simulation for " + durationMs + " ms");
        
        // Simulate data generation and processing
        for (int timeStep = 0; timeStep < durationMs; timeStep += 100) {
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
                Thread.sleep(1); // Sleep for 1ms to avoid CPU hogging
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
            
            // Log progress every 1000ms
            if (timeStep % 1000 == 0) {
                Log.printLine("Simulation progress: " + timeStep + "/" + durationMs + " ms");
            }
        }
        
        Log.printLine("Simulation completed after " + durationMs + " ms");
    }
    
    /**
     * Processes and displays simulation results
     */
    private void processResults() {
        // Collect metrics from all nodes
        results.collectIoTMetrics(iotDevices);
        results.collectEdgeMetrics(edgeNodes);
        results.collectFogMetrics(fogNodes);
        
        // Set detected attacks (if any)
        results.setDetectedAttacks(securityManager.getDetectedAttacks().size());
        
        // Log results
        results.logResults();
        
        // Perform analysis
        results.analyzeEnergyEfficiency();
        results.analyzeSecurityOverhead();
        results.analyzeDataReduction();
        
        // Generate detailed report
        String report = results.generateDetailedReport();
        Log.printLine("\nDetailed Report:\n" + report);
    }
    
    /**
     * Gets the simulation results
     * 
     * @return Simulation results
     */
    public SimulationResults getResults() {
        return results;
    }
}
