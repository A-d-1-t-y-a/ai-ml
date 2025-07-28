package com.nci.fogedge;

import com.nci.fogedge.model.SimulationConfig;
import com.nci.fogedge.model.SimulationResults;
import com.nci.fogedge.security.AttackType;
import com.nci.fogedge.security.CountermeasureType;
import com.nci.fogedge.util.LogManager;
import com.nci.fogedge.util.LogManager.LogLevel;

import java.io.File;
import java.util.List;

/**
 * Standalone demonstration class for the Fog and Edge Computing Simulation.
 * This class provides a simple way to run the simulation without any external dependencies.
 * It validates the simulation components and demonstrates their functionality.
 */
public class SimulationDemo {
    private static final String CONFIG_PATH = "src/main/resources/simulation.properties";
    private static LogManager logger;
    
    /**
     * Main method to run the simulation demonstration
     * @param args Command line arguments (not used)
     */
    public static void main(String[] args) {
        // Initialize the logger
        logger = new LogManager(LogLevel.INFO, true, true, "logs/simulation_demo.log");
        logger.info("Starting Fog and Edge Computing Simulation Demo");
        
        // Load configuration
        SimulationConfig config = loadConfiguration();
        if (config == null) {
            logger.error("Failed to load configuration. Exiting.");
            return;
        }
        
        // Display configuration summary
        displayConfigSummary(config);
        
        // Create and run the simulation
        FogEdgeSimulation simulation = new FogEdgeSimulation(config, logger);
        logger.info("Initializing simulation...");
        simulation.initialize();
        
        // Run the simulation
        logger.info("Running simulation for " + config.getSimulationDuration() + " ticks...");
        SimulationResults results = simulation.run();
        
        // Display results summary
        displayResultsSummary(results);
        
        logger.info("Simulation demo completed successfully");
    }
    
    /**
     * Loads the simulation configuration from the properties file
     * @return SimulationConfig object or null if loading fails
     */
    private static SimulationConfig loadConfiguration() {
        File configFile = new File(CONFIG_PATH);
        if (!configFile.exists()) {
            logger.error("Configuration file not found: " + CONFIG_PATH);
            return null;
        }
        
        try {
            logger.info("Loading configuration from: " + CONFIG_PATH);
            SimulationConfig config = new SimulationConfig(CONFIG_PATH);
            logger.info("Configuration loaded successfully");
            return config;
        } catch (Exception e) {
            logger.error("Error loading configuration: " + e.getMessage());
            return null;
        }
    }
    
    /**
     * Displays a summary of the simulation configuration
     * @param config SimulationConfig object
     */
    private static void displayConfigSummary(SimulationConfig config) {
        logger.info("=== Simulation Configuration Summary ===");
        logger.info("Simulation Duration: " + config.getSimulationDuration() + " ticks");
        logger.info("Simulation Area Size: " + config.getSimulationAreaSize() + " x " + config.getSimulationAreaSize());
        logger.info("Random Seed: " + config.getRandomSeed());
        
        // Device counts
        logger.info("\nDevice Counts:");
        logger.info("IoT Devices: " + config.getIoTDeviceCount());
        logger.info("Edge Nodes: " + config.getEdgeNodeCount());
        logger.info("Fog Nodes: " + config.getFogNodeCount());
        logger.info("Cloud Datacenters: " + config.getCloudDatacenterCount());
        
        // IoT device parameters
        logger.info("\nIoT Device Parameters:");
        logger.info("Mobility Percentage: " + config.getIoTMobilityPercentage() + "%");
        logger.info("Battery Capacity: " + config.getIoTBatteryCapacity() + " mAh");
        logger.info("CPU Capacity: " + config.getIoTCpuCapacity() + " MIPS");
        logger.info("RAM Capacity: " + config.getIoTRamCapacity() + " MB");
        logger.info("Storage Capacity: " + config.getIoTStorageCapacity() + " MB");
        logger.info("Task Generation Rate: " + config.getIoTTaskGenerationRate() + " tasks/tick");
        logger.info("Wireless Types: " + String.join(", ", config.getIoTWirelessTypes()));
        
        // Network parameters
        logger.info("\nNetwork Parameters:");
        logger.info("Base Bandwidth: " + config.getNetworkBaseBandwidth() + " Mbps");
        logger.info("Base Latency: " + config.getNetworkBaseLatency() + " ms");
        logger.info("Variability Factor: " + config.getNetworkVariabilityFactor());
        logger.info("Congestion Probability: " + config.getNetworkCongestionProbability());
        logger.info("Packet Loss Probability: " + config.getNetworkPacketLossProbability());
        logger.info("IoT Mesh Network: " + (config.getIoTMeshNetworkEnabled() ? "Enabled" : "Disabled"));
        logger.info("Edge Mesh Network: " + (config.getEdgeMeshNetworkEnabled() ? "Enabled" : "Disabled"));
        logger.info("Fog Mesh Network: " + (config.getFogMeshNetworkEnabled() ? "Enabled" : "Disabled"));
        
        // Security parameters
        logger.info("\nSecurity Parameters:");
        logger.info("Attacks Enabled: " + (config.isSecurityAttacksEnabled() ? "Yes" : "No"));
        if (config.isSecurityAttacksEnabled()) {
            logger.info("Attack Probability: " + config.getSecurityAttackProbability());
            logger.info("Detection Probability: " + config.getSecurityDetectionProbability());
            logger.info("Mitigation Probability: " + config.getSecurityMitigationProbability());
            
            List<AttackType> attackTypes = config.getSecurityAttackTypes();
            logger.info("Attack Types: " + (attackTypes.isEmpty() ? "None" : 
                attackTypes.stream().map(AttackType::name).reduce((a, b) -> a + ", " + b).orElse("None")));
            
            List<CountermeasureType> countermeasureTypes = config.getSecurityCountermeasureTypes();
            logger.info("Countermeasure Types: " + (countermeasureTypes.isEmpty() ? "None" : 
                countermeasureTypes.stream().map(CountermeasureType::name).reduce((a, b) -> a + ", " + b).orElse("None")));
        }
        
        logger.info("=== End of Configuration Summary ===\n");
    }
    
    /**
     * Displays a summary of the simulation results
     * @param results SimulationResults object
     */
    private static void displayResultsSummary(SimulationResults results) {
        logger.info("=== Simulation Results Summary ===");
        logger.info("Total Simulation Time: " + results.getTotalSimulationTime() + " ms");
        logger.info("Total Tasks Generated: " + results.getTotalTasksGenerated());
        logger.info("Total Tasks Completed: " + results.getTotalTasksCompleted());
        logger.info("Total Tasks Failed: " + results.getTotalTasksFailed());
        
        logger.info("\nTask Distribution:");
        logger.info("Tasks Executed on IoT Devices: " + results.getTasksExecutedOnIoT());
        logger.info("Tasks Executed on Edge Nodes: " + results.getTasksExecutedOnEdge());
        logger.info("Tasks Executed on Fog Nodes: " + results.getTasksExecutedOnFog());
        logger.info("Tasks Executed on Cloud: " + results.getTasksExecutedOnCloud());
        
        logger.info("\nNetwork Statistics:");
        logger.info("Total Data Transferred: " + results.getTotalDataTransferred() + " MB");
        logger.info("Average Network Latency: " + results.getAverageNetworkLatency() + " ms");
        logger.info("Network Congestion Events: " + results.getNetworkCongestionEvents());
        logger.info("Packet Loss Events: " + results.getPacketLossEvents());
        
        logger.info("\nSecurity Statistics:");
        logger.info("Total Attack Attempts: " + results.getTotalAttackAttempts());
        logger.info("Successful Attacks: " + results.getSuccessfulAttacks());
        logger.info("Detected Attacks: " + results.getDetectedAttacks());
        logger.info("Mitigated Attacks: " + results.getMitigatedAttacks());
        
        logger.info("\nEnergy Consumption:");
        logger.info("Total Energy Consumed: " + results.getTotalEnergyConsumed() + " mWh");
        logger.info("IoT Energy Consumed: " + results.getIoTEnergyConsumed() + " mWh");
        logger.info("Edge Energy Consumed: " + results.getEdgeEnergyConsumed() + " mWh");
        logger.info("Fog Energy Consumed: " + results.getFogEnergyConsumed() + " mWh");
        logger.info("Cloud Energy Consumed: " + results.getCloudEnergyConsumed() + " mWh");
        
        logger.info("=== End of Results Summary ===");
    }
}
