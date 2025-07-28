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
        // Initialize the logger with default settings
        SimulationConfig defaultConfig = new SimulationConfig();
        logger = new LogManager(defaultConfig);
        logger.log(LogLevel.INFO, "Starting Fog and Edge Computing Simulation Demo");
        
        // Load configuration
        SimulationConfig config = loadConfiguration();
        if (config == null) {
            logger.log(LogLevel.ERROR, "Failed to load configuration. Exiting.");
            return;
        }
        
        // Display configuration summary
        displayConfigSummary(config);
        
        // Create and run the simulation
        FogEdgeSimulation simulation = new FogEdgeSimulation(config, logger);
        logger.log(LogLevel.INFO, "Initializing simulation...");
        simulation.initialize();
        
        // Run the simulation
        logger.log(LogLevel.INFO, "Running simulation for " + config.getSimulationDuration() + " ticks...");
        SimulationResults results = simulation.run(config.getSimulationDuration());
        
        // Display results summary
        displayResultsSummary(results);
        
        logger.log(LogLevel.INFO, "Simulation demo completed successfully");
    }
    
    /**
     * Loads the simulation configuration from the properties file
     * @return SimulationConfig object or null if loading fails
     */
    private static SimulationConfig loadConfiguration() {
        File configFile = new File(CONFIG_PATH);
        if (!configFile.exists()) {
            logger.log(LogLevel.ERROR, "Configuration file not found: " + CONFIG_PATH);
            return null;
        }
        
        try {
            logger.log(LogLevel.INFO, "Loading configuration from: " + CONFIG_PATH);
            SimulationConfig config = new SimulationConfig(CONFIG_PATH);
            logger.log(LogLevel.INFO, "Configuration loaded successfully");
            return config;
        } catch (Exception e) {
            logger.log(LogLevel.ERROR, "Error loading configuration: " + e.getMessage());
            return null;
        }
    }
    
    /**
     * Displays a summary of the simulation configuration
     * @param config SimulationConfig object
     */
    private static void displayConfigSummary(SimulationConfig config) {
        logger.log(LogLevel.INFO, "=== Simulation Configuration Summary ===");
        logger.log(LogLevel.INFO, "Simulation Duration: " + config.getSimulationDuration() + " ticks");
        logger.log(LogLevel.INFO, "Simulation Area Size: " + config.getSimulationAreaSize() + " x " + config.getSimulationAreaSize());
        logger.log(LogLevel.INFO, "Random Seed: " + config.getRandomSeed());
        
        // Device counts
        logger.log(LogLevel.INFO, "\nDevice Counts:");
        logger.log(LogLevel.INFO, "IoT Devices: " + config.getIoTDeviceCount());
        logger.log(LogLevel.INFO, "Edge Nodes: " + config.getEdgeNodeCount());
        logger.log(LogLevel.INFO, "Fog Nodes: " + config.getFogNodeCount());
        logger.log(LogLevel.INFO, "Cloud Datacenters: " + config.getCloudDatacenterCount());
        
        // IoT device parameters
        logger.log(LogLevel.INFO, "\nIoT Device Parameters:");
        logger.log(LogLevel.INFO, "Mobility Percentage: " + config.getIoTMobilityPercentage() + "%");
        logger.log(LogLevel.INFO, "Battery Capacity: " + config.getIoTBatteryCapacity() + " mAh");
        logger.log(LogLevel.INFO, "CPU Capacity: " + config.getIoTCpuCapacity() + " MIPS");
        logger.log(LogLevel.INFO, "RAM Capacity: " + config.getIoTRamCapacity() + " MB");
        logger.log(LogLevel.INFO, "Storage Capacity: " + config.getIoTStorageCapacity() + " MB");
        logger.log(LogLevel.INFO, "Task Generation Rate: " + config.getIoTTaskGenerationRate() + " tasks/tick");
        logger.log(LogLevel.INFO, "Wireless Types: " + String.join(", ", config.getIoTWirelessTypes()));
        
        // Network parameters
        logger.log(LogLevel.INFO, "\nNetwork Parameters:");
        logger.log(LogLevel.INFO, "Base Bandwidth: " + config.getNetworkBaseBandwidth() + " Mbps");
        logger.log(LogLevel.INFO, "Base Latency: " + config.getNetworkBaseLatency() + " ms");
        logger.log(LogLevel.INFO, "Variability Factor: " + config.getNetworkVariabilityFactor());
        logger.log(LogLevel.INFO, "Congestion Probability: " + config.getNetworkCongestionProbability());
        logger.log(LogLevel.INFO, "Packet Loss Probability: " + config.getNetworkPacketLossProbability());
        logger.log(LogLevel.INFO, "IoT Mesh Network: " + (config.getIoTMeshNetworkEnabled() ? "Enabled" : "Disabled"));
        logger.log(LogLevel.INFO, "Edge Mesh Network: " + (config.getEdgeMeshNetworkEnabled() ? "Enabled" : "Disabled"));
        logger.log(LogLevel.INFO, "Fog Mesh Network: " + (config.getFogMeshNetworkEnabled() ? "Enabled" : "Disabled"));
        
        // Security parameters
        logger.log(LogLevel.INFO, "\nSecurity Parameters:");
        logger.log(LogLevel.INFO, "Attacks Enabled: " + (config.isSecurityAttacksEnabled() ? "Yes" : "No"));
        if (config.isSecurityAttacksEnabled()) {
            logger.log(LogLevel.INFO, "Attack Probability: " + config.getSecurityAttackProbability());
            logger.log(LogLevel.INFO, "Detection Probability: " + config.getSecurityDetectionProbability());
            logger.log(LogLevel.INFO, "Mitigation Probability: " + config.getSecurityMitigationProbability());
            
            List<AttackType> attackTypes = config.getSecurityAttackTypes();
            logger.log(LogLevel.INFO, "Attack Types: " + (attackTypes.isEmpty() ? "None" : 
                attackTypes.stream().map(AttackType::name).reduce((a, b) -> a + ", " + b).orElse("None")));
            
            List<CountermeasureType> countermeasureTypes = config.getSecurityCountermeasureTypes();
            logger.log(LogLevel.INFO, "Countermeasure Types: " + (countermeasureTypes.isEmpty() ? "None" : 
                countermeasureTypes.stream().map(CountermeasureType::name).reduce((a, b) -> a + ", " + b).orElse("None")));
        }
        
        logger.log(LogLevel.INFO, "=== End of Configuration Summary ===\n");
    }
    
    /**
     * Displays a summary of the simulation results
     * @param results SimulationResults object
     */
    private static void displayResultsSummary(SimulationResults results) {
        logger.log(LogLevel.INFO, "=== Simulation Results Summary ===");
        logger.log(LogLevel.INFO, "Total Simulation Time: " + results.getTotalSimulationTime() + " ms");
        logger.log(LogLevel.INFO, "Total Tasks Generated: " + results.getTotalTasksGenerated());
        logger.log(LogLevel.INFO, "Total Tasks Completed: " + results.getTotalTasksCompleted());
        logger.log(LogLevel.INFO, "Total Tasks Failed: " + results.getTotalTasksFailed());
        
        logger.log(LogLevel.INFO, "\nTask Distribution:");
        logger.log(LogLevel.INFO, "Tasks Executed on IoT Devices: " + results.getTasksExecutedOnIoT());
        logger.log(LogLevel.INFO, "Tasks Executed on Edge Nodes: " + results.getTasksExecutedOnEdge());
        logger.log(LogLevel.INFO, "Tasks Executed on Fog Nodes: " + results.getTasksExecutedOnFog());
        logger.log(LogLevel.INFO, "Tasks Executed on Cloud: " + results.getTasksExecutedOnCloud());
        
        logger.log(LogLevel.INFO, "\nNetwork Statistics:");
        logger.log(LogLevel.INFO, "Total Data Transferred: " + results.getTotalDataTransferred() + " MB");
        logger.log(LogLevel.INFO, "Average Network Latency: " + results.getAverageNetworkLatency() + " ms");
        logger.log(LogLevel.INFO, "Network Congestion Events: " + results.getNetworkCongestionEvents());
        logger.log(LogLevel.INFO, "Packet Loss Events: " + results.getPacketLossEvents());
        
        logger.log(LogLevel.INFO, "\nSecurity Statistics:");
        logger.log(LogLevel.INFO, "Total Attack Attempts: " + results.getTotalAttackAttempts());
        logger.log(LogLevel.INFO, "Successful Attacks: " + results.getSuccessfulAttacks());
        logger.log(LogLevel.INFO, "Detected Attacks: " + results.getDetectedAttacks());
        logger.log(LogLevel.INFO, "Mitigated Attacks: " + results.getMitigatedAttacks());
        
        logger.log(LogLevel.INFO, "\nEnergy Consumption:");
        logger.log(LogLevel.INFO, "Total Energy Consumed: " + results.getTotalEnergyConsumed() + " mWh");
        logger.log(LogLevel.INFO, "IoT Energy Consumed: " + results.getIoTEnergyConsumed() + " mWh");
        logger.log(LogLevel.INFO, "Edge Energy Consumed: " + results.getEdgeEnergyConsumed() + " mWh");
        logger.log(LogLevel.INFO, "Fog Energy Consumed: " + results.getFogEnergyConsumed() + " mWh");
        logger.log(LogLevel.INFO, "Cloud Energy Consumed: " + results.getCloudEnergyConsumed() + " mWh");
        
        logger.log(LogLevel.INFO, "=== End of Results Summary ===");
    }
}
