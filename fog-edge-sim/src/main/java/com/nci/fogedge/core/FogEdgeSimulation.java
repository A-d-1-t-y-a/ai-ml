package com.nci.fogedge.core;

import com.nci.fogedge.model.SimulationConfig;
import com.nci.fogedge.model.SimulationResults;
import com.nci.fogedge.network.NetworkModel;
import com.nci.fogedge.devices.DeviceManager;
import com.nci.fogedge.tasks.TaskManager;
import com.nci.fogedge.security.SecurityManager;
import com.nci.fogedge.util.LogManager;

/**
 * Main simulation class for the Fog and Edge Computing Simulation.
 * This class orchestrates the entire simulation process based on the
 * PureEdgeSim framework architecture as described in:
 * "PureEdgeSim: A simulation framework for performance evaluation of cloud, edge and mist computing environments"
 * by Mechalikh et al. (2021)
 */
public class FogEdgeSimulation {
    private SimulationConfig config;
    private SimulationResults results;
    private NetworkModel networkModel;
    private DeviceManager deviceManager;
    private TaskManager taskManager;
    private SecurityManager securityManager;
    private LogManager logManager;
    private boolean isRunning;
    private int currentTick;
    
    /**
     * Constructor for the simulation
     * @param configFilePath Path to the configuration file
     */
    public FogEdgeSimulation(String configFilePath) {
        // Initialize the simulation components
        this.config = new SimulationConfig(configFilePath);
        this.results = new SimulationResults();
        this.logManager = new LogManager(config.getLogLevel());
        this.networkModel = new NetworkModel(config);
        this.deviceManager = new DeviceManager(config, networkModel);
        this.taskManager = new TaskManager(config, results, networkModel, securityManager);
        this.securityManager = new SecurityManager(config);
        this.currentTick = 0;
        this.isRunning = false;
        
        logManager.log("Simulation initialized with config from: " + configFilePath);
    }
    
    /**
     * Starts the simulation
     */
    public void startSimulation() {
        if (isRunning) {
            logManager.log("Simulation is already running");
            return;
        }
        
        logManager.log("Starting simulation...");
        isRunning = true;
        
        // Initialize all components for the simulation
        deviceManager.initialize();
        networkModel.initialize();
        taskManager.initialize();
        securityManager.initialize();
        
        // Run the simulation for the configured duration
        while (currentTick < config.getSimulationDuration() && isRunning) {
            simulationStep();
            currentTick++;
        }
        
        // Finalize the simulation and collect results
        finalizeSimulation();
    }
    
    /**
     * Performs a single simulation step
     */
    private void simulationStep() {
        logManager.log("Simulation step: " + currentTick);
        
        // Update device positions if mobility is enabled
        if (config.isMobilityEnabled()) {
            deviceManager.updateDevicePositions(currentTick);
        }
        
        // Generate new tasks based on the configured task generation rate
        taskManager.generateTasks(currentTick);
        
        // Process tasks (offload, execute, etc.)
        taskManager.processTasks(currentTick);
        
        // Update network conditions
        networkModel.updateNetworkConditions(currentTick);
        
        // Apply security measures and simulate attacks if enabled
        if (config.isSecurityEnabled()) {
            securityManager.applySecurityMeasures(deviceManager.getAllDevices(), currentTick);
            securityManager.simulateAttacks(deviceManager.getAllDevices(), currentTick);
        }
        
        // Collect metrics for this step
        collectMetrics();
    }
    
    /**
     * Collects metrics for the current simulation step
     */
    private void collectMetrics() {
        // Collect performance metrics
        results.updateExecutionTime(taskManager.getAverageExecutionTime());
        results.updateNetworkDelay(networkModel.getAverageNetworkDelay());
        results.updateEnergyConsumption(deviceManager.getTotalEnergyConsumption());
        results.updateResourceUtilization(deviceManager.getAverageResourceUtilization());
        results.updateTaskSuccessRate(taskManager.getTaskSuccessRate());
        
        // Collect security metrics if enabled
        if (config.isSecurityEnabled()) {
            results.updateAttackDetectionRate(securityManager.getAttackDetectionRate());
            results.updateSecurityOverhead(securityManager.getSecurityOverhead());
        }
    }
    
    /**
     * Finalizes the simulation and prepares the results
     */
    private void finalizeSimulation() {
        isRunning = false;
        logManager.log("Simulation completed after " + currentTick + " steps");
        
        // Calculate final metrics
        results.calculateFinalMetrics();
        
        // Log summary of results
        logManager.log("Simulation Results Summary:");
        logManager.log("- Average Execution Time: " + results.getAverageExecutionTime() + " ms");
        logManager.log("- Average Network Delay: " + results.getAverageNetworkDelay() + " ms");
        logManager.log("- Total Energy Consumption: " + results.getTotalEnergyConsumption() + " J");
        logManager.log("- Average Resource Utilization: " + results.getAverageResourceUtilization() + "%");
        logManager.log("- Task Success Rate: " + results.getTaskSuccessRate() + "%");
        
        if (config.isSecurityEnabled()) {
            logManager.log("- Attack Detection Rate: " + results.getAttackDetectionRate() + "%");
            logManager.log("- Security Overhead: " + results.getSecurityOverhead() + "%");
        }
    }
    
    /**
     * Stops the simulation
     */
    public void stopSimulation() {
        logManager.log("Stopping simulation...");
        isRunning = false;
    }
    
    /**
     * Gets the simulation results
     * @return The simulation results
     */
    public SimulationResults getResults() {
        return results;
    }
    
    /**
     * Gets the current simulation tick
     * @return The current tick
     */
    public int getCurrentTick() {
        return currentTick;
    }
    
    /**
     * Checks if the simulation is running
     * @return True if the simulation is running, false otherwise
     */
    public boolean isRunning() {
        return isRunning;
    }
}
