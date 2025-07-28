package com.nci.fogedge;

import com.nci.fogedge.devices.*;
import com.nci.fogedge.model.SimulationConfig;
import com.nci.fogedge.model.SimulationResults;
import com.nci.fogedge.network.NetworkModel;
import com.nci.fogedge.security.SecurityManager;
import com.nci.fogedge.tasks.TaskManager;
import com.nci.fogedge.topology.TopologyManager;
import com.nci.fogedge.util.LogManager;

import java.io.FileInputStream;
import java.io.IOException;
import java.util.Properties;

/**
 * Main simulation class for the fog and edge computing simulation.
 * This class coordinates all components of the simulation.
 */
public class FogEdgeSimulation {
    private SimulationConfig config;
    private SimulationResults results;
    private DeviceManager deviceManager;
    private NetworkModel networkModel;
    private SecurityManager securityManager;
    private TaskManager taskManager;
    private TopologyManager topologyManager;
    private LogManager logManager;
    private int currentTick;
    private boolean isRunning;
    
    /**
     * Constructor for FogEdgeSimulation
     * 
     * @param configFilePath Path to the configuration file
     */
    public FogEdgeSimulation(String configFilePath) {
        // Load configuration
        this.config = loadConfiguration(configFilePath);
        
        // Initialize results collector
        this.results = new SimulationResults();
        
        // Initialize log manager
        this.logManager = new LogManager(config);
        
        // Initialize network model
        this.networkModel = new NetworkModel(config);
        
        // Initialize security manager
        this.securityManager = new SecurityManager(config, results);
        
        // Initialize device manager
        this.deviceManager = new DeviceManager(config, results, securityManager);
        
        // Initialize topology manager
        this.topologyManager = new TopologyManager(config, networkModel);
        
        // Initialize task manager
        this.taskManager = new TaskManager(config, results, networkModel, securityManager);
        
        // Initialize simulation state
        this.currentTick = 0;
        this.isRunning = false;
    }
    
    /**
     * Loads configuration from a properties file
     * 
     * @param configFilePath Path to the configuration file
     * @return SimulationConfig object
     */
    private SimulationConfig loadConfiguration(String configFilePath) {
        Properties properties = new Properties();
        
        try (FileInputStream fis = new FileInputStream(configFilePath)) {
            properties.load(fis);
        } catch (IOException e) {
            logManager.logError("Error loading configuration file: " + e.getMessage());
            // Use default configuration if file cannot be loaded
            return new SimulationConfig();
        }
        
        return new SimulationConfig(properties);
    }
    
    /**
     * Initializes the simulation
     */
    public void initialize() {
        logManager.logInfo("Initializing simulation...");
        
        // Reset simulation state
        currentTick = 0;
        isRunning = false;
        
        // Initialize results
        results.initialize();
        
        // Initialize components
        securityManager.initialize();
        deviceManager.initialize();
        networkModel.initialize();
        
        // Initialize topology after devices are created
        topologyManager.initialize(deviceManager.getDevices());
        
        // Initialize task manager after devices and topology are initialized
        taskManager.initialize(deviceManager.getDevices());
        
        logManager.logInfo("Simulation initialized successfully.");
    }
    
    /**
     * Runs the simulation for the specified number of ticks
     * 
     * @param numTicks Number of ticks to run
     */
    public void run(int numTicks) {
        logManager.logInfo("Starting simulation for " + numTicks + " ticks...");
        
        isRunning = true;
        
        for (int i = 0; i < numTicks && isRunning; i++) {
            // Update current tick
            currentTick++;
            
            // Run a single tick
            runTick();
            
            // Log progress periodically
            if (currentTick % 10 == 0 || currentTick == numTicks) {
                logManager.logInfo("Simulation progress: " + currentTick + "/" + numTicks + " ticks completed.");
                logManager.logInfo("Tasks completed: " + results.getCompletedTasksCount() + 
                                 ", Failed: " + results.getFailedTasksCount() + 
                                 ", Offloaded: " + results.getOffloadedTasksCount());
            }
        }
        
        isRunning = false;
        
        logManager.logInfo("Simulation completed after " + currentTick + " ticks.");
    }
    
    /**
     * Runs a single simulation tick
     */
    private void runTick() {
        // Update network conditions
        networkModel.updateNetworkConditions(currentTick);
        
        // Update devices
        deviceManager.updateDevices(currentTick);
        
        // Update topology
        topologyManager.updateTopology(currentTick);
        
        // Simulate security attacks
        securityManager.simulateAttacks(deviceManager.getDevices(), currentTick);
        
        // Update security measures
        securityManager.updateSecurityMeasures(deviceManager.getDevices(), currentTick);
        
        // Generate new tasks from IoT devices
        taskManager.generateTasks(deviceManager.getDevices(), currentTick);
        
        // Assign tasks to devices
        taskManager.assignTasks(deviceManager.getDevices(), topologyManager, currentTick);
        
        // Execute tasks
        taskManager.executeTasks(deviceManager.getDevices(), currentTick);
        
        // Update task status
        taskManager.updateTaskStatus(deviceManager.getDevices(), currentTick);
        
        // Collect metrics
        collectMetrics();
    }
    
    /**
     * Collects metrics for the current tick
     */
    private void collectMetrics() {
        // Update tick count
        results.setTotalTicks(currentTick);
        
        // Collect device metrics
        collectDeviceMetrics();
        
        // Collect network metrics
        collectNetworkMetrics();
        
        // Collect security metrics
        collectSecurityMetrics();
        
        // Collect task metrics
        collectTaskMetrics();
    }
    
    /**
     * Collects device metrics
     */
    private void collectDeviceMetrics() {
        // Calculate average resource utilization
        double totalUtilization = 0;
        int activeDeviceCount = 0;
        
        for (Device device : deviceManager.getDevices().values()) {
            if (device.isActive()) {
                totalUtilization += device.getResourceUtilization();
                activeDeviceCount++;
            }
        }
        
        double avgUtilization = activeDeviceCount > 0 ? totalUtilization / activeDeviceCount : 0;
        results.setAverageResourceUtilization(avgUtilization);
        
        // Calculate average energy level
        double totalEnergyLevel = 0;
        int batteryDeviceCount = 0;
        
        for (Device device : deviceManager.getDevices().values()) {
            if (device.isActive() && !(device instanceof CloudDatacenter)) {
                totalEnergyLevel += device.getEnergyLevel();
                batteryDeviceCount++;
            }
        }
        
        double avgEnergyLevel = batteryDeviceCount > 0 ? totalEnergyLevel / batteryDeviceCount : 0;
        results.setAverageEnergyLevel(avgEnergyLevel);
    }
    
    /**
     * Collects network metrics
     */
    private void collectNetworkMetrics() {
        // Calculate average network bandwidth and latency
        double totalBandwidth = 0;
        double totalLatency = 0;
        int linkCount = 0;
        
        for (var conditionEntry : networkModel.getNetworkConditions().entrySet()) {
            var condition = conditionEntry.getValue();
            totalBandwidth += condition.getCurrentBandwidth();
            totalLatency += condition.getCurrentLatency();
            linkCount++;
        }
        
        double avgBandwidth = linkCount > 0 ? totalBandwidth / linkCount : 0;
        double avgLatency = linkCount > 0 ? totalLatency / linkCount : 0;
        
        results.setAverageNetworkBandwidth(avgBandwidth);
        results.setAverageNetworkLatency(avgLatency);
    }
    
    /**
     * Collects security metrics
     */
    private void collectSecurityMetrics() {
        // Count compromised devices
        int compromisedCount = 0;
        
        for (var entry : securityManager.getCompromisedDevices().entrySet()) {
            if (entry.getValue()) {
                compromisedCount++;
            }
        }
        
        results.setCompromisedDeviceCount(compromisedCount);
    }
    
    /**
     * Collects task metrics
     */
    private void collectTaskMetrics() {
        // Task metrics are already collected by the TaskManager
        // Just update any derived metrics here
        
        // Calculate task completion rate
        int totalTasks = results.getTotalTasksCount();
        int completedTasks = results.getCompletedTasksCount();
        
        double completionRate = totalTasks > 0 ? (double) completedTasks / totalTasks : 0;
        results.setTaskCompletionRate(completionRate);
        
        // Calculate average task execution time
        results.calculateAverageTaskExecutionTime();
    }
    
    /**
     * Stops the simulation
     */
    public void stop() {
        isRunning = false;
        logManager.logInfo("Simulation stopped at tick " + currentTick);
    }
    
    /**
     * Gets the simulation results
     * 
     * @return SimulationResults object
     */
    public SimulationResults getResults() {
        return results;
    }
    
    /**
     * Gets the current tick
     * 
     * @return Current tick
     */
    public int getCurrentTick() {
        return currentTick;
    }
    
    /**
     * Checks if the simulation is running
     * 
     * @return True if running, false otherwise
     */
    public boolean isRunning() {
        return isRunning;
    }
    
    /**
     * Gets the device manager
     * 
     * @return DeviceManager object
     */
    public DeviceManager getDeviceManager() {
        return deviceManager;
    }
    
    /**
     * Gets the network model
     * 
     * @return NetworkModel object
     */
    public NetworkModel getNetworkModel() {
        return networkModel;
    }
    
    /**
     * Gets the security manager
     * 
     * @return SecurityManager object
     */
    public SecurityManager getSecurityManager() {
        return securityManager;
    }
    
    /**
     * Gets the task manager
     * 
     * @return TaskManager object
     */
    public TaskManager getTaskManager() {
        return taskManager;
    }
    
    /**
     * Gets the topology manager
     * 
     * @return TopologyManager object
     */
    public TopologyManager getTopologyManager() {
        return topologyManager;
    }
    
    /**
     * Gets the log manager
     * 
     * @return LogManager object
     */
    public LogManager getLogManager() {
        return logManager;
    }
    
    /**
     * Gets the simulation configuration
     * 
     * @return SimulationConfig object
     */
    public SimulationConfig getConfig() {
        return config;
    }
}
