package org.todg.simulation.util;

import java.io.FileInputStream;
import java.io.IOException;
import java.util.Properties;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Configuration class for the TODG simulation.
 * Holds all parameters needed to configure the simulation.
 */
public class SimulationConfig {
    private static final Logger logger = LoggerFactory.getLogger(SimulationConfig.class);
    
    // Simulation parameters
    private double simulationDuration = 3600.0; // seconds
    private double timeStep = 1.0; // seconds
    private double simulationAreaWidth = 1000.0; // meters
    private double simulationAreaHeight = 1000.0; // meters
    
    // TODG algorithm parameters
    private double alpha = 0.5; // Weight for delay in utility function
    private double beta = 0.3; // Weight for energy in utility function
    private double gamma = 0.2; // Weight for load balancing in utility function
    private double V = 10.0; // Control parameter for delay-energy tradeoff
    
    // IoT device parameters
    private int numDevices = 20;
    private double deviceMipsMin = 500.0;
    private double deviceMipsMax = 1000.0;
    private double deviceMemoryMin = 256.0; // MB
    private double deviceMemoryMax = 512.0; // MB
    private double deviceEnergyMin = 5.0; // Joules
    private double deviceEnergyMax = 10.0; // Joules
    private double taskGenerationRateMin = 0.1; // tasks per second
    private double taskGenerationRateMax = 0.5; // tasks per second
    private double deviceUploadBwMin = 1.0; // Mbps
    private double deviceUploadBwMax = 5.0; // Mbps
    private double deviceDownloadBwMin = 5.0; // Mbps
    private double deviceDownloadBwMax = 20.0; // Mbps
    
    // Edge server parameters
    private int numServers = 5;
    private double serverMipsMin = 10000.0;
    private double serverMipsMax = 20000.0;
    private double serverMemoryMin = 4096.0; // MB
    private double serverMemoryMax = 8192.0; // MB
    private double serverStorageMin = 100.0; // GB
    private double serverStorageMax = 500.0; // GB
    private double serverPowerMin = 100.0; // Watts
    private double serverPowerMax = 200.0; // Watts
    private int serverMaxLoadMin = 50;
    private int serverMaxLoadMax = 100;
    
    // Channel parameters
    private double channelBandwidthMin = 10.0; // Mbps
    private double channelBandwidthMax = 50.0; // Mbps
    private double channelInterferenceMin = 0.1;
    private double channelInterferenceMax = 0.3;
    private double channelReliabilityMin = 0.8;
    private double channelReliabilityMax = 0.99;
    private double channelDynamicsInterval = 10.0; // seconds
    private double channelInterferenceVariability = 0.1;
    private double networkLatencyMin = 10.0; // ms
    private double networkLatencyMax = 50.0; // ms
    
    // Task parameters
    private double taskDataSizeMin = 0.5; // MB
    private double taskDataSizeMax = 5.0; // MB
    private double taskComputationalRequirementMin = 100.0; // MI (Million Instructions)
    private double taskComputationalRequirementMax = 1000.0; // MI
    private double taskDeadlineMin = 5.0; // seconds
    private double taskDeadlineMax = 20.0; // seconds
    
    /**
     * Default constructor with default values.
     */
    public SimulationConfig() {
        // Use default values
    }
    
    /**
     * Constructor that loads configuration from a properties file.
     * 
     * @param configFile Path to the properties file
     */
    public SimulationConfig(String configFile) {
        loadFromFile(configFile);
    }
    
    /**
     * Loads configuration from a properties file.
     * 
     * @param configFile Path to the properties file
     */
    public void loadFromFile(String configFile) {
        Properties props = new Properties();
        try (FileInputStream fis = new FileInputStream(configFile)) {
            props.load(fis);
            
            // Load simulation parameters
            simulationDuration = getDoubleProperty(props, "simulation.duration", simulationDuration);
            timeStep = getDoubleProperty(props, "simulation.timeStep", timeStep);
            simulationAreaWidth = getDoubleProperty(props, "simulation.areaWidth", simulationAreaWidth);
            simulationAreaHeight = getDoubleProperty(props, "simulation.areaHeight", simulationAreaHeight);
            
            // Load TODG algorithm parameters
            alpha = getDoubleProperty(props, "algorithm.alpha", alpha);
            beta = getDoubleProperty(props, "algorithm.beta", beta);
            gamma = getDoubleProperty(props, "algorithm.gamma", gamma);
            V = getDoubleProperty(props, "algorithm.V", V);
            
            // Load IoT device parameters
            numDevices = getIntProperty(props, "device.count", numDevices);
            deviceMipsMin = getDoubleProperty(props, "device.mips.min", deviceMipsMin);
            deviceMipsMax = getDoubleProperty(props, "device.mips.max", deviceMipsMax);
            deviceMemoryMin = getDoubleProperty(props, "device.memory.min", deviceMemoryMin);
            deviceMemoryMax = getDoubleProperty(props, "device.memory.max", deviceMemoryMax);
            deviceEnergyMin = getDoubleProperty(props, "device.energy.min", deviceEnergyMin);
            deviceEnergyMax = getDoubleProperty(props, "device.energy.max", deviceEnergyMax);
            taskGenerationRateMin = getDoubleProperty(props, "device.taskRate.min", taskGenerationRateMin);
            taskGenerationRateMax = getDoubleProperty(props, "device.taskRate.max", taskGenerationRateMax);
            deviceUploadBwMin = getDoubleProperty(props, "device.uploadBw.min", deviceUploadBwMin);
            deviceUploadBwMax = getDoubleProperty(props, "device.uploadBw.max", deviceUploadBwMax);
            deviceDownloadBwMin = getDoubleProperty(props, "device.downloadBw.min", deviceDownloadBwMin);
            deviceDownloadBwMax = getDoubleProperty(props, "device.downloadBw.max", deviceDownloadBwMax);
            
            // Load edge server parameters
            numServers = getIntProperty(props, "server.count", numServers);
            serverMipsMin = getDoubleProperty(props, "server.mips.min", serverMipsMin);
            serverMipsMax = getDoubleProperty(props, "server.mips.max", serverMipsMax);
            serverMemoryMin = getDoubleProperty(props, "server.memory.min", serverMemoryMin);
            serverMemoryMax = getDoubleProperty(props, "server.memory.max", serverMemoryMax);
            serverStorageMin = getDoubleProperty(props, "server.storage.min", serverStorageMin);
            serverStorageMax = getDoubleProperty(props, "server.storage.max", serverStorageMax);
            serverPowerMin = getDoubleProperty(props, "server.power.min", serverPowerMin);
            serverPowerMax = getDoubleProperty(props, "server.power.max", serverPowerMax);
            serverMaxLoadMin = getIntProperty(props, "server.maxLoad.min", serverMaxLoadMin);
            serverMaxLoadMax = getIntProperty(props, "server.maxLoad.max", serverMaxLoadMax);
            
            // Load channel parameters
            channelBandwidthMin = getDoubleProperty(props, "channel.bandwidth.min", channelBandwidthMin);
            channelBandwidthMax = getDoubleProperty(props, "channel.bandwidth.max", channelBandwidthMax);
            channelInterferenceMin = getDoubleProperty(props, "channel.interference.min", channelInterferenceMin);
            channelInterferenceMax = getDoubleProperty(props, "channel.interference.max", channelInterferenceMax);
            channelReliabilityMin = getDoubleProperty(props, "channel.reliability.min", channelReliabilityMin);
            channelReliabilityMax = getDoubleProperty(props, "channel.reliability.max", channelReliabilityMax);
            channelDynamicsInterval = getDoubleProperty(props, "channel.dynamicsInterval", channelDynamicsInterval);
            channelInterferenceVariability = getDoubleProperty(props, "channel.interferenceVariability", channelInterferenceVariability);
            networkLatencyMin = getDoubleProperty(props, "network.latency.min", networkLatencyMin);
            networkLatencyMax = getDoubleProperty(props, "network.latency.max", networkLatencyMax);
            
            // Load task parameters
            taskDataSizeMin = getDoubleProperty(props, "task.dataSize.min", taskDataSizeMin);
            taskDataSizeMax = getDoubleProperty(props, "task.dataSize.max", taskDataSizeMax);
            taskComputationalRequirementMin = getDoubleProperty(props, "task.computationalRequirement.min", taskComputationalRequirementMin);
            taskComputationalRequirementMax = getDoubleProperty(props, "task.computationalRequirement.max", taskComputationalRequirementMax);
            taskDeadlineMin = getDoubleProperty(props, "task.deadline.min", taskDeadlineMin);
            taskDeadlineMax = getDoubleProperty(props, "task.deadline.max", taskDeadlineMax);
            
            logger.info("Configuration loaded from file: {}", configFile);
        } catch (IOException e) {
            logger.error("Error loading configuration from file: {}", e.getMessage());
        }
    }
    
    /**
     * Helper method to get a double property with a default value.
     */
    private double getDoubleProperty(Properties props, String key, double defaultValue) {
        String value = props.getProperty(key);
        if (value != null) {
            try {
                return Double.parseDouble(value);
            } catch (NumberFormatException e) {
                logger.warn("Invalid double value for {}: {}", key, value);
            }
        }
        return defaultValue;
    }
    
    /**
     * Helper method to get an integer property with a default value.
     */
    private int getIntProperty(Properties props, String key, int defaultValue) {
        String value = props.getProperty(key);
        if (value != null) {
            try {
                return Integer.parseInt(value);
            } catch (NumberFormatException e) {
                logger.warn("Invalid integer value for {}: {}", key, value);
            }
        }
        return defaultValue;
    }
    
    // Getters and setters for all properties
    
    public double getSimulationDuration() {
        return simulationDuration;
    }
    
    public void setSimulationDuration(double simulationDuration) {
        this.simulationDuration = simulationDuration;
    }
    
    public double getTimeStep() {
        return timeStep;
    }
    
    public void setTimeStep(double timeStep) {
        this.timeStep = timeStep;
    }
    
    public double getSimulationAreaWidth() {
        return simulationAreaWidth;
    }
    
    public void setSimulationAreaWidth(double simulationAreaWidth) {
        this.simulationAreaWidth = simulationAreaWidth;
    }
    
    public double getSimulationAreaHeight() {
        return simulationAreaHeight;
    }
    
    public void setSimulationAreaHeight(double simulationAreaHeight) {
        this.simulationAreaHeight = simulationAreaHeight;
    }
    
    public double getAlpha() {
        return alpha;
    }
    
    public void setAlpha(double alpha) {
        this.alpha = alpha;
    }
    
    public double getBeta() {
        return beta;
    }
    
    public void setBeta(double beta) {
        this.beta = beta;
    }
    
    public double getGamma() {
        return gamma;
    }
    
    public void setGamma(double gamma) {
        this.gamma = gamma;
    }
    
    public double getV() {
        return V;
    }
    
    public void setV(double v) {
        V = v;
    }
    
    public int getNumDevices() {
        return numDevices;
    }
    
    public void setNumDevices(int numDevices) {
        this.numDevices = numDevices;
    }
    
    public double getDeviceMipsMin() {
        return deviceMipsMin;
    }
    
    public void setDeviceMipsMin(double deviceMipsMin) {
        this.deviceMipsMin = deviceMipsMin;
    }
    
    public double getDeviceMipsMax() {
        return deviceMipsMax;
    }
    
    public void setDeviceMipsMax(double deviceMipsMax) {
        this.deviceMipsMax = deviceMipsMax;
    }
    
    public double getDeviceMemoryMin() {
        return deviceMemoryMin;
    }
    
    public void setDeviceMemoryMin(double deviceMemoryMin) {
        this.deviceMemoryMin = deviceMemoryMin;
    }
    
    public double getDeviceMemoryMax() {
        return deviceMemoryMax;
    }
    
    public void setDeviceMemoryMax(double deviceMemoryMax) {
        this.deviceMemoryMax = deviceMemoryMax;
    }
    
    public double getDeviceEnergyMin() {
        return deviceEnergyMin;
    }
    
    public void setDeviceEnergyMin(double deviceEnergyMin) {
        this.deviceEnergyMin = deviceEnergyMin;
    }
    
    public double getDeviceEnergyMax() {
        return deviceEnergyMax;
    }
    
    public void setDeviceEnergyMax(double deviceEnergyMax) {
        this.deviceEnergyMax = deviceEnergyMax;
    }
    
    public double getTaskGenerationRateMin() {
        return taskGenerationRateMin;
    }
    
    public void setTaskGenerationRateMin(double taskGenerationRateMin) {
        this.taskGenerationRateMin = taskGenerationRateMin;
    }
    
    public double getTaskGenerationRateMax() {
        return taskGenerationRateMax;
    }
    
    public void setTaskGenerationRateMax(double taskGenerationRateMax) {
        this.taskGenerationRateMax = taskGenerationRateMax;
    }
    
    public double getDeviceUploadBwMin() {
        return deviceUploadBwMin;
    }
    
    public void setDeviceUploadBwMin(double deviceUploadBwMin) {
        this.deviceUploadBwMin = deviceUploadBwMin;
    }
    
    public double getDeviceUploadBwMax() {
        return deviceUploadBwMax;
    }
    
    public void setDeviceUploadBwMax(double deviceUploadBwMax) {
        this.deviceUploadBwMax = deviceUploadBwMax;
    }
    
    public double getDeviceDownloadBwMin() {
        return deviceDownloadBwMin;
    }
    
    public void setDeviceDownloadBwMin(double deviceDownloadBwMin) {
        this.deviceDownloadBwMin = deviceDownloadBwMin;
    }
    
    public double getDeviceDownloadBwMax() {
        return deviceDownloadBwMax;
    }
    
    public void setDeviceDownloadBwMax(double deviceDownloadBwMax) {
        this.deviceDownloadBwMax = deviceDownloadBwMax;
    }
    
    public int getNumServers() {
        return numServers;
    }
    
    public void setNumServers(int numServers) {
        this.numServers = numServers;
    }
    
    public double getServerMipsMin() {
        return serverMipsMin;
    }
    
    public void setServerMipsMin(double serverMipsMin) {
        this.serverMipsMin = serverMipsMin;
    }
    
    public double getServerMipsMax() {
        return serverMipsMax;
    }
    
    public void setServerMipsMax(double serverMipsMax) {
        this.serverMipsMax = serverMipsMax;
    }
    
    public double getServerMemoryMin() {
        return serverMemoryMin;
    }
    
    public void setServerMemoryMin(double serverMemoryMin) {
        this.serverMemoryMin = serverMemoryMin;
    }
    
    public double getServerMemoryMax() {
        return serverMemoryMax;
    }
    
    public void setServerMemoryMax(double serverMemoryMax) {
        this.serverMemoryMax = serverMemoryMax;
    }
    
    public double getServerStorageMin() {
        return serverStorageMin;
    }
    
    public void setServerStorageMin(double serverStorageMin) {
        this.serverStorageMin = serverStorageMin;
    }
    
    public double getServerStorageMax() {
        return serverStorageMax;
    }
    
    public void setServerStorageMax(double serverStorageMax) {
        this.serverStorageMax = serverStorageMax;
    }
    
    public double getServerPowerMin() {
        return serverPowerMin;
    }
    
    public void setServerPowerMin(double serverPowerMin) {
        this.serverPowerMin = serverPowerMin;
    }
    
    public double getServerPowerMax() {
        return serverPowerMax;
    }
    
    public void setServerPowerMax(double serverPowerMax) {
        this.serverPowerMax = serverPowerMax;
    }
    
    public int getServerMaxLoadMin() {
        return serverMaxLoadMin;
    }
    
    public void setServerMaxLoadMin(int serverMaxLoadMin) {
        this.serverMaxLoadMin = serverMaxLoadMin;
    }
    
    public int getServerMaxLoadMax() {
        return serverMaxLoadMax;
    }
    
    public void setServerMaxLoadMax(int serverMaxLoadMax) {
        this.serverMaxLoadMax = serverMaxLoadMax;
    }
    
    public double getChannelBandwidthMin() {
        return channelBandwidthMin;
    }
    
    public void setChannelBandwidthMin(double channelBandwidthMin) {
        this.channelBandwidthMin = channelBandwidthMin;
    }
    
    public double getChannelBandwidthMax() {
        return channelBandwidthMax;
    }
    
    public void setChannelBandwidthMax(double channelBandwidthMax) {
        this.channelBandwidthMax = channelBandwidthMax;
    }
    
    public double getChannelInterferenceMin() {
        return channelInterferenceMin;
    }
    
    public void setChannelInterferenceMin(double channelInterferenceMin) {
        this.channelInterferenceMin = channelInterferenceMin;
    }
    
    public double getChannelInterferenceMax() {
        return channelInterferenceMax;
    }
    
    public void setChannelInterferenceMax(double channelInterferenceMax) {
        this.channelInterferenceMax = channelInterferenceMax;
    }
    
    public double getChannelReliabilityMin() {
        return channelReliabilityMin;
    }
    
    public void setChannelReliabilityMin(double channelReliabilityMin) {
        this.channelReliabilityMin = channelReliabilityMin;
    }
    
    public double getChannelReliabilityMax() {
        return channelReliabilityMax;
    }
    
    public void setChannelReliabilityMax(double channelReliabilityMax) {
        this.channelReliabilityMax = channelReliabilityMax;
    }
    
    public double getChannelDynamicsInterval() {
        return channelDynamicsInterval;
    }
    
    public void setChannelDynamicsInterval(double channelDynamicsInterval) {
        this.channelDynamicsInterval = channelDynamicsInterval;
    }
    
    public double getChannelInterferenceVariability() {
        return channelInterferenceVariability;
    }
    
    public void setChannelInterferenceVariability(double channelInterferenceVariability) {
        this.channelInterferenceVariability = channelInterferenceVariability;
    }
    
    public double getNetworkLatencyMin() {
        return networkLatencyMin;
    }
    
    public void setNetworkLatencyMin(double networkLatencyMin) {
        this.networkLatencyMin = networkLatencyMin;
    }
    
    public double getNetworkLatencyMax() {
        return networkLatencyMax;
    }
    
    public void setNetworkLatencyMax(double networkLatencyMax) {
        this.networkLatencyMax = networkLatencyMax;
    }
    
    public double getTaskDataSizeMin() {
        return taskDataSizeMin;
    }
    
    public void setTaskDataSizeMin(double taskDataSizeMin) {
        this.taskDataSizeMin = taskDataSizeMin;
    }
    
    public double getTaskDataSizeMax() {
        return taskDataSizeMax;
    }
    
    public void setTaskDataSizeMax(double taskDataSizeMax) {
        this.taskDataSizeMax = taskDataSizeMax;
    }
    
    public double getTaskComputationalRequirementMin() {
        return taskComputationalRequirementMin;
    }
    
    public void setTaskComputationalRequirementMin(double taskComputationalRequirementMin) {
        this.taskComputationalRequirementMin = taskComputationalRequirementMin;
    }
    
    public double getTaskComputationalRequirementMax() {
        return taskComputationalRequirementMax;
    }
    
    public void setTaskComputationalRequirementMax(double taskComputationalRequirementMax) {
        this.taskComputationalRequirementMax = taskComputationalRequirementMax;
    }
    
    public double getTaskDeadlineMin() {
        return taskDeadlineMin;
    }
    
    public void setTaskDeadlineMin(double taskDeadlineMin) {
        this.taskDeadlineMin = taskDeadlineMin;
    }
    
    public double getTaskDeadlineMax() {
        return taskDeadlineMax;
    }
    
    public void setTaskDeadlineMax(double taskDeadlineMax) {
        this.taskDeadlineMax = taskDeadlineMax;
    }
}
