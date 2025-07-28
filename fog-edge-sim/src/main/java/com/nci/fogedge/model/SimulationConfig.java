package com.nci.fogedge.model;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.Properties;

/**
 * Configuration class for the fog and edge computing simulation.
 * Loads and provides access to all simulation parameters from a properties file.
 */
public class SimulationConfig {
    private Properties properties;
    private String configFilePath;
    
    /**
     * Creates a new SimulationConfig with default configuration file path.
     */
    public SimulationConfig() {
        this("src/main/resources/simulation.properties");
    }
    
    /**
     * Creates a new SimulationConfig with the specified configuration file path.
     * 
     * @param configFilePath Path to the configuration file
     */
    public SimulationConfig(String configFilePath) {
        this.configFilePath = configFilePath;
        this.properties = new Properties();
        loadProperties();
    }
    
    /**
     * Loads properties from the configuration file.
     */
    private void loadProperties() {
        try (InputStream input = new FileInputStream(configFilePath)) {
            properties.load(input);
        } catch (IOException e) {
            System.err.println("Error loading configuration file: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    /**
     * Gets a property as a string.
     * 
     * @param key Property key
     * @param defaultValue Default value if property is not found
     * @return Property value as a string
     */
    public String getProperty(String key, String defaultValue) {
        return properties.getProperty(key, defaultValue);
    }
    
    /**
     * Gets a property as an integer.
     * 
     * @param key Property key
     * @param defaultValue Default value if property is not found
     * @return Property value as an integer
     */
    public int getIntProperty(String key, int defaultValue) {
        String value = properties.getProperty(key);
        if (value == null) {
            return defaultValue;
        }
        try {
            return Integer.parseInt(value);
        } catch (NumberFormatException e) {
            System.err.println("Invalid integer property: " + key + " = " + value);
            return defaultValue;
        }
    }
    
    /**
     * Gets a property as a double.
     * 
     * @param key Property key
     * @param defaultValue Default value if property is not found
     * @return Property value as a double
     */
    public double getDoubleProperty(String key, double defaultValue) {
        String value = properties.getProperty(key);
        if (value == null) {
            return defaultValue;
        }
        try {
            return Double.parseDouble(value);
        } catch (NumberFormatException e) {
            System.err.println("Invalid double property: " + key + " = " + value);
            return defaultValue;
        }
    }
    
    /**
     * Gets a property as a boolean.
     * 
     * @param key Property key
     * @param defaultValue Default value if property is not found
     * @return Property value as a boolean
     */
    public boolean getBooleanProperty(String key, boolean defaultValue) {
        String value = properties.getProperty(key);
        if (value == null) {
            return defaultValue;
        }
        return Boolean.parseBoolean(value);
    }
    
    /**
     * Gets a property as a string array by splitting on commas.
     * 
     * @param key Property key
     * @return Property value as a string array
     */
    public String[] getArrayProperty(String key) {
        String value = properties.getProperty(key, "");
        return value.split(",");
    }
    
    // Simulation parameters
    
    /**
     * Gets the simulation duration in seconds.
     * 
     * @return Simulation duration
     */
    public int getSimulationDuration() {
        return getIntProperty("simulation.duration", 3600);
    }
    
    /**
     * Gets the simulation area size in meters.
     * 
     * @return Simulation area size
     */
    public int getSimulationAreaSize() {
        return getIntProperty("simulation.area.size", 1000);
    }
    
    /**
     * Gets the simulation area width in meters.
     * 
     * @return Simulation area width
     */
    public int getSimulationAreaWidth() {
        return getSimulationAreaSize();
    }
    
    /**
     * Gets the simulation area height in meters.
     * 
     * @return Simulation area height
     */
    public int getSimulationAreaHeight() {
        return getSimulationAreaSize();
    }
    
    /**
     * Gets the random seed for the simulation.
     * 
     * @return Random seed
     */
    public int getRandomSeed() {
        return getIntProperty("simulation.random.seed", 12345);
    }
    
    // Device counts
    
    /**
     * Gets the number of IoT devices in the simulation.
     * 
     * @return Number of IoT devices
     */
    public int getIoTDeviceCount() {
        return getIntProperty("device.iot.count", 100);
    }
    
    /**
     * Gets the number of edge nodes in the simulation.
     * 
     * @return Number of edge nodes
     */
    public int getEdgeNodeCount() {
        return getIntProperty("device.edge.count", 10);
    }
    
    /**
     * Gets the number of fog nodes in the simulation.
     * 
     * @return Number of fog nodes
     */
    public int getFogNodeCount() {
        return getIntProperty("device.fog.count", 5);
    }
    
    /**
     * Gets the number of cloud datacenters in the simulation.
     * 
     * @return Number of cloud datacenters
     */
    public int getCloudDatacenterCount() {
        return getIntProperty("device.cloud.count", 2);
    }
    
    // IoT device parameters
    
    /**
     * Gets the percentage of IoT devices that are mobile.
     * 
     * @return Percentage of mobile IoT devices
     */
    public double getIoTMobilityPercentage() {
        return getDoubleProperty("device.iot.mobility.percentage", 30) / 100.0;
    }
    
    /**
     * Gets the battery capacity of IoT devices in mAh.
     * 
     * @return IoT device battery capacity
     */
    public double getIoTBatteryCapacity() {
        return getDoubleProperty("device.iot.battery.capacity", 5000);
    }
    
    /**
     * Gets the CPU capacity of IoT devices in MIPS.
     * 
     * @return IoT device CPU capacity
     */
    public double getIoTCpuCapacity() {
        return getDoubleProperty("device.iot.cpu.capacity", 500);
    }
    
    /**
     * Gets the RAM capacity of IoT devices in MB.
     * 
     * @return IoT device RAM capacity
     */
    public double getIoTRamCapacity() {
        return getDoubleProperty("device.iot.ram.capacity", 512);
    }
    
    /**
     * Gets the storage capacity of IoT devices in GB.
     * 
     * @return IoT device storage capacity
     */
    public double getIoTStorageCapacity() {
        return getDoubleProperty("device.iot.storage.capacity", 4);
    }
    
    /**
     * Gets the task generation rate of IoT devices in tasks per second.
     * 
     * @return IoT device task generation rate
     */
    public double getIoTTaskGenerationRate() {
        return getDoubleProperty("device.iot.task.generation.rate", 0.1);
    }
    
    /**
     * Gets the wireless types supported by IoT devices.
     * 
     * @return IoT device wireless types
     */
    public String[] getIoTWirelessTypes() {
        return getArrayProperty("device.iot.wireless.types");
    }
    
    // Edge node parameters
    
    /**
     * Gets the CPU capacity of edge nodes in MIPS.
     * 
     * @return Edge node CPU capacity
     */
    public double getEdgeCpuCapacity() {
        return getDoubleProperty("device.edge.cpu.capacity", 2000);
    }
    
    /**
     * Gets the RAM capacity of edge nodes in MB.
     * 
     * @return Edge node RAM capacity
     */
    public double getEdgeRamCapacity() {
        return getDoubleProperty("device.edge.ram.capacity", 4096);
    }
    
    /**
     * Gets the storage capacity of edge nodes in GB.
     * 
     * @return Edge node storage capacity
     */
    public double getEdgeStorageCapacity() {
        return getDoubleProperty("device.edge.storage.capacity", 128);
    }
    
    /**
     * Gets the battery capacity of edge nodes in mAh.
     * 
     * @return Edge node battery capacity
     */
    public double getEdgeBatteryCapacity() {
        return getDoubleProperty("device.edge.battery.capacity", 10000);
    }
    
    /**
     * Gets the maximum number of connections for edge nodes.
     * 
     * @return Edge node maximum connections
     */
    public int getEdgeMaxConnections() {
        return getIntProperty("device.edge.max.connections", 20);
    }
    
    // Fog node parameters
    
    /**
     * Gets the CPU capacity of fog nodes in MIPS.
     * 
     * @return Fog node CPU capacity
     */
    public double getFogCpuCapacity() {
        return getDoubleProperty("device.fog.cpu.capacity", 5000);
    }
    
    /**
     * Gets the RAM capacity of fog nodes in MB.
     * 
     * @return Fog node RAM capacity
     */
    public double getFogRamCapacity() {
        return getDoubleProperty("device.fog.ram.capacity", 8192);
    }
    
    /**
     * Gets the storage capacity of fog nodes in GB.
     * 
     * @return Fog node storage capacity
     */
    public double getFogStorageCapacity() {
        return getDoubleProperty("device.fog.storage.capacity", 512);
    }
    
    /**
     * Gets the battery capacity of fog nodes in mAh.
     * 
     * @return Fog node battery capacity
     */
    public double getFogBatteryCapacity() {
        return getDoubleProperty("device.fog.battery.capacity", 20000);
    }
    
    /**
     * Gets the maximum number of connections for fog nodes.
     * 
     * @return Fog node maximum connections
     */
    public int getFogMaxConnections() {
        return getIntProperty("device.fog.max.connections", 50);
    }
    
    // Cloud datacenter parameters
    
    /**
     * Gets the CPU capacity of cloud datacenters in MIPS.
     * 
     * @return Cloud datacenter CPU capacity
     */
    public double getCloudCpuCapacity() {
        return getDoubleProperty("device.cloud.cpu.capacity", 20000);
    }
    
    /**
     * Gets the RAM capacity of cloud datacenters in MB.
     * 
     * @return Cloud datacenter RAM capacity
     */
    public double getCloudRamCapacity() {
        return getDoubleProperty("device.cloud.ram.capacity", 65536);
    }
    
    /**
     * Gets the storage capacity of cloud datacenters in GB.
     * 
     * @return Cloud datacenter storage capacity
     */
    public double getCloudStorageCapacity() {
        return getDoubleProperty("device.cloud.storage.capacity", 10240);
    }
    
    /**
     * Gets the maximum number of connections for cloud datacenters.
     * 
     * @return Cloud datacenter maximum connections
     */
    public int getCloudMaxConnections() {
        return getIntProperty("device.cloud.max.connections", 200);
    }
    
    // Task parameters
    
    /**
     * Gets the minimum CPU requirement for tasks in MIPS.
     * 
     * @return Minimum task CPU requirement
     */
    public double getTaskMinCpu() {
        return getDoubleProperty("task.min.cpu", 100);
    }
    
    /**
     * Gets the maximum CPU requirement for tasks in MIPS.
     * 
     * @return Maximum task CPU requirement
     */
    public double getTaskMaxCpu() {
        return getDoubleProperty("task.max.cpu", 1000);
    }
    
    /**
     * Gets the minimum RAM requirement for tasks in MB.
     * 
     * @return Minimum task RAM requirement
     */
    public double getTaskMinRam() {
        return getDoubleProperty("task.min.ram", 64);
    }
    
    /**
     * Gets the maximum RAM requirement for tasks in MB.
     * 
     * @return Maximum task RAM requirement
     */
    public double getTaskMaxRam() {
        return getDoubleProperty("task.max.ram", 1024);
    }
    
    /**
     * Gets the minimum storage requirement for tasks in GB.
     * 
     * @return Minimum task storage requirement
     */
    public double getTaskMinStorage() {
        return getDoubleProperty("task.min.storage", 1);
    }
    
    /**
     * Gets the maximum storage requirement for tasks in GB.
     * 
     * @return Maximum task storage requirement
     */
    public double getTaskMaxStorage() {
        return getDoubleProperty("task.max.storage", 100);
    }
    
    /**
     * Gets the minimum duration for tasks in seconds.
     * 
     * @return Minimum task duration
     */
    public double getTaskMinDuration() {
        return getDoubleProperty("task.min.duration", 10);
    }
    
    /**
     * Gets the maximum duration for tasks in seconds.
     * 
     * @return Maximum task duration
     */
    public double getTaskMaxDuration() {
        return getDoubleProperty("task.max.duration", 300);
    }
    
    /**
     * Checks if task offloading is enabled.
     * 
     * @return True if task offloading is enabled, false otherwise
     */
    public boolean isTaskOffloadingEnabled() {
        return getBooleanProperty("task.offloading.enabled", true);
    }
    
    /**
     * Gets the task offloading threshold.
     * 
     * @return Task offloading threshold
     */
    public double getTaskOffloadingThreshold() {
        return getDoubleProperty("task.offloading.threshold", 0.7);
    }
    
    /**
     * Gets the task scheduling policy.
     * 
     * @return Task scheduling policy
     */
    public String getTaskSchedulingPolicy() {
        return getProperty("task.scheduling.policy", "PRIORITY");
    }
    
    /**
     * Gets the task output size in KB.
     * 
     * @return Task output size
     */
    public double getTaskOutputSize() {
        return getDoubleProperty("task.output.size", 512);
    }
    
    // Network parameters
    
    /**
     * Gets the base network bandwidth in Mbps.
     * 
     * @return Base network bandwidth
     */
    public double getNetworkBaseBandwidth() {
        return getDoubleProperty("network.base.bandwidth", 100);
    }
    
    /**
     * Gets the base network latency in ms.
     * 
     * @return Base network latency
     */
    public double getNetworkBaseLatency() {
        return getDoubleProperty("network.base.latency", 10);
    }
    
    /**
     * Gets the network variability factor.
     * 
     * @return Network variability factor
     */
    public double getNetworkVariabilityFactor() {
        return getDoubleProperty("network.variability.factor", 0.2);
    }
    
    /**
     * Gets the network congestion probability.
     * 
     * @return Network congestion probability
     */
    public double getNetworkCongestionProbability() {
        return getDoubleProperty("network.congestion.probability", 0.05);
    }
    
    /**
     * Gets the network packet loss probability.
     * 
     * @return Network packet loss probability
     */
    public double getNetworkPacketLossProbability() {
        return getDoubleProperty("network.packet.loss.probability", 0.01);
    }
    
    /**
     * Checks if IoT mesh networking is enabled.
     * 
     * @return True if IoT mesh networking is enabled, false otherwise
     */
    public boolean isIoTMeshEnabled() {
        return getBooleanProperty("network.iot.mesh.enabled", false);
    }
    
    /**
     * Checks if edge mesh networking is enabled.
     * 
     * @return True if edge mesh networking is enabled, false otherwise
     */
    public boolean isEdgeMeshEnabled() {
        return getBooleanProperty("network.edge.mesh.enabled", true);
    }
    
    /**
     * Checks if fog mesh networking is enabled.
     * 
     * @return True if fog mesh networking is enabled, false otherwise
     */
    public boolean isFogMeshEnabled() {
        return getBooleanProperty("network.fog.mesh.enabled", true);
    }
    
    // Security parameters
    
    /**
     * Checks if security attacks are enabled.
     * 
     * @return True if security attacks are enabled, false otherwise
     */
    public boolean areSecurityAttacksEnabled() {
        return getBooleanProperty("security.attacks.enabled", true);
    }
    
    /**
     * Gets the security attack probability.
     * 
     * @return Security attack probability
     */
    public double getSecurityAttackProbability() {
        return getDoubleProperty("security.attack.probability", 0.02);
    }
    
    /**
     * Gets the security detection probability.
     * 
     * @return Security detection probability
     */
    public double getSecurityDetectionProbability() {
        return getDoubleProperty("security.detection.probability", 0.8);
    }
    
    /**
     * Gets the security mitigation probability.
     * 
     * @return Security mitigation probability
     */
    public double getSecurityMitigationProbability() {
        return getDoubleProperty("security.mitigation.probability", 0.7);
    }
    
    /**
     * Gets the security attack types.
     * 
     * @return Security attack types
     */
    public String[] getSecurityAttackTypes() {
        return getArrayProperty("security.attack.types");
    }
    
    /**
     * Gets the security countermeasure types.
     * 
     * @return Security countermeasure types
     */
    public String[] getSecurityCountermeasureTypes() {
        return getArrayProperty("security.countermeasure.types");
    }
    
    // Logging parameters
    
    /**
     * Checks if console logging is enabled.
     * 
     * @return True if console logging is enabled, false otherwise
     */
    public boolean isConsoleLoggingEnabled() {
        return getBooleanProperty("logging.console.enabled", true);
    }
    
    /**
     * Checks if file logging is enabled.
     * 
     * @return True if file logging is enabled, false otherwise
     */
    public boolean isFileLoggingEnabled() {
        return getBooleanProperty("logging.file.enabled", true);
    }
    
    /**
     * Gets the log file path.
     * 
     * @return Log file path
     */
    public String getLogFilePath() {
        return getProperty("logging.file.path", "simulation_log.txt");
    }
    
    /**
     * Gets the minimum log level.
     * 
     * @return Minimum log level
     */
    public String getMinLogLevel() {
        return getProperty("logging.min.level", "INFO");
    }
}
