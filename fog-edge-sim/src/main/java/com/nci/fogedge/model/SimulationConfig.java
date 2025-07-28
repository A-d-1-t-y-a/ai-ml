package com.nci.fogedge.model;

import com.nci.fogedge.security.AttackType;
import com.nci.fogedge.security.CountermeasureType;
import com.nci.fogedge.util.LogManager.LogLevel;

import java.io.FileInputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Properties;

/**
 * Configuration class for the Fog and Edge Computing Simulation.
 * This class loads and manages all simulation parameters.
 */
public class SimulationConfig {
    // Simulation parameters
    private int simulationDuration;
    private int simulationAreaSize;
    private long randomSeed;
    
    // Device counts
    private int ioTDeviceCount;
    private int edgeNodeCount;
    private int fogNodeCount;
    private int cloudDatacenterCount;
    
    // IoT device parameters
    private int ioTMobilityPercentage;
    private int ioTBatteryCapacity;
    private int ioTCpuCapacity;
    private int ioTRamCapacity;
    private int ioTStorageCapacity;
    private double ioTTaskGenerationRate;
    private List<String> ioTWirelessTypes;
    
    // Edge node parameters
    private int edgeCpuCapacity;
    private int edgeRamCapacity;
    private int edgeStorageCapacity;
    private int edgeBatteryCapacity;
    private int edgeMaxConnections;
    
    // Fog node parameters
    private int fogCpuCapacity;
    private int fogRamCapacity;
    private int fogStorageCapacity;
    private int fogBatteryCapacity;
    private int fogMaxConnections;
    
    // Cloud datacenter parameters
    private int cloudCpuCapacity;
    private int cloudRamCapacity;
    private int cloudStorageCapacity;
    private int cloudMaxConnections;
    
    // Task parameters
    private int taskMinCpu;
    private int taskMaxCpu;
    private int taskMinRam;
    private int taskMaxRam;
    private int taskMinStorage;
    private int taskMaxStorage;
    private int taskMinDuration;
    private int taskMaxDuration;
    private boolean taskOffloadingEnabled;
    private double taskOffloadingThreshold;
    private String taskSchedulingPolicy;
    private int taskOutputSize; // Size of task output in KB
    
    // Network parameters
    private double networkBaseBandwidth;
    private double networkBaseLatency;
    private double networkVariabilityFactor;
    private double networkCongestionProbability;
    private double networkPacketLossProbability;
    private boolean ioTMeshNetworkEnabled;
    private boolean edgeMeshNetworkEnabled;
    private boolean fogMeshNetworkEnabled;
    
    // Security parameters
    private boolean securityAttacksEnabled;
    private double securityAttackProbability;
    private double securityDetectionProbability;
    private double securityMitigationProbability;
    private List<AttackType> securityAttackTypes;
    private List<CountermeasureType> securityCountermeasureTypes;
    
    // Logging parameters
    private boolean consoleLoggingEnabled;
    private boolean fileLoggingEnabled;
    private String logFilePath;
    private LogLevel minLogLevel;
    
    /**
     * Default constructor that loads default configuration values
     */
    public SimulationConfig() {
        loadDefaults();
    }
    
    /**
     * Constructor that loads configuration from properties
     * @param properties Properties object containing configuration values
     */
    public SimulationConfig(Properties properties) {
        loadDefaults();
        loadFromProperties(properties);
    }
    
    /**
     * Constructor that loads configuration from a file
     * @param configFilePath Path to the configuration file
     */
    public SimulationConfig(String configFilePath) {
        loadDefaults();
        if (configFilePath != null && !configFilePath.isEmpty()) {
            try {
                loadFromFile(configFilePath);
            } catch (IOException e) {
                System.err.println("Error loading configuration file: " + e.getMessage());
                System.err.println("Using default configuration values");
            }
        }
    }
    
    /**
     * Loads configuration from a file
     * @param configFilePath Path to the configuration file
     * @throws IOException If there is an error reading the file
     */
    private void loadFromFile(String configFilePath) throws IOException {
        Properties properties = new Properties();
        try (FileInputStream fis = new FileInputStream(configFilePath)) {
            properties.load(fis);
            loadFromProperties(properties);
        }
    }
    
    /**
     * Loads default configuration values
     */
    private void loadDefaults() {
        // Simulation defaults
        simulationDuration = 1000;
        simulationAreaSize = 1000;
        randomSeed = 12345;
        
        // Device counts
        ioTDeviceCount = 100;
        edgeNodeCount = 10;
        fogNodeCount = 5;
        cloudDatacenterCount = 1;
        
        // IoT device parameters
        ioTMobilityPercentage = 30;
        ioTBatteryCapacity = 5000;
        ioTCpuCapacity = 1000;
        ioTRamCapacity = 512;
        ioTStorageCapacity = 1024;
        ioTTaskGenerationRate = 0.2;
        ioTWirelessTypes = Arrays.asList("BLUETOOTH", "WIFI", "CELLULAR", "ZIGBEE", "LORA");
        
        // Edge node parameters
        edgeCpuCapacity = 5000;
        edgeRamCapacity = 8192;
        edgeStorageCapacity = 51200;
        edgeBatteryCapacity = 20000;
        edgeMaxConnections = 20;
        
        // Fog node parameters
        fogCpuCapacity = 10000;
        fogRamCapacity = 32768;
        fogStorageCapacity = 102400;
        fogBatteryCapacity = 50000;
        fogMaxConnections = 50;
        
        // Cloud datacenter parameters
        cloudCpuCapacity = 100000;
        cloudRamCapacity = 1048576;
        cloudStorageCapacity = 10485760;
        cloudMaxConnections = 1000;
        
        // Task parameters
        taskMinCpu = 100;
        taskMaxCpu = 1000;
        taskMinRam = 10;
        taskMaxRam = 100;
        taskMinStorage = 5;
        taskMaxStorage = 50;
        taskMinDuration = 10;
        taskMaxDuration = 100;
        taskOffloadingEnabled = true;
        taskOffloadingThreshold = 0.7;
        taskSchedulingPolicy = "FIFO";
        
        // Network parameters
        networkBaseBandwidth = 100;
        networkBaseLatency = 10;
        networkVariabilityFactor = 0.2;
        networkCongestionProbability = 0.1;
        networkPacketLossProbability = 0.05;
        ioTMeshNetworkEnabled = true;
        edgeMeshNetworkEnabled = true;
        fogMeshNetworkEnabled = true;
        
        // Security parameters
        securityAttacksEnabled = true;
        securityAttackProbability = 0.05;
        securityDetectionProbability = 0.7;
        securityMitigationProbability = 0.6;
        securityAttackTypes = Arrays.asList(
            AttackType.DDOS,
            AttackType.DATA_THEFT,
            AttackType.EAVESDROPPING,
            AttackType.MAN_IN_THE_MIDDLE,
            AttackType.MALWARE,
            AttackType.PHYSICAL_TAMPERING
        );
        securityCountermeasureTypes = Arrays.asList(
            CountermeasureType.TRAFFIC_FILTERING,
            CountermeasureType.ENCRYPTION,
            CountermeasureType.SECURE_COMMUNICATION,
            CountermeasureType.AUTHENTICATION,
            CountermeasureType.MALWARE_SCANNING,
            CountermeasureType.PHYSICAL_SECURITY,
            CountermeasureType.INTRUSION_DETECTION
        );
        
        // Logging parameters
        consoleLoggingEnabled = true;
        fileLoggingEnabled = true;
        logFilePath = "logs/simulation.log";
        minLogLevel = LogLevel.INFO;
    }
    
    /**
     * Loads configuration from a Properties object
     * @param properties Properties object containing configuration values
     */
    private void loadFromProperties(Properties properties) {
        if (properties == null) {
            return;
        }
        
        // Load simulation parameters
        simulationDuration = Integer.parseInt(properties.getProperty("simulation.duration", String.valueOf(simulationDuration)));
        simulationAreaSize = Integer.parseInt(properties.getProperty("simulation.areaSize", String.valueOf(simulationAreaSize)));
        randomSeed = Long.parseLong(properties.getProperty("simulation.randomSeed", String.valueOf(randomSeed)));
        
        // Load device counts
        ioTDeviceCount = Integer.parseInt(properties.getProperty("device.iotCount", String.valueOf(ioTDeviceCount)));
        edgeNodeCount = Integer.parseInt(properties.getProperty("device.edgeCount", String.valueOf(edgeNodeCount)));
        fogNodeCount = Integer.parseInt(properties.getProperty("device.fogCount", String.valueOf(fogNodeCount)));
        cloudDatacenterCount = Integer.parseInt(properties.getProperty("device.cloudCount", String.valueOf(cloudDatacenterCount)));
        
        // Load IoT device parameters
        ioTMobilityPercentage = Integer.parseInt(properties.getProperty("iot.mobilityPercentage", String.valueOf(ioTMobilityPercentage)));
        ioTBatteryCapacity = Integer.parseInt(properties.getProperty("iot.batteryCapacity", String.valueOf(ioTBatteryCapacity)));
        ioTCpuCapacity = Integer.parseInt(properties.getProperty("iot.cpuCapacity", String.valueOf(ioTCpuCapacity)));
        ioTRamCapacity = Integer.parseInt(properties.getProperty("iot.ramCapacity", String.valueOf(ioTRamCapacity)));
        ioTStorageCapacity = Integer.parseInt(properties.getProperty("iot.storageCapacity", String.valueOf(ioTStorageCapacity)));
        ioTTaskGenerationRate = Double.parseDouble(properties.getProperty("iot.taskGenerationRate", String.valueOf(ioTTaskGenerationRate)));
        
        // Load IoT wireless types
        String wirelessTypesStr = properties.getProperty("iot.wirelessTypes");
        if (wirelessTypesStr != null && !wirelessTypesStr.isEmpty()) {
            ioTWirelessTypes = Arrays.asList(wirelessTypesStr.split(","));
        }
        
        // Load Edge node parameters
        edgeCpuCapacity = Integer.parseInt(properties.getProperty("edge.cpuCapacity", String.valueOf(edgeCpuCapacity)));
        edgeRamCapacity = Integer.parseInt(properties.getProperty("edge.ramCapacity", String.valueOf(edgeRamCapacity)));
        edgeStorageCapacity = Integer.parseInt(properties.getProperty("edge.storageCapacity", String.valueOf(edgeStorageCapacity)));
        edgeBatteryCapacity = Integer.parseInt(properties.getProperty("edge.batteryCapacity", String.valueOf(edgeBatteryCapacity)));
        edgeMaxConnections = Integer.parseInt(properties.getProperty("edge.maxConnections", String.valueOf(edgeMaxConnections)));
        
        // Load Fog node parameters
        fogCpuCapacity = Integer.parseInt(properties.getProperty("fog.cpuCapacity", String.valueOf(fogCpuCapacity)));
        fogRamCapacity = Integer.parseInt(properties.getProperty("fog.ramCapacity", String.valueOf(fogRamCapacity)));
        fogStorageCapacity = Integer.parseInt(properties.getProperty("fog.storageCapacity", String.valueOf(fogStorageCapacity)));
        fogBatteryCapacity = Integer.parseInt(properties.getProperty("fog.batteryCapacity", String.valueOf(fogBatteryCapacity)));
        fogMaxConnections = Integer.parseInt(properties.getProperty("fog.maxConnections", String.valueOf(fogMaxConnections)));
        
        // Load Cloud datacenter parameters
        cloudCpuCapacity = Integer.parseInt(properties.getProperty("cloud.cpuCapacity", String.valueOf(cloudCpuCapacity)));
        cloudRamCapacity = Integer.parseInt(properties.getProperty("cloud.ramCapacity", String.valueOf(cloudRamCapacity)));
        cloudStorageCapacity = Integer.parseInt(properties.getProperty("cloud.storageCapacity", String.valueOf(cloudStorageCapacity)));
        cloudMaxConnections = Integer.parseInt(properties.getProperty("cloud.maxConnections", String.valueOf(cloudMaxConnections)));
        
        // Task parameters
        taskMinCpu = getIntProperty(properties, "task.min.cpu", taskMinCpu);
        taskMaxCpu = getIntProperty(properties, "task.max.cpu", taskMaxCpu);
        taskMinRam = getIntProperty(properties, "task.min.ram", taskMinRam);
        taskMaxRam = getIntProperty(properties, "task.max.ram", taskMaxRam);
        taskMinStorage = getIntProperty(properties, "task.min.storage", taskMinStorage);
        taskMaxStorage = getIntProperty(properties, "task.max.storage", taskMaxStorage);
        taskMinDuration = getIntProperty(properties, "task.min.duration", taskMinDuration);
        taskMaxDuration = getIntProperty(properties, "task.max.duration", taskMaxDuration);
        taskOffloadingEnabled = getBooleanProperty(properties, "task.offloading.enabled", taskOffloadingEnabled);
        taskOffloadingThreshold = getDoubleProperty(properties, "task.offloading.threshold", taskOffloadingThreshold);
        taskSchedulingPolicy = getStringProperty(properties, "task.scheduling.policy", taskSchedulingPolicy);
        
        // Load Network parameters
        networkBaseBandwidth = Double.parseDouble(properties.getProperty("network.baseBandwidth", String.valueOf(networkBaseBandwidth)));
        networkBaseLatency = Double.parseDouble(properties.getProperty("network.baseLatency", String.valueOf(networkBaseLatency)));
        networkVariabilityFactor = Double.parseDouble(properties.getProperty("network.variabilityFactor", String.valueOf(networkVariabilityFactor)));
        networkCongestionProbability = Double.parseDouble(properties.getProperty("network.congestionProbability", String.valueOf(networkCongestionProbability)));
        networkPacketLossProbability = Double.parseDouble(properties.getProperty("network.packetLossProbability", String.valueOf(networkPacketLossProbability)));
        ioTMeshNetworkEnabled = Boolean.parseBoolean(properties.getProperty("network.iotMeshEnabled", String.valueOf(ioTMeshNetworkEnabled)));
        edgeMeshNetworkEnabled = Boolean.parseBoolean(properties.getProperty("network.edgeMeshEnabled", String.valueOf(edgeMeshNetworkEnabled)));
        fogMeshNetworkEnabled = Boolean.parseBoolean(properties.getProperty("network.fogMeshEnabled", String.valueOf(fogMeshNetworkEnabled)));
        
        // Load Security parameters
        securityAttacksEnabled = Boolean.parseBoolean(properties.getProperty("security.attacksEnabled", String.valueOf(securityAttacksEnabled)));
        securityAttackProbability = Double.parseDouble(properties.getProperty("security.attackProbability", String.valueOf(securityAttackProbability)));
        securityDetectionProbability = Double.parseDouble(properties.getProperty("security.detectionProbability", String.valueOf(securityDetectionProbability)));
        securityMitigationProbability = Double.parseDouble(properties.getProperty("security.mitigationProbability", String.valueOf(securityMitigationProbability)));
        
        // Load attack types
        String attackTypesStr = properties.getProperty("security.attackTypes");
        if (attackTypesStr != null && !attackTypesStr.isEmpty()) {
            securityAttackTypes = new ArrayList<>();
            String[] attackTypeNames = attackTypesStr.split(",");
            for (String name : attackTypeNames) {
                try {
                    securityAttackTypes.add(AttackType.valueOf(name.trim()));
                } catch (IllegalArgumentException e) {
                    // Skip invalid attack type
                }
            }
        }
        
        // Load countermeasure types
        String countermeasureTypesStr = properties.getProperty("security.countermeasureTypes");
        if (countermeasureTypesStr != null && !countermeasureTypesStr.isEmpty()) {
            securityCountermeasureTypes = new ArrayList<>();
            String[] countermeasureTypeNames = countermeasureTypesStr.split(",");
            for (String name : countermeasureTypeNames) {
                try {
                    securityCountermeasureTypes.add(CountermeasureType.valueOf(name.trim()));
                } catch (IllegalArgumentException e) {
                    // Skip invalid countermeasure type
                }
            }
        }
        
        // Load Logging parameters
        consoleLoggingEnabled = Boolean.parseBoolean(properties.getProperty("logging.consoleEnabled", String.valueOf(consoleLoggingEnabled)));
        fileLoggingEnabled = Boolean.parseBoolean(properties.getProperty("logging.fileEnabled", String.valueOf(fileLoggingEnabled)));
        logFilePath = properties.getProperty("logging.filePath", logFilePath);
        
        // Load log level
        String logLevelStr = properties.getProperty("logging.minLevel");
        if (logLevelStr != null && !logLevelStr.isEmpty()) {
            try {
                minLogLevel = LogLevel.valueOf(logLevelStr.trim());
            } catch (IllegalArgumentException e) {
                // Use default log level
            }
        }
    }
    
    private double getDoubleProperty(Properties properties, String key, double defaultValue) {
        String value = properties.getProperty(key);
        if (value == null || value.isEmpty()) {
            return defaultValue;
        }
        try {
            return Double.parseDouble(value);
        } catch (NumberFormatException e) {
            System.err.println("Invalid value for " + key + ": " + value + ". Using default: " + defaultValue);
            return defaultValue;
        }
    }
    
    private String getStringProperty(Properties properties, String key, String defaultValue) {
        String value = properties.getProperty(key);
        if (value == null || value.isEmpty()) {
            return defaultValue;
        }
        return value;
    }
    
    private int getIntProperty(Properties properties, String key, int defaultValue) {
        String value = properties.getProperty(key);
        if (value == null || value.isEmpty()) {
            return defaultValue;
        }
        try {
            return Integer.parseInt(value);
        } catch (NumberFormatException e) {
            System.err.println("Invalid value for " + key + ": " + value + ". Using default: " + defaultValue);
            return defaultValue;
        }
    }
    
    private boolean getBooleanProperty(Properties properties, String key, boolean defaultValue) {
        String value = properties.getProperty(key);
        if (value == null || value.isEmpty()) {
            return defaultValue;
        }
        return Boolean.parseBoolean(value);
    }
    
    // Getters for all configuration parameters
    
    // Simulation parameters
    public int getSimulationDuration() {
        return simulationDuration;
    }
    
    public int getSimulationAreaSize() {
        return simulationAreaSize;
    }
    
    public long getRandomSeed() {
        return randomSeed;
    }
    
    // Device counts
    public int getIoTDeviceCount() {
        return ioTDeviceCount;
    }
    
    public int getEdgeNodeCount() {
        return edgeNodeCount;
    }
    
    public int getFogNodeCount() {
        return fogNodeCount;
    }
    
    public int getCloudDatacenterCount() {
        return cloudDatacenterCount;
    }
    
    // IoT device parameters
    public int getIoTMobilityPercentage() {
        return ioTMobilityPercentage;
    }
    
    public int getIoTBatteryCapacity() {
        return ioTBatteryCapacity;
    }
    
    public int getIoTCpuCapacity() {
        return ioTCpuCapacity;
    }
    
    public int getIoTRamCapacity() {
        return ioTRamCapacity;
    }
    
    public int getIoTStorageCapacity() {
        return ioTStorageCapacity;
    }
    
    public double getIoTTaskGenerationRate() {
        return ioTTaskGenerationRate;
    }
    
    public List<String> getIoTWirelessTypes() {
        return new ArrayList<>(ioTWirelessTypes);
    }
    
    // Edge node parameters
    public int getEdgeCpuCapacity() {
        return edgeCpuCapacity;
    }
    
    public int getEdgeRamCapacity() {
        return edgeRamCapacity;
    }
    
    public int getEdgeStorageCapacity() {
        return edgeStorageCapacity;
    }
    
    public int getEdgeBatteryCapacity() {
        return edgeBatteryCapacity;
    }
    
    public int getEdgeMaxConnections() {
        return edgeMaxConnections;
    }
    
    // Fog node parameters
    public int getFogCpuCapacity() {
        return fogCpuCapacity;
    }
    
    public int getFogRamCapacity() {
        return fogRamCapacity;
    }
    
    public int getFogStorageCapacity() {
        return fogStorageCapacity;
    }
    
    public int getFogBatteryCapacity() {
        return fogBatteryCapacity;
    }
    
    public int getFogMaxConnections() {
        return fogMaxConnections;
    }
    
    // Cloud datacenter parameters
    public int getCloudCpuCapacity() {
        return cloudCpuCapacity;
    }
    
    public int getCloudRamCapacity() {
        return cloudRamCapacity;
    }
    
    public int getCloudStorageCapacity() {
        return cloudStorageCapacity;
    }
    
    public int getCloudMaxConnections() {
        return cloudMaxConnections;
    }
    
    // Task parameters
    public int getTaskMinCpu() {
        return taskMinCpu;
    }
    
    public int getTaskMaxCpu() {
        return taskMaxCpu;
    }
    
    public int getTaskMinRam() {
        return taskMinRam;
    }
    
    public int getTaskMaxRam() {
        return taskMaxRam;
    }
    
    public int getTaskMinStorage() {
        return taskMinStorage;
    }
    
    public int getTaskMaxStorage() {
        return taskMaxStorage;
    }
    
    public int getTaskMinDuration() {
        return taskMinDuration;
    }
    
    public int getTaskMaxDuration() {
        return taskMaxDuration;
    }
    
    public boolean isTaskOffloadingEnabled() {
        return taskOffloadingEnabled;
    }
    
    public double getTaskOffloadingThreshold() {
        return taskOffloadingThreshold;
    }
    
    // Network parameters
    public double getNetworkBaseBandwidth() {
        return networkBaseBandwidth;
    }
    
    public double getNetworkBaseLatency() {
        return networkBaseLatency;
    }
    
    public double getNetworkVariabilityFactor() {
        return networkVariabilityFactor;
    }
    
    public double getNetworkCongestionProbability() {
        return networkCongestionProbability;
    }
    
    public double getNetworkPacketLossProbability() {
        return networkPacketLossProbability;
    }
    
    public boolean getIoTMeshNetworkEnabled() {
        return ioTMeshNetworkEnabled;
    }
    
    public boolean getEdgeMeshNetworkEnabled() {
        return edgeMeshNetworkEnabled;
    }
    
    public boolean getFogMeshNetworkEnabled() {
        return fogMeshNetworkEnabled;
    }
    
    // Security parameters
    public boolean isSecurityAttacksEnabled() {
        return securityAttacksEnabled;
    }
    
    public double getSecurityAttackProbability() {
        return securityAttackProbability;
    }
    
    public double getSecurityDetectionProbability() {
        return securityDetectionProbability;
    }
    
    public double getSecurityMitigationProbability() {
        return securityMitigationProbability;
    }
    
    public List<AttackType> getSecurityAttackTypes() {
        return new ArrayList<>(securityAttackTypes);
    }
    
    public List<CountermeasureType> getSecurityCountermeasureTypes() {
        return new ArrayList<>(securityCountermeasureTypes);
    }
    
    // Logging parameters
    public boolean isConsoleLoggingEnabled() {
        return consoleLoggingEnabled;
    }
    
    public boolean isFileLoggingEnabled() {
        return fileLoggingEnabled;
    }
    
    public String getLogFilePath() {
        return logFilePath;
    }
    
    public LogLevel getMinLogLevel() {
        return minLogLevel;
    }
    
    /**
     * Gets the task scheduling policy
     * @return Task scheduling policy (e.g., "FIFO", "PRIORITY", "DEADLINE")
     */
    public String getTaskSchedulingPolicy() {
        return taskSchedulingPolicy;
    }
    
    /**
     * Get the task output size
     * @return Task output size in KB
     */
    public int getTaskOutputSize() {
        return taskOutputSize;
    }
}
