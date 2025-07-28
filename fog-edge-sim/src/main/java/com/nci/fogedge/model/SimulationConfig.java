package com.nci.fogedge.model;

import java.io.FileInputStream;
import java.io.IOException;
import java.util.Properties;

/**
 * Configuration class for the Fog and Edge Computing Simulation.
 * This class loads and manages all simulation parameters.
 */
public class SimulationConfig {
    // Simulation parameters
    private int simulationDuration;
    private boolean mobilityEnabled;
    private boolean securityEnabled;
    private String logLevel;
    
    // Device parameters
    private int numIoTDevices;
    private int numEdgeNodes;
    private int numFogNodes;
    private int numCloudDatacenters;
    
    // Network parameters
    private double wanBandwidth;
    private double lanBandwidth;
    private double wanLatency;
    private double lanLatency;
    private double wirelessRange;
    
    // Task parameters
    private int taskGenerationRate;
    private int taskLength;
    private int taskInputSize;
    private int taskOutputSize;
    
    // Security parameters
    private double attackProbability;
    private double detectionRate;
    private double securityOverhead;
    
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
     * Loads default configuration values
     */
    private void loadDefaults() {
        // Simulation defaults
        simulationDuration = 1000; // simulation ticks
        mobilityEnabled = true;
        securityEnabled = true;
        logLevel = "INFO";
        
        // Device defaults
        numIoTDevices = 100;
        numEdgeNodes = 10;
        numFogNodes = 5;
        numCloudDatacenters = 1;
        
        // Network defaults
        wanBandwidth = 100.0; // Mbps
        lanBandwidth = 1000.0; // Mbps
        wanLatency = 100.0; // ms
        lanLatency = 5.0; // ms
        wirelessRange = 100.0; // meters
        
        // Task defaults
        taskGenerationRate = 5; // tasks per device per 100 ticks
        taskLength = 1000; // MI (Million Instructions)
        taskInputSize = 1000; // KB
        taskOutputSize = 100; // KB
        
        // Security defaults
        attackProbability = 0.01; // 1% chance of attack per tick
        detectionRate = 0.8; // 80% detection rate
        securityOverhead = 0.1; // 10% overhead
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
            
            // Load simulation parameters
            simulationDuration = Integer.parseInt(properties.getProperty("simulation.duration", String.valueOf(simulationDuration)));
            mobilityEnabled = Boolean.parseBoolean(properties.getProperty("simulation.mobility.enabled", String.valueOf(mobilityEnabled)));
            securityEnabled = Boolean.parseBoolean(properties.getProperty("simulation.security.enabled", String.valueOf(securityEnabled)));
            logLevel = properties.getProperty("simulation.log.level", logLevel);
            
            // Load device parameters
            numIoTDevices = Integer.parseInt(properties.getProperty("devices.iot.count", String.valueOf(numIoTDevices)));
            numEdgeNodes = Integer.parseInt(properties.getProperty("devices.edge.count", String.valueOf(numEdgeNodes)));
            numFogNodes = Integer.parseInt(properties.getProperty("devices.fog.count", String.valueOf(numFogNodes)));
            numCloudDatacenters = Integer.parseInt(properties.getProperty("devices.cloud.count", String.valueOf(numCloudDatacenters)));
            
            // Load network parameters
            wanBandwidth = Double.parseDouble(properties.getProperty("network.wan.bandwidth", String.valueOf(wanBandwidth)));
            lanBandwidth = Double.parseDouble(properties.getProperty("network.lan.bandwidth", String.valueOf(lanBandwidth)));
            wanLatency = Double.parseDouble(properties.getProperty("network.wan.latency", String.valueOf(wanLatency)));
            lanLatency = Double.parseDouble(properties.getProperty("network.lan.latency", String.valueOf(lanLatency)));
            wirelessRange = Double.parseDouble(properties.getProperty("network.wireless.range", String.valueOf(wirelessRange)));
            
            // Load task parameters
            taskGenerationRate = Integer.parseInt(properties.getProperty("tasks.generation.rate", String.valueOf(taskGenerationRate)));
            taskLength = Integer.parseInt(properties.getProperty("tasks.length", String.valueOf(taskLength)));
            taskInputSize = Integer.parseInt(properties.getProperty("tasks.input.size", String.valueOf(taskInputSize)));
            taskOutputSize = Integer.parseInt(properties.getProperty("tasks.output.size", String.valueOf(taskOutputSize)));
            
            // Load security parameters
            attackProbability = Double.parseDouble(properties.getProperty("security.attack.probability", String.valueOf(attackProbability)));
            detectionRate = Double.parseDouble(properties.getProperty("security.detection.rate", String.valueOf(detectionRate)));
            securityOverhead = Double.parseDouble(properties.getProperty("security.overhead", String.valueOf(securityOverhead)));
        }
    }
    
    // Getters and setters for all configuration parameters
    
    public int getSimulationDuration() {
        return simulationDuration;
    }

    public boolean isMobilityEnabled() {
        return mobilityEnabled;
    }

    public boolean isSecurityEnabled() {
        return securityEnabled;
    }

    public String getLogLevel() {
        return logLevel;
    }

    public int getNumIoTDevices() {
        return numIoTDevices;
    }

    public int getNumEdgeNodes() {
        return numEdgeNodes;
    }

    public int getNumFogNodes() {
        return numFogNodes;
    }

    public int getNumCloudDatacenters() {
        return numCloudDatacenters;
    }

    public double getWanBandwidth() {
        return wanBandwidth;
    }

    public double getLanBandwidth() {
        return lanBandwidth;
    }

    public double getWanLatency() {
        return wanLatency;
    }

    public double getLanLatency() {
        return lanLatency;
    }

    public double getWirelessRange() {
        return wirelessRange;
    }

    public int getTaskGenerationRate() {
        return taskGenerationRate;
    }

    public int getTaskLength() {
        return taskLength;
    }

    public int getTaskInputSize() {
        return taskInputSize;
    }

    public int getTaskOutputSize() {
        return taskOutputSize;
    }

    public double getAttackProbability() {
        return attackProbability;
    }

    public double getDetectionRate() {
        return detectionRate;
    }

    public double getSecurityOverhead() {
        return securityOverhead;
    }
}
