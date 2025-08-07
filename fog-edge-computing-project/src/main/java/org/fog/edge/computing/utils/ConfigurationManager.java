package org.fog.edge.computing.utils;

import java.io.FileInputStream;
import java.io.IOException;
import java.util.Properties;

/**
 * Configuration Manager for the Fog and Edge Computing project
 * 
 * This class provides centralized configuration management following
 * the PureEdgeSim paper's approach to externalized configuration.
 * It supports both properties files and programmatic configuration.
 * 
 * @author Student
 * @version 1.0
 */
public class ConfigurationManager {
    
    private static ConfigurationManager instance;
    private Properties properties;
    
    // Default configuration values
    private static final String DEFAULT_SIMULATION_DURATION = "1800"; // 30 minutes
    private static final String DEFAULT_UPDATE_INTERVAL = "0.01";
    private static final String DEFAULT_EDGE_DEVICES = "100";
    private static final String DEFAULT_EDGE_DATACENTERS = "10";
    private static final String DEFAULT_CLOUD_DATACENTERS = "1";
    private static final String DEFAULT_WAN_BANDWIDTH = "20.0";
    private static final String DEFAULT_LAN_BANDWIDTH = "300.0";
    private static final String DEFAULT_WAN_LATENCY = "100.0";
    private static final String DEFAULT_LAN_LATENCY = "5.0";
    
    /**
     * Private constructor for singleton pattern
     */
    private ConfigurationManager() {
        properties = new Properties();
        loadDefaultConfiguration();
    }
    
    /**
     * Get singleton instance
     * 
     * @return ConfigurationManager instance
     */
    public static synchronized ConfigurationManager getInstance() {
        if (instance == null) {
            instance = new ConfigurationManager();
        }
        return instance;
    }
    
    /**
     * Load configuration from file
     * 
     * @param configFilePath Path to configuration file
     * @throws IOException if file cannot be read
     */
    public void loadConfiguration(String configFilePath) throws IOException {
        try (FileInputStream fis = new FileInputStream(configFilePath)) {
            properties.load(fis);
            System.out.println("Configuration loaded from: " + configFilePath);
        }
    }
    
    /**
     * Load default configuration values
     */
    private void loadDefaultConfiguration() {
        // Simulation parameters
        properties.setProperty("simulation.duration", DEFAULT_SIMULATION_DURATION);
        properties.setProperty("simulation.update_interval", DEFAULT_UPDATE_INTERVAL);
        
        // Device configuration
        properties.setProperty("devices.edge_devices", DEFAULT_EDGE_DEVICES);
        properties.setProperty("devices.edge_datacenters", DEFAULT_EDGE_DATACENTERS);
        properties.setProperty("devices.cloud_datacenters", DEFAULT_CLOUD_DATACENTERS);
        
        // Network configuration
        properties.setProperty("network.wan_bandwidth", DEFAULT_WAN_BANDWIDTH);
        properties.setProperty("network.lan_bandwidth", DEFAULT_LAN_BANDWIDTH);
        properties.setProperty("network.wan_latency", DEFAULT_WAN_LATENCY);
        properties.setProperty("network.lan_latency", DEFAULT_LAN_LATENCY);
        
        // Orchestration configuration
        properties.setProperty("orchestration.algorithm", "FuzzyDecisionTree");
        properties.setProperty("orchestration.enable_comparison", "true");
        
        // Output configuration
        properties.setProperty("output.directory", "./output");
        properties.setProperty("output.generate_graphs", "true");
        properties.setProperty("output.csv_format", "true");
        
        System.out.println("Default configuration loaded successfully.");
    }
    
    /**
     * Get configuration value as string
     * 
     * @param key Configuration key
     * @return Configuration value
     */
    public String getString(String key) {
        return properties.getProperty(key);
    }
    
    /**
     * Get configuration value as integer
     * 
     * @param key Configuration key
     * @return Configuration value as integer
     */
    public int getInt(String key) {
        String value = properties.getProperty(key);
        return value != null ? Integer.parseInt(value) : 0;
    }
    
    /**
     * Get configuration value as double
     * 
     * @param key Configuration key
     * @return Configuration value as double
     */
    public double getDouble(String key) {
        String value = properties.getProperty(key);
        return value != null ? Double.parseDouble(value) : 0.0;
    }
    
    /**
     * Get configuration value as boolean
     * 
     * @param key Configuration key
     * @return Configuration value as boolean
     */
    public boolean getBoolean(String key) {
        String value = properties.getProperty(key);
        return value != null ? Boolean.parseBoolean(value) : false;
    }
    
    /**
     * Set configuration value
     * 
     * @param key Configuration key
     * @param value Configuration value
     */
    public void setProperty(String key, String value) {
        properties.setProperty(key, value);
    }
    
    /**
     * Get all properties
     * 
     * @return Properties object
     */
    public Properties getAllProperties() {
        return new Properties(properties);
    }
    
    /**
     * Print current configuration
     */
    public void printConfiguration() {
        System.out.println("\n=== CURRENT CONFIGURATION ===");
        System.out.println("Simulation Duration: " + getString("simulation.duration") + " seconds");
        System.out.println("Update Interval: " + getString("simulation.update_interval") + " seconds");
        System.out.println("Edge Devices: " + getString("devices.edge_devices"));
        System.out.println("Edge Datacenters: " + getString("devices.edge_datacenters"));
        System.out.println("Cloud Datacenters: " + getString("devices.cloud_datacenters"));
        System.out.println("WAN Bandwidth: " + getString("network.wan_bandwidth") + " Mbps");
        System.out.println("LAN Bandwidth: " + getString("network.lan_bandwidth") + " Mbps");
        System.out.println("WAN Latency: " + getString("network.wan_latency") + " ms");
        System.out.println("LAN Latency: " + getString("network.lan_latency") + " ms");
        System.out.println("Orchestration Algorithm: " + getString("orchestration.algorithm"));
        System.out.println("Enable Comparison: " + getString("orchestration.enable_comparison"));
        System.out.println("Output Directory: " + getString("output.directory"));
        System.out.println("Generate Graphs: " + getString("output.generate_graphs"));
        System.out.println("CSV Format: " + getString("output.csv_format"));
        System.out.println("===============================\n");
    }
}
