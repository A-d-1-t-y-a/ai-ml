package com.nci.fogedge.utils;

import org.yaml.snakeyaml.Yaml;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.InputStream;
import java.util.Map;
import java.util.HashMap;

/**
 * Configuration Manager for Fog and Edge Computing System
 * 
 * This class manages system configuration settings for the entire fog and edge computing system.
 * It loads configuration from YAML files and provides access to various system parameters.
 * 
 * @author National College of Ireland - Fog and Edge Computing Project
 * @version 1.0.0
 * @since 2024
 */
public class ConfigurationManager {
    
    private static final Logger logger = LoggerFactory.getLogger(ConfigurationManager.class);
    
    private static final String CONFIG_FILE = "config/application.yml";
    private static final String DEFAULT_CONFIG_FILE = "config/default.yml";
    
    private Map<String, Object> configuration;
    
    /**
     * Constructor for ConfigurationManager
     */
    public ConfigurationManager() {
        this.configuration = new HashMap<>();
        logger.info("ConfigurationManager initialized");
    }
    
    /**
     * Load configuration from file
     */
    public void loadConfiguration() {
        try {
            // Try to load custom configuration first
            if (!loadFromFile(CONFIG_FILE)) {
                // Fall back to default configuration
                loadFromFile(DEFAULT_CONFIG_FILE);
            }
            
            // Set default values for missing configurations
            setDefaultValues();
            
            logger.info("Configuration loaded successfully");
            
        } catch (Exception e) {
            logger.error("Error loading configuration", e);
            setDefaultValues();
        }
    }
    
    /**
     * Load configuration from specific file
     * 
     * @param filename Configuration file name
     * @return True if loaded successfully
     */
    private boolean loadFromFile(String filename) {
        try (InputStream input = getClass().getClassLoader().getResourceAsStream(filename)) {
            if (input == null) {
                logger.warn("Configuration file not found: {}", filename);
                return false;
            }
            
            Yaml yaml = new Yaml();
            Map<String, Object> loadedConfig = yaml.load(input);
            
            if (loadedConfig != null) {
                configuration.putAll(loadedConfig);
                logger.info("Configuration loaded from: {}", filename);
                return true;
            }
            
            return false;
            
        } catch (Exception e) {
            logger.error("Error loading configuration from: {}", filename, e);
            return false;
        }
    }
    
    /**
     * Set default configuration values
     */
    private void setDefaultValues() {
        // System configuration
        configuration.putIfAbsent("system.name", "Fog and Edge Computing System");
        configuration.putIfAbsent("system.version", "1.0.0");
        configuration.putIfAbsent("system.environment", "development");
        
        // IoT device configuration
        configuration.putIfAbsent("iot.device.count", 10);
        configuration.putIfAbsent("iot.device.types", new String[]{"TEMPERATURE", "HUMIDITY", "PRESSURE", "LIGHT", "MOTION"});
        configuration.putIfAbsent("iot.data.interval", 5000); // 5 seconds
        configuration.putIfAbsent("iot.transmission.interval", 10000); // 10 seconds
        
        // Edge node configuration
        configuration.putIfAbsent("edge.node.count", 3);
        configuration.putIfAbsent("edge.node.types", new String[]{"GATEWAY", "DATA_PROCESSING", "ANALYTICS"});
        configuration.putIfAbsent("edge.processing.capacity", 1000); // MB/s
        configuration.putIfAbsent("edge.storage.capacity", 10000); // MB
        
        // Cloud service configuration
        configuration.putIfAbsent("cloud.service.count", 2);
        configuration.putIfAbsent("cloud.service.types", new String[]{"DATA_ANALYTICS", "MACHINE_LEARNING"});
        configuration.putIfAbsent("cloud.processing.capacity", 5000); // MB/s
        configuration.putIfAbsent("cloud.storage.capacity", 100000); // MB
        
        // Network configuration
        configuration.putIfAbsent("network.lorawan.enabled", true);
        configuration.putIfAbsent("network.lorawan.frequency", 868.0); // MHz
        configuration.putIfAbsent("network.lorawan.bandwidth", 125.0); // kHz
        configuration.putIfAbsent("network.lorawan.spreading_factor", 7);
        configuration.putIfAbsent("network.lorawan.tx_power", 14); // dBm
        
        configuration.putIfAbsent("network.5g.enabled", true);
        configuration.putIfAbsent("network.5g.frequency", 3500.0); // MHz
        configuration.putIfAbsent("network.5g.bandwidth", 100.0); // MHz
        configuration.putIfAbsent("network.5g.tx_power", 23); // dBm
        
        // Performance configuration
        configuration.putIfAbsent("performance.target.latency", 50.0); // ms
        configuration.putIfAbsent("performance.target.throughput", 100.0); // Mbps
        configuration.putIfAbsent("performance.target.energy_efficiency", 80.0); // %
        configuration.putIfAbsent("performance.target.data_reduction", 70.0); // %
        
        // Monitoring configuration
        configuration.putIfAbsent("monitoring.metrics.interval", 30); // seconds
        configuration.putIfAbsent("monitoring.export.interval", 60); // seconds
        configuration.putIfAbsent("monitoring.analysis.interval", 120); // seconds
        configuration.putIfAbsent("monitoring.log.level", "INFO");
        
        // Data export configuration
        configuration.putIfAbsent("export.enabled", true);
        configuration.putIfAbsent("export.format", "CSV");
        configuration.putIfAbsent("export.directory", "data");
        configuration.putIfAbsent("export.retention.days", 30);
        
        logger.info("Default configuration values set");
    }
    
    /**
     * Get configuration value
     * 
     * @param key Configuration key
     * @return Configuration value
     */
    @SuppressWarnings("unchecked")
    public <T> T getValue(String key) {
        return (T) configuration.get(key);
    }
    
    /**
     * Get configuration value with default
     * 
     * @param key Configuration key
     * @param defaultValue Default value
     * @return Configuration value or default
     */
    @SuppressWarnings("unchecked")
    public <T> T getValue(String key, T defaultValue) {
        T value = (T) configuration.get(key);
        return value != null ? value : defaultValue;
    }
    
    /**
     * Set configuration value
     * 
     * @param key Configuration key
     * @param value Configuration value
     */
    public void setValue(String key, Object value) {
        configuration.put(key, value);
        logger.debug("Configuration updated: {} = {}", key, value);
    }
    
    /**
     * Get IoT device count
     * 
     * @return Number of IoT devices
     */
    public int getIoTDeviceCount() {
        return getValue("iot.device.count", 10);
    }
    
    /**
     * Get IoT device types
     * 
     * @return Array of device types
     */
    public String[] getIoTDeviceTypes() {
        return getValue("iot.device.types", new String[]{"TEMPERATURE", "HUMIDITY", "PRESSURE", "LIGHT", "MOTION"});
    }
    
    /**
     * Get IoT data interval
     * 
     * @return Data collection interval in milliseconds
     */
    public int getIoTDataInterval() {
        return getValue("iot.data.interval", 5000);
    }
    
    /**
     * Get IoT transmission interval
     * 
     * @return Transmission interval in milliseconds
     */
    public int getIoTTransmissionInterval() {
        return getValue("iot.transmission.interval", 10000);
    }
    
    /**
     * Get edge node count
     * 
     * @return Number of edge nodes
     */
    public int getEdgeNodeCount() {
        return getValue("edge.node.count", 3);
    }
    
    /**
     * Get edge node types
     * 
     * @return Array of node types
     */
    public String[] getEdgeNodeTypes() {
        return getValue("edge.node.types", new String[]{"GATEWAY", "DATA_PROCESSING", "ANALYTICS"});
    }
    
    /**
     * Get edge processing capacity
     * 
     * @return Processing capacity in MB/s
     */
    public int getEdgeProcessingCapacity() {
        return getValue("edge.processing.capacity", 1000);
    }
    
    /**
     * Get edge storage capacity
     * 
     * @return Storage capacity in MB
     */
    public int getEdgeStorageCapacity() {
        return getValue("edge.storage.capacity", 10000);
    }
    
    /**
     * Get cloud service count
     * 
     * @return Number of cloud services
     */
    public int getCloudServiceCount() {
        return getValue("cloud.service.count", 2);
    }
    
    /**
     * Get cloud service types
     * 
     * @return Array of service types
     */
    public String[] getCloudServiceTypes() {
        return getValue("cloud.service.types", new String[]{"DATA_ANALYTICS", "MACHINE_LEARNING"});
    }
    
    /**
     * Get cloud processing capacity
     * 
     * @return Processing capacity in MB/s
     */
    public int getCloudProcessingCapacity() {
        return getValue("cloud.processing.capacity", 5000);
    }
    
    /**
     * Get cloud storage capacity
     * 
     * @return Storage capacity in MB
     */
    public int getCloudStorageCapacity() {
        return getValue("cloud.storage.capacity", 100000);
    }
    
    /**
     * Check if LoRaWAN is enabled
     * 
     * @return True if LoRaWAN is enabled
     */
    public boolean isLoRaWANEnabled() {
        return getValue("network.lorawan.enabled", true);
    }
    
    /**
     * Get LoRaWAN frequency
     * 
     * @return Frequency in MHz
     */
    public double getLoRaWANFrequency() {
        return getValue("network.lorawan.frequency", 868.0);
    }
    
    /**
     * Get LoRaWAN bandwidth
     * 
     * @return Bandwidth in kHz
     */
    public double getLoRaWANBandwidth() {
        return getValue("network.lorawan.bandwidth", 125.0);
    }
    
    /**
     * Check if 5G is enabled
     * 
     * @return True if 5G is enabled
     */
    public boolean is5GEnabled() {
        return getValue("network.5g.enabled", true);
    }
    
    /**
     * Get 5G frequency
     * 
     * @return Frequency in MHz
     */
    public double get5GFrequency() {
        return getValue("network.5g.frequency", 3500.0);
    }
    
    /**
     * Get 5G bandwidth
     * 
     * @return Bandwidth in MHz
     */
    public double get5GBandwidth() {
        return getValue("network.5g.bandwidth", 100.0);
    }
    
    /**
     * Get target latency
     * 
     * @return Target latency in milliseconds
     */
    public double getTargetLatency() {
        return getValue("performance.target.latency", 50.0);
    }
    
    /**
     * Get target throughput
     * 
     * @return Target throughput in Mbps
     */
    public double getTargetThroughput() {
        return getValue("performance.target.throughput", 100.0);
    }
    
    /**
     * Get target energy efficiency
     * 
     * @return Target energy efficiency percentage
     */
    public double getTargetEnergyEfficiency() {
        return getValue("performance.target.energy_efficiency", 80.0);
    }
    
    /**
     * Get target data reduction
     * 
     * @return Target data reduction percentage
     */
    public double getTargetDataReduction() {
        return getValue("performance.target.data_reduction", 70.0);
    }
    
    /**
     * Get metrics collection interval
     * 
     * @return Interval in seconds
     */
    public int getMetricsInterval() {
        return getValue("monitoring.metrics.interval", 30);
    }
    
    /**
     * Get export interval
     * 
     * @return Interval in seconds
     */
    public int getExportInterval() {
        return getValue("monitoring.export.interval", 60);
    }
    
    /**
     * Get analysis interval
     * 
     * @return Interval in seconds
     */
    public int getAnalysisInterval() {
        return getValue("monitoring.analysis.interval", 120);
    }
    
    /**
     * Get log level
     * 
     * @return Log level
     */
    public String getLogLevel() {
        return getValue("monitoring.log.level", "INFO");
    }
    
    /**
     * Check if export is enabled
     * 
     * @return True if export is enabled
     */
    public boolean isExportEnabled() {
        return getValue("export.enabled", true);
    }
    
    /**
     * Get export format
     * 
     * @return Export format
     */
    public String getExportFormat() {
        return getValue("export.format", "CSV");
    }
    
    /**
     * Get export directory
     * 
     * @return Export directory
     */
    public String getExportDirectory() {
        return getValue("export.directory", "data");
    }
    
    /**
     * Get all configuration
     * 
     * @return Configuration map
     */
    public Map<String, Object> getAllConfiguration() {
        return new HashMap<>(configuration);
    }
    
    @Override
    public String toString() {
        return String.format("ConfigurationManager{entries=%d, system=%s, version=%s}",
            configuration.size(),
            getValue("system.name", "Unknown"),
            getValue("system.version", "Unknown"));
    }
} 