package org.nci.fogedge.util;

import org.cloudbus.cloudsim.Log;
import org.json.JSONObject;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Properties;

/**
 * Manages configuration settings for the fog computing simulation.
 * Handles loading, saving, and accessing configuration parameters.
 */
public class ConfigurationManager {
    
    private Properties properties;
    private static final String CONFIG_FILE = "config.properties";
    private static final String DEFAULT_CONFIG_JSON = "{\n" +
            "  \"simulation\": {\n" +
            "    \"durationMs\": 10000,\n" +
            "    \"securityEnabled\": true\n" +
            "  },\n" +
            "  \"topology\": {\n" +
            "    \"numIoTDevices\": 50,\n" +
            "    \"numEdgeNodes\": 5,\n" +
            "    \"numFogNodes\": 2\n" +
            "  },\n" +
            "  \"iot\": {\n" +
            "    \"dataGenerationRateKBs\": 10.0,\n" +
            "    \"energyCapacityJ\": 1000.0\n" +
            "  },\n" +
            "  \"edge\": {\n" +
            "    \"processingCapacityMIPS\": 2000.0,\n" +
            "    \"storageCapacityMB\": 1000.0,\n" +
            "    \"bandwidthMbps\": 100.0\n" +
            "  },\n" +
            "  \"fog\": {\n" +
            "    \"processingCapacityMIPS\": 10000.0,\n" +
            "    \"storageCapacityGB\": 10.0,\n" +
            "    \"bandwidthMbps\": 1000.0\n" +
            "  },\n" +
            "  \"security\": {\n" +
            "    \"encryptionAlgorithm\": \"AES\",\n" +
            "    \"keySize\": 256,\n" +
            "    \"authenticationEnabled\": true,\n" +
            "    \"intrusionDetectionEnabled\": true\n" +
            "  }\n" +
            "}";
    
    /**
     * Creates a new ConfigurationManager and loads configuration
     */
    public ConfigurationManager() {
        properties = new Properties();
        loadConfiguration();
    }
    
    /**
     * Loads configuration from the config file or creates default if not found
     */
    private void loadConfiguration() {
        File configFile = new File(CONFIG_FILE);
        
        if (configFile.exists()) {
            try (FileInputStream fis = new FileInputStream(configFile)) {
                properties.load(fis);
                Log.printLine("Configuration loaded from " + CONFIG_FILE);
            } catch (IOException e) {
                Log.printLine("Error loading configuration: " + e.getMessage());
                createDefaultConfiguration();
            }
        } else {
            createDefaultConfiguration();
        }
    }
    
    /**
     * Creates a default configuration file
     */
    private void createDefaultConfiguration() {
        try {
            // Parse the default JSON configuration
            JSONObject config = new JSONObject(DEFAULT_CONFIG_JSON);
            
            // Convert JSON to Properties
            flattenJsonToProperties(config, "", properties);
            
            // Save to file
            try (FileOutputStream fos = new FileOutputStream(CONFIG_FILE)) {
                properties.store(fos, "Default Fog Computing Simulation Configuration");
                Log.printLine("Default configuration created at " + CONFIG_FILE);
            }
        } catch (Exception e) {
            Log.printLine("Error creating default configuration: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    /**
     * Recursively flattens a JSON object to Properties
     * 
     * @param json JSON object to flatten
     * @param prefix Current key prefix
     * @param props Properties to populate
     */
    private void flattenJsonToProperties(JSONObject json, String prefix, Properties props) {
        for (String key : json.keySet()) {
            String fullKey = prefix.isEmpty() ? key : prefix + "." + key;
            Object value = json.get(key);
            
            if (value instanceof JSONObject) {
                flattenJsonToProperties((JSONObject) value, fullKey, props);
            } else {
                props.setProperty(fullKey, value.toString());
            }
        }
    }
    
    /**
     * Gets a string property value
     * 
     * @param key Property key
     * @param defaultValue Default value if not found
     * @return Property value
     */
    public String getString(String key, String defaultValue) {
        return properties.getProperty(key, defaultValue);
    }
    
    /**
     * Gets an integer property value
     * 
     * @param key Property key
     * @param defaultValue Default value if not found
     * @return Property value
     */
    public int getInt(String key, int defaultValue) {
        try {
            return Integer.parseInt(properties.getProperty(key, String.valueOf(defaultValue)));
        } catch (NumberFormatException e) {
            return defaultValue;
        }
    }
    
    /**
     * Gets a double property value
     * 
     * @param key Property key
     * @param defaultValue Default value if not found
     * @return Property value
     */
    public double getDouble(String key, double defaultValue) {
        try {
            return Double.parseDouble(properties.getProperty(key, String.valueOf(defaultValue)));
        } catch (NumberFormatException e) {
            return defaultValue;
        }
    }
    
    /**
     * Gets a boolean property value
     * 
     * @param key Property key
     * @param defaultValue Default value if not found
     * @return Property value
     */
    public boolean getBoolean(String key, boolean defaultValue) {
        return Boolean.parseBoolean(properties.getProperty(key, String.valueOf(defaultValue)));
    }
    
    /**
     * Sets a property value
     * 
     * @param key Property key
     * @param value Property value
     */
    public void setProperty(String key, String value) {
        properties.setProperty(key, value);
    }
    
    /**
     * Saves the current configuration to file
     */
    public void saveConfiguration() {
        try (FileOutputStream fos = new FileOutputStream(CONFIG_FILE)) {
            properties.store(fos, "Fog Computing Simulation Configuration");
            Log.printLine("Configuration saved to " + CONFIG_FILE);
        } catch (IOException e) {
            Log.printLine("Error saving configuration: " + e.getMessage());
            e.printStackTrace();
        }
    }
}
