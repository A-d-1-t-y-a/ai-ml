package org.nci.fogedge.util;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.nci.fogedge.model.SimulationConfig;
import org.nci.fogedge.security.AttackType;
import org.nci.fogedge.security.SecurityLevel;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Properties;

/**
 * Utility class to load and manage simulation configuration
 * Based on the paper: "An Overview of Fog Computing and Edge Computing Security and Privacy Issues"
 * (Sensors 2021, 21, 8226, https://doi.org/10.3390/s21248226)
 */
public class ConfigurationManager {
    private static final Logger logger = LogManager.getLogger(ConfigurationManager.class);
    private static final String CONFIG_FILE = "simulation.properties";
    private static SimulationConfig customConfig = null;
    
    /**
     * Set a custom configuration to override the properties file
     * @param config The configuration to set
     */
    public static void setConfig(SimulationConfig config) {
        customConfig = config;
        logger.info("Custom configuration set: {}", config);
    }
    
    /**
     * Load simulation configuration from properties file
     * @return The loaded configuration
     */
    public static SimulationConfig loadConfiguration() {
        // If custom config is set, return it
        if (customConfig != null) {
            logger.info("Using custom configuration");
            return customConfig;
        }
        
        SimulationConfig config = new SimulationConfig();
        Properties properties = new Properties();
        
        try {
            // Try to load from file
            try (InputStream input = new FileInputStream(CONFIG_FILE)) {
                properties.load(input);
                logger.info("Loaded configuration from file: {}", CONFIG_FILE);
            } catch (IOException e) {
                // If file not found, load from classpath
                try (InputStream input = ConfigurationManager.class.getClassLoader().getResourceAsStream(CONFIG_FILE)) {
                    if (input != null) {
                        properties.load(input);
                        logger.info("Loaded configuration from classpath: {}", CONFIG_FILE);
                    } else {
                        logger.warn("Configuration file not found, using default values");
                        return getDefaultConfig();
                    }
                }
            }
            
            // Parse configuration
            config.setNumIoTDevices(getIntProperty(properties, "num.iot.devices", 20));
            config.setNumEdgeNodes(getIntProperty(properties, "num.edge.nodes", 5));
            config.setNumFogNodes(getIntProperty(properties, "num.fog.nodes", 2));
            config.setSimulationSteps(getIntProperty(properties, "simulation.steps", 100));
            
            // Security configuration
            String securityLevelStr = properties.getProperty("security.level", "MEDIUM");
            try {
                config.setSecurityLevel(SecurityLevel.valueOf(securityLevelStr.toUpperCase()));
            } catch (IllegalArgumentException e) {
                logger.warn("Invalid security level: {}, using MEDIUM", securityLevelStr);
                config.setSecurityLevel(SecurityLevel.MEDIUM);
            }
            
            config.setSecurityEnabledAtIoT(getBooleanProperty(properties, "security.enabled.iot", true));
            config.setSecurityEnabledAtEdge(getBooleanProperty(properties, "security.enabled.edge", true));
            config.setSecurityEnabledAtFog(getBooleanProperty(properties, "security.enabled.fog", true));
            
            // Attack configuration
            config.setAttackSimulationEnabled(getBooleanProperty(properties, "attack.simulation.enabled", true));
            
            String attackTypesStr = properties.getProperty("attack.types", "ALL");
            List<AttackType> attackTypes = parseAttackTypes(attackTypesStr);
            config.setAttackTypes(attackTypes);
            
        } catch (Exception e) {
            logger.error("Error loading configuration: {}", e.getMessage());
            logger.info("Using default configuration");
            return getDefaultConfig();
        }
        
        logger.info("Configuration loaded: {}", config);
        return config;
    }
    
    /**
     * Create a default configuration
     * @return The default configuration
     */
    private static SimulationConfig getDefaultConfig() {
        SimulationConfig config = new SimulationConfig();
        
        config.setNumIoTDevices(20);
        config.setNumEdgeNodes(5);
        config.setNumFogNodes(2);
        config.setSimulationSteps(100);
        config.setSecurityLevel(SecurityLevel.MEDIUM);
        config.setSecurityEnabledAtIoT(true);
        config.setSecurityEnabledAtEdge(true);
        config.setSecurityEnabledAtFog(true);
        config.setAttackSimulationEnabled(true);
        
        // Enable all attack types by default
        config.setAttackTypes(Arrays.asList(AttackType.values()));
        
        return config;
    }
    
    /**
     * Parse attack types from configuration string
     * @param attackTypesStr The attack types string
     * @return List of attack types
     */
    private static List<AttackType> parseAttackTypes(String attackTypesStr) {
        List<AttackType> attackTypes = new ArrayList<>();
        
        if (attackTypesStr == null || attackTypesStr.trim().isEmpty()) {
            return attackTypes;
        }
        
        if (attackTypesStr.equalsIgnoreCase("ALL")) {
            return Arrays.asList(AttackType.values());
        }
        
        String[] types = attackTypesStr.split(",");
        for (String type : types) {
            try {
                AttackType attackType = AttackType.valueOf(type.trim().toUpperCase());
                attackTypes.add(attackType);
            } catch (IllegalArgumentException e) {
                logger.warn("Invalid attack type: {}", type);
            }
        }
        
        return attackTypes;
    }
    
    /**
     * Get an integer property with default value
     * @param properties The properties object
     * @param key The property key
     * @param defaultValue The default value
     * @return The property value
     */
    private static int getIntProperty(Properties properties, String key, int defaultValue) {
        String value = properties.getProperty(key);
        if (value == null) {
            return defaultValue;
        }
        
        try {
            return Integer.parseInt(value);
        } catch (NumberFormatException e) {
            logger.warn("Invalid integer value for {}: {}, using default: {}", key, value, defaultValue);
            return defaultValue;
        }
    }
    
    /**
     * Get a boolean property with default value
     * @param properties The properties object
     * @param key The property key
     * @param defaultValue The default value
     * @return The property value
     */
    private static boolean getBooleanProperty(Properties properties, String key, boolean defaultValue) {
        String value = properties.getProperty(key);
        if (value == null) {
            return defaultValue;
        }
        
        return Boolean.parseBoolean(value);
    }
}
