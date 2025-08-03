package com.fog.eedto.util;

import java.io.IOException;
import java.io.InputStream;
import java.util.Properties;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.awt.Color;

/**
 * Configuration manager for the EEDTO simulation.
 * Loads and provides access to configuration parameters from properties file.
 */
public class ConfigurationManager {
    private static final Logger logger = Logger.getLogger(ConfigurationManager.class.getName());
    private static final String CONFIG_FILE = "config.properties";
    private static Properties properties = new Properties();
    private static boolean initialized = false;
    
    /**
     * Initialize the configuration manager by loading properties from the config file.
     * This method should be called before any other method in this class.
     * 
     * @return true if initialization was successful, false otherwise
     */
    public static boolean initialize() {
        if (initialized) {
            return true;
        }
        
        try (InputStream input = ConfigurationManager.class.getClassLoader().getResourceAsStream(CONFIG_FILE)) {
            if (input == null) {
                logger.log(Level.SEVERE, "Unable to find " + CONFIG_FILE);
                return false;
            }
            
            properties.load(input);
            initialized = true;
            logger.info("Configuration loaded successfully from " + CONFIG_FILE);
            return true;
        } catch (IOException e) {
            logger.log(Level.SEVERE, "Error loading configuration: " + e.getMessage(), e);
            return false;
        }
    }
    
    /**
     * Get a string property value.
     * 
     * @param key Property key
     * @param defaultValue Default value if property is not found
     * @return Property value or default value
     */
    public static String getString(String key, String defaultValue) {
        ensureInitialized();
        return properties.getProperty(key, defaultValue);
    }
    
    /**
     * Get an integer property value.
     * 
     * @param key Property key
     * @param defaultValue Default value if property is not found or invalid
     * @return Property value or default value
     */
    public static int getInt(String key, int defaultValue) {
        ensureInitialized();
        String value = properties.getProperty(key);
        if (value == null) {
            return defaultValue;
        }
        
        try {
            return Integer.parseInt(value);
        } catch (NumberFormatException e) {
            logger.log(Level.WARNING, "Invalid integer value for key " + key + ": " + value);
            return defaultValue;
        }
    }
    
    /**
     * Get a double property value.
     * 
     * @param key Property key
     * @param defaultValue Default value if property is not found or invalid
     * @return Property value or default value
     */
    public static double getDouble(String key, double defaultValue) {
        ensureInitialized();
        String value = properties.getProperty(key);
        if (value == null) {
            return defaultValue;
        }
        
        try {
            return Double.parseDouble(value);
        } catch (NumberFormatException e) {
            logger.log(Level.WARNING, "Invalid double value for key " + key + ": " + value);
            return defaultValue;
        }
    }
    
    /**
     * Get a long property value.
     * 
     * @param key Property key
     * @param defaultValue Default value if property is not found or invalid
     * @return Property value or default value
     */
    public static long getLong(String key, long defaultValue) {
        ensureInitialized();
        String value = properties.getProperty(key);
        if (value == null) {
            return defaultValue;
        }
        
        try {
            return Long.parseLong(value);
        } catch (NumberFormatException e) {
            logger.log(Level.WARNING, "Invalid long value for key " + key + ": " + value);
            return defaultValue;
        }
    }
    
    /**
     * Get a boolean property value.
     * 
     * @param key Property key
     * @param defaultValue Default value if property is not found or invalid
     * @return Property value or default value
     */
    public static boolean getBoolean(String key, boolean defaultValue) {
        ensureInitialized();
        String value = properties.getProperty(key);
        if (value == null) {
            return defaultValue;
        }
        
        return Boolean.parseBoolean(value);
    }
    
    /**
     * Get a color property value.
     * 
     * @param key Property key
     * @param defaultValue Default value if property is not found or invalid
     * @return Color value or default value
     */
    public static Color getColor(String key, Color defaultValue) {
        ensureInitialized();
        String value = properties.getProperty(key);
        if (value == null) {
            return defaultValue;
        }
        
        try {
            String[] rgb = value.split(",");
            if (rgb.length != 3) {
                throw new NumberFormatException("Color must have 3 components");
            }
            
            int r = Integer.parseInt(rgb[0].trim());
            int g = Integer.parseInt(rgb[1].trim());
            int b = Integer.parseInt(rgb[2].trim());
            
            return new Color(r, g, b);
        } catch (Exception e) {
            logger.log(Level.WARNING, "Invalid color value for key " + key + ": " + value);
            return defaultValue;
        }
    }
    
    /**
     * Ensure that the configuration manager is initialized.
     * If not, attempt to initialize it.
     */
    private static void ensureInitialized() {
        if (!initialized) {
            initialize();
        }
    }
}
