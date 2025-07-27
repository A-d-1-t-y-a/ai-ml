package org.nci.fogedge.util;

import org.apache.logging.log4j.Level;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.apache.logging.log4j.core.config.Configurator;
import org.apache.logging.log4j.core.config.DefaultConfiguration;

/**
 * Utility class for configuring logging
 * Based on the paper: "An Overview of Fog Computing and Edge Computing Security and Privacy Issues"
 * (Sensors 2021, 21, 8226, https://doi.org/10.3390/s21248226)
 */
public class LoggingUtil {
    private static final Logger logger = LogManager.getLogger(LoggingUtil.class);
    
    /**
     * Configure logging for the application
     */
    public static void configureLogging() {
        // Configure Log4j
        Configurator.initialize(new DefaultConfiguration());
        Configurator.setRootLevel(Level.INFO);
        
        logger.info("Logging configured successfully");
    }
    
    /**
     * Set the logging level
     * @param level The logging level to set
     */
    public static void setLoggingLevel(Level level) {
        Configurator.setRootLevel(level);
        logger.info("Logging level set to: {}", level);
    }
}
