package com.fog.eedto;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

/**
 * Simple test class to verify Log4j dependencies
 */
public class LogTest {
    private static final Logger logger = LogManager.getLogger(LogTest.class);
    
    public static void main(String[] args) {
        logger.info("Log4j test successful");
    }
}
