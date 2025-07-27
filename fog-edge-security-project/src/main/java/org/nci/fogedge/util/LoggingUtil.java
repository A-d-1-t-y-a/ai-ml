package org.nci.fogedge.util;

import org.apache.log4j.ConsoleAppender;
import org.apache.log4j.FileAppender;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.log4j.PatternLayout;
import java.util.logging.Logger;

import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.Date;

/**
 * Utility class for logging in the fog computing simulation.
 * Provides methods for configuring and using Log4j for logging.
 */
public class LoggingUtil {
    
    private static final Logger logger = Logger.getLogger(LoggingUtil.class);
    private static final String LOG_PATTERN = "%d{yyyy-MM-dd HH:mm:ss} %-5p %c{1}:%L - %m%n";
    private static boolean initialized = false;
    
    /**
     * Initializes the logging system
     * 
     * @param logToFile Whether to log to a file in addition to console
     * @param logLevel The logging level to use
     */
    public static void initializeLogging(boolean logToFile, Level logLevel) {
        if (initialized) {
            return;
        }
        
        try {
            // Configure Log4j
            PatternLayout layout = new PatternLayout(LOG_PATTERN);
            
            // Console appender
            ConsoleAppender consoleAppender = new ConsoleAppender();
            consoleAppender.setLayout(layout);
            consoleAppender.setThreshold(logLevel);
            consoleAppender.activateOptions();
            Logger.getRootLogger().addAppender(consoleAppender);
            
            // File appender (optional)
            if (logToFile) {
                String timestamp = new SimpleDateFormat("yyyyMMdd_HHmmss").format(new Date());
                String logFileName = "simulation_" + timestamp + ".log";
                
                FileAppender fileAppender = new FileAppender();
                fileAppender.setFile(logFileName);
                fileAppender.setLayout(layout);
                fileAppender.setThreshold(logLevel);
                fileAppender.setAppend(true);
                fileAppender.activateOptions();
                Logger.getRootLogger().addAppender(fileAppender);
                
                Log.printLine("Logging to file: " + logFileName);
            }
            
            // Set CloudSim logging level
            Log.setDisabled(!logLevel.isGreaterOrEqual(Level.INFO));
            
            initialized = true;
            Log.printLine("Logging initialized with level: " + logLevel.toString());
        } catch (Exception e) {
            System.err.println("Failed to initialize logging: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    /**
     * Logs a message at INFO level
     * 
     * @param message The message to log
     */
    public static void info(String message) {
        logger.info(message);
    }
    
    /**
     * Logs a message at DEBUG level
     * 
     * @param message The message to log
     */
    public static void debug(String message) {
        logger.debug(message);
    }
    
    /**
     * Logs a message at WARN level
     * 
     * @param message The message to log
     */
    public static void warn(String message) {
        logger.warn(message);
    }
    
    /**
     * Logs a message at ERROR level
     * 
     * @param message The message to log
     */
    public static void error(String message) {
        logger.error(message);
    }
    
    /**
     * Logs a message at ERROR level with exception details
     * 
     * @param message The message to log
     * @param e The exception to log
     */
    public static void error(String message, Throwable e) {
        logger.error(message, e);
    }
    
    /**
     * Creates a log file for simulation results
     * 
     * @param simulationName Name of the simulation
     * @return Path to the created log file
     */
    public static String createSimulationResultsLog(String simulationName) {
        String timestamp = new SimpleDateFormat("yyyyMMdd_HHmmss").format(new Date());
        String resultsFileName = "results_" + simulationName + "_" + timestamp + ".csv";
        
        try {
            // Create CSV header
            StringBuilder header = new StringBuilder();
            header.append("Timestamp,");
            header.append("SecurityEnabled,");
            header.append("TotalIoTDevices,");
            header.append("TotalEdgeNodes,");
            header.append("TotalFogNodes,");
            header.append("TotalDataGenerated(KB),");
            header.append("TotalDataProcessed(KB),");
            header.append("TotalEnergyConsumption(J),");
            header.append("TotalProcessingTime(ms),");
            header.append("TotalSecurityOverhead(ms),");
            header.append("DetectedAttacks\n");
            
            // Write header to file
            java.nio.file.Files.write(
                java.nio.file.Paths.get(resultsFileName), 
                header.toString().getBytes()
            );
            
            Log.printLine("Created simulation results log: " + resultsFileName);
            return resultsFileName;
        } catch (IOException e) {
            Log.printLine("Failed to create simulation results log: " + e.getMessage());
            e.printStackTrace();
            return null;
        }
    }
    
    /**
     * Appends a simulation result entry to the results log
     * 
     * @param resultsFileName The results file to append to
     * @param securityEnabled Whether security was enabled
     * @param totalIoTDevices Total number of IoT devices
     * @param totalEdgeNodes Total number of edge nodes
     * @param totalFogNodes Total number of fog nodes
     * @param totalDataGenerated Total data generated in KB
     * @param totalDataProcessed Total data processed in KB
     * @param totalEnergyConsumption Total energy consumption in Joules
     * @param totalProcessingTime Total processing time in ms
     * @param totalSecurityOverhead Total security overhead in ms
     * @param detectedAttacks Number of detected attacks
     */
    public static void appendSimulationResult(
            String resultsFileName,
            boolean securityEnabled,
            int totalIoTDevices,
            int totalEdgeNodes,
            int totalFogNodes,
            double totalDataGenerated,
            double totalDataProcessed,
            double totalEnergyConsumption,
            double totalProcessingTime,
            double totalSecurityOverhead,
            int detectedAttacks) {
        
        try {
            // Create CSV line
            StringBuilder line = new StringBuilder();
            line.append(new SimpleDateFormat("yyyy-MM-dd HH:mm:ss").format(new Date())).append(",");
            line.append(securityEnabled).append(",");
            line.append(totalIoTDevices).append(",");
            line.append(totalEdgeNodes).append(",");
            line.append(totalFogNodes).append(",");
            line.append(String.format("%.2f", totalDataGenerated)).append(",");
            line.append(String.format("%.2f", totalDataProcessed)).append(",");
            line.append(String.format("%.2f", totalEnergyConsumption)).append(",");
            line.append(String.format("%.2f", totalProcessingTime)).append(",");
            line.append(String.format("%.2f", totalSecurityOverhead)).append(",");
            line.append(detectedAttacks).append("\n");
            
            // Append to file
            java.nio.file.Files.write(
                java.nio.file.Paths.get(resultsFileName), 
                line.toString().getBytes(),
                java.nio.file.StandardOpenOption.APPEND
            );
        } catch (IOException e) {
            Log.printLine("Failed to append simulation result: " + e.getMessage());
            e.printStackTrace();
        }
    }
}
