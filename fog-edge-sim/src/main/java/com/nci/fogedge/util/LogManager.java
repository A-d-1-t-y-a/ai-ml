package com.nci.fogedge.util;

import com.nci.fogedge.model.SimulationConfig;
import com.nci.fogedge.model.SimulationResults;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;

/**
 * Manages logging and reporting for the simulation.
 * This class handles console output, file logging, and report generation.
 */
public class LogManager {
    private SimulationConfig config;
    private List<LogEntry> logEntries;
    private boolean consoleLoggingEnabled;
    private boolean fileLoggingEnabled;
    private String logFilePath;
    private LogLevel minLogLevel;
    
    /**
     * Constructor for LogManager
     * 
     * @param config Simulation configuration
     */
    public LogManager(SimulationConfig config) {
        this.config = config;
        this.logEntries = new ArrayList<>();
        this.consoleLoggingEnabled = config.isConsoleLoggingEnabled();
        this.fileLoggingEnabled = config.isFileLoggingEnabled();
        this.logFilePath = config.getLogFilePath();
        this.minLogLevel = config.getMinLogLevel();
        
        // Create log directory if it doesn't exist
        if (fileLoggingEnabled) {
            File logDir = new File(logFilePath).getParentFile();
            if (logDir != null && !logDir.exists()) {
                logDir.mkdirs();
            }
        }
        
        // Log initial message
        logInfo("LogManager initialized. Console logging: " + consoleLoggingEnabled + 
                ", File logging: " + fileLoggingEnabled + ", Min log level: " + minLogLevel);
    }
    
    /**
     * Logs an info message
     * 
     * @param message Message to log
     */
    public void logInfo(String message) {
        log(LogLevel.INFO, message);
    }
    
    /**
     * Logs a warning message
     * 
     * @param message Message to log
     */
    public void logWarning(String message) {
        log(LogLevel.WARNING, message);
    }
    
    /**
     * Logs an error message
     * 
     * @param message Message to log
     */
    public void logError(String message) {
        log(LogLevel.ERROR, message);
    }
    
    /**
     * Logs a debug message
     * 
     * @param message Message to log
     */
    public void logDebug(String message) {
        log(LogLevel.DEBUG, message);
    }
    
    /**
     * Logs a message with the specified level
     * 
     * @param level Log level
     * @param message Message to log
     */
    public void log(LogLevel level, String message) {
        // Skip if log level is below minimum
        if (level.ordinal() < minLogLevel.ordinal()) {
            return;
        }
        
        // Create log entry
        LogEntry entry = new LogEntry(level, message);
        
        // Add to log entries
        logEntries.add(entry);
        
        // Log to console
        if (consoleLoggingEnabled) {
            logToConsole(entry);
        }
        
        // Log to file
        if (fileLoggingEnabled) {
            logToFile(entry);
        }
    }
    
    /**
     * Logs an entry to the console
     * 
     * @param entry Log entry
     */
    private void logToConsole(LogEntry entry) {
        String logMessage = formatLogEntry(entry);
        
        switch (entry.getLevel()) {
            case ERROR:
                System.err.println(logMessage);
                break;
            default:
                System.out.println(logMessage);
                break;
        }
    }
    
    /**
     * Logs an entry to a file
     * 
     * @param entry Log entry
     */
    private void logToFile(LogEntry entry) {
        String logMessage = formatLogEntry(entry);
        
        try (FileWriter fw = new FileWriter(logFilePath, true);
             PrintWriter pw = new PrintWriter(fw)) {
            pw.println(logMessage);
        } catch (IOException e) {
            System.err.println("Error writing to log file: " + e.getMessage());
            // Disable file logging to prevent further errors
            fileLoggingEnabled = false;
        }
    }
    
    /**
     * Formats a log entry
     * 
     * @param entry Log entry
     * @return Formatted log message
     */
    private String formatLogEntry(LogEntry entry) {
        return String.format("[%s] [%s] %s",
                entry.getTimestamp(),
                entry.getLevel(),
                entry.getMessage());
    }
    
    /**
     * Generates a simulation report
     * 
     * @param results Simulation results
     * @return Report as a string
     */
    public String generateReport(SimulationResults results) {
        StringBuilder report = new StringBuilder();
        
        // Add header
        report.append("==========================================================\n");
        report.append("                 SIMULATION REPORT                        \n");
        report.append("==========================================================\n\n");
        
        // Add timestamp
        
        // Add simulation results using the toString method
        report.append(results.toString());
        
        // Add footer
        report.append("\n==========================================================\n");
        report.append("                   END OF REPORT                          \n");
        report.append("==========================================================\n");
        
        return report.toString();
    }
    
    /**
     * Saves a report to a file
     * 
     * @param report Report to save
     * @param filePath File path
     * @return True if successful, false otherwise
     */
    public boolean saveReportToFile(String report, String filePath) {
        try (FileWriter fw = new FileWriter(filePath);
             PrintWriter pw = new PrintWriter(fw)) {
            pw.println(report);
            return true;
        } catch (IOException e) {
            logError("Error saving report to file: " + e.getMessage());
            return false;
        }
    }
    
    /**
     * Gets all log entries
     * 
     * @return List of log entries
     */
    public List<LogEntry> getLogEntries() {
        return new ArrayList<>(logEntries);
    }
    
    /**
     * Gets log entries filtered by level
     * 
     * @param level Log level
     * @return List of log entries
     */
    public List<LogEntry> getLogEntries(LogLevel level) {
        List<LogEntry> filtered = new ArrayList<>();
        
        for (LogEntry entry : logEntries) {
            if (entry.getLevel() == level) {
                filtered.add(entry);
            }
        }
        
        return filtered;
    }
    
    /**
     * Clears all log entries
     */
    public void clearLogEntries() {
        logEntries.clear();
    }
    
    /**
     * Log entry class
     */
    public static class LogEntry {
        private Date timestamp;
        private LogLevel level;
        private String message;
        
        /**
         * Constructor for LogEntry
         * 
         * @param level Log level
         * @param message Log message
         */
        public LogEntry(LogLevel level, String message) {
            this.timestamp = new Date();
            this.level = level;
            this.message = message;
        }
        
        /**
         * Gets the timestamp
         * 
         * @return Timestamp
         */
        public String getTimestamp() {
            SimpleDateFormat sdf = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss.SSS");
            return sdf.format(timestamp);
        }
        
        /**
         * Gets the log level
         * 
         * @return Log level
         */
        public LogLevel getLevel() {
            return level;
        }
        
        /**
         * Gets the log message
         * 
         * @return Log message
         */
        public String getMessage() {
            return message;
        }
    }
    
    /**
     * Log level enum
     */
    public enum LogLevel {
        DEBUG,
        INFO,
        WARNING,
        ERROR
    }
}
