package com.nci.fogedge.utils;

import java.util.Map;
import java.util.HashMap;

/**
 * Diagnostic Result for Fog and Edge Computing System
 * 
 * This class represents the result of a diagnostic operation performed on
 * system components such as IoT devices, edge nodes, and cloud services.
 * 
 * @author National College of Ireland - Fog and Edge Computing Project
 * @version 1.0.0
 * @since 2024
 */
public class DiagnosticResult {
    
    private final boolean passed;
    private final String message;
    private final Map<String, Object> details;
    
    /**
     * Constructor for DiagnosticResult
     * 
     * @param passed Whether the diagnostic passed
     * @param message Diagnostic message
     * @param details Diagnostic details
     */
    public DiagnosticResult(boolean passed, String message, Map<String, Object> details) {
        this.passed = passed;
        this.message = message;
        this.details = details != null ? new HashMap<>(details) : new HashMap<>();
    }
    
    /**
     * Constructor for DiagnosticResult with default details
     * 
     * @param passed Whether the diagnostic passed
     * @param message Diagnostic message
     */
    public DiagnosticResult(boolean passed, String message) {
        this(passed, message, new HashMap<>());
    }
    
    /**
     * Check if diagnostic passed
     * 
     * @return True if diagnostic passed
     */
    public boolean isPassed() {
        return passed;
    }
    
    /**
     * Get diagnostic message
     * 
     * @return Diagnostic message
     */
    public String getMessage() {
        return message;
    }
    
    /**
     * Get diagnostic details
     * 
     * @return Diagnostic details
     */
    public Map<String, Object> getDetails() {
        return new HashMap<>(details);
    }
    
    /**
     * Add diagnostic detail
     * 
     * @param key Detail key
     * @param value Detail value
     */
    public void addDetail(String key, Object value) {
        details.put(key, value);
    }
    
    /**
     * Get a specific diagnostic detail
     * 
     * @param key Detail key
     * @return Detail value
     */
    public Object getDetail(String key) {
        return details.get(key);
    }
    
    /**
     * Check if diagnostic has a specific detail
     * 
     * @param key Detail key
     * @return True if detail exists
     */
    public boolean hasDetail(String key) {
        return details.containsKey(key);
    }
    
    /**
     * Get diagnostic severity level
     * 
     * @return Severity level (INFO, WARNING, ERROR, CRITICAL)
     */
    public String getSeverity() {
        if (!passed) {
            if (details.containsKey("error_count") && (Integer) details.get("error_count") > 10) {
                return "CRITICAL";
            } else if (details.containsKey("error_count") && (Integer) details.get("error_count") > 5) {
                return "ERROR";
            } else {
                return "WARNING";
            }
        }
        return "INFO";
    }
    
    /**
     * Get diagnostic score
     * 
     * @return Diagnostic score (0-100)
     */
    public double getScore() {
        if (passed) {
            return 100.0;
        }
        
        // Calculate score based on details
        double score = 100.0;
        
        if (details.containsKey("error_count")) {
            int errorCount = (Integer) details.get("error_count");
            score -= errorCount * 5.0; // Deduct 5 points per error
        }
        
        if (details.containsKey("latency")) {
            double latency = (Double) details.get("latency");
            if (latency > 200.0) {
                score -= 20.0; // Deduct 20 points for high latency
            } else if (latency > 100.0) {
                score -= 10.0; // Deduct 10 points for moderate latency
            }
        }
        
        if (details.containsKey("cpu_usage")) {
            double cpuUsage = (Double) details.get("cpu_usage");
            if (cpuUsage > 90.0) {
                score -= 15.0; // Deduct 15 points for high CPU usage
            } else if (cpuUsage > 80.0) {
                score -= 10.0; // Deduct 10 points for moderate CPU usage
            }
        }
        
        if (details.containsKey("memory_usage")) {
            double memoryUsage = (Double) details.get("memory_usage");
            if (memoryUsage > 90.0) {
                score -= 15.0; // Deduct 15 points for high memory usage
            } else if (memoryUsage > 80.0) {
                score -= 10.0; // Deduct 10 points for moderate memory usage
            }
        }
        
        return Math.max(0.0, score);
    }
    
    /**
     * Get diagnostic summary
     * 
     * @return Diagnostic summary
     */
    public String getSummary() {
        StringBuilder summary = new StringBuilder();
        summary.append("Diagnostic ").append(passed ? "PASSED" : "FAILED");
        summary.append(" - ").append(message);
        summary.append(" (Score: ").append(String.format("%.1f", getScore())).append("/100)");
        summary.append(" - Severity: ").append(getSeverity());
        
        return summary.toString();
    }
    
    @Override
    public String toString() {
        return String.format("DiagnosticResult{passed=%s, message='%s', details=%s, score=%.1f}",
            passed, message, details, getScore());
    }
    
    /**
     * Create a successful diagnostic result
     * 
     * @param message Success message
     * @return Diagnostic result
     */
    public static DiagnosticResult success(String message) {
        return new DiagnosticResult(true, message);
    }
    
    /**
     * Create a successful diagnostic result with details
     * 
     * @param message Success message
     * @param details Success details
     * @return Diagnostic result
     */
    public static DiagnosticResult success(String message, Map<String, Object> details) {
        return new DiagnosticResult(true, message, details);
    }
    
    /**
     * Create a failed diagnostic result
     * 
     * @param message Failure message
     * @return Diagnostic result
     */
    public static DiagnosticResult failure(String message) {
        return new DiagnosticResult(false, message);
    }
    
    /**
     * Create a failed diagnostic result with details
     * 
     * @param message Failure message
     * @param details Failure details
     * @return Diagnostic result
     */
    public static DiagnosticResult failure(String message, Map<String, Object> details) {
        return new DiagnosticResult(false, message, details);
    }
} 