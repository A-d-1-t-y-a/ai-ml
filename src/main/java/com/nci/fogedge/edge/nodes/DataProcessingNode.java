package com.nci.fogedge.edge.nodes;

import com.nci.fogedge.edge.BaseEdgeNode;
import com.nci.fogedge.network.NetworkManager;
import com.nci.fogedge.utils.MetricsCollector;
import com.nci.fogedge.utils.DiagnosticResult;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.HashMap;
import java.util.Map;
import java.util.Random;

/**
 * Data Processing Edge Node implementation for the Fog and Edge Computing System
 * 
 * This class implements a data processing edge node that performs real-time
 * data filtering, aggregation, and preprocessing before sending to cloud or
 * other edge nodes. Based on the research paper's edge processing implementation.
 * 
 * @author National College of Ireland - Fog and Edge Computing Project
 * @version 1.0.0
 * @since 2024
 */
public class DataProcessingNode extends BaseEdgeNode {
    
    private static final Logger logger = LoggerFactory.getLogger(DataProcessingNode.class);
    
    // Data processing specific properties
    private int dataFilterCount;
    private int dataAggregationCount;
    private double dataReductionRate;
    private Random random;
    
    // Processing algorithms
    private double filterAccuracy;
    private double aggregationEfficiency;
    
    /**
     * Constructor for Data Processing Edge Node
     * 
     * @param nodeId Unique node identifier
     * @param networkManager Network manager for communication
     * @param metricsCollector Metrics collector for performance tracking
     */
    public DataProcessingNode(String nodeId, NetworkManager networkManager, MetricsCollector metricsCollector) {
        super(nodeId, "DATA_PROCESSING", networkManager, metricsCollector);
        
        this.random = new Random();
        this.dataFilterCount = 0;
        this.dataAggregationCount = 0;
        this.dataReductionRate = 0.75; // 75% data reduction
        this.filterAccuracy = 0.95; // 95% filter accuracy
        this.aggregationEfficiency = 0.90; // 90% aggregation efficiency
        
        logger.debug("Data processing edge node initialized: {}", nodeId);
    }
    
    @Override
    protected void initializeNode() {
        logger.debug("Initializing data processing edge node: {}", nodeId);
        
        // Set node-specific configuration
        configuration.put("processingType", "REAL_TIME");
        configuration.put("filterThreshold", 0.8);
        configuration.put("aggregationWindow", 60); // seconds
        configuration.put("dataCompression", 0.6);
        configuration.put("qualityOfService", "HIGH");
        
        logger.debug("Data processing edge node {} initialized successfully", nodeId);
    }
    
    @Override
    protected void cleanupNode() {
        logger.debug("Cleaning up data processing edge node: {}", nodeId);
        
        // Save processing statistics
        saveProcessingStats();
        
        logger.debug("Data processing edge node {} cleanup completed", nodeId);
    }
    
    @Override
    public String processData(String data) {
        try {
            logger.debug("Processing data in data processing node: {}", nodeId);
            
            // Simulate data processing pipeline
            String filteredData = applyDataFilter(data);
            String aggregatedData = applyDataAggregation(filteredData);
            String compressedData = applyDataCompression(aggregatedData);
            
            // Update processing statistics
            dataFilterCount++;
            dataAggregationCount++;
            
            // Create processing result
            String processingResult = "Processed data: " + compressedData;
            
            logger.debug("Data processed successfully in data processing node: {}", nodeId);
            return processingResult;
            
        } catch (Exception e) {
            logger.error("Error processing data in data processing node: {}", nodeId, e);
            return "Error processing data";
        }
    }

    public Object processData(Object data) {
        if (data instanceof String) {
            return processData((String) data);
        }
        return "";
    }
    
    private String applyDataFilter(String data) {
        try {
            // Simulate data filtering with realistic processing time
            Thread.sleep(random.nextInt(50) + 10); // 10-60ms processing time
            
            // Apply filter based on configuration
            double filterThreshold = (Double) configuration.get("filterThreshold");
            
            // Simulate filtering logic (keep data that meets threshold)
            if (Math.random() > filterThreshold) {
                return data + " [FILTERED]";
            }
            
            return data;
            
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            logger.error("Data filtering interrupted in node: {}", nodeId);
            return data;
        }
    }
    
    private String applyDataAggregation(String data) {
        try {
            // Simulate data aggregation with realistic processing time
            Thread.sleep(random.nextInt(100) + 20); // 20-120ms processing time
            
            // Apply aggregation based on configuration
            int aggregationWindow = (Integer) configuration.get("aggregationWindow");
            
            // Simulate aggregation logic
            String aggregatedData = data + " [AGGREGATED over " + aggregationWindow + "s]";
            
            return aggregatedData;
            
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            logger.error("Data aggregation interrupted in node: {}", nodeId);
            return data;
        }
    }
    
    private String applyDataCompression(String data) {
        try {
            // Simulate data compression with realistic processing time
            Thread.sleep(random.nextInt(30) + 5); // 5-35ms processing time
            
            // Apply compression based on configuration
            double compressionRatio = (Double) configuration.get("dataCompression");
            
            // Simulate compression logic
            String compressedData = data + " [COMPRESSED " + String.format("%.1f", compressionRatio * 100) + "%]";
            
            return compressedData;
            
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            logger.error("Data compression interrupted in node: {}", nodeId);
            return data;
        }
    }
    
    /**
     * Save processing statistics
     */
    private void saveProcessingStats() {
        // In a real implementation, this would save to persistent storage
        logger.debug("Processing statistics saved for data processing edge node: {}", nodeId);
    }
    
    /**
     * Get data filter count
     * 
     * @return Number of data filtering operations performed
     */
    public int getDataFilterCount() {
        return dataFilterCount;
    }
    
    /**
     * Get data aggregation count
     * 
     * @return Number of data aggregation operations performed
     */
    public int getDataAggregationCount() {
        return dataAggregationCount;
    }
    
    /**
     * Get data reduction rate
     * 
     * @return Data reduction rate as percentage
     */
    public double getDataReductionRate() {
        return dataReductionRate;
    }
    
    /**
     * Get filter accuracy
     * 
     * @return Filter accuracy as percentage
     */
    public double getFilterAccuracy() {
        return filterAccuracy;
    }
    
    /**
     * Get aggregation efficiency
     * 
     * @return Aggregation efficiency as percentage
     */
    public double getAggregationEfficiency() {
        return aggregationEfficiency;
    }
    
    @Override
    public Map<String, Object> getPerformanceMetrics() {
        Map<String, Object> metrics = super.getPerformanceMetrics();
        
        // Add data processing-specific metrics
        metrics.put("dataFilterCount", dataFilterCount);
        metrics.put("dataAggregationCount", dataAggregationCount);
        metrics.put("dataReductionRate", dataReductionRate);
        metrics.put("filterAccuracy", filterAccuracy);
        metrics.put("aggregationEfficiency", aggregationEfficiency);
        
        return metrics;
    }
    
    @Override
    public long getLastTaskOffloadingTime() {
        return lastTaskOffloadingTime;
    }

    @Override
    public boolean offloadTaskToCloud(String task) {
        try {
            logger.debug("Offloading task to cloud from data processing node: {}", nodeId);
            
            // Simulate task offloading to cloud
            boolean offloadingSuccess = networkManager.offloadTaskToCloud(nodeId, task);
            
            if (offloadingSuccess) {
                lastTaskOffloadingTime = System.currentTimeMillis();
                logger.debug("Task offloaded successfully from data processing node: {}", nodeId);
            } else {
                logger.warn("Task offloading failed from data processing node: {}", nodeId);
            }
            
            return offloadingSuccess;
            
        } catch (Exception e) {
            logger.error("Error offloading task from data processing node: {}", nodeId, e);
            return false;
        }
    }

    @Override
    public DiagnosticResult performDiagnostic() {
        DiagnosticResult baseResult = super.performDiagnostic();
        
        Map<String, Object> details = new HashMap<>(baseResult.getDetails());
        boolean passed = baseResult.isPassed();
        String message = baseResult.getMessage();
        
        // Add data processing-specific diagnostic checks
        if (filterAccuracy < 0.8) {
            passed = false;
            message = "Low filter accuracy";
        }
        details.put("filterAccuracy", filterAccuracy);
        details.put("minFilterAccuracy", 0.8);
        
        if (aggregationEfficiency < 0.7) {
            passed = false;
            message = "Low aggregation efficiency";
        }
        details.put("aggregationEfficiency", aggregationEfficiency);
        details.put("minAggregationEfficiency", 0.7);
        
        if (dataReductionRate < 0.5) {
            passed = false;
            message = "Low data reduction rate";
        }
        details.put("dataReductionRate", dataReductionRate);
        details.put("minDataReductionRate", 0.5);
        
        details.put("dataFilterCount", dataFilterCount);
        details.put("dataAggregationCount", dataAggregationCount);
        
        return new DiagnosticResult(passed, message, details);
    }

    @Override
    public boolean isRunning() {
        return this.isRunning;
    }

    @Override
    public String getLocation() {
        return "UNKNOWN";
    }
} 