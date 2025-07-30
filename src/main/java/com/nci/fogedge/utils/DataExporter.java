package com.nci.fogedge.utils;

import com.opencsv.CSVWriter;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.Instant;
import java.time.format.DateTimeFormatter;
import java.util.List;

/**
 * Data Exporter for Fog and Edge Computing System
 * 
 * This class exports system metrics to CSV files for analysis and visualization.
 * It supports exporting device metrics, node metrics, and system-wide statistics.
 * 
 * @author National College of Ireland - Fog and Edge Computing Project
 * @version 1.0.0
 * @since 2024
 */
public class DataExporter {
    
    private static final Logger logger = LoggerFactory.getLogger(DataExporter.class);
    
    private static final String DATA_DIR = "data";
    private static final String METRICS_DIR = "metrics";
    private static final String DEVICES_DIR = "devices";
    private static final String NODES_DIR = "nodes";
    private static final String SYSTEM_DIR = "system";
    
    private final DateTimeFormatter timestampFormatter;
    
    /**
     * Constructor for DataExporter
     */
    public DataExporter() {
        this.timestampFormatter = DateTimeFormatter.ofPattern("yyyy-MM-dd_HH-mm-ss");
        createDirectories();
        logger.info("DataExporter initialized");
    }
    
    /**
     * Export system metrics
     * 
     * @param metrics System metrics to export
     */
    public void exportMetrics(SystemMetrics metrics) {
        try {
            exportSystemMetrics(metrics);
            exportDeviceMetrics(metrics.getDeviceMetrics());
            exportNodeMetrics(metrics.getNodeMetrics());
            exportNetworkMetrics(metrics.getNetworkStatistics());
            
            logger.info("Metrics exported successfully");
            
        } catch (Exception e) {
            logger.error("Error exporting metrics", e);
        }
    }
    
    /**
     * Export system-wide metrics
     * 
     * @param metrics System metrics
     */
    private void exportSystemMetrics(SystemMetrics metrics) {
        String filename = String.format("system_metrics_%s.csv", 
            timestampFormatter.format(Instant.now()));
        Path filepath = Paths.get(DATA_DIR, SYSTEM_DIR, filename);
        
        try (CSVWriter writer = new CSVWriter(new FileWriter(filepath.toFile()))) {
            // Write header
            String[] header = {
                "Timestamp", "TotalDataProcessed", "TotalDevicesActive", "TotalNodesActive",
                "AverageLatency", "LatencyReduction", "DataReductionAtEdge", "EnergyEfficiency",
                "BandwidthOptimization", "SystemHealthScore", "SystemEfficiencyScore"
            };
            writer.writeNext(header);
            
            // Write data
            String[] data = {
                Instant.now().toString(),
                String.valueOf(metrics.getTotalDataProcessed()),
                String.valueOf(metrics.getTotalDevicesActive()),
                String.valueOf(metrics.getTotalNodesActive()),
                String.format("%.2f", metrics.getAverageLatency()),
                String.format("%.2f", metrics.getLatencyReduction()),
                String.format("%.2f", metrics.getDataReductionAtEdge()),
                String.format("%.2f", metrics.getEnergyEfficiency()),
                String.format("%.2f", metrics.getBandwidthOptimization()),
                String.format("%.2f", metrics.getSystemHealthScore()),
                String.format("%.2f", metrics.getSystemEfficiencyScore())
            };
            writer.writeNext(data);
            
            logger.debug("System metrics exported to {}", filepath);
            
        } catch (IOException e) {
            logger.error("Error exporting system metrics", e);
        }
    }
    
    /**
     * Export device metrics
     * 
     * @param deviceMetrics Device metrics map
     */
    private void exportDeviceMetrics(java.util.Map<String, DeviceMetrics> deviceMetrics) {
        if (deviceMetrics.isEmpty()) {
            return;
        }
        
        String filename = String.format("device_metrics_%s.csv", 
            timestampFormatter.format(Instant.now()));
        Path filepath = Paths.get(DATA_DIR, DEVICES_DIR, filename);
        
        try (CSVWriter writer = new CSVWriter(new FileWriter(filepath.toFile()))) {
            // Write header
            String[] header = {
                "Timestamp", "DeviceId", "AverageLatency", "AverageThroughput", 
                "AverageEnergyConsumption", "TotalDataProcessed", "HealthScore", 
                "PerformanceTrend", "UptimePercentage", "IsHealthy"
            };
            writer.writeNext(header);
            
            // Write data for each device
            for (DeviceMetrics metrics : deviceMetrics.values()) {
                String[] data = {
                    Instant.now().toString(),
                    metrics.getDeviceId(),
                    String.format("%.2f", metrics.getAverageLatency()),
                    String.format("%.2f", metrics.getAverageThroughput()),
                    String.format("%.2f", metrics.getAverageEnergyConsumption()),
                    String.valueOf(metrics.getTotalDataProcessed()),
                    String.format("%.2f", metrics.getHealthScore()),
                    metrics.getPerformanceTrend().toString(),
                    String.format("%.2f", metrics.getUptimePercentage()),
                    String.valueOf(metrics.isHealthy())
                };
                writer.writeNext(data);
            }
            
            logger.debug("Device metrics exported to {}", filepath);
            
        } catch (IOException e) {
            logger.error("Error exporting device metrics", e);
        }
    }
    
    /**
     * Export node metrics
     * 
     * @param nodeMetrics Node metrics map
     */
    private void exportNodeMetrics(java.util.Map<String, NodeMetrics> nodeMetrics) {
        if (nodeMetrics.isEmpty()) {
            return;
        }
        
        String filename = String.format("node_metrics_%s.csv", 
            timestampFormatter.format(Instant.now()));
        Path filepath = Paths.get(DATA_DIR, NODES_DIR, filename);
        
        try (CSVWriter writer = new CSVWriter(new FileWriter(filepath.toFile()))) {
            // Write header
            String[] header = {
                "Timestamp", "NodeId", "AverageLatency", "AverageThroughput", 
                "AverageEnergyConsumption", "TotalDataProcessed", "HealthScore", 
                "PerformanceTrend", "UptimePercentage", "ProcessingEfficiency",
                "LoadBalancingScore", "IsHealthy"
            };
            writer.writeNext(header);
            
            // Write data for each node
            for (NodeMetrics metrics : nodeMetrics.values()) {
                String[] data = {
                    Instant.now().toString(),
                    metrics.getNodeId(),
                    String.format("%.2f", metrics.getAverageLatency()),
                    String.format("%.2f", metrics.getAverageThroughput()),
                    String.format("%.2f", metrics.getAverageEnergyConsumption()),
                    String.valueOf(metrics.getTotalDataProcessed()),
                    String.format("%.2f", metrics.getHealthScore()),
                    metrics.getPerformanceTrend().toString(),
                    String.format("%.2f", metrics.getUptimePercentage()),
                    String.format("%.2f", metrics.getProcessingEfficiency()),
                    String.format("%.2f", metrics.getLoadBalancingScore()),
                    String.valueOf(metrics.isHealthy())
                };
                writer.writeNext(data);
            }
            
            logger.debug("Node metrics exported to {}", filepath);
            
        } catch (IOException e) {
            logger.error("Error exporting node metrics", e);
        }
    }
    
    /**
     * Export network metrics
     * 
     * @param networkStatistics Network statistics
     */
    private void exportNetworkMetrics(com.nci.fogedge.network.NetworkStatistics networkStatistics) {
        if (networkStatistics == null) {
            return;
        }
        
        String filename = String.format("network_metrics_%s.csv", 
            timestampFormatter.format(Instant.now()));
        Path filepath = Paths.get(DATA_DIR, METRICS_DIR, filename);
        
        try (CSVWriter writer = new CSVWriter(new FileWriter(filepath.toFile()))) {
            // Write header
            String[] header = {
                "Timestamp", "TotalPacketsTransmitted", "TotalPacketsReceived", 
                "TotalBytesTransmitted", "TotalBytesReceived", "AverageLatency", 
                "PacketLossRate", "ActiveNodeCount", "ActiveConnectionCount",
                "PacketSuccessRate", "DataTransferEfficiency", "ThroughputMbps",
                "NetworkHealthScore", "NetworkUtilization", "IsNetworkHealthy"
            };
            writer.writeNext(header);
            
            // Write data
            String[] data = {
                Instant.now().toString(),
                String.valueOf(networkStatistics.getTotalPacketsTransmitted()),
                String.valueOf(networkStatistics.getTotalPacketsReceived()),
                String.valueOf(networkStatistics.getTotalBytesTransmitted()),
                String.valueOf(networkStatistics.getTotalBytesReceived()),
                String.format("%.2f", networkStatistics.getAverageLatency()),
                String.format("%.4f", networkStatistics.getPacketLossRate()),
                String.valueOf(networkStatistics.getActiveNodeCount()),
                String.valueOf(networkStatistics.getActiveConnectionCount()),
                String.format("%.4f", networkStatistics.getPacketSuccessRate()),
                String.format("%.4f", networkStatistics.getDataTransferEfficiency()),
                String.format("%.2f", networkStatistics.getThroughputMbps()),
                String.format("%.2f", networkStatistics.getNetworkHealthScore()),
                String.format("%.2f", networkStatistics.getNetworkUtilization()),
                String.valueOf(networkStatistics.isNetworkHealthy())
            };
            writer.writeNext(data);
            
            logger.debug("Network metrics exported to {}", filepath);
            
        } catch (IOException e) {
            logger.error("Error exporting network metrics", e);
        }
    }
    
    /**
     * Export performance comparison data
     * 
     * @param metrics System metrics
     */
    public void exportPerformanceComparison(SystemMetrics metrics) {
        String filename = String.format("performance_comparison_%s.csv", 
            timestampFormatter.format(Instant.now()));
        Path filepath = Paths.get(DATA_DIR, METRICS_DIR, filename);
        
        try (CSVWriter writer = new CSVWriter(new FileWriter(filepath.toFile()))) {
            // Write header
            String[] header = {
                "Timestamp", "Metric", "FogEdgeValue", "CloudOnlyValue", "Improvement"
            };
            writer.writeNext(header);
            
            // Write comparison data
            String[][] comparisons = {
                {Instant.now().toString(), "Latency (ms)", 
                 String.format("%.2f", metrics.getAverageLatency()), "200.0", 
                 String.format("%.2f", metrics.getLatencyReduction()) + "%"},
                {Instant.now().toString(), "Data Reduction (%)", 
                 String.format("%.2f", metrics.getDataReductionAtEdge()), "0.0", 
                 String.format("%.2f", metrics.getDataReductionAtEdge()) + "%"},
                {Instant.now().toString(), "Energy Efficiency (%)", 
                 String.format("%.2f", metrics.getEnergyEfficiency()), "0.0", 
                 String.format("%.2f", metrics.getEnergyEfficiency()) + "%"},
                {Instant.now().toString(), "Bandwidth Optimization (%)", 
                 String.format("%.2f", metrics.getBandwidthOptimization()), "0.0", 
                 String.format("%.2f", metrics.getBandwidthOptimization()) + "%"}
            };
            
            for (String[] comparison : comparisons) {
                writer.writeNext(comparison);
            }
            
            logger.debug("Performance comparison exported to {}", filepath);
            
        } catch (IOException e) {
            logger.error("Error exporting performance comparison", e);
        }
    }
    
    /**
     * Create necessary directories
     */
    private void createDirectories() {
        try {
            Files.createDirectories(Paths.get(DATA_DIR));
            Files.createDirectories(Paths.get(DATA_DIR, METRICS_DIR));
            Files.createDirectories(Paths.get(DATA_DIR, DEVICES_DIR));
            Files.createDirectories(Paths.get(DATA_DIR, NODES_DIR));
            Files.createDirectories(Paths.get(DATA_DIR, SYSTEM_DIR));
            
            logger.debug("Data directories created successfully");
            
        } catch (IOException e) {
            logger.error("Error creating data directories", e);
        }
    }
    
    /**
     * Get data directory path
     * 
     * @return Data directory path
     */
    public String getDataDirectory() {
        return DATA_DIR;
    }
} 