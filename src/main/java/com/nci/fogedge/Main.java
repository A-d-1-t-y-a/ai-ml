package com.nci.fogedge;

import com.nci.fogedge.iot.IoTDeviceManager;
import com.nci.fogedge.edge.EdgeNodeManager;
import com.nci.fogedge.cloud.CloudManager;
import com.nci.fogedge.network.NetworkManager;
import com.nci.fogedge.utils.MetricsCollector;
import com.nci.fogedge.utils.DataExporter;
import com.nci.fogedge.utils.ConfigurationManager;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

/**
 * Main application class for the Fog and Edge Computing System
 * 
 * This class orchestrates the entire three-tier architecture (IoT-Edge-Cloud)
 * implementing the research paper: "Edge-Fog-Cloud Architecture for Real-Time 
 * IoT Data Processing: A Hierarchical Approach to Service Distribution"
 * 
 * @author National College of Ireland - Fog and Edge Computing Project
 * @version 1.0.0
 * @since 2024
 */
public class Main {
    
    private static final Logger logger = LoggerFactory.getLogger(Main.class);
    
    // System components
    private IoTDeviceManager iotManager;
    private EdgeNodeManager edgeManager;
    private CloudManager cloudManager;
    private NetworkManager networkManager;
    private MetricsCollector metricsCollector;
    private DataExporter dataExporter;
    private ConfigurationManager configManager;
    
    // Thread management
    private ScheduledExecutorService executorService;
    
    /**
     * Main entry point for the Fog and Edge Computing System
     * 
     * @param args Command line arguments
     */
    public static void main(String[] args) {
        logger.info("=== FOG AND EDGE COMPUTING SYSTEM STARTUP ===");
        logger.info("Based on IEEE INFOCOM 2022 Research Paper");
        logger.info("System Version: 1.0.0");
        
        Main application = new Main();
        application.initialize();
        application.start();
        
        // Add shutdown hook for graceful termination
        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            logger.info("Shutdown signal received. Terminating system gracefully...");
            application.shutdown();
        }));
    }
    
    /**
     * Initialize all system components
     */
    private void initialize() {
        logger.info("Initializing system components...");
        
        try {
            // Load configuration
            configManager = new ConfigurationManager();
            configManager.loadConfiguration();
            logger.info("Configuration loaded successfully");
            
            // Initialize metrics collector
            metricsCollector = new MetricsCollector();
            logger.info("Metrics collector initialized");
            
            // Initialize data exporter
            dataExporter = new DataExporter();
            logger.info("Data exporter initialized");
            
            // Initialize network manager
            networkManager = new NetworkManager(configManager);
            logger.info("Network manager initialized");
            
            // Initialize IoT device manager
            iotManager = new IoTDeviceManager(configManager, networkManager, metricsCollector);
            logger.info("IoT device manager initialized");
            
            // Initialize edge node manager
            edgeManager = new EdgeNodeManager(configManager, networkManager, metricsCollector);
            logger.info("Edge node manager initialized");
            
            // Initialize cloud manager
            cloudManager = new CloudManager(configManager, networkManager, metricsCollector);
            logger.info("Cloud manager initialized");
            
            // Initialize thread pool
            executorService = Executors.newScheduledThreadPool(10);
            logger.info("Thread pool initialized with 10 threads");
            
            logger.info("All system components initialized successfully");
            
        } catch (Exception e) {
            logger.error("Failed to initialize system components", e);
            throw new RuntimeException("System initialization failed", e);
        }
    }
    
    /**
     * Start the system and begin data processing
     */
    private void start() {
        logger.info("Starting Fog and Edge Computing System...");
        
        try {
            // Start IoT devices
            iotManager.start();
            logger.info("IoT devices started successfully");
            
            // Start edge nodes
            edgeManager.start();
            logger.info("Edge nodes started successfully");
            
            // Start cloud services
            cloudManager.start();
            logger.info("Cloud services started successfully");
            
            // Start network monitoring
            networkManager.start();
            logger.info("Network monitoring started");
            
            // Schedule periodic metrics collection
            executorService.scheduleAtFixedRate(() -> {
                try {
                    metricsCollector.collectMetrics();
                    logger.debug("Periodic metrics collection completed");
                } catch (Exception e) {
                    logger.error("Error during metrics collection", e);
                }
            }, 5, 30, TimeUnit.SECONDS);
            
            // Schedule periodic data export
            executorService.scheduleAtFixedRate(() -> {
                try {
                    dataExporter.exportMetrics(metricsCollector.getMetrics());
                    logger.debug("Periodic data export completed");
                } catch (Exception e) {
                    logger.error("Error during data export", e);
                }
            }, 10, 60, TimeUnit.SECONDS);
            
            // Schedule periodic performance analysis
            executorService.scheduleAtFixedRate(() -> {
                try {
                    analyzePerformance();
                    logger.debug("Periodic performance analysis completed");
                } catch (Exception e) {
                    logger.error("Error during performance analysis", e);
                }
            }, 15, 120, TimeUnit.SECONDS);
            
            logger.info("=== SYSTEM STARTED SUCCESSFULLY ===");
            logger.info("System is now processing IoT data with Fog and Edge Computing architecture");
            logger.info("Performance metrics will be collected every 30 seconds");
            logger.info("Data will be exported every 60 seconds");
            logger.info("Performance analysis will be conducted every 2 minutes");
            
        } catch (Exception e) {
            logger.error("Failed to start system", e);
            throw new RuntimeException("System startup failed", e);
        }
    }
    
    /**
     * Perform periodic performance analysis
     */
    private void analyzePerformance() {
        logger.info("=== PERFORMANCE ANALYSIS ===");
        
        // Analyze latency metrics
        double avgLatency = metricsCollector.getAverageLatency();
        double latencyReduction = metricsCollector.getLatencyReduction();
        logger.info("Average Latency: {} ms", String.format("%.2f", avgLatency));
        logger.info("Latency Reduction: {}%", String.format("%.2f", latencyReduction));
        
        // Analyze data processing metrics
        double dataReduction = metricsCollector.getDataReductionAtEdge();
        double energyEfficiency = metricsCollector.getEnergyEfficiency();
        logger.info("Data Reduction at Edge: {}%", String.format("%.2f", dataReduction));
        logger.info("Energy Efficiency: {}%", String.format("%.2f", energyEfficiency));
        
        // Analyze bandwidth usage
        double bandwidthOptimization = metricsCollector.getBandwidthOptimization();
        logger.info("Bandwidth Optimization: {}%", String.format("%.2f", bandwidthOptimization));
        
        // Log system health
        logger.info("Active IoT Devices: {}", iotManager.getActiveDeviceCount());
        logger.info("Active Edge Nodes: {}", edgeManager.getActiveNodeCount());
        logger.info("Cloud Services Status: {}", cloudManager.getServiceStatus());
        
        logger.info("=== END PERFORMANCE ANALYSIS ===");
    }
    
    /**
     * Gracefully shutdown the system
     */
    private void shutdown() {
        logger.info("Initiating system shutdown...");
        
        try {
            // Shutdown thread pool
            if (executorService != null) {
                executorService.shutdown();
                if (!executorService.awaitTermination(30, TimeUnit.SECONDS)) {
                    executorService.shutdownNow();
                }
                logger.info("Thread pool shutdown completed");
            }
            
            // Stop IoT devices
            if (iotManager != null) {
                iotManager.stop();
                logger.info("IoT devices stopped");
            }
            
            // Stop edge nodes
            if (edgeManager != null) {
                edgeManager.stop();
                logger.info("Edge nodes stopped");
            }
            
            // Stop cloud services
            if (cloudManager != null) {
                cloudManager.stop();
                logger.info("Cloud services stopped");
            }
            
            // Stop network monitoring
            if (networkManager != null) {
                networkManager.stop();
                logger.info("Network monitoring stopped");
            }
            
            // Export final metrics
            if (dataExporter != null && metricsCollector != null) {
                dataExporter.exportMetrics(metricsCollector.getMetrics());
                logger.info("Final metrics exported");
            }
            
            logger.info("=== SYSTEM SHUTDOWN COMPLETED ===");
            
        } catch (Exception e) {
            logger.error("Error during system shutdown", e);
        }
    }
} 