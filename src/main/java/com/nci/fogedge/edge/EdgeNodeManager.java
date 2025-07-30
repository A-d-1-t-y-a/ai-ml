package com.nci.fogedge.edge;

import com.nci.fogedge.network.NetworkManager;
import com.nci.fogedge.utils.MetricsCollector;
import com.nci.fogedge.utils.ConfigurationManager;
import com.nci.fogedge.edge.nodes.*;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Edge Node Manager for the Fog and Edge Computing System
 * 
 * This class manages multiple edge computing nodes that process IoT data
 * locally and perform intelligent task offloading to the cloud layer.
 * Based on the research paper's edge computing implementation.
 * 
 * @author National College of Ireland - Fog and Edge Computing Project
 * @version 1.0.0
 * @since 2024
 */
public class EdgeNodeManager {
    
    private static final Logger logger = LoggerFactory.getLogger(EdgeNodeManager.class);
    
    // Configuration and dependencies
    private final ConfigurationManager configManager;
    private final NetworkManager networkManager;
    private final MetricsCollector metricsCollector;
    
    // Edge node management
    private final Map<String, EdgeNode> edgeNodes;
    private final List<EdgeNode> activeNodes;
    private final AtomicInteger nodeCounter;
    
    // Thread management
    private final ScheduledExecutorService edgeExecutor;
    private final List<Future<?>> edgeTasks;
    
    // Performance tracking
    private final AtomicInteger totalDataProcessed;
    private final AtomicInteger tasksOffloaded;
    private final AtomicInteger localProcessingTime;
    
    /**
     * Constructor for Edge Node Manager
     * 
     * @param configManager Configuration manager for system settings
     * @param networkManager Network manager for communication
     * @param metricsCollector Metrics collector for performance tracking
     */
    public EdgeNodeManager(ConfigurationManager configManager, 
                          NetworkManager networkManager, 
                          MetricsCollector metricsCollector) {
        this.configManager = configManager;
        this.networkManager = networkManager;
        this.metricsCollector = metricsCollector;
        
        this.edgeNodes = new ConcurrentHashMap<>();
        this.activeNodes = Collections.synchronizedList(new ArrayList<>());
        this.nodeCounter = new AtomicInteger(0);
        
        this.edgeExecutor = Executors.newScheduledThreadPool(15);
        this.edgeTasks = Collections.synchronizedList(new ArrayList<>());
        
        this.totalDataProcessed = new AtomicInteger(0);
        this.tasksOffloaded = new AtomicInteger(0);
        this.localProcessingTime = new AtomicInteger(0);
        
        logger.info("Edge Node Manager initialized");
    }
    
    /**
     * Start the Edge Node Manager and initialize edge nodes
     */
    public void start() {
        logger.info("Starting Edge Node Manager...");
        
        try {
            // Create and initialize edge nodes
            createEdgeNodes();
            
            // Start all edge nodes
            startAllEdgeNodes();
            
            // Start edge monitoring
            startEdgeMonitoring();
            
            logger.info("Edge Node Manager started successfully with {} nodes", activeNodes.size());
            
        } catch (Exception e) {
            logger.error("Failed to start Edge Node Manager", e);
            throw new RuntimeException("Edge Node Manager startup failed", e);
        }
    }
    
    /**
     * Create various types of edge nodes
     */
    private void createEdgeNodes() {
        logger.info("Creating edge nodes...");
        
        // Data processing edge nodes
        for (int i = 0; i < 5; i++) {
            String nodeId = "EDGE_DATA_" + String.format("%03d", i);
            DataProcessingNode node = new DataProcessingNode(nodeId, networkManager, metricsCollector);
            edgeNodes.put(nodeId, node);
            activeNodes.add(node);
            logger.debug("Created data processing edge node: {}", nodeId);
        }
        
        // Analytics edge nodes
        for (int i = 0; i < 3; i++) {
            String nodeId = "EDGE_ANALYTICS_" + String.format("%03d", i);
            AnalyticsNode node = new AnalyticsNode(nodeId, networkManager, metricsCollector);
            edgeNodes.put(nodeId, node);
            activeNodes.add(node);
            logger.debug("Created analytics edge node: {}", nodeId);
        }
        
        // Gateway edge nodes
        for (int i = 0; i < 2; i++) {
            String nodeId = "EDGE_GATEWAY_" + String.format("%03d", i);
            GatewayNode node = new GatewayNode(nodeId, networkManager, metricsCollector);
            edgeNodes.put(nodeId, node);
            activeNodes.add(node);
            logger.debug("Created gateway edge node: {}", nodeId);
        }
        
        logger.info("Created {} edge nodes successfully", activeNodes.size());
    }
    
    /**
     * Start all edge nodes
     */
    private void startAllEdgeNodes() {
        logger.info("Starting all edge nodes...");
        
        for (EdgeNode node : activeNodes) {
            try {
                node.start();
                logger.debug("Started edge node: {}", node.getNodeId());
            } catch (Exception e) {
                logger.error("Failed to start edge node: {}", node.getNodeId(), e);
            }
        }
        
        logger.info("All edge nodes started");
    }
    
    /**
     * Start edge monitoring and data collection
     */
    private void startEdgeMonitoring() {
        logger.info("Starting edge monitoring...");
        
        // Monitor edge node health
        Future<?> healthMonitor = edgeExecutor.scheduleAtFixedRate(() -> {
            try {
                monitorEdgeHealth();
            } catch (Exception e) {
                logger.error("Error in edge health monitoring", e);
            }
        }, 10, 60, TimeUnit.SECONDS);
        edgeTasks.add(healthMonitor);
        
        // Monitor processing performance
        Future<?> performanceMonitor = edgeExecutor.scheduleAtFixedRate(() -> {
            try {
                monitorProcessingPerformance();
            } catch (Exception e) {
                logger.error("Error in processing performance monitoring", e);
            }
        }, 15, 45, TimeUnit.SECONDS);
        edgeTasks.add(performanceMonitor);
        
        // Monitor task offloading
        Future<?> offloadingMonitor = edgeExecutor.scheduleAtFixedRate(() -> {
            try {
                monitorTaskOffloading();
            } catch (Exception e) {
                logger.error("Error in task offloading monitoring", e);
            }
        }, 20, 90, TimeUnit.SECONDS);
        edgeTasks.add(offloadingMonitor);
        
        logger.info("Edge monitoring started");
    }
    
    /**
     * Monitor the health of all edge nodes
     */
    private void monitorEdgeHealth() {
        logger.debug("Monitoring edge node health...");
        
        int healthyNodes = 0;
        int totalNodes = activeNodes.size();
        
        for (EdgeNode node : activeNodes) {
            if (node.isHealthy()) {
                healthyNodes++;
            } else {
                logger.warn("Edge node {} is unhealthy", node.getNodeId());
            }
        }
        
        double healthPercentage = (double) healthyNodes / totalNodes * 100;
        logger.info("Edge Node Health Status: {}/{} nodes healthy ({:.2f}%)", 
                   healthyNodes, totalNodes, healthPercentage);
        
        // Update metrics
        metricsCollector.updateEdgeHealth(healthPercentage);
    }
    
    /**
     * Monitor processing performance
     */
    private void monitorProcessingPerformance() {
        logger.debug("Monitoring processing performance...");
        
        int totalProcessed = totalDataProcessed.get();
        int totalTime = localProcessingTime.get();
        double avgProcessingTime = totalProcessed > 0 ? (double) totalTime / totalProcessed : 0;
        
        logger.info("Processing Performance Stats:");
        logger.info("  Total Data Processed: {} bytes", totalProcessed);
        logger.info("  Average Processing Time: {:.2f} ms", avgProcessingTime);
        logger.info("  Active Edge Nodes: {}", activeNodes.size());
        
        // Update metrics
        metricsCollector.updateProcessingStats(totalProcessed, avgProcessingTime);
    }
    
    /**
     * Monitor task offloading statistics
     */
    private void monitorTaskOffloading() {
        logger.debug("Monitoring task offloading...");
        
        int totalOffloaded = tasksOffloaded.get();
        int totalTasks = totalDataProcessed.get();
        double offloadingRate = totalTasks > 0 ? (double) totalOffloaded / totalTasks * 100 : 0;
        
        logger.info("Task Offloading Stats:");
        logger.info("  Total Tasks: {}", totalTasks);
        logger.info("  Tasks Offloaded: {}", totalOffloaded);
        logger.info("  Offloading Rate: {:.2f}%", offloadingRate);
        
        // Update metrics
        metricsCollector.updateOffloadingStats(totalTasks, totalOffloaded, offloadingRate);
    }
    
    /**
     * Get the count of active edge nodes
     * 
     * @return Number of active edge nodes
     */
    public int getActiveNodeCount() {
        return activeNodes.size();
    }
    
    /**
     * Get a specific edge node by ID
     * 
     * @param nodeId Edge node identifier
     * @return Edge node or null if not found
     */
    public EdgeNode getEdgeNode(String nodeId) {
        return edgeNodes.get(nodeId);
    }
    
    /**
     * Get all active edge nodes
     * 
     * @return List of active edge nodes
     */
    public List<EdgeNode> getAllEdgeNodes() {
        return new ArrayList<>(activeNodes);
    }
    
    /**
     * Record data processing
     * 
     * @param dataSize Size of processed data in bytes
     * @param processingTime Processing time in milliseconds
     */
    public void recordDataProcessing(int dataSize, int processingTime) {
        totalDataProcessed.addAndGet(dataSize);
        localProcessingTime.addAndGet(processingTime);
    }
    
    /**
     * Record task offloading
     */
    public void recordTaskOffloading() {
        tasksOffloaded.incrementAndGet();
    }
    
    /**
     * Stop the Edge Node Manager
     */
    public void stop() {
        logger.info("Stopping Edge Node Manager...");
        
        try {
            // Stop all edge nodes
            for (EdgeNode node : activeNodes) {
                try {
                    node.stop();
                    logger.debug("Stopped edge node: {}", node.getNodeId());
                } catch (Exception e) {
                    logger.error("Error stopping edge node: {}", node.getNodeId(), e);
                }
            }
            
            // Cancel all monitoring tasks
            for (Future<?> task : edgeTasks) {
                if (!task.isCancelled()) {
                    task.cancel(true);
                }
            }
            
            // Shutdown executor
            edgeExecutor.shutdown();
            if (!edgeExecutor.awaitTermination(30, TimeUnit.SECONDS)) {
                edgeExecutor.shutdownNow();
            }
            
            logger.info("Edge Node Manager stopped successfully");
            
        } catch (Exception e) {
            logger.error("Error stopping Edge Node Manager", e);
        }
    }
    
    /**
     * Get performance statistics
     * 
     * @return Map containing performance statistics
     */
    public Map<String, Object> getPerformanceStats() {
        Map<String, Object> stats = new HashMap<>();
        stats.put("totalNodes", activeNodes.size());
        stats.put("totalDataProcessed", totalDataProcessed.get());
        stats.put("tasksOffloaded", tasksOffloaded.get());
        stats.put("localProcessingTime", localProcessingTime.get());
        stats.put("offloadingRate", totalDataProcessed.get() > 0 ? 
                  (double) tasksOffloaded.get() / totalDataProcessed.get() * 100 : 0);
        
        return stats;
    }
} 