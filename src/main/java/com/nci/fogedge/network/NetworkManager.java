package com.nci.fogedge.network;

import com.nci.fogedge.utils.ConfigurationManager;
import com.nci.fogedge.utils.MetricsCollector;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.Map;
import java.util.Queue;
import java.util.concurrent.ConcurrentLinkedQueue;

/**
 * Network Manager for Fog and Edge Computing System
 * 
 * This class manages network communication between IoT devices, edge nodes, and cloud services.
 * It simulates LoRaWAN and 5G connectivity as described in the IEEE INFOCOM 2022 research paper.
 * 
 * @author National College of Ireland - Fog and Edge Computing Project
 * @version 1.0.0
 * @since 2024
 */
public class NetworkManager {
    
    private static final Logger logger = LoggerFactory.getLogger(NetworkManager.class);
    
    // Network configuration
    private final ConfigurationManager configManager;
    private final MetricsCollector metricsCollector;
    
    // Network state
    private boolean isRunning;
    private final Map<String, NetworkNode> networkNodes;
    private final Queue<NetworkPacket> packetQueue;
    private final Map<String, NetworkConnection> connections;
    
    // Thread management
    private ScheduledExecutorService executorService;
    private Thread packetProcessorThread;
    
    // Network statistics
    private long totalPacketsTransmitted;
    private long totalPacketsReceived;
    private long totalBytesTransmitted;
    private long totalBytesReceived;
    private double averageLatency;
    private double packetLossRate;
    
    /**
     * Constructor for NetworkManager
     * 
     * @param configManager Configuration manager for network settings
     */
    public NetworkManager(ConfigurationManager configManager) {
        this.configManager = configManager;
        this.metricsCollector = new MetricsCollector();
        this.networkNodes = new ConcurrentHashMap<>();
        this.packetQueue = new ConcurrentLinkedQueue<>();
        this.connections = new ConcurrentHashMap<>();
        
        logger.info("NetworkManager initialized");
    }
    
    /**
     * Start the network manager
     */
    public void start() {
        if (isRunning) {
            logger.warn("NetworkManager is already running");
            return;
        }
        
        logger.info("Starting NetworkManager...");
        
        try {
            // Initialize thread pool
            executorService = Executors.newScheduledThreadPool(5);
            
            // Start packet processor
            startPacketProcessor();
            
            // Start network monitoring
            startNetworkMonitoring();
            
            // Start connection management
            startConnectionManagement();
            
            isRunning = true;
            logger.info("NetworkManager started successfully");
            
        } catch (Exception e) {
            logger.error("Failed to start NetworkManager", e);
            throw new RuntimeException("NetworkManager startup failed", e);
        }
    }
    
    /**
     * Stop the network manager
     */
    public void stop() {
        if (!isRunning) {
            logger.warn("NetworkManager is not running");
            return;
        }
        
        logger.info("Stopping NetworkManager...");
        
        try {
            isRunning = false;
            
            // Stop packet processor
            if (packetProcessorThread != null) {
                packetProcessorThread.interrupt();
                packetProcessorThread.join(5000);
            }
            
            // Stop thread pool
            if (executorService != null) {
                executorService.shutdown();
                if (!executorService.awaitTermination(10, TimeUnit.SECONDS)) {
                    executorService.shutdownNow();
                }
            }
            
            logger.info("NetworkManager stopped successfully");
            
        } catch (Exception e) {
            logger.error("Error during NetworkManager shutdown", e);
        }
    }
    
    /**
     * Transmit data from a device to the network
     * 
     * @param sourceId Source device/node ID
     * @param data Data to transmit
     */
    public void transmitData(String sourceId, byte[] data) {
        if (!isRunning) {
            logger.warn("NetworkManager is not running, cannot transmit data");
            return;
        }
        
        try {
            NetworkPacket packet = new NetworkPacket(sourceId, data);
            packetQueue.offer(packet);
            
            totalPacketsTransmitted++;
            totalBytesTransmitted += data.length;
            
            logger.debug("Data queued for transmission from {}: {} bytes", sourceId, data.length);
            
        } catch (Exception e) {
            logger.error("Error transmitting data from {}", sourceId, e);
        }
    }
    
    /**
     * Register a network node
     * 
     * @param nodeId Node identifier
     * @param nodeType Type of node (IoT, EDGE, CLOUD)
     * @param location Node location coordinates
     */
    public void registerNode(String nodeId, String nodeType, NetworkLocation location) {
        NetworkNode node = new NetworkNode(nodeId, nodeType, location);
        networkNodes.put(nodeId, node);
        
        logger.info("Registered network node: {} ({}) at {}", nodeId, nodeType, location);
    }
    
    /**
     * Unregister a network node
     * 
     * @param nodeId Node identifier
     */
    public void unregisterNode(String nodeId) {
        NetworkNode node = networkNodes.remove(nodeId);
        if (node != null) {
            logger.info("Unregistered network node: {}", nodeId);
        }
    }
    
    /**
     * Get network statistics
     * 
     * @return Network statistics
     */
    public NetworkStatistics getNetworkStatistics() {
        return new NetworkStatistics(
            totalPacketsTransmitted,
            totalPacketsReceived,
            totalBytesTransmitted,
            totalBytesReceived,
            averageLatency,
            packetLossRate,
            networkNodes.size(),
            connections.size()
        );
    }
    
    /**
     * Start packet processing thread
     */
    private void startPacketProcessor() {
        packetProcessorThread = new Thread(() -> {
            logger.info("Packet processor started");
            
            while (isRunning && !Thread.currentThread().isInterrupted()) {
                try {
                    NetworkPacket packet = packetQueue.poll();
                    if (packet != null) {
                        processPacket(packet);
                    } else {
                        Thread.sleep(10); // Small delay to prevent busy waiting
                    }
                } catch (InterruptedException e) {
                    logger.info("Packet processor interrupted");
                    Thread.currentThread().interrupt();
                    break;
                } catch (Exception e) {
                    logger.error("Error processing packet", e);
                }
            }
            
            logger.info("Packet processor stopped");
        });
        
        packetProcessorThread.start();
    }
    
    /**
     * Process a network packet
     * 
     * @param packet Packet to process
     */
    private void processPacket(NetworkPacket packet) {
        try {
            // Simulate network latency
            long latency = simulateNetworkLatency(packet.getSourceId());
            Thread.sleep(latency);
            
            // Simulate packet loss
            if (Math.random() < packetLossRate) {
                logger.debug("Packet lost from {}", packet.getSourceId());
                return;
            }
            
            // Route packet to destination
            routePacket(packet);
            
            totalPacketsReceived++;
            totalBytesReceived += packet.getData().length;
            
            // Update average latency
            updateAverageLatency(latency);
            
            logger.debug("Packet processed from {}: {} bytes, latency: {}ms", 
                packet.getSourceId(), packet.getData().length, latency);
            
        } catch (Exception e) {
            logger.error("Error processing packet from {}", packet.getSourceId(), e);
        }
    }
    
    /**
     * Simulate network latency based on source type
     * 
     * @param sourceId Source device/node ID
     * @return Simulated latency in milliseconds
     */
    private long simulateNetworkLatency(String sourceId) {
        NetworkNode node = networkNodes.get(sourceId);
        if (node == null) {
            return 50; // Default latency
        }
        
        switch (node.getNodeType()) {
            case "IoT":
                return 20 + (long)(Math.random() * 30); // LoRaWAN: 20-50ms
            case "EDGE":
                return 5 + (long)(Math.random() * 15);  // 5G Edge: 5-20ms
            case "CLOUD":
                return 100 + (long)(Math.random() * 200); // Cloud: 100-300ms
            default:
                return 50;
        }
    }
    
    /**
     * Route packet to appropriate destination
     * 
     * @param packet Packet to route
     */
    private void routePacket(NetworkPacket packet) {
        // Simple routing logic - in a real system, this would be more sophisticated
        String sourceId = packet.getSourceId();
        NetworkNode sourceNode = networkNodes.get(sourceId);
        
        if (sourceNode == null) {
            logger.warn("Unknown source node: {}", sourceId);
            return;
        }
        
        // Route based on source type
        switch (sourceNode.getNodeType()) {
            case "IoT":
                routeToEdge(packet);
                break;
            case "EDGE":
                routeToCloud(packet);
                break;
            case "CLOUD":
                routeToEdge(packet);
                break;
            default:
                logger.warn("Unknown node type: {}", sourceNode.getNodeType());
        }
    }
    
    /**
     * Route packet to edge nodes
     * 
     * @param packet Packet to route
     */
    private void routeToEdge(NetworkPacket packet) {
        // Find nearest edge node
        NetworkNode nearestEdge = findNearestEdgeNode(packet.getSourceId());
        if (nearestEdge != null) {
            logger.debug("Routing packet from {} to edge node {}", 
                packet.getSourceId(), nearestEdge.getNodeId());
        }
    }
    
    /**
     * Route packet to cloud
     * 
     * @param packet Packet to route
     */
    private void routeToCloud(NetworkPacket packet) {
        // Find cloud service
        NetworkNode cloudNode = findCloudNode();
        if (cloudNode != null) {
            logger.debug("Routing packet from {} to cloud node {}", 
                packet.getSourceId(), cloudNode.getNodeId());
        }
    }
    
    /**
     * Find nearest edge node
     * 
     * @param sourceId Source node ID
     * @return Nearest edge node
     */
    private NetworkNode findNearestEdgeNode(String sourceId) {
        return networkNodes.values().stream()
            .filter(node -> "EDGE".equals(node.getNodeType()))
            .findFirst()
            .orElse(null);
    }
    
    /**
     * Find cloud node
     * 
     * @return Cloud node
     */
    private NetworkNode findCloudNode() {
        return networkNodes.values().stream()
            .filter(node -> "CLOUD".equals(node.getNodeType()))
            .findFirst()
            .orElse(null);
    }
    
    /**
     * Update average latency
     * 
     * @param latency New latency measurement
     */
    private void updateAverageLatency(long latency) {
        if (totalPacketsReceived > 0) {
            averageLatency = ((averageLatency * (totalPacketsReceived - 1)) + latency) / totalPacketsReceived;
        }
    }
    
    /**
     * Start network monitoring
     */
    private void startNetworkMonitoring() {
        executorService.scheduleAtFixedRate(() -> {
            try {
                monitorNetworkHealth();
            } catch (Exception e) {
                logger.error("Error during network monitoring", e);
            }
        }, 30, 30, TimeUnit.SECONDS);
    }
    
    /**
     * Start connection management
     */
    private void startConnectionManagement() {
        executorService.scheduleAtFixedRate(() -> {
            try {
                manageConnections();
            } catch (Exception e) {
                logger.error("Error during connection management", e);
            }
        }, 60, 60, TimeUnit.SECONDS);
    }
    
    /**
     * Monitor network health
     */
    private void monitorNetworkHealth() {
        NetworkStatistics stats = getNetworkStatistics();
        
        // Update packet loss rate
        if (totalPacketsTransmitted > 0) {
            packetLossRate = (double)(totalPacketsTransmitted - totalPacketsReceived) / totalPacketsTransmitted;
        }
        
        // Log network health
        logger.info("Network Health - Packets: {}/{}, Bytes: {}/{}, Latency: {:.2f}ms, Loss: {:.2f}%",
            totalPacketsReceived, totalPacketsTransmitted,
            totalBytesReceived, totalBytesTransmitted,
            averageLatency, packetLossRate * 100);
        
        // Update metrics
        metricsCollector.updateNetworkMetrics(stats);
    }
    
    /**
     * Manage network connections
     */
    private void manageConnections() {
        // Clean up stale connections
        connections.entrySet().removeIf(entry -> {
            NetworkConnection conn = entry.getValue();
            if (conn.isStale(300)) { // 5 minutes timeout
                logger.debug("Removing stale connection: {}", entry.getKey());
                return true;
            }
            return false;
        });
        
        logger.debug("Active connections: {}", connections.size());
    }
    
    /**
     * Get active node count
     * 
     * @return Number of active nodes
     */
    public int getActiveNodeCount() {
        return networkNodes.size();
    }
    
    /**
     * Get active connection count
     * 
     * @return Number of active connections
     */
    public int getActiveConnectionCount() {
        return connections.size();
    }
} 