package com.nci.fogedge.network;

/**
 * Network Statistics for Fog and Edge Computing System
 * 
 * This class holds network performance metrics and statistics
 * for monitoring and analysis purposes.
 * 
 * @author National College of Ireland - Fog and Edge Computing Project
 * @version 1.0.0
 * @since 2024
 */
public class NetworkStatistics {
    
    private final long totalPacketsTransmitted;
    private final long totalPacketsReceived;
    private final long totalBytesTransmitted;
    private final long totalBytesReceived;
    private final double averageLatency;
    private final double packetLossRate;
    private final int activeNodeCount;
    private final int activeConnectionCount;
    
    /**
     * Constructor for NetworkStatistics
     * 
     * @param totalPacketsTransmitted Total packets transmitted
     * @param totalPacketsReceived Total packets received
     * @param totalBytesTransmitted Total bytes transmitted
     * @param totalBytesReceived Total bytes received
     * @param averageLatency Average latency in milliseconds
     * @param packetLossRate Packet loss rate (0-1)
     * @param activeNodeCount Number of active nodes
     * @param activeConnectionCount Number of active connections
     */
    public NetworkStatistics(long totalPacketsTransmitted, long totalPacketsReceived,
                           long totalBytesTransmitted, long totalBytesReceived,
                           double averageLatency, double packetLossRate,
                           int activeNodeCount, int activeConnectionCount) {
        this.totalPacketsTransmitted = totalPacketsTransmitted;
        this.totalPacketsReceived = totalPacketsReceived;
        this.totalBytesTransmitted = totalBytesTransmitted;
        this.totalBytesReceived = totalBytesReceived;
        this.averageLatency = averageLatency;
        this.packetLossRate = packetLossRate;
        this.activeNodeCount = activeNodeCount;
        this.activeConnectionCount = activeConnectionCount;
    }
    
    /**
     * Get total packets transmitted
     * 
     * @return Total packets transmitted
     */
    public long getTotalPacketsTransmitted() {
        return totalPacketsTransmitted;
    }
    
    /**
     * Get total packets received
     * 
     * @return Total packets received
     */
    public long getTotalPacketsReceived() {
        return totalPacketsReceived;
    }
    
    /**
     * Get total bytes transmitted
     * 
     * @return Total bytes transmitted
     */
    public long getTotalBytesTransmitted() {
        return totalBytesTransmitted;
    }
    
    /**
     * Get total bytes received
     * 
     * @return Total bytes received
     */
    public long getTotalBytesReceived() {
        return totalBytesReceived;
    }
    
    /**
     * Get average latency
     * 
     * @return Average latency in milliseconds
     */
    public double getAverageLatency() {
        return averageLatency;
    }
    
    /**
     * Get packet loss rate
     * 
     * @return Packet loss rate (0-1)
     */
    public double getPacketLossRate() {
        return packetLossRate;
    }
    
    /**
     * Get active node count
     * 
     * @return Number of active nodes
     */
    public int getActiveNodeCount() {
        return activeNodeCount;
    }
    
    /**
     * Get active connection count
     * 
     * @return Number of active connections
     */
    public int getActiveConnectionCount() {
        return activeConnectionCount;
    }
    
    /**
     * Get packet success rate
     * 
     * @return Packet success rate (0-1)
     */
    public double getPacketSuccessRate() {
        return totalPacketsTransmitted > 0 ? 
            (double) totalPacketsReceived / totalPacketsTransmitted : 1.0;
    }
    
    /**
     * Get data transfer efficiency
     * 
     * @return Data transfer efficiency (0-1)
     */
    public double getDataTransferEfficiency() {
        return totalBytesTransmitted > 0 ? 
            (double) totalBytesReceived / totalBytesTransmitted : 1.0;
    }
    
    /**
     * Get network throughput in Mbps
     * 
     * @return Network throughput in Mbps
     */
    public double getThroughputMbps() {
        // Assuming average packet size of 1500 bytes
        double avgPacketSize = totalPacketsTransmitted > 0 ? 
            (double) totalBytesTransmitted / totalPacketsTransmitted : 1500.0;
        
        return (totalPacketsReceived * avgPacketSize * 8.0) / (1024.0 * 1024.0); // Convert to Mbps
    }
    
    /**
     * Get network health score
     * 
     * @return Network health score (0-100)
     */
    public double getNetworkHealthScore() {
        double packetSuccessScore = getPacketSuccessRate() * 100.0;
        double latencyScore = Math.max(0.0, 100.0 - averageLatency / 10.0); // Normalize latency
        double connectionScore = activeConnectionCount > 0 ? 100.0 : 50.0;
        
        return (packetSuccessScore + latencyScore + connectionScore) / 3.0;
    }
    
    /**
     * Check if network is healthy
     * 
     * @return True if network is healthy
     */
    public boolean isNetworkHealthy() {
        return getNetworkHealthScore() > 70.0 &&
               packetLossRate < 0.05 && // Less than 5% packet loss
               averageLatency < 100.0; // Less than 100ms average latency
    }
    
    /**
     * Get network utilization percentage
     * 
     * @return Network utilization (0-100)
     */
    public double getNetworkUtilization() {
        // Simplified calculation based on active connections and throughput
        double maxConnections = 100.0; // Assumed maximum
        double maxThroughput = 1000.0; // 1 Gbps assumed maximum
        
        double connectionUtilization = (activeConnectionCount / maxConnections) * 100.0;
        double throughputUtilization = (getThroughputMbps() / maxThroughput) * 100.0;
        
        return Math.min(100.0, (connectionUtilization + throughputUtilization) / 2.0);
    }
    
    @Override
    public String toString() {
        return String.format("NetworkStatistics{packets=%d/%d, bytes=%d/%d, latency=%.2fms, loss=%.2f%%, health=%.1f}",
            totalPacketsReceived, totalPacketsTransmitted,
            totalBytesReceived, totalBytesTransmitted,
            averageLatency, packetLossRate * 100, getNetworkHealthScore());
    }
} 