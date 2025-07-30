package com.nci.fogedge.network;

import java.time.Instant;

/**
 * Network Connection for Fog and Edge Computing System
 * 
 * This class represents a network connection between two nodes.
 * It tracks connection state, quality, and performance metrics.
 * 
 * @author National College of Ireland - Fog and Edge Computing Project
 * @version 1.0.0
 * @since 2024
 */
public class NetworkConnection {
    
    private final String connectionId;
    private final String sourceNodeId;
    private final String targetNodeId;
    private final Instant creationTime;
    private Instant lastActivity;
    private ConnectionState state;
    private double signalStrength;
    private double bandwidth;
    private long totalBytesTransferred;
    private int packetLossCount;
    private int totalPackets;
    
    /**
     * Connection states
     */
    public enum ConnectionState {
        ESTABLISHING,
        ESTABLISHED,
        DEGRADED,
        DISCONNECTED
    }
    
    /**
     * Constructor for NetworkConnection
     * 
     * @param sourceNodeId Source node ID
     * @param targetNodeId Target node ID
     */
    public NetworkConnection(String sourceNodeId, String targetNodeId) {
        this.connectionId = generateConnectionId(sourceNodeId, targetNodeId);
        this.sourceNodeId = sourceNodeId;
        this.targetNodeId = targetNodeId;
        this.creationTime = Instant.now();
        this.lastActivity = Instant.now();
        this.state = ConnectionState.ESTABLISHING;
        this.signalStrength = 100.0;
        this.bandwidth = 100.0;
        this.totalBytesTransferred = 0;
        this.packetLossCount = 0;
        this.totalPackets = 0;
    }
    
    /**
     * Get connection ID
     * 
     * @return Connection identifier
     */
    public String getConnectionId() {
        return connectionId;
    }
    
    /**
     * Get source node ID
     * 
     * @return Source node ID
     */
    public String getSourceNodeId() {
        return sourceNodeId;
    }
    
    /**
     * Get target node ID
     * 
     * @return Target node ID
     */
    public String getTargetNodeId() {
        return targetNodeId;
    }
    
    /**
     * Get connection creation time
     * 
     * @return Creation timestamp
     */
    public Instant getCreationTime() {
        return creationTime;
    }
    
    /**
     * Get last activity time
     * 
     * @return Last activity timestamp
     */
    public Instant getLastActivity() {
        return lastActivity;
    }
    
    /**
     * Get connection state
     * 
     * @return Current connection state
     */
    public ConnectionState getState() {
        return state;
    }
    
    /**
     * Get signal strength
     * 
     * @return Signal strength (0-100)
     */
    public double getSignalStrength() {
        return signalStrength;
    }
    
    /**
     * Get bandwidth
     * 
     * @return Bandwidth in Mbps
     */
    public double getBandwidth() {
        return bandwidth;
    }
    
    /**
     * Get total bytes transferred
     * 
     * @return Total bytes transferred
     */
    public long getTotalBytesTransferred() {
        return totalBytesTransferred;
    }
    
    /**
     * Get packet loss rate
     * 
     * @return Packet loss rate (0-1)
     */
    public double getPacketLossRate() {
        return totalPackets > 0 ? (double) packetLossCount / totalPackets : 0.0;
    }
    
    /**
     * Update connection activity
     */
    public void updateActivity() {
        this.lastActivity = Instant.now();
    }
    
    /**
     * Set connection state
     * 
     * @param state New connection state
     */
    public void setState(ConnectionState state) {
        this.state = state;
        updateActivity();
    }
    
    /**
     * Update signal strength
     * 
     * @param signalStrength New signal strength (0-100)
     */
    public void updateSignalStrength(double signalStrength) {
        this.signalStrength = Math.max(0.0, Math.min(100.0, signalStrength));
        updateActivity();
    }
    
    /**
     * Update bandwidth
     * 
     * @param bandwidth New bandwidth in Mbps
     */
    public void updateBandwidth(double bandwidth) {
        this.bandwidth = Math.max(0.0, bandwidth);
        updateActivity();
    }
    
    /**
     * Record data transfer
     * 
     * @param bytes Number of bytes transferred
     * @param packetLost Whether the packet was lost
     */
    public void recordTransfer(long bytes, boolean packetLost) {
        this.totalBytesTransferred += bytes;
        this.totalPackets++;
        
        if (packetLost) {
            this.packetLossCount++;
        }
        
        updateActivity();
    }
    
    /**
     * Check if connection is stale
     * 
     * @param timeoutSeconds Timeout in seconds
     * @return True if connection is stale
     */
    public boolean isStale(long timeoutSeconds) {
        return Instant.now().isAfter(lastActivity.plusSeconds(timeoutSeconds));
    }
    
    /**
     * Check if connection is healthy
     * 
     * @return True if connection is healthy
     */
    public boolean isHealthy() {
        return state == ConnectionState.ESTABLISHED &&
               signalStrength > 50.0 &&
               getPacketLossRate() < 0.1; // Less than 10% packet loss
    }
    
    /**
     * Get connection quality score
     * 
     * @return Quality score (0-100)
     */
    public double getQualityScore() {
        double signalScore = signalStrength;
        double packetLossScore = (1.0 - getPacketLossRate()) * 100.0;
        double stateScore = (state == ConnectionState.ESTABLISHED) ? 100.0 : 50.0;
        
        return (signalScore + packetLossScore + stateScore) / 3.0;
    }
    
    /**
     * Generate connection ID
     * 
     * @param sourceNodeId Source node ID
     * @param targetNodeId Target node ID
     * @return Connection ID
     */
    private static String generateConnectionId(String sourceNodeId, String targetNodeId) {
        return sourceNodeId + "_TO_" + targetNodeId + "_" + System.currentTimeMillis();
    }
    
    @Override
    public String toString() {
        return String.format("NetworkConnection{id=%s, %s->%s, state=%s, quality=%.1f}",
            connectionId, sourceNodeId, targetNodeId, state, getQualityScore());
    }
    
    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (obj == null || getClass() != obj.getClass()) return false;
        NetworkConnection that = (NetworkConnection) obj;
        return connectionId.equals(that.connectionId);
    }
    
    @Override
    public int hashCode() {
        return connectionId.hashCode();
    }
} 