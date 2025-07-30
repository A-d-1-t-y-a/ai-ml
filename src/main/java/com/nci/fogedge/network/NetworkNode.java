package com.nci.fogedge.network;

import java.time.Instant;

/**
 * Network Node for Fog and Edge Computing System
 * 
 * This class represents a network node in the three-tier architecture.
 * Nodes can be IoT devices, edge nodes, or cloud services.
 * 
 * @author National College of Ireland - Fog and Edge Computing Project
 * @version 1.0.0
 * @since 2024
 */
public class NetworkNode {
    
    private final String nodeId;
    private final String nodeType;
    private final NetworkLocation location;
    private final Instant registrationTime;
    private Instant lastActivity;
    private boolean isActive;
    private double healthScore;
    
    /**
     * Constructor for NetworkNode
     * 
     * @param nodeId Node identifier
     * @param nodeType Type of node (IoT, EDGE, CLOUD)
     * @param location Node location coordinates
     */
    public NetworkNode(String nodeId, String nodeType, NetworkLocation location) {
        this.nodeId = nodeId;
        this.nodeType = nodeType;
        this.location = location;
        this.registrationTime = Instant.now();
        this.lastActivity = Instant.now();
        this.isActive = true;
        this.healthScore = 100.0;
    }
    
    /**
     * Get node identifier
     * 
     * @return Node ID
     */
    public String getNodeId() {
        return nodeId;
    }
    
    /**
     * Get node type
     * 
     * @return Node type (IoT, EDGE, CLOUD)
     */
    public String getNodeType() {
        return nodeType;
    }
    
    /**
     * Get node location
     * 
     * @return Node location coordinates
     */
    public NetworkLocation getLocation() {
        return location;
    }
    
    /**
     * Get registration time
     * 
     * @return Time when node was registered
     */
    public Instant getRegistrationTime() {
        return registrationTime;
    }
    
    /**
     * Get last activity time
     * 
     * @return Time of last activity
     */
    public Instant getLastActivity() {
        return lastActivity;
    }
    
    /**
     * Check if node is active
     * 
     * @return True if node is active
     */
    public boolean isActive() {
        return isActive;
    }
    
    /**
     * Get node health score
     * 
     * @return Health score (0-100)
     */
    public double getHealthScore() {
        return healthScore;
    }
    
    /**
     * Update last activity time
     */
    public void updateActivity() {
        this.lastActivity = Instant.now();
    }
    
    /**
     * Set node active status
     * 
     * @param active Active status
     */
    public void setActive(boolean active) {
        this.isActive = active;
    }
    
    /**
     * Update health score
     * 
     * @param healthScore New health score (0-100)
     */
    public void updateHealthScore(double healthScore) {
        this.healthScore = Math.max(0.0, Math.min(100.0, healthScore));
    }
    
    /**
     * Calculate distance to another node
     * 
     * @param otherNode Other network node
     * @return Distance in meters
     */
    public double calculateDistance(NetworkNode otherNode) {
        if (location == null || otherNode.getLocation() == null) {
            return Double.MAX_VALUE;
        }
        
        return location.calculateDistance(otherNode.getLocation());
    }
    
    /**
     * Check if node is stale (inactive for too long)
     * 
     * @param timeoutSeconds Timeout in seconds
     * @return True if node is stale
     */
    public boolean isStale(long timeoutSeconds) {
        return Instant.now().isAfter(lastActivity.plusSeconds(timeoutSeconds));
    }
    
    @Override
    public String toString() {
        return String.format("NetworkNode{id=%s, type=%s, location=%s, active=%s, health=%.1f}",
            nodeId, nodeType, location, isActive, healthScore);
    }
    
    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (obj == null || getClass() != obj.getClass()) return false;
        NetworkNode that = (NetworkNode) obj;
        return nodeId.equals(that.nodeId);
    }
    
    @Override
    public int hashCode() {
        return nodeId.hashCode();
    }
} 