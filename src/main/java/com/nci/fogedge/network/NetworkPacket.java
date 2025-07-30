package com.nci.fogedge.network;

import java.time.Instant;

/**
 * Network Packet for Fog and Edge Computing System
 * 
 * This class represents a data packet transmitted through the network.
 * It includes metadata such as source, timestamp, and payload.
 * 
 * @author National College of Ireland - Fog and Edge Computing Project
 * @version 1.0.0
 * @since 2024
 */
public class NetworkPacket {
    
    private final String sourceId;
    private final byte[] data;
    private final Instant timestamp;
    private final String packetId;
    
    /**
     * Constructor for NetworkPacket
     * 
     * @param sourceId Source device/node ID
     * @param data Packet payload data
     */
    public NetworkPacket(String sourceId, byte[] data) {
        this.sourceId = sourceId;
        this.data = data;
        this.timestamp = Instant.now();
        this.packetId = generatePacketId();
    }
    
    /**
     * Get source device/node ID
     * 
     * @return Source ID
     */
    public String getSourceId() {
        return sourceId;
    }
    
    /**
     * Get packet payload data
     * 
     * @return Packet data
     */
    public byte[] getData() {
        return data;
    }
    
    /**
     * Get packet timestamp
     * 
     * @return Timestamp when packet was created
     */
    public Instant getTimestamp() {
        return timestamp;
    }
    
    /**
     * Get packet ID
     * 
     * @return Unique packet identifier
     */
    public String getPacketId() {
        return packetId;
    }
    
    /**
     * Get packet size in bytes
     * 
     * @return Packet size
     */
    public int getSize() {
        return data != null ? data.length : 0;
    }
    
    /**
     * Generate unique packet ID
     * 
     * @return Unique packet identifier
     */
    private String generatePacketId() {
        return sourceId + "_" + System.currentTimeMillis() + "_" + System.nanoTime();
    }
    
    @Override
    public String toString() {
        return String.format("NetworkPacket{id=%s, source=%s, size=%d bytes, timestamp=%s}",
            packetId, sourceId, getSize(), timestamp);
    }
} 