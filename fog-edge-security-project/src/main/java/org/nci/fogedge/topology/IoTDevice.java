package org.nci.fogedge.topology;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Class representing an IoT device in the network topology
 * 
 * This class models an IoT device that generates data and sends it to an edge node.
 * It implements various security features based on the research paper.
 * 
 * @author NCI H9FEC Student
 * @version 1.0
 */
public class IoTDevice {
    
    private String id;
    private Location location;
    private String deviceType;
    private double dataGenerationRate; // packets per second
    private EdgeNode connectedEdgeNode; // For backward compatibility
    private List<EdgeNode> connectedEdgeNodes; // Support for multiple edge nodes
    private boolean encryptionEnabled;
    private String encryptionAlgorithm;
    private boolean authenticationEnabled;
    private List<String> generatedDataIds;
    private Random random;
    
    /**
     * Constructor with parameters
     * @param id Device ID
     * @param location Device location
     * @param deviceType Type of device
     * @param dataGenerationRate Data generation rate
     */
    public IoTDevice(String id, Location location, String deviceType, double dataGenerationRate) {
        this.id = id;
        this.location = location;
        this.deviceType = deviceType;
        this.dataGenerationRate = dataGenerationRate;
        this.encryptionEnabled = true;
        this.encryptionAlgorithm = "AES-256";
        this.authenticationEnabled = true;
        this.generatedDataIds = new ArrayList<>();
        this.connectedEdgeNodes = new ArrayList<>();
        this.random = new Random(System.currentTimeMillis());
    }
    
    /**
     * Generate data packet
     * @param currentTime Current simulation time
     * @return Generated data packet ID or null if no data generated
     */
    public String generateData(double currentTime) {
        // Check if data should be generated based on rate
        if (random.nextDouble() > dataGenerationRate) {
            return null; // No data generated this time
        }
        
        // Generate data packet ID
        String dataId = id + "_" + currentTime + "_" + random.nextInt(1000);
        generatedDataIds.add(dataId);
        
        return dataId;
    }
    
    /**
     * Send data to connected edge node
     * @param dataId ID of data to send
     * @param currentTime Current simulation time
     * @return True if data sent successfully, false otherwise
     */
    public boolean sendData(String dataId, double currentTime) {
        if (connectedEdgeNodes.isEmpty() && connectedEdgeNode == null) {
            System.err.println("Error: IoT device " + id + " not connected to any edge node");
            return false;
        }
        
        // Apply security measures before sending
        String securedDataId = applySecurityMeasures(dataId);
        
        boolean success = false;
        
        // If we have multiple edge nodes, use those
        if (!connectedEdgeNodes.isEmpty()) {
            // For simplicity, send to the first connected edge node
            EdgeNode targetEdge = connectedEdgeNodes.get(0);
            success = targetEdge.receiveData(securedDataId, this, currentTime);
        } else {
            // For backward compatibility
            success = connectedEdgeNode.receiveData(securedDataId, this, currentTime);
        }
        
        return success;
    }
    
    /**
     * Apply security measures to data before sending
     * @param dataId Data ID to secure
     * @return Secured data ID
     */
    private String applySecurityMeasures(String dataId) {
        String securedDataId = dataId;
        
        // Apply encryption if enabled
        if (encryptionEnabled) {
            securedDataId = "ENC(" + securedDataId + ")";
        }
        
        // Apply authentication if enabled
        if (authenticationEnabled) {
            securedDataId = "AUTH(" + securedDataId + ")";
        }
        
        return securedDataId;
    }
    
    /**
     * Get device ID
     * @return Device ID
     */
    public String getId() {
        return id;
    }
    
    /**
     * Get device location
     * @return Device location
     */
    public Location getLocation() {
        return location;
    }
    
    /**
     * Get device type
     * @return Device type
     */
    public String getDeviceType() {
        return deviceType;
    }
    
    /**
     * Get data generation rate
     * @return Data generation rate
     */
    public double getDataGenerationRate() {
        return dataGenerationRate;
    }
    
    /**
     * Set data generation rate
     * @param dataGenerationRate New data generation rate
     */
    public void setDataGenerationRate(double dataGenerationRate) {
        this.dataGenerationRate = dataGenerationRate;
    }
    
    /**
     * Get connected edge node
     * @return Connected edge node
     */
    public EdgeNode getConnectedEdgeNode() {
        return connectedEdgeNode;
    }
    
    /**
     * Set connected edge node
     * @param connectedEdgeNode Edge node to connect to
     */
    public void setConnectedEdgeNode(EdgeNode connectedEdgeNode) {
        this.connectedEdgeNode = connectedEdgeNode;
        
        // Also add to the list of connected edge nodes if not already present
        if (connectedEdgeNode != null && !connectedEdgeNodes.contains(connectedEdgeNode)) {
            connectedEdgeNodes.add(connectedEdgeNode);
        }
    }
    
    /**
     * Add a connected edge node
     * @param edgeNode Edge node to connect to
     */
    public void addConnectedEdgeNode(EdgeNode edgeNode) {
        if (edgeNode != null && !connectedEdgeNodes.contains(edgeNode)) {
            connectedEdgeNodes.add(edgeNode);
        }
        
        // For backward compatibility, if this is the first edge node, also set it as the primary
        if (connectedEdgeNode == null && !connectedEdgeNodes.isEmpty()) {
            connectedEdgeNode = connectedEdgeNodes.get(0);
        }
    }
    
    /**
     * Get all connected edge nodes
     * @return List of connected edge nodes
     */
    public List<EdgeNode> getConnectedEdgeNodes() {
        return connectedEdgeNodes;
    }
    
    /**
     * Check if encryption is enabled
     * @return True if encryption is enabled, false otherwise
     */
    public boolean isEncryptionEnabled() {
        return encryptionEnabled;
    }
    
    /**
     * Enable or disable encryption
     * @param encryptionEnabled True to enable encryption, false to disable
     */
    public void setEncryptionEnabled(boolean encryptionEnabled) {
        this.encryptionEnabled = encryptionEnabled;
    }
    
    /**
     * Get encryption algorithm
     * @return Encryption algorithm
     */
    public String getEncryptionAlgorithm() {
        return encryptionAlgorithm;
    }
    
    /**
     * Set encryption algorithm
     * @param encryptionAlgorithm New encryption algorithm
     */
    public void setEncryptionAlgorithm(String encryptionAlgorithm) {
        this.encryptionAlgorithm = encryptionAlgorithm;
    }
    
    /**
     * Check if authentication is enabled
     * @return True if authentication is enabled, false otherwise
     */
    public boolean isAuthenticationEnabled() {
        return authenticationEnabled;
    }
    
    /**
     * Enable or disable authentication
     * @param authenticationEnabled True to enable authentication, false to disable
     */
    public void setAuthenticationEnabled(boolean authenticationEnabled) {
        this.authenticationEnabled = authenticationEnabled;
    }
    
    /**
     * Get list of generated data IDs
     * @return List of generated data IDs
     */
    public List<String> getGeneratedDataIds() {
        return generatedDataIds;
    }
    
    @Override
    public String toString() {
        return "IoTDevice{" +
                "id='" + id + '\'' +
                ", location=" + location +
                ", deviceType='" + deviceType + '\'' +
                ", dataGenerationRate=" + dataGenerationRate +
                '}';
    }
}
