package org.nci.fogedge.topology;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

/**
 * Class representing an Edge Node in the network topology
 * 
 * This class models an edge computing node that receives data from IoT devices,
 * performs initial processing and security checks, and forwards data to fog nodes.
 * It implements security countermeasures mentioned in the research paper.
 * 
 * @author NCI H9FEC Student
 * @version 1.0
 */
public class EdgeNode {
    
    private String id;
    private Location location;
    private double processingCapacity; // in MIPS
    private double memory; // in MB
    private List<IoTDevice> connectedDevices;
    private FogNode connectedFogNode;
    private List<FogNode> connectedFogNodes; // Support for multiple fog nodes
    private Map<String, Double> receivedData; // Data ID -> Receive time
    private Map<String, Double> processedData; // Data ID -> Process time
    private boolean intrusionDetectionEnabled;
    private boolean encryptionEnabled;
    private boolean authenticationEnabled;
    private Random random;
    
    // Security parameters
    private int securityIncidentsDetected;
    private int securityIncidentsMitigated;
    
    /**
     * Constructor with parameters
     * @param id Node ID
     * @param location Node location
     * @param processingCapacity Processing capacity in MIPS
     * @param memory Memory in MB
     */
    public EdgeNode(String id, Location location, double processingCapacity, double memory) {
        this.id = id;
        this.location = location;
        this.processingCapacity = processingCapacity;
        this.memory = memory;
        this.connectedDevices = new ArrayList<>();
        this.connectedFogNodes = new ArrayList<>();
        this.receivedData = new HashMap<>();
        this.processedData = new HashMap<>();
        this.intrusionDetectionEnabled = true;
        this.encryptionEnabled = true;
        this.authenticationEnabled = true;
        this.securityIncidentsDetected = 0;
        this.securityIncidentsMitigated = 0;
        this.random = new Random(System.currentTimeMillis());
    }
    
    /**
     * Receive data from an IoT device
     * @param dataId ID of received data
     * @param sourceDevice Source IoT device
     * @param currentTime Current simulation time
     * @return True if data received successfully, false otherwise
     */
    public boolean receiveData(String dataId, IoTDevice sourceDevice, double currentTime) {
        // Record received data
        receivedData.put(dataId, currentTime);
        
        // Check for security incidents
        boolean securityIncident = checkForSecurityIncident(dataId, sourceDevice);
        
        if (securityIncident) {
            securityIncidentsDetected++;
            
            // Attempt to mitigate security incident
            boolean mitigated = mitigateSecurityIncident(dataId, sourceDevice);
            
            if (mitigated) {
                securityIncidentsMitigated++;
                System.out.println("Edge node " + id + " mitigated security incident for data " + dataId);
            } else {
                System.out.println("Edge node " + id + " failed to mitigate security incident for data " + dataId);
                return false; // Data rejected due to security incident
            }
        }
        
        // Process data
        double processingTime = calculateProcessingTime(dataId);
        double processCompleteTime = currentTime + processingTime;
        
        // Record processed data
        processedData.put(dataId, processCompleteTime);
        
        return true;
    }
    
    /**
     * Process data and forward to fog node
     * @param dataId ID of data to process
     * @param currentTime Current simulation time
     * @return True if data processed and forwarded successfully, false otherwise
     */
    public boolean processAndForwardData(String dataId, double currentTime) {
        // Check if data has been received
        if (!receivedData.containsKey(dataId)) {
            return false;
        }
        
        // Check if data has been processed
        if (!processedData.containsKey(dataId)) {
            return false;
        }
        
        // Check if processing is complete
        if (processedData.get(dataId) > currentTime) {
            return false; // Processing not yet complete
        }
        
        // Apply edge analytics and filtering
        String processedDataId = applyEdgeAnalytics(dataId);
        
        // Forward to fog node
        if (connectedFogNode != null) {
            boolean success = connectedFogNode.receiveData(processedDataId, this, currentTime);
            return success;
        } else {
            System.err.println("Error: Edge node " + id + " not connected to any fog node");
            return false;
        }
    }
    
    /**
     * Apply edge analytics and filtering to data
     * @param dataId ID of data to process
     * @return Processed data ID
     */
    private String applyEdgeAnalytics(String dataId) {
        // Apply edge analytics (data filtering, aggregation, etc.)
        return "EDGE_PROCESSED(" + dataId + ")";
    }
    
    /**
     * Check for security incidents in received data
     * @param dataId ID of data to check
     * @param sourceDevice Source IoT device
     * @return True if security incident detected, false otherwise
     */
    private boolean checkForSecurityIncident(String dataId, IoTDevice sourceDevice) {
        if (!intrusionDetectionEnabled) {
            return false;
        }
        
        // Simulate security incident detection
        // In a real implementation, this would involve actual security checks
        double incidentProbability = 0.05; // 5% chance of security incident
        return random.nextDouble() < incidentProbability;
    }
    
    /**
     * Attempt to mitigate a detected security incident
     * @param dataId ID of data with security incident
     * @param sourceDevice Source IoT device
     * @return True if incident mitigated successfully, false otherwise
     */
    private boolean mitigateSecurityIncident(String dataId, IoTDevice sourceDevice) {
        // Simulate security incident mitigation
        // In a real implementation, this would involve actual mitigation techniques
        double mitigationSuccessProbability = 0.8; // 80% chance of successful mitigation
        return random.nextDouble() < mitigationSuccessProbability;
    }
    
    /**
     * Calculate processing time for data
     * @param dataId ID of data to process
     * @return Processing time in simulation time units
     */
    public double calculateProcessingTime(String dataId) {
        // Base processing time
        double baseTime = 0.01; // 10 ms
        
        // Adjust based on processing capacity
        baseTime = baseTime * (1000 / processingCapacity);
        
        // Add security overhead if enabled
        if (intrusionDetectionEnabled) {
            baseTime *= 1.1; // 10% overhead
        }
        
        if (encryptionEnabled) {
            baseTime *= 1.05; // 5% overhead
        }
        
        if (authenticationEnabled) {
            baseTime *= 1.05; // 5% overhead
        }
        
        return baseTime;
    }
    
    /**
     * Add a connected IoT device
     * @param device IoT device to connect
     */
    public void addConnectedDevice(IoTDevice device) {
        connectedDevices.add(device);
    }
    
    /**
     * Get node ID
     * @return Node ID
     */
    public String getId() {
        return id;
    }
    
    /**
     * Get node location
     * @return Node location
     */
    public Location getLocation() {
        return location;
    }
    
    /**
     * Get processing capacity
     * @return Processing capacity in MIPS
     */
    public double getProcessingCapacity() {
        return processingCapacity;
    }
    
    /**
     * Get memory
     * @return Memory in MB
     */
    public double getMemory() {
        return memory;
    }
    
    /**
     * Get connected IoT devices
     * @return List of connected IoT devices
     */
    public List<IoTDevice> getConnectedDevices() {
        return connectedDevices;
    }
    
    /**
     * Get connected fog node
     * @return Connected fog node
     */
    public FogNode getConnectedFogNode() {
        return connectedFogNode;
    }
    
    /**
     * Set connected fog node
     * @param connectedFogNode Fog node to connect to
     */
    public void setConnectedFogNode(FogNode connectedFogNode) {
        this.connectedFogNode = connectedFogNode;
        
        // Also add to the list of connected fog nodes if not already present
        if (connectedFogNode != null && !connectedFogNodes.contains(connectedFogNode)) {
            connectedFogNodes.add(connectedFogNode);
        }
    }
    
    /**
     * Add a connected fog node
     * @param fogNode Fog node to connect to
     */
    public void addConnectedFogNode(FogNode fogNode) {
        if (fogNode != null && !connectedFogNodes.contains(fogNode)) {
            connectedFogNodes.add(fogNode);
        }
        
        // For backward compatibility, if this is the first fog node, also set it as the primary
        if (connectedFogNode == null && !connectedFogNodes.isEmpty()) {
            connectedFogNode = connectedFogNodes.get(0);
        }
    }
    
    /**
     * Get all connected fog nodes
     * @return List of connected fog nodes
     */
    public List<FogNode> getConnectedFogNodes() {
        return connectedFogNodes;
    }
    
    /**
     * Check if intrusion detection is enabled
     * @return True if intrusion detection is enabled, false otherwise
     */
    public boolean isIntrusionDetectionEnabled() {
        return intrusionDetectionEnabled;
    }
    
    /**
     * Enable or disable intrusion detection
     * @param intrusionDetectionEnabled True to enable intrusion detection, false to disable
     */
    public void setIntrusionDetectionEnabled(boolean intrusionDetectionEnabled) {
        this.intrusionDetectionEnabled = intrusionDetectionEnabled;
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
     * Get number of security incidents detected
     * @return Number of security incidents detected
     */
    public int getSecurityIncidentsDetected() {
        return securityIncidentsDetected;
    }
    
    /**
     * Get number of security incidents mitigated
     * @return Number of security incidents mitigated
     */
    public int getSecurityIncidentsMitigated() {
        return securityIncidentsMitigated;
    }
    
    /**
     * Process data and return processed data ID
     * @param dataId ID of data to process
     * @param currentTime Current simulation time
     * @return Processed data ID
     */
    public String processData(String dataId, double currentTime) {
        // Apply edge analytics to data
        String processedDataId = applyEdgeAnalytics(dataId);
        
        // Record processed data with timestamp
        processedData.put(processedDataId, currentTime + calculateProcessingTime(dataId));
        
        return processedDataId;
    }
    
    /**
     * Handle a security incident
     * @param dataId ID of data with security incident
     * @param currentTime Current simulation time
     * @return True if incident handled successfully, false otherwise
     */
    public boolean handleSecurityIncident(String dataId, double currentTime) {
        securityIncidentsDetected++;
        
        // Simulate security incident handling with 70% success rate
        boolean mitigated = random.nextDouble() < 0.7;
        
        if (mitigated) {
            securityIncidentsMitigated++;
        }
        
        return mitigated;
    }
    
    @Override
    public String toString() {
        return "EdgeNode{" +
                "id='" + id + '\'' +
                ", location=" + location +
                ", processingCapacity=" + processingCapacity +
                ", memory=" + memory +
                ", connectedDevices=" + connectedDevices.size() +
                '}';
    }
}
