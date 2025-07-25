package org.nci.fogedge.topology;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

/**
 * Class representing a Fog Node in the network topology
 * 
 * This class models a fog computing node that receives data from edge nodes,
 * performs advanced processing and security measures, and forwards data to the cloud.
 * It implements advanced security countermeasures mentioned in the research paper.
 * 
 * @author NCI H9FEC Student
 * @version 1.0
 */
public class FogNode {
    
    private String id;
    private Location location;
    private double processingCapacity; // in MIPS
    private double memory; // in MB
    private List<EdgeNode> connectedEdgeNodes;
    private CloudDatacenter connectedCloud;
    private Map<String, Double> receivedData; // Data ID -> Receive time
    private Map<String, Double> processedData; // Data ID -> Process time
    private Random random;
    
    // Security features
    private boolean intrusionDetectionEnabled;
    private boolean encryptionEnabled;
    private boolean authenticationEnabled;
    private boolean blockchainEnabled;
    private boolean decoyTechniqueEnabled;
    private int securityIncidentsDetected;
    private int securityIncidentsMitigated;
    
    /**
     * Constructor with parameters
     * @param id Node ID
     * @param location Node location
     * @param processingCapacity Processing capacity in MIPS
     * @param memory Memory in MB
     */
    public FogNode(String id, Location location, double processingCapacity, double memory) {
        this.id = id;
        this.location = location;
        this.processingCapacity = processingCapacity;
        this.memory = memory;
        this.connectedEdgeNodes = new ArrayList<>();
        this.receivedData = new HashMap<>();
        this.processedData = new HashMap<>();
        this.intrusionDetectionEnabled = true;
        this.encryptionEnabled = true;
        this.authenticationEnabled = true;
        this.blockchainEnabled = true;
        this.decoyTechniqueEnabled = true;
        this.securityIncidentsDetected = 0;
        this.securityIncidentsMitigated = 0;
        this.random = new Random(System.currentTimeMillis());
    }
    
    /**
     * Receive data from an edge node
     * @param dataId ID of received data
     * @param sourceEdge Source edge node
     * @param currentTime Current simulation time
     * @return True if data received successfully, false otherwise
     */
    public boolean receiveData(String dataId, EdgeNode sourceEdge, double currentTime) {
        // Record received data
        receivedData.put(dataId, currentTime);
        
        // Check for security incidents
        boolean securityIncident = checkForSecurityIncident(dataId, sourceEdge);
        
        if (securityIncident) {
            securityIncidentsDetected++;
            
            // Attempt to mitigate security incident
            boolean mitigated = mitigateSecurityIncident(dataId, sourceEdge);
            
            if (mitigated) {
                securityIncidentsMitigated++;
                System.out.println("Fog node " + id + " mitigated security incident for data " + dataId);
            } else {
                System.out.println("Fog node " + id + " failed to mitigate security incident for data " + dataId);
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
     * Process data and forward to cloud
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
        
        // Apply fog analytics and aggregation
        String processedDataId = applyFogAnalytics(dataId);
        
        // Forward to cloud if needed
        // In fog computing, not all data needs to go to the cloud
        if (shouldForwardToCloud(processedDataId)) {
            if (connectedCloud != null) {
                boolean success = connectedCloud.receiveData(processedDataId, this, currentTime);
                return success;
            } else {
                System.err.println("Error: Fog node " + id + " not connected to any cloud datacenter");
                return false;
            }
        }
        
        return true; // Data processed successfully at fog level
    }
    
    /**
     * Apply fog analytics and aggregation to data
     * @param dataId ID of data to process
     * @return Processed data ID
     */
    private String applyFogAnalytics(String dataId) {
        // Apply fog analytics (data aggregation, advanced processing, etc.)
        return "FOG_PROCESSED(" + dataId + ")";
    }
    
    /**
     * Determine if data should be forwarded to cloud
     * @param dataId ID of data to check
     * @return True if data should be forwarded to cloud, false otherwise
     */
    private boolean shouldForwardToCloud(String dataId) {
        // In a real implementation, this would involve actual decision logic
        // For simulation, forward 30% of data to cloud
        return random.nextDouble() < 0.3;
    }
    
    /**
     * Check for security incidents in received data
     * @param dataId ID of data to check
     * @param sourceEdge Source edge node
     * @return True if security incident detected, false otherwise
     */
    private boolean checkForSecurityIncident(String dataId, EdgeNode sourceEdge) {
        if (!intrusionDetectionEnabled) {
            return false;
        }
        
        // Simulate security incident detection
        // In a real implementation, this would involve actual security checks
        double incidentProbability = 0.03; // 3% chance of security incident
        return random.nextDouble() < incidentProbability;
    }
    
    /**
     * Attempt to mitigate a detected security incident
     * @param dataId ID of data with security incident
     * @param sourceEdge Source edge node
     * @return True if incident mitigated successfully, false otherwise
     */
    private boolean mitigateSecurityIncident(String dataId, EdgeNode sourceEdge) {
        // Simulate security incident mitigation
        // In a real implementation, this would involve actual mitigation techniques
        
        // Calculate base mitigation success probability
        double mitigationSuccessProbability = 0.85; // 85% base chance of successful mitigation
        
        // Adjust based on enabled security features
        if (blockchainEnabled) {
            mitigationSuccessProbability += 0.05; // +5% with blockchain
        }
        
        if (decoyTechniqueEnabled) {
            mitigationSuccessProbability += 0.05; // +5% with decoy technique
        }
        
        // Cap at 95% success probability
        mitigationSuccessProbability = Math.min(mitigationSuccessProbability, 0.95);
        
        return random.nextDouble() < mitigationSuccessProbability;
    }
    
    /**
     * Calculate processing time for data
     * @param dataId ID of data to process
     * @return Processing time in simulation time units
     */
    private double calculateProcessingTime(String dataId) {
        // Base processing time
        double baseTime = 0.005; // 5 ms
        
        // Adjust based on processing capacity
        baseTime = baseTime * (2000 / processingCapacity);
        
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
        
        if (blockchainEnabled) {
            baseTime *= 1.15; // 15% overhead
        }
        
        if (decoyTechniqueEnabled) {
            baseTime *= 1.1; // 10% overhead
        }
        
        return baseTime;
    }
    
    /**
     * Add a connected edge node
     * @param edge Edge node to connect
     */
    public void addConnectedEdgeNode(EdgeNode edge) {
        connectedEdgeNodes.add(edge);
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
     * Get connected edge nodes
     * @return List of connected edge nodes
     */
    public List<EdgeNode> getConnectedEdgeNodes() {
        return connectedEdgeNodes;
    }
    
    /**
     * Get connected cloud datacenter
     * @return Connected cloud datacenter
     */
    public CloudDatacenter getConnectedCloud() {
        return connectedCloud;
    }
    
    /**
     * Set connected cloud datacenter
     * @param connectedCloud Cloud datacenter to connect to
     */
    public void setConnectedCloud(CloudDatacenter connectedCloud) {
        this.connectedCloud = connectedCloud;
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
     * Check if blockchain is enabled
     * @return True if blockchain is enabled, false otherwise
     */
    public boolean isBlockchainEnabled() {
        return blockchainEnabled;
    }
    
    /**
     * Enable or disable blockchain
     * @param blockchainEnabled True to enable blockchain, false to disable
     */
    public void setBlockchainEnabled(boolean blockchainEnabled) {
        this.blockchainEnabled = blockchainEnabled;
    }
    
    /**
     * Check if decoy technique is enabled
     * @return True if decoy technique is enabled, false otherwise
     */
    public boolean isDecoyTechniqueEnabled() {
        return decoyTechniqueEnabled;
    }
    
    /**
     * Enable or disable decoy technique
     * @param decoyTechniqueEnabled True to enable decoy technique, false to disable
     */
    public void setDecoyTechniqueEnabled(boolean decoyTechniqueEnabled) {
        this.decoyTechniqueEnabled = decoyTechniqueEnabled;
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
    
    @Override
    public String toString() {
        return "FogNode{" +
                "id='" + id + '\'' +
                ", location=" + location +
                ", processingCapacity=" + processingCapacity +
                ", memory=" + memory +
                ", connectedEdgeNodes=" + connectedEdgeNodes.size() +
                '}';
    }
}
