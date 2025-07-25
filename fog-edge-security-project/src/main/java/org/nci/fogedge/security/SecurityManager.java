package org.nci.fogedge.security;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

/**
 * Security Manager class for the fog computing environment
 * 
 * This class manages security features and countermeasures for the fog computing
 * environment as described in the research paper. It implements various security
 * mechanisms including encryption, authentication, intrusion detection, blockchain,
 * and decoy techniques.
 * 
 * @author NCI H9FEC Student
 * @version 1.0
 */
public class SecurityManager {
    
    /**
     * Enum for authentication schemes
     */
    public enum AuthScheme {
        BASIC_AUTHENTICATION,
        MUTUAL_AUTHENTICATION,
        MULTI_TIER_AUTHENTICATION
    }
    
    /**
     * Enum for encryption algorithms
     */
    public enum EncryptionAlgorithm {
        AES_128,
        AES_256,
        RSA_2048
    }
    
    private boolean encryptionEnabled;
    private boolean intrusionDetectionEnabled;
    private boolean blockchainEnabled;
    private boolean decoyTechniqueEnabled;
    private AuthScheme authenticationScheme;
    private EncryptionAlgorithm encryptionAlgorithm;
    
    // Security incident tracking
    private List<SecurityIncident> securityIncidents;
    private Map<String, Integer> incidentsByType;
    private Random random;
    
    /**
     * Default constructor
     */
    public SecurityManager() {
        this.encryptionEnabled = true;
        this.intrusionDetectionEnabled = true;
        this.blockchainEnabled = false;
        this.decoyTechniqueEnabled = false;
        this.authenticationScheme = AuthScheme.BASIC_AUTHENTICATION;
        this.encryptionAlgorithm = EncryptionAlgorithm.AES_256;
        this.securityIncidents = new ArrayList<>();
        this.incidentsByType = new HashMap<>();
        this.random = new Random(System.currentTimeMillis());
        
        // Initialize incident types
        incidentsByType.put("DoS", 0);
        incidentsByType.put("DDoS", 0);
        incidentsByType.put("ManInTheMiddle", 0);
        incidentsByType.put("DataTampering", 0);
        incidentsByType.put("Eavesdropping", 0);
    }
    
    /**
     * Encrypt data using the configured encryption algorithm
     * @param data Data to encrypt
     * @return Encrypted data
     */
    public String encryptData(String data) {
        if (!encryptionEnabled) {
            return data;
        }
        
        // In a real implementation, this would use actual encryption
        // For simulation, we just indicate the encryption algorithm used
        return encryptionAlgorithm.name() + "(" + data + ")";
    }
    
    /**
     * Decrypt data using the configured encryption algorithm
     * @param encryptedData Encrypted data
     * @return Decrypted data
     */
    public String decryptData(String encryptedData) {
        if (!encryptionEnabled) {
            return encryptedData;
        }
        
        // In a real implementation, this would use actual decryption
        // For simulation, we just remove the encryption indicator
        for (EncryptionAlgorithm algo : EncryptionAlgorithm.values()) {
            String prefix = algo.name() + "(";
            if (encryptedData.startsWith(prefix) && encryptedData.endsWith(")")) {
                return encryptedData.substring(prefix.length(), encryptedData.length() - 1);
            }
        }
        
        return encryptedData; // Not encrypted or unknown format
    }
    
    /**
     * Authenticate a device or node
     * @param deviceId ID of device or node to authenticate
     * @param credentials Authentication credentials
     * @return True if authentication successful, false otherwise
     */
    public boolean authenticate(String deviceId, String credentials) {
        // In a real implementation, this would use actual authentication
        // For simulation, we use a simple success probability based on the scheme
        double successProbability;
        
        switch (authenticationScheme) {
            case BASIC_AUTHENTICATION:
                successProbability = 0.95; // 95% success
                break;
            case MUTUAL_AUTHENTICATION:
                successProbability = 0.98; // 98% success
                break;
            case MULTI_TIER_AUTHENTICATION:
                successProbability = 0.99; // 99% success
                break;
            default:
                successProbability = 0.9; // Default 90% success
        }
        
        return random.nextDouble() < successProbability;
    }
    
    /**
     * Check for security incidents in data
     * @param data Data to check
     * @param sourceId ID of data source
     * @param destinationId ID of data destination
     * @return Security incident if detected, null otherwise
     */
    public SecurityIncident checkForSecurityIncident(String data, String sourceId, String destinationId) {
        if (!intrusionDetectionEnabled) {
            return null;
        }
        
        // In a real implementation, this would use actual security checks
        // For simulation, we use a simple incident probability
        double incidentProbability = 0.05; // 5% chance of incident
        
        if (random.nextDouble() < incidentProbability) {
            // Generate a random incident type
            String[] incidentTypes = {"DoS", "DDoS", "ManInTheMiddle", "DataTampering", "Eavesdropping"};
            String incidentType = incidentTypes[random.nextInt(incidentTypes.length)];
            
            // Create and record the incident
            SecurityIncident incident = new SecurityIncident(
                    incidentType,
                    "Detected " + incidentType + " attack from " + sourceId + " to " + destinationId,
                    System.currentTimeMillis(),
                    calculateSeverity(incidentType),
                    sourceId,
                    destinationId
            );
            
            securityIncidents.add(incident);
            incidentsByType.put(incidentType, incidentsByType.get(incidentType) + 1);
            
            return incident;
        }
        
        return null;
    }
    
    /**
     * Attempt to mitigate a security incident
     * @param incident Security incident to mitigate
     * @return True if mitigation successful, false otherwise
     */
    public boolean mitigateSecurityIncident(SecurityIncident incident) {
        // In a real implementation, this would use actual mitigation techniques
        // For simulation, we use a success probability based on enabled features
        
        double baseProbability = 0.7; // 70% base success probability
        
        // Adjust based on enabled features
        if (blockchainEnabled) {
            baseProbability += 0.1; // +10% with blockchain
        }
        
        if (decoyTechniqueEnabled) {
            baseProbability += 0.1; // +10% with decoy technique
        }
        
        // Adjust based on incident type
        switch (incident.getType()) {
            case "DoS":
            case "DDoS":
                // These are harder to mitigate
                baseProbability -= 0.1;
                break;
            case "Eavesdropping":
                // Easier to mitigate with encryption
                if (encryptionEnabled) {
                    baseProbability += 0.1;
                }
                break;
        }
        
        // Cap probability between 0.5 and 0.95
        baseProbability = Math.max(0.5, Math.min(0.95, baseProbability));
        
        boolean success = random.nextDouble() < baseProbability;
        
        // Update incident with mitigation result
        incident.setMitigated(success);
        incident.setMitigationTime(System.currentTimeMillis());
        
        return success;
    }
    
    /**
     * Calculate severity of an incident type
     * @param incidentType Type of incident
     * @return Severity level (1-5)
     */
    private int calculateSeverity(String incidentType) {
        switch (incidentType) {
            case "DoS":
                return 3;
            case "DDoS":
                return 4;
            case "ManInTheMiddle":
                return 5;
            case "DataTampering":
                return 4;
            case "Eavesdropping":
                return 2;
            default:
                return 3;
        }
    }
    
    /**
     * Generate a decoy response for a potential attacker
     * @param originalData Original data
     * @return Decoy data
     */
    public String generateDecoyResponse(String originalData) {
        if (!decoyTechniqueEnabled) {
            return originalData;
        }
        
        // In a real implementation, this would generate actual decoy data
        // For simulation, we just indicate it's a decoy
        return "DECOY(" + originalData + ")";
    }
    
    /**
     * Apply blockchain security to data
     * @param data Data to secure
     * @return Blockchain-secured data
     */
    public String applyBlockchainSecurity(String data) {
        if (!blockchainEnabled) {
            return data;
        }
        
        // In a real implementation, this would use actual blockchain techniques
        // For simulation, we just indicate blockchain is used
        return "BLOCKCHAIN(" + data + ")";
    }
    
    /**
     * Enable or disable encryption
     * @param enabled True to enable, false to disable
     */
    public void enableEncryption(boolean enabled) {
        this.encryptionEnabled = enabled;
    }
    
    /**
     * Enable or disable intrusion detection
     * @param enabled True to enable, false to disable
     */
    public void enableIntrusionDetection(boolean enabled) {
        this.intrusionDetectionEnabled = enabled;
    }
    
    /**
     * Enable or disable blockchain
     * @param enabled True to enable, false to disable
     */
    public void enableBlockchain(boolean enabled) {
        this.blockchainEnabled = enabled;
    }
    
    /**
     * Enable or disable decoy technique
     * @param enabled True to enable, false to disable
     */
    public void enableDecoyTechnique(boolean enabled) {
        this.decoyTechniqueEnabled = enabled;
    }
    
    /**
     * Set authentication scheme
     * @param scheme Authentication scheme to use
     */
    public void enableAuthenticationScheme(AuthScheme scheme) {
        this.authenticationScheme = scheme;
    }
    
    /**
     * Set encryption algorithm
     * @param algorithm Encryption algorithm to use
     */
    public void setEncryptionAlgorithm(EncryptionAlgorithm algorithm) {
        this.encryptionAlgorithm = algorithm;
    }
    
    /**
     * Get list of security incidents
     * @return List of security incidents
     */
    public List<SecurityIncident> getSecurityIncidents() {
        return securityIncidents;
    }
    
    /**
     * Get map of incidents by type
     * @return Map of incident counts by type
     */
    public Map<String, Integer> getIncidentsByType() {
        return incidentsByType;
    }
    
    /**
     * Get total number of security incidents
     * @return Total number of security incidents
     */
    public int getTotalSecurityIncidents() {
        return securityIncidents.size();
    }
    
    /**
     * Get number of mitigated security incidents
     * @return Number of mitigated security incidents
     */
    public int getMitigatedSecurityIncidents() {
        int count = 0;
        for (SecurityIncident incident : securityIncidents) {
            if (incident.isMitigated()) {
                count++;
            }
        }
        return count;
    }
    
    /**
     * Get mitigation success rate
     * @return Mitigation success rate (0-1)
     */
    public double getMitigationSuccessRate() {
        if (securityIncidents.isEmpty()) {
            return 1.0; // No incidents, perfect success rate
        }
        return (double) getMitigatedSecurityIncidents() / getTotalSecurityIncidents();
    }
}
