package org.nci.fogedge.security;

/**
 * Class representing a security incident in the fog computing environment
 * 
 * This class models a security incident detected in the network, including
 * its type, description, timestamp, severity, and mitigation status.
 * 
 * @author NCI H9FEC Student
 * @version 1.0
 */
public class SecurityIncident {
    
    private String type;
    private String description;
    private long detectionTime;
    private int severity; // 1-5 scale, 5 being most severe
    private String sourceId;
    private String destinationId;
    private boolean mitigated;
    private long mitigationTime;
    
    /**
     * Constructor with parameters
     * @param type Type of security incident
     * @param description Description of the incident
     * @param detectionTime Time of detection
     * @param severity Severity level (1-5)
     * @param sourceId Source ID of the incident
     * @param destinationId Destination ID of the incident
     */
    public SecurityIncident(String type, String description, long detectionTime, 
            int severity, String sourceId, String destinationId) {
        this.type = type;
        this.description = description;
        this.detectionTime = detectionTime;
        this.severity = severity;
        this.sourceId = sourceId;
        this.destinationId = destinationId;
        this.mitigated = false;
        this.mitigationTime = 0;
    }
    
    /**
     * Get incident type
     * @return Incident type
     */
    public String getType() {
        return type;
    }
    
    /**
     * Get incident description
     * @return Incident description
     */
    public String getDescription() {
        return description;
    }
    
    /**
     * Get detection time
     * @return Detection time
     */
    public long getDetectionTime() {
        return detectionTime;
    }
    
    /**
     * Get severity level
     * @return Severity level (1-5)
     */
    public int getSeverity() {
        return severity;
    }
    
    /**
     * Get source ID
     * @return Source ID
     */
    public String getSourceId() {
        return sourceId;
    }
    
    /**
     * Get destination ID
     * @return Destination ID
     */
    public String getDestinationId() {
        return destinationId;
    }
    
    /**
     * Check if incident has been mitigated
     * @return True if mitigated, false otherwise
     */
    public boolean isMitigated() {
        return mitigated;
    }
    
    /**
     * Set mitigation status
     * @param mitigated True if mitigated, false otherwise
     */
    public void setMitigated(boolean mitigated) {
        this.mitigated = mitigated;
    }
    
    /**
     * Get mitigation time
     * @return Mitigation time
     */
    public long getMitigationTime() {
        return mitigationTime;
    }
    
    /**
     * Set mitigation time
     * @param mitigationTime Mitigation time
     */
    public void setMitigationTime(long mitigationTime) {
        this.mitigationTime = mitigationTime;
    }
    
    /**
     * Get mitigation response time in milliseconds
     * @return Mitigation response time
     */
    public long getMitigationResponseTime() {
        if (!mitigated || mitigationTime == 0) {
            return 0;
        }
        return mitigationTime - detectionTime;
    }
    
    /**
     * Get severity as a string
     * @return Severity string
     */
    public String getSeverityString() {
        switch (severity) {
            case 1:
                return "Low";
            case 2:
                return "Medium-Low";
            case 3:
                return "Medium";
            case 4:
                return "Medium-High";
            case 5:
                return "High";
            default:
                return "Unknown";
        }
    }
    
    @Override
    public String toString() {
        return "SecurityIncident{" +
                "type='" + type + '\'' +
                ", severity=" + getSeverityString() +
                ", source='" + sourceId + '\'' +
                ", destination='" + destinationId + '\'' +
                ", mitigated=" + mitigated +
                ", responseTime=" + (mitigated ? getMitigationResponseTime() + "ms" : "N/A") +
                '}';
    }
}
