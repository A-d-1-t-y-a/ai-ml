package org.nci.fogedge.model;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Class to hold the results of the fog computing simulation
 * 
 * This class captures all the metrics and statistics generated during
 * the simulation for later analysis and reporting.
 * 
 * @author NCI H9FEC Student
 * @version 1.0
 */
public class SimulationResults {
    
    private long totalPacketsGenerated;
    private long packetsProcessedAtEdge;
    private long packetsProcessedAtFog;
    private long packetsProcessedAtCloud;
    private long packetsProcessedLocally;
    private int securityIncidentsDetected;
    private int securityIncidentsMitigated;
    private int securityIncidentsUnmitigated;
    private long packetsTransmittedToEdge;
    private long packetsTransmittedToFog;
    private long packetsTransmittedToCloud;
    private double averageLatency; // in milliseconds
    private double bandwidthSaved; // in MB
    private double energyConsumption; // in kWh
    
    // Additional metrics for detailed reports
    private double minEndToEndLatency = Double.MAX_VALUE;
    private double maxEndToEndLatency = 0.0;
    private double latencyStandardDeviation = 0.0;
    private double averageIoTProcessingTime = 0.0;
    private double averageEdgeProcessingTime = 0.0;
    private double averageFogProcessingTime = 0.0;
    private double averageCloudProcessingTime = 0.0;
    private double averageIoTToEdgeTransmissionTime = 0.0;
    private double averageEdgeToFogTransmissionTime = 0.0;
    private double averageFogToCloudTransmissionTime = 0.0;
    private double averageEncryptionOverhead = 0.0;
    private double averageAuthenticationOverhead = 0.0;
    private double averageIntrusionDetectionOverhead = 0.0;
    private double averageBlockchainOverhead = 0.0;
    private double averageDecoyTechniqueOverhead = 0.0;
    private double totalEnergyConsumed = 0.0;
    private double energyConsumedByIoT = 0.0;
    private double energyConsumedByEdge = 0.0;
    private double energyConsumedByFog = 0.0;
    private double energyConsumedByCloud = 0.0;
    private double energySavedByEdgeProcessing = 0.0;
    private double energySavedByFogProcessing = 0.0;
    private double totalEnergySaved = 0.0;
    private double energyConsumedByProcessing = 0.0;
    private double energyConsumedByCommunication = 0.0;
    private double energyConsumedBySecurity = 0.0;
    private double energyEfficiencyRatio = 0.0;
    private double energyPerPacket = 0.0;
    private double energyPerBit = 0.0;
    private double bandwidthSavedByEdgeProcessing = 0.0;
    private double bandwidthSavedByFogProcessing = 0.0;
    private double totalBandwidthSaved = 0.0;
    private double averageEndToEndLatency = 0.0;
    private double averageSecurityOverheadIoT = 0.0;
    private double averageSecurityOverheadEdge = 0.0;
    private double averageSecurityOverheadFog = 0.0;
    private double totalSecurityOverhead = 0.0;
    private double mitigationSuccessRate = 0.0;
    private double encryptionEffectiveness = 0.0;
    private double intrusionDetectionEffectiveness = 0.0;
    private double blockchainEffectiveness = 0.0;
    private double decoyTechniqueEffectiveness = 0.0;
    private double averageIncidentDetectionTime = 0.0;
    private double averageIncidentMitigationTime = 0.0;
    private double averageSecurityResponseTime = 0.0;
    private double ioTLayerEnergyConsumption = 0.0;
    private double edgeLayerEnergyConsumption = 0.0;
    private double fogLayerEnergyConsumption = 0.0;
    private double cloudLayerEnergyConsumption = 0.0;
    private double totalEnergyConsumption = 0.0;
    private double processingEnergyConsumption = 0.0;
    private double transmissionEnergyConsumption = 0.0;
    private double storageEnergyConsumption = 0.0;
    private double securityEnergyConsumption = 0.0;
    
    // Detailed metrics
    private List<Double> latencyValues = new ArrayList<>();
    private Map<String, Integer> securityIncidentsByType = new HashMap<>();
    private Map<String, Double> energyConsumptionByLayer = new HashMap<>();
    private List<Double> processingTimeByLayer = new ArrayList<>();
    
    /**
     * Default constructor
     */
    public SimulationResults() {
        // Initialize maps
        securityIncidentsByType.put("DoS", 0);
        securityIncidentsByType.put("DDoS", 0);
        securityIncidentsByType.put("ManInTheMiddle", 0);
        securityIncidentsByType.put("DataTampering", 0);
        securityIncidentsByType.put("Eavesdropping", 0);
        
        energyConsumptionByLayer.put("IoT", 0.0);
        energyConsumptionByLayer.put("Edge", 0.0);
        energyConsumptionByLayer.put("Fog", 0.0);
        energyConsumptionByLayer.put("Cloud", 0.0);
    }
    
    /**
     * Records a latency value for a processed packet
     * @param latency Latency value in milliseconds
     */
    public void recordLatency(double latency) {
        latencyValues.add(latency);
        // Recalculate average
        double sum = 0;
        for (Double value : latencyValues) {
            sum += value;
        }
        this.averageLatency = sum / latencyValues.size();
    }
    
    /**
     * Records a security incident
     * @param incidentType Type of security incident
     * @param mitigated Whether the incident was successfully mitigated
     */
    public void recordSecurityIncident(String incidentType, boolean mitigated) {
        securityIncidentsDetected++;
        
        if (mitigated) {
            securityIncidentsMitigated++;
        } else {
            securityIncidentsUnmitigated++;
        }
        
        // Record incident by type
        securityIncidentsByType.put(incidentType, 
            securityIncidentsByType.getOrDefault(incidentType, 0) + 1);
    }
    
    /**
     * Increment the total packets generated counter
     */
    public void incrementTotalPacketsGenerated() {
        totalPacketsGenerated++;
    }
    
    /**
     * Increment the packets transmitted to edge counter
     */
    public void incrementPacketsTransmittedToEdge() {
        packetsTransmittedToEdge++;
    }
    
    /**
     * Increment the packets processed at edge counter
     */
    public void incrementPacketsProcessedAtEdge() {
        packetsProcessedAtEdge++;
    }
    
    /**
     * Increment the packets transmitted to fog counter
     */
    public void incrementPacketsTransmittedToFog() {
        packetsTransmittedToFog++;
    }
    
    /**
     * Increment the packets processed at fog counter
     */
    public void incrementPacketsProcessedAtFog() {
        packetsProcessedAtFog++;
    }
    
    /**
     * Increment the packets transmitted to cloud counter
     */
    public void incrementPacketsTransmittedToCloud() {
        packetsTransmittedToCloud++;
    }
    
    /**
     * Increment the packets processed at cloud counter
     */
    public void incrementPacketsProcessedAtCloud() {
        packetsProcessedAtCloud++;
    }
    
    /**
     * Increment the packets processed locally counter
     */
    public void incrementPacketsProcessedLocally() {
        packetsProcessedLocally++;
    }
    
    /**
     * Increment the security incidents detected counter
     */
    public void incrementSecurityIncidentsDetected() {
        securityIncidentsDetected++;
    }
    
    /**
     * Increment the security incidents mitigated counter
     */
    public void incrementSecurityIncidentsMitigated() {
        securityIncidentsMitigated++;
    }
    
    /**
     * Increment the security incidents unmitigated counter
     */
    public void incrementSecurityIncidentsUnmitigated() {
        securityIncidentsUnmitigated++;
    }
    
    /**
     * Calculate derived metrics based on raw data
     */
    public void calculateDerivedMetrics() {
        // Calculate average latency if we have values
        if (!latencyValues.isEmpty()) {
            double sum = 0.0;
            for (Double value : latencyValues) {
                sum += value;
            }
            averageLatency = sum / latencyValues.size();
        }
        
        // Calculate energy metrics
        totalEnergyConsumed = energyConsumedByIoT + energyConsumedByEdge + 
                             energyConsumedByFog + energyConsumedByCloud;
        
        // Calculate bandwidth saved
        bandwidthSaved = bandwidthSavedByEdgeProcessing + bandwidthSavedByFogProcessing;
        
        // Calculate security metrics
        if (securityIncidentsDetected > 0) {
            mitigationSuccessRate = (double) securityIncidentsMitigated / securityIncidentsDetected;
        }
        
        // Calculate energy efficiency
        if (totalPacketsGenerated > 0) {
            energyPerPacket = totalEnergyConsumed / totalPacketsGenerated;
        }
        
        // Calculate other derived metrics as needed
        totalSecurityOverhead = averageSecurityOverheadIoT + averageSecurityOverheadEdge + 
                               averageSecurityOverheadFog;
    }
    
    /**
     * Records energy consumption for a specific layer
     * @param layer Layer name (IoT, Edge, Fog, Cloud)
     * @param energy Energy consumed in kWh
     */
    public void recordEnergyConsumption(String layer, double energy) {
        if (energyConsumptionByLayer.containsKey(layer)) {
            energyConsumptionByLayer.put(layer, energyConsumptionByLayer.get(layer) + energy);
        } else {
            energyConsumptionByLayer.put(layer, energy);
        }
        
        // Update total energy consumption
        this.energyConsumption += energy;
    }
    
    // Getters and setters
    
    public long getTotalPacketsGenerated() {
        return totalPacketsGenerated;
    }
    
    public void setTotalPacketsGenerated(long totalPacketsGenerated) {
        this.totalPacketsGenerated = totalPacketsGenerated;
    }
    
    public long getPacketsProcessedAtEdge() {
        return packetsProcessedAtEdge;
    }
    
    public void setPacketsProcessedAtEdge(long packetsProcessedAtEdge) {
        this.packetsProcessedAtEdge = packetsProcessedAtEdge;
    }
    
    public long getPacketsProcessedAtFog() {
        return packetsProcessedAtFog;
    }
    
    public void setPacketsProcessedAtFog(long packetsProcessedAtFog) {
        this.packetsProcessedAtFog = packetsProcessedAtFog;
    }
    
    public long getPacketsProcessedAtCloud() {
        return packetsProcessedAtCloud;
    }
    
    public void setPacketsProcessedAtCloud(long packetsProcessedAtCloud) {
        this.packetsProcessedAtCloud = packetsProcessedAtCloud;
    }
    
    public int getSecurityIncidentsDetected() {
        return securityIncidentsDetected;
    }
    
    public int getSecurityIncidentsMitigated() {
        return securityIncidentsMitigated;
    }
    
    public double getAverageLatency() {
        return averageLatency;
    }
    
    public double getBandwidthSaved() {
        return bandwidthSaved;
    }
    
    public void setBandwidthSaved(double bandwidthSaved) {
        this.bandwidthSaved = bandwidthSaved;
    }
    
    public double getEnergyConsumption() {
        return energyConsumption;
    }
    
    public List<Double> getLatencyValues() {
        return latencyValues;
    }
    
    public Map<String, Integer> getSecurityIncidentsByType() {
        return securityIncidentsByType;
    }
    
    public Map<String, Double> getEnergyConsumptionByLayer() {
        return energyConsumptionByLayer;
    }
    
    public List<Double> getProcessingTimeByLayer() {
        return processingTimeByLayer;
    }
    
    public void setProcessingTimeByLayer(List<Double> processingTimeByLayer) {
        this.processingTimeByLayer = processingTimeByLayer;
    }
    
    public double getTotalEnergyConsumed() {
        return totalEnergyConsumed;
    }
    
    public void setTotalEnergyConsumed(double totalEnergyConsumed) {
        this.totalEnergyConsumed = totalEnergyConsumed;
    }
    
    public double getEnergyConsumedByIoT() {
        return energyConsumedByIoT;
    }
    
    /**
     * Get the number of packets transmitted to edge
     * @return Number of packets transmitted to edge
     */
    public long getPacketsTransmittedToEdge() {
        return packetsTransmittedToEdge;
    }
    
    /**
     * Set the number of packets transmitted to edge
     * @param packetsTransmittedToEdge Number of packets transmitted to edge
     */
    public void setPacketsTransmittedToEdge(long packetsTransmittedToEdge) {
        this.packetsTransmittedToEdge = packetsTransmittedToEdge;
    }
    
    /**
     * Get the number of packets transmitted to fog
     * @return Number of packets transmitted to fog
     */
    public long getPacketsTransmittedToFog() {
        return packetsTransmittedToFog;
    }
    
    /**
     * Set the number of packets transmitted to fog
     * @param packetsTransmittedToFog Number of packets transmitted to fog
     */
    public void setPacketsTransmittedToFog(long packetsTransmittedToFog) {
        this.packetsTransmittedToFog = packetsTransmittedToFog;
    }
    
    /**
     * Get the number of packets transmitted to cloud
     * @return Number of packets transmitted to cloud
     */
    public long getPacketsTransmittedToCloud() {
        return packetsTransmittedToCloud;
    }
    
    /**
     * Set the number of packets transmitted to cloud
     * @param packetsTransmittedToCloud Number of packets transmitted to cloud
     */
    public void setPacketsTransmittedToCloud(long packetsTransmittedToCloud) {
        this.packetsTransmittedToCloud = packetsTransmittedToCloud;
    }
    
    /**
     * Get the number of packets processed locally
     * @return Number of packets processed locally
     */
    public long getPacketsProcessedLocally() {
        return packetsProcessedLocally;
    }
    
    /**
     * Set the number of packets processed locally
     * @param packetsProcessedLocally Number of packets processed locally
     */
    public void setPacketsProcessedLocally(long packetsProcessedLocally) {
        this.packetsProcessedLocally = packetsProcessedLocally;
    }
    
    public void setEnergyConsumedByIoT(double energyConsumedByIoT) {
        this.energyConsumedByIoT = energyConsumedByIoT;
    }
    
    public double getEnergyConsumedByEdgeNodes() {
        return energyConsumedByEdge;
    }
    
    public void setEnergyConsumedByEdgeNodes(double energyConsumedByEdge) {
        this.energyConsumedByEdge = energyConsumedByEdge;
    }
    
    public double getEnergyConsumedByFogNodes() {
        return energyConsumedByFog;
    }
    
    public void setEnergyConsumedByFogNodes(double energyConsumedByFog) {
        this.energyConsumedByFog = energyConsumedByFog;
    }
    
    public double getEnergyConsumedByCloud() {
        return energyConsumedByCloud;
    }
    
    public void setEnergyConsumedByCloud(double energyConsumedByCloud) {
        this.energyConsumedByCloud = energyConsumedByCloud;
    }
    
    public double getMinEndToEndLatency() {
        return minEndToEndLatency;
    }
    
    public void setMinEndToEndLatency(double minEndToEndLatency) {
        this.minEndToEndLatency = minEndToEndLatency;
    }
    
    public double getMaxEndToEndLatency() {
        return maxEndToEndLatency;
    }
    
    public void setMaxEndToEndLatency(double maxEndToEndLatency) {
        this.maxEndToEndLatency = maxEndToEndLatency;
    }
    
    public double getLatencyStandardDeviation() {
        return latencyStandardDeviation;
    }
    
    public void setLatencyStandardDeviation(double latencyStandardDeviation) {
        this.latencyStandardDeviation = latencyStandardDeviation;
    }
    
    public double getAverageIoTProcessingTime() {
        return averageIoTProcessingTime;
    }
    
    public void setAverageIoTProcessingTime(double averageIoTProcessingTime) {
        this.averageIoTProcessingTime = averageIoTProcessingTime;
    }
    
    public double getAverageEdgeProcessingTime() {
        return averageEdgeProcessingTime;
    }
    
    public void setAverageEdgeProcessingTime(double averageEdgeProcessingTime) {
        this.averageEdgeProcessingTime = averageEdgeProcessingTime;
    }
    
    public double getAverageFogProcessingTime() {
        return averageFogProcessingTime;
    }
    
    public void setAverageFogProcessingTime(double averageFogProcessingTime) {
        this.averageFogProcessingTime = averageFogProcessingTime;
    }
    
    public double getAverageCloudProcessingTime() {
        return averageCloudProcessingTime;
    }
    
    public void setAverageCloudProcessingTime(double averageCloudProcessingTime) {
        this.averageCloudProcessingTime = averageCloudProcessingTime;
    }
    
    public double getAverageIoTToEdgeTransmissionTime() {
        return averageIoTToEdgeTransmissionTime;
    }
    
    public void setAverageIoTToEdgeTransmissionTime(double averageIoTToEdgeTransmissionTime) {
        this.averageIoTToEdgeTransmissionTime = averageIoTToEdgeTransmissionTime;
    }
    
    public double getAverageEdgeToFogTransmissionTime() {
        return averageEdgeToFogTransmissionTime;
    }
    
    public void setAverageEdgeToFogTransmissionTime(double averageEdgeToFogTransmissionTime) {
        this.averageEdgeToFogTransmissionTime = averageEdgeToFogTransmissionTime;
    }
    
    public double getAverageFogToCloudTransmissionTime() {
        return averageFogToCloudTransmissionTime;
    }
    
    public void setAverageFogToCloudTransmissionTime(double averageFogToCloudTransmissionTime) {
        this.averageFogToCloudTransmissionTime = averageFogToCloudTransmissionTime;
    }
    
    public double getAverageEncryptionOverhead() {
        return averageEncryptionOverhead;
    }
    
    public void setAverageEncryptionOverhead(double averageEncryptionOverhead) {
        this.averageEncryptionOverhead = averageEncryptionOverhead;
    }
    
    public double getAverageAuthenticationOverhead() {
        return averageAuthenticationOverhead;
    }
    
    public void setAverageAuthenticationOverhead(double averageAuthenticationOverhead) {
        this.averageAuthenticationOverhead = averageAuthenticationOverhead;
    }
    
    public double getAverageIntrusionDetectionOverhead() {
        return averageIntrusionDetectionOverhead;
    }
    
    public void setAverageIntrusionDetectionOverhead(double averageIntrusionDetectionOverhead) {
        this.averageIntrusionDetectionOverhead = averageIntrusionDetectionOverhead;
    }
    
    public double getAverageBlockchainOverhead() {
        return averageBlockchainOverhead;
    }
    
    public void setAverageBlockchainOverhead(double averageBlockchainOverhead) {
        this.averageBlockchainOverhead = averageBlockchainOverhead;
    }
    
    public double getAverageDecoyTechniqueOverhead() {
        return averageDecoyTechniqueOverhead;
    }
    
    public void setAverageDecoyTechniqueOverhead(double averageDecoyTechniqueOverhead) {
        this.averageDecoyTechniqueOverhead = averageDecoyTechniqueOverhead;
    }
    
    // These methods are already defined elsewhere in the class
    // Removed duplicate declarations
    
    public int getSecurityIncidentsUnmitigated() {
        return securityIncidentsUnmitigated;
    }
    
    public void setSecurityIncidentsUnmitigated(int securityIncidentsUnmitigated) {
        this.securityIncidentsUnmitigated = securityIncidentsUnmitigated;
    }
    
    public double getMitigationSuccessRate() {
        if (securityIncidentsDetected == 0) {
            return 0.0;
        }
        return (double) securityIncidentsMitigated / securityIncidentsDetected;
    }
    
    public double getEncryptionEffectiveness() {
        return encryptionEffectiveness;
    }
    
    public void setEncryptionEffectiveness(double encryptionEffectiveness) {
        this.encryptionEffectiveness = encryptionEffectiveness;
    }
    
    public double getIntrusionDetectionEffectiveness() {
        return intrusionDetectionEffectiveness;
    }
    
    public void setIntrusionDetectionEffectiveness(double intrusionDetectionEffectiveness) {
        this.intrusionDetectionEffectiveness = intrusionDetectionEffectiveness;
    }
    
    public double getBlockchainEffectiveness() {
        return blockchainEffectiveness;
    }
    
    public void setBlockchainEffectiveness(double blockchainEffectiveness) {
        this.blockchainEffectiveness = blockchainEffectiveness;
    }
    
    public double getDecoyTechniqueEffectiveness() {
        return decoyTechniqueEffectiveness;
    }
    
    public void setDecoyTechniqueEffectiveness(double decoyTechniqueEffectiveness) {
        this.decoyTechniqueEffectiveness = decoyTechniqueEffectiveness;
    }
    
    public double getAverageIncidentDetectionTime() {
        return averageIncidentDetectionTime;
    }
    
    public void setAverageIncidentDetectionTime(double averageIncidentDetectionTime) {
        this.averageIncidentDetectionTime = averageIncidentDetectionTime;
    }
    
    public double getAverageIncidentMitigationTime() {
        return averageIncidentMitigationTime;
    }
    
    public void setAverageIncidentMitigationTime(double averageIncidentMitigationTime) {
        this.averageIncidentMitigationTime = averageIncidentMitigationTime;
    }
    
    public double getAverageSecurityResponseTime() {
        return averageSecurityResponseTime;
    }
    
    public void setAverageSecurityResponseTime(double averageSecurityResponseTime) {
        this.averageSecurityResponseTime = averageSecurityResponseTime;
    }
    
    public double getAverageEndToEndLatency() {
        return averageEndToEndLatency;
    }
    
    public void setAverageEndToEndLatency(double averageEndToEndLatency) {
        this.averageEndToEndLatency = averageEndToEndLatency;
    }
    
    public double getAverageSecurityOverheadIoT() {
        return averageSecurityOverheadIoT;
    }
    
    public void setAverageSecurityOverheadIoT(double averageSecurityOverheadIoT) {
        this.averageSecurityOverheadIoT = averageSecurityOverheadIoT;
    }
    
    public double getAverageSecurityOverheadEdge() {
        return averageSecurityOverheadEdge;
    }
    
    public void setAverageSecurityOverheadEdge(double averageSecurityOverheadEdge) {
        this.averageSecurityOverheadEdge = averageSecurityOverheadEdge;
    }
    
    public double getAverageSecurityOverheadFog() {
        return averageSecurityOverheadFog;
    }
    
    public void setAverageSecurityOverheadFog(double averageSecurityOverheadFog) {
        this.averageSecurityOverheadFog = averageSecurityOverheadFog;
    }
    
    public double getTotalSecurityOverhead() {
        return totalSecurityOverhead;
    }
    
    public void setTotalSecurityOverhead(double totalSecurityOverhead) {
        this.totalSecurityOverhead = totalSecurityOverhead;
    }
    
    public double getBandwidthSavedByEdgeProcessing() {
        return bandwidthSavedByEdgeProcessing;
    }
    
    public void setBandwidthSavedByEdgeProcessing(double bandwidthSavedByEdgeProcessing) {
        this.bandwidthSavedByEdgeProcessing = bandwidthSavedByEdgeProcessing;
    }
    
    public double getBandwidthSavedByFogProcessing() {
        return bandwidthSavedByFogProcessing;
    }
    
    public void setBandwidthSavedByFogProcessing(double bandwidthSavedByFogProcessing) {
        this.bandwidthSavedByFogProcessing = bandwidthSavedByFogProcessing;
    }
    
    public double getTotalBandwidthSaved() {
        return totalBandwidthSaved;
    }
    
    public void setTotalBandwidthSaved(double totalBandwidthSaved) {
        this.totalBandwidthSaved = totalBandwidthSaved;
    }
    
    public double getEnergyPerPacket() {
        return energyPerPacket;
    }
    
    public void setEnergyPerPacket(double energyPerPacket) {
        this.energyPerPacket = energyPerPacket;
    }
    
    public double getEnergyPerBit() {
        return energyPerBit;
    }
    
    public void setEnergyPerBit(double energyPerBit) {
        this.energyPerBit = energyPerBit;
    }
    
    public double getIoTLayerEnergyConsumption() {
        return ioTLayerEnergyConsumption;
    }
    
    public void setIoTLayerEnergyConsumption(double ioTLayerEnergyConsumption) {
        this.ioTLayerEnergyConsumption = ioTLayerEnergyConsumption;
    }
    
    public double getEdgeLayerEnergyConsumption() {
        return edgeLayerEnergyConsumption;
    }
    
    public void setEdgeLayerEnergyConsumption(double edgeLayerEnergyConsumption) {
        this.edgeLayerEnergyConsumption = edgeLayerEnergyConsumption;
    }
    
    public double getFogLayerEnergyConsumption() {
        return fogLayerEnergyConsumption;
    }
    
    public void setFogLayerEnergyConsumption(double fogLayerEnergyConsumption) {
        this.fogLayerEnergyConsumption = fogLayerEnergyConsumption;
    }
    
    public double getCloudLayerEnergyConsumption() {
        return cloudLayerEnergyConsumption;
    }
    
    public void setCloudLayerEnergyConsumption(double cloudLayerEnergyConsumption) {
        this.cloudLayerEnergyConsumption = cloudLayerEnergyConsumption;
    }
    
    public double getTotalEnergyConsumption() {
        return totalEnergyConsumption;
    }
    
    public void setTotalEnergyConsumption(double totalEnergyConsumption) {
        this.totalEnergyConsumption = totalEnergyConsumption;
    }
    
    public double getProcessingEnergyConsumption() {
        return processingEnergyConsumption;
    }
    
    public void setProcessingEnergyConsumption(double processingEnergyConsumption) {
        this.processingEnergyConsumption = processingEnergyConsumption;
    }
    
    public double getTransmissionEnergyConsumption() {
        return transmissionEnergyConsumption;
    }
    
    public void setTransmissionEnergyConsumption(double transmissionEnergyConsumption) {
        this.transmissionEnergyConsumption = transmissionEnergyConsumption;
    }
    
    public double getStorageEnergyConsumption() {
        return storageEnergyConsumption;
    }
    
    public void setStorageEnergyConsumption(double storageEnergyConsumption) {
        this.storageEnergyConsumption = storageEnergyConsumption;
    }
    
    public double getSecurityEnergyConsumption() {
        return securityEnergyConsumption;
    }
    
    public void setSecurityEnergyConsumption(double securityEnergyConsumption) {
        this.securityEnergyConsumption = securityEnergyConsumption;
    }
}
