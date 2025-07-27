package org.nci.fogedge.model;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

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
    private Map<String, Integer> securityIncidentsMitigatedByType = new HashMap<>();
    
    // Per-device metrics
    private Map<String, Integer> packetsGeneratedByDevice = new HashMap<>();
    private Map<String, Double> energyConsumedByDevice = new HashMap<>();
    private Map<String, List<Double>> latenciesByDevice = new HashMap<>();
    
    // Per-node metrics
    private Map<String, Integer> packetsProcessedByEdgeNode = new HashMap<>();
    private Map<String, Integer> packetsProcessedByFogNode = new HashMap<>();
    private Map<String, Double> energyConsumedByEdgeNode = new HashMap<>();
    private Map<String, Double> energyConsumedByFogNode = new HashMap<>();
    
    // Offloading metrics
    private Map<String, List<OffloadingRecord>> offloadingRecords = new HashMap<>();
    
    // Security metrics
    private Map<String, List<SecurityIncidentRecord>> securityIncidentRecords = new HashMap<>();
    
    // Packet completion metrics
    private Map<String, PacketCompletionRecord> packetCompletionRecords = new HashMap<>();
    
    // Inner classes for detailed records
    private class OffloadingRecord {
        String sourceId;
        String destinationId;
        String dataId;
        int dataSize;
        double timestamp;
        String offloadingType;
        
        public OffloadingRecord(String sourceId, String destinationId, String dataId, int dataSize, double timestamp, String offloadingType) {
            this.sourceId = sourceId;
            this.destinationId = destinationId;
            this.dataId = dataId;
            this.dataSize = dataSize;
            this.timestamp = timestamp;
            this.offloadingType = offloadingType;
        }
    }
    
    private class SecurityIncidentRecord {
        String incidentType;
        String sourceId;
        String destinationId;
        String dataId;
        double timestamp;
        boolean mitigated;
        double mitigationTime;
        double energyOverhead;
        
        public SecurityIncidentRecord(String incidentType, String sourceId, String destinationId, String dataId, double timestamp) {
            this.incidentType = incidentType;
            this.sourceId = sourceId;
            this.destinationId = destinationId;
            this.dataId = dataId;
            this.timestamp = timestamp;
            this.mitigated = false;
            this.mitigationTime = 0.0;
            this.energyOverhead = 0.0;
        }
    }
    
    private class PacketCompletionRecord {
        String dataId;
        String sourceDeviceId;
        double latency;
        double completionTime;
        
        public PacketCompletionRecord(String dataId, String sourceDeviceId, double latency, double completionTime) {
            this.dataId = dataId;
            this.sourceDeviceId = sourceDeviceId;
            this.latency = latency;
            this.completionTime = completionTime;
        }
    }
    
    // Packet processing logs
    private List<Map<String, Object>> packetProcessingLogs = new ArrayList<>();
    private List<Double> processingTimeByLayer = new ArrayList<>();
    
    // Per-device metrics for additional tracking
    private Map<String, Double> processingTimeByDevice = new HashMap<>();
    private Map<String, Integer> packetsProcessedByDevice = new HashMap<>();
    
    // Offloading logs
    private List<Map<String, Object>> dataOffloadingLogs = new ArrayList<>();
    
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
            double sumSquared = 0.0;
            minEndToEndLatency = Double.MAX_VALUE;
            maxEndToEndLatency = 0.0;
            
            for (Double value : latencyValues) {
                sum += value;
                sumSquared += value * value;
                minEndToEndLatency = Math.min(minEndToEndLatency, value);
                maxEndToEndLatency = Math.max(maxEndToEndLatency, value);
            }
            
            averageLatency = sum / latencyValues.size();
            averageEndToEndLatency = averageLatency;
            
            // Calculate standard deviation
            double mean = sum / latencyValues.size();
            double variance = (sumSquared / latencyValues.size()) - (mean * mean);
            latencyStandardDeviation = Math.sqrt(variance);
        }
        
        // Calculate energy metrics
        totalEnergyConsumed = energyConsumedByIoT + energyConsumedByEdge + 
                             energyConsumedByFog + energyConsumedByCloud;
        
        // Set layer energy consumption for reporting
        ioTLayerEnergyConsumption = energyConsumedByIoT;
        edgeLayerEnergyConsumption = energyConsumedByEdge;
        fogLayerEnergyConsumption = energyConsumedByFog;
        cloudLayerEnergyConsumption = energyConsumedByCloud;
        totalEnergyConsumption = totalEnergyConsumed;
        
        // Calculate bandwidth saved
        bandwidthSaved = bandwidthSavedByEdgeProcessing + bandwidthSavedByFogProcessing;
        totalBandwidthSaved = bandwidthSaved;
        
        // Calculate security metrics
        if (securityIncidentsDetected > 0) {
            mitigationSuccessRate = (double) securityIncidentsMitigated / securityIncidentsDetected;
        }
        
        // Calculate energy efficiency
        if (totalPacketsGenerated > 0) {
            energyPerPacket = totalEnergyConsumed / totalPacketsGenerated;
            // Assuming average packet size of 1KB = 8192 bits
            energyPerBit = energyPerPacket / 8192.0;
        }
        
        // Calculate security overhead metrics
        totalSecurityOverhead = 0.0;
        for (Double overhead : securityOverheadByLayer.values()) {
            totalSecurityOverhead += overhead;
        }
        
        // Calculate average security overhead by layer
        if (securityOverheadByLayer.containsKey("IoT")) {
            averageSecurityOverheadIoT = securityOverheadByLayer.get("IoT");
        }
        if (securityOverheadByLayer.containsKey("Edge")) {
            averageSecurityOverheadEdge = securityOverheadByLayer.get("Edge");
        }
        if (securityOverheadByLayer.containsKey("Fog")) {
            averageSecurityOverheadFog = securityOverheadByLayer.get("Fog");
        }
        
        // Calculate energy consumption breakdown
        processingEnergyConsumption = totalEnergyConsumed * 0.7; // Assuming 70% for processing
        transmissionEnergyConsumption = totalEnergyConsumed * 0.2; // Assuming 20% for transmission
        storageEnergyConsumption = totalEnergyConsumed * 0.05; // Assuming 5% for storage
        securityEnergyConsumption = totalEnergyConsumed * 0.05; // Assuming 5% for security
        
        // Calculate energy saved metrics
        energySavedByEdgeProcessing = packetsProcessedAtEdge * 0.001; // 1mWh saved per packet processed at edge
        energySavedByFogProcessing = packetsProcessedAtFog * 0.0005; // 0.5mWh saved per packet processed at fog
        totalEnergySaved = energySavedByEdgeProcessing + energySavedByFogProcessing;
        
        // Calculate energy efficiency ratio
        if (totalEnergyConsumed > 0) {
            energyEfficiencyRatio = totalEnergySaved / totalEnergyConsumed;
        }
    }
    
    /**
     * Records energy consumption for a specific layer
     * @param layer Layer name (IoT, Edge, Fog, Cloud)
     * @param energy Energy consumed in kWh
     */
    public void recordEnergyConsumption(String layer, double energy) {
        if (layer.equals("IoT")) {
            energyConsumedByIoT += energy;
        } else if (layer.equals("Edge")) {
            energyConsumedByEdge += energy;
        } else if (layer.equals("Fog")) {
            energyConsumedByFog += energy;
        } else if (layer.equals("Cloud")) {
            energyConsumedByCloud += energy;
        }
        
        // Update energy consumption by layer map
        Double currentEnergy = energyConsumptionByLayer.getOrDefault(layer, 0.0);
        energyConsumptionByLayer.put(layer, currentEnergy + energy);
    }
    
    /**
     * Records energy consumption for a specific device
     * @param deviceId Device ID
     * @param energy Energy consumed in kWh
     */
    public void recordEnergyConsumptionByDevice(String deviceId, double energy) {
        Double currentEnergy = energyConsumedByDevice.getOrDefault(deviceId, 0.0);
        energyConsumedByDevice.put(deviceId, currentEnergy + energy);
    }
    
    /**
     * Records security overhead by type
     * @param type Security measure type (encryption, authentication, etc.)
     * @param overhead Overhead in milliseconds
     */
    public void recordSecurityOverhead(String type, double overhead) {
        Double currentOverhead = securityOverheadByType.getOrDefault(type, 0.0);
        securityOverheadByType.put(type, currentOverhead + overhead);
        
        // Update specific overhead metrics
        if (type.equals("encryption")) {
            averageEncryptionOverhead += overhead;
        } else if (type.equals("authentication")) {
            averageAuthenticationOverhead += overhead;
        } else if (type.equals("intrusion_detection")) {
            averageIntrusionDetectionOverhead += overhead;
        } else if (type.equals("blockchain")) {
            averageBlockchainOverhead += overhead;
        } else if (type.equals("decoy")) {
            averageDecoyTechniqueOverhead += overhead;
        }
    }
    
    /**
     * Records security overhead by layer
     * @param layer Layer name (IoT, Edge, Fog)
     * @param overhead Overhead in milliseconds
     */
    public void recordSecurityOverheadByLayer(String layer, double overhead) {
        Double currentOverhead = securityOverheadByLayer.getOrDefault(layer, 0.0);
        securityOverheadByLayer.put(layer, currentOverhead + overhead);
    }
    
    /**
     * Records a packet generation event
     * @param deviceId Source device ID
     * @param packetId Packet ID
     * @param size Packet size in bytes
     * @param timestamp Generation timestamp
     */
    public void recordPacketGeneration(String deviceId, String packetId, int size, double timestamp) {
        // Increment packets generated by device
        Integer currentCount = packetsGeneratedByDevice.getOrDefault(deviceId, 0);
        packetsGeneratedByDevice.put(deviceId, currentCount + 1);
        
        // Record packet processing log
        Map<String, Object> log = new HashMap<>();
        log.put("packetId", packetId);
        log.put("sourceId", deviceId);
        log.put("size", size);
        log.put("generationTime", timestamp);
        log.put("status", "generated");
        packetProcessingLogs.add(log);
    }
    
    /**
     * Records data offloading from one node to another
     * @param sourceId Source node ID
     * @param destinationId Destination node ID
     * @param packetId Packet ID
     * @param size Packet size in bytes
     * @param timestamp Offloading timestamp
     * @param type Offloading type (IoT-to-Edge, Edge-to-Fog, etc.)
     */
    public void recordDataOffloading(String sourceId, String destinationId, String packetId, int size, double timestamp, String type) {
        Map<String, Object> log = new HashMap<>();
        log.put("packetId", packetId);
        log.put("sourceId", sourceId);
        log.put("destinationId", destinationId);
        log.put("size", size);
        log.put("timestamp", timestamp);
        log.put("type", type);
        dataOffloadingLogs.add(log);
    }
    
    /**
     * Increments transmission energy consumption
     * @param energy Energy consumed in kWh
     */
    public void incrementTransmissionEnergyConsumption(double energy) {
        transmissionEnergyConsumption += energy;
    }
    
    /**
     * Get energy consumed by a specific device
     * @param deviceId Device ID
     * @return Energy consumed in kWh
     */
    public double getDeviceEnergyConsumption(String deviceId) {
        return energyConsumedByDevice.getOrDefault(deviceId, 0.0);
    }
    
    /**
     * Get packets generated by a specific device
     * @param deviceId Device ID
     * @return Number of packets generated
     */
    public int getPacketsGeneratedByDevice(String deviceId) {
        return packetsGeneratedByDevice.getOrDefault(deviceId, 0);
    }
    
    /**
     * Get packets processed by a specific device
     * @param deviceId Device ID
     * @return Number of packets processed
     */
    public int getPacketsProcessedByDevice(String deviceId) {
        return packetsProcessedByDevice.getOrDefault(deviceId, 0);
    }
    
    /**
     * Get all device IDs that have generated packets
     * @return Set of device IDs
     */
    public Set<String> getAllDeviceIds() {
        return packetsGeneratedByDevice.keySet();
    }
    
    /**
     * Get all data offloading logs
     * @return List of offloading logs
     */
    public List<Map<String, Object>> getDataOffloadingLogs() {
        return dataOffloadingLogs;
    }
    
    /**
     * Get security overhead by type
     * @param type Security measure type
     * @return Overhead in milliseconds
     */
    public double getSecurityOverheadByType(String type) {
        return securityOverheadByType.getOrDefault(type, 0.0);
    }
    
    /**
     * Get security overhead by layer
     * @param layer Layer name
     * @return Overhead in milliseconds
     */
    public double getSecurityOverheadByLayer(String layer) {
        return securityOverheadByLayer.getOrDefault(layer, 0.0);
    }
    
    /**
     * Get all packet processing logs
     * @return List of packet processing logs
     */
    public List<Map<String, Object>> getPacketProcessingLogs() {
        return packetProcessingLogs;
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
    
    /**
     * Record a security incident with detailed information
     * @param incidentType Type of security incident
     * @param sourceId Source ID where incident originated
     * @param destinationId Destination ID where incident was detected
     * @param dataId Data packet ID involved in the incident
     * @param timestamp Time when the incident was detected
     */
    public void recordSecurityIncident(String incidentType, String sourceId, String destinationId, String dataId, double timestamp) {
        // Increment the count for this incident type
        securityIncidentsByType.put(incidentType, securityIncidentsByType.getOrDefault(incidentType, 0) + 1);
        
        // Create a new security incident record
        SecurityIncidentRecord record = new SecurityIncidentRecord(incidentType, sourceId, destinationId, dataId, timestamp);
        
        // Add to the records map
        List<SecurityIncidentRecord> records = securityIncidentRecords.getOrDefault(incidentType, new ArrayList<>());
        records.add(record);
        securityIncidentRecords.put(incidentType, records);
    }
    
    /**
     * Record security incident handling details
     * @param sourceId Source ID where incident originated
     * @param destinationId Destination ID where incident was detected
     * @param dataId Data packet ID involved in the incident
     * @param mitigated Whether the incident was successfully mitigated
     * @param mitigationTime Time taken to mitigate the incident
     * @param energyOverhead Energy consumed for mitigation
     * @param timestamp Time when mitigation was completed
     */
    public void recordSecurityIncidentHandling(String sourceId, String destinationId, String dataId, boolean mitigated, 
                                             double mitigationTime, double energyOverhead, double timestamp) {
        // Find the incident record
        for (List<SecurityIncidentRecord> records : securityIncidentRecords.values()) {
            for (SecurityIncidentRecord record : records) {
                if (record.dataId.equals(dataId) && record.destinationId.equals(destinationId)) {
                    record.mitigated = mitigated;
                    record.mitigationTime = mitigationTime;
                    record.energyOverhead = energyOverhead;
                    break;
                }
            }
        }
        
        // Record energy consumption for security
        recordEnergyConsumption("Security", energyOverhead);
    }
    
    /**
     * Record offloading of a data packet
     * @param sourceId Source ID
     * @param destinationId Destination ID
     * @param dataId Data packet ID
     * @param dataSize Size of the data packet
     * @param timestamp Time of offloading
     * @param offloadingType Type of offloading (e.g., "IoT-to-Edge", "Edge-to-Fog", etc.)
     */
    public void recordOffloading(String sourceId, String destinationId, String dataId, int dataSize, double timestamp, String offloadingType) {
        OffloadingRecord record = new OffloadingRecord(sourceId, destinationId, dataId, dataSize, timestamp, offloadingType);
        
        List<OffloadingRecord> records = offloadingRecords.getOrDefault(offloadingType, new ArrayList<>());
        records.add(record);
        offloadingRecords.put(offloadingType, records);
    }
    
    /**
     * Record packet completion with latency
     * @param dataId Data packet ID
     * @param sourceDeviceId Source device ID
     * @param latency End-to-end latency
     * @param completionTime Time of completion
     */
    public void recordPacketCompletion(String dataId, String sourceDeviceId, double latency, double completionTime) {
        PacketCompletionRecord record = new PacketCompletionRecord(dataId, sourceDeviceId, latency, completionTime);
        packetCompletionRecords.put(dataId, record);
        
        // Also record the latency for the device
        List<Double> deviceLatencies = latenciesByDevice.getOrDefault(sourceDeviceId, new ArrayList<>());
        deviceLatencies.add(latency);
        latenciesByDevice.put(sourceDeviceId, deviceLatencies);
    }
    
    /**
     * Increment security overhead
     * @param overhead Overhead in milliseconds
     */
    public void incrementSecurityOverhead(double overhead) {
        totalSecurityOverhead += overhead;
    }
    
    /**
     * Calculate device-specific metrics
     * @param deviceId Device ID
     */
    public void calculateDeviceMetrics(String deviceId) {
        // Calculate average latency for this device
        List<Double> deviceLatencies = latenciesByDevice.getOrDefault(deviceId, new ArrayList<>());
        if (!deviceLatencies.isEmpty()) {
            double sum = 0.0;
            for (Double latency : deviceLatencies) {
                sum += latency;
            }
            double avgLatency = sum / deviceLatencies.size();
            System.out.println("Device " + deviceId + " average latency: " + String.format("%.3f", avgLatency) + " ms");
        }
        
        // Calculate energy consumption for this device
        double energy = energyConsumedByDevice.getOrDefault(deviceId, 0.0);
        System.out.println("Device " + deviceId + " energy consumption: " + String.format("%.6f", energy) + " mWh");
        
        // Calculate packets generated by this device
        int packets = packetsGeneratedByDevice.getOrDefault(deviceId, 0);
        System.out.println("Device " + deviceId + " packets generated: " + packets);
    }
    
    /**
     * Calculate edge node specific metrics
     * @param edgeId Edge node ID
     */
    public void calculateEdgeNodeMetrics(String edgeId) {
        // Calculate packets processed by this edge node
        int packets = packetsProcessedByEdgeNode.getOrDefault(edgeId, 0);
        System.out.println("Edge node " + edgeId + " packets processed: " + packets);
        
        // Calculate energy consumption for this edge node
        double energy = energyConsumedByEdgeNode.getOrDefault(edgeId, 0.0);
        System.out.println("Edge node " + edgeId + " energy consumption: " + String.format("%.6f", energy) + " mWh");
    }
    
    /**
     * Calculate fog node specific metrics
     * @param fogId Fog node ID
     */
    public void calculateFogNodeMetrics(String fogId) {
        // Calculate packets processed by this fog node
        int packets = packetsProcessedByFogNode.getOrDefault(fogId, 0);
        System.out.println("Fog node " + fogId + " packets processed: " + packets);
        
        // Calculate energy consumption for this fog node
        double energy = energyConsumedByFogNode.getOrDefault(fogId, 0.0);
        System.out.println("Fog node " + fogId + " energy consumption: " + String.format("%.6f", energy) + " mWh");
    }
    
    /**
     * Calculate cloud datacenter metrics
     * @param cloudId Cloud datacenter ID
     */
    public void calculateCloudMetrics(String cloudId) {
        // For now, just print the total packets processed at cloud
        System.out.println("Cloud datacenter " + cloudId + " packets processed: " + packetsProcessedAtCloud);
        
        // Calculate energy consumption for cloud
        double energy = energyConsumptionByLayer.getOrDefault("Cloud", 0.0);
        System.out.println("Cloud datacenter " + cloudId + " energy consumption: " + String.format("%.6f", energy) + " mWh");
    }
    
    /**
     * Calculate security-related metrics
     */
    public void calculateSecurityMetrics() {
        // Calculate security mitigation rate
        if (securityIncidentsDetected > 0) {
            mitigationSuccessRate = (double) securityIncidentsMitigated / securityIncidentsDetected;
        }
        
        System.out.println("Security incidents by type:");
        for (Map.Entry<String, Integer> entry : securityIncidentsByType.entrySet()) {
            System.out.println("  " + entry.getKey() + ": " + entry.getValue());
        }
        
        System.out.println("Security overhead by type:");
        for (Map.Entry<String, Double> entry : securityOverheadByType.entrySet()) {
            System.out.println("  " + entry.getKey() + ": " + String.format("%.3f", entry.getValue()) + " ms");
        }
        
        System.out.println("Security overhead by layer:");
        for (Map.Entry<String, Double> entry : securityOverheadByLayer.entrySet()) {
            System.out.println("  " + entry.getKey() + ": " + String.format("%.3f", entry.getValue()) + " ms");
        }
    }
    
    /**
     * Calculate offloading-related metrics
     */
    public void calculateOffloadingMetrics() {
        System.out.println("Offloading statistics:");
        for (Map.Entry<String, List<OffloadingRecord>> entry : offloadingRecords.entrySet()) {
            System.out.println("  " + entry.getKey() + ": " + entry.getValue().size() + " packets");
        }
    }
    
    /**
     * Calculate network-related metrics
     */
    public void calculateNetworkMetrics() {
        // Calculate bandwidth saved by edge and fog processing
        double totalDataSize = 0.0;
        double edgeDataSize = 0.0;
        double fogDataSize = 0.0;
        
        for (Map.Entry<String, List<OffloadingRecord>> entry : offloadingRecords.entrySet()) {
            for (OffloadingRecord record : entry.getValue()) {
                if (entry.getKey().equals("IoT-to-Edge")) {
                    totalDataSize += record.dataSize;
                } else if (entry.getKey().equals("Edge-to-Fog")) {
                    edgeDataSize += record.dataSize;
                } else if (entry.getKey().equals("Fog-to-Cloud")) {
                    fogDataSize += record.dataSize;
                }
            }
        }
        
        // Calculate bandwidth saved (data that didn't need to go to cloud)
        bandwidthSavedByEdgeProcessing = totalDataSize - edgeDataSize;
        bandwidthSavedByFogProcessing = edgeDataSize - fogDataSize;
        totalBandwidthSaved = bandwidthSavedByEdgeProcessing + bandwidthSavedByFogProcessing;
        
        System.out.println("Network metrics:");
        System.out.println("  Total data size: " + String.format("%.2f", totalDataSize / 1024.0) + " KB");
        System.out.println("  Bandwidth saved by edge processing: " + String.format("%.2f", bandwidthSavedByEdgeProcessing / 1024.0) + " KB");
        System.out.println("  Bandwidth saved by fog processing: " + String.format("%.2f", bandwidthSavedByFogProcessing / 1024.0) + " KB");
        System.out.println("  Total bandwidth saved: " + String.format("%.2f", totalBandwidthSaved / 1024.0) + " KB");
    }
    
    /**
     * Get security mitigation rate
     * @return Security mitigation rate (0.0 to 1.0)
     */
    public double getSecurityMitigationRate() {
        return mitigationSuccessRate;
    }
    
    /**
     * Increment bandwidth saved by edge processing
     * @param bandwidthSaved Bandwidth saved in bytes
     */
    public void incrementBandwidthSavedByEdgeProcessing(double bandwidthSaved) {
        this.bandwidthSavedByEdgeProcessing += bandwidthSaved;
    }
    
    /**
     * Increment bandwidth saved by fog processing
     * @param bandwidthSaved Bandwidth saved in bytes
     */
    public void incrementBandwidthSavedByFogProcessing(double bandwidthSaved) {
        this.bandwidthSavedByFogProcessing += bandwidthSaved;
    }
    
    /**
     * Increment energy saved by edge processing
     * @param energySaved Energy saved in mWh
     */
    public void incrementEnergySavedByEdgeProcessing(double energySaved) {
        this.energySavedByEdgeProcessing += energySaved;
    }
    
    /**
     * Increment energy saved by fog processing
     * @param energySaved Energy saved in mWh
     */
    public void incrementEnergySavedByFogProcessing(double energySaved) {
        this.energySavedByFogProcessing += energySaved;
    }
    
    /**
     * Get all device IDs in the simulation
     * @return Set of device IDs
     */
    public java.util.Set<String> getAllDeviceIds() {
        return packetsGeneratedByDevice.keySet();
    }
}
