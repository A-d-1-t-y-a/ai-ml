package org.nci.fogedge.simulation;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Random;

import org.nci.fogedge.model.SimulationParameters;
import org.nci.fogedge.model.SimulationResults;
import org.nci.fogedge.security.SecurityIncident;
import org.nci.fogedge.security.SecurityManager;
import org.nci.fogedge.topology.CloudDatacenter;
import org.nci.fogedge.topology.EdgeNode;
import org.nci.fogedge.topology.FogNode;
import org.nci.fogedge.topology.IoTDevice;
import org.nci.fogedge.topology.NetworkTopology;

/**
 * Simulation Engine for the fog computing environment
 * 
 * This class manages the simulation execution, including event scheduling,
 * data generation, transmission, processing, and security events.
 * 
 * @author NCI H9FEC Student
 * @version 1.0
 */
public class SimulationEngine {
    
    /**
     * Enum for simulation event types
     */
    public enum EventType {
        GENERATE_DATA,
        TRANSMIT_TO_EDGE,
        PROCESS_AT_EDGE,
        TRANSMIT_TO_FOG,
        PROCESS_AT_FOG,
        TRANSMIT_TO_CLOUD,
        PROCESS_AT_CLOUD,
        SECURITY_CHECK,
        SECURITY_INCIDENT,
        SIMULATION_END
    }
    
    private SimulationParameters parameters;
    private NetworkTopology topology;
    private SecurityManager securityManager;
    private SimulationResults results;
    private double currentTime;
    private PriorityQueue<SimulationEvent> eventQueue;
    private Random random;
    private Map<String, String> dataPacketMap; // Maps data IDs to their content
    private Map<String, Map<String, Double>> packetTimingMap; // Maps data IDs to their timing information
    private Map<String, Integer> packetSizeMap; // Maps data IDs to their size in bytes
    private Map<String, String> packetSourceMap; // Maps data IDs to their source device ID
    private Map<String, String> packetDestinationMap; // Maps data IDs to their current destination
    
    /**
     * Constructor with parameters
     * @param parameters Simulation parameters
     * @param topology Network topology
     * @param securityManager Security manager
     */
    public SimulationEngine(SimulationParameters parameters, NetworkTopology topology, SecurityManager securityManager) {
        this.parameters = parameters;
        this.topology = topology;
        this.securityManager = securityManager;
        this.results = new SimulationResults();
        this.currentTime = 0.0;
        this.eventQueue = new PriorityQueue<>();
        this.random = new Random(System.currentTimeMillis());
        this.dataPacketMap = new HashMap<>();
        this.packetTimingMap = new HashMap<>();
        this.packetSizeMap = new HashMap<>();
        this.packetSourceMap = new HashMap<>();
        this.packetDestinationMap = new HashMap<>();
    }
    
    /**
     * Initialize the simulation
     */
    public void initialize() {
        // Schedule initial data generation events for all IoT devices
        for (IoTDevice device : topology.getIotDevices()) {
            scheduleEvent(new SimulationEvent(
                EventType.GENERATE_DATA,
                random.nextDouble() * parameters.getInitialDataGenerationDelay(),
                device.getId(),
                null,
                null
            ));
        }
        
        // Schedule simulation end event
        scheduleEvent(new SimulationEvent(
            EventType.SIMULATION_END,
            parameters.getSimulationLength(),
            "SYSTEM",
            null,
            null
        ));
        
        System.out.println("Simulation initialized with " + topology.getIotDevices().size() + " IoT devices, " +
                topology.getEdgeNodes().size() + " edge nodes, " + topology.getFogNodes().size() + " fog nodes, and " +
                (topology.getCloudDatacenter() != null ? "1" : "0") + " cloud datacenter.");
    }
    
    /**
     * Run the simulation
     * @return Simulation results
     */
    public SimulationResults runSimulation() {
        System.out.println("Starting simulation...");
        
        // Process events until simulation end
        while (!eventQueue.isEmpty()) {
            SimulationEvent event = eventQueue.poll();
            currentTime = event.getTime();
            
            // Process the event
            processEvent(event);
            
            // Check if simulation end
            if (event.getType() == EventType.SIMULATION_END) {
                break;
            }
        }
        
        System.out.println("Simulation completed at time " + currentTime);
        return results;
    }
    
    /**
     * Process a simulation event
     * @param event Event to process
     */
    private void processEvent(SimulationEvent event) {
        switch (event.getType()) {
            case GENERATE_DATA:
                handleDataGeneration(event);
                break;
            case TRANSMIT_TO_EDGE:
                handleTransmitToEdge(event);
                break;
            case PROCESS_AT_EDGE:
                handleProcessAtEdge(event);
                break;
            case TRANSMIT_TO_FOG:
                handleTransmitToFog(event);
                break;
            case PROCESS_AT_FOG:
                handleProcessAtFog(event);
                break;
            case TRANSMIT_TO_CLOUD:
                handleTransmitToCloud(event);
                break;
            case PROCESS_AT_CLOUD:
                handleProcessAtCloud(event);
                break;
            case SECURITY_CHECK:
                handleSecurityCheck(event);
                break;
            case SECURITY_INCIDENT:
                handleSecurityIncident(event);
                break;
            case SIMULATION_END:
                handleSimulationEnd(event);
                break;
        }
    }
    
    /**
     * Handle data generation event
     * @param event Event to handle
     */
    private void handleDataGeneration(SimulationEvent event) {
        String deviceId = event.getSourceId();
        IoTDevice device = findIoTDevice(deviceId);
        
        if (device != null) {
            // Generate data
            String dataId = deviceId + "-" + currentTime;
            String data = "DATA_" + dataId;
            int dataSize = 1024; // 1KB per packet
            
            // Record start time for this packet's lifecycle
            Map<String, Double> packetTimings = new HashMap<>();
            packetTimings.put("generationTime", currentTime);
            packetTimings.put("processingStartTime", currentTime);
            
            // Apply security measures and track overhead
            double securityStartTime = currentTime;
            if (parameters.isIotEncryptionEnabled()) {
                data = securityManager.encryptData(data);
                double encryptionTime = 0.001 + (random.nextDouble() * 0.002); // 1-3ms encryption time
                results.recordSecurityOverhead("encryption", encryptionTime);
                results.recordSecurityOverheadByLayer("IoT", encryptionTime);
            }
            double securityEndTime = currentTime + 0.001 + (random.nextDouble() * 0.002);
            packetTimings.put("securityEndTime", securityEndTime);
            
            // Store data and timings
            dataPacketMap.put(dataId, data);
            
        }
    }
    
    /**
     * Handle transmit to edge event
     * @param event Event to handle
     */
    private void handleTransmitToEdge(SimulationEvent event) {
        String sourceId = event.getSourceId();
        String edgeId = event.getDestinationId();
        String dataId = event.getDataId();
        EdgeNode edge = findEdgeNode(edgeId);
        
        if (edge != null && dataPacketMap.containsKey(dataId)) {
            String data = dataPacketMap.get(dataId);
            int dataSize = packetSizeMap.getOrDefault(dataId, 1024); // Default to 1KB if not found
            
            // Record transmission
            results.incrementPacketsTransmittedToEdge();
            
            // Record packet timing
            Map<String, Double> timingMap = packetTimingMap.getOrDefault(dataId, new HashMap<>());
            timingMap.put("arrivedAtEdge", currentTime);
            packetTimingMap.put(dataId, timingMap);
            
            // Calculate and record transmission energy consumption
            double transmissionEnergy = 0.0001 * dataSize; // 0.1 mWh per KB
            results.recordEnergyConsumption("IoT", transmissionEnergy);
            results.recordEnergyConsumptionByDevice(sourceId, transmissionEnergy);
            results.incrementTransmissionEnergyConsumption(transmissionEnergy);
            
            // Record data offloading metrics
            results.recordDataOffloading(sourceId, edgeId, dataId, dataSize, currentTime, "IoT-to-Edge");
            
            System.out.println("[" + currentTime + "] Data packet " + dataId + " transmitted from " + sourceId + " to edge node " + edgeId);
            
            // Schedule security check
            scheduleEvent(new SimulationEvent(
                EventType.SECURITY_CHECK,
                currentTime + 0.001, // Small delay for security check
                sourceId,
                edgeId,
                dataId
            ));
            
            // Schedule processing at edge
            double processingDelay = edge.calculateProcessingTime(data);
            
            scheduleEvent(new SimulationEvent(
                EventType.PROCESS_AT_EDGE,
                currentTime + processingDelay,
                edgeId,
                null,
                dataId
            ));
        }
    }
    
    /**
     * Handle process at edge event
     * @param event Event to handle
     */
    private void handleProcessAtEdge(SimulationEvent event) {
        String edgeId = event.getSourceId();
        String dataId = event.getDataId();
        EdgeNode edge = findEdgeNode(edgeId);
        
        if (edge != null && dataPacketMap.containsKey(dataId)) {
            String data = dataPacketMap.get(dataId);
            int dataSize = packetSizeMap.getOrDefault(dataId, 1024); // Default to 1KB if not found
            String sourceDeviceId = packetSourceMap.get(dataId);
            
            // Record packet timing
            Map<String, Double> timingMap = packetTimingMap.getOrDefault(dataId, new HashMap<>());
            timingMap.put("processedAtEdge", currentTime);
            packetTimingMap.put(dataId, timingMap);
            
            // Process data at edge
            edge.processData(data, currentTime);
            
            // Record processing
            results.incrementPacketsProcessedAtEdge();
            
            // Calculate and record processing energy consumption
            double processingEnergy = 0.0002 * dataSize; // 0.2 mWh per KB
            results.recordEnergyConsumption("Edge", processingEnergy);
            
            // Record bandwidth saved by edge processing
            double bandwidthSaved = dataSize * 0.7; // Assume 70% bandwidth saved
            results.incrementBandwidthSavedByEdgeProcessing(bandwidthSaved);
            
            // Record energy saved by edge processing
            double energySaved = 0.0001 * dataSize; // 0.1 mWh per KB saved
            results.incrementEnergySavedByEdgeProcessing(energySaved);
            
            System.out.println("[" + currentTime + "] Data packet " + dataId + " processed at edge node " + edgeId);
            
            // Decide whether to offload to fog or not
            boolean offloadToFog = random.nextDouble() < parameters.getEdgeToFogOffloadingProbability();
            
            if (offloadToFog && !edge.getConnectedFogNodes().isEmpty()) {
                // Select a fog node to offload to
                FogNode targetFog = edge.getConnectedFogNodes().get(
                        random.nextInt(edge.getConnectedFogNodes().size()));
                String fogId = targetFog.getId();
                
                // Calculate transmission delay
                double transmissionDelay = calculateTransmissionDelay(edge, targetFog);
                
                // Calculate and record transmission energy consumption
                double transmissionEnergy = 0.00005 * dataSize; // 0.05 mWh per KB (less than IoT-to-Edge)
                results.recordEnergyConsumption("Edge", transmissionEnergy);
                results.incrementTransmissionEnergyConsumption(transmissionEnergy);
                
                // Record data offloading metrics
                results.recordDataOffloading(edgeId, fogId, dataId, dataSize, currentTime, "Edge-to-Fog");
                
                // Schedule transmission to fog
                scheduleEvent(new SimulationEvent(
                    EventType.TRANSMIT_TO_FOG,
                    currentTime + transmissionDelay,
                    edgeId,
                    fogId,
                    dataId
                ));
                
                System.out.println("[" + currentTime + "] Data packet " + dataId + " offloaded from edge node " + edgeId + " to fog node " + fogId);
            } else {
                // Data processing completed at edge
                // Calculate end-to-end latency
                double generationTime = timingMap.getOrDefault("generated", 0.0);
                double latency = currentTime - generationTime;
                results.recordLatency(latency);
                
                System.out.println("[" + currentTime + "] Data packet " + dataId + " processing completed at edge node " + edgeId + ". Latency: " + String.format("%.3f", latency) + " ms");
            }
        }
    }
    
    /**
     * Handle transmit to fog event
     * @param event Event to handle
     */
    private void handleTransmitToFog(SimulationEvent event) {
        String sourceId = event.getSourceId();
        String fogId = event.getDestinationId();
        String dataId = event.getDataId();
        FogNode fog = findFogNode(fogId);
        
        if (fog != null && dataPacketMap.containsKey(dataId)) {
            String data = dataPacketMap.get(dataId);
            int dataSize = packetSizeMap.getOrDefault(dataId, 1024); // Default to 1KB if not found
            
            // Record transmission
            results.incrementPacketsTransmittedToFog();
            
            // Record packet timing
            Map<String, Double> timingMap = packetTimingMap.getOrDefault(dataId, new HashMap<>());
            timingMap.put("arrivedAtFog", currentTime);
            packetTimingMap.put(dataId, timingMap);
            
            System.out.println("[" + currentTime + "] Data packet " + dataId + " transmitted from " + sourceId + " to fog node " + fogId);
            
            // Schedule security check
            scheduleEvent(new SimulationEvent(
                EventType.SECURITY_CHECK,
                currentTime + 0.001, // Small delay for security check
                sourceId,
                fogId,
                dataId
            ));
            
            // Schedule processing at fog
            double processingDelay = fog.calculateProcessingTime(data);
            
            scheduleEvent(new SimulationEvent(
                EventType.PROCESS_AT_FOG,
                currentTime + processingDelay,
                fogId,
                null,
                dataId
            ));
        }
    }
    
    /**
     * Handle process at fog event
     * @param event Event to handle
     */
    private void handleProcessAtFog(SimulationEvent event) {
        String fogId = event.getSourceId();
        String dataId = event.getDataId();
        FogNode fog = findFogNode(fogId);
        
        if (fog != null && dataPacketMap.containsKey(dataId)) {
            String data = dataPacketMap.get(dataId);
            int dataSize = packetSizeMap.getOrDefault(dataId, 1024); // Default to 1KB if not found
            String sourceDeviceId = packetSourceMap.get(dataId);
            
            // Record packet timing
            Map<String, Double> timingMap = packetTimingMap.getOrDefault(dataId, new HashMap<>());
            timingMap.put("processedAtFog", currentTime);
            packetTimingMap.put(dataId, timingMap);
            
            // Process data at fog
            fog.processData(data, currentTime);
            
            // Record processing
            results.incrementPacketsProcessedAtFog();
            
            // Calculate and record processing energy consumption
            double processingEnergy = 0.00015 * dataSize; // 0.15 mWh per KB (less than edge)
            results.recordEnergyConsumption("Fog", processingEnergy);
            
            // Record bandwidth saved by fog processing
            double bandwidthSaved = dataSize * 0.5; // Assume 50% bandwidth saved
            results.incrementBandwidthSavedByFogProcessing(bandwidthSaved);
            
            // Record energy saved by fog processing
            double energySaved = 0.00005 * dataSize; // 0.05 mWh per KB saved
            results.incrementEnergySavedByFogProcessing(energySaved);
            
            System.out.println("[" + currentTime + "] Data packet " + dataId + " processed at fog node " + fogId);
            
            // Decide whether to offload to cloud or not
            boolean offloadToCloud = random.nextDouble() < parameters.getFogToCloudOffloadingProbability();
            
            if (offloadToCloud && topology.getCloudDatacenter() != null) {
                String cloudId = topology.getCloudDatacenter().getId();
                
                // Calculate transmission delay
                double transmissionDelay = calculateTransmissionDelay(fog, topology.getCloudDatacenter());
                
                // Calculate and record transmission energy consumption
                double transmissionEnergy = 0.00003 * dataSize; // 0.03 mWh per KB (less than Edge-to-Fog)
                results.recordEnergyConsumption("Fog", transmissionEnergy);
                results.incrementTransmissionEnergyConsumption(transmissionEnergy);
                
                // Record data offloading metrics
                results.recordDataOffloading(fogId, cloudId, dataId, dataSize, currentTime, "Fog-to-Cloud");
                
                // Schedule transmission to cloud
                scheduleEvent(new SimulationEvent(
                    EventType.TRANSMIT_TO_CLOUD,
                    currentTime + transmissionDelay,
                    fogId,
                    cloudId,
                    dataId
                ));
                
                System.out.println("[" + currentTime + "] Data packet " + dataId + " offloaded from fog node " + fogId + " to cloud datacenter " + cloudId);
            } else {
                // Data processing completed at fog
                // Calculate end-to-end latency
                double generationTime = timingMap.getOrDefault("generated", 0.0);
                double latency = currentTime - generationTime;
                results.recordLatency(latency);
                
                System.out.println("[" + currentTime + "] Data packet " + dataId + " processing completed at fog node " + fogId + ". Latency: " + String.format("%.3f", latency) + " ms");
            }
        }
    }
    
    /**
     * Handle transmit to cloud event
     * @param event Event to handle
     */
    private void handleTransmitToCloud(SimulationEvent event) {
        String sourceId = event.getSourceId();
        String cloudId = event.getDestinationId();
        String dataId = event.getDataId();
        CloudDatacenter cloud = topology.getCloudDatacenter();
        
        if (cloud != null && cloud.getId().equals(cloudId) && dataPacketMap.containsKey(dataId)) {
            String data = dataPacketMap.get(dataId);
            int dataSize = packetSizeMap.getOrDefault(dataId, 1024); // Default to 1KB if not found
            
            // Record transmission
            results.incrementPacketsTransmittedToCloud();
            
            // Record packet timing
            Map<String, Double> timingMap = packetTimingMap.getOrDefault(dataId, new HashMap<>());
            timingMap.put("arrivedAtCloud", currentTime);
            packetTimingMap.put(dataId, timingMap);
            
            System.out.println("[" + currentTime + "] Data packet " + dataId + " transmitted from " + sourceId + " to cloud datacenter " + cloudId);
            
            // Schedule processing at cloud
            double processingDelay = 0.005; // Base cloud processing delay
            
            scheduleEvent(new SimulationEvent(
                EventType.PROCESS_AT_CLOUD,
                currentTime + processingDelay,
                cloudId,
                null,
                dataId
            ));
        }
    }
    
    /**
     * Handle process at cloud event
     * @param event Event to handle
     */
    private void handleProcessAtCloud(SimulationEvent event) {
        String cloudId = event.getSourceId();
        String dataId = event.getDataId();
        CloudDatacenter cloud = topology.getCloudDatacenter();
        
        if (cloud != null && cloud.getId().equals(cloudId) && dataPacketMap.containsKey(dataId)) {
            String data = dataPacketMap.get(dataId);
            int dataSize = packetSizeMap.getOrDefault(dataId, 1024); // Default to 1KB if not found
            String sourceDeviceId = packetSourceMap.get(dataId);
            
            // Record packet timing
            Map<String, Double> timingMap = packetTimingMap.getOrDefault(dataId, new HashMap<>());
            timingMap.put("processedAtCloud", currentTime);
            packetTimingMap.put(dataId, timingMap);
            
            // Process data at cloud
            cloud.processData(dataId, data);
            
            // Record processing
            results.incrementPacketsProcessedAtCloud();
            
            // Calculate and record processing energy consumption
            double processingEnergy = 0.0001 * dataSize; // 0.1 mWh per KB (less than fog)
            results.recordEnergyConsumption("Cloud", processingEnergy);
            
            // Calculate end-to-end latency
            double generationTime = timingMap.getOrDefault("generated", 0.0);
            double latency = currentTime - generationTime;
            results.recordLatency(latency);
            
            // Record completion of data processing
            results.recordPacketCompletion(dataId, sourceDeviceId, latency, currentTime);
            
            System.out.println("[" + currentTime + "] Data packet " + dataId + " processed at cloud datacenter " + cloudId + ". End-to-end latency: " + String.format("%.3f", latency) + " ms");
        }
    }
    
    /**
     * Handle security check event
     * @param event Event to handle
     */
    private void handleSecurityCheck(SimulationEvent event) {
        String sourceId = event.getSourceId();
        String destinationId = event.getDestinationId();
        String dataId = event.getDataId();
        
        if (dataPacketMap.containsKey(dataId)) {
            String data = dataPacketMap.get(dataId);
            int dataSize = packetSizeMap.getOrDefault(dataId, 1024); // Default to 1KB if not found
            
            // Record packet timing
            Map<String, Double> timingMap = packetTimingMap.getOrDefault(dataId, new HashMap<>());
            timingMap.put("securityCheck", currentTime);
            packetTimingMap.put(dataId, timingMap);
            
            // Calculate and record security check overhead
            double securityCheckTime = 0.001; // 1ms for security check
            double securityEnergy = 0.00002 * dataSize; // 0.02 mWh per KB for security check
            
            // Record security overhead
            results.incrementSecurityOverhead(securityCheckTime);
            results.recordEnergyConsumption("Security", securityEnergy);
            
            System.out.println("[" + currentTime + "] Security check for data packet " + dataId + " from " + sourceId + " to " + destinationId);
            
            // Check for security incident
            SecurityIncident incident = securityManager.checkForSecurityIncident(data, sourceId, destinationId);
            if (incident != null) {
                // Schedule security incident handling
                scheduleEvent(new SimulationEvent(
                    EventType.SECURITY_INCIDENT,
                    currentTime + 0.002, // Small delay for incident handling
                    sourceId,
                    destinationId,
                    dataId
                ));
                
                // Record security incident
                results.incrementSecurityIncidentsDetected();
                results.recordSecurityIncident(incident.getType(), sourceId, destinationId, dataId, currentTime);
                
                System.out.println("[" + currentTime + "] Security incident detected for data packet " + dataId + ": " + incident.getType());
            }
        }
    }
    
    /**
     * Handle security incident event
     * @param event Event to handle
     */
    private void handleSecurityIncident(SimulationEvent event) {
        String sourceId = event.getSourceId();
        String destinationId = event.getDestinationId();
        String dataId = event.getDataId();
        
        if (dataPacketMap.containsKey(dataId)) {
            int dataSize = packetSizeMap.getOrDefault(dataId, 1024); // Default to 1KB if not found
            
            // Record packet timing
            Map<String, Double> timingMap = packetTimingMap.getOrDefault(dataId, new HashMap<>());
            timingMap.put("securityIncident", currentTime);
            packetTimingMap.put(dataId, timingMap);
            
            // Calculate security incident handling overhead
            double securityIncidentTime = 0.005; // 5ms for incident handling
            double securityIncidentEnergy = 0.0001 * dataSize; // 0.1 mWh per KB for incident handling
            
            // Record security overhead
            results.incrementSecurityOverhead(securityIncidentTime);
            results.recordEnergyConsumption("Security", securityIncidentEnergy);
            
            System.out.println("[" + currentTime + "] Handling security incident for data packet " + dataId + " at " + destinationId);
            
            // Attempt to mitigate the incident
            boolean mitigated = false;
            
            // Determine which node handles the incident
            if (destinationId.startsWith("EDGE")) {
                EdgeNode edge = findEdgeNode(destinationId);
                if (edge != null) {
                    mitigated = edge.handleSecurityIncident(dataId, currentTime);
                }
            } else if (destinationId.startsWith("FOG")) {
                FogNode fog = findFogNode(destinationId);
                if (fog != null) {
                    mitigated = fog.handleSecurityIncident(dataId, currentTime);
                }
            } else if (destinationId.startsWith("CLOUD")) {
                // Cloud datacenter security incident handling
                CloudDatacenter cloud = topology.getCloudDatacenter();
                if (cloud != null && cloud.getId().equals(destinationId)) {
                    // For now, assume cloud always successfully mitigates incidents
                    mitigated = true;
                }
            }
            
            // Record mitigation result
            if (mitigated) {
                results.incrementSecurityIncidentsMitigated();
                System.out.println("[" + currentTime + "] Security incident for data packet " + dataId + " successfully mitigated at " + destinationId);
            } else {
                results.incrementSecurityIncidentsUnmitigated();
                System.out.println("[" + currentTime + "] Failed to mitigate security incident for data packet " + dataId + " at " + destinationId);
            }
            
            // Record detailed security incident metrics
            results.recordSecurityIncidentHandling(sourceId, destinationId, dataId, mitigated, securityIncidentTime, securityIncidentEnergy, currentTime);
        }
    }
    
    /**
     * Handle simulation end event
     * @param event Event to handle
     */
    private void handleSimulationEnd(SimulationEvent event) {
        // Calculate final metrics
        results.calculateDerivedMetrics();
        
        // Calculate per-device statistics
        for (IoTDevice device : topology.getIotDevices()) {
            String deviceId = device.getId();
            results.calculateDeviceMetrics(deviceId);
        }
        
        // Calculate per-edge-node statistics
        for (EdgeNode edge : topology.getEdgeNodes()) {
            String edgeId = edge.getId();
            results.calculateEdgeNodeMetrics(edgeId);
        }
        
        // Calculate per-fog-node statistics
        for (FogNode fog : topology.getFogNodes()) {
            String fogId = fog.getId();
            results.calculateFogNodeMetrics(fogId);
        }
        
        // Calculate cloud datacenter statistics
        if (topology.getCloudDatacenter() != null) {
            results.calculateCloudMetrics(topology.getCloudDatacenter().getId());
        }
        
        // Calculate security metrics
        results.calculateSecurityMetrics();
        
        // Calculate offloading metrics
        results.calculateOffloadingMetrics();
        
        // Calculate network metrics
        results.calculateNetworkMetrics();
        
        System.out.println("Simulation ended at time " + currentTime);
        System.out.println("Total packets generated: " + results.getTotalPacketsGenerated());
        System.out.println("Packets processed at edge: " + results.getPacketsProcessedAtEdge());
        System.out.println("Packets processed at fog: " + results.getPacketsProcessedAtFog());
        System.out.println("Packets processed at cloud: " + results.getPacketsProcessedAtCloud());
        System.out.println("Average end-to-end latency: " + String.format("%.3f", results.getAverageLatency()) + " ms");
        System.out.println("Total energy consumption: " + String.format("%.3f", results.getTotalEnergyConsumption()) + " mWh");
        System.out.println("Security incidents detected: " + results.getSecurityIncidentsDetected());
        System.out.println("Security incidents mitigated: " + results.getSecurityIncidentsMitigated());
        System.out.println("Security mitigation rate: " + String.format("%.2f", results.getSecurityMitigationRate() * 100) + "%");
    }
    
    /**
     * Schedule a simulation event
     * @param event Event to schedule
     */
    private void scheduleEvent(SimulationEvent event) {
        eventQueue.add(event);
    }
    
    /**
     * Calculate transmission delay between two nodes
     * @param source Source node
     * @param destination Destination node
     * @return Transmission delay
     */
    private double calculateTransmissionDelay(Object source, Object destination) {
        // Base delay
        double baseDelay = 0.002; // 2ms base delay
        
        // Add random variation
        baseDelay += random.nextDouble() * 0.002; // 0-2ms random variation
        
        return baseDelay;
    }
    
    /**
     * Find an IoT device by ID
     * @param deviceId Device ID
     * @return IoT device, or null if not found
     */
    private IoTDevice findIoTDevice(String deviceId) {
        for (IoTDevice device : topology.getIotDevices()) {
            if (device.getId().equals(deviceId)) {
                return device;
            }
        }
        return null;
    }
    
    /**
     * Find an edge node by ID
     * @param edgeId Edge node ID
     * @return Edge node, or null if not found
     */
    private EdgeNode findEdgeNode(String edgeId) {
        for (EdgeNode edge : topology.getEdgeNodes()) {
            if (edge.getId().equals(edgeId)) {
                return edge;
            }
        }
        return null;
    }
    
    /**
     * Find a fog node by ID
     * @param fogId Fog node ID
     * @return Fog node, or null if not found
     */
    private FogNode findFogNode(String fogId) {
        for (FogNode fog : topology.getFogNodes()) {
            if (fog.getId().equals(fogId)) {
                return fog;
            }
        }
        return null;
    }
    
    /**
     * Get simulation results
     * @return Simulation results
     */
    public SimulationResults getResults() {
        return results;
    }
    
    /**
     * Get current simulation time
     * @return Current simulation time
     */
    public double getCurrentTime() {
        return currentTime;
    }
    
    /**
     * Inner class representing a simulation event
     */
    private class SimulationEvent implements Comparable<SimulationEvent> {
        private EventType type;
        private double time;
        private String sourceId;
        private String destinationId;
        private String dataId;
        
        /**
         * Constructor with parameters
         * @param type Event type
         * @param time Event time
         * @param sourceId Source ID
         * @param destinationId Destination ID
         * @param dataId Data ID
         */
        public SimulationEvent(EventType type, double time, String sourceId, String destinationId, String dataId) {
            this.type = type;
            this.time = time;
            this.sourceId = sourceId;
            this.destinationId = destinationId;
            this.dataId = dataId;
        }
        
        /**
         * Get event type
         * @return Event type
         */
        public EventType getType() {
            return type;
        }
        
        /**
         * Get event time
         * @return Event time
         */
        public double getTime() {
            return time;
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
         * Get data ID
         * @return Data ID
         */
        public String getDataId() {
            return dataId;
        }
        
        @Override
        public int compareTo(SimulationEvent other) {
            return Double.compare(this.time, other.time);
        }
    }
}
