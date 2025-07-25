package org.nci.fogedge.simulation;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Random;

import org.nci.fogedge.model.SimulationParameters;
import org.nci.fogedge.model.SimulationResults;
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
            
            // Apply security measures
            if (parameters.isIotEncryptionEnabled()) {
                data = securityManager.encryptData(data);
            }
            
            // Store data
            dataPacketMap.put(dataId, data);
            
            // Record data generation
            results.incrementTotalPacketsGenerated();
            
            // Schedule transmission to edge
            if (!device.getConnectedEdgeNodes().isEmpty()) {
                EdgeNode targetEdge = device.getConnectedEdgeNodes().get(
                        random.nextInt(device.getConnectedEdgeNodes().size()));
                
                double transmissionDelay = calculateTransmissionDelay(device, targetEdge);
                
                scheduleEvent(new SimulationEvent(
                    EventType.TRANSMIT_TO_EDGE,
                    currentTime + transmissionDelay,
                    deviceId,
                    targetEdge.getId(),
                    dataId
                ));
            }
            
            // Schedule next data generation
            double nextGenerationTime = currentTime + parameters.getDataGenerationInterval() * (0.8 + random.nextDouble() * 0.4);
            if (nextGenerationTime < parameters.getSimulationLength()) {
                scheduleEvent(new SimulationEvent(
                    EventType.GENERATE_DATA,
                    nextGenerationTime,
                    deviceId,
                    null,
                    null
                ));
            }
        }
    }
    
    /**
     * Handle transmit to edge event
     * @param event Event to handle
     */
    private void handleTransmitToEdge(SimulationEvent event) {
        String edgeId = event.getDestinationId();
        String dataId = event.getDataId();
        EdgeNode edge = findEdgeNode(edgeId);
        
        if (edge != null && dataPacketMap.containsKey(dataId)) {
            String data = dataPacketMap.get(dataId);
            
            // Record transmission
            results.incrementPacketsTransmittedToEdge();
            
            // Schedule security check
            scheduleEvent(new SimulationEvent(
                EventType.SECURITY_CHECK,
                currentTime + 0.001, // Small delay for security check
                event.getSourceId(),
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
            // Process data at edge
            String processedData = edge.processData(dataPacketMap.get(dataId), currentTime);
            dataPacketMap.put(dataId, processedData);
            
            // Record processing
            results.incrementPacketsProcessedAtEdge();
            
            // Decide whether to forward to fog
            if (random.nextDouble() < parameters.getEdgeToFogForwardingProbability() && !edge.getConnectedFogNodes().isEmpty()) {
                // Select a fog node
                FogNode targetFog = edge.getConnectedFogNodes().get(
                        random.nextInt(edge.getConnectedFogNodes().size()));
                
                // Calculate transmission delay
                double transmissionDelay = calculateTransmissionDelay(edge, targetFog);
                
                // Schedule transmission to fog
                scheduleEvent(new SimulationEvent(
                    EventType.TRANSMIT_TO_FOG,
                    currentTime + transmissionDelay,
                    edgeId,
                    targetFog.getId(),
                    dataId
                ));
            } else {
                // Data processed locally at edge, not forwarded
                results.incrementPacketsProcessedLocally();
            }
        }
    }
    
    /**
     * Handle transmit to fog event
     * @param event Event to handle
     */
    private void handleTransmitToFog(SimulationEvent event) {
        String fogId = event.getDestinationId();
        String dataId = event.getDataId();
        FogNode fog = findFogNode(fogId);
        
        if (fog != null && dataPacketMap.containsKey(dataId)) {
            String data = dataPacketMap.get(dataId);
            
            // Record transmission
            results.incrementPacketsTransmittedToFog();
            
            // Schedule security check
            scheduleEvent(new SimulationEvent(
                EventType.SECURITY_CHECK,
                currentTime + 0.001, // Small delay for security check
                event.getSourceId(),
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
            // Process data at fog
            String processedData = fog.processData(dataPacketMap.get(dataId), currentTime);
            dataPacketMap.put(dataId, processedData);
            
            // Record processing
            results.incrementPacketsProcessedAtFog();
            
            // Apply blockchain if enabled
            if (parameters.isFogBlockchainEnabled()) {
                processedData = securityManager.applyBlockchainSecurity(processedData);
                dataPacketMap.put(dataId, processedData);
            }
            
            // Decide whether to forward to cloud
            if (random.nextDouble() < parameters.getFogToCloudForwardingProbability() && topology.getCloudDatacenter() != null) {
                // Calculate transmission delay
                double transmissionDelay = calculateTransmissionDelay(fog, topology.getCloudDatacenter());
                
                // Schedule transmission to cloud
                scheduleEvent(new SimulationEvent(
                    EventType.TRANSMIT_TO_CLOUD,
                    currentTime + transmissionDelay,
                    fogId,
                    topology.getCloudDatacenter().getId(),
                    dataId
                ));
            } else {
                // Data processed locally at fog, not forwarded
                results.incrementPacketsProcessedLocally();
            }
        }
    }
    
    /**
     * Handle transmit to cloud event
     * @param event Event to handle
     */
    private void handleTransmitToCloud(SimulationEvent event) {
        String cloudId = event.getDestinationId();
        String dataId = event.getDataId();
        CloudDatacenter cloud = topology.getCloudDatacenter();
        
        if (cloud != null && cloud.getId().equals(cloudId) && dataPacketMap.containsKey(dataId)) {
            // Record transmission
            results.incrementPacketsTransmittedToCloud();
            
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
            // Process data at cloud
            cloud.processAndStoreData(dataId, currentTime);
            
            // Record processing
            results.incrementPacketsProcessedAtCloud();
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
            
            // Check for security incident
            if (securityManager.checkForSecurityIncident(data, sourceId, destinationId) != null) {
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
        }
        
        // Record mitigation result
        if (mitigated) {
            results.incrementSecurityIncidentsMitigated();
        } else {
            results.incrementSecurityIncidentsUnmitigated();
        }
    }
    
    /**
     * Handle simulation end event
     * @param event Event to handle
     */
    private void handleSimulationEnd(SimulationEvent event) {
        // Calculate final metrics
        results.calculateDerivedMetrics();
        
        System.out.println("Simulation ended at time " + currentTime);
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
