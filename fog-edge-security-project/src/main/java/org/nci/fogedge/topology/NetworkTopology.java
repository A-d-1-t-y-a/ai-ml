package org.nci.fogedge.topology;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.nci.fogedge.topology.EdgeNode.EdgeData;
import org.nci.fogedge.topology.IoTDevice.IoTData;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

/**
 * Class representing the network topology of IoT devices, edge nodes, and fog nodes
 * Based on the paper: "An Overview of Fog Computing and Edge Computing Security and Privacy Issues"
 * (Sensors 2021, 21, 8226, https://doi.org/10.3390/s21248226)
 */
public class NetworkTopology {
    private static final Logger logger = LogManager.getLogger(NetworkTopology.class);
    private static final Random random = new Random();
    
    private List<IoTDevice> iotDevices;
    private List<EdgeNode> edgeNodes;
    private List<FogNode> fogNodes;
    
    // Metrics
    private double totalDataGenerated;
    private double totalDataProcessedAtEdge;
    private double totalDataProcessedAtFog;
    private double totalProcessingTime;
    private double totalEnergyConsumption;
    
    public NetworkTopology() {
        this.iotDevices = new ArrayList<>();
        this.edgeNodes = new ArrayList<>();
        this.fogNodes = new ArrayList<>();
        
        this.totalDataGenerated = 0.0;
        this.totalDataProcessedAtEdge = 0.0;
        this.totalDataProcessedAtFog = 0.0;
        this.totalProcessingTime = 0.0;
        this.totalEnergyConsumption = 0.0;
    }
    
    /**
     * Add an IoT device to the topology
     * @param device The IoT device to add
     */
    public void addIoTDevice(IoTDevice device) {
        iotDevices.add(device);
        logger.debug("Added IoT device {} to topology", device.getId());
    }
    
    /**
     * Add an edge node to the topology
     * @param edgeNode The edge node to add
     */
    public void addEdgeNode(EdgeNode edgeNode) {
        edgeNodes.add(edgeNode);
        logger.debug("Added edge node {} to topology", edgeNode.getId());
    }
    
    /**
     * Add a fog node to the topology
     * @param fogNode The fog node to add
     */
    public void addFogNode(FogNode fogNode) {
        fogNodes.add(fogNode);
        logger.debug("Added fog node {} to topology", fogNode.getId());
    }
    
    /**
     * Connect IoT devices to edge nodes based on proximity and load balancing
     */
    public void connectDevicesToEdgeNodes() {
        if (edgeNodes.isEmpty()) {
            logger.warn("No edge nodes available to connect devices");
            return;
        }
        
        // Simple load balancing: distribute devices evenly across edge nodes
        int edgeNodeIndex = 0;
        for (IoTDevice device : iotDevices) {
            EdgeNode targetEdge = edgeNodes.get(edgeNodeIndex);
            targetEdge.connectDevice(device);
            
            // Move to next edge node (round-robin)
            edgeNodeIndex = (edgeNodeIndex + 1) % edgeNodes.size();
        }
        
        logger.info("Connected {} IoT devices to {} edge nodes", iotDevices.size(), edgeNodes.size());
    }
    
    /**
     * Connect edge nodes to fog nodes based on proximity and load balancing
     */
    public void connectEdgeNodesToFogNodes() {
        if (fogNodes.isEmpty()) {
            logger.warn("No fog nodes available to connect edge nodes");
            return;
        }
        
        // Simple load balancing: distribute edge nodes evenly across fog nodes
        int fogNodeIndex = 0;
        for (EdgeNode edgeNode : edgeNodes) {
            FogNode targetFog = fogNodes.get(fogNodeIndex);
            targetFog.connectEdgeNode(edgeNode);
            
            // Move to next fog node (round-robin)
            fogNodeIndex = (fogNodeIndex + 1) % fogNodes.size();
        }
        
        logger.info("Connected {} edge nodes to {} fog nodes", edgeNodes.size(), fogNodes.size());
    }
    
    /**
     * Get data for a specific edge node from generated IoT data
     * @param edgeNode The target edge node
     * @param generatedData List of generated data from IoT devices
     * @return List of data objects for this edge node
     */
    public List<Object> getDataForEdgeNode(EdgeNode edgeNode, List<Object> generatedData) {
        List<Object> nodeData = new ArrayList<>();
        
        // Filter data for devices connected to this edge node
        for (Object data : generatedData) {
            if (data instanceof IoTData) {
                IoTData iotData = (IoTData) data;
                
                // Find the device that generated this data
                for (IoTDevice device : iotDevices) {
                    if (device.getId().equals(iotData.getDeviceId()) && 
                            device.getConnectedEdgeNode() != null && 
                            device.getConnectedEdgeNode().getId().equals(edgeNode.getId())) {
                        nodeData.add(data);
                        break;
                    }
                }
            }
        }
        
        return nodeData;
    }
    
    /**
     * Get data for a specific fog node from processed edge data
     * @param fogNode The target fog node
     * @param edgeData List of processed data from edge nodes
     * @return List of data objects for this fog node
     */
    public List<Object> getDataForFogNode(FogNode fogNode, List<Object> edgeData) {
        List<Object> nodeData = new ArrayList<>();
        
        // Filter data for edge nodes connected to this fog node
        for (Object data : edgeData) {
            if (data instanceof EdgeData) {
                EdgeData processedData = (EdgeData) data;
                
                // Find the edge node that processed this data
                for (EdgeNode edgeNode : edgeNodes) {
                    if (edgeNode.getId().equals(processedData.getEdgeNodeId()) && 
                            edgeNode.getConnectedFogNode() != null && 
                            edgeNode.getConnectedFogNode().getId().equals(fogNode.getId())) {
                        nodeData.add(data);
                        break;
                    }
                }
            }
        }
        
        return nodeData;
    }
    
    /**
     * Calculate total data generated by IoT devices in the current step
     * @return Total data generated in KB
     */
    public double calculateTotalDataGenerated() {
        double total = 0.0;
        for (IoTDevice device : iotDevices) {
            total += device.getDataGenerationRate();
        }
        
        // Add to cumulative total
        totalDataGenerated += total;
        
        return total;
    }
    
    /**
     * Calculate data processed at edge layer in the current step
     * @return Total data processed at edge in KB
     */
    public double calculateDataProcessedAtEdge() {
        double total = calculateTotalDataGenerated();
        double processed = 0.0;
        
        for (EdgeNode edgeNode : edgeNodes) {
            // Estimate based on connected devices and reduction ratio
            double nodeData = 0.0;
            for (IoTDevice device : edgeNode.getConnectedDevices()) {
                nodeData += device.getDataGenerationRate();
            }
            
            processed += nodeData * edgeNode.getDataReductionRatio();
        }
        
        // Add to cumulative total
        totalDataProcessedAtEdge += processed;
        
        return processed;
    }
    
    /**
     * Calculate data processed at fog layer in the current step
     * @return Total data processed at fog in KB
     */
    public double calculateDataProcessedAtFog() {
        double edgeProcessed = calculateDataProcessedAtEdge();
        double processed = 0.0;
        
        for (FogNode fogNode : fogNodes) {
            // Estimate based on connected edge nodes and reduction ratio
            double nodeData = 0.0;
            for (EdgeNode edgeNode : fogNode.getConnectedEdgeNodes()) {
                // Calculate data from this edge node after reduction
                double edgeNodeData = 0.0;
                for (IoTDevice device : edgeNode.getConnectedDevices()) {
                    edgeNodeData += device.getDataGenerationRate();
                }
                
                nodeData += edgeNodeData * edgeNode.getDataReductionRatio();
            }
            
            processed += nodeData * fogNode.getDataReductionRatio();
        }
        
        // Add to cumulative total
        totalDataProcessedAtFog += processed;
        
        return processed;
    }
    
    /**
     * Calculate processing time for the current step
     * @return Processing time in ms
     */
    public double calculateProcessingTime() {
        double iotData = calculateTotalDataGenerated();
        
        // Calculate processing time at each layer
        double edgeProcessingTime = 0.0;
        for (EdgeNode edgeNode : edgeNodes) {
            // Estimate processing time based on connected devices
            double nodeData = 0.0;
            for (IoTDevice device : edgeNode.getConnectedDevices()) {
                nodeData += device.getDataGenerationRate();
            }
            
            edgeProcessingTime += (nodeData / edgeNode.getProcessingCapacity()) * 1000;
        }
        
        double fogProcessingTime = 0.0;
        for (FogNode fogNode : fogNodes) {
            // Estimate processing time based on connected edge nodes
            double nodeData = 0.0;
            for (EdgeNode edgeNode : fogNode.getConnectedEdgeNodes()) {
                // Calculate data from this edge node after reduction
                double edgeNodeData = 0.0;
                for (IoTDevice device : edgeNode.getConnectedDevices()) {
                    edgeNodeData += device.getDataGenerationRate();
                }
                
                nodeData += edgeNodeData * edgeNode.getDataReductionRatio();
            }
            
            fogProcessingTime += (nodeData / fogNode.getProcessingCapacity()) * 1000;
        }
        
        double totalTime = edgeProcessingTime + fogProcessingTime;
        
        // Add to cumulative total
        totalProcessingTime += totalTime;
        
        return totalTime;
    }
    
    /**
     * Calculate total energy consumption for the current step
     * @return Energy consumption in mJ
     */
    public double calculateTotalEnergyConsumption() {
        double iotEnergy = 0.0;
        for (IoTDevice device : iotDevices) {
            iotEnergy += device.getWirelessType().getEnergyConsumption() * device.getDataGenerationRate();
        }
        
        double edgeEnergy = 0.0;
        for (EdgeNode edgeNode : edgeNodes) {
            // Calculate data from this edge node
            double nodeData = 0.0;
            for (IoTDevice device : edgeNode.getConnectedDevices()) {
                nodeData += device.getDataGenerationRate();
            }
            
            edgeEnergy += edgeNode.calculateEnergyConsumption(nodeData);
        }
        
        double fogEnergy = 0.0;
        for (FogNode fogNode : fogNodes) {
            // Calculate data from this fog node
            double nodeData = 0.0;
            for (EdgeNode edgeNode : fogNode.getConnectedEdgeNodes()) {
                // Calculate data from this edge node after reduction
                double edgeNodeData = 0.0;
                for (IoTDevice device : edgeNode.getConnectedDevices()) {
                    edgeNodeData += device.getDataGenerationRate();
                }
                
                nodeData += edgeNodeData * edgeNode.getDataReductionRatio();
            }
            
            fogEnergy += fogNode.calculateEnergyConsumption(nodeData);
        }
        
        double totalEnergy = iotEnergy + edgeEnergy + fogEnergy;
        
        // Add to cumulative total
        totalEnergyConsumption += totalEnergy;
        
        return totalEnergy;
    }
    
    // Getters
    
    public List<IoTDevice> getIoTDevices() {
        return iotDevices;
    }
    
    public List<EdgeNode> getEdgeNodes() {
        return edgeNodes;
    }
    
    public List<FogNode> getFogNodes() {
        return fogNodes;
    }
    
    public double getTotalDataGenerated() {
        return totalDataGenerated;
    }
    
    public double getTotalDataProcessedAtEdge() {
        return totalDataProcessedAtEdge;
    }
    
    public double getTotalDataProcessedAtFog() {
        return totalDataProcessedAtFog;
    }
    
    public double getTotalProcessingTime() {
        return totalProcessingTime;
    }
    
    public double getTotalEnergyConsumption() {
        return totalEnergyConsumption;
    }
}
