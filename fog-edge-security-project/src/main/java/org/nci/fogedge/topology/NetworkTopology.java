package org.nci.fogedge.topology;

import java.util.ArrayList;
import java.util.List;

/**
 * Class representing the network topology for the fog computing environment
 * 
 * This class models the physical network structure including IoT devices,
 * edge nodes, fog nodes, and cloud datacenter.
 * 
 * @author NCI H9FEC Student
 * @version 1.0
 */
public class NetworkTopology {
    
    private List<IoTDevice> iotDevices;
    private List<EdgeNode> edgeNodes;
    private List<FogNode> fogNodes;
    private CloudDatacenter cloudDatacenter;
    
    /**
     * Default constructor
     */
    public NetworkTopology() {
        this.iotDevices = new ArrayList<>();
        this.edgeNodes = new ArrayList<>();
        this.fogNodes = new ArrayList<>();
    }
    
    /**
     * Add an IoT device to the topology
     * @param device IoT device to add
     */
    public void addIoTDevice(IoTDevice device) {
        iotDevices.add(device);
    }
    
    /**
     * Add an edge node to the topology
     * @param node Edge node to add
     */
    public void addEdgeNode(EdgeNode node) {
        edgeNodes.add(node);
    }
    
    /**
     * Add a fog node to the topology
     * @param node Fog node to add
     */
    public void addFogNode(FogNode node) {
        fogNodes.add(node);
    }
    
    /**
     * Set the cloud datacenter for the topology
     * @param datacenter Cloud datacenter
     */
    public void setCloudDatacenter(CloudDatacenter datacenter) {
        this.cloudDatacenter = datacenter;
    }
    
    /**
     * Get all IoT devices in the topology
     * @return List of IoT devices
     */
    public List<IoTDevice> getIotDevices() {
        return iotDevices;
    }
    
    /**
     * Get all edge nodes in the topology
     * @return List of edge nodes
     */
    public List<EdgeNode> getEdgeNodes() {
        return edgeNodes;
    }
    
    /**
     * Get all fog nodes in the topology
     * @return List of fog nodes
     */
    public List<FogNode> getFogNodes() {
        return fogNodes;
    }
    
    /**
     * Get the cloud datacenter
     * @return Cloud datacenter
     */
    public CloudDatacenter getCloudDatacenter() {
        return cloudDatacenter;
    }
    
    /**
     * Print the topology information
     * @return String representation of the topology
     */
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("Network Topology:\n");
        sb.append("----------------\n");
        sb.append("IoT Devices: ").append(iotDevices.size()).append("\n");
        sb.append("Edge Nodes: ").append(edgeNodes.size()).append("\n");
        sb.append("Fog Nodes: ").append(fogNodes.size()).append("\n");
        sb.append("Cloud Datacenter: ").append(cloudDatacenter != null ? "1" : "0").append("\n");
        
        return sb.toString();
    }
}
