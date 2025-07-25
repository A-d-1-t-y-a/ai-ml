package org.nci.fogedge;

import org.nci.fogedge.model.*;
import org.nci.fogedge.security.*;
import org.nci.fogedge.simulation.*;
import org.nci.fogedge.topology.*;
import org.nci.fogedge.utils.*;

import java.util.Calendar;
import java.util.List;

/**
 * Main class for the Secure Fog Computing Simulation
 * 
 * This class serves as the entry point for the simulation, setting up
 * the network topology, security features, and running the simulation.
 * It implements a proof-of-concept prototype for a secure fog computing
 * architecture based on the research paper "An Overview of Fog Computing
 * and Edge Computing Security and Privacy Issues" (MDPI, 2021).
 * 
 * @author NCI H9FEC Student
 * @version 1.0
 */
public class SecureFogSimulation {

    /**
     * Main method
     * @param args Command line arguments
     */
    public static void main(String[] args) {
        System.out.println("=======================================================");
        System.out.println("      SECURE FOG COMPUTING SIMULATION                  ");
        System.out.println("      Based on MDPI 2021 Research Paper                ");
        System.out.println("=======================================================");
        System.out.println();
        
        try {
            // 1. Create simulation parameters
            System.out.println("Setting up simulation parameters...");
            SimulationParameters parameters = createSimulationParameters();
            
            // 2. Build network topology
            System.out.println("Building network topology...");
            NetworkTopology topology = buildNetworkTopology();
            
            // Initialize security components
            System.out.println("Initializing security components...");
            org.nci.fogedge.security.SecurityManager securityManager = new org.nci.fogedge.security.SecurityManager();
            securityManager.enableEncryption(true);
            securityManager.enableIntrusionDetection(true);
            securityManager.enableAuthenticationScheme(org.nci.fogedge.security.SecurityManager.AuthScheme.MUTUAL_AUTHENTICATION);
            
            // Start simulation
            System.out.println("Starting simulation...");
            SimulationEngine engine = new SimulationEngine(parameters, topology, securityManager);
            engine.initialize();
            SimulationResults results = engine.runSimulation();
            
            // Display results
            System.out.println("\nSimulation completed successfully!");
            System.out.println("Results Summary:");
            System.out.println("----------------");
            System.out.println("Total IoT data packets generated: " + results.getTotalPacketsGenerated());
            System.out.println("Data packets processed at edge: " + results.getPacketsProcessedAtEdge());
            System.out.println("Data packets processed at fog: " + results.getPacketsProcessedAtFog());
            System.out.println("Data packets processed at cloud: " + results.getPacketsProcessedAtCloud());
            System.out.println("Security incidents detected: " + results.getSecurityIncidentsDetected());
            System.out.println("Security incidents mitigated: " + results.getSecurityIncidentsMitigated());
            System.out.println("Average processing latency (ms): " + results.getAverageLatency());
            System.out.println("Network bandwidth saved (MB): " + results.getBandwidthSaved());
            System.out.println("Energy consumption (kWh): " + results.getEnergyConsumption());
            
            // Generate detailed reports
            ReportGenerator reportGenerator = new ReportGenerator(results, topology, securityManager);
            reportGenerator.generateLatencyReport("latency_report.pdf");
            reportGenerator.generateSecurityReport("security_report.pdf");
            reportGenerator.generateEnergyReport("energy_report.pdf");
            
            System.out.println("\nDetailed reports generated successfully!");
            
        } catch (Exception e) {
            System.err.println("Error in simulation: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    /**
     * Create simulation parameters with default values
     * @return Simulation parameters
     */
    private static SimulationParameters createSimulationParameters() {
        SimulationParameters parameters = new SimulationParameters();
        
        // Set custom parameters for this simulation
        parameters.setSimulationLength(1800); // 30 minutes
        parameters.setIotDataGenerationRate(1.0); // 1 packet per second per device
        parameters.setSecurityOverhead(20); // 20% overhead for security processing
        
        return parameters;
    }
    
    /**
     * Build network topology for the simulation
     * @return Network topology
     */
    private static NetworkTopology buildNetworkTopology() {
        NetworkTopology topology = new NetworkTopology();
        
        // Add IoT devices
        for (int i = 1; i <= 10; i++) {
            IoTDevice device = new IoTDevice("iot-" + i, new Location(2.0 * i, 2.0), "sensor", 1.0);
            topology.addIoTDevice(device);
        }
        
        // Add edge nodes
        for (int i = 1; i <= 3; i++) {
            EdgeNode edge = new EdgeNode("edge-" + i, new Location(5.0 * i, 5.0), 1000.0, 2048.0);
            topology.addEdgeNode(edge);
        }
        
        // Add fog nodes
        for (int i = 1; i <= 2; i++) {
            FogNode fog = new FogNode("fog-" + i, new Location(10.0 * i, 10.0), 5000.0, 4096.0);
            topology.addFogNode(fog);
        }
        
        // Add cloud datacenter
        CloudDatacenter cloud = new CloudDatacenter("cloud-1", new Location(0.0, 0.0), 20000.0, 8192.0);
        topology.setCloudDatacenter(cloud);
        
        // Connect IoT devices to edge nodes (round-robin assignment)
        List<IoTDevice> iotDevices = topology.getIotDevices();
        List<EdgeNode> edgeNodes = topology.getEdgeNodes();
        for (int i = 0; i < iotDevices.size(); i++) {
            // Connect each IoT device to one edge node (round-robin)
            EdgeNode targetEdge = edgeNodes.get(i % edgeNodes.size());
            iotDevices.get(i).addConnectedEdgeNode(targetEdge);
            
            // For redundancy, also connect to a second edge node if available
            if (edgeNodes.size() > 1) {
                EdgeNode secondEdge = edgeNodes.get((i + 1) % edgeNodes.size());
                iotDevices.get(i).addConnectedEdgeNode(secondEdge);
            }
        }
        
        // Connect edge nodes to fog nodes (round-robin assignment)
        List<FogNode> fogNodes = topology.getFogNodes();
        for (int i = 0; i < edgeNodes.size(); i++) {
            // Connect each edge node to one fog node (round-robin)
            FogNode targetFog = fogNodes.get(i % fogNodes.size());
            edgeNodes.get(i).addConnectedFogNode(targetFog);
            
            // For redundancy, also connect to a second fog node if available
            if (fogNodes.size() > 1) {
                FogNode secondFog = fogNodes.get((i + 1) % fogNodes.size());
                edgeNodes.get(i).addConnectedFogNode(secondFog);
            }
        }
        
        // Connect fog nodes to cloud datacenter
        for (FogNode fog : fogNodes) {
            fog.setConnectedCloud(cloud);
        }
        
        return topology;
    }
}
