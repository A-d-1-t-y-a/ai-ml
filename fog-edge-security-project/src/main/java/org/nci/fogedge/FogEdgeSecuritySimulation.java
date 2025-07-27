package org.nci.fogedge;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.nci.fogedge.model.SimulationConfig;
import org.nci.fogedge.model.SimulationResults;
import org.nci.fogedge.security.AttackSimulator;
import org.nci.fogedge.security.SecurityManager;
import org.nci.fogedge.topology.EdgeNode;
import org.nci.fogedge.topology.FogNode;
import org.nci.fogedge.topology.IoTDevice;
import org.nci.fogedge.topology.NetworkTopology;
import org.nci.fogedge.util.ConfigurationManager;
import org.nci.fogedge.util.DataProcessor;
import org.nci.fogedge.util.LoggingUtil;

import java.util.ArrayList;
import java.util.List;

/**
 * Main simulation class for the Fog and Edge Computing Security Framework
 * Based on the paper: "An Overview of Fog Computing and Edge Computing Security and Privacy Issues"
 * (Sensors 2021, 21, 8226, https://doi.org/10.3390/s21248226)
 */
public class FogEdgeSecuritySimulation {
    private static final Logger logger = LogManager.getLogger(FogEdgeSecuritySimulation.class);
    
    private SimulationConfig config;
    private NetworkTopology topology;
    private SecurityManager securityManager;
    private AttackSimulator attackSimulator;
    private SimulationResults results;
    
    public FogEdgeSecuritySimulation() {
        this.config = ConfigurationManager.loadConfiguration();
        this.topology = new NetworkTopology();
        this.securityManager = new SecurityManager();
        this.attackSimulator = new AttackSimulator();
        this.results = new SimulationResults();
    }
    
    public void initialize() {
        logger.info("Initializing Fog and Edge Computing Security Simulation");
        
        // Create IoT devices
        for (int i = 0; i < config.getNumIoTDevices(); i++) {
            IoTDevice device = new IoTDevice("IoT-" + i, config.getRandomWirelessType());
            topology.addIoTDevice(device);
        }
        
        // Create edge nodes
        for (int i = 0; i < config.getNumEdgeNodes(); i++) {
            EdgeNode edgeNode = new EdgeNode("Edge-" + i);
            topology.addEdgeNode(edgeNode);
        }
        
        // Create fog nodes
        for (int i = 0; i < config.getNumFogNodes(); i++) {
            FogNode fogNode = new FogNode("Fog-" + i);
            topology.addFogNode(fogNode);
        }
        
        // Connect devices to edge nodes
        topology.connectDevicesToEdgeNodes();
        
        // Connect edge nodes to fog nodes
        topology.connectEdgeNodesToFogNodes();
        
        // Initialize security manager
        securityManager.initialize(topology, config.getSecurityLevel());
        
        // Initialize attack simulator with attack types from the paper
        attackSimulator.initialize(topology, config.getAttackTypes());
        
        logger.info("Simulation initialized with {} IoT devices, {} edge nodes, and {} fog nodes",
                config.getNumIoTDevices(), config.getNumEdgeNodes(), config.getNumFogNodes());
    }
    
    public void runSimulation() {
        logger.info("Starting Fog and Edge Computing Security Simulation");
        
        // Run for the configured number of simulation steps
        for (int step = 0; step < config.getSimulationSteps(); step++) {
            logger.info("Simulation step {}/{}", step + 1, config.getSimulationSteps());
            
            // Generate data from IoT devices
            List<Object> generatedData = generateIoTData();
            
            // Process data at edge nodes
            List<Object> processedDataAtEdge = processDataAtEdge(generatedData);
            
            // Process data at fog nodes
            List<Object> processedDataAtFog = processDataAtFog(processedDataAtEdge);
            
            // Simulate attacks based on the paper's attack types
            simulateAttacks(step);
            
            // Apply security measures
            applySecurityMeasures(step);
            
            // Collect metrics
            collectMetrics(step);
        }
        
        // Finalize results
        results.finalizeResults();
        
        logger.info("Simulation completed successfully");
    }
    
    private List<Object> generateIoTData() {
        List<Object> generatedData = new ArrayList<>();
        
        for (IoTDevice device : topology.getIoTDevices()) {
            Object data = device.generateData();
            generatedData.add(data);
            
            // Apply security at IoT level if configured
            if (config.isSecurityEnabledAtIoT()) {
                securityManager.secureIoTData(device, data);
            }
        }
        
        logger.info("Generated data from {} IoT devices", topology.getIoTDevices().size());
        return generatedData;
    }
    
    private List<Object> processDataAtEdge(List<Object> generatedData) {
        List<Object> processedData = new ArrayList<>();
        
        // Distribute data to edge nodes based on topology
        for (EdgeNode edgeNode : topology.getEdgeNodes()) {
            List<Object> nodeData = topology.getDataForEdgeNode(edgeNode, generatedData);
            
            // Apply security at Edge level
            if (config.isSecurityEnabledAtEdge()) {
                securityManager.secureEdgeProcessing(edgeNode, nodeData);
            }
            
            // Process data (filter, aggregate)
            Object processed = edgeNode.processData(nodeData);
            processedData.add(processed);
        }
        
        logger.info("Processed data at {} edge nodes", topology.getEdgeNodes().size());
        return processedData;
    }
    
    private List<Object> processDataAtFog(List<Object> edgeData) {
        List<Object> processedData = new ArrayList<>();
        
        // Distribute data to fog nodes based on topology
        for (FogNode fogNode : topology.getFogNodes()) {
            List<Object> nodeData = topology.getDataForFogNode(fogNode, edgeData);
            
            // Apply security at Fog level
            if (config.isSecurityEnabledAtFog()) {
                securityManager.secureFogProcessing(fogNode, nodeData);
            }
            
            // Process data (advanced analytics)
            Object processed = fogNode.processData(nodeData);
            processedData.add(processed);
        }
        
        logger.info("Processed data at {} fog nodes", topology.getFogNodes().size());
        return processedData;
    }
    
    private void simulateAttacks(int step) {
        // Simulate attacks based on the paper's attack types
        if (config.isAttackSimulationEnabled()) {
            attackSimulator.simulateAttacks(step);
            
            // Log attack details
            logger.info("Simulated attacks: {}", attackSimulator.getActiveAttacks());
        }
    }
    
    private void applySecurityMeasures(int step) {
        // Apply security countermeasures based on the paper
        securityManager.applyCountermeasures(attackSimulator.getActiveAttacks(), step);
        
        // Log security measures
        logger.info("Applied security countermeasures");
    }
    
    private void collectMetrics(int step) {
        // Collect performance metrics
        results.collectPerformanceMetrics(topology, step);
        
        // Collect security metrics
        results.collectSecurityMetrics(securityManager, attackSimulator, step);
        
        // Collect energy consumption metrics
        results.collectEnergyMetrics(topology, securityManager, step);
        
        logger.info("Collected metrics for step {}", step);
    }
    
    public SimulationResults getResults() {
        return results;
    }
    
    public void printResults() {
        results.printResults();
    }
    
    public static void main(String[] args) {
        // Configure logging
        LoggingUtil.configureLogging();
        
        // Create and run simulation
        FogEdgeSecuritySimulation simulation = new FogEdgeSecuritySimulation();
        simulation.initialize();
        simulation.runSimulation();
        simulation.printResults();
    }
}
