package org.nci.fogedge.test;

import org.apache.log4j.Level;
import org.cloudbus.cloudsim.Log;
import org.junit.Before;
import org.junit.Test;
import org.nci.fogedge.SecureFogSimulation;
import org.nci.fogedge.model.SimulationResults;
import org.nci.fogedge.security.SecurityManager;
import org.nci.fogedge.security.SecurityLevel;
import org.nci.fogedge.topology.EdgeNode;
import org.nci.fogedge.topology.FogNode;
import org.nci.fogedge.topology.IoTDevice;
import org.nci.fogedge.util.DataProcessor;
import org.nci.fogedge.util.LoggingUtil;

import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.*;

/**
 * Test class for the Secure Fog Computing Framework.
 * Tests the basic functionality of the simulation components.
 */
public class SimulationTest {

    private SecurityManager securityManager;
    private FogNode fogNode;
    private EdgeNode edgeNode;
    private IoTDevice iotDevice;
    
    @Before
    public void setUp() {
        // Initialize logging
        LoggingUtil.initializeLogging(false, Level.INFO);
        
        // Create security manager with security enabled
        securityManager = new SecurityManager(true);
        
        // Create a simple topology for testing
        fogNode = new FogNode("fog-1", securityManager);
        edgeNode = new EdgeNode("edge-1", fogNode, securityManager);
        iotDevice = new IoTDevice("iot-1", IoTDevice.WirelessType.WIFI, edgeNode, securityManager);
    }
    
    @Test
    public void testDataFlow() {
        // Generate test data
        byte[] testData = "Test data for secure fog computing simulation".getBytes();
        
        // Process data through the topology
        iotDevice.generateAndSendData(testData.length);
        
        // Verify data flow
        assertTrue("IoT device should generate data", iotDevice.getTotalDataGenerated() > 0);
        assertTrue("IoT device should consume energy", iotDevice.getEnergyConsumption() > 0);
        
        // If security is enabled, security overhead should be > 0
        if (securityManager.isSecurityEnabled()) {
            assertTrue("Security overhead should be positive", iotDevice.getSecurityOverhead() > 0);
        }
    }
    
    @Test
    public void testSecurityOperations() {
        // Test encryption and decryption
        String testMessage = "Secure message for testing";
        byte[] originalData = testMessage.getBytes();
        
        // Encrypt with HIGH security level
        byte[] encryptedData = securityManager.encryptData(originalData, SecurityLevel.HIGH);
        
        // Data should be different after encryption
        assertFalse("Encrypted data should differ from original", 
                java.util.Arrays.equals(originalData, encryptedData));
        
        // Decrypt
        byte[] decryptedData = securityManager.decryptData(encryptedData);
        
        // Data should match original after decryption
        assertTrue("Decrypted data should match original", 
                java.util.Arrays.equals(originalData, decryptedData));
    }
    
    @Test
    public void testAuthentication() {
        String deviceId = "test-device";
        
        // Generate challenge
        byte[] challenge = securityManager.generateChallenge();
        
        // Generate correct response
        byte[] correctResponse = securityManager.generateResponse(deviceId, challenge);
        
        // Authentication should succeed with correct response
        assertTrue("Authentication should succeed with correct response", 
                securityManager.authenticate(deviceId, challenge, correctResponse));
        
        // Generate incorrect response
        byte[] incorrectResponse = new byte[correctResponse.length];
        System.arraycopy(correctResponse, 0, incorrectResponse, 0, correctResponse.length);
        incorrectResponse[0] = (byte) (incorrectResponse[0] + 1); // Modify one byte
        
        // Authentication should fail with incorrect response
        assertFalse("Authentication should fail with incorrect response", 
                securityManager.authenticate(deviceId, challenge, incorrectResponse));
    }
    
    @Test
    public void testDataProcessor() {
        // Generate test data
        byte[] testData = new byte[1000]; // 1KB of data
        for (int i = 0; i < testData.length; i++) {
            testData[i] = (byte) (i % 256);
        }
        
        // Process at edge level
        byte[] edgeProcessed = DataProcessor.processDataAtEdge(testData);
        
        // Edge processing should reduce data size
        assertTrue("Edge processing should reduce data size", edgeProcessed.length < testData.length);
        
        // Process at fog level
        byte[] fogProcessed = DataProcessor.processDataAtFog(edgeProcessed);
        
        // Fog processing should produce a fixed-size summary
        assertEquals("Fog processing should produce a fixed-size summary", 100, fogProcessed.length);
    }
    
    @Test
    public void testSimulationResults() {
        // Create simulation results
        SimulationResults results = new SimulationResults(true);
        
        // Create test topology
        List<IoTDevice> iotDevices = new ArrayList<>();
        List<EdgeNode> edgeNodes = new ArrayList<>();
        List<FogNode> fogNodes = new ArrayList<>();
        
        // Add test devices
        FogNode testFog = new FogNode("test-fog", securityManager);
        fogNodes.add(testFog);
        
        EdgeNode testEdge = new EdgeNode("test-edge", testFog, securityManager);
        edgeNodes.add(testEdge);
        
        IoTDevice testDevice = new IoTDevice("test-iot", IoTDevice.WirelessType.WIFI, testEdge, securityManager);
        iotDevices.add(testDevice);
        
        // Generate some test data
        testDevice.generateAndSendData(1024); // 1KB
        
        // Collect metrics
        results.collectIoTMetrics(iotDevices);
        results.collectEdgeMetrics(edgeNodes);
        results.collectFogMetrics(fogNodes);
        
        // Verify results
        assertTrue("Total data generated should be positive", results.getTotalDataGenerated() > 0);
        assertTrue("Total energy consumption should be positive", results.getTotalEnergyConsumption() > 0);
        
        // Generate report
        String report = results.generateDetailedReport();
        assertNotNull("Report should not be null", report);
        assertTrue("Report should contain configuration section", report.contains("## Configuration"));
    }
    
    @Test
    public void testFullSimulation() {
        // Run a small simulation
        SecureFogSimulation simulation = new SecureFogSimulation();
        simulation.startSimulation(true, 5, 2, 1); // 5 IoT devices, 2 edge nodes, 1 fog node
        
        // Get results
        SimulationResults results = simulation.getResults();
        assertNotNull("Simulation results should not be null", results);
        
        // Verify simulation ran correctly
        assertEquals("Should have correct number of IoT devices", 5, results.getTotalIoTDevices());
        assertEquals("Should have correct number of edge nodes", 2, results.getTotalEdgeNodes());
        assertEquals("Should have correct number of fog nodes", 1, results.getTotalFogNodes());
        assertTrue("Should have generated data", results.getTotalDataGenerated() > 0);
    }
}
