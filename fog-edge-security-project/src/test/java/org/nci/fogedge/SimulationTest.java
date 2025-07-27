package org.nci.fogedge;

import org.junit.Before;
import org.junit.Test;
import org.nci.fogedge.model.SimulationConfig;
import org.nci.fogedge.model.SimulationResults;
import org.nci.fogedge.security.AttackType;
import org.nci.fogedge.security.SecurityLevel;
import org.nci.fogedge.util.ConfigurationManager;
import org.nci.fogedge.util.LoggingUtil;

import java.util.Arrays;

import static org.junit.Assert.*;

/**
 * Test class for validating the Fog and Edge Computing Security Simulation
 * Based on the paper: "An Overview of Fog Computing and Edge Computing Security and Privacy Issues"
 * (Sensors 2021, 21, 8226, https://doi.org/10.3390/s21248226)
 */
public class SimulationTest {
    
    private FogEdgeSecuritySimulation simulation;
    private SimulationConfig testConfig;
    
    @Before
    public void setUp() {
        // Configure logging for tests
        LoggingUtil.configureLogging();
        
        // Create a test configuration
        testConfig = new SimulationConfig();
        testConfig.setNumIoTDevices(10); // Smaller number for faster tests
        testConfig.setNumEdgeNodes(3);
        testConfig.setNumFogNodes(1);
        testConfig.setSimulationSteps(10); // Fewer steps for faster tests
        testConfig.setSecurityLevel(SecurityLevel.MEDIUM);
        testConfig.setSecurityEnabledAtIoT(true);
        testConfig.setSecurityEnabledAtEdge(true);
        testConfig.setSecurityEnabledAtFog(true);
        testConfig.setAttackSimulationEnabled(true);
        testConfig.setAttackTypes(Arrays.asList(AttackType.values()));
        
        // Set the test configuration
        ConfigurationManager.setConfig(testConfig);
        
        // Create simulation instance
        simulation = new FogEdgeSecuritySimulation();
    }
    
    @Test
    public void testSimulationInitialization() {
        // Initialize the simulation
        simulation.initialize();
        
        // Verify initialization (indirect testing through results)
        SimulationResults results = simulation.getResults();
        assertNotNull("Results object should be initialized", results);
    }
    
    @Test
    public void testSimulationExecution() {
        // Initialize and run the simulation
        simulation.initialize();
        simulation.runSimulation();
        
        // Get results
        SimulationResults results = simulation.getResults();
        
        // Verify that results contain data
        assertNotNull("Results object should not be null", results);
        
        // Note: Since attacks are random, we can't assert specific values
        // but we can verify that the simulation ran and collected metrics
    }
    
    @Test
    public void testWithoutSecurityMeasures() {
        // Create config with security disabled
        SimulationConfig noSecurityConfig = new SimulationConfig();
        noSecurityConfig.setNumIoTDevices(10);
        noSecurityConfig.setNumEdgeNodes(3);
        noSecurityConfig.setNumFogNodes(1);
        noSecurityConfig.setSimulationSteps(10);
        noSecurityConfig.setSecurityLevel(SecurityLevel.LOW);
        noSecurityConfig.setSecurityEnabledAtIoT(false);
        noSecurityConfig.setSecurityEnabledAtEdge(false);
        noSecurityConfig.setSecurityEnabledAtFog(false);
        noSecurityConfig.setAttackSimulationEnabled(true);
        noSecurityConfig.setAttackTypes(Arrays.asList(AttackType.values()));
        
        // Set the no-security configuration
        ConfigurationManager.setConfig(noSecurityConfig);
        
        // Create and run simulation
        FogEdgeSecuritySimulation noSecuritySimulation = new FogEdgeSecuritySimulation();
        noSecuritySimulation.initialize();
        noSecuritySimulation.runSimulation();
        
        // Get results
        SimulationResults noSecurityResults = noSecuritySimulation.getResults();
        
        // Verify that results contain data
        assertNotNull("Results object should not be null", noSecurityResults);
    }
    
    @Test
    public void testWithHighSecurity() {
        // Create config with high security
        SimulationConfig highSecurityConfig = new SimulationConfig();
        highSecurityConfig.setNumIoTDevices(10);
        highSecurityConfig.setNumEdgeNodes(3);
        highSecurityConfig.setNumFogNodes(1);
        highSecurityConfig.setSimulationSteps(10);
        highSecurityConfig.setSecurityLevel(SecurityLevel.VERY_HIGH);
        highSecurityConfig.setSecurityEnabledAtIoT(true);
        highSecurityConfig.setSecurityEnabledAtEdge(true);
        highSecurityConfig.setSecurityEnabledAtFog(true);
        highSecurityConfig.setAttackSimulationEnabled(true);
        highSecurityConfig.setAttackTypes(Arrays.asList(AttackType.values()));
        
        // Set the high-security configuration
        ConfigurationManager.setConfig(highSecurityConfig);
        
        // Create and run simulation
        FogEdgeSecuritySimulation highSecuritySimulation = new FogEdgeSecuritySimulation();
        highSecuritySimulation.initialize();
        highSecuritySimulation.runSimulation();
        
        // Get results
        SimulationResults highSecurityResults = highSecuritySimulation.getResults();
        
        // Verify that results contain data
        assertNotNull("Results object should not be null", highSecurityResults);
    }
    
    @Test
    public void testWithoutAttacks() {
        // Create config without attacks
        SimulationConfig noAttacksConfig = new SimulationConfig();
        noAttacksConfig.setNumIoTDevices(10);
        noAttacksConfig.setNumEdgeNodes(3);
        noAttacksConfig.setNumFogNodes(1);
        noAttacksConfig.setSimulationSteps(10);
        noAttacksConfig.setSecurityLevel(SecurityLevel.MEDIUM);
        noAttacksConfig.setSecurityEnabledAtIoT(true);
        noAttacksConfig.setSecurityEnabledAtEdge(true);
        noAttacksConfig.setSecurityEnabledAtFog(true);
        noAttacksConfig.setAttackSimulationEnabled(false);
        
        // Set the no-attacks configuration
        ConfigurationManager.setConfig(noAttacksConfig);
        
        // Create and run simulation
        FogEdgeSecuritySimulation noAttacksSimulation = new FogEdgeSecuritySimulation();
        noAttacksSimulation.initialize();
        noAttacksSimulation.runSimulation();
        
        // Get results
        SimulationResults noAttacksResults = noAttacksSimulation.getResults();
        
        // Verify that results contain data
        assertNotNull("Results object should not be null", noAttacksResults);
    }
}
