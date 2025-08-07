package org.fog.edge.computing.orchestration;

import static org.junit.Assert.*;

import org.fog.edge.computing.simulation.SimulationScenario;
import org.fog.edge.computing.utils.SimulationParameters;
import org.junit.Before;
import org.junit.Test;

/**
 * Unit tests for OrchestratorComparison
 * 
 * This test class validates the orchestrator comparison framework,
 * ensuring it can properly compare different algorithms and generate
 * meaningful performance metrics.
 * 
 * @author Student
 * @version 1.0
 */
public class OrchestratorComparisonTest {
    
    private OrchestratorComparison comparison;
    private SimulationScenario scenario;
    private SimulationParameters parameters;
    
    @Before
    public void setUp() {
        scenario = new SimulationScenario();
        parameters = new SimulationParameters();
        comparison = new OrchestratorComparison(scenario, parameters);
    }
    
    @Test
    public void testComparisonInitialization() {
        // Test that the comparison framework initializes correctly
        assertNotNull("Comparison object should not be null", comparison);
    }
    
    @Test
    public void testRunComparison() {
        // Test running comparison with a small number of tasks
        try {
            OrchestratorComparison.ComparisonResults results = comparison.runComparison(10);
            assertNotNull("Comparison results should not be null", results);
            assertNotNull("Results map should not be null", results.getResults());
            assertTrue("Should have results for multiple algorithms", results.getResults().size() > 0);
        } catch (Exception e) {
            fail("Comparison should run without exceptions: " + e.getMessage());
        }
    }
    
    @Test
    public void testPerformanceMetrics() {
        // Test performance metrics calculation
        OrchestratorComparison.PerformanceMetrics metrics = 
            new OrchestratorComparison.PerformanceMetrics();
        
        // Test setters and getters
        metrics.setCloudTasksPercentage(30.0);
        metrics.setFogTasksPercentage(40.0);
        metrics.setMistTasksPercentage(30.0);
        metrics.setAverageDecisionTime(1.5);
        metrics.setTotalEnergyConsumption(100.0);
        metrics.setTaskSuccessRate(95.0);
        
        assertEquals("Cloud tasks percentage should be set correctly", 30.0, metrics.getCloudTasksPercentage(), 0.01);
        assertEquals("Fog tasks percentage should be set correctly", 40.0, metrics.getFogTasksPercentage(), 0.01);
        assertEquals("Mist tasks percentage should be set correctly", 30.0, metrics.getMistTasksPercentage(), 0.01);
        assertEquals("Average decision time should be set correctly", 1.5, metrics.getAverageDecisionTime(), 0.01);
        assertEquals("Total energy consumption should be set correctly", 100.0, metrics.getTotalEnergyConsumption(), 0.01);
        assertEquals("Task success rate should be set correctly", 95.0, metrics.getTaskSuccessRate(), 0.01);
    }
    
    @Test
    public void testComparisonWithDifferentTaskCounts() {
        // Test comparison with different numbers of tasks
        try {
            // Small comparison
            OrchestratorComparison.ComparisonResults smallResults = comparison.runComparison(5);
            assertNotNull("Small comparison results should not be null", smallResults);
            
            // Medium comparison
            OrchestratorComparison.ComparisonResults mediumResults = comparison.runComparison(20);
            assertNotNull("Medium comparison results should not be null", mediumResults);
            
            // Both should have the same number of algorithms
            assertEquals("Both comparisons should test the same algorithms", 
                        smallResults.getResults().size(), mediumResults.getResults().size());
            
        } catch (Exception e) {
            fail("Comparisons with different task counts should work: " + e.getMessage());
        }
    }
}
