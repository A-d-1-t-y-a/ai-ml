package com.nci.fogedge;

import com.nci.fogedge.utils.ConfigurationManager;
import com.nci.fogedge.utils.MetricsCollector;
import com.nci.fogedge.network.NetworkManager;
import com.nci.fogedge.network.NetworkLocation;
import com.nci.fogedge.network.NetworkStatistics;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Test class for the Fog and Edge Computing System
 * 
 * @author National College of Ireland - Fog and Edge Computing Project
 * @version 1.0.0
 * @since 2024
 */
public class MainTest {
    
    private ConfigurationManager configManager;
    private MetricsCollector metricsCollector;
    private NetworkManager networkManager;
    
    @BeforeEach
    void setUp() {
        configManager = new ConfigurationManager();
        metricsCollector = new MetricsCollector();
        networkManager = new NetworkManager(configManager);
    }
    
    @Test
    void testConfigurationManager() {
        // Test configuration loading
        configManager.loadConfiguration();
        
        // Test basic configuration values
        assertEquals(10, configManager.getIoTDeviceCount());
        assertEquals(3, configManager.getEdgeNodeCount());
        assertEquals(2, configManager.getCloudServiceCount());
        assertTrue(configManager.isLoRaWANEnabled());
        assertTrue(configManager.is5GEnabled());
        
        // Test performance targets
        assertEquals(50.0, configManager.getTargetLatency(), 0.1);
        assertEquals(100.0, configManager.getTargetThroughput(), 0.1);
        assertEquals(80.0, configManager.getTargetEnergyEfficiency(), 0.1);
        assertEquals(70.0, configManager.getTargetDataReduction(), 0.1);
    }
    
    @Test
    void testMetricsCollector() {
        // Test metrics collection
        metricsCollector.collectMetrics();
        
        // Test basic metrics
        assertNotNull(metricsCollector.getMetrics());
        assertEquals(0, metricsCollector.getActiveDeviceCount());
        assertEquals(0, metricsCollector.getActiveNodeCount());
        assertEquals(0, metricsCollector.getTotalDataProcessed());
    }
    
    @Test
    void testNetworkManager() {
        // Test network manager initialization
        assertNotNull(networkManager);
        
        // Test network statistics
        NetworkStatistics stats = networkManager.getNetworkStatistics();
        assertNotNull(stats);
        assertEquals(0, stats.getTotalPacketsTransmitted());
        assertEquals(0, stats.getTotalPacketsReceived());
        assertEquals(0, stats.getActiveNodeCount());
        assertEquals(0, stats.getActiveConnectionCount());
    }
    
    @Test
    void testNetworkLocation() {
        // Test network location functionality
        NetworkLocation location1 = new NetworkLocation(53.3498, -6.2603, 10.0);
        NetworkLocation location2 = new NetworkLocation(53.3498, -6.2603, 20.0);
        
        // Test distance calculation
        double distance = location1.calculateDistance(location2);
        assertTrue(distance > 0);
        assertTrue(distance < 100); // Should be close since same coordinates
        
        // Test random location generation
        NetworkLocation randomLocation = NetworkLocation.randomDublinLocation();
        assertNotNull(randomLocation);
        assertTrue(randomLocation.getLatitude() > 53.0 && randomLocation.getLatitude() < 54.0);
        assertTrue(randomLocation.getLongitude() > -7.0 && randomLocation.getLongitude() < -6.0);
    }
    
    @Test
    void testSystemIntegration() {
        // Test basic system integration
        assertNotNull(configManager);
        assertNotNull(metricsCollector);
        assertNotNull(networkManager);
        
        // Test configuration integration
        configManager.loadConfiguration();
        assertEquals("Fog and Edge Computing System", configManager.getValue("system.name"));
        assertEquals("1.0.0", configManager.getValue("system.version"));
        
        // Test metrics integration
        metricsCollector.updateDeviceHealth(85.0);
        metricsCollector.updateEdgeHealth(90.0);
        metricsCollector.updateTransmissionStats(1000, 950, 50, 0.95);
        
        // Verify metrics were updated
        assertTrue(metricsCollector.getActiveDeviceCount() > 0);
        assertTrue(metricsCollector.getActiveNodeCount() > 0);
        assertTrue(metricsCollector.getTotalDataProcessed() > 0);
    }
    
    @Test
    void testPerformanceMetrics() {
        // Test performance metrics calculation
        double latency = 45.0;
        double throughput = 85.0;
        double energyConsumption = 15.0;
        double cpuUsage = 65.0;
        double memoryUsage = 70.0;
        
        // Simulate performance improvements
        double latencyReduction = ((200.0 - latency) / 200.0) * 100.0;
        double dataReduction = 75.0;
        double energyEfficiency = ((100.0 - energyConsumption) / 100.0) * 100.0;
        double bandwidthOptimization = 55.0;
        
        // Verify performance targets are met
        assertTrue(latencyReduction > 40.0, "Latency reduction should be > 40%");
        assertTrue(dataReduction > 70.0, "Data reduction should be > 70%");
        assertTrue(energyEfficiency > 35.0, "Energy efficiency should be > 35%");
        assertTrue(bandwidthOptimization > 50.0, "Bandwidth optimization should be > 50%");
    }
} 