package com.fog.eedto;

import java.util.logging.Logger;
import java.util.logging.Level;

import com.fog.eedto.model.Task;
import com.fog.eedto.model.IoTDevice;
import com.fog.eedto.model.EdgeServer;
import com.fog.eedto.model.CloudServer;
import com.fog.eedto.util.ConfigurationManager;

/**
 * Simple test class to verify that the model classes compile correctly
 */
public class TestMain {
    private static final Logger logger = Logger.getLogger(TestMain.class.getName());
    
    public static void main(String[] args) {
        logger.info("Starting EEDTO test");
        
        try {
            // Initialize configuration
            if (!ConfigurationManager.initialize()) {
                logger.severe("Failed to initialize configuration. Exiting.");
                return;
            }
            
            // Create test objects to verify compilation
            IoTDevice iotDevice = new IoTDevice(
                1, "Test IoT Device", 1000, 512, 1024, 10, 200, 5000, 0.5
            );
            
            EdgeServer edgeServer = new EdgeServer(
                1, "Test Edge Server", 5000, 4096, 1024 * 1024, 100, 100,
                200, 10, 20, 0.1
            );
            
            CloudServer cloudServer = new CloudServer(
                1, "Test Cloud Server", 20000, 16384, 10 * 1024 * 1024, 1000, 80,
                500, 100, 100, 0.5, 2.0
            );
            
            Task task = new Task(
                1, 1000, 10240, 5120, 10.0, 0.0, Task.TaskType.INTENSIVE
            );
            
            logger.info("Created test objects successfully:");
            logger.info("IoT Device: " + iotDevice);
            logger.info("Edge Server: " + edgeServer);
            logger.info("Cloud Server: " + cloudServer);
            logger.info("Task: " + task);
            
            logger.info("Test completed successfully!");
            
        } catch (Exception e) {
            logger.log(Level.SEVERE, "Error in test: " + e.getMessage(), e);
        }
    }
}
