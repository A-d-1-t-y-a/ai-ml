package com.fog.eedto;

import static org.junit.Assert.*;

import java.util.ArrayList;
import java.util.List;

import org.junit.Before;
import org.junit.Test;

import com.fog.eedto.algorithm.EEDTOAlgorithm;
import com.fog.eedto.blockchain.BlockchainService;
import com.fog.eedto.model.CloudServer;
import com.fog.eedto.model.Device;
import com.fog.eedto.model.EdgeServer;
import com.fog.eedto.model.IoTDevice;
import com.fog.eedto.model.Task;

/**
 * Unit tests for the EEDTO implementation.
 */
public class EEDTOTest {
    private IoTDevice iotDevice;
    private EdgeServer edgeServer;
    private CloudServer cloudServer;
    private BlockchainService blockchainService;
    private EEDTOAlgorithm eedtoAlgorithm;
    private List<EdgeServer> edgeServers;
    private List<CloudServer> cloudServers;
    
    @Before
    public void setUp() {
        // Create IoT device
        iotDevice = new IoTDevice(1, "IoT-1", 800, 512, 2048, 20, 200, 15000, 0.2);
        
        // Create edge server
        edgeServer = new EdgeServer(1, "Edge-1", 8000, 4096, 102400, 200, 400, 15, 10, 15, 0.00001);
        
        // Create cloud server
        cloudServer = new CloudServer(1, "Cloud-1", 30000, 16384, 1048576, 800, 600, 80, 80, 150, 0.000008, 2.0);
        
        // Create blockchain service
        blockchainService = new BlockchainService(2);
        
        // Create EEDTO algorithm
        eedtoAlgorithm = new EEDTOAlgorithm(0.33, 0.33, 0.33, 0.2, 5, 3, blockchainService);
        
        // Create lists for servers
        edgeServers = new ArrayList<>();
        edgeServers.add(edgeServer);
        
        cloudServers = new ArrayList<>();
        cloudServers.add(cloudServer);
    }
    
    @Test
    public void testTaskCreation() {
        Task task = new Task(1, 5000, 1024 * 1024, 512 * 1024, 10, 0, Task.TaskType.MEDIUM);
        
        assertEquals(1, task.getId());
        assertEquals(5000, task.getLength());
        assertEquals(1024 * 1024, task.getInputSize());
        assertEquals(512 * 1024, task.getOutputSize());
        assertEquals(10, task.getDeadline(), 0.001);
        assertEquals(0, task.getArrivalTime(), 0.001);
        assertEquals(Task.TaskType.MEDIUM, task.getTaskType());
        assertEquals(Task.TaskStatus.CREATED, task.getStatus());
    }
    
    @Test
    public void testTaskExecutionTime() {
        Task task = new Task(1, 5000, 1024 * 1024, 512 * 1024, 10, 0, Task.TaskType.MEDIUM);
        
        // Execution time = task length / MIPS
        double expectedExecutionTime = 5000.0 / 800; // IoT device MIPS = 800
        assertEquals(expectedExecutionTime, task.calculateExecutionTime(iotDevice.getMips()), 0.001);
        
        expectedExecutionTime = 5000.0 / 8000; // Edge server MIPS = 8000
        assertEquals(expectedExecutionTime, task.calculateExecutionTime(edgeServer.getMips()), 0.001);
        
        expectedExecutionTime = 5000.0 / 30000; // Cloud server MIPS = 30000
        assertEquals(expectedExecutionTime, task.calculateExecutionTime(cloudServer.getMips()), 0.001);
    }
    
    @Test
    public void testIoTDeviceCanExecuteTask() {
        // Lightweight task that IoT device can execute
        Task lightTask = new Task(1, 1000, 100 * 1024, 50 * 1024, 10, 0, Task.TaskType.LIGHTWEIGHT);
        assertTrue(iotDevice.canExecuteTask(lightTask));
        
        // Intensive task that exceeds IoT device capabilities
        Task heavyTask = new Task(2, 9000, 600 * 1024, 300 * 1024, 10, 0, Task.TaskType.INTENSIVE);
        assertTrue(iotDevice.canExecuteTask(heavyTask)); // Should still be able to execute, just slowly
    }
    
    @Test
    public void testEdgeServerCanExecuteTask() {
        Task task = new Task(1, 5000, 1024 * 1024, 512 * 1024, 10, 0, Task.TaskType.MEDIUM);
        assertTrue(edgeServer.canExecuteTask(task));
        
        // Add multiple tasks to reach capacity
        for (int i = 0; i < edgeServer.getMaxConcurrentTasks(); i++) {
            Task t = new Task(i + 2, 1000, 100 * 1024, 50 * 1024, 10, 0, Task.TaskType.LIGHTWEIGHT);
            assertTrue(edgeServer.addActiveTask(t));
        }
        
        // Now the server should be at capacity
        assertFalse(edgeServer.canExecuteTask(task));
    }
    
    @Test
    public void testCloudServerCanExecuteTask() {
        Task task = new Task(1, 20000, 2048 * 1024, 1024 * 1024, 10, 0, Task.TaskType.INTENSIVE);
        assertTrue(cloudServer.canExecuteTask(task));
        
        // Add multiple tasks to reach capacity
        for (int i = 0; i < cloudServer.getMaxConcurrentTasks(); i++) {
            Task t = new Task(i + 2, 1000, 100 * 1024, 50 * 1024, 10, 0, Task.TaskType.LIGHTWEIGHT);
            assertTrue(cloudServer.addActiveTask(t));
        }
        
        // Now the server should be at capacity
        assertFalse(cloudServer.canExecuteTask(task));
    }
    
    @Test
    public void testBlockchainFunctionality() {
        Task task = new Task(1, 5000, 1024 * 1024, 512 * 1024, 10, 0, Task.TaskType.MEDIUM);
        
        // Add a transaction
        String transactionId = blockchainService.addTaskOffloadingTransaction(task, iotDevice, edgeServer);
        assertNotNull(transactionId);
        assertEquals(1, blockchainService.getPendingTransactionsCount());
        
        // Mine pending transactions
        boolean mined = blockchainService.minePendingTransactions();
        assertTrue(mined);
        assertEquals(0, blockchainService.getPendingTransactionsCount());
        assertEquals(2, blockchainService.getBlockchainSize()); // Genesis block + 1 new block
        
        // Verify blockchain integrity
        assertTrue(blockchainService.isChainValid());
    }
    
    @Test
    public void testEEDTOAlgorithmOffloadingDecision() {
        // Lightweight task
        Task lightTask = new Task(1, 1000, 100 * 1024, 50 * 1024, 10, 0, Task.TaskType.LIGHTWEIGHT);
        Device selectedDevice = eedtoAlgorithm.makeOffloadingDecision(lightTask, iotDevice, edgeServers, cloudServers, 0);
        assertNotNull(selectedDevice);
        // Lightweight tasks should typically be executed locally
        assertEquals(iotDevice, selectedDevice);
        
        // Intensive task
        Task heavyTask = new Task(2, 9000, 600 * 1024, 300 * 1024, 10, 0, Task.TaskType.INTENSIVE);
        selectedDevice = eedtoAlgorithm.makeOffloadingDecision(heavyTask, iotDevice, edgeServers, cloudServers, 0);
        assertNotNull(selectedDevice);
        // Intensive tasks should typically be offloaded
        assertNotEquals(iotDevice, selectedDevice);
        
        // Task with tight deadline
        Task urgentTask = new Task(3, 5000, 400 * 1024, 200 * 1024, 1, 0, Task.TaskType.MEDIUM);
        selectedDevice = eedtoAlgorithm.makeOffloadingDecision(urgentTask, iotDevice, edgeServers, cloudServers, 0);
        assertNotNull(selectedDevice);
        // Urgent tasks should be offloaded to faster servers
        assertNotEquals(iotDevice, selectedDevice);
    }
}
