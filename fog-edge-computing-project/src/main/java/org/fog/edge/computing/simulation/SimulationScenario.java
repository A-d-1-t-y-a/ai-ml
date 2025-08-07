package org.fog.edge.computing.simulation;

import java.util.ArrayList;
import java.util.List;

/**
 * SimulationScenario class for the Fog and Edge Computing project.
 * 
 * This simplified version works with CloudSim Plus and provides a basic
 * fog and edge computing scenario for demonstration purposes.
 * 
 * @author Student
 * @version 1.0
 */
public class SimulationScenario {
    
    /**
     * Default constructor for the SimulationScenario
     */
    public SimulationScenario() {
        // Simple constructor for basic scenario
    }
    
    /**
     * Initializes the simulation scenario
     */
    public void initialize() {
        // Initialize scenario components
        System.out.println("Initializing simulation scenario...");
        System.out.println("- Cloud datacenters: 2");
        System.out.println("- Edge nodes: 4");
        System.out.println("- IoT devices: 20");
        System.out.println("- Applications: 5");
    }
    
    /**
     * Gets the list of cloud datacenters in the simulation
     * 
     * @return List of cloud datacenters
     */
    public List<Object> getCloudDatacenters() {
        // Placeholder implementation
        List<Object> cloudDCs = new ArrayList<>();
        cloudDCs.add(new MockDatacenter("Cloud-DC-1"));
        cloudDCs.add(new MockDatacenter("Cloud-DC-2"));
        return cloudDCs;
    }
    
    /**
     * Gets the list of fog datacenters in the simulation
     * 
     * @return List of fog datacenters
     */
    public List<Object> getFogDatacenters() {
        // Placeholder implementation
        List<Object> fogDCs = new ArrayList<>();
        fogDCs.add(new MockDatacenter("Fog-DC-1"));
        fogDCs.add(new MockDatacenter("Fog-DC-2"));
        fogDCs.add(new MockDatacenter("Fog-DC-3"));
        return fogDCs;
    }
    
    /**
     * Gets the list of edge devices in the simulation
     * 
     * @return List of edge devices
     */
    public List<org.fog.edge.computing.orchestration.FuzzyDecisionTreeOrchestrator.DeviceInfo> getEdgeDevices() {
        // Create a list of mock edge devices
        List<org.fog.edge.computing.orchestration.FuzzyDecisionTreeOrchestrator.DeviceInfo> edgeDevices = new ArrayList<>();
        
        // Create a temporary instance of FuzzyDecisionTreeOrchestrator to access its inner classes
        org.fog.edge.computing.orchestration.FuzzyDecisionTreeOrchestrator orchestrator = 
            new org.fog.edge.computing.orchestration.FuzzyDecisionTreeOrchestrator();
        
        // Create 5 mock edge devices directly without reflection
        for (int i = 1; i <= 5; i++) {
            org.fog.edge.computing.orchestration.FuzzyDecisionTreeOrchestrator.MockDeviceInfo deviceInfo = 
                orchestrator.new MockDeviceInfo(i, "Edge-Device-" + i);
            edgeDevices.add(deviceInfo);
        }
        
        return edgeDevices;
    }
    
    /**
     * Mock implementation of a datacenter for testing purposes
     */
    public class MockDatacenter {
        private String name;
        private List<MockHost> hostList;
        
        public MockDatacenter(String name) {
            this.name = name;
            this.hostList = new ArrayList<>();
            // Add some mock hosts
            hostList.add(new MockHost(name + "-Host-1"));
            hostList.add(new MockHost(name + "-Host-2"));
        }
        
        public String getName() {
            return name;
        }
        
        public List<MockHost> getHostList() {
            return hostList;
        }
    }
    
    /**
     * Mock implementation of a host for testing purposes
     */
    public class MockHost {
        private String name;
        private List<MockVm> vmList;
        private List<MockVm> vmCreatedList;
        private double totalMips;
        private double availableMips;
        
        public MockHost(String name) {
            this.name = name;
            this.vmList = new ArrayList<>();
            this.vmCreatedList = new ArrayList<>();
            this.totalMips = 10000.0;
            this.availableMips = 5000.0;
            
            // Add some mock VMs
            MockVm vm1 = new MockVm(name + "-VM-1");
            MockVm vm2 = new MockVm(name + "-VM-2");
            vmList.add(vm1);
            vmCreatedList.add(vm1);
            vmCreatedList.add(vm2);
        }
        
        public String getName() {
            return name;
        }
        
        public List<MockVm> getVmList() {
            return vmList;
        }
        
        public List<MockVm> getVmCreatedList() {
            return vmCreatedList;
        }
        
        public double getTotalMipsCapacity() {
            return totalMips;
        }
        
        public double getAvailableMips() {
            return availableMips;
        }
    }
    
    /**
     * Mock implementation of a VM for testing purposes
     */
    public class MockVm {
        private String name;
        private int numberOfPes;
        private double mips;
        
        public MockVm(String name) {
            this.name = name;
            this.numberOfPes = 4;
            this.mips = 1000.0;
        }
        
        public String getName() {
            return name;
        }
        
        public int getNumberOfPes() {
            return numberOfPes;
        }
        
        public double getMips() {
            return mips;
        }
    }
}
