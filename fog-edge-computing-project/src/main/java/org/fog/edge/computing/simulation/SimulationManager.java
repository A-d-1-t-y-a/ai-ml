package org.fog.edge.computing.simulation;

import java.util.ArrayList;
import java.util.Calendar;
import java.util.List;

import org.cloudsimplus.brokers.DatacenterBroker;
import org.cloudsimplus.brokers.DatacenterBrokerSimple;
import org.cloudsimplus.builders.tables.CloudletsTableBuilder;
import org.cloudsimplus.cloudlets.Cloudlet;
import org.cloudsimplus.cloudlets.CloudletSimple;
import org.cloudsimplus.core.CloudSim;
import org.cloudsimplus.datacenters.Datacenter;
import org.cloudsimplus.datacenters.DatacenterSimple;
import org.cloudsimplus.hosts.Host;
import org.cloudsimplus.hosts.HostSimple;
import org.cloudsimplus.resources.Pe;
import org.cloudsimplus.resources.PeSimple;
import org.cloudsimplus.schedulers.cloudlet.CloudletSchedulerTimeShared;
import org.cloudsimplus.schedulers.vm.VmSchedulerTimeShared;
import org.cloudsimplus.utilizationmodels.UtilizationModel;
import org.cloudsimplus.utilizationmodels.UtilizationModelFull;
import org.cloudsimplus.vms.Vm;
import org.cloudsimplus.vms.VmSimple;
import org.fog.edge.computing.utils.SimulationResults;

/**
 * Manages the simulation lifecycle including initialization, execution, and result collection.
 * This class serves as the central coordinator for the CloudSim Plus-based simulation,
 * handling the initialization of CloudSim Plus, setting up the simulation scenario,
 * and managing the simulation execution.
 * 
 * Migrated from PureEdgeSim to use CloudSim Plus directly for better stability.
 * 
 * @author Student
 * @version 1.0
 */
public class SimulationManager {
    /**
     * Output directory path for simulation results
     */
    private String outputFolder;
    
    /**
     * CloudSim Plus simulation engine
     */
    private CloudSim simulation;
    
    /**
     * Results collector for the simulation
     */
    private SimulationResults simulationResults;
    
    /**
     * Constructor for the SimulationManager
     * 
     * @param outputFolder Output directory path for storing simulation results
     */
    public SimulationManager(String outputFolder) {
        this.outputFolder = outputFolder;
        this.simulation = new CloudSim();
        this.simulationResults = new SimulationResults(outputFolder);
    }
    

    
    /**
     * Starts the simulation with the configured settings
     * 
     * This method orchestrates the complete simulation lifecycle:
     * 
     * 1. Creates cloud and edge datacenters
     * 2. Creates VMs and cloudlets (tasks)
     * 3. Runs the simulation
     * 4. Processes and saves results
     * 
     * @throws Exception if there's an error during simulation execution
     */
    public void startSimulation() throws Exception {
        System.out.println("Creating simulation scenario...");
        
        // Create datacenters (Cloud and Edge)
        Datacenter cloudDatacenter = createCloudDatacenter();
        Datacenter edgeDatacenter = createEdgeDatacenter();
        
        // Create broker
        DatacenterBroker broker = new DatacenterBrokerSimple(simulation);
        
        // Create VMs
        List<Vm> vmList = createVms();
        broker.submitVmList(vmList);
        
        // Create Cloudlets (tasks)
        List<Cloudlet> cloudletList = createCloudlets();
        broker.submitCloudletList(cloudletList);
        
        // Start simulation
        System.out.println("Starting CloudSim Plus simulation...");
        simulation.start();
        
        // Process results
        processSimulationResults(broker);
        
        System.out.println("Simulation finished!");
    }
    
    /**
     * Creates a cloud datacenter with high-performance hosts
     */
    private Datacenter createCloudDatacenter() {
        List<Host> hostList = new ArrayList<>();
        
        // Create cloud hosts with high specifications
        for (int i = 0; i < 2; i++) {
            List<Pe> peList = new ArrayList<>();
            for (int j = 0; j < 8; j++) {
                peList.add(new PeSimple(10000)); // 10000 MIPS per PE
            }
            
            Host host = new HostSimple(32768, 1000000, 10000000, peList)
                .setVmScheduler(new VmSchedulerTimeShared());
            hostList.add(host);
        }
        
        return new DatacenterSimple(simulation, hostList);
    }
    
    /**
     * Creates an edge datacenter with moderate performance hosts
     */
    private Datacenter createEdgeDatacenter() {
        List<Host> hostList = new ArrayList<>();
        
        // Create edge hosts with moderate specifications
        for (int i = 0; i < 4; i++) {
            List<Pe> peList = new ArrayList<>();
            for (int j = 0; j < 4; j++) {
                peList.add(new PeSimple(5000)); // 5000 MIPS per PE
            }
            
            Host host = new HostSimple(16384, 500000, 5000000, peList)
                .setVmScheduler(new VmSchedulerTimeShared());
            hostList.add(host);
        }
        
        return new DatacenterSimple(simulation, hostList);
    }
    
    /**
     * Creates VMs for the simulation
     */
    private List<Vm> createVms() {
        List<Vm> vmList = new ArrayList<>();
        
        // Create cloud VMs
        for (int i = 0; i < 4; i++) {
            Vm vm = new VmSimple(i, 2000, 2) // 2000 MIPS, 2 PEs
                .setRam(4096).setBw(1000).setSize(100000)
                .setCloudletScheduler(new CloudletSchedulerTimeShared());
            vmList.add(vm);
        }
        
        // Create edge VMs
        for (int i = 4; i < 8; i++) {
            Vm vm = new VmSimple(i, 1000, 2) // 1000 MIPS, 2 PEs
                .setRam(2048).setBw(500).setSize(50000)
                .setCloudletScheduler(new CloudletSchedulerTimeShared());
            vmList.add(vm);
        }
        
        return vmList;
    }
    
    /**
     * Creates cloudlets (tasks) for the simulation
     */
    private List<Cloudlet> createCloudlets() {
        List<Cloudlet> cloudletList = new ArrayList<>();
        UtilizationModel utilizationModel = new UtilizationModelFull();
        
        // Create various types of tasks
        for (int i = 0; i < 20; i++) {
            long length = 10000 + (i * 1000); // Task length in MI
            long fileSize = 1000 + (i * 100);  // Input file size
            long outputSize = 500 + (i * 50);  // Output file size
            
            Cloudlet cloudlet = new CloudletSimple(i, length, 1)
                .setFileSize(fileSize)
                .setOutputSize(outputSize)
                .setUtilizationModel(utilizationModel);
            
            cloudletList.add(cloudlet);
        }
        
        return cloudletList;
    }
    
    /**
     * Processes simulation results and generates reports
     */
    private void processSimulationResults(DatacenterBroker broker) {
        List<Cloudlet> finishedCloudlets = broker.getCloudletFinishedList();
        
        System.out.println("\n=== Simulation Results ===");
        new CloudletsTableBuilder(finishedCloudlets).build();
        
        // Record results for CSV generation and graph creation
        for (Cloudlet cloudlet : finishedCloudlets) {
            simulationResults.recordTaskResult(
                (int) cloudlet.getId(),
                0, // source device ID
                (int) cloudlet.getVm().getId(), // destination VM ID
                0.0, // offloading time
                cloudlet.getActualCpuTime(),
                cloudlet.getWaitingTime(),
                cloudlet.isFinished(),
                cloudlet.getVm().getId() < 4 ? "Cloud" : "Edge"
            );
            
            // Record energy consumption (simulated)
            simulationResults.recordEnergyConsumption(
                "VM_" + cloudlet.getVm().getId(),
                cloudlet.getActualCpuTime() * 0.1 // Simplified energy model
            );
            
            // Record resource utilization (simulated)
            simulationResults.recordResourceUtilization(
                "VM_" + cloudlet.getVm().getId(),
                0.7 + (Math.random() * 0.3) // Random utilization between 70-100%
            );
            
            // Record network usage (simulated)
            simulationResults.recordNetworkUsage(
                "Network_" + (cloudlet.getVm().getId() < 4 ? "Cloud" : "Edge"),
                cloudlet.getFileSize() + cloudlet.getOutputSize()
            );
        }
        
        // Process and save results (including graph generation)
        simulationResults.processResults();
    }
}
