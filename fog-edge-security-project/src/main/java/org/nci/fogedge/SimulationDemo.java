package org.nci.fogedge;

import org.apache.logging.log4j.Level;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.nci.fogedge.model.SimulationConfig;
import org.nci.fogedge.model.SimulationResults;
import org.nci.fogedge.security.AttackType;
import org.nci.fogedge.security.SecurityLevel;
import org.nci.fogedge.util.ConfigurationManager;
import org.nci.fogedge.util.LoggingUtil;

import java.util.Arrays;
import java.util.Scanner;

/**
 * Standalone demo class for the Fog and Edge Computing Security Simulation
 * Based on the paper: "An Overview of Fog Computing and Edge Computing Security and Privacy Issues"
 * (Sensors 2021, 21, 8226, https://doi.org/10.3390/s21248226)
 */
public class SimulationDemo {
    private static final Logger logger = LogManager.getLogger(SimulationDemo.class);
    
    public static void main(String[] args) {
        // Configure logging
        LoggingUtil.configureLogging();
        
        System.out.println("=============================================================");
        System.out.println("  Fog and Edge Computing Security Simulation Demo");
        System.out.println("  Based on: \"An Overview of Fog Computing and Edge Computing");
        System.out.println("  Security and Privacy Issues\" (Sensors 2021, 21, 8226)");
        System.out.println("=============================================================");
        System.out.println();
        
        // Check if we should use interactive mode
        boolean interactive = args.length > 0 && args[0].equalsIgnoreCase("--interactive");
        
        if (interactive) {
            runInteractiveDemo();
        } else {
            runDefaultDemo();
        }
    }
    
    private static void runDefaultDemo() {
        System.out.println("Running default simulation with configuration from simulation.properties");
        System.out.println();
        
        // Create and run simulation
        FogEdgeSecuritySimulation simulation = new FogEdgeSecuritySimulation();
        simulation.initialize();
        simulation.runSimulation();
        simulation.printResults();
    }
    
    private static void runInteractiveDemo() {
        Scanner scanner = new Scanner(System.in);
        
        System.out.println("Interactive Simulation Mode");
        System.out.println("==========================");
        
        // Set logging level
        System.out.println("\nSelect logging level:");
        System.out.println("1. INFO (Default)");
        System.out.println("2. DEBUG (Verbose)");
        System.out.println("3. WARN (Minimal)");
        System.out.print("Enter choice [1-3]: ");
        
        int logChoice = getIntInput(scanner, 1, 3, 1);
        Level logLevel = Level.INFO;
        
        switch (logChoice) {
            case 1:
                logLevel = Level.INFO;
                break;
            case 2:
                logLevel = Level.DEBUG;
                break;
            case 3:
                logLevel = Level.WARN;
                break;
        }
        
        LoggingUtil.setLoggingLevel(logLevel);
        
        // Create custom configuration
        SimulationConfig config = new SimulationConfig();
        
        // Set topology parameters
        System.out.println("\nTopology Configuration:");
        System.out.print("Number of IoT devices [5-50]: ");
        config.setNumIoTDevices(getIntInput(scanner, 5, 50, 20));
        
        System.out.print("Number of edge nodes [1-10]: ");
        config.setNumEdgeNodes(getIntInput(scanner, 1, 10, 5));
        
        System.out.print("Number of fog nodes [1-5]: ");
        config.setNumFogNodes(getIntInput(scanner, 1, 5, 2));
        
        System.out.print("Number of simulation steps [10-1000]: ");
        config.setSimulationSteps(getIntInput(scanner, 10, 1000, 100));
        
        // Set security parameters
        System.out.println("\nSecurity Configuration:");
        System.out.println("Select security level:");
        System.out.println("1. LOW");
        System.out.println("2. MEDIUM");
        System.out.println("3. HIGH");
        System.out.println("4. VERY_HIGH");
        System.out.print("Enter choice [1-4]: ");
        
        int secChoice = getIntInput(scanner, 1, 4, 2);
        SecurityLevel secLevel = SecurityLevel.MEDIUM;
        
        switch (secChoice) {
            case 1:
                secLevel = SecurityLevel.LOW;
                break;
            case 2:
                secLevel = SecurityLevel.MEDIUM;
                break;
            case 3:
                secLevel = SecurityLevel.HIGH;
                break;
            case 4:
                secLevel = SecurityLevel.VERY_HIGH;
                break;
        }
        
        config.setSecurityLevel(secLevel);
        
        System.out.print("Enable security at IoT layer? (y/n): ");
        config.setSecurityEnabledAtIoT(getBooleanInput(scanner, true));
        
        System.out.print("Enable security at Edge layer? (y/n): ");
        config.setSecurityEnabledAtEdge(getBooleanInput(scanner, true));
        
        System.out.print("Enable security at Fog layer? (y/n): ");
        config.setSecurityEnabledAtFog(getBooleanInput(scanner, true));
        
        // Set attack parameters
        System.out.println("\nAttack Simulation Configuration:");
        System.out.print("Enable attack simulation? (y/n): ");
        config.setAttackSimulationEnabled(getBooleanInput(scanner, true));
        
        if (config.isAttackSimulationEnabled()) {
            System.out.println("Select attack types to enable:");
            System.out.println("1. All attack types");
            System.out.println("2. IoT layer attacks only");
            System.out.println("3. Edge layer attacks only");
            System.out.println("4. Fog layer attacks only");
            System.out.println("5. Network layer attacks only");
            System.out.print("Enter choice [1-5]: ");
            
            int attackChoice = getIntInput(scanner, 1, 5, 1);
            
            switch (attackChoice) {
                case 1:
                    config.setAttackTypes(Arrays.asList(AttackType.values()));
                    break;
                case 2:
                    config.setAttackTypes(Arrays.asList(
                            AttackType.IOT_PHYSICAL_TAMPERING,
                            AttackType.IOT_MALWARE_INJECTION,
                            AttackType.IOT_BATTERY_DRAINING
                    ));
                    break;
                case 3:
                    config.setAttackTypes(Arrays.asList(
                            AttackType.EDGE_DOS,
                            AttackType.EDGE_MAN_IN_MIDDLE,
                            AttackType.EDGE_AUTHENTICATION_BYPASS
                    ));
                    break;
                case 4:
                    config.setAttackTypes(Arrays.asList(
                            AttackType.FOG_DATA_THEFT,
                            AttackType.FOG_PRIVILEGE_ESCALATION,
                            AttackType.FOG_VM_ESCAPE
                    ));
                    break;
                case 5:
                    config.setAttackTypes(Arrays.asList(
                            AttackType.NETWORK_EAVESDROPPING,
                            AttackType.NETWORK_TRAFFIC_ANALYSIS,
                            AttackType.NETWORK_ROUTING_ATTACK
                    ));
                    break;
            }
        }
        
        System.out.println("\nRunning simulation with custom configuration...");
        System.out.println();
        
        // Override the configuration in ConfigurationManager
        ConfigurationManager.setConfig(config);
        
        // Create and run simulation
        FogEdgeSecuritySimulation simulation = new FogEdgeSecuritySimulation();
        simulation.initialize();
        simulation.runSimulation();
        
        // Print results
        System.out.println("\nSimulation Results:");
        System.out.println("===================");
        simulation.printResults();
        
        // Save results option
        System.out.print("\nDo you want to save detailed results to a file? (y/n): ");
        boolean saveResults = getBooleanInput(scanner, false);
        
        if (saveResults) {
            System.out.print("Enter filename (default: simulation_results.txt): ");
            String filename = scanner.nextLine().trim();
            if (filename.isEmpty()) {
                filename = "simulation_results.txt";
            }
            
            SimulationResults results = simulation.getResults();
            results.saveToFile(filename);
            System.out.println("Results saved to " + filename);
        }
        
        scanner.close();
    }
    
    private static int getIntInput(Scanner scanner, int min, int max, int defaultValue) {
        int value = defaultValue;
        
        try {
            String input = scanner.nextLine().trim();
            if (!input.isEmpty()) {
                value = Integer.parseInt(input);
            }
        } catch (NumberFormatException e) {
            System.out.println("Invalid input, using default: " + defaultValue);
            return defaultValue;
        }
        
        if (value < min || value > max) {
            System.out.println("Value out of range, using default: " + defaultValue);
            return defaultValue;
        }
        
        return value;
    }
    
    private static boolean getBooleanInput(Scanner scanner, boolean defaultValue) {
        String input = scanner.nextLine().trim().toLowerCase();
        
        if (input.isEmpty()) {
            return defaultValue;
        }
        
        return input.startsWith("y");
    }
}
