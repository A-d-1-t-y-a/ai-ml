package org.nci.fogedge.topology;

/**
 * Enum representing different wireless technologies used in IoT devices
 * Based on the paper: "An Overview of Fog Computing and Edge Computing Security and Privacy Issues"
 * (Sensors 2021, 21, 8226, https://doi.org/10.3390/s21248226)
 */
public enum WirelessType {
    WIFI("WiFi", 54.0, 100.0, 3.0),
    BLUETOOTH("Bluetooth", 3.0, 10.0, 1.0),
    ZIGBEE("ZigBee", 0.25, 100.0, 0.5),
    LORA("LoRa", 0.05, 10000.0, 0.1),
    CELLULAR_5G("5G", 1000.0, 1000.0, 5.0),
    CELLULAR_4G("4G", 100.0, 500.0, 4.0),
    CELLULAR_NB_IOT("NB-IoT", 0.2, 1000.0, 0.2);
    
    private final String name;
    private final double bandwidth; // Mbps
    private final double range; // meters
    private final double energyConsumption; // mW per packet
    
    WirelessType(String name, double bandwidth, double range, double energyConsumption) {
        this.name = name;
        this.bandwidth = bandwidth;
        this.range = range;
        this.energyConsumption = energyConsumption;
    }
    
    public String getName() {
        return name;
    }
    
    public double getBandwidth() {
        return bandwidth;
    }
    
    public double getRange() {
        return range;
    }
    
    public double getEnergyConsumption() {
        return energyConsumption;
    }
    
    @Override
    public String toString() {
        return name;
    }
}
