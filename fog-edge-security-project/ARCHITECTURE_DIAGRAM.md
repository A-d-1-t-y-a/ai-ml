# System Architecture Diagram

## Secure Fog Computing Architecture

```
+------------------+      +------------------+      +------------------+      +------------------+
|                  |      |                  |      |                  |      |                  |
|   IoT Devices    |----->|   Edge Nodes     |----->|   Fog Nodes      |----->|  Cloud Datacenter|
|                  |      |                  |      |                  |      |                  |
+------------------+      +------------------+      +------------------+      +------------------+
        |                        |                         |                         |
        v                        v                         v                         v
+------------------+      +------------------+      +------------------+      +------------------+
|  Security Layer  |      |  Security Layer  |      |  Security Layer  |      |  Security Layer  |
|  - Encryption    |      |  - Authentication|      |  - Blockchain    |      |  - Advanced     |
|  - Basic Auth    |      |  - Intrusion     |      |  - Decoy         |      |    Analytics    |
|                  |      |    Detection     |      |    Techniques    |      |                  |
+------------------+      +------------------+      +------------------+      +------------------+
        |                        |                         |                         |
        v                        v                         v                         v
+------------------+      +------------------+      +------------------+      +------------------+
|  Data Generation |      |  Initial         |      |  Advanced        |      |  Final          |
|  - Sensors       |      |  Processing      |      |  Processing      |      |  Processing     |
|  - Actuators     |      |  - Filtering     |      |  - Analytics     |      |  - Storage      |
|                  |      |  - Aggregation   |      |  - Decision      |      |  - Long-term    |
|                  |      |                  |      |    Making        |      |    Analytics    |
+------------------+      +------------------+      +------------------+      +------------------+

```

## Data Flow

1. **IoT Layer**:
   - IoT devices generate data
   - Data is encrypted using AES-256
   - Basic authentication is applied
   - Data is transmitted to connected Edge Nodes

2. **Edge Layer**:
   - Edge nodes receive encrypted data from IoT devices
   - Authentication is verified
   - Intrusion detection is performed
   - Data is processed (filtering, aggregation)
   - Processed data is either:
     - Stored locally if processing is complete
     - Forwarded to Fog Nodes for further processing

3. **Fog Layer**:
   - Fog nodes receive data from Edge Nodes
   - Advanced security measures are applied:
     - Blockchain for data integrity
     - Decoy techniques for misleading attackers
   - Advanced processing is performed
   - Processed data is either:
     - Stored locally if processing is complete
     - Forwarded to Cloud Datacenter for final processing

4. **Cloud Layer**:
   - Cloud datacenter receives data from Fog Nodes
   - Final processing and storage
   - Long-term analytics and decision making

## Security Countermeasures

| Layer | Security Countermeasures |
|-------|--------------------------|
| IoT   | Encryption (AES-256), Basic Authentication |
| Edge  | Authentication, Intrusion Detection |
| Fog   | Blockchain, Decoy Techniques |
| Cloud | Advanced Analytics, Comprehensive Security |

## Implementation Classes

| Component | Main Classes |
|-----------|--------------|
| IoT Layer | `IoTDevice.java` |
| Edge Layer | `EdgeNode.java` |
| Fog Layer | `FogNode.java` |
| Cloud Layer | `CloudDatacenter.java` |
| Security | `SecurityManager.java`, `SecurityIncident.java` |
| Simulation | `SimulationEngine.java` |
| Reporting | `ReportGenerator.java` |
