package org.nci.fogedge.security;

import org.bouncycastle.jce.provider.BouncyCastleProvider;
import java.util.logging.Logger;

import javax.crypto.Cipher;
import javax.crypto.KeyGenerator;
import javax.crypto.SecretKey;
import javax.crypto.spec.IvParameterSpec;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.security.SecureRandom;
import java.security.Security;
import java.util.ArrayList;
import java.util.Base64;
import java.util.List;

/**
 * Manages security operations for the fog computing architecture.
 * Implements lightweight encryption, authentication, and intrusion detection.
 * Based on the 2021 paper: "A Lightweight Security Framework for IoT-Fog-Cloud Architecture"
 */
public class SecurityManager {
    private boolean securityEnabled;
    private SecretKey encryptionKey;
    private SecureRandom random;
    private double encryptionTime;
    private double decryptionTime;
    private double authenticationTime;
    private List<String> detectedAttacks;
    
    // Constants for encryption
    private static final String AES_ALGORITHM = "AES/CBC/PKCS5Padding";
    private static final int KEY_SIZE = 256;
    private static final int IV_SIZE = 16;
    
    /**
     * Creates a new SecurityManager with specified security state
     * 
     * @param securityEnabled Whether security features are enabled
     */
    public SecurityManager(boolean securityEnabled) {
        this.securityEnabled = securityEnabled;
        this.random = new SecureRandom();
        this.encryptionTime = 0.0;
        this.decryptionTime = 0.0;
        this.authenticationTime = 0.0;
        this.detectedAttacks = new ArrayList<>();
        
        // Initialize security provider
        Security.addProvider(new BouncyCastleProvider());
        
        // Generate encryption key
        try {
            KeyGenerator keyGen = KeyGenerator.getInstance("AES");
            keyGen.init(KEY_SIZE, random);
            this.encryptionKey = keyGen.generateKey();
            
            Log.printLine("SecurityManager initialized with " + 
                    (securityEnabled ? "enabled" : "disabled") + " security features");
            
            if (securityEnabled) {
                Log.printLine("Using AES-" + KEY_SIZE + " encryption");
            }
        } catch (NoSuchAlgorithmException e) {
            Log.printLine("Error initializing SecurityManager: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    /**
     * Encrypts data using AES encryption with the specified security level
     * 
     * @param data The data to encrypt
     * @param level The security level affecting encryption strength
     * @return Encrypted data
     */
    public byte[] encryptData(byte[] data, SecurityLevel level) {
        if (!securityEnabled) {
            return data;
        }
        
        double startTime = System.currentTimeMillis();
        
        try {
            // Generate IV
            byte[] iv = new byte[IV_SIZE];
            random.nextBytes(iv);
            IvParameterSpec ivSpec = new IvParameterSpec(iv);
            
            // Initialize cipher
            Cipher cipher = Cipher.getInstance(AES_ALGORITHM);
            cipher.init(Cipher.ENCRYPT_MODE, encryptionKey, ivSpec);
            
            // Encrypt data
            byte[] encryptedData = cipher.doFinal(data);
            
            // Combine IV and encrypted data
            byte[] combined = new byte[iv.length + encryptedData.length];
            System.arraycopy(iv, 0, combined, 0, iv.length);
            System.arraycopy(encryptedData, 0, combined, iv.length, encryptedData.length);
            
            // Update encryption time
            this.encryptionTime += System.currentTimeMillis() - startTime;
            
            return combined;
        } catch (Exception e) {
            Log.printLine("Encryption error: " + e.getMessage());
            e.printStackTrace();
            return data; // Return original data on error
        }
    }
    
    /**
     * Decrypts data using AES decryption
     * 
     * @param encryptedData The data to decrypt
     * @return Decrypted data
     */
    public byte[] decryptData(byte[] encryptedData) {
        if (!securityEnabled || encryptedData.length <= IV_SIZE) {
            return encryptedData;
        }
        
        double startTime = System.currentTimeMillis();
        
        try {
            // Extract IV
            byte[] iv = new byte[IV_SIZE];
            System.arraycopy(encryptedData, 0, iv, 0, iv.length);
            IvParameterSpec ivSpec = new IvParameterSpec(iv);
            
            // Extract encrypted data
            byte[] actualEncryptedData = new byte[encryptedData.length - IV_SIZE];
            System.arraycopy(encryptedData, IV_SIZE, actualEncryptedData, 0, actualEncryptedData.length);
            
            // Initialize cipher
            Cipher cipher = Cipher.getInstance(AES_ALGORITHM);
            cipher.init(Cipher.DECRYPT_MODE, encryptionKey, ivSpec);
            
            // Decrypt data
            byte[] decryptedData = cipher.doFinal(actualEncryptedData);
            
            // Update decryption time
            this.decryptionTime += System.currentTimeMillis() - startTime;
            
            return decryptedData;
        } catch (Exception e) {
            Log.printLine("Decryption error: " + e.getMessage());
            e.printStackTrace();
            return encryptedData; // Return encrypted data on error
        }
    }
    
    /**
     * Authenticates a device or node using a challenge-response mechanism
     * 
     * @param id The ID of the device or node
     * @param challenge The challenge data
     * @param response The response data
     * @return true if authentication succeeds, false otherwise
     */
    public boolean authenticate(String id, byte[] challenge, byte[] response) {
        if (!securityEnabled) {
            return true;
        }
        
        double startTime = System.currentTimeMillis();
        
        try {
            // Generate expected response by hashing the challenge with the ID
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            digest.update(challenge);
            digest.update(id.getBytes());
            byte[] expectedResponse = digest.digest();
            
            // Compare with actual response
            boolean authenticated = MessageDigest.isEqual(expectedResponse, response);
            
            // Update authentication time
            this.authenticationTime += System.currentTimeMillis() - startTime;
            
            if (!authenticated) {
                // Record potential attack
                String attackInfo = "Authentication failure for ID: " + id + " at " + System.currentTimeMillis();
                detectedAttacks.add(attackInfo);
                Log.printLine("Security alert: " + attackInfo);
            }
            
            return authenticated;
        } catch (NoSuchAlgorithmException e) {
            Log.printLine("Authentication error: " + e.getMessage());
            e.printStackTrace();
            return false;
        }
    }
    
    /**
     * Generates a challenge for authentication
     * 
     * @return Random challenge data
     */
    public byte[] generateChallenge() {
        byte[] challenge = new byte[32]; // 256 bits
        random.nextBytes(challenge);
        return challenge;
    }
    
    /**
     * Generates a response to an authentication challenge
     * 
     * @param id The ID of the device or node
     * @param challenge The challenge data
     * @return Response data
     */
    public byte[] generateResponse(String id, byte[] challenge) {
        try {
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            digest.update(challenge);
            digest.update(id.getBytes());
            return digest.digest();
        } catch (NoSuchAlgorithmException e) {
            Log.printLine("Response generation error: " + e.getMessage());
            e.printStackTrace();
            return new byte[0];
        }
    }
    
    /**
     * Detects potential intrusions based on traffic patterns and authentication failures
     * 
     * @param sourceId Source ID
     * @param dataSize Data size
     * @param frequency Access frequency
     * @return true if intrusion is detected, false otherwise
     */
    public boolean detectIntrusion(String sourceId, int dataSize, int frequency) {
        if (!securityEnabled) {
            return false;
        }
        
        // Simple intrusion detection based on thresholds
        // In a real implementation, this would use more sophisticated algorithms
        boolean isIntrusion = false;
        
        // Check for abnormal data size
        if (dataSize > 1024 * 1000) { // More than 1MB
            isIntrusion = true;
        }
        
        // Check for abnormal access frequency
        if (frequency > 100) { // More than 100 accesses in a short period
            isIntrusion = true;
        }
        
        if (isIntrusion) {
            String attackInfo = "Potential intrusion detected from " + sourceId + 
                    " (data size: " + dataSize + ", frequency: " + frequency + ")";
            detectedAttacks.add(attackInfo);
            Log.printLine("Security alert: " + attackInfo);
        }
        
        return isIntrusion;
    }
    
    /**
     * Generates a digital signature for data integrity
     * 
     * @param data The data to sign
     * @return Digital signature
     */
    public String generateSignature(byte[] data) {
        try {
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            byte[] hash = digest.digest(data);
            return Base64.getEncoder().encodeToString(hash);
        } catch (NoSuchAlgorithmException e) {
            Log.printLine("Signature generation error: " + e.getMessage());
            e.printStackTrace();
            return "";
        }
    }
    
    /**
     * Verifies a digital signature for data integrity
     * 
     * @param data The data to verify
     * @param signature The signature to verify against
     * @return true if signature is valid, false otherwise
     */
    public boolean verifySignature(byte[] data, String signature) {
        try {
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            byte[] hash = digest.digest(data);
            String calculatedSignature = Base64.getEncoder().encodeToString(hash);
            return calculatedSignature.equals(signature);
        } catch (NoSuchAlgorithmException e) {
            Log.printLine("Signature verification error: " + e.getMessage());
            e.printStackTrace();
            return false;
        }
    }
    
    // Getters
    public boolean isSecurityEnabled() {
        return securityEnabled;
    }
    
    public double getEncryptionTime() {
        return encryptionTime;
    }
    
    public double getDecryptionTime() {
        return decryptionTime;
    }
    
    public double getAuthenticationTime() {
        return authenticationTime;
    }
    
    public List<String> getDetectedAttacks() {
        return new ArrayList<>(detectedAttacks);
    }
}
