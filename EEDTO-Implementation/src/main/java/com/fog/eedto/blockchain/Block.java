package com.fog.eedto.blockchain;

import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.List;

/**
 * Represents a block in the blockchain for the EEDTO system.
 * Each block contains a list of task offloading transactions and is linked to the previous block.
 */
public class Block {
    private final int index;
    private final long timestamp;
    private final List<TaskTransaction> transactions;
    private final String previousHash;
    private String hash;
    private int nonce;
    
    /**
     * Constructor for the Block class
     * 
     * @param index Block index in the blockchain
     * @param timestamp Block creation timestamp
     * @param transactions List of transactions in the block
     * @param previousHash Hash of the previous block
     */
    public Block(int index, long timestamp, List<TaskTransaction> transactions, String previousHash) {
        this.index = index;
        this.timestamp = timestamp;
        this.transactions = transactions;
        this.previousHash = previousHash;
        this.hash = calculateHash();
        this.nonce = 0;
    }
    
    /**
     * Calculate the hash of the block
     * 
     * @return SHA-256 hash of the block
     */
    public String calculateHash() {
        String dataToHash = index + timestamp + previousHash + nonce + transactions.toString();
        
        try {
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            byte[] hashBytes = digest.digest(dataToHash.getBytes(StandardCharsets.UTF_8));
            
            // Convert byte array to hexadecimal string
            StringBuilder hexString = new StringBuilder();
            for (byte hashByte : hashBytes) {
                String hex = Integer.toHexString(0xff & hashByte);
                if (hex.length() == 1) {
                    hexString.append('0');
                }
                hexString.append(hex);
            }
            
            return hexString.toString();
        } catch (NoSuchAlgorithmException e) {
            throw new RuntimeException("Error calculating hash: " + e.getMessage(), e);
        }
    }
    
    /**
     * Mine the block with the specified difficulty
     * 
     * @param difficulty Number of leading zeros required in the hash
     */
    public void mineBlock(int difficulty) {
        String target = new String(new char[difficulty]).replace('\0', '0');
        
        while (!hash.substring(0, difficulty).equals(target)) {
            nonce++;
            hash = calculateHash();
        }
        
        System.out.println("Block mined: " + hash);
    }
    
    // Getters
    public int getIndex() {
        return index;
    }
    
    public long getTimestamp() {
        return timestamp;
    }
    
    public List<TaskTransaction> getTransactions() {
        return transactions;
    }
    
    public String getPreviousHash() {
        return previousHash;
    }
    
    public String getHash() {
        return hash;
    }
    
    public int getNonce() {
        return nonce;
    }
    
    @Override
    public String toString() {
        return "Block{" +
                "index=" + index +
                ", timestamp=" + timestamp +
                ", transactions=" + transactions.size() +
                ", previousHash='" + previousHash + '\'' +
                ", hash='" + hash + '\'' +
                ", nonce=" + nonce +
                '}';
    }
}
