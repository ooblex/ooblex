// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/security/Pausable.sol";
import "@openzeppelin/contracts/security/ReentrancyGuard.sol";
import "@openzeppelin/contracts/utils/Counters.sol";

/**
 * @title ContentRegistry
 * @dev Smart contract for registering and verifying AI-processed video content
 * Provides cryptographic proof of authenticity and chain of custody tracking
 */
contract ContentRegistry is Ownable, Pausable, ReentrancyGuard {
    using Counters for Counters.Counter;
    
    // Content registration struct
    struct Content {
        string contentHash;
        uint256 timestamp;
        address creator;
        string metadata; // JSON metadata
        uint256 blockNumber;
        bool exists;
    }
    
    // Content history entry
    struct HistoryEntry {
        string contentHash;
        uint256 timestamp;
        address actor;
        string action;
        string metadata;
    }
    
    // Access control roles
    struct Verifier {
        bool isActive;
        string name;
        uint256 addedAt;
    }
    
    // State variables
    mapping(string => Content) private contents;
    mapping(string => HistoryEntry[]) private contentHistory;
    mapping(string => string[]) private contentDerivatives; // Parent -> children
    mapping(address => Verifier) private verifiers;
    mapping(address => uint256) private userContentCount;
    
    Counters.Counter private _contentIdCounter;
    
    // Events
    event ContentRegistered(
        string indexed contentHash,
        address indexed creator,
        uint256 timestamp,
        uint256 blockNumber
    );
    
    event ContentVerified(
        string indexed contentHash,
        address indexed verifier,
        bool isAuthentic,
        uint256 timestamp
    );
    
    event DerivativeRegistered(
        string indexed parentHash,
        string indexed derivativeHash,
        address indexed creator,
        uint256 timestamp
    );
    
    event VerifierAdded(address indexed verifier, string name);
    event VerifierRemoved(address indexed verifier);
    
    // Modifiers
    modifier onlyVerifier() {
        require(verifiers[msg.sender].isActive, "Not an authorized verifier");
        _;
    }
    
    modifier contentExists(string memory contentHash) {
        require(contents[contentHash].exists, "Content not found");
        _;
    }
    
    modifier validContentHash(string memory contentHash) {
        require(bytes(contentHash).length == 64, "Invalid content hash length");
        _;
    }
    
    constructor() {
        // Add contract deployer as first verifier
        verifiers[msg.sender] = Verifier({
            isActive: true,
            name: "System",
            addedAt: block.timestamp
        });
    }
    
    /**
     * @dev Register new content on the blockchain
     * @param contentHash SHA256 hash of the content
     * @param timestamp Unix timestamp of content creation
     * @param creator Address of content creator
     * @param metadata JSON string with additional metadata
     */
    function registerContent(
        string memory contentHash,
        uint256 timestamp,
        address creator,
        string memory metadata
    ) external whenNotPaused nonReentrant validContentHash(contentHash) {
        require(!contents[contentHash].exists, "Content already registered");
        require(creator != address(0), "Invalid creator address");
        require(timestamp <= block.timestamp, "Invalid timestamp");
        
        contents[contentHash] = Content({
            contentHash: contentHash,
            timestamp: timestamp,
            creator: creator,
            metadata: metadata,
            blockNumber: block.number,
            exists: true
        });
        
        // Add to history
        contentHistory[contentHash].push(HistoryEntry({
            contentHash: contentHash,
            timestamp: block.timestamp,
            actor: msg.sender,
            action: "REGISTERED",
            metadata: metadata
        }));
        
        userContentCount[creator]++;
        _contentIdCounter.increment();
        
        emit ContentRegistered(contentHash, creator, timestamp, block.number);
    }
    
    /**
     * @dev Register derivative content with parent reference
     * @param parentHash Hash of the parent content
     * @param derivativeHash Hash of the derivative content
     * @param timestamp Creation timestamp
     * @param metadata Metadata including processing details
     */
    function registerDerivative(
        string memory parentHash,
        string memory derivativeHash,
        uint256 timestamp,
        string memory metadata
    ) external whenNotPaused nonReentrant 
      contentExists(parentHash) 
      validContentHash(derivativeHash) {
        
        require(!contents[derivativeHash].exists, "Derivative already registered");
        
        // Register the derivative as new content
        contents[derivativeHash] = Content({
            contentHash: derivativeHash,
            timestamp: timestamp,
            creator: msg.sender,
            metadata: metadata,
            blockNumber: block.number,
            exists: true
        });
        
        // Link to parent
        contentDerivatives[parentHash].push(derivativeHash);
        
        // Add to history
        contentHistory[derivativeHash].push(HistoryEntry({
            contentHash: derivativeHash,
            timestamp: block.timestamp,
            actor: msg.sender,
            action: "DERIVED",
            metadata: string(abi.encodePacked("Parent: ", parentHash))
        }));
        
        emit DerivativeRegistered(parentHash, derivativeHash, msg.sender, timestamp);
    }
    
    /**
     * @dev Verify content authenticity
     * @param contentHash Hash to verify
     * @return isAuthentic Whether content is registered
     * @return contentData The content data if found
     */
    function verifyContent(string memory contentHash) 
        external 
        view 
        validContentHash(contentHash)
        returns (bool isAuthentic, Content memory contentData) {
        
        if (contents[contentHash].exists) {
            return (true, contents[contentHash]);
        }
        
        return (false, contentData);
    }
    
    /**
     * @dev Get content details
     * @param contentHash Hash of the content
     * @return Content struct with all details
     */
    function getContent(string memory contentHash) 
        external 
        view 
        contentExists(contentHash)
        returns (
            string memory,
            uint256,
            address,
            string memory,
            uint256
        ) {
        Content memory content = contents[contentHash];
        return (
            content.contentHash,
            content.timestamp,
            content.creator,
            content.metadata,
            content.blockNumber
        );
    }
    
    /**
     * @dev Get content history (chain of custody)
     * @param contentHash Hash of the content
     * @return Array of history entries
     */
    function getContentHistory(string memory contentHash)
        external
        view
        contentExists(contentHash)
        returns (HistoryEntry[] memory) {
        return contentHistory[contentHash];
    }
    
    /**
     * @dev Get derivatives of a content
     * @param parentHash Hash of the parent content
     * @return Array of derivative content hashes
     */
    function getDerivatives(string memory parentHash)
        external
        view
        contentExists(parentHash)
        returns (string[] memory) {
        return contentDerivatives[parentHash];
    }
    
    /**
     * @dev Add action to content history
     * @param contentHash Hash of the content
     * @param action Action performed
     * @param metadata Additional metadata
     */
    function addHistoryEntry(
        string memory contentHash,
        string memory action,
        string memory metadata
    ) external onlyVerifier contentExists(contentHash) {
        contentHistory[contentHash].push(HistoryEntry({
            contentHash: contentHash,
            timestamp: block.timestamp,
            actor: msg.sender,
            action: action,
            metadata: metadata
        }));
    }
    
    /**
     * @dev Batch verify multiple content hashes
     * @param contentHashes Array of content hashes
     * @return results Array of verification results
     */
    function batchVerify(string[] memory contentHashes)
        external
        view
        returns (bool[] memory results) {
        results = new bool[](contentHashes.length);
        
        for (uint256 i = 0; i < contentHashes.length; i++) {
            results[i] = contents[contentHashes[i]].exists;
        }
        
        return results;
    }
    
    /**
     * @dev Get user's content count
     * @param user Address of the user
     * @return Number of contents registered by user
     */
    function getUserContentCount(address user) external view returns (uint256) {
        return userContentCount[user];
    }
    
    /**
     * @dev Get total registered content count
     * @return Total number of registered contents
     */
    function getTotalContentCount() external view returns (uint256) {
        return _contentIdCounter.current();
    }
    
    // Access Control Functions
    
    /**
     * @dev Add a new verifier
     * @param verifier Address of the verifier
     * @param name Name of the verifier
     */
    function addVerifier(address verifier, string memory name) 
        external 
        onlyOwner {
        require(verifier != address(0), "Invalid verifier address");
        require(!verifiers[verifier].isActive, "Verifier already exists");
        
        verifiers[verifier] = Verifier({
            isActive: true,
            name: name,
            addedAt: block.timestamp
        });
        
        emit VerifierAdded(verifier, name);
    }
    
    /**
     * @dev Remove a verifier
     * @param verifier Address of the verifier
     */
    function removeVerifier(address verifier) external onlyOwner {
        require(verifiers[verifier].isActive, "Verifier not found");
        
        verifiers[verifier].isActive = false;
        
        emit VerifierRemoved(verifier);
    }
    
    /**
     * @dev Check if an address is a verifier
     * @param account Address to check
     * @return Whether the address is an active verifier
     */
    function isVerifier(address account) external view returns (bool) {
        return verifiers[account].isActive;
    }
    
    // Emergency Functions
    
    /**
     * @dev Pause the contract (emergency only)
     */
    function pause() external onlyOwner {
        _pause();
    }
    
    /**
     * @dev Unpause the contract
     */
    function unpause() external onlyOwner {
        _unpause();
    }
    
    /**
     * @dev Emergency function to update content metadata
     * @param contentHash Hash of the content
     * @param newMetadata New metadata
     */
    function updateContentMetadata(
        string memory contentHash,
        string memory newMetadata
    ) external onlyOwner contentExists(contentHash) {
        contents[contentHash].metadata = newMetadata;
        
        // Add to history
        contentHistory[contentHash].push(HistoryEntry({
            contentHash: contentHash,
            timestamp: block.timestamp,
            actor: msg.sender,
            action: "METADATA_UPDATED",
            metadata: newMetadata
        }));
    }
}