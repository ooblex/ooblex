/**
 * Deployment script for ContentRegistry smart contract
 * 
 * Usage: node deploy.js --network <network_name>
 */

const Web3 = require('web3');
const fs = require('fs');
const path = require('path');

// Network configurations
const networks = {
    ethereum: {
        rpc: process.env.ETHEREUM_RPC_URL || 'https://mainnet.infura.io/v3/YOUR_PROJECT_ID',
        chainId: 1,
        gasPrice: '50000000000' // 50 gwei
    },
    polygon: {
        rpc: process.env.POLYGON_RPC_URL || 'https://polygon-rpc.com',
        chainId: 137,
        gasPrice: '30000000000' // 30 gwei
    },
    avalanche: {
        rpc: process.env.AVALANCHE_RPC_URL || 'https://api.avax.network/ext/bc/C/rpc',
        chainId: 43114,
        gasPrice: '25000000000' // 25 gwei
    },
    arbitrum: {
        rpc: process.env.ARBITRUM_RPC_URL || 'https://arb1.arbitrum.io/rpc',
        chainId: 42161,
        gasPrice: '100000000' // 0.1 gwei
    },
    optimism: {
        rpc: process.env.OPTIMISM_RPC_URL || 'https://mainnet.optimism.io',
        chainId: 10,
        gasPrice: '1000000' // 0.001 gwei
    },
    bsc: {
        rpc: process.env.BSC_RPC_URL || 'https://bsc-dataseed.binance.org/',
        chainId: 56,
        gasPrice: '5000000000' // 5 gwei
    },
    // Test networks
    goerli: {
        rpc: process.env.GOERLI_RPC_URL || 'https://goerli.infura.io/v3/YOUR_PROJECT_ID',
        chainId: 5,
        gasPrice: '20000000000' // 20 gwei
    },
    mumbai: {
        rpc: process.env.MUMBAI_RPC_URL || 'https://rpc-mumbai.maticvigil.com',
        chainId: 80001,
        gasPrice: '10000000000' // 10 gwei
    }
};

async function deploy() {
    // Parse command line arguments
    const args = process.argv.slice(2);
    const networkIndex = args.indexOf('--network');
    const networkName = networkIndex !== -1 ? args[networkIndex + 1] : 'polygon';
    
    if (!networks[networkName]) {
        console.error(`Unknown network: ${networkName}`);
        console.log('Available networks:', Object.keys(networks).join(', '));
        process.exit(1);
    }
    
    const network = networks[networkName];
    console.log(`Deploying to ${networkName}...`);
    
    // Initialize Web3
    const web3 = new Web3(new Web3.providers.HttpProvider(network.rpc));
    
    // Load contract
    const contractPath = path.join(__dirname, 'ContentRegistry.json');
    const contractData = JSON.parse(fs.readFileSync(contractPath, 'utf8'));
    const abi = contractData.abi;
    const bytecode = contractData.bytecode || '0x608060405234801561001057600080fd5b50...'; // Add compiled bytecode
    
    // Get deployer account
    const privateKey = process.env.BLOCKCHAIN_PRIVATE_KEY;
    if (!privateKey) {
        console.error('BLOCKCHAIN_PRIVATE_KEY environment variable not set');
        process.exit(1);
    }
    
    const account = web3.eth.accounts.privateKeyToAccount(privateKey);
    web3.eth.accounts.wallet.add(account);
    
    console.log(`Deploying from account: ${account.address}`);
    
    // Check balance
    const balance = await web3.eth.getBalance(account.address);
    console.log(`Account balance: ${web3.utils.fromWei(balance, 'ether')} ETH`);
    
    if (balance === '0') {
        console.error('Insufficient balance for deployment');
        process.exit(1);
    }
    
    // Create contract instance
    const contract = new web3.eth.Contract(abi);
    
    // Estimate gas
    const deployTx = contract.deploy({ data: bytecode });
    const estimatedGas = await deployTx.estimateGas({ from: account.address });
    console.log(`Estimated gas: ${estimatedGas}`);
    
    // Deploy contract
    try {
        const deployedContract = await deployTx.send({
            from: account.address,
            gas: Math.floor(estimatedGas * 1.1), // Add 10% buffer
            gasPrice: network.gasPrice
        });
        
        console.log(`Contract deployed successfully!`);
        console.log(`Contract address: ${deployedContract.options.address}`);
        console.log(`Transaction hash: ${deployedContract.options.transactionHash}`);
        
        // Update contract addresses in JSON file
        contractData.networks = contractData.networks || {};
        contractData.networks[network.chainId] = {
            address: deployedContract.options.address,
            transactionHash: deployedContract.options.transactionHash,
            deployedAt: new Date().toISOString()
        };
        
        fs.writeFileSync(contractPath, JSON.stringify(contractData, null, 2));
        console.log('Contract addresses updated in ContentRegistry.json');
        
        // Verify contract on block explorer (if applicable)
        console.log('\nTo verify contract on block explorer:');
        console.log(`1. Go to the explorer for ${networkName}`);
        console.log(`2. Search for contract address: ${deployedContract.options.address}`);
        console.log('3. Click "Verify and Publish"');
        console.log('4. Use Solidity version: 0.8.19');
        console.log('5. Enable optimization: Yes, 200 runs');
        
    } catch (error) {
        console.error('Deployment failed:', error.message);
        process.exit(1);
    }
}

// Run deployment
deploy().catch(console.error);