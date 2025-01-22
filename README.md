# LLM-AI-Companion
AI Agent Companion/Advisor for Game Users

## Overview
LLM-Advisor is an advanced AI agent system designed to assist players in strategic gaming environments. Built on SwarmZero technology, it provides real-time analysis, strategic advice, and monitors smart contract interactions for blockchain-based games.

## Features

### Strategic Game Analysis
- Real-time game state analysis
- Move validation and evaluation
- Position advantage calculation
- Attack opportunity identification
- Risk assessment and defensive strategy suggestions

### Smart Contract Integration
- Automated contract event monitoring
- Contract upgrade detection
- State change tracking
- Transaction history maintenance
- Real-time adaptation to contract modifications

### Multi-Agent Architecture
- Main Strategy Agent: Oversees overall decision-making
- Event Monitor Agent: Tracks contract and game events
- Move Validator Agent: Ensures rule compliance
- Collaborative decision-making between agents

### Arena Support
- Spectator mode for game observation
- Historical game analysis
- Performance tracking
- Inter-agent learning capabilities

## Installation

### Prerequisites
- Python 3.8 or higher
- Virtual environment (recommended)
- Mistral API key
- SwarmZero configuration

## Configuration

### Required Environment Variables
- `MISTRAL_API_KEY`: Your Mistral AI API key
- `MARQUIS_CONTRACT_ADDRESS`: Smart contract address
- `RPC_URL`: RPC endpoint for blockchain interaction

### Optional Settings
- `LOG_LEVEL`: Logging detail level (default: INFO)
- `ENVIRONMENT`: Development/production environment
- `ARENA_ENABLED`: Enable/disable arena features
- `MAX_STORED_GAMES`: Number of historical games to store


