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
- Python 3.11 or higher
- Virtual environment (recommended)
- Mistral API key
- GEMINI API key
- SwarmZero configuration

### Setup
1. Clone the repository:
```bash
git clone https://github.com/yourusername/LLM-Advisor.git
cd LLM-Advisor
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration values
```

5. Run the Advisor you want in a given terminal:
```bash
python run GEMINI_Advisor.py
python run Mistral_Advisor.py
python run Mistral_Advisor_for_Creator.py
```

## Configuration

### Required Environment Variables
- `MISTRAL_API_KEY`: Your Mistral AI API key
- `GEMINI_API_KEY`: GEMINI API key


