"""Centralized configuration with environment-aware settings"""
import os
import json
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum

class TradingMode(Enum):
    BACKTEST = "backtest"
    PAPER = "paper"
    LIVE = "live"

class AssetClass(Enum):
    CRYPTO = "crypto"
    FOREX = "forex"
    EQUITIES = "equities"

@dataclass
class RLConfig:
    """Reinforcement Learning Configuration"""
    gamma: float = 0.99
    learning_rate: float = 0.0003
    buffer_size: int = 100000
    batch_size: int = 64
    tau: float = 0.005  # Target network update rate
    exploration_noise: float = 0.1
    policy_delay: int = 2  # TD3-style delayed updates
    
@dataclass
class NeuroevolutionConfig:
    """NEAT Configuration for Neuroevolution"""
    population_size: int = 100
    elitism: int = 2
    survival_threshold: float = 0.2
    activation_mutation_rate: float = 0.1
    node_add_rate: float = 0.05
    node_delete_rate: float = 0.02
    weight_mutation_rate: float = 0.8
    weight_perturbation_scale: float = 0.5
    
@dataclass
class TradingConfig:
    """Trading System Configuration"""
    mode: TradingMode = TradingMode.PAPER
    asset_class: AssetClass = AssetClass.CRYPTO
    symbols: List[str] = field(default_factory=lambda: ["BTC/USDT", "ETH/USDT"])
    timeframe: str = "15m"
    initial_capital: float = 10000.0
    max_position_size: float = 0.1  # 10% of capital
    max_daily_loss: float = 0.02  # 2% daily stop
    transaction_cost: float = 0.001  # 0.1%
    
@dataclass
class FirebaseConfig:
    """Firebase Configuration"""
    project_id: str = "evolution-trading-system"
    collection_agents: str = "neuroevolution_agents"
    collection_trades: str = "executed_trades"
    collection_metrics: str = "performance_metrics"
    enable_realtime: bool = True
    
class ConfigManager:
    """Central configuration manager with validation"""
    
    def __init__(self, env: str = "production"):
        self.env = env
        self.rl = RLConfig()
        self.neuro = NeuroevolutionConfig()
        self.trading = TradingConfig()
        self.firebase = FirebaseConfig()
        self._validate()
        
    def _validate(self):
        """Validate configuration integrity"""
        assert 0 < self.rl.gamma <= 1, "Gamma must be in (0,1]"
        assert self.trading.max_position_size > 0, "Position size must be positive"
        assert self.neuro.population_size > 0, "Population size must be positive"
        
        # Environment-specific overrides
        if self.env == "development":
            self.trading.mode = TradingMode.BACKTEST
            self.neuro.population_size = 20
            
    def to_dict(self) -> Dict:
        """Serialize configuration for logging"""
        return {
            "env": self.env,
            "rl": self.rl.__dict__,
            "neuro": self.neuro.__dict__,
            "trading": self.trading.__dict__,
            "firebase": self.firebase.__dict__
        }