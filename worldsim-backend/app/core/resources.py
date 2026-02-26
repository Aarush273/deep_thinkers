"""
Resource Pool and Management
"""
from dataclasses import dataclass, field
from typing import Dict
import numpy as np

@dataclass
class ResourcePool:
    """Represents resources in a region"""
    water: float = 1000.0
    food: float = 1000.0
    energy: float = 1000.0
    land: float = 1000.0
    
    # Maximum capacity constraints
    max_water: float = 2000.0
    max_food: float = 2000.0
    max_energy: float = 2000.0
    max_land: float = 1000.0
    
    def __post_init__(self):
        """Initialize with constraints"""
        self.water = min(self.water, self.max_water)
        self.food = min(self.food, self.max_food)
        self.energy = min(self.energy, self.max_energy)
        self.land = min(self.land, self.max_land)
    
    def get_as_dict(self) -> Dict[str, float]:
        """Convert to dictionary"""
        return {
            'water': self.water,
            'food': self.food,
            'energy': self.energy,
            'land': self.land
        }
    
    def deplete(self, water: float = 0, food: float = 0, energy: float = 0, land: float = 0):
        """Deplete resources"""
        self.water = max(0, self.water - water)
        self.food = max(0, self.food - food)
        self.energy = max(0, self.energy - energy)
        self.land = max(0, self.land - land)
    
    def replenish(self, water: float = 0, food: float = 0, energy: float = 0, land: float = 0):
        """Replenish resources with cap"""
        self.water = min(self.max_water, self.water + water)
        self.food = min(self.max_food, self.food + food)
        self.energy = min(self.max_energy, self.energy + energy)
        self.land = min(self.max_land, self.land + land)
    
    def get_total_value(self) -> float:
        """Calculate total resource value"""
        return self.water + self.food + self.energy + self.land * 10  # Land weighted more
    
    def is_critical(self) -> bool:
        """Check if any resource is critically low"""
        return (self.water < 100 or self.food < 100 or self.energy < 100)


@dataclass
class RegionState:
    """Complete state of a region"""
    region_id: str
    name: str
    resources: ResourcePool = field(default_factory=ResourcePool)
    population: int = 100
    development_level: float = 0.5  # 0-1 scale
    growth_rate: float = 0.02
    stability: float = 0.8  # 0-1, affects agent decisions
    trade_partners: Dict[str, float] = field(default_factory=dict)  # region_id -> trade_strength
    
    # Environmental factors
    temperature: float = 20.0  # Celsius
    rainfall: float = 100.0  # mm/month
    disaster_risk: float = 0.05  # Probability per cycle
    
    def get_state_dict(self) -> Dict:
        """Get complete state as dictionary"""
        return {
            'region_id': self.region_id,
            'name': self.name,
            'resources': self.resources.get_as_dict(),
            'population': self.population,
            'development_level': self.development_level,
            'growth_rate': self.growth_rate,
            'stability': self.stability,
            'trade_partners': self.trade_partners,
            'temperature': self.temperature,
            'rainfall': self.rainfall,
            'disaster_risk': self.disaster_risk
        }
