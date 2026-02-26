"""
FastAPI Routes for WorldSim
"""
from fastapi import APIRouter, HTTPException, Query

router = APIRouter()

# Global simulation instance (will be set in main.py)
simulation = None

@router.on_event("startup")
async def startup_event():
    """Initialize simulation on startup"""
    global simulation
    if simulation is None:
        from app.simulation.engine import WorldSimulation
        simulation = WorldSimulation(num_regions=6)

@router.get("/")
async def root():
    """API info"""
    return {
        "name": "WorldSim API",
        "version": "1.0",
        "description": "Adaptive Resource Scarcity & Agent Strategy Simulator"
    }

@router.post("/simulation/step")
async def step_simulation():
    """Execute one simulation cycle"""
    if simulation is None:
        raise HTTPException(status_code=500, detail="Simulation not initialized")
    
    try:
        cycle_data = simulation.step()
        return cycle_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/simulation/state")
async def get_world_state():
    """Get current world state"""
    if simulation is None:
        raise HTTPException(status_code=500, detail="Simulation not initialized")
    
    return simulation.get_world_state()

@router.get("/simulation/statistics")
async def get_statistics():
    """Get simulation statistics"""
    if simulation is None:
        raise HTTPException(status_code=500, detail="Simulation not initialized")
    
    return simulation.get_statistics()

@router.get("/simulation/history")
async def get_cycle_history(limit: int = Query(50, le=500)):
    """Get recent cycle history"""
    if simulation is None:
        raise HTTPException(status_code=500, detail="Simulation not initialized")
    
    return {
        'cycles': simulation.cycle_history[-limit:],
        'total_cycles': len(simulation.cycle_history)
    }

@router.get("/regions")
async def get_regions():
    """Get all regions"""
    if simulation is None:
        raise HTTPException(status_code=500, detail="Simulation not initialized")
    
    return {
        rid: region.get_state_dict() 
        for rid, region in simulation.regions.items()
    }

@router.get("/regions/{region_id}")
async def get_region(region_id: str):
    """Get specific region state"""
    if simulation is None:
        raise HTTPException(status_code=500, detail="Simulation not initialized")
    
    if region_id not in simulation.regions:
        raise HTTPException(status_code=404, detail="Region not found")
    
    region = simulation.regions[region_id]
    agent = simulation.agents[region_id]
    
    return {
        'region': region.get_state_dict(),
        'agent_stats': {
            'actions_taken': len(agent.action_history),
            'avg_reward': float(sum(agent.reward_history) / len(agent.reward_history)) if agent.reward_history else 0,
            'recent_actions': [agent.ACTIONS[a]['name'] for a in agent.action_history[-5:]],
            'epsilon': float(agent.dqn.epsilon)
        }
    }

@router.get("/trade-network")
async def get_trade_network():
    """Get trade network state"""
    if simulation is None:
        raise HTTPException(status_code=500, detail="Simulation not initialized")
    
    return {
        'nodes': list(simulation.regions.keys()),
        'edges': [
            {'source': u, 'target': v, 'weight': data.get('weight', 0.5)}
            for u, v, data in simulation.trade_network.trade_graph.edges(data=True)
        ],
        'recent_trades': simulation.trade_network.trade_history[-20:]
    }

@router.get("/events")
async def get_events(limit: int = Query(50, le=1000)):
    """Get event history"""
    if simulation is None:
        raise HTTPException(status_code=500, detail="Simulation not initialized")
    
    events = simulation.event_history[-limit:]
    return {
        'count': len(events),
        'events': [
            {
                'type': event.event_type,
                'affected_regions': event.affected_regions,
                'severity': event.severity,
                'timestamp': event.timestamp.isoformat()
            }
            for event in events
        ]
    }

@router.post("/simulation/reset")
async def reset_simulation():
    """Reset simulation to initial state"""
    global simulation
    from app.simulation.engine import WorldSimulation
    simulation = WorldSimulation(num_regions=6)
    return {"status": "Reset successful", "cycle": simulation.current_cycle}

@router.get("/simulation/analysis")
async def get_analysis():
    """Get analysis of emergent strategies and sustainability"""
    if simulation is None:
        raise HTTPException(status_code=500, detail="Simulation not initialized")
    
    analysis = {
        'cycle': simulation.current_cycle,
        'regions': {},
        'sustainability_metrics': {}
    }
    
    # Per-region analysis
    for region_id, region in simulation.regions.items():
        agent = simulation.agents[region_id]
        
        # Strategy analysis
        action_counts = {}
        for action in agent.action_history:
            action_name = agent.ACTIONS[action]['name']
            action_counts[action_name] = action_counts.get(action_name, 0) + 1
        
        # Sort by frequency
        top_actions = sorted(action_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        
        analysis['regions'][region_id] = {
            'name': region.name,
            'status': 'critical' if region.is_critical() else 'stable' if region.stability > 0.6 else 'fragile',
            'population': region.population,
            'development': region.development_level,
            'stability': region.stability,
            'total_actions': len(agent.action_history),
            'top_strategies': [{'action': a, 'frequency': f} for a, f in top_actions],
            'learning_progress': float(agent.dqn.epsilon),
            'avg_reward': float(sum(agent.reward_history) / len(agent.reward_history)) if agent.reward_history else 0
        }
    
    # Global sustainability
    total_pop = sum(r.population for r in simulation.regions.values())
    avg_resources = {
        'water': sum(r.resources.water for r in simulation.regions.values()) / simulation.num_regions,
        'food': sum(r.resources.food for r in simulation.regions.values()) / simulation.num_regions,
        'energy': sum(r.resources.energy for r in simulation.regions.values()) / simulation.num_regions,
    }
    
    analysis['sustainability_metrics'] = {
        'total_population': total_pop,
        'avg_resources': avg_resources,
        'collapsed_regions': sum(1 for r in simulation.regions.values() if r.is_critical()),
        'trade_intensity': simulation.trade_network.trade_graph.number_of_edges() / (simulation.num_regions * 2),
        'events_total': len(simulation.event_history)
    }
    
    return analysis
