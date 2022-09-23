import pytest

from aintelope.aintelope.environments.savanna_zoo import SavannaZooEnv

def test_qagent_in_savanna_zoo_sequential():
    # TODO: refactor out into test constants? Or leave here? /shrug
    env_params = {
    "NUM_ITERS": 40,  # duration of the game
    "MAP_MIN": 0,
    "MAP_MAX": 20,
    "render_map_max": 20,
    "AMOUNT_AGENTS": 1,  # for now only one agent
    "AMOUNT_GRASS_PATCHES": 2,
    }
    e = SavannaZooEnv(env_params=env_params)
    
    

    