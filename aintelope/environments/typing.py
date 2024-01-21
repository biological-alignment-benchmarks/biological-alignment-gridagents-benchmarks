from typing import Dict, Union

import numpy as np
import gymnasium as gym
from pettingzoo import AECEnv, ParallelEnv

ObservationFloat = np.float32
PositionFloat = np.float32
Action = int
AgentId = str
AgentStates = Dict[AgentId, np.ndarray]

Observation = np.ndarray
Reward = float
Info = dict

PettingZooEnv = Union[AECEnv, ParallelEnv]
Environment = Union[gym.Env, PettingZooEnv]
