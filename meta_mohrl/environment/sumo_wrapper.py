"""
Multi-objective SUMO-RL environment wrapper.

Wraps sumo_rl to provide:
- Multi-objective reward vector [speed, waiting_time, queue_length]
- Task descriptor computation for meta-controller
- Context features extraction
- Heterogeneous multi-agent support (vehicles + traffic lights)
- Compatible with both single-agent and multi-agent (PettingZoo) modes
"""

import os
import sys
import numpy as np
import gymnasium as gym
from typing import Dict, Optional, Tuple, Any, List

# Ensure SUMO_HOME is set
SUMO_HOME = os.environ.get('SUMO_HOME', r'C:\Program Files (x86)\Eclipse\Sumo')
if SUMO_HOME not in os.environ.get('PATH', ''):
    os.environ['SUMO_HOME'] = SUMO_HOME
    tools_path = os.path.join(SUMO_HOME, 'tools')
    if tools_path not in sys.path:
        sys.path.append(tools_path)

# Enforce LibSumo for Windows stability to bypass WinError 10038
os.environ['LIBSUMO_AS_TRACI'] = '1'

from meta_mohrl.environment.reward_functions import (
    speed_reward, waiting_time_reward, queue_length_reward
)


class MultiObjectiveSumoEnv:
    """SUMO-RL environment wrapper with multi-objective rewards.

    Wraps the sumo_rl SumoEnvironment to provide:
    1. Multi-objective reward vector R_t = [r_speed, r_wait, r_queue]
    2. Task descriptor d for meta-controller
    3. Context features for MOHRL-ci augmented state
    """

    def __init__(
        self,
        net_file: str,
        route_file: str,
        num_seconds: int = 3600,
        delta_time: int = 5,
        yellow_time: int = 2,
        min_green: int = 5,
        max_green: int = 60,
        use_gui: bool = False,
        single_agent: bool = True,
        reward_fn: str = 'multi_objective',
        additional_sumo_cmd: Optional[str] = None,
        sumo_seed: int = 42
    ):
        self.net_file = net_file
        self.route_file = route_file
        self.num_seconds = num_seconds
        self.single_agent = single_agent
        self.use_gui = use_gui

        # Store initial state samples for task descriptor
        self._initial_states: List[np.ndarray] = []
        self._num_agents = 0
        self._obs_dim = 0

        try:
            import sumo_rl
            if single_agent:
                self.env = sumo_rl.SumoEnvironment(
                    net_file=net_file,
                    route_file=route_file,
                    single_agent=True,
                    use_gui=use_gui,
                    num_seconds=num_seconds,
                    delta_time=delta_time,
                    yellow_time=yellow_time,
                    min_green=min_green,
                    max_green=max_green,
                    sumo_seed=sumo_seed,
                    reward_fn='diff-waiting-time',
                )
                self._num_agents = 1
            else:
                self.env = sumo_rl.parallel_env(
                    net_file=net_file,
                    route_file=route_file,
                    use_gui=use_gui,
                    num_seconds=num_seconds,
                    delta_time=delta_time,
                    yellow_time=yellow_time,
                    min_green=min_green,
                    max_green=max_green,
                    sumo_seed=sumo_seed,
                    reward_fn='diff-waiting-time',
                )

            self._sumo_available = True
        except Exception as e:
            print(f"SUMO-RL environment could not be initialized: {e}")
            print("Falling back to mock environment for testing.")
            self._sumo_available = False
            self._setup_mock_env()

    def _setup_mock_env(self):
        """Create a mock environment for testing without SUMO installed."""
        self._obs_dim = 48
        self._num_agents = 4  # 2x2 grid
        self._num_actions = 4
        self._step_count = 0
        self._max_steps = self.num_seconds // 5  # delta_time = 5

    def reset(self) -> Tuple[np.ndarray, Dict]:
        """Reset environment and return initial observation."""
        if self._sumo_available:
            if self.single_agent:
                obs, info = self.env.reset()
                obs = np.array(obs, dtype=np.float32)
                self._obs_dim = len(obs)
                self._initial_states.append(obs.copy())
                return obs, info
            else:
                observations = self.env.reset()
                self._num_agents = len(self.env.agents) if hasattr(self.env, 'agents') else 4
                first_obs = list(observations.values())[0] if isinstance(observations, dict) else observations
                if isinstance(first_obs, np.ndarray):
                    self._obs_dim = len(first_obs)
                return observations, {}
        else:
            # Mock reset
            self._step_count = 0
            obs = np.random.randn(self._obs_dim).astype(np.float32) * 0.1
            self._initial_states.append(obs.copy())
            return obs, {}

    def step(self, action) -> Tuple[np.ndarray, np.ndarray, bool, bool, Dict]:
        """Step environment and return multi-objective reward vector.

        Returns:
            obs: next observation
            reward_vector: [speed_reward, waiting_reward, queue_reward]
            terminated: episode done
            truncated: episode truncated
            info: additional info including per-objective rewards
        """
        if self._sumo_available:
            if self.single_agent:
                obs, scalar_reward, terminated, truncated, info = self.env.step(action)
                obs = np.array(obs, dtype=np.float32)

                # Compute multi-objective rewards from traffic signal
                ts = list(self.env.traffic_signals.values())[0]
                reward_vector = np.array([
                    speed_reward(ts),
                    waiting_time_reward(ts),
                    queue_length_reward(ts)
                ], dtype=np.float32)

                info['reward_vector'] = reward_vector
                info['scalar_reward'] = scalar_reward
                info['per_objective'] = {
                    'speed': reward_vector[0],
                    'waiting': reward_vector[1],
                    'queue': reward_vector[2]
                }
                return obs, reward_vector, terminated, truncated, info
            else:
                observations, rewards, terminations, truncations, infos = \
                    self.env.step(action)
                # Multi-agent: compute multi-objective reward per agent
                mo_rewards = {}
                for agent_id in (self.env.agents if hasattr(self.env, 'agents') else []):
                    ts = self.env.traffic_signals.get(agent_id)
                    if ts:
                        mo_rewards[agent_id] = np.array([
                            speed_reward(ts),
                            waiting_time_reward(ts),
                            queue_length_reward(ts),
                        ], dtype=np.float32)
                    else:
                        mo_rewards[agent_id] = np.array([0., 0., 0.], dtype=np.float32)
                return observations, mo_rewards, terminations, truncations, infos
        else:
            # Mock step
            self._step_count += 1
            obs = np.random.randn(self._obs_dim).astype(np.float32) * 0.1
            # Mock rewards with some structure
            reward_vector = np.array([
                0.5 + 0.1 * np.sin(self._step_count * 0.01),   # speed
                -0.3 + 0.05 * np.cos(self._step_count * 0.02), # waiting
                -0.2 + 0.05 * np.sin(self._step_count * 0.015) # queue
            ], dtype=np.float32)
            terminated = self._step_count >= self._max_steps
            return obs, reward_vector, terminated, False, {'reward_vector': reward_vector}

    def get_task_descriptor(self) -> np.ndarray:
        """Compute task descriptor for meta-controller.

        d = [n/n_max, H_est/H_max, K/K_max, μ(s₀), σ²(s₀)]
        """
        from meta_mohrl.meta_controller.task_encoder import TaskEncoder

        initial_states = np.array(self._initial_states) if self._initial_states else np.zeros((1, self._obs_dim))
        return TaskEncoder.compute_descriptor(
            num_agents=self._num_agents,
            horizon_estimate=self.num_seconds // 5,  # steps at delta_time=5
            num_objectives=3,
            initial_states=initial_states
        )

    def get_context_features(self) -> np.ndarray:
        """Extract context features for MOHRL-ci augmented state."""
        if self._sumo_available and self.single_agent:
            try:
                ts = list(self.env.traffic_signals.values())[0]
                # Density, phase, queue per lane
                features = []
                for lane in ts.lanes:
                    features.append(ts.sumo.lane.getLastStepVehicleNumber(lane))
                    features.append(ts.sumo.lane.getLastStepOccupancy(lane))
                return np.array(features, dtype=np.float32)
            except Exception:
                return np.zeros(8, dtype=np.float32)
        return np.zeros(8, dtype=np.float32)

    @property
    def observation_space(self):
        if self._sumo_available:
            if self.single_agent:
                return self.env.observation_space
            else:
                first_agent = list(self.env.observation_spaces.values())[0] \
                    if hasattr(self.env, 'observation_spaces') else None
                return first_agent
        return gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self._obs_dim,))

    @property
    def action_space(self):
        if self._sumo_available:
            if self.single_agent:
                return self.env.action_space
            else:
                first_agent = list(self.env.action_spaces.values())[0] \
                    if hasattr(self.env, 'action_spaces') else None
                return first_agent
        return gym.spaces.Discrete(self._num_actions if hasattr(self, '_num_actions') else 4)

    @property
    def obs_dim(self) -> int:
        if self._obs_dim > 0:
            return self._obs_dim
        space = self.observation_space
        if hasattr(space, 'shape'):
            return space.shape[0]
        return 48

    @property
    def num_actions(self) -> int:
        space = self.action_space
        if hasattr(space, 'n'):
            return space.n
        return 4

    def close(self):
        if self._sumo_available:
            try:
                self.env.close()
                import time
                time.sleep(2.0) # Ensure OS releases the active TraCI TCP port
            except Exception:
                pass
