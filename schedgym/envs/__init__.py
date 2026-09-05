import logging

from gymnasium.envs.registration import register

from .compact_env import CompactRmEnv
from .deeprm_env import DeepRmEnv

logger = logging.getLogger(__name__)

register(
    id="DeepRM-v0",
    nondeterministic=False,
    entry_point=f"schedgym.envs.deeprm_env:{DeepRmEnv.__name__}",
)

register(
    id="CompactRM-v0",
    nondeterministic=False,
    entry_point=f"schedgym.envs.compact_env:{CompactRmEnv.__name__}",
)
