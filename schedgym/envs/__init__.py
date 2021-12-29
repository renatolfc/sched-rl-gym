#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from .deeprm_env import DeepRmEnv
from .compact_env import CompactRmEnv
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='DeepRM-v0',
    nondeterministic=False,
    entry_point='schedgym.envs:DeepRmEnv',
)

register(
    id='CompactRM-v0',
    nondeterministic=False,
    entry_point='schedgym.envs:CompactRmEnv',
)
