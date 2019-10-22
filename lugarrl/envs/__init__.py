#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from .deeprm_env import DeepRmEnv
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='DeepRM-v0',
    nondeterministic=True,
    entry_point='lugarrl.envs:DeepRmEnv',
)
