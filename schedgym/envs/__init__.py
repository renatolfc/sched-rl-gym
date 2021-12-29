#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='DeepRM-v0',
    nondeterministic=False,
    entry_point='schedgym.envs.deeprm_env:DeepRmEnv',
)

register(
    id='CompactRM-v0',
    nondeterministic=False,
    entry_point='schedgym.envs.compact_env:CompactRmEnv',
)
