Design of sched-rl-gym
======================

This page documents the overall design of the environment, and may be useful in
understanding its components.

`sched-rl-gym` was designed with a view of having multiple layers to try and
separate functionality between them. Conceptually, we have three layers:

 1. Simulator primitives
 2. Simulator
 3. OpenAI Gym <-> Simulator Glue

With user code living in a fourth layer atop 3. The good thing of using this
design is that one can also access each layer directly, which is useful for:

 1. Unit testing (the code is tested with `coverage on coveralls.io
    <https://coveralls.io/github/renatolfc/sched-rl-gym>`_)
 2. Using the simulator directly (to replicate results, for example)
