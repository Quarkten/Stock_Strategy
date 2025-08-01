from stable_baselines3 import DQN

class Manager(DQN):
    """
    The high-level Manager agent in the HRL framework.
    This agent decides on the overall strategy and the target R-multiple.
    """
    def __init__(self, policy, env, **kwargs):
        super(Manager, self).__init__(policy=policy, env=env, **kwargs)
