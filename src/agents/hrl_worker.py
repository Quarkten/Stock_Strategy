from sb3_contrib import TQC

class Worker(TQC):
    """
    The low-level Worker agent in the HRL framework.
    This agent is responsible for executing the strategy defined by the Manager.
    """
    def __init__(self, policy, env, **kwargs):
        super(Worker, self).__init__(policy=policy, env=env, **kwargs)
