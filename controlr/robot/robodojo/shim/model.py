"""No-op XPolicyLab model: RoboDojo's eval client expects a policy server with a ``Model``,
but controlr's controller runs outside it (``deploy.py`` talks to ``controlr robodojo-serve``).
Anything asking this model for an action is a wiring error."""

from XPolicyLab.model_template import ModelTemplate


class Model(ModelTemplate):
    def __init__(self, model_cfg):
        self.model_cfg = model_cfg

    def reset(self):
        return None

    def update_obs(self, obs):
        del obs

    def update_obs_batch(self, obs_list):
        del obs_list

    def get_action(self):
        raise RuntimeError("controlr serves no model: actions come from controlr robodojo-serve via deploy.py")

    def get_action_batch(self, env_idx_list=None):
        raise RuntimeError("controlr serves no model: actions come from controlr robodojo-serve via deploy.py")
