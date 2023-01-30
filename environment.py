from fourrooms import Fourrooms, LTLFourrooms

from gym.wrappers import AtariPreprocessing, TransformReward
from gym.wrappers import FrameStack as FrameStack_

def make_env(env_name, render):

    if env_name == 'fourrooms':
        return Fourrooms(render), False
    elif env_name == 'ltl_fourrooms':
        return LTLFourrooms(render), False

    env = gym.make(env_name)
    is_atari = hasattr(gym.envs, 'atari') and isinstance(env.unwrapped, gym.envs.atari.atari_env.AtariEnv)
    if is_atari:
        env = AtariPreprocessing(env, grayscale_obs=True, scale_obs=True, terminal_on_life_loss=True)
        env = TransformReward(env, lambda r: np.clip(r, -1, 1))
        env = FrameStack(env, 4)
    return env, is_atari