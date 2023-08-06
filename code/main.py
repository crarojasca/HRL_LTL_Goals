from dataclasses import dataclass, asdict

from logger import WanDBLogger as Logger
from env.fourrooms import Fourrooms, LTLFourrooms
from env.breakout import Breakout, LTLBreakout
from env.sapientino import Sapientino

from agent.sarsa import Sarsa
from agent.dqn import DQN
from agent.option_critic import OptionCritic
from agent.actor_critic import ActorCritic

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf


agents = {
    "DQN": DQN,
    "OC": OptionCritic,
    "Sarsa": Sarsa
}

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg : DictConfig) -> None:

    experiment = cfg.experiment
    if cfg.agent.name=="OC": experiment=f"{experiment}_{cfg.agent.num_options}opt"
    run_name="{}_{}_{}".format(cfg.agent.name, cfg.env.name, experiment)

    print("\nRUN: {}\n\nPARAMETERS:\n{}".format(run_name, OmegaConf.to_yaml(cfg)))

    # Settng logger
    logger = Logger(
        cfg, run_name
    )

    if cfg.env.name=="fourrooms":
        env = LTLFourrooms(**cfg.env)
    elif cfg.env.name=="breakout":
        env = LTLBreakout(**cfg.env)
    elif cfg.env.name=="sapientino":
        env = Sapientino(**cfg.env)
    else:
        raise("Please declare an environment: fourrooms - breakout - sapientino")
        
    agent = agents[cfg.agent.name](
            observation_space=env.observation_space.shape[0], 
            action_space=env.action_space.n,
            logger=logger,
            args=cfg.agent
    )

    # Load  pretrained model if set
    # agent.load(cfg)

    # Train
    print("First Stage")
    env.spec.end_state = 1
    agent.run(env)

    print("Second Stage")
    # Update Goal
    env.spec.end_state = 2
    # Clear Buffer
    agent.buffer.buffer.clear()
    # Set New Number of Max Episodes
    agent.max_episodes = cfg.agent.max_episodes*2
    # Run
    agent.run(env)

    print("Third Stage")
    env.spec.end_state = 3
    # Clear Buffer
    agent.buffer.buffer.clear()
    # Set New Number of Max Episodes
    agent.max_episodes = cfg.agent.max_episodes*3
    # Run
    agent.run(env)


    # Save Model
    # agent.save(cfg, run_name)

    print("FINISHED RUNNING")

if __name__ == "__main__":
    main()