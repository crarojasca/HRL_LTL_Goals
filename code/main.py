from dataclasses import dataclass, asdict

from logger import WanDBLogger, TensorboardLogger
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
    "Sarsa": Sarsa,
    "A2C": ActorCritic
}

envs = {
    "fourrooms": LTLFourrooms,
    "breakout": LTLBreakout,
    "Sapientino": Sapientino
}

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg : DictConfig) -> None:

    
         
    # Load env   
    ##########
    if not cfg.env.name in envs:
        raise("Please declare an environment: fourrooms - breakout - sapientino")

    env = envs[cfg.env.name](**cfg.env)
        
    # Load Agent
    ############
    agent = agents[cfg.agent.name](
            observation_space=env.observation_space.shape[0], 
            action_space=env.action_space.n,
            args=cfg.agent
    )

    # Add additional parameters config:
    parameters = OmegaConf.to_container(cfg, resolve=True)
    parameters["agent"]["number_parameters"] = agent.number_parameters
    parameters["agent"]["device"] = str(agent.device)

    # Setting logger
    experiment = cfg.experiment
    if cfg.agent.name=="OC": experiment=f"{experiment}_{cfg.agent.num_options}opt"
    run_name="{}_{}_{}".format(cfg.agent.name, cfg.env.name, experiment)

    logger = TensorboardLogger(
        parameters, run_name
    )

    # Load  pretrained model if set
    # agent.load(cfg)

    # Train
    print("First Stage")
    env.spec.end_state = 1
    agent.run(env, logger)

    print("Second Stage")
    # Update Goal
    env.spec.end_state = 2
    # Clear Buffer
    agent.buffer.buffer.clear()
    # Set New Number of Max Episodes
    agent.max_episodes = cfg.agent.max_episodes*2
    # Run
    agent.run(env, logger)

    print("Third Stage")
    env.spec.end_state = 3
    # Clear Buffer
    agent.buffer.buffer.clear()
    # Set New Number of Max Episodes
    agent.max_episodes = cfg.agent.max_episodes*3
    # Run
    agent.run(env, logger)


    # Save Model
    # agent.save(cfg, run_name)

    logger.close()
    print("FINISHED RUNNING")
    

if __name__ == "__main__":
    main()