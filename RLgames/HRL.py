import os
from dataclasses import dataclass, asdict
from game import (
    loadGameModule, loadAgentModule, load, writeinfo,
    evaluate, learn
)
from logger import WanDBLogger, TensorboardLogger

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf


# @dataclass
# class args:
#     name = "1"
#     seed = 42
#     game = "Sapientino3C" # BreakoutNRA Sapientino3C
#     agent = "Sarsa"
#     trainfile = "Sapiento_Best_2" 
#     rows = 2
#     cols = 5
#     gamma = 0.99
#     epsilon = 0.1
#     alpha = -1
#     lambdae = -1
#     nstep = 100
#     niter = 20000

#     debug = False
#     gui = True
#     sound = False
#     eval = False
#     stopongoal = False

# @dataclass
# class agent_args:
#     num_options=4
#     name="OC_10"
#     env='NRA_NewHyper' 
#     logdir='runs' 
#     max_episodes=50000
#     batch_size=240 
#     cuda=True 
#     entropy_reg=0.01 
#     epsilon_decay=20000
#     epsilon_min=0.1
#     epsilon_start=1.0 
#     freeze_interval=1000
#     gamma=0.999 
#     learning_rate=0.00025 
#     max_history=1000000 
#     max_steps_ep=18000 
#     max_steps_total=5000
#     optimal_eps=0.05 
#     seed=42 
#     temp=1 
#     termination_reg=0.01 
#     update_frequency=4


def init_game(args, trainname, agent):
    game = loadGameModule(args, trainname)
    # set parameters
    game.debug = False
    game.gui_visible = args.env.render
    game.sound_enabled = False
    
    game.setRandomSeed(args.env.seed)
    game.init(agent)

        # load saved data
    load(trainname, game, agent)
    print("Game iteration: %d" %game.iteration)
    print("Game elapsedtime: %d" %game.elapsedtime)

    if (game.iteration==0):
        writeinfo(args, trainname, game, agent, init=True)
    return game

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg : DictConfig) -> None: 

    run_name="{}_{}_{}".format(cfg.agent.name, cfg.env.name, cfg.experiment)

    if not os.path.exists(f"data/{run_name}/"):
        print("Creating directory")
        os.makedirs(f"data/{run_name}/")

    # load game and agent modules

    agent = loadAgentModule(cfg)
    agent.gamma = cfg.agent.gamma
    agent.epsilon = cfg.agent.epsilon
    agent.alpha = cfg.agent.alpha
    agent.nstepsupdates = cfg.agent.nstep
    agent.lambdae = cfg.agent.lambdae
    agent.debug = False
    agent.setRandomSeed(cfg.agent.seed)

    parameters = OmegaConf.to_container(cfg, resolve=True)
    # parameters["agent"]["number_parameters"] = agent.number_parameters
    parameters["agent"]["device"] = "cpu"

    logger = WanDBLogger(
        parameters, run_name
    )

    # # First round
    # args.game = "BreakoutN"
    # game = init_game(args, run_name, agent)
    # optimalPolicyFound = learn(args, game, agent, logger)
    # writeinfo(args, run_name,game, agent, False, optimalPolicyFound)

    # # Second Round
    # args.game = "BreakoutNRA"
    # args.game = "Sapientino3C"
    
    game = init_game(cfg, run_name, agent)
    # game.iteration = args.niter
    # args.niter = args.niter*2
    optimalPolicyFound = learn(cfg, game, agent, logger)
    writeinfo(cfg, run_name, game, agent, False, optimalPolicyFound)


    ## OPTION CRITIC
    # agent_args.stateSpace = 1
    # agent_oc = Agent(agent_args)
    # agent_oc.run(game)

    print("Experiment terminated after iteration: %d!!!\n" %game.iteration)
    #print('saving ...')
    #save()
    print('Game over')
    game.quit()

if __name__=="__main__":
    main()

