import os
from dataclasses import dataclass
from game import (
    loadGameModule, loadAgentModule, load, writeinfo,
    evaluate, learn
)

from logger import Logger, WanDBLogger, TensorboardLogger

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

# @dataclass
# class args:
#     seed = 42
#     game = "BreakoutNRA"
#     agent = "Sarsa"
#     trainfile = "Sarsa_5_OriginalRewards"
#     rows = 2
#     cols = 5
#     gamma = 0.999
#     epsilon = 0.2
#     alpha = -1
#     lambdae = -1
#     nstep = 100
#     niter = 10000

#     debug = False
#     gui = True
#     sound = False
#     eval = False
#     stopongoal = False

loggers = {
    "base": Logger,
    "wandb": WanDBLogger,
    "tensorboard": TensorboardLogger
}


def init_game(args, trainname, agent):
    game = loadGameModule(args, trainname)
        # set parameters
    game.debug = False
    game.gui_visible = False
    game.sound_enabled = False

    
    game.setRandomSeed(args.agent.seed)
    game.init(agent)
    return game

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(args : DictConfig) -> None: 
  
    # trainname = args.trainfile.replace('.npz','')

    # if not os.path.exists(f"data/{trainname}/"):
    #     print("Creating directory")
    #     os.makedirs(f"data/{trainname}/")

    run_name="{}_{}_{}".format(args.agent.name, args.env.name, args.experiment)

    # print(run_name)
    # print(args)

    parameters = OmegaConf.to_container(args, resolve=True)
    parameters["agent"]["device"] = "cpu"

    logger = loggers[args.logger.type](
        parameters, run_name
    )

    # load game and agent modules

    agent = loadAgentModule(args)
    agent.gamma = args.agent.gamma
    agent.epsilon = args.agent.epsilon
    agent.alpha = args.agent.alpha
    agent.nstepsupdates = args.agent.nstep
    agent.lambdae = args.agent.lambdae
    agent.debug = False
    agent.setRandomSeed(args.agent.seed)


    game = init_game(args, run_name, agent)


    print("Game iteration: %d" %game.iteration)
    print("Game elapsedtime: %d" %game.elapsedtime)

    # if (game.iteration==0):
    #     writeinfo(args, run_name, game, agent, init=True)


    # First round
    optimalPolicyFound = learn(args.agent, game, agent, logger)
    # writeinfo(args, run_name, game, agent, False, optimalPolicyFound)

    # # Second Round
    # game.RA.left_right=False
    # game.iteration = 0
    # optimalPolicyFound = learn(args, game, agent)
    # writeinfo(args, trainname ,game, agent, False, optimalPolicyFound)


    print("Experiment terminated after iteration: %d!!!\n" %game.iteration)
    #print('saving ...')
    #save()
    print('Game over')
    logger.close()
    game.quit()
    print("FINISHED RUNNING")

if __name__=="__main__":
    main()