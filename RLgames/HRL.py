import os
from dataclasses import dataclass, asdict
from game import (
    loadGameModule, loadAgentModule, load, writeinfo,
    evaluate, learn
)
from option_critic.logger import WanDBLogger as Logger
# from option_critic.option_critic import Agent

@dataclass
class args:
    name = "1"
    seed = 42
    game = "Sapientino3C" # BreakoutNRA Sapientino3C
    agent = "Sarsa"
    trainfile = "Sapiento_Best_2" 
    rows = 2
    cols = 5
    gamma = 0.99
    epsilon = 0.1
    alpha = -1
    lambdae = -1
    nstep = 100
    niter = 20000

    debug = False
    gui = True
    sound = False
    eval = False
    stopongoal = False

@dataclass
class agent_args:
    num_options=4
    name="OC_10"
    env='NRA_NewHyper' 
    logdir='runs' 
    max_episodes=50000
    batch_size=240 
    cuda=True 
    entropy_reg=0.01 
    epsilon_decay=20000
    epsilon_min=0.1
    epsilon_start=1.0 
    freeze_interval=1000
    gamma=0.999 
    learning_rate=0.00025 
    max_history=1000000 
    max_steps_ep=18000 
    max_steps_total=5000
    optimal_eps=0.05 
    seed=42 
    temp=1 
    termination_reg=0.01 
    update_frequency=4


def init_game(args, trainname, agent):
    game = loadGameModule(args, trainname)
    # set parameters
    game.debug = args.debug
    game.gui_visible = args.gui
    game.sound_enabled = args.sound
    if (args.debug):
        game.sleeptime = 1.0
        game.gui_visible = True
    
    game.setRandomSeed(args.seed)
    game.init(agent)

        # load saved data
    load(trainname, game, agent)
    print("Game iteration: %d" %game.iteration)
    print("Game elapsedtime: %d" %game.elapsedtime)

    if (game.iteration==0):
        writeinfo(args, trainname, game, agent, init=True)
    return game



if not os.path.exists(f"data/{args.trainfile}/"):
    print("Creating directory")
    os.makedirs(f"data/{args.trainfile}/")

# load game and agent modules

agent = loadAgentModule(args)
agent.gamma = args.gamma
agent.epsilon = args.epsilon
agent.alpha = args.alpha
agent.nstepsupdates = args.nstep
agent.lambdae = args.lambdae
agent.debug = args.debug
agent.setRandomSeed(args.seed)


# learning or evaluation process
if (args.eval):
    game = init_game(args, args.trainfile, agent)
    evaluate(game, agent, 10)
else:       

    run_name = "{}-{}-{}".format(args.agent, args.game, args.name) 

    logger = Logger(
        "HRL", run_name
    )

    # # First round
    # args.game = "BreakoutN"
    # game = init_game(args, args.trainfile, agent)
    # optimalPolicyFound = learn(args, game, agent, logger)
    # writeinfo(args, args.trainfile,game, agent, False, optimalPolicyFound)

    # # Second Round
    # args.game = "BreakoutNRA"
    # args.game = "Sapientino3C"
    
    game = init_game(args, args.trainfile, agent)
    # game.iteration = args.niter
    # args.niter = args.niter*2
    optimalPolicyFound = learn(args, game, agent, logger)
    writeinfo(args, args.trainfile, game, agent, False, optimalPolicyFound)


    ## OPTION CRITIC
    # agent_args.stateSpace = 1
    # agent_oc = Agent(agent_args)
    # agent_oc.run(game)

print("Experiment terminated after iteration: %d!!!\n" %game.iteration)
#print('saving ...')
#save()
print('Game over')
game.quit()

