import os
from argparse import Namespace
from dataclasses import dataclass
from game import (
    loadGameModule, loadAgentModule, load, writeinfo,
    evaluate, learn
)
from datetime import datetime
from option_critic.option_critic import Agent

@dataclass
class args:
    seed = 42
    game = "BreakoutNRA"
    agent = "Sarsa"
    trainfile = "HRLtest"
    rows = 2
    cols = 5
    gamma = 1.0
    epsilon = 0.2
    alpha = -1
    lambdae = -1
    nstep = 100
    niter = -1
    maxtime = -1

    debug = True
    gui = True
    sound = False
    eval = False
    stopongoal = False

    # Option Critic Args
    num_options=8
    name="OC_9"
    env='N_NRA' 
    logdir='runs' 
    max_episodes=50000
    batch_size=32 
    cuda=True 
    entropy_reg=0.01 
    epsilon_decay=20000
    epsilon_min=0.1
    epsilon_start=1.0 
    freeze_interval=64
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

                 
def init_game(args, trainname):
    game = loadGameModule(args, trainname)
        # set parameters
    game.debug = args.debug
    game.gui_visible = args.gui
    game.sound_enabled = args.sound

    
    game.setRandomSeed(args.seed)
    # game.init(agent)
    return game

# agent = loadAgentModule(args)
# agent.gamma = args.gamma
# agent.epsilon = args.epsilon
# agent.alpha = args.alpha
# agent.nstepsupdates = args.nstep
# agent.lambdae = args.lambdae
# agent.debug = args.debug
# agent.setRandomSeed(args.seed)

if not os.path.exists(f"data/{args.trainfile}/"):
    print("Creating directory")
    os.makedirs(f"data/{args.trainfile}/")

# args_agent = Namespace(batch_size=32, cuda=True, entropy_reg=0.01, env='Breakout', 
#                  epsilon_decay=20000, epsilon_min=0.2, epsilon_start=1.0, 
#                  freeze_interval=64, gamma=0.99, learning_rate=0.0005, 
#                  logdir='runs', max_history=10000, max_steps_ep=18000, max_steps_total=5000,
#                  num_options=2, optimal_eps=0.05, seed=42, temp=1, 
#                  termination_reg=0.01, update_frequency=4,
#                  name="OC_3")



# agent_oc.gamma = args.gamma
game = init_game(args, args.trainfile)
args.stateSpace = 1 #game.getStateSpace()
agent_oc = Agent(args)

# # First round
# # game.RA.type="left2rightx2"
# args.game = "BreakoutA"
# game = init_game(args, args.trainfile)
# game.init(agent_oc)
# agent_oc.run(game)

# Second round
game.RA.type="left2right"
# args.game = "BreakoutNRA"
game = init_game(args, args.trainfile)
game.init(agent_oc)
agent_oc.run(game)

print("FINISHED RUNNING")