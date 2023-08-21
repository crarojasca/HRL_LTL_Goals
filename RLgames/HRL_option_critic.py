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
    gamma = 0.999
    epsilon = 0.2
    alpha = -1
    lambdae = -1
    nstep = 100
    niter = 5000
    maxtime = -1

    debug = False
    gui = True
    sound = False
    eval = False
    stopongoal = False

    # Option Critic Args
    num_options=8
    name="OC_4"
    env='OriginalRewards' 
    logdir='runs' 
    batch_size=32 
    cuda=True 
    entropy_reg=0.01 
    epsilon_decay=20000 
    epsilon_min=0.2 
    epsilon_start=1.0 
    freeze_interval=64 
    gamma=0.99 
    learning_rate=0.0005 
    max_history=10000 
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
    return game

if not os.path.exists(f"data/{args.trainfile}/"):
    print("Creating directory")
    os.makedirs(f"data/{args.trainfile}/")

agent_oc = Agent(args, 5000)
agent_oc.gamma = args.gamma
game = init_game(args, args.trainfile, agent_oc)

# First round
# game.RA.type="left2rightx2"
agent_oc.run(game)

# # Second round
# game.RA.type="left2right"
# agent_oc.run(game)