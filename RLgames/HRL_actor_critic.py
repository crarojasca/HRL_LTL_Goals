import os
from argparse import Namespace
from dataclasses import dataclass
from game import (
    loadGameModule, loadAgentModule, load, writeinfo,
    evaluate, learn
)
from datetime import datetime
from actor_critic.actor_critic import Agent

@dataclass
class args:
    seed = 42
    game = "BreakoutNRA"
    agent = "Sarsa"
    trainfile = "HRLtest"
    rows = 3
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
    name="A2C_2"
    env='N_NRA' 
    logdir='runs' 
    max_episodes=50000
    batch_size=32 
    cuda=True 
    entropy_reg=0.01 
    epsilon_decay=1000000 
    epsilon_min=0.1
    epsilon_start=1.0 
    freeze_interval=10000 
    gamma=0.99 
    learning_rate=0.0001 
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
    if (args.debug):
        game.sleeptime = 1.0
        game.gui_visible = True
    
    game.setRandomSeed(args.seed)
    # game.init(agent)
    return game

if not os.path.exists(f"data/{args.trainfile}/"):
    print("Creating directory")
    os.makedirs(f"data/{args.trainfile}/")

# agent_oc.gamma = args.gamma
game = init_game(args, args.trainfile)
agent_a2c = Agent(
    lr=args.learning_rate,
    input_dims=(game.getStateSpace(), ), 
    n_actions=3,
    args=args
)
agent_a2c.run(game)

# # First round
# # game.RA.type="left2rightx2"
# args.game = "BreakoutA"
# game = init_game(args, args.trainfile)
# game.init(agent_oc)
# agent_oc.run(game)