import os
import torch
import numpy as np

from game import loadGameModule
from dataclasses import dataclass

from option_critic.logger import Logger
from feudalnets.storage import Storage
from feudalnets.feudalnet import FeudalNetwork, feudal_loss
from feudalnets.utils import make_envs, take_action, init_obj

from option_critic.utils import to_tensor

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
    cuda = True

    run_name="baseline"
    seed=42
    num_workers=1
    hidden_dim_manager=256
    hidden_dim_worker=16
    time_horizon=10
    dilation=10
    mlp=True
    max_steps=int(1e8)
    eps=int(1e-5)
    lr=0.0005
    num_steps=400
    max_episodes=5000
    max_steps_ep=20000

                 
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



env = init_game(args, args.trainfile)    

save_steps = list(torch.arange(0, int(args.max_steps),
                                   int(args.max_steps) // 10).numpy())

logger = Logger(
            logdir="runs", 
            run_name=args.run_name)
cuda_is_available = torch.cuda.is_available() and args.cuda
device = torch.device("cuda" if cuda_is_available else "cpu")
args.device = device

torch.manual_seed(args.seed)
if cuda_is_available:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

feudalnet = FeudalNetwork(
    num_workers=args.num_workers,
    input_dim=[env.getStateSpace()],
    hidden_dim_manager=args.hidden_dim_manager,
    hidden_dim_worker=args.hidden_dim_worker,
    n_actions=3,
    time_horizon=args.time_horizon,
    dilation=args.dilation,
    device=device,
    mlp=args.mlp,
    args=args)

optimizer = torch.optim.RMSprop(feudalnet.parameters(), lr=args.lr,
                                    alpha=0.99, eps=1e-5)
goals, states, masks = feudalnet.init_obj()

episodes = 0
steps = 0
rewards = []
while episodes < args.max_episodes:

    # Detaching LSTMs and goals
    feudalnet.repackage_hidden()
    goals = [g.detach() for g in goals]
    storage = Storage(size=args.max_steps_ep,
                        keys=['r', 'r_i', 'v_w', 'v_m', 'logp', 'entropy',
                            's_goal_cos', 'mask', 'ret_w', 'ret_m',
                            'adv_m', 'adv_w'])
    
    env.reset()
    env.draw()
    obs   = env.getstate()
    obs = to_tensor(obs).reshape((1, -1))
    
    
    done = False ; ep_steps = 0 ; cumreward = 0 ; 
                
    while not done and ep_steps < args.max_steps_ep:


        action_dist, goals, states, value_m, value_w \
                 = feudalnet(obs, goals, states, masks[-1])

        # Take a step, log the info, get the next state
        action, logp, entropy = take_action(action_dist)
        # x, reward, done, _, info = env.step(action)

        env.update(action)
        obs = env.getstate()
        obs = to_tensor(obs).reshape((1, -1))
        reward = env.getreward()
        done = env.finished
        env.draw()

        mask = torch.FloatTensor(1 - done).unsqueeze(-1).to(args.device)
        masks.pop(0)
        masks.append(mask)

        storage.add({
            'r': torch.FloatTensor(reward).unsqueeze(-1).to(device),
            'r_i': feudalnet.intrinsic_reward(states, goals, masks),
            'v_w': value_w,
            'v_m': value_m,
            'logp': logp.unsqueeze(-1),
            'entropy': entropy.unsqueeze(-1),
            's_goal_cos': feudalnet.state_goal_cosine(states, goals, masks),
            'm': mask
        })
               
        if steps % args.num_steps == 0 and steps>0:
            with torch.no_grad():
                *_, next_v_m, next_v_w = feudalnet(
                    obs, goals, states, mask, save=False)
                next_v_m = next_v_m.detach()
                next_v_w = next_v_w.detach()

            optimizer.zero_grad()
            loss, loss_dict = feudal_loss(storage, next_v_m, next_v_w, args)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(feudalnet.parameters(), args.grad_clip)
            optimizer.step()

            if len(save_steps) > 0 and steps > save_steps[0]:
                torch.save({
                    'model': feudalnet.state_dict(),
                    'args': args,
                    'processor_mean': feudalnet.preprocessor.rms.mean,
                    'optim': optimizer.state_dict()},
                    f'models/{args.env_name}_{args.run_name}.pt')
                save_steps.pop(0)

        cumreward += reward
        ep_steps += 1
        steps += 1 

    rewards += [cumreward]
    mean_reward = np.mean(rewards[-100:])
    logger.log_episode(episodes, cumreward, mean_reward, {}, steps, 0)
    episodes += 1