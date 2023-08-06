import torch
import torch.nn.functional as F
import logging
import numpy as np

import gym
from gym import spaces
from gym.utils import seeding

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
matplotlib.use('TkAgg')
plt.ion()

# from .specs import LDBA, Spec_Controller


logger = logging.getLogger(__name__)

class Fourrooms(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self, noise=0.3, render=False, seed=42, name="fourrooms"):

        layout = """\
wwwwwwwwwwwww
w     w     w
w     w     w
w           w
w     w     w
w     w     w
ww wwww     w
w     www www
w     w     w
w     w     w
w           w
w     w     w
wwwwwwwwwwwww
"""
        self.occupancy = np.array([list(map(lambda c: 1 if c=='w' else 0, line)) for line in layout.splitlines()])

        # From any state the agent can perform one of four actions, up, down, left or right
        self.action_space = spaces.Discrete(4)
        
        self.observation_space = spaces.Box(low=0., high=1., shape=(np.sum(self.occupancy == 0),))

        self.directions = [np.array((-1,0)), np.array((1,0)), np.array((0,-1)), np.array((0,1))]
        self.rng = np.random.RandomState(1234)

        self.tostate = {}
        statenum = 0
        for i in range(13):
            for j in range(13):
                if self.occupancy[i, j] == 0:
                    self.tostate[(i,j)] = statenum
                    statenum += 1
        self.tocell = {v:k for k,v in self.tostate.items()}

        self.goal = self.tostate[(1,1)] # East doorway
        self.init_states = [self.tostate[(11,11)]]# list(range(self.observation_space.shape[0]))
        if self.goal in self.init_states:
            self.init_states.remove(self.goal)
        self.ep_steps = 0

        self.noise = noise
        self.render_ = render
        
        if self.render_:
            fig, self.ax = plt.subplots()    
            plt.show()

        self.seed(seed)

    def seed(self, seed=None):
        return self._seed(seed)

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def empty_around(self, cell):
        avail = []
        for action in range(self.action_space.n):
            nextcell = tuple(cell + self.directions[action])
            if not self.occupancy[nextcell]:
                avail.append(nextcell)
        return avail

    def reset(self):
        state = self.rng.choice(self.init_states)
        self.currentcell = self.tocell[state]
        self.ep_steps = 0
        return self.get_state(state), None

    def switch_goal(self, goal=None):
        prev_goal = self.goal
        self.goal = goal if goal else self.rng.choice(self.init_states)
        self.init_states.append(prev_goal)
        self.init_states.remove(self.goal)
        assert prev_goal in self.init_states
        assert self.goal not in self.init_states

    def get_state(self, state):
        return np.array(state)

    def render(self, delay=0.01):
        """
        Renders the grid current state.
        """

        grid = np.array(self.occupancy)
        current_pos = self.currentcell
        goal_pos = self.tocell[self.goal]

        current_pos = list(current_pos)
        current_pos.reverse()
        goal_pos = list(goal_pos)
        goal_pos.reverse()

        # Clear the FIG
        self.ax.clear()
        # Plot Grid
        matrice = self.ax.matshow(grid, vmin=0, vmax=1)
        # Circle
        circle = patches.Circle(current_pos, radius=0.25, color='green')
        self.ax.add_patch(circle)
        # Annulus
        annulus = patches.Annulus(goal_pos, r=0.25, width=0.01, color='red')
        self.ax.add_patch(annulus)

        plt.draw()
        plt.pause(delay)

    def step(self, action):
        """
        The agent can perform one of four actions,
        up, down, left or right, which have a stochastic effect. With probability 2/3, the actions
        cause the agent to move one cell in the corresponding direction, and with probability 1/3,
        the agent moves instead in one of the other three directions, each with 1/9 probability. In
        either case, if the movement would take the agent into a wall then the agent remains in the
        same cell.
        We consider a case in which rewards are zero on all state transitions.
        """

        nextcell = tuple(self.currentcell + self.directions[action])
        if not self.occupancy[nextcell]:
            if self.rng.uniform() < self.noise:
                empty_cells = self.empty_around(self.currentcell)
                self.currentcell = empty_cells[self.rng.randint(len(empty_cells))]
            else:
                self.currentcell = nextcell

        state = self.tostate[self.currentcell]
        done = state == self.goal
        reward = float(done)

        if self.render_:
            self.render()

        return self.get_state(state), reward, done, False, None


class RoomsSpec:
    def __init__(self):
        self.num_states = [1]

    def reset(self):
        return [0], True

    def step(self, label):
        if label!="r2": return [0], True
        return [1], False
    
    def label(self, cell):
        x, y = cell

        if x < 6:
            if y < 6:
                # R1, -R2, -R3, -R4
                return "r1"
            else:
                return "r2"
        else:
            if y < 7:
                return "r3"
            else:
                return "r4"
            
class RouteSpec:
    def __init__(self):
        self.route = ["x11y1", "x1y1", "x1y11"]
        self.state = 0
        self.end_state = len(self.route)

    def reset(self):
        self.state = 0
        return self.encode_state(self.state)
    
    def __len__(self):
        return len(self.route)
    
    def encode_state(self, state):
        encoded_state = np.zeros((self.__len__()))
        if state>=self.end_state:
            return encoded_state
        encoded_state[state] = 1
        return encoded_state

    def step(self, label):
        reward = 0
        done = False

        # Check Label transitions the StateMachine
        if label == self.route[self.state]:
            reward += 1
            self.state += 1
            # Check end state Machine
            if self.state == self.end_state:
                done = True
                
        encoded_state = self.encode_state(self.state)
        return encoded_state, reward, done
    
    def label(self, cell):
        x, y = cell
        return f"x{x}y{y}"

class LTLFourrooms(Fourrooms):
    def __init__(self, noise, render, seed, name):
        Fourrooms.__init__(self, noise, render, seed, name)
        self.spec = RouteSpec()
         #Spec_Controller(["G (phi -> (X psi))"], save_to="runs/")
        self.fourrooms_space_len = np.sum(self.occupancy == 0)
        self.observation_space = spaces.Box(
            low=0., high=1., shape=(np.sum(self.occupancy == 0)+len(self.spec),))
                
    def reset(self):
        room_state, info = Fourrooms.reset(self)
        spec_state = self.spec.reset()
        next_prod_state = np.concatenate([room_state, spec_state], 0)
        return next_prod_state, None

    def get_state(self, state):
        s = np.zeros(self.fourrooms_space_len)
        s[state] = 1
        return s

    def step(self, action):

        # Fourroom State
        next_room_step, _, _, _, _  = Fourrooms.step(self, action)

        # Spec State
        labeled_actions = self.spec.label(self.currentcell)
        next_spec_state, reward, done = self.spec.step(labeled_actions)

        # Prod State
        next_prod_state = np.concatenate([next_room_step, next_spec_state], 0)

        return next_prod_state, reward, done, False, None

    def reward(self, acceptances):
        self.rewards = [1 if acc else 0 for acc in acceptances]
        return self.rewards