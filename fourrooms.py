import torch
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

from specs import LDBA, Spec_Controller

plt.ion()
logger = logging.getLogger(__name__)

class Fourrooms(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self, render):

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

        self.goal = 62 # East doorway
        self.init_states = list(range(self.observation_space.shape[0]))
        self.init_states.remove(self.goal)
        self.ep_steps = 0

        self.render_ = render
        
        if self.render_:
            fig, self.ax = plt.subplots()    
            plt.show()

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
        return self.get_state(state)

    def switch_goal(self):
        prev_goal = self.goal
        self.goal = self.rng.choice(self.init_states)
        self.init_states.append(prev_goal)
        self.init_states.remove(self.goal)
        assert prev_goal in self.init_states
        assert self.goal not in self.init_states

    def get_state(self, state):
        s = np.zeros(self.observation_space.shape[0])
        s[state] = 1
        return s

    def render(self, delay=0.001):
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
        self.ep_steps += 1

        nextcell = tuple(self.currentcell + self.directions[action])
        if not self.occupancy[nextcell]:
            if self.rng.uniform() < 1/3.:
                empty_cells = self.empty_around(self.currentcell)
                self.currentcell = empty_cells[self.rng.randint(len(empty_cells))]
            else:
                self.currentcell = nextcell

        state = self.tostate[self.currentcell]
        done = state == self.goal
        reward = float(done)

        if not done and self.ep_steps >= 1000:
            done = True ; reward = 0.0

        if self.render_:
            self.render()

        return self.get_state(state), reward, done, None


class LTLFourrooms(Fourrooms):
    def __init__(self, render):
        Fourrooms.__init__(self, render)
        self.spec = Spec_Controller(["G (phi -> (X psi))"], save_to="runs/")

    def label(self, actions):
        decode = {62: "phi", 52: "psi", None: None}
        labeled_actions = [(decode[a]) if a in decode else ("") for a in actions]
        return labeled_actions

    def step(self, action):

        # Fourroom State
        next_room_step, _, _, _  = Fourrooms.step(self, action)

        # Spec State
        labeled_actions = self.label([action])
        next_spec_spec, acceptances = self.spec.step(labeled_actions)

        # Prod State
        next_prod_state = torch.cat([torch.tensor(next_room_step)] + next_spec_spec, 0)

        # Rewards
        rewards = self.reward(acceptances)

        return next_prod_state, rewards, False, None

    def reward(self, acceptances):
        rewards = [1 if acc else 0 for acc in acceptances]
        return rewards

if __name__=="__main__":
    env = Fourrooms()
    env.seed(3)
