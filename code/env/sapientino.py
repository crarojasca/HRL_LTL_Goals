import pygame, sys
import numpy as np
import atexit
import random
import time
import math
from math import fabs

import gym
from gym import spaces
from gym.utils import seeding
from collections import namedtuple

COLORS = ['red', 'green', 'blue', 'pink', 'brown', 'gray', 'purple' ]

TOKENS = [ ['r1', COLORS[0], 0, 0],  ['r2', 'red', 1, 1], ['r3', 'red', 6, 3],   
    ['g1', 'green', 4, 0], ['g2', 'green', 5, 2], ['g3', 'green', 5, 4],
    ['b1', 'blue', 1, 3], ['b2', 'blue', 2, 4],  ['b3', 'blue', 6, 0], 
    ['p1', 'pink', 2, 1], ['p2', 'pink', 2, 3], ['p3', 'pink', 4, 2], 
    ['n1', 'brown', 3, 0], ['n2', 'brown', 3, 4], ['n3', 'brown', 6, 1],
    ['y1', 'gray', 0, 2], ['y2', 'gray', 3, 1], ['y3', 'gray', 4, 3],
    ['u1', 'purple', 0, 4], ['u2', 'purple', 1, 0], ['u3', 'purple', 5, 1]
]

# only positive rewards

STATES = {
    'Init':0,
    'Alive':0,
    'Dead':-1,
    'Score':0,
    'Hit':0,
    'GoodColor':0,
    'GoalStep':100,
    'RAFail':-1,
    'RAGoal':1000
}

# Reward automa

class RewardAutoma(object):

    def __init__(self, ncol, nvisitpercol):
        # RA states
        self.ncolors = ncol
        self.nvisitpercol = nvisitpercol
        self.nRAstates = int(math.pow(2,self.ncolors*3)+2)  # number of RA states
        self.RAGoal = self.nRAstates
        self.RAFail = self.nRAstates+1        
        self.goalreached = 0 # number of RA goals reached for statistics
        self.visits = {} # number of visits for each RA state
        self.success = {} # number of good transitions for each RA state
        self.reward_shaping_enabled = False
        self.reset()

    def init(self, game):
        self.game = game
        
    def reset(self):
        self.RAnode = 0
        self.last_node = self.RAnode
        self.past_colors = []
        self.consecutive_turns = 0 # number of consecutive turn actions
        self.countupdates = 0 # count state transitions (for the score)
        if (self.RAnode in self.visits):
            self.visits[self.RAnode] += 1
        else:
            self.visits[self.RAnode] = 1

    def encode_tokenbip(self):
        c = 0
        b = 1
        for t in TOKENS:
            c = c + self.game.tokenbip[t[0]] * b
            b *= 2
        return c

    # RewardAutoma Transition
    def update(self, a=None): # last action executed
        reward = 0
        state_changed = False
        self.last_node = self.RAnode

        # check consecutive turns in differential mode
        if (a == 0 or a == 1): # turn left/right
            self.consecutive_turns += 1
        else:
            self.consecutive_turns = 0

        if (self.consecutive_turns>=4):
            self.RAnode = self.RAFail  # FAIL
            reward += STATES['RAFail']   

        # check double bip
        for t in self.game.tokenbip:
            if self.game.tokenbip[t]>1:                
                self.RAnode = self.RAFail  # FAIL
                reward += STATES['RAFail']
                #print("  *** RA FAIL (two bips) *** ")


        if (self.RAnode != self.RAFail):
            self.RAnode = self.encode_tokenbip()

            #print("  -- encode tokenbip: %d" %self.RAnode)
            # Check rule
            # nvisitpercol
            c = np.zeros(self.ncolors)
            kc = -1
            #print(self.game.colorbip)
            for i in range(len(COLORS)):
                if (self.game.colorbip[COLORS[i]]>self.nvisitpercol):
                    self.RAnode = self.RAFail
                    break
                elif (self.game.colorbip[COLORS[i]]<self.nvisitpercol):
                    break
                kc = i # last color with nvisitsper col satisfied
            #print("%d visits until color %d" %(self.nvisitpercol,kc))

            if (kc==self.ncolors-1): #  GOAL
                self.RAnode = self.RAGoal

            # check bips in colors >= kc+2
            if (self.RAnode != self.RAFail and self.RAnode != self.RAGoal):
                for i in range(kc+2,len(COLORS)):
                    if (self.game.colorbip[COLORS[i]]>0):
                        #print("RA failure for color %r" %i)
                        self.RAnode = self.RAFail
                        break


            if (self.last_node != self.RAnode):
                state_changed = True
                #print("  ++ changed state ++")
                if (self.RAnode == self.RAFail):
                    reward += STATES['RAFail']
                #elif (self.last_id_colvisited != kc): # new state in which color has been visited right amunt of time
                #    self.last_id_colvisited = kc
                #    reward += STATES['GoalStep']
                else: # new state good for the goal
                    self.countupdates += 1
                    if self.reward_shaping_enabled:
                        rs = self.reward_shape(self.last_node, self.RAnode)
                        #print(' -- added reward shape F(%d,a,%d) = %f ' %(self.last_node, self.RAnode, rs))
                        reward += rs
                    else:
                        #reward += STATES['GoalStep']
                        reward += self.countupdates * STATES['GoalStep']
                if (self.RAnode == self.RAGoal): #  GOAL
                    reward += STATES['RAGoal']
                    #print("RAGoal")

        #print("  -- RA reward %d" %(reward))

        if (state_changed):
            if (self.RAnode in self.visits):
                self.visits[self.RAnode] += 1
            else:
                self.visits[self.RAnode] = 1

            if (self.RAnode != self.RAFail):
                #print("Success for last_node ",self.last_node)
                if (self.last_node in self.success):
                    self.success[self.last_node] += 1
                else:
                    self.success[self.last_node] = 1
        
        return (reward, state_changed)

    def current_successrate(self):
        s = 0.0
        v = 1.0
        if (self.RAnode in self.success):
            s = float(self.success[self.RAnode])
        if (self.RAnode in self.visits):
            v = float(self.visits[self.RAnode])
        #print("   -- success rate: ",s," / ",v)
        return s/v


    def print_successrate(self):
        r = []
        for i in range(len(self.success)):
            v = 0
            if (i in self.success):
                v = float(self.success[i])/self.visits[i]
            r.append(v)
        print('RA success: %s' %str(r))


    # TODO reward shaping function
    def reward_shape(self, s, snext):
        egamma = math.pow(0.99, 10) # estimated discount to reach a new RA state
        return egamma * self.reward_phi(snext) - self.reward_phi(s)


    # TODO reward shaping function
    def reward_phi(self, state):
        # state = current node (encoding of tokenbip)        
        return state * 100


class Sapientino(object):

    def __init__(self, seed=42, rows=5, cols=7, name=None, trainsessionname='test', 
                 ncol=7, nvisitpercol=2, type_state="features", render=False):

        if seed:
            random.seed(seed)
            np.random.seed(seed)

        self.isAuto = True
        self.gui_visible = render
        self.userquit = False
        self.optimalPolicyUser = False  # optimal policy set by user
        self.trainsessionname = trainsessionname
        self.rows = rows
        self.cols = cols
        self.nvisitpercol = nvisitpercol
        self.ncolors = ncol
        self.differential = False
        self.colorsensor = True
        self.type_state = type_state
        
        # Configuration
        self.pause = False # game is paused
        self.debug = False
        
        self.sleeptime = 0.0
        self.command = 0
        self.iteration = 0
        self.score = 0
        self.cumreward = 0
        self.cumreward100 = 0 # cumulative reward for statistics
        self.cumscore100 = 0 
        self.ngoalreached = 0
        self.numactions = 0 # number of actions in this run
        self.reward_shaping_enabled = False

        self.hiscore = 0
        self.hireward = -1000000
        self.elapsedtime = 0 # elapsed time for this experiment

        self.win_width = 480
        self.win_height = 520

        self.size_square = 40
        self.offx = 40
        self.offy = 100
        self.radius = 5

        self.action_names = ['<-','->','^','v','x']

        if (self.cols>10):
            self.win_width += self.size_square * (self.cols-10)
        if (self.rows>10):
            self.win_height += self.size_square * (self.rows-10)

        self.RA_exploration_enabled = False  # Use options to speed-up learning process
        self.report_str = ''

        pygame.init()
        pygame.display.set_caption('Sapientino')

        flags = pygame.HIDDEN if (not self.gui_visible) else pygame.SHOWN
        self.screen = pygame.display.set_mode([self.win_width,self.win_height], flags=flags)
        self.myfont = pygame.font.SysFont("Arial",  30)

        self.nactions = 5  # 0: left, 1: right, 2: up, 3: down, 4: bip

        self.RA = RewardAutoma(self.ncolors, self.nvisitpercol)
        self.RA.init(self)

        self.nstates = self.rows * self.cols
        if (self.differential):
            self.nstates *= 4
        if (self.colorsensor):
            self.nstates *= self.ncolors+1

        self.action_space = spaces.Discrete(5)

        State_cfg = namedtuple("State_cfg", ["observation_space", "get_state"])
        self.state_cfg = {
            "integer": State_cfg(
                observation_space=spaces.Discrete(1), get_state=self.integerState),
            "features": State_cfg(
                spaces.Box(low=0., high=1., shape=(4 + len(TOKENS) + len(COLORS),)), 
                self.featuresState),
            # "screen": State_cfg(
            #     observation_space={
            #         "env": spaces.Box(low=0., high=255., shape=(self.win_height, self.win_height, 3)),
            #         "spec": spaces.Discrete(1)
            #     }, 
            #     get_state=self.screenState),
        }

        self.observation_space = self.state_cfg[self.type_state].observation_space

        self.step_cell_bip = None

      

    def reset(self):
        
        self.pos_x = 3
        self.pos_y = 2
        self.pos_th = 90

        self.score = 0
        self.cumreward = 0
        self.cumscore = 0  
        self.gamman = 1.0 # cumulative gamma over time
        self.current_reward = 0 # accumulate reward over all events happened during this action until next different state

        self.prev_state = 0 # previous state
        self.firstAction = True # first action of the episode
        self.finished = False # episode finished
        self.newstate = True # new state reached
        self.numactions = 0 # number of actions in this episode
        self.iteration += 1

        self.tokenbip = {}
        self.colorbip = {}        
        for t in TOKENS:
            self.tokenbip[t[0]] = 0
            self.colorbip[t[1]] = 0
        self.countbip=0
        # self.RA.reset()

        # RA exploration
        # self.RA_exploration()
        self.draw()

        self.step_cell_bip = None

        return self.getstate(), None

        
    def getSizeStateSpace(self):
        return self.nstates


    def integerState(self):
        x = self.pos_x + self.cols * self.pos_y
        if (self.differential):
            x += (self.pos_th/90) * (self.rows * self.cols)
        if (self.colorsensor):
            x += self.encode_color() * (self.rows * self.cols * 4)
        # x += self.nstates * self.RA.RAnode     
        return [x]
    
    def featuresState(self):
        pos = [self.pos_x, self.pos_y]
        diff = [self.pos_th/90]
        color = [self.encode_color()]

        # Tokens
        tokens = list(self.tokenbip.values())
        # Colors
        colors = list(self.colorbip.values())

        # ra_state = [self.RA.RAnode]

        state = pos + diff + color + tokens + colors #+ ra_state
        return state
    
    # def screenState(self):

    #     screen_state = pygame.surfarray.array3d(pygame.display.get_surface())
    #     screen_state = screen_state.swapaxes(0,1)
    #     # ra_state = np.array([self.RA.RAnode])

    #     state = {
    #         "env": screen_state,
    #         "spec": ra_state
    #     }

    #     return screen_state
    
    def getstate(self):
        return self.state_cfg[self.type_state].get_state()

    def goal_reached(self):
        return self.RA.RAnode==self.RA.RAGoal


    def update_color(self):
        self.countbip += 1
        colfound = None
        for t in TOKENS:
            if (self.pos_x == t[2] and self.pos_y == t[3]):
                self.tokenbip[t[0]] += 1 # token id
                self.colorbip[t[1]] += 1 # color
                colfound = t[1]
        #print ("pos %d %d %d - col %r" %(self.pos_x, self.pos_y, self.pos_th, colfound))


    def check_color(self):
        r = ' '
        for t in TOKENS:
            if (self.pos_x == t[2] and self.pos_y == t[3]):
                r = t[1]
                break
        return r

 
    def encode_color(self):
        r = 0
        for t in TOKENS:
            r += 1
            if (self.pos_x == t[2] and self.pos_y == t[3]):
                break
        return r

    # def RA_exploration(self):
    #     if not self.RA_exploration_enabled:
    #         return
    #     #print("RA state: ",self.RA.RAnode)
    #     success_rate = max(min(self.RA.current_successrate(),0.9),0.1)
    #     #print("RA exploration policy: current state success rate ",success_rate)
    #     er = random.random()
    #     #print("RA exploration policy: optimal ",self.agent.partialoptimal, "\n")
        
    def step(self, action):
        
        self.step_cell_bip = None
        self.command = action

        self.prev_state = self.getstate() # remember previous state
        
        # print(" == Update start ",self.prev_state," action",self.command)
        
        self.current_reward = 0 # accumulate reward over all events happened during this action until next different state
        self.numactions += 1 # total number of actions axecuted in this episode
        
        white_bip = False
        
        if (self.firstAction):
            self.firstAction = False
            self.current_reward += STATES['Init']
        

        if (not self.differential):
            # omni directional motion
            if self.command == 0: # moving left
                self.pos_x -= 1
                if (self.pos_x < 0):
                    self.pos_x = 0 
                    self.current_reward += STATES['Hit']
            elif self.command == 1:  # moving right
                self.pos_x += 1
                if (self.pos_x >= self.cols):
                    self.pos_x = self.cols-1
                    self.current_reward += STATES['Hit']
            elif self.command == 2:  # moving up
                self.pos_y += 1
                if (self.pos_y >= self.rows):
                    self.pos_y = self.rows-1
                    self.current_reward += STATES['Hit']
            elif self.command == 3:  # moving down
                self.pos_y -= 1
                if (self.pos_y< 0):
                    self.pos_y = 0 
                    self.current_reward += STATES['Hit']
        else:
            # differential motion
            if self.command == 0: # turn left
                self.pos_th += 90
                if (self.pos_th >= 360):
                    self.pos_th -= 360
                #print ("left") 
            elif self.command == 1:  # turn right
                self.pos_th -= 90
                if (self.pos_th < 0):
                    self.pos_th += 360 
                #print ("right") 
            elif (self.command == 2 or self.command == 3):
                dx = 0
                dy = 0
                if (self.pos_th == 0): # right
                    dx = 1
                elif (self.pos_th == 90): # up
                    dy = 1
                elif (self.pos_th == 180): # left
                    dx = -1
                elif (self.pos_th == 270): # down
                    dy = -1
                if (self.command == 3):  # backward
                    dx = -dx
                    dy = -dy
                    #print ("backward") 
                #else:
                    #print ("forward") 
        
                self.pos_x += dx
                if (self.pos_x >= self.cols):
                    self.pos_x = self.cols-1
                    self.current_reward += STATES['Hit']
                if (self.pos_x < 0):
                    self.pos_x = 0 
                    self.current_reward += STATES['Hit']
                self.pos_y += dy
                if (self.pos_y >= self.rows):
                    self.pos_y = self.rows-1
                    self.current_reward += STATES['Hit']
                if (self.pos_y < 0):
                    self.pos_y = 0 
                    self.current_reward += STATES['Hit']


        #print ("pos %d %d %d" %(self.pos_x, self.pos_y, self.pos_th))

        if self.command == 4:  # bip
            self.update_color()
            self.step_cell_bip = (self.pos_x, self.pos_y)
            if (self.check_color()!=' '):
                pass
                #self.current_reward += STATES['Score']
                #if self.debug:
                #    print("bip on color")
            else:
                white_bip = True


        self.current_reward += STATES['Alive']


        # if (self.differential):
        #     (RAr,state_changed) = self.RA.update(a)  # consider also turn actions
        # else:
        #     (RAr,state_changed) = self.RA.update()

        # self.current_reward += RAr

        # RA exploration
        # if (state_changed):
        #     self.RA_exploration()

        # set score
        # RAnode = self.RA.RAnode
        # if (RAnode==self.RA.RAFail):
        #     RAnode = self.RA.last_node

        # self.score = self.RA.countupdates

        
        # check if episode finished
        if self.goal_reached():
            self.current_reward += STATES['Score']
            self.ngoalreached += 1
            self.finished = True
        if (self.numactions>(self.cols*self.rows)*10):
            self.current_reward += STATES['Dead']
            self.finished = True
        # if (self.RA.RAnode==self.RA.RAFail):
        #     self.finished = True
        if (white_bip):
            self.current_reward += STATES['Dead']
            self.finished = True
           

        if (not self.finished and self.reward_shaping_enabled):
            self.current_reward += self.reward_shape(self.prev_state, self.getstate())

        # Updates the screen
        self.draw()

        return self.getstate(), self.current_reward, self.finished, False, None




    # reward shaping function
    def reward_shape(self, s, snext):
        return self.agent.gamma * self.reward_phi(snext) - self.reward_phi(s)


    # reward shaping function
    def reward_phi(self, state):
        # state = current node (encoding of tokenbip)
        RAstate = int(state / self.nstates)
        return RAstate


    def draw(self):
        pygame.event.get()
        self.screen.fill(pygame.color.THECOLORS['white'])

        score_label = self.myfont.render(str(self.score), 100, pygame.color.THECOLORS['black'])
        self.screen.blit(score_label, (20, 10))

        #count_label = self.myfont.render(str(self.paddle_hit_count), 100, pygame.color.THECOLORS['brown'])
        #self.screen.blit(count_label, (70, 10))

        x = self.getstate()
        cmd = ' '
        if self.command==0:
            cmd = '<'
        elif self.command==1:
            cmd = '>'
        elif self.command==2:
            cmd = '^'
        elif self.command==3:
            cmd = 'v'
        elif self.command==4:
            cmd = 'x'
        #s = '%d %s %d' %(self.prev_state,cmd,x)
        s = '%s' %(cmd,)
        count_label = self.myfont.render(s, 100, pygame.color.THECOLORS['brown'])
        self.screen.blit(count_label, (60, 10))
        

        if self.isAuto is True:
            auto_label = self.myfont.render("Auto", 100, pygame.color.THECOLORS['red'])
            self.screen.blit(auto_label, (self.win_width-200, 10))


        
        # grid
        for i in range (0,self.cols+1):
            ox = self.offx + i*self.size_square
            pygame.draw.line(self.screen, pygame.color.THECOLORS['black'], [ox, self.offy], [ox, self.offy+self.rows*self.size_square])
        for i in range (0,self.rows+1):
            oy = self.offy + i*self.size_square
            pygame.draw.line(self.screen, pygame.color.THECOLORS['black'], [self.offx , oy], [self.offx + self.cols*self.size_square, oy])


        # color tokens
        for t in TOKENS:
            tk = t[0]
            col = t[1]
            u = t[2]
            v = t[3]
            dx = int(self.offx + u * self.size_square)
            dy = int(self.offy + (self.rows-v-1) * self.size_square)
            sqsz = (dx+5,dy+5,self.size_square-10,self.size_square-10)
            pygame.draw.rect(self.screen, pygame.color.THECOLORS[col], sqsz)
            if (self.tokenbip[tk]==1):
                pygame.draw.rect(self.screen, pygame.color.THECOLORS['black'], (dx+15,dy+15,self.size_square-30,self.size_square-30))

        # agent position
        dx = int(self.offx + self.pos_x * self.size_square)
        dy = int(self.offy + (self.rows-self.pos_y-1) * self.size_square)
        pygame.draw.circle(self.screen, pygame.color.THECOLORS['orange'], [int(dx+self.size_square/2), int(dy+self.size_square/2)], 2*self.radius, 0)

        # agent orientation

        ox = 0
        oy = 0
        if (self.pos_th == 0): # right
            ox = self.radius
        elif (self.pos_th == 90): # up
            oy = -self.radius
        elif (self.pos_th == 180): # left
            ox = -self.radius
        elif (self.pos_th == 270): # down
            oy = self.radius

        pygame.draw.circle(self.screen, pygame.color.THECOLORS['black'], [int(dx+self.size_square/2+ox), int(dy+self.size_square/2+oy)], 5, 0)

        pygame.display.update()


    def quit(self):
        pygame.quit()


COLORS = ['red', 'green', 'blue', 'pink', 'brown', 'gray', 'purple' ]

TOKENS = [ ['r1', COLORS[0], 0, 0],  ['r2', 'red', 1, 1], ['r3', 'red', 6, 3],   
    ['g1', 'green', 4, 0], ['g2', 'green', 5, 2], ['g3', 'green', 5, 4],
    ['b1', 'blue', 1, 3], ['b2', 'blue', 2, 4],  ['b3', 'blue', 6, 0], 
    ['p1', 'pink', 2, 1], ['p2', 'pink', 2, 3], ['p3', 'pink', 4, 2], 
    ['n1', 'brown', 3, 0], ['n2', 'brown', 3, 4], ['n3', 'brown', 6, 1],
    ['y1', 'gray', 0, 2], ['y2', 'gray', 3, 1], ['y3', 'gray', 4, 3],
    ['u1', 'purple', 0, 4], ['u2', 'purple', 1, 0], ['u3', 'purple', 5, 1]
]

class ColorSpec():

    def __init__(self, colors, tokens) -> None:

        self.colors = colors
        self.tokens = {} 

        for token in tokens:
            color = token[1]
            x, y = token[2], token[3]

            if color in self.tokens:
                self.tokens[color].append((x, y))

        self.current_color = None


    def step(self, cell_bip):

        pass

        

class LTLSapientino(Sapientino):

    def __init__(self, seed=42, rows=5, cols=7, name=None, trainsessionname='test', 
                 ncol=7, nvisitpercol=2, type_state="features", render=False):
        
        super().__init__(seed, rows, cols, name, trainsessionname, ncol, 
                         nvisitpercol, type_state, render)
        
        self.spec = self.step_cell_bip = None
        
    def step(self, action):
        return super().step(action)