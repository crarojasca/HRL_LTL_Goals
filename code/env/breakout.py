import pygame
import numpy as np
import random
import math
from math import fabs

import gym
from gym import spaces
from gym.utils import seeding


black = [0, 0, 0]
white = [255,255,255]
grey = [180,180,180]
orange = [180,100,20]
red = [180,0,0]

# game's constant variables
ball_radius = 10
paddle_width = 80
paddle_height = 10

block_width = 60
block_height = 12
block_xdistance = 20
            
resolutionx = 20
resolutiony = 10


class Brick(object):

    def __init__(self, i, j):
        self.i = i
        self.j = j
        self.x = (block_width+block_xdistance)*i+block_xdistance
        self.y = 70+(block_height+8)*j
        self.rect = pygame.Rect(self.x, self.y, block_width, block_height)

class Breakout(gym.Env):

    def __init__(self, brick_rows=3, brick_cols=3, fire_enabled=False, 
                 seed=42, render=False, name="breakout"):

        random.seed(seed)
        np.random.seed(seed)

        self.render = render
        self.isAuto = True
        self.gui_visible = False
        self.sound_enabled = True
        self.fire_enabled = fire_enabled
        self.userquit = False
        self.name = name
        self.brick_rows = brick_rows
        self.brick_cols = brick_cols

        if (self.brick_cols<5):
            self.block_xdistance = 50

        self.STATES = {
            'Init':0,
            'Alive':0,
            'Dead':0,
            'PaddleNotMoving':0,
            'Scores':10,    # brick removed
            'Hit':0,        # paddle hit
            'Goal':0,     # level completed
        }
        
        # Configuration
        self.deterministic = True   # deterministic ball bouncing
        self.simple_state = False   # simple = do not consider paddle x
        self.paddle_normal_bump = True  # only left/right bounces
        self.paddle_complex_bump = False  # straigth/left/right complex bounces
        
        self.init_ball_speed_x = 2
        self.init_ball_speed_y = 5
        self.accy = 1.00
        self.score = 0
        self.ball_hit_count = 0
        self.brick_hit_count = 0
        self.paddle_hit_count = 0
        self.action = 0
        self.iteration = 0
        self.ngoalreached = 0 # number of goals reached for stats
        self.numactions = 0 # number of actions in this run

        self.action_names = ['--','<-','->','o'] # stay, left, right, fire

        # firing variables
        self.fire_posx = 0
        self.fire_posy = 0
        self.fire_speedy = 0 # 0 = not firing, <0 firing up

        self.win_width = int((block_width + block_xdistance) * self.brick_cols + block_xdistance )
        self.win_height = 480

        pygame.init()
        pygame.display.set_caption('Breakout')
        
        #allows for holding of key
        pygame.key.set_repeat(1,0)

        flags = pygame.HIDDEN if (not self.gui_visible) else pygame.SHOWN
        self.screen = pygame.display.set_mode([self.win_width,self.win_height], flags=flags)
        self.myfont = pygame.font.SysFont("Arial",  30)

        self.se_brick = None
        self.se_wall = None
        self.se_paddle = None

        self.hit_brick = None

        if self.fire_enabled:
            self.action_space = spaces.Discrete(4)
        else:
            self.action_space = spaces.Discrete(3)
            
        self.observation_space = spaces.Box(
            low=0., high=1., shape=(5 + self.brick_rows*self.brick_cols,))
    
    def initBricks(self):
        self.bricks = []
        self.bricksgrid = np.zeros((self.brick_cols, self.brick_rows))
        for i in range(0,self.brick_cols):
            for j in range(0,self.brick_rows):
                temp = Brick(i,j)
                self.bricks.append(temp)
                self.bricksgrid[i][j]=1

        
    def reset(self):
        self.ball_x = self.win_width/2
        self.ball_y = self.win_height-100-ball_radius
        self.ball_speed_x = self.init_ball_speed_x
        self.ball_speed_y = self.init_ball_speed_y

        self.randomAngle('i')

        self.paddle_x = self.win_width/2
        self.paddle_y = self.win_height-20
        self.paddle_speed = 10 # same as resolution
        self.com_vec = 0

        self.score = 0
        self.ball_hit_count = 0
        self.paddle_hit_count = 0
        self.brick_hit_count = 0
        self.gamman = 1.0 # cumulative gamma over time

        self.paddle_hit_without_brick = 0
        
        self.current_reward = 0 # accumulate reward over all events happened during this action until next different state
        
        self.prev_state = None # previous state
        self.firstAction = True # first action of the episode
        self.finished = False # episode finished
        self.current_state = None # currrent state
        self.numactions = 0 # number of actions in this run
        self.iteration += 1
        self.hit_brick = None
        
        self.initBricks()

        # firing variables
        self.fire_posx = 0
        self.fire_posy = 0
        self.fire_speedy = 0 # 0 = not firing, <0 firing up

        self.current_state = self.getState()

        return self.current_state, None


    def goal_reached(self):
        return len(self.bricks) == 0
        
    def step(self, a):
        
        self.action = a

        self.prev_state = self.current_state # remember previous state
        
        self.current_reward = 0 # accumulate reward over all events happened during this action until next different state
        self.numactions += 1
        self.last_brikcsremoved = []

        while (self.prev_state == self.current_state):
        
            if (self.firstAction):
                self.current_reward += self.STATES['Init']
                self.firstAction = False
            
            if self.action == 0:  # not moving
                # do nothing
                self.current_reward += self.STATES['PaddleNotMoving']
                pass
            elif self.action == 1:  # moving left
                self.paddle_x -= self.paddle_speed
            elif self.action == 2:  # moving right
                self.paddle_x += self.paddle_speed                

            if self.paddle_x < 0:
                self.paddle_x = 0
            if self.paddle_x > self.screen.get_width() - paddle_width:
                self.paddle_x = self.screen.get_width() - paddle_width

            if self.action == 3:  # fire
                if (self.fire_speedy==0):
                    self.fire_posx = self.paddle_x + paddle_width/2
                    self.fire_posy = self.paddle_y
                    if (self.init_ball_speed_y>0):
                        self.fire_speedy = -self.init_ball_speed_y*2
                    else:
                        self.fire_speedy = -10


            self.current_reward += self.STATES['Alive']
            ##MOVE THE BALL
            self.ball_y += self.ball_speed_y
            self.ball_x += self.ball_speed_x

            # firing
            if (self.fire_speedy < 0):
                self.fire_posy += self.fire_speedy

            self.hitDetect()

            self.current_state = self.getState()

        if self.render:
            self.draw()

        return self.current_state, self.current_reward, self.finished, False, None
    

    def getState(self):
        ball_pos = [self.ball_x, self.ball_y]
        ball_speed = [self.ball_speed_x, self.ball_speed_y]
        paddle = [self.paddle_x]
        bricksgrid = [
            self.bricksgrid[i][j] for i in range(0, self.brick_cols) for j in range(0, self.brick_rows)]
        state = ball_pos + ball_speed + paddle + bricksgrid
        return state


    def randomAngle(self, ev):
        if (ev=='i'): # init
            self.randomAngle1()
        if (ev=='b'): # brick hit
            self.randomAngle2()
        if (ev=='i'): # paddle hit
            self.randomAngle3()
        
    def randomAngle1(self):
        if (not self.deterministic):
            ran = random.uniform(0.75, 1.5)
            self.ball_speed_x *= ran
            #print("random ball_speed_x = %.2f" %self.ball_speed_x)

    def randomAngle2(self):
        if (not self.deterministic):
            ran = random.uniform(0.0, 1.0)
            if (random.uniform(0.0, 1.0) < 0.5):
                self.ball_speed_x *= -1
            #print("random ball_speed_x = %.2f" %self.ball_speed_x)

    def randomAngle3(self):
        if (not self.deterministic):
            ran = random.uniform(0.0, 1.0)
            if (random.uniform(0.0, 1.0) < 0.1):
                self.ball_speed_x *= 0.75
            elif (random.uniform(0.0, 1.0) > 0.9):
                self.ball_speed_x *= 1.5
            sign = self.ball_speed_x/abs(self.ball_speed_x)    
            self.ball_speed_x = min(self.ball_speed_x,6)*sign
            self.ball_speed_x = max(self.ball_speed_x,0.5)*sign
            #print("random ball_speed_x = %.2f" %self.ball_speed_x)
            
    def hitDetect(self):
        ##COLLISION DETECTION
        ball_rect = pygame.Rect(self.ball_x-ball_radius, self.ball_y-ball_radius, ball_radius*2,ball_radius*2) #circles are measured from the center, so have to subtract 1 radius from the x and y
        paddle_rect = pygame.Rect(self.paddle_x, self.paddle_y, paddle_width, paddle_height)

        fire_rect = pygame.Rect(self.fire_posx-1, self.fire_posy-1, 3, 3)

        # TERMINATION OF EPISODE
        if (not self.finished):
            #check if the ball is off the bottom of the self.screen
            end1 = self.ball_y > self.screen.get_height() - ball_radius
            end2 = self.goal_reached()
            end3 = self.paddle_hit_without_brick == 30
            if (end1 or end2 or end3):
                if (pygame.display.get_active() and (not self.se_wall is None)):
                    self.se_wall.play()
                if (end1):    
                    self.current_reward += self.STATES['Dead']
                if (end2):
                    self.ngoalreached += 1
                    self.current_reward += self.STATES['Goal']

                self.finished = True # game will be reset at the beginning of next iteration
                return 
        
        #for screen border
        if self.ball_y < ball_radius:
            self.ball_y = ball_radius
            self.ball_speed_y = -self.ball_speed_y
            if (pygame.display.get_active() and (not self.se_wall is None)):
                self.se_wall.play()
        if self.ball_x < ball_radius:
            self.ball_x = ball_radius
            self.ball_speed_x = -self.ball_speed_x
            if (pygame.display.get_active() and (not self.se_wall is None)):
                self.se_wall.play()
        if self.ball_x > self.screen.get_width() - ball_radius:
            self.ball_x = self.screen.get_width() - ball_radius
            self.ball_speed_x = -self.ball_speed_x
            if (pygame.display.get_active() and (not self.se_wall is None)):
                self.se_wall.play()

        #for paddle
        if ball_rect.colliderect(paddle_rect):
            if (self.paddle_complex_bump):
                dbp = math.fabs(self.ball_x-(self.paddle_x+paddle_width/2))
                if (dbp<20):
                    #print 'straight'
                    if (self.ball_speed_x<-5):
                        self.ball_speed_x += 2
                    elif (self.ball_speed_x>5):
                        self.ball_speed_x -= 2
                    elif (self.ball_speed_x<=-0.5): 
                        self.ball_speed_x += 0.5
                    elif (self.ball_speed_x>=0.5): 
                        self.ball_speed_x -= 0.5

                dbp = math.fabs(self.ball_x-(self.paddle_x+0))
                if (dbp<10):
                    #print 'left' 
                    self.ball_speed_x = -abs(self.ball_speed_x)-1
                dbp = math.fabs(self.ball_x-(self.paddle_x+paddle_width))
                if (dbp<10):
                    #print 'right'
                    self.ball_speed_x = abs(self.ball_speed_x)+1

            elif (self.paddle_normal_bump):
                dbp = math.fabs(self.ball_x-(self.paddle_x+paddle_width/2))
                if (dbp<20):
                    #print 'straight'
                    if (self.ball_speed_x!=0):
                        self.ball_speed_x = 2*abs(self.ball_speed_x)/self.ball_speed_x
                dbp = math.fabs(self.ball_x-(self.paddle_x+0))
                if (dbp<20):
                    #print 'left' 
                    self.ball_speed_x = -5
                    self.randomAngle('p')
                dbp = math.fabs(self.ball_x-(self.paddle_x+paddle_width))
                if (dbp<20):
                    #print 'right'
                    self.ball_speed_x = 5
                    self.randomAngle('p')
                    
            self.ball_speed_y = - abs(self.ball_speed_y)
            self.current_reward += self.STATES['Hit']
            self.ball_hit_count +=1
            self.paddle_hit_count +=1
            self.paddle_hit_without_brick += 1
            if (pygame.display.get_active() and (not self.se_wall is None)):
                self.se_paddle.play()

            # reset after paddle hits the ball
            if len(self.bricks) == 0:
                self.initBricks()
                self.ball_speed_y = self.init_ball_speed_y
                
        self.hit_brick = None
        for brick in self.bricks:
            if brick.rect.colliderect(ball_rect):
                #print 'brick hit ',brick.i,brick.j
                if ((not self.se_brick is None)):  #pygame.display.get_active() and 
                    self.se_brick.play()
                    #print('self.se_brick.play()')

                self.score = self.score + 1
                self.brick_hit_count += 1
                self.bricks.remove(brick)
                self.last_brikcsremoved.append(brick)
                self.bricksgrid[(brick.i,brick.j)] = 0
                self.ball_speed_y = -self.ball_speed_y
                self.current_reward += self.STATES['Scores']
                self.paddle_hit_without_brick = 0
                self.hit_brick = brick
                break

        #firing
        if (self.fire_posy < 5):
            #reset
            self.fire_posx = 0
            self.fire_posy = 0
            self.fire_speedy = 0

        for brick in self.bricks:
            if brick.rect.colliderect(fire_rect):
                #print 'brick hit with fire ',brick.i,brick.j
                if (pygame.display.get_active() and (not self.se_wall is None)):
                    self.se_brick.play()
                self.score = self.score + 1
                self.bricks.remove(brick)
                self.last_brikcsremoved.append(brick)
                self.bricksgrid[(brick.i,brick.j)] = 0
                self.current_reward += self.STATES['Scores']
                self.paddle_hit_without_brick = 0
                # reset firing
                self.fire_posx = 0
                self.fire_posy = 0
                self.fire_speedy = 0
                self.hit_brick = brick
                break

        if self.brick_hit_count > 0:
            self.randomAngle('b')
            self.brick_hit_count = 0

    def draw(self):
        self.screen.fill(white)

        score_label = self.myfont.render(str(self.score), 100, pygame.color.THECOLORS['black'])
        self.screen.blit(score_label, (20, 10))

        cmd = ' '
        if self.action==1:
            cmd = '<'
        elif self.action==2:
            cmd = '>'
        elif self.action==3:
            cmd = 'o'

        s = '%s' %(cmd)
        count_label = self.myfont.render(s, 100, pygame.color.THECOLORS['brown'])
        self.screen.blit(count_label, (60, 10))

        if self.isAuto is True:
            auto_label = self.myfont.render("Auto", 100, pygame.color.THECOLORS['red'])
            self.screen.blit(auto_label, (self.win_width-200, 10))
            
        for brick in self.bricks:
            pygame.draw.rect(self.screen,grey,brick.rect,0)
        pygame.draw.circle(self.screen, orange, [int(self.ball_x), int(self.ball_y)], ball_radius, 0)
        pygame.draw.rect(self.screen, grey, [self.paddle_x, self.paddle_y, paddle_width, paddle_height], 0)

        if (self.fire_speedy<0):
            pygame.draw.rect(self.screen, red, [self.fire_posx, self.fire_posy, 5, 5], 0)

        pygame.display.update()

class HalfScreenSpec:
    def __init__(self, row, cols):
        self.half = cols//2
        self.state = 0
        self.transitions = [
            {"a": 1, "b": 2},
            {"a": 1, "b": 2},
            {"a": 1, "b": 2},
        ]
        self.acceptances = [
            {"a": True, "b": True},
            {"a": False, "b": True},
            {"a": True, "b": False},
        ]

    def reset(self):
        self.state = 0
        return self.encode_state(self.state)
    
    def __len__(self):
        return 3
        # return len(self.sequence)
    
    def encode_state(self, state):
        encoded_state = np.zeros((self.__len__()))
        encoded_state[state] = 1
        return encoded_state

    def step(self, label):
        
        reward = 0
        done = False
        # Check Label transitions the StateMachine

        if label:
            
            if self.acceptances[self.state][label]:
                reward = 10
                done = False
            # else:
            #     done = True

            self.state = self.transitions[self.state][label]
                
        encoded_state = self.encode_state(self.state)
        return encoded_state, reward, done
    
    def label(self, brick):
        if brick and brick.j == 0:
            if brick.i < self.half:
                return "a"
            return "b"
        
class BricksOrderSpec:
    def __init__(self, nrow, ncols):
        self.half = ncols//2
        self.state = 0

        self.transitions = []
        self.acceptances = []
        for c1 in range(ncols):
            state_transitions = {c2:c2 for c2 in range(ncols)} 
            state_transitions[None] = c1
            self.transitions.append(state_transitions)
   
            state_acceptances = {c2: (True if c2>=c1 else False) for c2 in range(ncols)} 
            state_acceptances[None] = False
            self.acceptances.append(state_acceptances)

    def reset(self):
        self.state = 0
        return self.encode_state(self.state)
    
    def __len__(self):
        return len(self.transitions)
    
    def encode_state(self, state):
        encoded_state = np.zeros((self.__len__()))
        encoded_state[state] = 1
        return encoded_state

    def step(self, label):
        
        reward = 0
        done = False
        # Check Label transitions the StateMachine

            
        if self.acceptances[self.state][label]:
            reward = 10

        self.state = self.transitions[self.state][label]
                
        encoded_state = self.encode_state(self.state)
        return encoded_state, reward, done
    
    def label(self, brick):
        if brick:
            return brick.i
        return None
    
class DummySpec:
    def __init__(self, nrow, ncols):
        self.state = 0
        self.nrow = nrow

    def reset(self):
        self.state = 0
        return self.encode_state(self.state)
    
    def __len__(self):
        return self.nrow
    
    def encode_state(self, state):
        encoded_state = np.zeros((self.__len__()))
        encoded_state[state] = 1
        return encoded_state

    def step(self, label):
        
        reward = 0
        done = False                
        encoded_state = self.encode_state(self.state)
        return encoded_state, reward, done
    
    def label(self, brick):
        return None
    

class LTLBreakout(Breakout):
    def __init__(self, brick_rows=3, brick_cols=3, fire_enabled=False, 
                 seed=42, render=False, name="breakout"):
        Breakout.__init__(self, brick_rows, brick_cols, seed, render, name)
        self.spec = BricksOrderSpec(self.brick_rows, self.brick_cols)
        self.observation_space = spaces.Box(
            low=0., high=1., shape=(self.observation_space.shape[0]+len(self.spec),))
                
    def reset(self):
        env_state, info = Breakout.reset(self)
        spec_state = self.spec.reset()
        next_prod_state = np.concatenate([env_state, spec_state], 0)
        return next_prod_state, None

    def step(self, action):

        # Fourroom State
        evn_state, env_reward, env_done, _, _  = Breakout.step(self, action)

        # Spec State

        labeled_actions = self.spec.label(self.hit_brick)
        spec_state, spec_reward, spec_done = self.spec.step(labeled_actions)

        # Prod State
        next_prod_state = np.concatenate([evn_state, spec_state], 0)

        return next_prod_state, spec_reward, env_done, False, None

    def reward(self, acceptances):

        self.rewards = [1 if acc else 0 for acc in acceptances]
        return self.rewards

class RewardAutoma(object):

    def __init__(self, brick_cols=0, type="left2right"): # brick_cols=0 -> RA disabled
        # RA states
        self.brick_cols = brick_cols
        if (self.brick_cols>0):
            self.nRAstates = brick_cols+2  # number of RA states
            self.RAGoal = self.nRAstates-2
            self.RAFail = self.nRAstates-1
        else: # RA disabled
            self.nRAstates = 2  # number of RA states
            self.RAGoal = 1 # never reached
            self.RAFail = 2 # never reached

        self.STATES = {
            'RAGoalStep':100,   # goal step of reward automa
            'RAGoal':1000,      # goal of reward automa
            'RAFail':0,         # fail of reward automa
            'GoodBrick':10,      # good brick removed for next RA state
            'WrongBrick':0      # wrong brick removed for next RA state
        }

        self.type = type
        self.goalreached = 0 # number of RA goals reached for statistics
        self.visits = {} # number of visits for each state
        self.success = {} # number of good transitions for each state
        self.reset()
        
    def init(self, game):
        self.game = game
        
    def reset(self):
        self.current_node = 0
        self.last_node = self.current_node
        self.countupdates = 0 # count state transitions (for the score)
        if (self.current_node in self.visits):
            self.visits[self.current_node] += 1
        else:
            self.visits[self.current_node] = 1


    # check if a column is free (used by RA)
    def check_free_cols(self, cols):
        cond = True
        for c in cols:
            for j in range(0,self.game.brick_rows):
                if (self.game.bricksgrid[c][j]==1):
                    cond = False
                    break
        return cond

    # RewardAutoma Transition
    def update(self):
        reward = 0
        state_changed = False

        # RA disabled
        if (self.brick_cols==0):
            return (reward, state_changed)
            
        for b in self.game.last_brikcsremoved:
            if b.i == self.current_node:
                reward += self.STATES['GoodBrick']
                #print 'Hit right brick for next RA state'
            else:
                reward += self.STATES['WrongBrick']
                #print 'Hit wrong brick for next RA state'

        f = np.zeros(self.brick_cols)
        for c in range(0,self.brick_cols):
            f[c] = self.check_free_cols([c])  # vector of free columns

        if (self.current_node<self.brick_cols): # 0 ... brick_cols
            if self.type == "left2right":
                goal_column = self.current_node
                cbegin = goal_column + 1
                cend = self.brick_cols
                cinc = 1
            elif self.type == "left2rightx2":
                goal_column = self.current_node
                cbegin = goal_column + 1
                cend = self.brick_cols
                cinc = 2
            elif self.type == "right2left":
                goal_column = self.brick_cols - self.current_node - 1
                cbegin = goal_column - 1
                cend = -1
                cinc = -1
            else:
                raise(f"Type {self.type} not implemented.")

            if f[goal_column]:
                state_changed = True
                self.countupdates += 1
                self.last_node = self.current_node
                self.current_node += 1
                reward += self.STATES['RAGoal'] * self.countupdates / self.brick_cols
                #print("  -- RA state transition to %d, " %(self.current_node))
                if (self.current_node==self.RAGoal):
                    # print("  <<< RA GOAL >>>")
                    reward += self.STATES['RAGoal']
                    self.goalreached += 1
            else:
                for c in range(cbegin, cend, cinc):
                    if f[c]:
                        self.last_node = self.current_node
                        self.current_node = self.RAFail  # FAIL
                        reward += self.STATES['RAFail']
                        #print("  *** RA FAIL *** ")
                        break

        elif (self.current_node==self.RAGoal): #  GOAL
            pass

        elif (self.current_node==self.RAFail): #  FAIL
            pass

        if (state_changed):
            if (self.current_node in self.visits):
                self.visits[self.current_node] += 1
            else:
                self.visits[self.current_node] = 1

            if (self.current_node != self.RAFail):
                #print "Success for last_node ",self.last_node
                if (self.last_node in self.success):
                    self.success[self.last_node] += 1
                else:
                    self.success[self.last_node] = 1


        return (reward, state_changed)

    def current_successrate(self):
        s = 0.0
        v = 1.0
        if (self.current_node in self.success):
            s = float(self.success[self.current_node])
        if (self.current_node in self.visits):
            v = float(self.visits[self.current_node])
        #print "   -- success rate: ",s," / ",v
        return s/v

    def print_successrate(self):
        r = []
        for i in range(len(self.success)):
            r.append(float(self.success[i])/self.visits[i])
        print('RA success: %s' %str(r))

class BreakoutNRA(Breakout):

    def __init__(self, brick_rows=3, brick_cols=3, fire_enabled=False, 
                 seed=42, render=False, name="breakout"):
        Breakout.__init__(self, brick_rows, brick_cols, fire_enabled, 
                 seed, render, name)
        
        self.RA_exploration_enabled = False  # Use options to speed-up learning process

        self.RA = RewardAutoma(brick_cols)
        self.RA.init(self)
        self.STATES = {
            'Init':0,
            'Alive':0,
            'Dead':-1,
            'PaddleNotMoving':0,
            'Scores':0,    # brick removed
            'Hit':0,       # paddle hit
            'Goal':0,      # level completed
        }

        self.observation_space = spaces.Box(
            low=0., high=1., shape=(5 + self.brick_rows*self.brick_cols + 1,))


    def savedata(self):
        return [self.iteration, self.hiscore, self.hireward, self.elapsedtime, self.RA.visits, self.RA.success, self.agent.SA_failure, random.getstate(),np.random.get_state()]

         
    def loaddata(self,data):
        self.iteration = data[0]
        self.hiscore = data[1]
        self.hireward = data[2]
        self.elapsedtime = data[3]
        self.RA.visits = data[4]
        self.RA.success = data[5]
        if (len(data)>6):
            self.agent.SA_failure = data[6]
        if (len(data)>7):
            print('Set random generator state from file.')
            random.setstate(data[7])
            np.random.set_state(data[8])   


    def setStateActionSpace(self):
        super(BreakoutNRA, self).setStateActionSpace()
        self.nstates *= self.RA.nRAstates
        print('Number of states: %d' %self.nstates)
        print('Number of actions: %d' %self.nactions)

    def getState(self):
        state = super(BreakoutNRA, self).getState()
        # return x + (self.n_ball_x*self.n_ball_y*self.n_ball_dir*self.n_paddle_x) * self.RA.current_node
        return state + [self.RA.current_node]

    def reset(self):
        super(BreakoutNRA, self).reset()
        self.RA.reset()
        self.RA_exploration()

        return self.getState(), None

    def step(self, a):
        super(BreakoutNRA, self).step(a)
        (RAr, state_changed) = self.RA.update()
        if (state_changed):
            self.RA_exploration()
        self.current_reward += RAr
        if (self.RA.current_node==self.RA.RAFail):
            self.finished = True

        self.current_state = self.getState()
        return self.current_state, int(self.current_reward), self.finished, False, None 
         
    def goal_reached(self):
        return self.RA.current_node==self.RA.RAGoal
       
    def getreward(self):
        r = self.current_reward
        #if (self.current_reward>0 and self.RA.current_node>0 and self.RA.current_node<=self.RA.RAGoal):
        #    r *= (self.RA.current_node+1)
            #print "MAXI REWARD ",r
        if (self.current_reward>0 and self.RA.current_node==self.RA.RAFail):  # FAIL RA state
            r = 0
        self.cumreward += self.gamman * r
        self.gamman *= self.agent.gamma

        #if (r<0):
        #    print("Neg reward: %.1f" %r)
        return r

    def RA_exploration(self):
        if not self.RA_exploration_enabled:
            return
        #print("RA state: ",self.RA.current_node)
        success_rate = max(min(self.RA.current_successrate(),0.9),0.1)
        #print("RA exploration policy: current state success rate ",success_rate)
        er = random.random()
        self.agent.option_enabled = (er<success_rate)
        #print("RA exploration policy: optimal ",self.agent.partialoptimal, "\n")