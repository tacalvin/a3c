import time
import random
import numpy as np


NUM_STATE = env_test.env.observation_space.shape[0]
NUM_ACTIONS = env_test.env.action_space.n
NONE_STATE = np.zeros(NUM_STATE)
GAMMA = .99
NUM_STEP_RETURN = 8

class Policy:
    """Policy NN"""
    #s, a, r, s', s' terminal mask
    train_queue = [[], [], [], [], []]

    def predict(self,state):
        return 5

policy = Policy()

#policy needs to be global to manage multiple agents
#frames needs to be global
class Agent:
    """Agent that acts based of a policy"""
    #If an episode ends in <NUM_STEPS_RETURN than the computation is incorrect
    def __init__(self,eps_start, eps_stop, eps_step):
        self.eps_start = eps_start
        self.eps_stop = eps_stop
        self.eps_step = eps_step

        self.reward = 0
        #using the nstep version of A3C
        self.memory = []

    def getEpsilon(self):
        if(frames >= self.eps_step):
            return self.eps_stop
        else:
            return self.eps_start + frames * (self.eps_end - self.eps_start) /self.eps_step

    #WHY: why is it when an agent acts is the answer chosen randomly from the distribution
    def act(self, state):
        epsilon = self.getEpsilon()
        
        if random.random() < epsilon:
            return random.randint(0,NUM_ACTIONS-1)
        else:
            p = policy.predict(state)
            return np.random.choice(NUM_ACTIONs,p=p)

    #Will grab the current state, action, reward and s_
    def getSample(self,numStep):
        s, a, _, _ = self.memory[0]
        _, _, _, s_ = self.memory[numStep-1]

        return s, a, self.R, s_


    def train(self, s, a, r, s_):
        a_cats = np.zeros(NUM_ACTIONS)
        a_cats[a] = 1

        self.memory.append((s, a_cats, r, s_))

        #Simplified relation of reward instead of scanning through memory and summing them up
        self.reward = (self.reward + r * (GAMMA ** NUM_STEP_RETURN))/GAMMA

        #Terminal state
        #Here the reward will be computed for each iteration with N_STEP being the length of the buffer essentially clearing out the buffer
        if s_ is None:
            while(len(self.memory)) > 0:
                n = len(self.memory)
                s, a, r, s_ = self.getSample(n)
                policy.train_push(s, a, r, s_)

                #What happens to GAMMA_N (GAMMA**NUM_STEP_RETURN)
                self.reward = (self.reward -self.memory[0][2]) / GAMMA
            self.reward = 0

        #computes when there is enough states in buffer
        if len(self.memory) >= NUM_STEP_RETURN:
            s, a, r, s_ = self.getSample(NUM_STEP_RETURN)
            policy.train_push(s, a, r, s_)

            #Why is the reward now just a difference
            self.reward = self.reward - self.memory[0][2]
            self.memory.pop(0)


def Optimizer:
    """Optimizer"""
    def __init__(self):
        print("Optimizer")

