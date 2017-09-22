import time,threading

import gym
from universe import envs
from a3c import Agent

#-- constants
ENV = 'CartPole-v0'

RUN_TIME = 30
THREADS = 8
OPTIMIZERS = 2
THREAD_DELAY = 0.001

GAMMA = 0.99

N_STEP_RETURN = 8
GAMMA_N = GAMMA ** N_STEP_RETURN

EPS_START = 0.4
EPS_STOP  = .15
EPS_STEPS = 75000

MIN_BATCH = 32
LEARNING_RATE = 5e-3

LOSS_V = .5			# v loss coefficient
LOSS_ENTROPY = .01 # entropy coefficient

class Enviroment(threading.Thread):
    """"Enviroment for running A3C agents"""

    stop_signal = False

    def __init__(self, render=True, eps_start=EPS_START, eps_end=EPS_STOP, eps_steps=EPS_STEPS):
        threading.Thread.__init__(self)

        self.redner = render
        self.env = gym.make(ENV)
        self.agent = Agent(eps_start,eps_end,eps_steps)

    def run(self):
        while not self.stop_signal:
            self.runEpisode()

    def runEpisode(self):
         s = self.env.reset()
         while True:
             time.sleep(THREAD_DELAY)

             a = self.agent.act(s)
             s_, r, done, info = self.env.step(a)

             if done:
                 s_ = None
             self.agent.train(s,a,r,s_)
             s = s_

             if done or self.stop_signal:
                 break
    def stop(self):
        self.stop_signal = True


