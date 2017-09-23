import time,threading
import random
import numpy as np
from keras.models import *
from keras.layers import *
from keras import backend as K
from env import Enviroment
from env import *
import tensorflow as tf

GAMMA = .99
NUM_STEP_RETURN = 8
L_VC = .5 #coefficient for loss of value function
L_E = .01 #coefficient for entropy

#GOAL Convert to entirely tensorflow
class Policy:
    """Policy NN"""
    #s, a, r, s', s' terminal mask
    train_queue = [[], [], [], [], []]
    lock_queue = threading.Lock()

    def __init__(self):
        self.session = tf.Session()
        K.set_session(self.session)
        K.manual_variable_initialization(True)

        self.model = self.build_model()
        self.graph = self.build_comp_graph(self.model)

        self.session.run(tf.global_variables_initializer())
        self.default_graph = tf.get_default_graph()
        self.default_graph.finalize() #to avoid modifications



    #Why is the loss just the negative Objective Function J(PI)
    def build_model(self):
        #The model will have two outputs one for the action function and one for the value function
        input_l = Input(batch_shape = (None,NUM_STATE))
        dense_l = Dense(16,activation='relu')(input_l)

        actions_out = Dense(NUM_ACTIONS, activation = 'softmax')(dense_l)
        value_out = Dense(1,activation = 'linear')(dense_l)


        model =  Model(inputs=[input_l],outputs=[actions_out,value_out])
        model._make_predict_function() #necessary to call model from multiple threads

        return model
    def build_comp_graph(self, model):
        #create placegolders

        #state batch placeholder
        s_t = tf.placeholder(tf.float32, shape=(None,NUM_STATE))
        #one hot encoded actions placeholders
        a_t = tf.placeholder(tf.float32, shape=(None,NUM_ACTIONS))

        #nstep reward
        r_t = tf.placeholder(tf.float32, shape=(None,1))

        p,v = model(s_t)

        #constant added to prevent NaN if probability was 0
        log_prob = tf.log(tf.reduce_sum(p * a_t, axis = 1, keep_dims=True ) + 1e-10)

        #advantage function = A(S,A) = Q(S,A) - V(S)
        advantage = r_t - v

        loss_policy = - log_prob * tf.stop_gradient(advantage)

        loss_value = L_VC * tf.square(advantage)

        entropy = L_E * tf.reduce_sum(p * tf.log(p + 1e-10), axis = 1, keep_dims=True)
        #lreg = 1/n sum H(pi(s))
        lreg = tf.reduce_mean(loss_policy + loss_value + entropy)

        optimizer = tf.train.RMSPropOptimizer(LEARNING_RATE,decay = .99)
        minimize = optimizer.minimize(lreg)

        return s_t, a_t, r_t, minimize



    def train_push(self, s, a, r, s_):
        with self.lock_queue:
            self.train_queue[0].append(s)
            self.train_queue[1].append(a)
            self.train_queue[2].append(r)

            if s_ is None:
                self.train_queue[3].append(NONE_STATE)
                self.train_queue[4].append(0.)
            else:
                self.train_queue[3].append(s_)
                self.train_queue[4].append(1.)
    def optimize(self):
        if len(self.train_queue[0]) < MIN_BATCH:
            time.sleep(0) #Yield here
            return
        with self.lock_queue:
            s, a, r, s_, s_mask = self.train_queue
            self.train_queue = [[], [], [], [], []]

        s = np.vstack(s)
        a = np.vstack(a)
        r = np.vstack(r)
        s_ = np.vstack(s_)
        s_mask = np.vstack(s_mask)

        v = self.v_predict(s_)
        r = r + (GAMMA ** NUM_STEP_RETURN) * v *s_mask

        s_t, a_t, r_t, minimize = self.graph
        self.session.run(minimize, feed_dict = {s_t: s, a_t: a, r_t: r})


    def predict(self, s):
        with self.default_graph.as_default():
            return self.model.predict(s)

    def predict_v(self, s):
        with self.default_graph.as_default():
            p,v = self.model.predict(s)
            return v

    def predict_p(self,s):
        with self.default_graph.as_default():
            p,v = self.model.predict(s)
            return p

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


class Optimizer(threading.Thread):
    """Optimizer"""

    stop_signal = False

    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):
        while not self.stop_signal:
            policy.optimize()
    def stop(self):
        self.stop_signal = True

#main program here

env_test = Enviroment(Agent, eps_start = 0., eps_end = 0.)
NUM_STATE = env_test.env.observation_space.shape[0]
NUM_ACTIONS = env_test.env.action_space.n
NONE_STATE = np.zeros(NUM_STATE)



policy = Policy()


envs = [Enviroment(Agent) for i in range(THREADS)]
opts = [Optimizer() for i in range(OPTIMIZERS)]


for o in opts:
    o.run()

for e in envs:
    e.run()

time.sleep(RUN_TIME)

for e in envs:
    e.stop()

for e in envs:
    e.join()

for o in opts:
    o.stop()

for o in opts:
    o.join()

print("Training Complete")
env_test.run()

