import time,threading
import env
from env import create_env
import random
import numpy as np
import gym
import gym.spaces
from keras.models import *
from keras.layers import *
from keras import backend as K
import tensorflow as tf
#from model import *
ENV = "SpaceInvaders-v0"


RUN_TIME = 30
THREADS = 2
OPTIMIZERS = 1
THREAD_DELAY = 0.001

GAMMA = 0.99

N_STEP_RETURN = 8
GAMMA_N = GAMMA ** N_STEP_RETURN

EPS_START = 0.4
EPS_STOP  = .15
EPS_STEPS = 75000

MIN_BATCH = 32
LEARNING_RATE = 5e-3

   # entropy coefficient
RUN_TIME = 30
THREADS = 8
OPTIMIZERS = 2
THREAD_DELAY = 0.001







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

    def __init__(self, ob_space, ac_space):
        self.session = tf.Session()
        K.set_session(self.session)
        K.manual_variable_initialization(True)
        self.ob_space = ob_space
        self.ac_space = ac_space

        self.model = self.build_model()
        self.graph = self.build_comp_graph(self.model)

        self.session.run(tf.global_variables_initializer())
        self.default_graph = tf.get_default_graph()
        self.default_graph.finalize() #to avoid modifications



    #Why is the loss just the negative Objective Function J(PI)
    def build_model(self):
        #The model will have two outputs one for the action function and one for the value function
        input_l = Input(shape = (None,NUM_STATE,None,None))
        dense_l = Dense(16,activation='relu')(input_l)

        actions_out = Dense(NUM_ACTIONS, activation = 'softmax')(dense_l)
        value_out = Dense(1,activation = 'linear')(dense_l)


        #model =  Model(inputs=[input_l],outputs=[actions_out,value_out])

#----------------------------------------------------
        print(self.ob_space)
        #input_l = Input(batch_shape = (None,NUM_STATE))
        #conv1 = Conv2D(32,kernel_size=3,strides=2,padding = 'same',activation = 'elu')(input_l)
        #conv2 = Conv2D(32,kernel_size=3,strides=2,padding = 'same',activation = 'elu')(conv1)
        #conv3 = Conv2D(32,kernel_size=3,strides=2,padding = 'same',activation = 'elu')(conv2)
        #conv4 = Conv2D(32,kernel_size=3,strides=2,padding = 'same',activation = 'elu')(conv3)
        #flat_l = Flatten()(conv4)
        #linear_l = Dense(256, activation='linear')(conv4)

        #linear_l = K.expand_dims(linear_l, [0])
        
        #lstm_l = LSTM(256)

        #actions_out = Dense(NUM_ACTIONS, activation = 'softmax')(linear_l)
        #value_out = Dense(1, activation = 'linear')(linear_l)

        model = Model(inputs = [input_l], outputs = [actions_out, value_out])

        model._make_predict_function() #necessary to call model from multiple threads
        return model
    def build_comp_graph(self, model):
        #create placegolders

        #state batch placeholder
        s_t = tf.placeholder(tf.float32, shape=(None,NUM_STATE,None,3))
        #one hot encoded actions placeholders
        a_t = tf.placeholder(tf.float32, shape=(None,NUM_ACTIONS))

        #nstep reward
        r_t = tf.placeholder(tf.float32, shape=(None,1))

        #fyi using keras functional api
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
        #optimizes default graph using this optimizer
        #feeding it with a loss function and having a keras model set as default will allow the optimizer to train the global variables in the session
        minimize = optimizer.minimize(lreg)

        return s_t, a_t, r_t, minimize



    def train_push(self, s, a, r, s_):
        # print("PUSHING: {}".format(s.shape))
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
        #if len(self.train_queue[0]) < MIN_BATCH:
        #    time.sleep(0) #Yield here
        #    return
        #print("MADE IT")
        #with self.lock_queue:
        #    if len(self.train_queue[0]) < MIN_BATCH:
        #        return

        #    s, a, r, s_, s_mask = self.train_queue
        #    self.train_queue = [[], [], [], [], []]

        #s = np.vstack(s)
        #a = np.vstack(a)
        #r = np.vstack(r)
        #s_ = np.vstack(s_)
        #s_mask = np.vstack(s_mask);print("Optimizing")

        #v = self.predict_v(s_)
        #r = r + (GAMMA ** NUM_STEP_RETURN) * v *s_mask

        #s_t, a_t, r_t, minimize = self.graph
        #print("Presesh")
        #self.session.run(minimize, feed_dict = {s_t: s, a_t: a, r_t: r})

        if len(self.train_queue[0]) < MIN_BATCH:
            time.sleep(0)    # yield
            return

        with self.lock_queue:
            if len(self.train_queue[0]) < MIN_BATCH:    # more thread could have passed without lock
                return                                     # we can't yield inside lock

            s, a, r, s_, s_mask = self.train_queue
            self.train_queue = [ [], [], [], [], [] ]

        s = np.vstack(s)
        a = np.vstack(a)
        r = np.vstack(r)
        s_ = np.vstack(s_)
        s_mask = np.vstack(s_mask);print("OPTIMIZING")
        if len(s) > 5*MIN_BATCH: print("Optimizer alert! Minimizing batch of %d" % len(s))

        
        v = self.predict_v(s_)
        r = r + (GAMMA ** NUM_STEP_RETURN) * v * s_mask    # set v to 0 where s_ is terminal state
        
        s_t, a_t, r_t, minimize = self.graph
        self.session.run(minimize, feed_dict={s_t: s, a_t: a, r_t: r})


    def predict(self, s):
        with self.default_graph.as_default():
            return self.model.predict(s)

    def predict_v(self, s):
        with self.default_graph.as_default():
            p,v = self.model.predict(s)
            return v[-1]

    def predict_p(self,s):
        with self.default_graph.as_default():
            p,v = self.model.predict(s)
            #print("MP")
            #print(len(m))
            #Actions for space invaders is the last value of the prediction
            return p[-1]

#policy needs to be global to manage multiple agents
#frames needs to be global
frames = 0
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
            return self.eps_start + frames * (self.eps_stop - self.eps_start) /self.eps_step

    #WHY: why is it when an agent acts is the answer chosen randomly from the distribution
    def act(self, state):
        epsilon = self.getEpsilon()
        global frames; frames = frames + 1
        # print("ACT {}".format(state.shape))
        if random.random() < epsilon:
            return random.randint(0,NUM_ACTIONS-1)
        else:
            s = np.array([state])
            print(s.shape)
            p,v = policy.predict(s)
            p = p[-1]
            print("SHAPE OF P \n ", p.shape)
          #  quit()
            a = np.random.choice(NUM_ACTIONS,p=p)
            return a

    #Will grab the current state, action, reward and s_
    


    def train(self, s, a, r, s_):
        def getSample(memory,numStep):
            s, a, _, _ = memory[0]
            _, _, _, s_ = memory[numStep-1]

            return s, a, self.reward, s_
        a_cats = np.zeros(NUM_ACTIONS)
        a_cats[a] = 1

        # print("wut",s.shape)
        # quit()
        self.memory.append((s, a_cats, r, s_))

        #Simplified relation of reward instead of scanning through memory and summing them up
        self.reward = (self.reward + r * (GAMMA ** NUM_STEP_RETURN))/GAMMA

        #Terminal state
        #Here the reward will be computed for each iteration with N_STEP being the length of the buffer essentially clearing out the buffer
        if s_ is None:
            while(len(self.memory)) > 0:
                n = len(self.memory)
                s, a, r, s_ = getSample(self.memory, n)
                policy.train_push(s, a, r, s_)

                #What happens to GAMMA_N (GAMMA**NUM_STEP_RETURN)
                self.reward = (self.reward -self.memory[0][2]) / GAMMA
                self.memory.pop(0)
            self.reward = 0

        #computes when there is enough states in buffer
        if len(self.memory) >= NUM_STEP_RETURN:
            s, a, r, s_ = getSample(self.memory, NUM_STEP_RETURN)
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


class Environment(threading.Thread):
    stop_signal = False

    def __init__(self, render=False, eps_start=EPS_START, eps_end=EPS_STOP, eps_steps=EPS_STEPS, id_num = 0):
        threading.Thread.__init__(self)

        self.render = render
        self.env = create_env('doom',id_num,None)
        self.agent = Agent(eps_start, eps_end, eps_steps)


    def runEpisode(self):
        s = self.env.reset()

        R = 0
        while True:         
            time.sleep(THREAD_DELAY) # yield 

            if self.render: self.env.render()

            print("State size", s.shape)
            #quit()
            a = self.agent.act(s)
            s_, r, done, info = self.env.step(a)
            print("STEP STATE SIZE {}".format(s_.shape))
            quit()
            if done: # terminal state
                s_ = None
            self.agent.train(s, a, r, s_)

            s = s_
            R += r

            if done or self.stop_signal:
                break

        print("Total R:", R)

    def run(self):
        while not self.stop_signal:
            self.runEpisode()

    def stop(self):
        self.stop_signal = True



#main program here
           
env_test = Environment(render=True, eps_start = 0., eps_end = 0.)
NUM_STATE = env_test.env.observation_space.shape[0]
NUM_ACTIONS = env_test.env.action_space.n
NONE_STATE = np.zeros(NUM_STATE)

print(NUM_STATE)
print(NUM_ACTIONS)
policy = Policy(env_test.env.observation_space.shape,NUM_ACTIONS)


envs = [Environment(id_num = 1)]# [Environment(id_num=i+1) for i in range(THREADS)]
opts = [Optimizer()]#[Optimizer() for i in range(OPTIMIZERS)]


print("RUNNING")
print(len(opts))
for o in opts:
    o.start()

for e in envs:
    e.start()

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

