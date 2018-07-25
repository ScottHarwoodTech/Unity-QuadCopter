#based on https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/blob/master/contents/7_Policy_gradient_softmax/RL_brain.py
import numpy as np
import control
import subprocess
import tensorflow as tf
import os
import time
from tqdm import tqdm
import math
#inputs = velocity vector (3x1) + rotaion vector(3x1)
import sys
def evaluate_score(vel,rot, lastVelM,lastRotM,step):
    velM = (vel[0] * vel[0]) + (vel[1] * vel[1]) + (vel[2] * vel[2])#find mag of velocity
    velM = math.sqrt(velM)
    rotM = (rot[0] * rot[0]) + (rot[1] * rot[1]) + (rot[2] * rot[2])#find mag of rotation
    rotM = math.sqrt(rotM)
    if rot[0] > 180 or rot[2] > 180 or rot[0] < -180 or rot[2] < -180:#check it hasnt flipped
        return velM,rotM,-1.0
    if velM < lastVelM or rotM < lastRotM:
        return velM,rotM,1.0 * step
    else:
        return velM,rotM,-1.0

class RLnetwork():

    def __init__(self,n_features,n_actions,lr=0.01, reward_decay = 0.95,output_graph = False):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = lr
        self.gamma = reward_decay
        self.ep_obs, self.ep_as, self.ep_rs = [],[],[]

        self._build_net()
        self.sess = tf.Session()
        self.step = 0
        if output_graph:
            self.merged = tf.summary.merge_all()
            self.grapher = tf.summary.FileWriter("logs/" + str(VERSION) + "/",self.sess.graph)
            self.writer = tf.summary.FileWriter("logs/" +str(VERSION) + "/")
        self.sess.run(tf.global_variables_initializer())
    def _build_net(self):
        with tf.name_scope("inputs"):
            self.tf_obs = tf.placeholder(tf.float32,[None,self.n_features],name = "observations")#here i will feed in the current state
            self.tf_acts = tf.placeholder(tf.int32,[None, ], name = "actions_num")
            self.tf_vt = tf.placeholder(tf.float32,[None, ], name = "actions_value")#Question wtf does this do
        #the first fully connected layer
        layer1 = tf.layers.dense(
                inputs = self.tf_obs,
                units = 256,#256 hidden neurons on this layer
                activation = tf.nn.relu, #using the relu activation
                kernel_initializer = tf.random_normal_initializer(mean = 0,stddev=0.3),
                bias_initializer = tf.constant_initializer(0.1),
                name="fc1"
                )
        # 2nd hidden layer
        layer2 = tf.layers.dense(
                inputs = layer1,
                units = 256, # 256 hidden neurons on this layer
                activation = tf.nn.relu, #using the relu activation
                kernel_initializer = tf.random_normal_initializer(mean = 0,stddev=0.3),
                bias_initializer = tf.constant_initializer(0.1),
                name="fc2"
                )
        #output layer
        all_act = tf.layers.dense(
                inputs= layer2,
                units = self.n_actions,#so it knows how many possible outputs it can have
                activation=tf.nn.softmax,
                kernel_initializer = tf.random_normal_initializer(mean=0,stddev=0.3),
                bias_initializer = tf.constant_initializer(0.1),
                name="output")

        self.all_act_prob = tf.nn.softmax(all_act,name="act_prob")#convert to probability distribution

        with tf.name_scope("loss"):#define the loss function
            neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=all_act,labels=self.tf_acts)#Question
            self.loss = tf.reduce_mean(neg_log_prob * self.tf_vt)
            tf.summary.scalar("loss",self.loss)
        with tf.name_scope("train"):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def choose_action(self,observation):
        observation = np.array(observation)
        prob_weights = self.sess.run(self.all_act_prob,feed_dict = {self.tf_obs:observation[np.newaxis, :]})
        #feed sample through network to calculate probabilities
        action = np.random.choice(range(prob_weights.shape[1]),p=prob_weights.ravel())
        #select the action using prob dist. 
        return action #Question, is this going to be a number? 

    def store_transition(self,s,a,r):
        self.ep_obs.append(s)#append the state
        self.ep_as.append(a)#append the action
        self.ep_rs.append(r)#append the reward
    
    def learn(self):#train the model
        discounted_ep_rs_norm = self._discount_and_norm_rewards()

        _ ,loss = self.sess.run([self.train_op,self.loss], feed_dict = {#using the adam optimizer train feed this tample with these targets
            self.tf_obs:np.vstack(self.ep_obs),#feed each state
            self.tf_acts: np.array(self.ep_as),#feed each action
            self.tf_vt: discounted_ep_rs_norm
            })
        self.step += 1
        summary  = tf.Summary()
        summary.value.add(tag="cookies",simple_value = sum(self.ep_rs))
        self.writer.add_summary(summary,self.step)
        if self.step % 1 ==0:
            ret = self.sess.run(self.merged, feed_dict = {#using the adam optimizer train feed this tample with these targets
            self.tf_obs:np.vstack(self.ep_obs),#feed each state
            self.tf_acts: np.array(self.ep_as),#feed each action
            self.tf_vt: discounted_ep_rs_norm
            })
            #s = tf.Summary()
           # s.value.add(tag = "Grad" + str(VERSION),simple_value = discounted_ep_rs_norm)
          #  self.writer.add_summary(s,self.step)
            self.writer.add_summary(ret,self.step)
        self.ep_obs,self.ep_as,self.ep_rs = [],[],[] #clears it out for the next episode
        return discounted_ep_rs_norm

    def _discount_and_norm_rewards(self):
        discounted_ep_rs = np.zeros_like(self.ep_rs)#Question tf is np.zeros_like
        running_add = 0
        for t in reversed(range(0,len(self.ep_rs))):#iterates over every reward backwards
            running_add = running_add * self.gamma * self.ep_rs[t]
            discounted_ep_rs[t] = running_add #save the current running add
        discounted_ep_rs -= np.mean(discounted_ep_rs)#Question how does this normalize?
        std = np.std(discounted_ep_rs)
        if std != 0:
            discounted_ep_rs /= std
        return discounted_ep_rs

VERSION = len(os.listdir("./logs"))
os.mkdir(os.path.join("./logs",str(VERSION)))
network = RLnetwork(6,16,0.01,0.95,True)#make network with 6 inputs (x,y,z of vel and rotation), 16 target outputs (basically change the different motors)
#os.system('"Unity QuadCopter.exe"')

p = subprocess.Popen(os.path.join(os.getcwd(),"builds.exe"))
time.sleep(5)
env = control.Drone_Control(25000,0.1)
NUM_EPISODES = int(sys.argv[1])
for x in tqdm(range(NUM_EPISODES)):
    observation = env.pollRotation() + env.pollVelocity()
    lastRotM = np.inf
    lastVelM = np.inf
    #this will be a full episode
    env.reset()#clean slate
    episodeStep = 0
    while not env.crashed:#picks random number and it applies it
        action = network.choose_action(observation)
        env.eval_action(action)
        vel = env.pollVelocity()
        rot = env.pollVelocity()
        next_ob = rot + vel
        velM,rotM,reward = evaluate_score(vel,rot,lastVelM,lastRotM,episodeStep)
        lastRotM = rotM
        lastVelM = velM
        network.store_transition(observation,action,reward)
        observation = next_ob
        if rot[0] > 180 or rot[2] > 180 or rot[0] < -180 or rot[2] < -180:
            env.crashed = True
        episodeStep += 1
     #when the episdoe completes your last action didnt work
    vt = network.learn()#when you crash learn from your mistakes
    env.crashed = False
env.s.close()
env.listena = False
env.listener.close()
p.kill()

