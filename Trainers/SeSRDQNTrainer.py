# This model is an updated model according to the TKDE reviews, written in 13/09/2023
from re import I
import sys
import random
from tabnanny import verbose
import time
from xmlrpc.client import boolean
sys.path.append("../../")
sys.path.append("../")
from Environment.JobMatcherLinux import JobMatcher
from Utils.JobReader import n_skill, skill_lst, sample_info, read_offline_samples, read_skill_graph, itemset_process, get_frequent_itemset
from Utils.Utils import sparse_to_dense, read_pkl_data, dense_to_sparse, dense_to_sparse_prob
from Utils.Functions import evaluate, evaluate_case_study

from Environment.DifficultyEstimatorGLinux import *
from Models.SeSRDQN import SeSRDQN 
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from Sampler import BestStrategyPoolSampler, EpsilonGreedyPoolSampler
from CONFIG import HOME_PATH
from tqdm import tqdm
from Environment.Environment import Environment
from Trainers.Memory import Memory2
from random import randint
from math import log
import argparse
import wandb
import ast

def write_data(names):
    with open(r'training_data.txt', 'a') as fp:
        for item in names:
            # write each item on a new line
            fp.write("%s\n" % dense_to_sparse(item))

class OnPolicyTrainer(object):

    #TODO: add a function to get the best action using kNN
    def get_kNN_act_pool(self, state): 
        return
      
    def get_act_pool(self, state):
        cnt = [0] * n_skill
        for i in range(n_skill):
            if state[i] == 1:
                for u in self.relational_lst[i]:
                    cnt[u] += 1
        skill_id = list(range(n_skill))
        skill_id.sort(key=lambda x: cnt[x] + 1e-5 * x, reverse=True)
        ret = []
        for u in range(n_skill):
            s = skill_id[u]
            if state[s] == 0:
                ret.append(s)
            if len(ret) == self.pool_size:
                break
        return ret

    def __init__(self, Qnet, Qtarget, environment, beta, train_samples, relational_lst, memory, pool_size, sess):
        self.environment = environment
        self.relational_lst = relational_lst
        self.pool_size = pool_size

        self.sampler = EpsilonGreedyPoolSampler(relation_lst, Qtarget, 0.7, n_skill, pool_size=pool_size)
        self.best_sampler = BestStrategyPoolSampler(relational_lst, Qtarget, n_skill, pool_size)

        self.Qnet = Qnet
        self.Qtarget = Qtarget

        self.beta = beta
        self.train_samples = train_samples
        self.avg_loss = 0
        self.step_count = 0

        self.memory = memory
        self.sess = sess
        self.cp_ops = [w.assign(self.Qnet.weights[name.replace("target", 'qnet')]) for name, w in self.Qtarget.weights.items()]
        self.target_update()

    def target_update(self):
        self.sess.run(self.cp_ops)

    def train(self, n_batch, batch_size, data_train, data_valid, verbose_batch=500, T=26, target_update_batch=20, save_path=None):
        n_data = len(data_train)
        for it in tqdm(range(n_batch)):

            if it != 0 and it % target_update_batch == 0:
                self.target_update()

            if it != 0 and it % verbose_batch == 0:
                evaluate(self.best_sampler, self.environment, data_valid, self.train_samples, it / verbose_batch, T=T, verbose=False)
                self.sampler.epsilon += (1 - self.sampler.epsilon) * 0.1
                if save_path is None:
                    self.Qnet.save(HOME_PATH + "data/model/%s_SeSRDQN_sal_%s_dif_%s_info_%s_freq_%s_div_%s_enc_%s_%s" %(direct_name, salary_prototype, difficulty_prototype, lambda_info, lambda_freq, lambda_div, lambda_enc, env_params))
                else:
                    self.Qnet.save(save_path)

            self.environment.clear()

            # sample a resume
            k = randint(0, n_data - 1)
            x, y = data_train[k]
            prefix = self.train_samples[x][0]

            salary_pre = self.environment.add_prefix(prefix)

            # Progressively sample the frequent set from the bootstrapped frequent set
            result = self.environment.get_frequent_set_step(prefix, flag=True)
            result = [sparse_to_dense(list(element), n_skill) for element in result] 
                   
            result_v = []
            if len(result) > batch_size * 3:
                result_v = random.sample(result, batch_size * 3)
            else:
                result_v = result

            for t in range(T):
                state_pre = self.environment.state_list.copy()
                s, _ = self.sampler.sample(self.environment.state)
                easy, salary, r = self.environment.add_skill(s, evaluate=True)

                self.memory.store((state_pre, s, (easy, salary - salary_pre), result_v))
                salary_pre = salary

            if self.memory.get_size() > batch_size * 10:
                data_batch = self.memory.sample(batch_size)
                data_batch = self.transform_train_batch(data_batch, batch_size)
                _, _, loss, info_loss, div_loss, freq_loss, enc_loss = self.Qnet.run(data_batch, batch_size, train=True)
                train_loss = np.average(loss).item()
                train_info_loss = np.average(info_loss).item()
                train_div_loss = np.average(div_loss).item()
                train_freq_loss = np.average(freq_loss).item()
                train_enc_loss = np.average(enc_loss).item()
                
                wandb.log({"train_loss": train_loss}, step=it)
                wandb.log({"train_info_loss": train_info_loss}, step=it)
                wandb.log({"train_div_loss": train_div_loss}, step=it)
                wandb.log({"train_freq_loss": train_freq_loss}, step=it)
                wandb.log({"train_enc_loss": train_enc_loss}, step=it)

        wandb.run.summary["train_loss"] = train_loss
        wandb.run.summary["train_info_loss"] = train_info_loss
        wandb.run.summary["train_div_loss"] = train_div_loss
        wandb.run.summary["train_freq_loss"] = train_freq_loss
        wandb.run.summary["train_enc_loss"] = train_enc_loss
        wandb.finish()

        return

    def transform_train_batch(self, data_batch, batch_size):
        data_state = [sparse_to_dense(u[0], n_skill) for u in data_batch]
        data_skill = [u[1] for u in data_batch]
        data_easy = [u[2][0] for u in data_batch]
        data_salary = [u[2][1] for u in data_batch]

        data_frequency = []
        for u in data_batch:
            data_frequency += u[3]
        if len(data_frequency) >= batch_size * 3:
            data_frequency = random.sample(data_frequency, batch_size * 3)
        else:
            print("This batch step is not full")

        # add some randomized elements from the frequent set
        training_freq = random.sample(frequency_set, batch_size)
        for item in training_freq:
            data_frequency.append(sparse_to_dense(item, n_skill))
        # print(len(data_frequency), dense_to_sparse(data_frequency[0]))

        pool_nxt = []
        for state, skill in zip(data_state, data_skill):
            state[skill] = 1
            pool_nxt.append(self.get_act_pool(state))

        data_salary_easy, _ = self.Qtarget.estimate_maxq_batch(data_state, pool_nxt)  #
        data_salary_q, data_easy_q = data_salary_easy

        for state, skill in zip(data_state, data_skill):
            state[skill] = 0

        data_easy = [r + (1 - self.beta) * q for r, q in zip(data_easy, data_easy_q)]
        data_salary = [r + (1 - self.beta) * q for r, q in zip(data_salary, data_salary_q)]

        return data_state, [[skill] * self.pool_size for skill in data_skill], data_salary, data_easy, data_frequency
    
    def test(self):
        return 

    def replace_prototype_embeddings(self):
        # salary_embeddings = self.get_mcts_embeddings('salary')
        salary_prototype_embeddings = tf.Variable(self.get_mcts_embeddings('salary'), name='qnet_salary_prototype', trainable=False)
        difficulty_prototype_embeddings = tf.Variable(self.get_mcts_embeddings('difficulty'), name='qnet_difficulty_prototype', trainable=False)
        salary_embeddings = tf.Variable(self.Qnet.weights['qnet_salary_embedding'], name='qnet_salary_embedding', trainable=False)
        difficulty_embeddings = tf.Variable(self.Qnet.weights['qnet_difficulty_embedding'], name='qnet_difficulty_embedding', trainable=False)      

        self.Qnet.weights['qnet_salary_prototype'] = salary_prototype_embeddings
        self.Qnet.weights['qnet_difficulty_prototype'] = difficulty_prototype_embeddings
        self.Qnet.weights['qnet_salary_embedding'] = salary_embeddings
        self.Qnet.weights['qnet_difficulty_embedding'] = difficulty_embeddings

        # print(self.Qnet.weights['qnet_difficulty_embedding'].trainable)

        # self.Qnet.lambda_freq = 1.0
        self.Qnet.lambda_info = 0.0
        self.Qnet.lambda_enc = 0.0

        self.Qnet.save(HOME_PATH + "data/model/%s_SeSRDQN_sal_%s_dif_%s_info_%s_freq_%s_div_%s_enc_%s_%s" %(direct_name, salary_prototype, difficulty_prototype, lambda_info, lambda_freq, lambda_div, lambda_enc, env_params))

    def get_mcts_embeddings(self, part):
        mcts_file = str("info_" + str(lambda_info) + "_freq_" + str(lambda_freq) + "_skill_sets" + '(' + part + ')')
        flag = True if part == 'salary' else False
        replaced_embeddings = []

        # read file line by line
        with open("output/mcts/" + mcts_file, 'r') as f:
            for line in f:
                vector_str, _ = line.split(' 0.0', 1)
                vectors = ast.literal_eval(vector_str) 
                # print(vectors, type(vectors))
                replaced_prototype = []
                for item in vectors:
                    replaced_prototype.append(skill_lst.index(item))
                # print(replaced_prototype)
                embed = self.Qnet.get_raw_embeddings(replaced_prototype, flag)
                # print(type(embed), embed.shape)
                replaced_embeddings.append(embed)
        results = tf.convert_to_tensor(np.array(replaced_embeddings), dtype=tf.float32)
        # print(results.shape, type(results))
        return results  

# Parameter Setting
sample_lst, skill_cnt, jd_len = sample_info()
skill_p = [log(u * 1.0 / len(sample_lst)) for u in skill_cnt]
relation_lst = read_skill_graph()
frequency_set = get_frequent_itemset(2, 9)
freq_len = len(frequency_set)


parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--salary_prototype', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--difficulty_prototype', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.01)

parser.add_argument('--lambda_info', type=float, default=0.0)
parser.add_argument('--lambda_freq', type=float, default=0.01)
parser.add_argument('--lambda_div', type=float, default=0.1)
parser.add_argument('--lambda_enc', type=float, default=0.1)

parser.add_argument('--lambda_d', type=float, default=0.1)
parser.add_argument('--emb_size', type=int, default=256)
parser.add_argument('--beta', type=float, default=0.2)
parser.add_argument('--pool_size', type=int, default=100)
parser.add_argument('--pooling', type=str, default="sum")
args = parser.parse_args()


if __name__ == "__main__":
    wandb.init(
       project = "SkillRec",
       tags=["Prototybe-based SkillRec", str(args.salary_prototype), str(args.difficulty_prototype), str(args.lambda_info), str(args.lambda_freq), str(args.lambda_div), str(args.lambda_enc), str(args.lambda_d), str(args.beta), str(args.pool_size)],
       entity="yangji721",
       config = wandb.config,
       # config = config,
       mode="disabled",
    )

    direct_name = "resume"
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    batch_size = int(args.batch_size)
    salary_prototype, difficulty_prototype = int(args.salary_prototype), int(args.difficulty_prototype)
    lambda_info, lambda_freq, lambda_div, lambda_enc = float(args.lambda_info), float(args.lambda_freq), float(args.lambda_div), float(args.lambda_enc)

    lr = float(args.lr)
        
    lambda_d, beta, pool_size = float(args.lambda_d), float(args.beta), int(args.pool_size)
    emb_size = int(args.emb_size)
    pooling = str(args.pooling)

    env_params = {"lambda_d": lambda_d, "beta": beta, 'pool_size': pool_size}
    
    # Difficulty & Reward
    itemset = itemset_process(skill_cnt)

    d_estimator = DifficultyEstimator(item_sets=[u[0] for u in itemset], item_freq=[u[1] for u in itemset], n_samples=len(sample_lst))
    job_matcher = JobMatcher(n_top=100, skill_list=[u[0] for u in sample_lst], salary=[u[1] for u in sample_lst], w=5, th=10.0 / 9)

    environment = Environment(lambda_d=env_params['lambda_d'], d_estimator=d_estimator,
                              job_matcher=job_matcher, n_skill=n_skill)

    train_samples = read_offline_samples(direct_name)  # Start at 1，skill_lst[1:i + 1], skill_lst[i+1], r_lst[i]

    memory = Memory2(20000)

    # Training Dataset
    data_train = read_pkl_data(HOME_PATH + "data/%s/train_test/traindata.pkl" % direct_name)
    N_train = len(data_train)

    # Testing Dataset
    data_valid = read_pkl_data(HOME_PATH + "data/%s/train_test/validdata.pkl" % direct_name)

    # ----------------- Init Model ---------------------
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)


    Qa = SeSRDQN(n_skill, lambda_info = lambda_info, lambda_div= lambda_div, lambda_enc = lambda_enc, lambda_freq = lambda_freq, verbose=1, learning_rate=lr, lambda_d=env_params['lambda_d'], embsize=emb_size, 
               pool_size=env_params['pool_size'], salary_prototype = salary_prototype, difficulty_prototype = difficulty_prototype,
               salary_encoder_layers=[emb_size, emb_size], salary_decoder_layers=[emb_size, emb_size], dropout_salary=[1.0, 1.0, 1.0], difficulty_decoder_layers=[emb_size, emb_size],
               difficulty_encoder_layers=[emb_size, emb_size], dropout_difficulty=[1.0, 1.0, 1.0], activation=tf.nn.leaky_relu, random_seed=2023, l2_reg=0.001, pool=pooling, name="qnet", sess=sess)

    Qa_ = SeSRDQN(n_skill, lambda_info = lambda_info, lambda_div= lambda_div, lambda_enc = lambda_enc, lambda_freq = lambda_freq, verbose=1, learning_rate=lr, lambda_d=env_params['lambda_d'], embsize=emb_size, pool_size=env_params['pool_size'], salary_prototype = salary_prototype, difficulty_prototype = difficulty_prototype, salary_encoder_layers=[emb_size, emb_size], salary_decoder_layers=[emb_size, emb_size], dropout_salary=[1.0, 1.0, 1.0], difficulty_encoder_layers=[emb_size, emb_size], difficulty_decoder_layers=[emb_size, emb_size], dropout_difficulty=[1.0, 1.0, 1.0], activation=tf.nn.leaky_relu, random_seed=2023, l2_reg=0.001, pool=pooling, name="target", sess=sess)

    sess.run(tf.global_variables_initializer())

    # ----------------- Model Reading ---------------------
    # Qa.load(HOME_PATH + "data/model/%s_SeSRDQN_sal_%s_dif_%s_info_%s_freq_%s_div_%s_enc_%s_%s" %(direct_name, salary_prototype, difficulty_prototype, lambda_info, lambda_freq, lambda_div, lambda_enc, env_params))

    # ----------------- Model Training ---------------------
    on_trainer = OnPolicyTrainer(Qa, Qa_, environment=environment, train_samples=train_samples, beta=env_params['beta'],
                                 memory=memory, sess=sess, relational_lst=relation_lst, pool_size=env_params['pool_size'])
    on_trainer.train(n_batch=320000, batch_size=batch_size, data_train=data_train, data_valid=data_valid, verbose_batch=2048, T=20, target_update_batch=64)

    # ----------------- Model Retrain ---------------------
    # on_trainer.replace_prototype_embeddings()
    # print(Qa.lambda_freq)
    # quit()
    # on_trainer.train(n_batch=80000, batch_size=batch_size, data_train=data_train, data_valid=data_valid, verbose_batch=2048, T=20, target_update_batch=64)
    
    sampler = BestStrategyPoolSampler(relation_lst, Qa, n_skill, pool_size=env_params['pool_size'])
    data_test = read_pkl_data(HOME_PATH + "data/%s/train_test/testdata.pkl" % direct_name)
    # general evaluate
    evaluate(sampler=sampler, environment=environment, data_test=data_valid, train_samples=train_samples, epoch=-1, T=20, verbose=False)
    # case study
    # evaluate_case_study(sampler=sampler, environment=environment, T=20, num1=salary_prototype, num2=difficulty_prototype)

    # ----------------- MOdel Save -----------------------
    Qa.save(HOME_PATH + "data/model/%s_SeSRDQN_sal_%s_dif_%s_info_%s_freq_%s_div_%s_enc_%s_%s" %(direct_name, salary_prototype, difficulty_prototype, lambda_info, lambda_freq, lambda_div, lambda_enc, env_params))