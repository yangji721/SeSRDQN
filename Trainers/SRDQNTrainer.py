import sys
sys.path.append("../../")
sys.path.append("../")
from Environment.JobMatcherLinux import JobMatcher
from Utils.JobReader import n_skill, sample_info, read_offline_samples, read_skill_graph, itemset_process, get_frequent_itemset
from Environment.DifficultyEstimatorGLinux import DifficultyEstimator
from Models.SRDQN import SRDQN
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import tensorflow as tf
from Utils.Utils import sparse_to_dense, read_pkl_data
from Utils.Functions import evaluate, evaluate_case_study, evaluate_with_example
from Sampler import BestStrategyPoolSampler, EpsilonGreedyPoolSampler
from CONFIG import HOME_PATH
from tqdm import tqdm
from Environment.Environment import Environment
from Trainers.Memory import Memory2
from random import randint
from math import log
import numpy as np
import time


class OnPolicyTrainer(object):

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
                    self.Qnet.save(HOME_PATH + "data/model/%s_SRDQN_%s" % (direct_name, env_params))
                else:
                    self.Qnet.save(save_path)

            self.environment.clear()

            # Sample a resume
            k = randint(0, n_data - 1)
            x, y = data_train[k]
            prefix = self.train_samples[x][0]

            salary_pre = self.environment.add_prefix(prefix)
            for t in range(T):
                state_pre = self.environment.state_list.copy()
                s, _ = self.sampler.sample(self.environment.state)
                easy, salary, r = self.environment.add_skill(s, evaluate=True)
                self.memory.store((state_pre, s, (easy, salary - salary_pre)))
                salary_pre = salary

            if self.memory.get_size() > batch_size * 10:
                data_batch = self.memory.sample(batch_size)
                data_batch = self.transform_train_batch(data_batch)
                self.Qnet.run(data_batch, batch_size, train=True)
        return

    def transform_train_batch(self, data_batch):
        data_state = [sparse_to_dense(u[0], n_skill) for u in data_batch]
        data_skill = [u[1] for u in data_batch]
        data_easy = [u[2][0] for u in data_batch]
        data_salary = [u[2][1] for u in data_batch]

        pool_nxt = []
        begin = time.time()
        for state, skill in zip(data_state, data_skill):
            state[skill] = 1
            pool_nxt.append(self.get_act_pool(state))
        end = time.time()
        # print("get_act_pool time: ", end - begin)


        data_salary_easy, _ = self.Qtarget.estimate_maxq_batch(data_state, pool_nxt)  #
        data_salary_q, data_easy_q = data_salary_easy

        for state, skill in zip(data_state, data_skill):
            state[skill] = 0

        data_easy = [r + (1 - self.beta) * q for r, q in zip(data_easy, data_easy_q)]
        data_salary = [r + (1 - self.beta) * q for r, q in zip(data_salary, data_salary_q)]

        return data_state, [[skill] * self.pool_size for skill in data_skill], data_salary, data_easy


# Parameter Settings
sample_lst, skill_cnt, jd_len = sample_info()
skill_p = [log(u * 1.0 / len(sample_lst)) for u in skill_cnt]
relation_lst = read_skill_graph()

if __name__ == "__main__":
    direct_name = "resume"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(0)
    lambda_d, beta, pool_size = float(0.1), float(0.2), int(50)
    env_params = {"lambda_d": lambda_d, "beta": beta, 'pool_size': pool_size}
    # env_params = {"lambda_d": 0.1, "beta": 0.2, 'pool_size': 100}
    
    # Difficulty & Reward
    itemset = itemset_process(skill_cnt)
    d_estimator = DifficultyEstimator(item_sets=[u[0] for u in itemset], item_freq=[u[1] for u in itemset], n_samples=len(sample_lst))
    job_matcher = JobMatcher(n_top=100, skill_list=[u[0] for u in sample_lst], salary=[u[1] for u in sample_lst], w=5, th=10.0 / 9)

    environment = Environment(lambda_d=env_params['lambda_d'], d_estimator=d_estimator,
                              job_matcher=job_matcher, n_skill=n_skill)

    train_samples = read_offline_samples(direct_name)  # start at 1，skill_lst[1:i + 1], skill_lst[i+1], r_lst[i]

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

    Qa = SRDQN(n_skill, verbose=1, lr=0.001, lambda_d=env_params['lambda_d'], embsize=256, pool_size=env_params['pool_size'],
               salary_deep_layers=[256, 256, 256, 256, 256], dropout_salary_deep=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
               easy_deep_layers=[256, 256, 256, 256, 256], dropout_easy_deep=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
               activation=tf.nn.leaky_relu, random_seed=2019, l2_reg=0.001, name="qnet", sess=sess)

    Qa_ = SRDQN(n_skill, verbose=1, lr=0.001, lambda_d=env_params['lambda_d'], embsize=256, pool_size=env_params['pool_size'],
                salary_deep_layers=[256, 256, 256, 256, 256], dropout_salary_deep=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                easy_deep_layers=[256, 256, 256, 256, 256], dropout_easy_deep=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                activation=tf.nn.leaky_relu, random_seed=2019, l2_reg=0.001, name="target", sess=sess)

    sess.run(tf.global_variables_initializer())

    # ----------------- Model Reading ---------------------
    Qa.load(HOME_PATH + "data/model/%s_SRDQN_%s" % (direct_name, env_params))

    # ----------------- Model Training ---------------------
    on_trainer = OnPolicyTrainer(Qa, Qa_, environment=environment, train_samples=train_samples, beta=env_params['beta'],
                                 memory=memory, sess=sess, relational_lst=relation_lst, pool_size=env_params['pool_size'])
    #on_trainer.train(n_batch=320000, batch_size=64, data_train=data_train, data_valid=data_valid, verbose_batch=4096,
    #                 T=20, target_update_batch=64)
    sampler = BestStrategyPoolSampler(relation_lst, Qa_, n_skill, pool_size=env_params['pool_size'])

    data_test = read_pkl_data(HOME_PATH + "data/%s/train_test/testdata.pkl" % direct_name)
    # evaluate(sampler=sampler, environment=environment, data_test=data_valid, train_samples=train_samples, epoch=-1, T=20, verbose=False)
    # evaluate_case_study(sampler=sampler, environment=environment, T=20)

    # ----------------- MOdel Saving -----------------------
    # Qa.save(HOME_PATH + "data/model/%s_SRDQN_%s" % (direct_name, env_params))
    evaluate_with_example(sampler=sampler, environment=environment, epoch=-1, T=20, verbose=True)
