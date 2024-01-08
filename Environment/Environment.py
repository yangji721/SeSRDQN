
from xmlrpc.client import boolean


class Environment(object):
    def __init__(self, lambda_d, d_estimator, job_matcher, n_skill):
        self.lambda_d = lambda_d
        self.d_estimator = d_estimator
        self.job_matcher = job_matcher
        self.n_skill = n_skill
        self.state = [0] * self.n_skill
        self.state_list = []

    def get_reward(self, easy, salary):
        return self.lambda_d * easy + salary

    def clear(self):
        self.job_matcher.reset()
        self.d_estimator.clear()
        self.state = [0] * self.n_skill
        self.state_list = []

    def add_prefix(self, prefix):
        for s in prefix:
            self.add_skill(s, evaluate=False)
        salary_start = self.job_matcher.top_average()
        return salary_start

    def add_skill(self, s, evaluate=True):
        self.job_matcher.add(int(s))
        easy = self.d_estimator.predict_and_add(int(s))
        salary = -1
        if evaluate:
            salary = self.job_matcher.top_average()
        self.state[s] = 1
        self.state_list.append(s)
        return easy, salary, self.get_reward(easy, salary)

    def get_added_skill_dif(self, s, evaluate=True):
        self.job_matcher.add(int(s))
        easy = self.d_estimator.predict_and_add(int(s))
        return easy
    
    def get_frequent_set(self, skill_set, flag):
        return self.d_estimator.get_frequent_set(skill_set, bool(flag))

    def get_frequent_set_step(self, s, flag):
        return self.d_estimator.get_frequent_set_step(int(s), bool(flag))
