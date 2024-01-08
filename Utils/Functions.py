import collections
from platform import mac_ver
from Utils.JobReader import skill_lst, skill_dict, n_skill, sample_info, read_offline_samples, read_skill_graph, itemset_process, get_frequent_itemset
from Utils.Utils import dense_to_sparse, dense_to_sparse_prob, sparse_to_dense
import time
import numpy as np
import pprint
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from CONFIG import HOME_PATH


def evaluate_with_example(sampler, environment, epoch, T=16, verbose=False):
    time_start = time.time()

    N_test = 1
    step_salary, step_easy, step_r = [0] * min(20, T * 10), [0] * min(20, T * 10), [0] * min(20, T * 10)
    avg_easy, avg_salary = 0, 0
    avg_start = 0


    # 初始化
    environment.clear()
    step_metrics = []
    skill_set = ['python', 'java', 'hadoop', 'hive', '性能', '编程', '性能优化', '优化', 'c/c++', '代码', '数据结构', 'coding', 'linux/unix', '并发', '架构', '算法', '统计', 'linux', 'c++', 'c']
    prefix = [skill_dict[u] for u in skill_set]

    salary_start = environment.add_prefix(prefix)
    prefix_names = [skill_lst[u] for u in prefix]
    append_skills = []
    for t in range(T):
        s, q = sampler.sample(environment.state)
        easy, salary, r = environment.add_skill(s, evaluate=True)
        step_metrics.append((r, easy, salary, q))

        append_skills.append(skill_lst[s])
    print(append_skills)


    # 评价指标
    avg_start += salary_start / N_test
    avg_easy += sum([val[1] for val in step_metrics]) / T / N_test
    avg_salary += sum([val[2] for val in step_metrics]) / T / N_test
    for j, k in enumerate(range(1, T + 1)):
        step_r[j] += step_metrics[k - 1][0] / N_test
        step_easy[j] += step_metrics[k - 1][1] / N_test
        step_salary[j] += step_metrics[k - 1][2] / N_test

    time_end = time.time()
    sec = time_end - time_start
    # 输出
    print("[time:%d, epoch %d] avg_easy: %.4f, avg_salary: %.4f, salary_start: %.4f\n step_salary: " % (sec, epoch, avg_easy, avg_salary, avg_start), end="")

    for j, k in enumerate(range(1, T + 1)):
        print("%.4f," % step_salary[j], end="")
    print("\n step_easy: ", end="")

    for j, k in enumerate(range(1, T + 1)):
        print("%.4f," % step_easy[j], end="")
    print("")

def evaluate(sampler, environment, data_test, train_samples, epoch, T=16, verbose=False):
    time_start = time.time()

    N_test = len(data_test)
    step_salary, step_easy, step_r = [0] * min(20, T * 10), [0] * min(20, T * 10), [0] * min(20, T * 10)
    avg_easy, avg_salary = 0, 0
    avg_start = 0
    # evaluate q range
    q_salary_range = collections.defaultdict(list)
    q_easy_range = collections.defaultdict(list)

    for x, y in data_test:
        # 初始化
        environment.clear()
        step_metrics = []
        prefix = train_samples[x][0]
        salary_start = environment.add_prefix(prefix)
        prefix_names = [skill_lst[u] for u in prefix]
        append_skills = []
        for t in range(T):
            s, q = sampler.sample(environment.state)
            easy, salary, r = environment.add_skill(s, evaluate=True)
            step_metrics.append((r, easy, salary, q))
            # evaluate q range

            append_skills.append(skill_lst[s])
        #if verbose:
            #print(prefix_names, append_skills)

        # 评价指标
        avg_start += salary_start / N_test
        avg_easy += sum([val[1] for val in step_metrics]) / T / N_test
        avg_salary += sum([val[2] for val in step_metrics]) / T / N_test
        for j, k in enumerate(range(1, T + 1)):
            step_r[j] += step_metrics[k - 1][0] / N_test
            step_easy[j] += step_metrics[k - 1][1] / N_test
            step_salary[j] += step_metrics[k - 1][2] / N_test

    time_end = time.time()
    sec = time_end - time_start
    # 输出
    print("[time:%d, epoch %d] avg_easy: %.4f, avg_salary: %.4f, salary_start: %.4f\n step_salary: " % (sec, epoch, avg_easy, avg_salary, avg_start), end="")

    for j, k in enumerate(range(1, T + 1)):
        print("%.4f," % step_salary[j], end="")
    print("\n step_easy: ", end="")

    for j, k in enumerate(range(1, T + 1)):
        print("%.4f," % step_easy[j], end="")
    print("")


def evaluate_step(sampler, environment, data_test, train_samples, T=16):
    step_easy_lst, step_salary_lst = [], []
    salary_lst = []
    for x, y in data_test:
        step_salary, step_easy = [], []
        # 初始化
        environment.clear()
        prefix = train_samples[x][0][:y]
        salary_lst.append(environment.add_prefix(prefix))
        append_skills = []
        for t in range(T):
            s, q = sampler.sample(environment.state)
            easy, salary, r = environment.add_skill(s, evaluate=True)
            step_easy.append(easy)
            step_salary.append(salary)
            append_skills.append(skill_lst[s])
        step_easy_lst.append(step_easy)
        step_salary_lst.append(step_salary)
    return salary_lst, step_easy_lst, step_salary_lst

def evaluate_case_study(sampler, environment, T, num1, num2):
    environment.clear()
    step_metrics = []
    step_salary, step_easy, step_r = [0] * min(20, T * 10), [0] * min(20, T * 10), [0] * min(20, T * 10)
    avg_easy, avg_salary = 0, 0
    avg_start = 0


    ### Case study 1
    # prefix = [6, 76, 248, 383, 502, 507, 581, 631, 725, 746, 812, 828, 846, 882, 958, 953, 985]
    ### Case study 2
    # prefix = [160, 671, 89, 856, 843, 631, 779, 552, 301, 434, 581, 212]
 
    # Case study 1 by Ying
    # skill_set = ['前端开发', '性能', 'h5', 'dom', 'electron', 'database', '编程', 'web前端', 'html', 'css', '代码', 'javascirpt', 'ui', 'web', 'http', '优化', '框架']
    # Case Study 2 by Ying
    skill_set = ['python', 'java', 'hadoop', 'hive', '性能', '编程', '性能优化', '优化', 'c/c++', '代码', '数据结构', 'coding', 'linux/unix', '并发', '架构', '算法', '统计', 'linux', 'c++', 'c']
    
    prefix = [skill_dict[u] for u in skill_set]

    # training set 
    # prefix = [14, 134, 160, 231, 434, 460, 490, 550, 610, 701, 756, 819, 873, 939, 944, 974]

    prefix_names = [skill_lst[u] for u in prefix]
    print("prefix is:", prefix_names)

    salary_start = environment.add_prefix(prefix)
    append_skills = []

    for t in range(T):
        s, q, attention_salary, salary_prototypes, attention_difficulty, difficulty_prototypes, salary_action_value, difficulty_action_value, frequency_set, orginal_frequency_set, chosen_action, difficulty_mapped = sampler.sample_with_batch(environment.state)

        #skill = 245
        #s, q, attention_salary, salary_prototypes, attention_difficulty, difficulty_prototypes, salary_action_value, difficulty_action_value, frequency_set, orginal_frequency_set, chosen_action, difficulty_mapped = sampler.sample_with_batch_with_skill(environment.state, skill)

        # add loss functions
        loss, encoder_loss, div_loss, freq_loss = sampler.sample_with_internal_things(environment.state)

        # evaluate candidate pool
        (q_salary, q_easy), act_lst, q = sampler.sample_top_candidates(environment.state)
   
        print("-------------------------------")
        print("Skill is:", skill_lst[s])
        print("-------------------------------")

        easy, salary, r = environment.add_skill(s, evaluate=True)
        step_metrics.append((r, easy, salary, q, attention_salary, salary_prototypes, attention_difficulty, difficulty_prototypes, salary_action_value[0], difficulty_action_value[0], frequency_set, orginal_frequency_set, loss, encoder_loss, div_loss, freq_loss, chosen_action, difficulty_mapped, (q_salary, q_easy), act_lst, q))
        append_skills.append(skill_lst[s])

    # 评价指标
    avg_start += salary_start
    avg_easy += sum([val[1] for val in step_metrics]) / T 
    avg_salary += sum([val[2] for val in step_metrics]) / T
    for j, k in enumerate(range(1, T + 1)):
        step_r[j] += step_metrics[k - 1][0] 
        step_easy[j] += step_metrics[k - 1][1]
        step_salary[j] += step_metrics[k - 1][2]

    # 输出
    print("avg_easy: %.4f, avg_salary: %.4f, salary_start: %.4f\n step_salary: " % (avg_easy, avg_salary, avg_start), end="")

    for j, k in enumerate(range(1, T + 1)):
        print("%.4f," % step_salary[j], end="")
    print("\n step_easy: ", end="")

    for j, k in enumerate(range(1, T + 1)):
        print("%.4f," % step_easy[j], end="")
    print("")

    np.set_printoptions(threshold=np.inf)
    m = 1
    # 9
    # 
    action_number = step_metrics[m-1][16]
    # action_number = [95]
    print("Pool has the following skills: ")
    print([skill_lst[u] for u in step_metrics[m-1][19][0]])
    
    ### define your action value here
    # action_number = [15]
    print(action_number)
    print("Selected action is:", append_skills[m-1])

    print("------------------- the %sth step detailed information-------------------" %(m))

    print("Learning Path is:", append_skills)
    #print("Learning Path salaries", [step_metrics[i][2] for i in range(20)])
    #print("Learning Path difficulties", [step_metrics[i][1] for i in range(20)])

    q_value_index = np.argsort(step_metrics[m-1][20][0])
    candidate_actions = []
    candidate_q_values = []
    candidate_salary_q_values = []
    candidate_difficulty_q_values = []
    candidate_salaries, candidate_difficulties = [], []
    pool_size = 50

    for i in range(pool_size):
        index = q_value_index[pool_size - i - 1]
        candidate_actions.append(step_metrics[m-1][19][0][index])

        environment.clear()
        salary_start = environment.add_prefix(prefix)
        easy, salary, r = environment.add_skill(step_metrics[m-1][19][0][index], evaluate=True)

        candidate_salaries.append(salary)
        candidate_difficulties.append(easy)
        candidate_q_values.append(step_metrics[m-1][20][0][index])
        candidate_salary_q_values.append(step_metrics[m-1][18][0][0][index])
        candidate_difficulty_q_values.append(step_metrics[m-1][18][1][0][index])

    print("Candidate Actions:")
    print([skill_lst[u] for u in candidate_actions])
    print("Candidate q values:")
    print(list(map(lambda number: round(number, 2), candidate_q_values)))
    print("Candidate Salary q values:")
    print(list(map(lambda number: round(number, 2), candidate_salary_q_values)))
    print("Candidate Difficulty q values:")
    print(list(map(lambda number: round(number, 2), candidate_difficulty_q_values)))
    print("Candidate Salaries:")
    print(list(map(lambda number: round(number, 2), candidate_salaries)))
    print("Candidate Difficulties:")
    print(list(map(lambda number: round(number, 2), candidate_difficulties)))

    
    print("----------------------------------------------------------")
    print("------------------- Salary Part ----------------------")
    print("----------------------------------------------------------")    
    
    print("original input freq set is:", [skill_lst[u] for u in dense_to_sparse_prob(step_metrics[m-1][11][0], 0.8)])
    print("input freq set mapping back is:", [skill_lst[u] for u in dense_to_sparse_prob(step_metrics[m-1][10][0], 0.8)])

    print("Similarity Values for Salary Part are:", step_metrics[m-1][4][0])
    print("The Chosen Action-Values for Salary Part are:", step_metrics[m-1][8][action_number][0])
    salary_scores = np.multiply(step_metrics[m-1][4][0], step_metrics[m-1][8][action_number][0])
    salary_rank = np.argsort(salary_scores)
    salary_dict = []
    for i, item in enumerate(salary_rank):
        x = (item + 1, step_metrics[m-1][4][0][item], step_metrics[m-1][8][action_number][0][item])
        salary_dict.append(x)
    
    for i in range(num1):
        print(salary_dict[num1 - 1 - i])

    #for i in range(40):
    #    print("%s-th salary prototype is:" %(i), [skill_lst[u] for u in dense_to_sparse_prob(step_metrics[m-1][5][i], 0.8)])

    print("----------------------------------------------------------")
    print("------------------- Difficulty Part ----------------------")
    print("----------------------------------------------------------")

    #print("original input freq set is:", [skill_lst[u] for u in dense_to_sparse_prob(step_metrics[m-1][11][0], 0.8)])
    #print("input freq set mapping back is:", [skill_lst[u] for u in dense_to_sparse_prob(step_metrics[m-1][17][0], 0.8)])

    # Similarity Part
    print("Similarity Values for Difficulty Part are:", step_metrics[m-1][6][0])
    print("The Chosen Action-Values for Difficulty Part are:", step_metrics[m-1][9][action_number][0])
    difficulty_scores = np.multiply(step_metrics[m-1][6][0], step_metrics[m-1][9][action_number][0])
    difficulty_rank = np.argsort(difficulty_scores)
    difficulty_dict = []
    for i, item in enumerate(difficulty_rank):
        x = (item + 1, step_metrics[m-1][6][0][item], step_metrics[m-1][9][action_number][0][item])
        difficulty_dict.append(x)
    
    for i in range(num2):
        print(difficulty_dict[num2 - 1 - i])
    

    #for i in range(60):
    #       print("%s-th difficulty prototype is:" %(i), [skill_lst[u] for u in dense_to_sparse_prob(step_metrics[m-1][7][i], 0.8)])
    
    print("------------", "loss function is", "------------")
    # for i in range(1, 20):
    print("(%s-th) overall loss is:" %(m), np.mean(step_metrics[m-1][12]))
    print("overall encoder loss is", np.mean(step_metrics[m-1][13]))
    print("overall diversity loss is", np.mean(step_metrics[m-1][14]))
    print("overall frequency loss is", np.mean(step_metrics[m-1][15]))
    #for i in range(2):
    #    print("%s-th easy prototype is:" %(i), [skill_lst[u] for u in dense_to_sparse(step_metrics[m-1][7][i])])
