from environment import Environment
from agent import Agent
import numpy as np
import time
from score_computer import Evaluator
import sys


class Logger:
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

    def close_file(self):
        self.log.close()


def load_data():
    try:
        expert_info = np.genfromtxt('tcdata/sic_semi_process_time_matrix.csv', delimiter=',', encoding='utf-8', dtype=np.int)
        task_info = np.genfromtxt('tcdata/sic_semi_work_order.csv', encoding='utf-8', dtype=np.int, delimiter=',')
    except IOError:
        expert_info = np.genfromtxt('process_time_matrix.csv', delimiter=',', encoding='utf-8', dtype=np.int)
        task_info = np.genfromtxt('work_order.csv', encoding='utf-8', dtype=np.int, delimiter=',')
    print("file reading succeeded")
    expert_info = np.delete(expert_info, [0], axis=0)
    expert_info = np.delete(expert_info, [0], axis=1)
    task_info[0, 0] = 1
    task_info[:, [0, 2]] -= 1  # 从0开始

    return task_info, expert_info


if __name__ == "__main__":
    GENERATE_FILE = False
    GENERATE_GRAPH = False
    UPLOAD = True
    # logger = Logger('build/log.txt')
    # sys.stdout = logger

    task_info, expert_info = load_data()
    # task_info = task_info[:1000]

    a = np.copy(expert_info)  # 确定专家等级
    expert_level = np.zeros(expert_info.shape[0])
    a[a < 999999] = 1
    a[a == 999999] = 0
    sum_row = a.sum(axis=1)
    sum_row = sum_row.astype(int)
    expert_level[np.where(sum_row < 10)] = 1

    is_done = False
    env = Environment(expert_level, expert_info=expert_info, task_info=task_info).reset()
    agent = Agent()
    strategy = np.zeros(shape=[0, 3], dtype=np.int)

    while not is_done:
        print("t={}\t执行中任务数：{}\t等待任务数：{}\t新来任务数：{}".format(env.t, len(env.processing), len(env.waiting), len(env.new_arrival)), end='\t')
        action = agent.choose_act(env)  # 普通任务，普通专家，重新分配的任务，重新分配对应的老专家，重新分配对应的新专家

        tasks, experts, rematch_tasks, old_experts, new_experts = action
        all_tasks = np.concatenate([tasks, rematch_tasks])
        all_experts = np.concatenate([experts, new_experts])
        t = env.t * np.ones(all_tasks.shape[0], dtype=np.int)
        strategy = np.vstack((strategy, np.vstack([all_tasks, all_experts, t]).T))

        is_done = env.step(action)  # action作用于env
        print(">>>分配后>>>\t执行中任务数：{}\t等待任务数：{}".format(len(env.processing), len(env.waiting)), end='\t')
        env.time_go(is_done)  # 时间流逝1min
        print()

    strategy[:, :2] += 1
    if GENERATE_FILE:
        submit_filename = time.strftime('build/submit_%y%m%d_%H%M%S.csv', time.localtime())
        np.savetxt(submit_filename, strategy, delimiter=',', fmt='%d')
    if UPLOAD:
        np.savetxt('result.csv', strategy, delimiter=',', fmt='%d')

    # 使用计分器得到分数
    task_info_ = task_info.copy()
    task_info_[:, [0, 2]] += 1
    evaluator = Evaluator(assignments=strategy, task_info=task_info_, expert_info=expert_info).fit()
    print("Score:", evaluator.get_score())
    evaluator.print_performance()
    # if GENERATE_GRAPH:
        # evaluator.draw_task_overview(save_fig=True)
        # evaluator.draw_expert_load(save_fig=True)
        # evaluator.draw_task_overtime(save_fig=True)
        # evaluator.draw_task_efficiency(save_fig=True)

    # sys.stdout = sys.__stdout__  # 恢复原来的重定向
    # logger.close_file()
    # env.expert_load_seq = env.expert_load_seq.reshape(-1, 133, 6)
    # import pickle
    # with open('visualization/data_la.pkl', 'wb') as file:
    #     pickle.dump([env.new_arrival_seq, env.waiting_seq, env.processing_seq, env.expert_load_seq, env.expert_work_time_seq], file)
