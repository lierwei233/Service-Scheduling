import numpy as np
from scipy.optimize import linear_sum_assignment
from itertools import product
from mip import Model, xsum, BINARY, maximize
from copy import deepcopy
# import sys


class Agent:
    def __init__(self):
        # self.action_seq = np.genfromtxt('submit_201112_213454.csv', delimiter=',', dtype='int64', encoding='utf-8')
        pass

    @staticmethod
    def argsort_a2b(a, b):
        idx_a = np.argsort(a)
        idx_b = np.argsort(b)
        idx = np.argsort(idx_b)  # idx为0，1，2，……可以复现b，从而idx_a如此也可以复现b
        return idx_a[idx]

    @staticmethod
    def hungarian(W, tasks, experts):
        row_ind, col_ind = linear_sum_assignment(W)
        matches = np.array([row_ind, col_ind])
        idx = np.where(W[row_ind, col_ind] > 449328)[0]  # 999999*np.exp(-0.8)
        matches = np.delete(matches, idx, axis=1)
        matches[0, :] = tasks[matches[0, :]]
        matches[1, :] = experts[matches[1, :]]
        return matches

    def choose_rematch_task(self, matching_expert, env):
        rematch = np.array([], dtype=np.int)
        old_expert = np.array([], dtype=np.int)
        if env.processing.size and matching_expert.size:
            time_left = env.processing_task_state[:, 1]
            # time_left = env.processing_task_state[:, 1]
            W = self.find_w(env.processing, matching_expert, env, np.empty([0,]))
            n_processing = env.processing.shape[0]
            for i in range(n_processing):
                # 当前专家处理的任务剩余时间（带权） > 当前任务被新专家处理的最小所需时间
                if time_left[i] > np.min(W[i, :]):
                    rematch = np.append(rematch, env.processing[i])
                    old_expert = np.append(old_expert,
                                           env.processing_task_state[i, 0])
        return rematch, old_expert

    @staticmethod
    def choose_match_expert(expert_state, processing_task_state):
        """
        :return: matching_expert: 可以参加匹配的空闲专家
                 place: 所有专家的空位的个数
                 free: 后4分钟各个专家的空位个数
        """
        expert_state = deepcopy(expert_state)
        processing_task_state = deepcopy(processing_task_state)
        place = np.zeros(expert_state.shape[0], dtype=np.int)
        place += np.sum(expert_state[:, [0, 4, 8]] == -1, axis=1)
        free = np.zeros((4, expert_state.shape[0]), dtype=np.int)
        for i in range(4):
            processing_task_state[:, 1] = np.maximum(processing_task_state[:, 1] - 1, 0)
            idx = np.where(processing_task_state[:, 1] == 0)
            free_expert = processing_task_state[idx, 0]
            free[i, free_expert] += 1
            processing_task_state = np.delete(processing_task_state, idx, axis=0)

        matching_expert = np.where(place >= 0)[0]
        return matching_expert, place, free

    @staticmethod
    def choose_match_task(rematch, matching_expert, env):
        matching_task = np.concatenate((rematch, env.waiting, env.new_arrival))
        t = env.t
        for i in range(4):
            # 时间+1
            t = t + 1
            # 新到来任务的处理
            if t >= 1199:
                break
            else:
                matching_task = np.concatenate((matching_task, env.task_info[env.task_info[:,1] == t, 0]))
        return matching_task

    @staticmethod
    def find_w(matching_task, matching_expert, env, rematch):
        n_tasks = matching_task.shape[0]
        n_experts = matching_expert.shape[0]
        matching_task_types = env.task_info[matching_task, 2]

        # W = np.zeros((n_tasks, n_experts), dtype=np.float)
        # 考虑等待中任务的紧迫性
        # 取出所有待匹配任务中的等待任务
        m_waiting_idx = np.nonzero(np.isin(matching_task, env.waiting))
        waiting_matching_task = matching_task[m_waiting_idx]
        # 取出所有等待任务中的待匹配任务
        waiting_idx = np.nonzero(np.isin(env.waiting, matching_task))
        waiting_task = env.waiting[waiting_idx]
        # 分别按任务号排序
        idx1 = np.argsort(waiting_matching_task)
        idx2 = np.argsort(waiting_task)
        # 再按索引排序
        idx3 = np.argsort(idx1)
        waiting_idx = idx2[idx3]
        # a = a[idx1[idx3]] = b[idx2[idx3]]
        time_waited = env.waiting_task_state[waiting_idx, 1]
        time_total = env.waiting_task_state[waiting_idx, 0]
        weight = np.ones(n_tasks)
        idx = np.where(time_waited >= time_total)[0]
        time_waited[idx] = time_total[idx]
        weight[m_waiting_idx] *= np.exp(-time_waited / time_total)
        time_needed = env.expert_info[np.tile(matching_expert, n_tasks), np.repeat(matching_task_types, n_experts)].reshape(n_tasks, n_experts)

        W = time_needed * weight.reshape(-1, 1)

        # 考虑专家已工作时长
        # all_L = env.expert_state[:, -1]
        # eta = (all_L[matching_expert] - np.mean(all_L)) / (1./133 + np.mean(all_L))
        # eta[np.where(eta>2)] = 2
        # eta[np.where(eta<-2)] = -2
        # W = W * np.exp(eta)
        return W

    def mipsolve(self, matching_expert, place, free, matching_task, rematch, env, old_expert):
        time_utility, time_needed = self.findT(matching_task, rematch, env)
        m = matching_task.size
        n = matching_expert.size
        width = 1  # 时间窗宽度
        time_utility = time_utility[:width, :, :]
        model = Model('schedule')
        # model.max_min_gap_abs = 1
        model.verbose = 0
        # c = model.add_var(name="C")
        X = model.add_var_tensor(shape=(width, m, n), name='X', var_type=BINARY)
        # x = [[[model.add_var(var_type=BINARY, name='x({},{},{})'.format(t, i, j)) for j in range(n)]for i in range(m)] for t in range(width)]

        model.objective = maximize(xsum((X * time_utility).ravel()))
        # model.objective = maximize(xsum(x[t][i][j] * time_utility[t,i,j] for t in range(width) for i in range(m) for j in range(n)))

        # C是分配结果的完成时间
        # for (t,i,j) in product(range(5), range(m), range(n)):
        #     model += (c >= t + x[t][i][j] * time_needed[i,j])

        model += X.sum(axis=(0, 2)) <= 1  # 只分一次
        # 新任务到来前不分配
        t_arrival = env.task_info[matching_task, 1] - env.t
        for t in range(width):
            model += X[t, t < t_arrival, :] == 0
        # 任务调度次数限制
        r_times = np.zeros(m, dtype=np.int)
        m_p_idx = np.nonzero(np.isin(matching_task, env.processing))[0]
        m_p = matching_task[m_p_idx]
        p_m_idx = np.nonzero(np.isin(env.processing, matching_task))[0]
        p_m = env.processing[p_m_idx]
        p2m_idx = self.argsort_a2b(p_m, m_p)
        r_times[m_p_idx] = env.processing_task_state[p_m_idx, 2][p2m_idx]
        model += X[:, r_times == 5, :] == 0  # 已经5次分配的不可再分配
        # 取消效果不好的重分配
        if p_m_idx.size:
            old_time = 1000000 * np.ones(m, dtype=np.int)
            old_time[m_p_idx] = env.processing_task_state[p_m_idx, 1][p2m_idx]
            model += X[:, time_needed >= old_time.reshape(-1, 1)] == 0

        # 只分一次
        # for i in range(m):
        #     # model += xsum(X[:, i, :].ravel()) <= 1
        #     model += (xsum(x[t][i][j] for t in range(width) for j in range(n)) <= 1)

        # t_arrival = env.task_info[matching_task, 1] - env.t
        # for t in range(width):
        #     model += X[t, t < t_arrival, :] == 0  # 新任务到来前不分配
        # r_times = np.zeros(m, dtype=np.int)
        # m_processing = matching_task[np.isin(matching_task, env.processing)]
        # r_times

        # for i in range(m):
        #     if not np.isin(matching_task[i], np.concatenate((env.waiting, env.processing))):  # 新任务
        #         # 新任务到来前不分配
        #         t_arrival = env.task_info[matching_task[i], 1] - env.t  # 0~4
        #         for t in range(width):
        #             if t < t_arrival:
        #                 for j in range(n):
        #                     model += (x[t][i][j] == 0)
        #     if np.isin(matching_task[i], rematch):
        #         # 重分配任务调度次数限制
        #         idx = np.where(env.processing==matching_task[i])[0]
        #         r_times = env.processing_task_state[idx,2][0]
        #         model += (xsum(x[t][i][j] for t in range(width) for j in range(n)) + r_times <= 5)
        #         # 重匹配要有提升
        #         old_time = env.processing_task_state[idx, 1][0]
        #         for (t,j) in product(range(width), range(n)):
        #             if time_needed[i, j] >= old_time:
        #                 model += (x[t][i][j] == 0)

        # 并发数不超限
        for j in range(n):  # n=133
            # 如果j是重分配的老专家，那么对应的重分配任务可以为其增加空位
            if np.isin(j, old_expert):
                old_expert_idx = np.where(old_expert == j)[0]  # 找出j在old_expert中的位置
                corespond_rematch = rematch[old_expert_idx]  # 找出相应的参加rematch的任务
                rematch_idx = np.nonzero(np.isin(matching_task, corespond_rematch))[0]  # 找出这些任务在全部待分配任务中的位置  matching task中corespond_rematch任务的索引
                other_idx = np.nonzero(~np.isin(matching_task, corespond_rematch))[0]  # matching task中其他任务的索引
                new_expert = np.delete(range(n), j)  # 对于重分配任务，除j以外都属于新专家
                # 在计算空位时，rematch_idx对应了取出的任务，这些任务分给谁（除了j自己）都要使空位+1
                # other_idx对应放入的任务，其他任务放进来总会使空位-1
                add = xsum(X[0][np.ix_(rematch_idx, new_expert)].ravel()) - xsum(X[0, other_idx, j])
                # add = xsum(x[0][i][j] for i in rematch_idx for j in new_expert) - xsum(x[0][i][j] for i in other_idx)
                model += place[j] + add >= 0
                for t in range(1, width):
                    add = add + xsum(X[t][np.ix_(rematch_idx, new_expert)].ravel()) - xsum(X[t, other_idx, j])
                    # add = add + xsum(x[t][i][j] for i in rematch_idx for j in new_expert) - xsum(x[t][i][j] for i in other_idx)
                    model += place[j] + add + free[t - 1, j] >= 0
            else:  # 如果j不是老专家，那么仅计算任务放入即可
                add = - xsum(X[0, :, j])
                model += (place[j] + add >= 0)
                for t in range(1, width):
                    add = add - xsum(X[t, :, j])
                    model += place[j] + add + free[t - 1, j] >= 0

        # 每轮重匹配次数不要太多
        # rematch_idx = np.nonzero(np.isin(matching_task, rematch))[0]
        # model += (xsum(x[t][i][j] for t in range(width) for i in rematch_idx for j in range(n)) <= 5)


        # # 保证至少要有几个匹配
        # t_size = matching_task.size
        # # e_size = np.sum(place) + np.sum(free[0]) + np.sum(free[1])
        # e_size = np.sum(place)
        # # if t_size == rematch.size: # 只剩重分配任务了
        # #     # 把能提升的位置拿出来
        # #     rematch_size =
        # size = min(t_size, e_size)
        # model += (xsum(x[t][i][j] for t in range(width) for i in range(m) for j in range(n)) >= size)

        model.optimize()

        match_result = np.zeros((m, n), dtype=np.int)
        for (i, j) in product(range(m), range(n)):
            match_result[i, j] = X[0, i, j].x
        if np.isnan(match_result[0, 0]):  # 以防nan
            task_idx = np.array([], dtype=np.int)
            expert_idx = np.array([], dtype=np.int)
        else:
            idx = np.nonzero(match_result)
            task_idx = idx[0]
            expert_idx = idx[1]

        return [matching_task[task_idx], matching_expert[expert_idx]]

    def findT(self, matching_task, rematch, env):
        n_tasks = matching_task.shape[0]
        n_experts = env.expert_state.shape[0]
        matching_task_types = env.task_info[matching_task, 2]
        waiting_m_idx = np.nonzero(np.isin(env.waiting, matching_task))[0]
        waiting_m = env.waiting[waiting_m_idx]
        m_waiting_idx = np.nonzero(np.isin(matching_task, env.waiting))[0]
        m_waiting = matching_task[m_waiting_idx]
        idx_waiting2m = self.argsort_a2b(waiting_m, m_waiting)
        time_waited_coef = np.zeros(n_tasks)
        time_waited_coef[m_waiting_idx] = (env.waiting_task_state[waiting_m_idx, 1] / env.waiting_task_state[waiting_m_idx, 0])[idx_waiting2m]
        time_waited_coef[time_waited_coef >= 1] = 1
        time_waited_coef = np.exp(time_waited_coef * 0.5)
        time_needed = env.expert_info[np.tile(range(n_experts), n_tasks), np.repeat(matching_task_types, n_experts)].reshape(n_tasks, n_experts)

        # weight = np.ones(5, n_tasks)
        # weight[m_waiting_idx] *= np.exp(-time_waited / time_total)

        time_utility = np.ones((5, n_tasks, n_experts), dtype=np.float)
        time_min = env.expert_info.min(0)[matching_task_types]

        for t in range(5):
            time_utility[t] = np.floor(time_min[:, np.newaxis] / time_needed * np.power(0.9, t) * time_waited_coef[:, np.newaxis] * 1000000).astype(np.int)
            for i in range(n_tasks):
                if np.isin(matching_task[i], rematch):
                    idx = np.where(env.processing == matching_task[i])[0]
                    old_time = env.processing_task_state[idx, 1][0]
                    time_utility[t][i] = np.floor((old_time - time_needed[i]) * np.power(0.9, t))
        return time_utility, time_needed

    @staticmethod
    def process_result(match_result, rematch, expert_state, old_expert, expert_info):
        tasks, experts = match_result
        idx = np.nonzero(np.in1d(tasks, rematch))[0] # 获得坐标
        rematch_tasks = tasks[idx]  # 最终得到rematch的任务
        new_experts = experts[idx]  # rematch的新专家

        idx = np.nonzero(np.logical_not(np.isin(tasks, rematch_tasks)))[0]
        tasks = tasks[idx]
        experts = experts[idx]

        old_experts = np.array([], dtype=np.int)  # rematch的老专家
        if rematch_tasks.size:
            old_expert_state = expert_state[old_expert, :]
            old_expert_task = old_expert_state[:, [0, 4, 8]].reshape(-1, 3)
            for i in range(rematch_tasks.size):
                idx = np.where(old_expert_task == rematch_tasks[i])[0][0]  #
                # 第一个坐标的第一个元素
                old_experts = np.append(old_experts, old_expert[idx])

        # 把没有提升的rematch结果取消
        # if rematch_tasks.size:
        #     old_time = np.zeros(rematch_tasks.shape)
        #     new_time = np.zeros(rematch_tasks.shape)
        #     for i in range(rematch_tasks.size):
        #         idx = np.where(old_expert_task == rematch_tasks[i])
        #         old_time[i] = old_expert_state[idx[0][0], (4*idx[1][0]+2)] -\
        #                       old_expert_state[idx[0][0], (4*idx[1][0]+3)]
        #         new_time[i] = expert_info[new_experts[i], old_expert_state[idx[0][0], (4*idx[1][0]+1)]]
        #     idx = np.where(old_time > new_time)
        #     rematch_tasks = rematch_tasks[idx]
        #     old_experts = old_experts[idx]
        #     new_experts = new_experts[idx]

        # 匹配中应避免老专家的任务分给了老专家
        if rematch_tasks.size:
            idx = np.array([], dtype=np.int)  # 出现连续分配的位置索引
            for i in range(rematch_tasks.size):
                if old_experts[i] == new_experts[i]:
                    idx = np.append(idx, i)
            rematch_tasks = np.delete(rematch_tasks, idx)
            old_experts = np.delete(old_experts, idx)
            new_experts = np.delete(new_experts, idx)

        # 普通任务，普通专家，重新分配的任务，重新分配对应的老专家，重新分配对应的新专家
        return [tasks, experts, rematch_tasks, old_experts, new_experts]

    def choose_act(self, env):
        matching_expert, place, free = self.choose_match_expert(env.expert_state, env.processing_task_state)
        # old_expert 是
        rematch, old_expert = self.choose_rematch_task(matching_expert, env)
        # rematch = env.processing
        # old_expert = env.processing_task_state[:, 0]

        # if len(rematch):
        #     print(rematch)


        matching_task = self.choose_match_task(rematch, matching_expert, env)

        # W = self.find_w(matching_task, matching_expert, env, rematch)
        if matching_task.size:
            match_result = self.mipsolve(matching_expert, place, free, matching_task, rematch, env, old_expert)
            # match_result = (self.action_seq[self.action_seq[:, 2] == env.t, :][:, :2] - 1).T
            match_result = self.process_result(match_result, rematch, env.expert_state, old_expert, env.expert_info)
        else:
            match_result = [np.array([], dtype=np.int) for _ in range(5)]
        # 普通任务，普通专家，重新分配的任务，重新分配对应的老专家，重新分配对应的新专家
        return match_result
