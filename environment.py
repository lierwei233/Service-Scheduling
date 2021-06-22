import numpy as np


class Environment:
    def __init__(self, expert_level, task_info, expert_info):
        self.max_t_begin = np.max(task_info[:, 1])
        self.min_t_begin = np.min(task_info[:, 1])
        self.t = None
        self.processing = None
        self.waiting = None
        self.new_arrival = None
        self.expert_state = None
        self.expert_work_time = None
        self.waiting_task_state = None
        self.processing_task_state = None
        self.expert_level = expert_level
        self.task_level = np.zeros(task_info.shape[0], dtype=np.int)
        self.task_info = task_info
        self.expert_info = expert_info
        self.waiting_seq = {}
        self.processing_seq = {}
        self.new_arrival_seq = {}
        self.expert_load_seq = {}  # [任务1，任务2，任务3，总工作时长，处理可处理任务的时长，处理不可处理任务的时长]
        self.expert_work_time_seq = {}

    def reset(self):
        self.t = self.min_t_begin
        self.processing = np.array([], dtype=np.int)
        self.waiting = np.array([], dtype=np.int)
        self.new_arrival = self.task_info[self.task_info[:, 1] == self.t, 0]

        self.expert_state = -1 * np.ones([self.expert_info.shape[0], 15], dtype=np.int)  # 任务号、任务类别、总需时间、已处理时间，最后一列是专家总工作时长
        self.expert_state[:, 12:] = 0
        self.expert_work_time = np.zeros(shape=[133, 107], dtype=np.int)
        self.waiting_task_state = np.zeros(shape=[0, 2], dtype=np.int)  # 总等待时间、已等待时间
        self.processing_task_state = np.zeros(shape=[0, 3], dtype=np.int)  # 专家、剩余时间、转换次数
        return self

    def step(self, action):
        tasks, experts, rematch_tasks, old_experts, new_experts = action
        is_done = (self.t > self.max_t_begin) and not self.processing.size and not self.waiting.size and not self.new_arrival.size

        if not is_done:
            # 任务分配
            n_rematch_tasks = rematch_tasks.shape[0]
            n_tasks = tasks.shape[0]
            # 遍历rematch任务
            for i in range(n_rematch_tasks):
                # 老专家进度置零
                old_expert_state = self.expert_state[old_experts[i], [0, 4, 8]]
                insert_idx = np.where(old_expert_state == rematch_tasks[i])[0][0] * 4
                insert_idx = np.array([0, 1, 2, 3]) + insert_idx
                self.expert_state[old_experts[i], insert_idx] = -1
            for i in range(n_rematch_tasks):
                # 把任务给新专家
                rematch_task_type = self.task_info[rematch_tasks[i], 2]
                new_expert_state = self.expert_state[new_experts[i], [0, 4, 8]]
                insert_idx = np.where(new_expert_state == -1)[0][0] * 4
                insert_idx = np.array([0, 1, 2, 3]) + insert_idx
                info = [rematch_tasks[i],
                        rematch_task_type,
                        self.expert_info[new_experts[i], rematch_task_type],
                        0]
                self.expert_state[new_experts[i], insert_idx] = info
                idx = np.where(self.processing == rematch_tasks[i])[0]
                self.processing_task_state[idx, :] = [new_experts[i], self.expert_info[new_experts[i], rematch_task_type],
                                                      self.processing_task_state[idx, 2] + 1]
            # 遍历新分配任务
            for i in range(n_tasks):
                expert_state = self.expert_state[experts[i], [0, 4, 8]]
                insert_idx = np.where(expert_state == -1)[0][0] * 4
                insert_idx = np.array([0, 1, 2, 3]) + insert_idx
                task_type = self.task_info[tasks[i], 2]
                process_time = self.expert_info[experts[i], task_type]
                info = [tasks[i], task_type, process_time, 0]
                self.expert_state[experts[i], insert_idx] = info
                self.processing = np.append(self.processing, tasks[i])
                self.processing_task_state = np.vstack((self.processing_task_state,
                                                        [experts[i], process_time, 1]))

            # tasks和waiting比对，重合部分从waiting删除，对应信息从wts删除
            idx = np.nonzero(np.logical_not(np.in1d(self.waiting, tasks)))[0]
            self.waiting = self.waiting[idx]
            self.waiting_task_state = self.waiting_task_state[idx].reshape(
                -1, 2)

            # new arrival 里没分配的放到waiting，在wts中添加对应信息
            idx = np.nonzero(np.logical_not(np.in1d(self.new_arrival, tasks)))[0]
            add_tasks = self.new_arrival[idx]
            self.waiting = np.concatenate((self.waiting, add_tasks))
            idx = np.nonzero(np.in1d(self.task_info[:, 0], add_tasks))[0]
            add_tasks_info = self.task_info[idx, 3]
            add_tasks_info = np.vstack((add_tasks_info, np.zeros(
                add_tasks.shape[0]))).T
            self.waiting_task_state = np.vstack((self.waiting_task_state,
                                                 add_tasks_info))

        return is_done

    def time_go(self, is_done):
        self.waiting_seq[self.t] = self.waiting
        self.new_arrival_seq[self.t] = self.new_arrival
        self.processing_seq[self.t] = self.processing
        self.expert_load_seq[self.t] = self.expert_state[:, [0, 4, 8, 12, 13, 14]]
        self.expert_work_time_seq[self.t] = self.expert_work_time

        if not is_done:
            # 专家处理时间+1，任务剩余时间-1
            if self.processing.size:
                self.processing_task_state[:, 1] = np.maximum(self.processing_task_state[:, 1] - 1, 0)
            idx = list(np.where(self.expert_state[:, [0, 4, 8]] != -1))
            idx[1] = idx[1] * 4 + 3  # 第二个坐标加3
            idx = tuple(idx)
            self.expert_state[idx] += 1
            self.expert_state[:, 12] += np.sum(self.expert_state[:, [0, 4, 8]] != -1, axis=1)
            self.expert_state[:, 13] += np.sum((self.expert_state[:, [2, 6, 10]] < 999999) * (self.expert_state[:, [2, 6, 10]] > 0), axis=1)
            self.expert_state[:, 14] += np.sum(self.expert_state[:, [2, 6, 10]] == 999999, axis=1)
            processing_task_types = self.task_info[self.processing, 2]
            processing_task_experts = self.processing_task_state[:, 0]
            self.expert_work_time[processing_task_experts, processing_task_types] += 1
            # 已经处理完的任务从processing中删除
            finish_idx = np.where(self.processing_task_state[:, 1] <= 0)[0]
            if finish_idx.size:
                finish_expert = self.processing_task_state[finish_idx, 0]
                finish_expert_tasks = self.expert_state[np.ix_(finish_expert,
                                                               [0, 4, 8])]
                insert_idx = np.where(finish_expert_tasks == self.processing[finish_idx].reshape(-1, 1))[
                    1]  # 取第二个坐标。理论上每行只能找到一个。
                finish_expert = np.repeat(finish_expert, 4)  # 每个index重复4次，便于后面的赋值
                insert_idx = np.tile(insert_idx * 4, (4, 1)).T  # 取转置便于后面的加1加2加3操作
                insert_idx += np.array([0, 1, 2, 3])  # 第i列加i
                insert_idx = insert_idx.ravel()  # 最后重新变为1维数组，idx成型
                # 理论上此时finish_expert有原来的4倍长度，insert_idx也是原来的4倍长度
                self.expert_state[finish_expert, insert_idx] = -1  # 这些idx都应该赋值-1
                self.processing = np.delete(self.processing, finish_idx)
                self.processing_task_state = np.delete(self.processing_task_state, finish_idx, axis=0)

            # wts里面时间加1
            if self.waiting.size:
                self.waiting_task_state[:, 1] += 1

            # 时间+1
            self.t += 1
            # 新到来任务的处理
            if self.t <= self.max_t_begin:
                self.new_arrival = self.task_info[self.task_info[:, 1] == self.t, 0]
            else:
                self.new_arrival = np.zeros(shape=[0, ], dtype=np.int)

