import numpy as np
import matplotlib.pyplot as plt


class Evaluator:
    def __init__(self, filename=None, assignments=None, task_info=None, expert_info=None):
        if expert_info is None:
            self.expert_info = np.genfromtxt('tcdata/sic_semi_process_time_matrix.csv', delimiter=',', encoding='utf-8', dtype='int64')
            self.expert_info = np.delete(self.expert_info, [0], axis=0)
            self.expert_info = np.delete(self.expert_info, [0], axis=1)
        else:
            self.expert_info = expert_info
        if assignments is None:
            self.assignments = np.genfromtxt(filename, delimiter=',', dtype='int64', encoding='utf-8')
        else:
            self.assignments = assignments
        if task_info is None:
            self.task_info = np.genfromtxt('tcdata/sic_semi_work_order.csv', encoding='utf-8', dtype='int64', delimiter=',')
            self.task_info[0, 0] = 1
        else:
            self.task_info = task_info
        self.is_assignments_available_ = False

    def check_expert_status(self, expert_id, time):
        """
        判断专家是否有空闲接任务（即小二并发数是否超限）
        """
        task_ids = self.expert_status_[expert_id]
        for i, task_id in enumerate(task_ids):
            task_type = self.task_type_[task_id]
            start_time = self.T_switch_[task_id, 0]
            process_time = self.expert_info[expert_id, task_type]
            if time - start_time >= process_time:
                self.expert_status_[expert_id, i] = -1  # 任务执行完毕，清空
        if (task_ids == -1).any():
            return True
        else:
            return False

    @staticmethod
    def check_valid_id(expert_id, task_id, n_experts, n_tasks):
        # 如果expert_id和task_id都是scalar，那么将它们变成np.array对象，这样就可以进行all运算。
        expert_id = np.asarray(expert_id)
        task_id = np.asarray(task_id)
        if (0 <= expert_id).all() and (expert_id <= n_experts - 1).all() and \
                (0 <= task_id).all() and (task_id <= n_tasks - 1).all():
            return True
        else:
            return False

    def check_valid_time(self, task_id, time):
        task_id = np.asarray(task_id)
        time = np.asarray(time)
        if (time >= self.task_info[task_id, 1]).all():
            return True
        else:
            return False
        
    def fit(self):
        n_tasks = self.task_info.shape[0]
        n_assignments = self.assignments.shape[0]
        n_experts = self.expert_info.shape[0]
        switch_counter = np.zeros(n_tasks, dtype='int64')
        self.n_tasks_ = n_tasks
        self.n_assignments_ = n_assignments
        self.n_experts_ = n_experts
        self.T_stay_ = np.zeros((n_experts, n_tasks), dtype='int64')
        self.T_start_ = -1 * np.ones(n_tasks, dtype='int64')
        self.T_last_ = -1 * np.ones(n_tasks, dtype='int64')
        self.T_finish_ = -1 * np.ones(n_tasks, dtype='int64')
        self.T_switch_ = -1 * np.ones((n_tasks, 2), dtype='int64')  #
        # 任务最后一次被分配时对应的时间(0)、专家(1)
        self.expert_status_ = -1 * np.ones((n_experts, 3), dtype='int64')  #
        # 保存对应的task_id
        self.task_type_ = self.task_info[:, 2] - 1

        self.is_assignments_available_ = True
        assignment_id = 0
        while assignment_id < n_assignments:
            # 将同一时刻的分配都提取出来
            assignments = []
            this_assignment = self.assignments[assignment_id]
            assignments.append(this_assignment)
            while (assignment_id < n_assignments - 1) and \
                    self.assignments[assignment_id+1, 2] == this_assignment[2]:
                assignment_id += 1
                assignments.append(self.assignments[assignment_id])
            assignments = np.array(assignments).reshape(-1, 3)  #
            # 保证assignments是一个二维数组
            # <任务ID>, <服务技术专家ID>，<任务开始处理时间(单位:分钟)>
            n_s_assignments = assignments.shape[0]
            task_ids = assignments[:, 0] - 1  # 因为表格中从0开始，所以要减1. 下同
            expert_ids = assignments[:, 1] - 1
            if not self.check_valid_id(expert_ids, task_ids, n_experts,
                                       n_tasks):
                print('存在无效id')
                self.is_assignments_available_ = False
                return self
            times = assignments[:, 2]
            if not self.check_valid_time(task_ids, times):
                print('任务调度时间小于任务到达时间')
                self.is_assignments_available_ = False
                return self

            for i in range(n_s_assignments):
                # if i==3 and assignment_id==401:
                #     print("haha")
                task_id = task_ids[i]
                task_type = self.task_type_[task_id]
                expert_id = expert_ids[i]
                time = times[i]
                # if time == 556:
                #     print("haha")
                # 判断专家是否忙到无法接任务
                if not self.check_expert_status(expert_id, time):
                    # 如果专家忙，那么判断该专家正在执行的任务是否在这一步被分配给其它专家。
                    # 注意 [i:]
                    if np.intersect1d(self.expert_status_[expert_id], task_ids[i:]).size == 0:
                        print('小二并发处理量超限')
                        self.is_assignments_available_ = False
                        return self
                    else:
                        self.expert_status_[expert_id, np.nonzero(np.in1d(self.expert_status_[expert_id], task_ids[i:]))[0][0]] = -1
                # 填充T_start。根据任务是否重新分配分情况考虑。
                if self.T_start_[task_id] == -1:
                    self.T_start_[task_id] = time
                else:
                    last_stay_time = time - self.T_switch_[task_id, 0]
                    time_needed = self.expert_info[self.T_switch_[task_id, 1],
                                                   task_type]
                    if last_stay_time >= time_needed:
                        # 如果任务已经被处理完毕而再次分配，说明调度出错。
                        print('已完成任务再次分配')
                        self.is_assignments_available_ = False
                        return self
                    else:
                        self.T_stay_[self.T_switch_[task_id, 1], task_id] += \
                            last_stay_time
                switch_counter[task_id] += 1
                if switch_counter[task_id] > 5:
                    print('任务调度次数超过限额')
                    self.is_assignments_available_ = False
                    return self

                self.T_switch_[task_id, :] = np.array([time, expert_id])
                self.expert_status_[np.where(self.expert_status_ == task_id)]\
                    = -1
                self.expert_status_[expert_id, np.argwhere(
                    self.expert_status_[expert_id, :] == -1)[0]] = task_id

            assignment_id += 1
        if (self.T_start_ == -1).any():  # 存在未分配的任务，失败
            print('存在未分配的任务')
            self.is_assignments_available_ = False
            return self
        # Too many indices for array?
        for i in range(n_tasks):
            final_time = self.expert_info[self.T_switch_[i, 1],
                                               self.task_type_[i]]
            self.T_stay_[self.T_switch_[i, 1], i] += final_time
            self.T_last_[i] = final_time
            self.T_finish_[i] = self.T_switch_[i, 0] + final_time
        # self.T_last_ = self.expert_info[tuple(zip(self.T_switch_[:, 1],
        #                                           self.task_type_))]
        # one_shot_ids = np.where(switch_counter == 0)[0]
        # one_shot_types = self.task_info[one_shot_ids, 2].ravel() - 1  #
        # # 每个task_id对应的问题类型
        # for i in range(len(one_shot_ids)):
        #     self.T_stay_[self.T_switch_[i, 1], i] = self.expert_info[
        #         self.T_switch_[i, 1], one_shot_types[i]]
        # self.T_stay_[:, one_shot_ids] = \
        #     self.expert_info[tuple(zip(self.T_switch_[one_shot_ids, 1],
        #                                one_shot_types))]

        return self
            
    def get_score(self):
        if self.is_assignments_available_:
            L = np.sum(self.T_stay_, axis=1).astype(np.float64) / (60 * 8 * 3)
            delta_waiting_time = self.T_start_ - self.task_info[:, 1] - self.task_info[:, 3]
            delta_waiting_time[delta_waiting_time < 0] = 0
            M = delta_waiting_time.astype(np.float64) / self.task_info[:, 3]
            # R = -1 * np.ones(len(self.T_finish_), dtype='float64')
            min_process_time_idx = np.argmin(self.expert_info[:, self.task_type_], axis=0)
            self.min_process_time_ = self.expert_info[min_process_time_idx, self.task_type_]
            self.task_total_time = self.T_finish_ - self.task_info[:, 1]
            R_ = self.T_last_.astype(np.float64) / (self.T_finish_ - self.task_info[:, 1])
            R = np.sum(self.min_process_time_) / np.sum(self.T_finish_ - self.task_info[:, 1])
            # R = np.sum(self.T_last_.astype(np.float64)) / np.sum((self.T_finish_ - self.task_info[:, 1]))
            # for i in range(self.T_finish_.shape[0]):
            #     R[i] = self.expert_info[self.T_switch_[i, 1],
            #                             self.task_type_[i]] / \
            #            (self.T_finish_[i] - self.task_info[i, 1])
            sigma_L = np.std(L)
            M_bar = np.mean(M)
            R_bar = np.mean(R_)
            a, b, c = 99, 99, 1000
            # score = (c * R_bar) / ((a * M_bar) + (b * sigma_L))
            score = c * R - a * M_bar - b * sigma_L
            self.L_ = L
            self.M_ = M
            self.R_ = R
            self.sigma_L_ = sigma_L
            self.M_bar_ = M_bar
            self.R_bar_ = R_bar
            self.score_ = score
            return score
        else:
            return 10**(-9)

    def draw_expert_load(self, save_fig=False):
        if self.is_assignments_available_:
            plt.figure(figsize=(30, 10), dpi=100, facecolor='w', edgecolor='k')
            x = np.arange(0, self.n_experts_)
            plt.bar(x, self.L_, width=0.8, label='$\\sigma_L={}$'.format(self.sigma_L_))
            plt.xticks(x, [str(i + 1) for i in range(self.n_experts_)], fontsize=6)
            plt.grid()
            plt.title("The load of each expert")
            plt.xlim([-0.5, 133.5])
            plt.legend(fontsize=36)
            if save_fig:
                plt.savefig('build/expert_load.png', dpi=300)
            plt.show()
            plt.close()
        else:
            print("Wrong Assignment, and no expert load plot.")

    def draw_task_overtime(self, save_fig=False):
        if self.is_assignments_available_:
            plt.figure(figsize=(30, 10), dpi=100, facecolor='w', edgecolor='k')
            x = np.arange(self.n_tasks_)
            plt.scatter(x, self.M_, s=1, label='$\\bar{{M}}={}$'.format(self.M_bar_))
            plt.title("The overtime of each task")
            plt.ylim([-0.5, np.max(self.M_)+1])
            # plt.yticks(np.arange(np.ceil(np.max(self.M_))))
            plt.grid()
            plt.legend(fontsize=36)
            if save_fig:
                plt.savefig('build/task overtime.png', dpi=300)
            plt.show()
            plt.close()
        else:
            print("Wrong assignment, and no task overtime plot.")

    def draw_task_efficiency(self, save_fig=False):
        if self.is_assignments_available_:
            plt.figure(figsize=(30, 10), dpi=100, facecolor='w', edgecolor='k')
            x = np.arange(self.n_tasks_)
            plt.scatter(x, self.min_process_time_/self.task_total_time, s=1, label='$R={}$'.format(self.R_))
            plt.title("The efficency of each task")
            # plt.ylim([-0.5, np.max(self.M_)+1])
            # plt.yticks(np.arange(np.ceil(np.max(self.M_))))
            plt.grid()
            plt.legend(fontsize=36)
            if save_fig:
                plt.savefig('build/task efficiency.png', dpi=300)
            plt.show()
            plt.close()
        else:
            print("Wrong assignment, and no task efficiency plot.")

    def draw_task_overview(self, save_fig=False):
        if self.is_assignments_available_:
            plt.figure(figsize=(50, 20), dpi=100, facecolor='w', edgecolor='k')
            x = np.arange(self.n_tasks_)
            y_sla = self.task_info[:, 3]
            y_wait = self.T_start_ - self.task_info[:, 1]
            y_total = self.T_finish_ - self.task_info[:, 1]
            y_min_process = self.min_process_time_
            y_process = self.T_finish_ - self.T_start_
            plt.subplot(2, 1, 1)
            s = None
            plt.stem(x, y_wait, label='$t^{wait}$', markerfmt=' ', linefmt='b')
            plt.stem(x, y_sla, label='$t^{sla}$', markerfmt=' ', linefmt='g')
            plt.grid()
            plt.legend(fontsize=20)
            plt.subplot(2, 1, 2)
            plt.stem(x, y_total, label='$t^{total}$', markerfmt=' ', linefmt='b')
            plt.stem(x, y_process, label='$t^{process}$', markerfmt=' ', linefmt='g')
            plt.stem(x, y_min_process, label='$t^{min process}$', markerfmt=' ', linefmt='r')
            # plt.title("Overview of each task")
            plt.grid()
            plt.legend(fontsize=20)
            plt.suptitle('Overview of each task')
            plt.tight_layout()
            if save_fig:
                plt.savefig('build/task overview.png', dpi=300)
            plt.show()
            plt.close()
        else:
            print("Wrong assignment, and no task efficiency plot.")

    def print_performance(self):
        if self.is_assignments_available_:
            print("R:", self.R_)
            print("sigma_L:", self.sigma_L_)
            print("M_bar:", self.M_bar_)


if __name__ == "__main__":
    evaluator = Evaluator(filename='result_width=3.csv').fit()
    print('score:', evaluator.get_score())
    # evaluator.draw_expert_load(save_fig=True)
    # evaluator.draw_task_overtime(save_fig=True)
    # evaluator.draw_task_efficiency(save_fig=True)
    # evaluator.draw_task_overview(save_fig=True)
    evaluator.print_performance()
