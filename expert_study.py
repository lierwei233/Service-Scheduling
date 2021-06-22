import matplotlib.pyplot as plt
import numpy as np
from main import load_data


def get_expert_process_time(filename, expert_id):
    expert_id -= 1
    assignments = np.genfromtxt(filename, delimiter=',', dtype='int64',
                                encoding='utf-8')
    assignments[0, 0] = 1
    assignments[:, 0] -= 1
    assignments[:, 1] -= 1
    task_info, expert_info = load_data()

    idx = np.where(assignments[:, 1] == expert_id)[0]
    tasks = assignments[idx, 0]
    task_types = task_info[tasks, 2]
    times = expert_info[expert_id, task_types]
    print(tasks)
    print(task_types)
    print(times)
    return times


def draw_expert_graph():
    expert_info = np.genfromtxt('process_time_matrix.csv', delimiter=',',
                                encoding='utf-8', dtype='int64')
    expert_info = np.delete(expert_info, [0], axis=0)
    expert_info = np.delete(expert_info, [0], axis=1)
    expert_info = expert_info.T
    plt.figure(1)
    plt.imshow(expert_info, cmap='gray')
    plt.colorbar()
    plt.xlabel('Expert')
    plt.ylabel('Task')
    # plt.xticks(np.arange(107), [])
    # plt.yticks(np.arange(133), [])
    plt.grid(b=True, which='both')
    plt.tight_layout()
    plt.savefig('build/heatmap of each expert.png', dpi=300)
    plt.show()

    plt.figure(figsize=(30, 10), dpi=100, facecolor='w', edgecolor='k')
    x = np.arange(0, 133) + 1
    num_skills = np.sum(expert_info < 999999, axis=0)
    plt.bar(x, num_skills, width=0.8)
    plt.xticks(x, [str(i + 1) for i in range(133)], fontsize=6)
    plt.grid(b=True, axis='y')
    plt.title("The number of skills of each expert")
    plt.xlim([-0.5, 133.5])
    plt.yticks(np.arange(0, np.max(num_skills) + 1, 1))
    plt.tight_layout()
    plt.savefig('build/num_skills of each expert.png', dpi=300)
    plt.show()


def draw_cluster_graph(experts, classes):
    n_experts = experts.shape[0]
    experts_idx = np.array([], dtype=np.int32)  # 用来存储在画出来的图里面横轴的坐标。
    classes_set = list(set(classes))
    classified_experts = {}
    for i in classes_set:
        classified_experts[i] = [[], []]  # 第一个保存专家，第二个保存原始index
    for i in range(n_experts):
        classified_experts[classes[i]][0].append(experts[i])
        classified_experts[classes[i]][1].append(i+1)

    plt.figure(figsize=(18, 12), dpi=100, facecolor='w', edgecolor='k')
    start_tick = 1
    for i in classes_set:
        y = np.array([], dtype=np.int32)
        x = np.array([], dtype=np.int32)
        for row in classified_experts[i][0]:
            idx = np.where(row == 1)[0] + 1  # 得到所有为1的点的坐标（从1开始）
            y = np.concatenate((y, idx))
            x = np.concatenate((x, start_tick * np.ones(len(idx))))
            start_tick += 1
        plt.scatter(x, y, marker='o', label='class '+str(i), linewidth=0.8)
        experts_idx = np.concatenate((experts_idx, classified_experts[i][1]))
    plt.legend()
    plt.xlim([-2, 136])
    plt.xticks(np.arange(133)+1, experts_idx, fontsize=3)
    plt.yticks(np.arange(107)+1, fontsize=3)
    plt.grid(b=True, axis='both')
    plt.title('Cluster Result')
    plt.xlabel('Experts')
    plt.ylabel('Tasks')
    plt.savefig('build/cluster_result.png', dpi=300)


def dbscan(eps, min_experts):
    expert_info = np.genfromtxt('process_time_matrix.csv', delimiter=',',
                                encoding='utf-8', dtype='int32')
    expert_info = np.delete(expert_info, [0], axis=0)
    expert_info = np.delete(expert_info, [0], axis=1)
    expert_info = (expert_info < 999999).astype(np.int32)

    class Expert:
        def __init__(self, data, marker):
            """
            :param data: 原始数据
            :param marker: -2：noise，-1：not visited，0：visited，1~N：cluster name
            """
            self.data = data
            self.marker = marker
    experts = []
    for expert_ in expert_info:
        expert = Expert(expert_, -1)
        experts.append(expert)

    def region_query(expert, eps):
        neighborhood_points = []
        for expert_ in experts:
            if np.sum(np.logical_and(expert.data, expert_.data).astype(
                    np.int32)) >= eps:
                neighborhood_points.append(expert_)
        return neighborhood_points

    def expand_cluster(expert, neighborhood_points, C, eps, min_experts):
        expert.marker = C
        for expert_ in neighborhood_points:
            if expert_.marker == -1:
                expert_.marker = 0
                neighborhood_points_ = region_query(expert_, eps)
                if len(neighborhood_points_) >= min_experts:
                    neighborhood_points = neighborhood_points + list(set(
                        neighborhood_points + neighborhood_points_) - set(
                        neighborhood_points))
            if expert_.marker == 0:
                expert_.marker = C

    C = 0
    for i, expert in enumerate(experts):
        if expert.marker != -1:
            continue
        expert.marker = 0
        neighbor_experts = region_query(expert, eps)
        if len(neighbor_experts) < min_experts:
            expert.marker = -2
        else:
            C += 1
            expand_cluster(expert, neighbor_experts, C, eps, min_experts)

    experts_data = []
    experts_class = []
    for expert in experts:
        experts_data.append(expert.data)
        experts_class.append(expert.marker)
    experts_data = np.array(experts_data)
    experts_class = np.array(experts_class)
    return experts_data, experts_class


if __name__ == "__main__":
    print("Main function of expert_study.py")
    experts, classes = dbscan(eps=25, min_experts=10)
    draw_cluster_graph(experts, classes)
