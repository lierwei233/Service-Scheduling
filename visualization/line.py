import pyecharts.options as opts
from pyecharts.charts import Line, Timeline
from pyecharts.options import LabelOpts
import pickle
import numpy as np
import matplotlib.pyplot as plt

expert_info = np.genfromtxt('../tcdata/sic_semi_process_time_matrix.csv', delimiter=',', encoding='utf-8', dtype='int64')
expert_info = np.delete(expert_info, [0], axis=0)
expert_info = np.delete(expert_info, [0], axis=1)

task_info = np.genfromtxt('../tcdata/sic_semi_work_order.csv', encoding='utf-8', dtype=np.int, delimiter=',')
task_info[0, 0] = 1
task_info[:, [0, 2]] -= 1  # 从0开始

with open('data_280.pkl', 'rb') as file:
    new_arrival_seq_3, waiting_seq_3, processing_seq_3, expert_load_seq_3, expert_work_time_seq_3 = pickle.load(file)
with open('data_272.pkl', 'rb') as file:
    new_arrival_seq_1, waiting_seq_1, processing_seq_1, expert_load_seq_1, expert_work_time_seq_1 = pickle.load(file)


time_1 = np.array(list(new_arrival_seq_1.keys()))
n_new_arrival_1 = np.array([len(new_arrival_seq_1[key]) for key in time_1])
n_waiting_1 = np.array([len(waiting_seq_1[key]) for key in time_1])
n_processing_1 = np.array([len(processing_seq_1[key]) for key in time_1])

time_3 = np.array(list(new_arrival_seq_3.keys()))
n_new_arrival_3 = np.array([len(new_arrival_seq_3[key]) for key in time_3])
n_waiting_3 = np.array([len(waiting_seq_3[key]) for key in time_3])
n_processing_3 = np.array([len(processing_seq_3[key]) for key in time_3])

tl = Timeline(init_opts=opts.InitOpts(width="1600px", height="800px"))
tl.add_schema(play_interval=100, is_loop_play=False, is_timeline_show=False, is_auto_play=True)
this_time = []
this_new_arrival = []
this_waiting = []
this_processing_1 = []
this_processing_3 = []
time_start = 600
time_end = 800
t_start = time_start - 481
t_end = time_end - 481
for i in range(t_end - t_start):
    # this_time.append([j for j in range(i+1)])
    # this_new_arrival.append([0 for j in range(i+1)])
    # this_waiting.append([1 for j in range(i+1)])
    # this_processing.append([-1 for j in range(i+1)])
    this_time.append(list(range(i+1)))
    # this_new_arrival.append(n_new_arrival[t_start:t_start+i+1].tolist())
    # this_waiting.append(n_waiting[t_start:t_start+i+1].tolist())
    this_processing_1.append(n_processing_1[t_start:t_start + i + 1].tolist())
    this_processing_3.append(n_processing_3[t_start:t_start + i + 1].tolist())
for i in range(t_end - t_start):
    line = (
        Line()
        .add_xaxis(xaxis_data=this_time[i])
        # .add_yaxis(
        #     series_name="new arrival",
        #     y_axis=this_new_arrival[i],
        # )
        .add_yaxis(
            series_name="有预测",
            y_axis=this_processing_3[i],
        )
        .add_yaxis(
            series_name='无预测',
            y_axis=this_processing_1[i],
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(title="任务数量变化"),
            # tooltip_opts=opts.TooltipOpts(trigger="axis"),
            # toolbox_opts=opts.ToolboxOpts(is_show=True),
            # xaxis_opts=opts.AxisOpts(type_="category", boundary_gap=False),
        )
        .set_series_opts(LabelOpts(is_show=False))
    )
    tl.add(line, 't={}'.format(i))
# tl.render("num_task_change.html")

plt.figure(figsize=(30, 20), dpi=100, facecolor='w', edgecolor='k')
plt.plot(np.arange(time_start, time_end + 1), n_processing_1[t_start:t_end + 1], marker='o', label='no prediction')
plt.plot(np.arange(time_start, time_end + 1), n_processing_3[t_start:t_end + 1], marker='o', label='with prediction')
plt.yticks(fontsize=36)
plt.xticks(fontsize=36)
plt.grid()
plt.legend(fontsize=36)
plt.savefig('processing_comparison.png', dpi=300)
plt.show()

x_1 = time_1
x_3 = time_3
y_1 = np.zeros(len(time_1), dtype=np.int)
y_3 = np.zeros(len(time_3), dtype=np.int)
for i in range(len(time_1)-1):
    t = time_1[i]
    y_1[i] = np.sum((expert_load_seq_1[t + 1] - expert_load_seq_1[t])[:, 5])
for i in range(len(time_3) - 1):
    t = time_3[i]
    y_3[i] = np.sum((expert_load_seq_3[t + 1] - expert_load_seq_3[t])[:, 5])
plt.figure(figsize=(30, 20), dpi=100, facecolor='w', edgecolor='k')
plt.plot(x_1, y_1, marker='o', label='no prediction')
plt.plot(x_3, y_3, marker='o', label='with prediction')
plt.yticks(fontsize=36)
plt.xticks(fontsize=36)
plt.grid()
plt.legend(fontsize=36)
plt.savefig('la_comparison.png', dpi=300)
plt.show()

x_1 = time_1
x_3 = time_3
# zuihuigan = np.argmin(expert_info, axis=1)
zuixiaoshijian = np.min(expert_info, axis=0)
zuihuigan = [np.where(expert_info[i] == zuixiaoshijian)[0] for i in range(len(expert_info))]
y_1 = np.zeros(len(time_1), dtype=np.int)
y_3 = np.zeros(len(time_3), dtype=np.int)
for i in range(len(time_1)):
    t = time_1[i]
    expert_load_seq_1_copy_t = expert_load_seq_1[t].copy()
    expert_load_seq_1_copy_t[expert_load_seq_1[t] == -1] = 0
    expert_load_seq_1_copy_t[:, 0] = task_info[expert_load_seq_1_copy_t[:, 0], 2]
    expert_load_seq_1_copy_t[:, 1] = task_info[expert_load_seq_1_copy_t[:, 1], 2]
    expert_load_seq_1_copy_t[:, 2] = task_info[expert_load_seq_1_copy_t[:, 2], 2]
    expert_load_seq_1_copy_t[expert_load_seq_1[t] == -1] = -1
    for j in range(133):
        y_1[i] += np.sum(np.isin(expert_load_seq_1_copy_t[j, [0, 1, 2]], zuihuigan[j]))
for i in range(len(time_3)):
    t = time_3[i]
    expert_load_seq_3_copy_t = expert_load_seq_3[t].copy()
    expert_load_seq_3_copy_t[expert_load_seq_3[t] == -1] = 0
    expert_load_seq_3_copy_t[:, 0] = task_info[expert_load_seq_3_copy_t[:, 0], 2]
    expert_load_seq_3_copy_t[:, 1] = task_info[expert_load_seq_3_copy_t[:, 1], 2]
    expert_load_seq_3_copy_t[:, 2] = task_info[expert_load_seq_3_copy_t[:, 2], 2]
    expert_load_seq_3_copy_t[expert_load_seq_3[t] == -1] = -1
    for j in range(133):
        y_3[i] += np.sum(np.isin(expert_load_seq_3_copy_t[j, [0, 1, 2]], zuihuigan[j]))
plt.figure(figsize=(30, 20), dpi=100, facecolor='w', edgecolor='k')
plt.plot(x_1, y_1, marker='o', label='no prediction')
plt.plot(x_3, y_3, marker='o', label='with prediction')
plt.yticks(fontsize=36)
plt.xticks(fontsize=36)
plt.grid()
plt.legend(fontsize=36)
plt.savefig('zuihuigan_comparison.png', dpi=300)
plt.show()
