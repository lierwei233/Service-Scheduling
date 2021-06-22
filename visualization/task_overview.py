import pickle
import numpy as np

with open('data_280.pkl', 'rb') as file:
    new_arrival_seq, waiting_seq, processing_seq, expert_load_seq, expert_work_time_seq = pickle.load(file)

from pyecharts import options as opts
from pyecharts.charts import Bar, Timeline

task_info = np.genfromtxt('../tcdata/sic_semi_work_order.csv', encoding='utf-8', dtype=np.int, delimiter=',')
task_info[0, 0] = 1
task_info[:, [0, 2]] -= 1  # 从0开始

x = ['{}'.format(i) for i in range(1, 31)]
tl = Timeline()
tl.add_schema(play_interval=100, is_loop_play=False, is_timeline_show=False, is_auto_play=True)
time_start = 900
time_end = 1100
t_start = time_start - 481
t_end = time_end - 481
for t in range(time_start, time_end+1):
    new_arrival_type = task_info[new_arrival_seq[t], 2]
    waiting_type = task_info[waiting_seq[t], 2]
    processing_type = task_info[processing_seq[t], 2]
    new_arrival_count = np.zeros(30, dtype=np.int)
    new_arrival_type = new_arrival_type[new_arrival_type < 30]
    unique_new_arrival, unique_new_arrival_count = np.unique(new_arrival_type, return_counts=True)
    new_arrival_count[unique_new_arrival] = unique_new_arrival_count
    waiting_count = np.zeros(30, dtype=np.int)
    waiting_type = waiting_type[waiting_type < 30]
    unique_waiting, unique_waiting_count = np.unique(waiting_type, return_counts=True)
    waiting_count[unique_waiting] = unique_waiting_count
    processing_count = np.zeros(30, dtype=np.int)
    processing_type = processing_type[processing_type < 30]
    unique_processing, unique_processing_count = np.unique(processing_type, return_counts=True)
    processing_count[unique_processing] = unique_processing_count
    bar = (
        Bar()
        .add_xaxis(x)
        .add_yaxis("new_arrival", new_arrival_count.tolist(), stack="stack1")
        .add_yaxis("waiting", waiting_count.tolist(), stack="stack1")
        .add_yaxis("processing", processing_count.tolist(), stack='stack1')
        .set_global_opts(
            title_opts=opts.TitleOpts("Arrived Tasks".format(t)),
        )
    )
    tl.add(bar, "{} min".format(t))

tl.render("task_type.html")
