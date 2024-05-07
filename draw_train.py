from tensorboard.backend.event_processing import event_accumulator  # 导入tensorboard的事件解析器
import utils


def get_train_data(log_file_path):
    ea = event_accumulator.EventAccumulator(log_file_path)  # 初始化EventAccumulator对象
    ea.Reload()  # 这一步是必须的，将事件的内容都导进去

    reward_events = ea.scalars.Items('reward')
    steps = []
    rewards = []

    for event in reward_events:
        steps.append(event.step*100)
        rewards.append(event.value)

    return steps, rewards


mfg_steps, mfg_rewards = get_train_data('train_log/events.out.tfevents.1714036861.nomad-virtual-machine')
low_lstm_steps, low_lstm_rewards = get_train_data('train_log/events.out.tfevents.1714055386.nomad-virtual-machine')

step_len = min(len(mfg_steps), len(low_lstm_steps))

utils.draw_list([mfg_rewards[:step_len], low_lstm_rewards[:step_len]], algos=['mfg', 'low_lstm_layer'],
                y_label='Reward',
                step_list=mfg_steps[:step_len],
                save_dir='plot_train/train_epoch_reward.png')
