import datetime
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import matplotlib.dates as mdates
from matplotlib import ticker

from agents.agent_aggregator_dqn import AggregatorAgent
from agents.agent_customer import CustomerAgent, construct_network
from environment.environment import Environment
from params import EPISODES, TIME_STEPS_TEST, TRAINING_START_DAY, NUM_RL_AGENTS, \
    AGENT_IDS, TIME_STEPS_TRAIN, TRAINING_PERIOD, TRAINING_END_DAY, AGENT_CLASS, CLASS_RHO, CLASS_DC, \
    DISSATISFACTION_COEFFICIENTS, REPLACE_TARGET_INTERVAL, TESTING_START_DAY, TESTING_END_DAY, BASELINE_START_DAY, \
    TESTING_PERIOD

env = Environment(AGENT_IDS, heterogeneous=False, baseline=False)
aggregator_agent = AggregatorAgent(env)
customer_agents = [CustomerAgent(agent_id, data_id, env, dummy=agent_id >= NUM_RL_AGENTS) for agent_id, data_id in enumerate(AGENT_IDS)]


def main():
    train(log=True, save=True)
    test_single_day(path=None, day=182)
    # test_single_day(path='save_files/MARL_IDR_2', day=182)
    # test_average(path='save_files/X2_a')
    # for day in range(181, 243):
    #     test_single_day(path='save_files/Case_1_b', day=day)


def train(log, save):
    # Use TensorBoard for the learning curves
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    start = time.time()
    # log_name = 'no_baseline_' + current_time
    log_name = 'MARL_IDR_6'
    if log:
        log_path = os.path.join('logs', log_name)
        tf_writer = tf.summary.create_file_writer(log_path)

    # Train with trained customer agents
    # for agent in customer_agents:
    #     agent.load('save_files/Case_2_b')

    # Train with trained aggregator
    # aggregator_agent.load('save_files/K8')

    aggregator_turn = True

    for episode in range(EPISODES):
        start_episode = time.time()

        # Reset environment and agents
        env.reset(max_steps=TIME_STEPS_TRAIN)
        aggregator_agent.reset()
        for agent in customer_agents:
            agent.reset()

        if episode % REPLACE_TARGET_INTERVAL == 0:
            aggregator_turn = False if aggregator_turn else True

        # Train single episode
        while not env.done:
            aggregator_agent.act(train=True)
            # aggregator_agent.act(train=True) if aggregator_turn else aggregator_agent.act(train=False)
            for agent in customer_agents:
                agent.act(train=True)
                # agent.act(train=False) if aggregator_turn else agent.act(train=True)
            env.step()

        if episode % 1 == 0:
            print('Episode:', episode)
            print('Aggregator turn:', aggregator_turn)
            print('Day:', datetime.datetime.strptime('{} {}'.format(env.day, 2018), '%j %Y'))
            print('Episode run time:', (time.time() - start_episode), 'sec')
            print('Cumulated run time:', (time.time() - start), 'sec')

        if log:
            with tf_writer.as_default():
                tf.summary.scalar("epsilon/agent_{}".format(customer_agents[0].data_id), customer_agents[0].epsilon, episode)
                tf.summary.scalar("reward/agent_{}".format(customer_agents[0].data_id), customer_agents[0].acc_reward, episode)
                tf.summary.scalar("reward/agent_{}".format(customer_agents[8].data_id), customer_agents[8].acc_reward, episode)
                tf.summary.scalar("reward/agent_{}".format(customer_agents[17].data_id), customer_agents[17].acc_reward, episode)
                tf.summary.scalar("reward/aggregator", aggregator_agent.acc_reward, episode)
                tf.summary.scalar("epsilon/aggregator", aggregator_agent.epsilon, episode)

    # Save trained networks
    if save:
        path = 'save_files/' + log_name
        os.mkdir(path)
        aggregator_agent.save(path)
        for agent in customer_agents:
            agent.save(path)

    print('Training done')
    print()


def test_single_day(path, day=TRAINING_START_DAY):
    env.reset(day=day, max_steps=TIME_STEPS_TEST)

    # Load agents
    if path is not None:
        aggregator_agent.load(path)
        for agent in customer_agents:
            agent.load(path)

    # Run single day
    start = time.time()
    for iteration in range(TIME_STEPS_TEST):
        aggregator_agent.act(train=False)
        for agent in customer_agents:
            agent.act(train=False)
        env.step()
    end = time.time()

    plot_agent = customer_agents[0]
    plot_hourly_average = False  # Average demands, consumptions and incentives per hour to make the plot more readable
    plot_incentive = True  # Plot the incentive
    time_labels = [datetime.datetime(year=2018, month=1, day=1) + datetime.timedelta(days=day - 1, minutes=i * 15) for i in range(TIME_STEPS_TEST)]

    print_metrics(end - start, path)
    for plot_agent in customer_agents:
        plot_aggregated_load_curve(time_labels, day, path)
        plot_single_load_curve(plot_agent, time_labels, day, path)
        plot_ac_reduction(plot_agent, time_labels, path, plot_incentive)
        plot_dissatisfaction(plot_agent, time_labels, path, plot_incentive)
        plot_schedule(plot_agent, time_labels, path)
        plt.show()


def test_average(path):
    agents_rewards = np.zeros((TESTING_PERIOD, NUM_RL_AGENTS))
    incentives_received = np.zeros((TESTING_PERIOD, NUM_RL_AGENTS))
    aggregator_rewards = np.zeros(TESTING_PERIOD)
    total_demands = np.zeros((TESTING_PERIOD, TIME_STEPS_TEST))
    total_consumptions = np.zeros((TESTING_PERIOD, TIME_STEPS_TEST))
    peak_demand = np.zeros(TESTING_PERIOD)
    peak_consumption = np.zeros(TESTING_PERIOD)
    mean_demand = np.zeros(TESTING_PERIOD)
    mean_consumption = np.zeros(TESTING_PERIOD)
    thresholds = np.zeros(TESTING_PERIOD)
    runtime = np.zeros(TESTING_PERIOD)

    for day in range(TESTING_START_DAY, TESTING_END_DAY):
        print('Test on', datetime.datetime.strptime('{} {}'.format(day, 2018), '%j %Y'))
        env.reset(day=day, max_steps=TIME_STEPS_TEST)

        # Load agents
        if path is not None:
            aggregator_agent.load(path)
            for agent in customer_agents:
                agent.load(path)

        # Run single day
        start = time.time()
        for iteration in range(TIME_STEPS_TEST):
            aggregator_agent.act(train=False)
            for agent in customer_agents:
                agent.act(train=False)
            env.step()
        end = time.time()

        agents_rewards[day - TESTING_START_DAY] = env.rewards_customers.sum(axis=0)[:NUM_RL_AGENTS]
        aggregator_rewards[day - TESTING_START_DAY] = env.rewards_aggregator.sum()
        incentives_received[day - TESTING_START_DAY] = env.incentive_received.sum(axis=0)[:NUM_RL_AGENTS]
        total_demands[day - TESTING_START_DAY] = env.get_total_demand()
        total_consumptions[day - TESTING_START_DAY] = env.get_total_consumption()
        peak_demand[day - TESTING_START_DAY] = np.max(env.get_total_demand())
        peak_consumption[day - TESTING_START_DAY] = np.max(env.get_total_consumption())
        mean_demand[day - TESTING_START_DAY] = np.mean(env.get_total_demand())
        mean_consumption[day - TESTING_START_DAY] = np.mean(env.get_total_consumption())
        thresholds[day - TESTING_START_DAY] = env.capacity_threshold
        runtime[day - TESTING_END_DAY] = end - start

    print('Path:', str(path))
    print('Test period', datetime.datetime.strptime('{} {}'.format(TESTING_START_DAY, 2018), '%j %Y'),
          datetime.datetime.strptime('{} {}'.format(TESTING_END_DAY, 2018), '%j %Y'))
    print('Mean run time:', np.mean(runtime))
    print('Mean customer reward:', np.mean(agents_rewards))
    print('Mean customer reward per agent:', np.mean(agents_rewards, axis=0))
    print('Mean customer reward per day:', np.mean(agents_rewards, axis=1))
    print('Mean aggregator reward per day:', np.mean(aggregator_rewards))
    print()
    print('Metrics averaged over testing period', '  No DR', '   with DR')
    print('Peak load:', np.mean(peak_demand), np.mean(peak_consumption))
    print('Mean load:', np.mean(mean_demand), np.mean(mean_consumption))
    print('PAR:', np.mean(peak_demand) / np.mean(mean_demand), np.mean(peak_consumption) / np.mean(mean_consumption))
    print('Mean incentive paid:', '0', np.mean(np.sum(incentives_received, axis=1)))
    print('Mean incentive received per agent:', '0', np.mean(incentives_received))
    print('Mean threshold exceedance:',
          np.mean(np.sum(np.maximum(0, total_demands - thresholds[:, None]), axis=1)),
          np.mean(np.sum(np.maximum(0, total_consumptions - thresholds[:, None]), axis=1)))


def print_metrics(run_time, path):
    agents_rewards = env.rewards_customers.sum(axis=0)[:NUM_RL_AGENTS]
    incentives_received = env.incentive_received.sum(axis=0)[:NUM_RL_AGENTS]
    test_day = datetime.datetime.strptime('{} {}'.format(env.day, 2018), '%j %Y')
    peak_demand = np.max(env.get_total_demand())
    peak_consumption = np.max(env.get_total_consumption())
    mean_demand = np.mean(env.get_total_demand())
    mean_consumption = np.mean(env.get_total_consumption())
    print('Path:', str(path))
    print('Test on', test_day)
    print('Run time:', run_time, 'sec')
    print('Customer agent rewards:', agents_rewards)
    print('Mean customer agent reward:', np.mean(agents_rewards))
    print('Aggregator agent reward:', env.rewards_aggregator.sum())
    print('Metrics', '  No DR', '   with DR')
    print('Peak load:', peak_demand, peak_consumption)
    print('Mean load:', mean_demand, mean_consumption)
    print('Std:', np.std(env.get_total_demand()), np.std(env.get_total_consumption()))
    print('PAR:', peak_demand / mean_demand, peak_consumption / mean_consumption)
    print('Total incentive paid:', '0', np.sum(incentives_received))
    print('Mean incentive received:', '0', np.mean(incentives_received))
    print('Threshold exceedance:',
          np.sum(np.maximum(0, env.get_total_demand() - env.capacity_threshold)),
          np.sum(np.maximum(0, env.get_total_consumption() - env.capacity_threshold)))

    print('Demand:', repr(env.get_total_demand()))
    print('Load curve:', repr(env.get_total_consumption()))
    print('Incentives:', repr(env.incentives))
    print('Capacity:', env.capacity_threshold)


def plot_aggregated_load_curve(time_labels, day, path):
    ax, ax2 = init_plot()
    # ax = init_plot(twinx=False)

    ax.hlines(env.capacity_threshold, time_labels[0], time_labels[-1], label='Capacity', colors='black', linestyles='dashed', linewidth=3, alpha=0.8)
    # ax.fill_between(time_labels, env.non_shiftable_load.sum(axis=1), label='Non-shiftable demand', color='orange')
    ax.plot(time_labels, env.get_total_demand(), label='Without DR', color='C1', linestyle='dashed', linewidth=3)
    # ax.plot(time_labels, env.baselines[:, day - TRAINING_START_DAY, :TIME_STEPS_TEST].sum(axis=0), label='Baseline', color='C3', linestyle='dashed', linewidth=3)
    ax.fill_between(time_labels, env.get_total_consumption(), label='With DR', color='C1', alpha=0.5)
    ax2.plot(time_labels, env.incentives, label='Incentive', color='C2', marker='x', markersize=8, linewidth=3, markeredgewidth=3)
    # ax.bar(time_labels, env.incentives, width=1/len(time_labels[48:]), label='Incentive', color='black')

    # Labels
    # ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.tick_params(axis='both', which='both', labelsize=18)
    ax2.tick_params(axis='both', which='both', labelsize=18)
    ax.set_xlim(time_labels[47], time_labels[-1])
    ax2.set_xlim(time_labels[47], time_labels[-1])
    ax.set_ylim(bottom=40)
    # ax.set_ylim(bottom=0, top=10)
    ax2.set_ylim(bottom=0, top=10)
    ax.set_ylabel('Aggregated consumption (kW)', fontsize=20)
    ax.set_ylabel('Incentive (¢)', fontsize=20)
    ax.legend(loc='upper left', fontsize=20)
    ax2.legend(loc='upper right', fontsize=20)
    ax.set_xlabel('Time (h)', fontsize=20)
    ax.grid(which='major', linewidth=1)
    ax.grid(which='minor', linewidth=0.5)
    plt.title('Load curve' + '\n' + str(path))
    # plt.show()


def plot_single_load_curve(plot_agent, time_labels, day, path):
    ax = init_plot(twinx=False)

    baseline = env.baselines[plot_agent.agent_id, day - BASELINE_START_DAY, :][:TIME_STEPS_TEST]

    # ax.fill_between(time_labels, env.non_shiftable_load[:, plot_agent.agent_id], label='Agent {} non-shiftable'.format(plot_agent.data_id), color='C0')
    ax.plot(time_labels, env.demand[:, plot_agent.agent_id], label='Without DR', color='C0', marker='o', markersize=8, linewidth=3)
    ax.plot(time_labels, env.consumptions[:, plot_agent.agent_id], label='MARL-DR', color='C3', marker='^', markersize=8, linewidth=3, alpha=1)
    ax.plot(time_labels, baseline, label='Baseline', color='black', linestyle='dashed', linewidth=3)
    ax.plot(time_labels, env.incentives, label='Incentive', color='C2', marker='x', markersize=8)
    # print('Incenties:', repr(env.incentives))

    # Labels
    ax.tick_params(axis='both', which='both', labelsize=20)
    ax.set_xlim(time_labels[48], time_labels[-1])
    # ax2.set_xlim(time_labels[47], time_labels[-1])
    ax.set_ylim(bottom=0)
    # ax2.set_ylim(bottom=0)
    ax.set_ylabel('Power (kW)', fontsize=32)
    # ax2.set_ylabel('Incentive (¢)')
    ax.legend(loc='upper left', fontsize=20)
    # ax2.legend(loc='upper right')
    ax.set_xlabel('Time (h)', fontsize=32)
    ax.grid(which='major', linewidth=1)
    # ax.grid(which='minor', linewidth=0.5)
    # plt.title('Load curve' + '\n' + str(path))
    # plt.show()


def plot_ac_reduction(plot_agent, time_labels, path, plot_incentive):
    ax, ax2 = init_plot()

    ac_demand = env.request_loads[:, plot_agent.agent_id, 0]
    ac_consumption = env.ac_rates[:, plot_agent.agent_id] * env.request_loads[:, plot_agent.agent_id, 0]

    ax.plot(time_labels, ac_demand, label='Agent {} AC original'.format(plot_agent.data_id), color='C0', alpha=0.5)
    ax.plot(time_labels, ac_consumption, label='Agent {} AC actual'.format(plot_agent.data_id), linestyle='dashed', color='C0', alpha=0.5)
    ax.fill_between(time_labels, env.ac_rates[:, plot_agent.agent_id] * env.request_loads[:, plot_agent.agent_id, 0], color='C0', alpha=0.2)
    ax2.plot(time_labels, env.ac_rates[:, plot_agent.agent_id] * 10, label='Agent {} AC rate'.format(plot_agent.data_id), color='C1', alpha=0.5)

    if plot_incentive:
        ax2.plot(time_labels, env.incentives, label='Incentive', color='C2', alpha=1.0)

    # Labels
    ax.tick_params(axis='x', which='both', rotation=45)
    ax.set_xlim(time_labels[0], time_labels[-1])
    ax2.set_xlim(time_labels[0], time_labels[-1])
    ax.set_ylim(bottom=0)
    ax2.set_ylim(bottom=0)
    ax.set_ylabel('Total demand (kW)')
    ax2.set_ylabel('Incentive (¢)')
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')
    ax.set_xlabel('Time')
    ax.grid(which='major', linewidth=1)
    ax.grid(which='minor', linewidth=0.5)
    plt.title('Load curve' + '\n' + str(path))
    # plt.show()


def plot_dissatisfaction(plot_agent, time_labels, path, plot_incentive):
    # ax, ax2 = init_plot()
    ax = init_plot(twinx=False)
    a, b, c, d, e = -env.dissatisfaction[:, plot_agent.agent_id, :].T
    rewards = np.roll(env.rewards_customers[:, plot_agent.agent_id], -1)
    rewards[-1] = 0

    ax.stackplot(time_labels, b, c, d, e, a, labels=['EV', 'WM', 'DW', 'Dryer', 'AC'])
    ax.plot(time_labels, env.incentive_received[:, plot_agent.agent_id], label='Profit', color='black', marker='s', markersize=8, linewidth=3)
    ax.plot(time_labels, env.rewards_customers[:, plot_agent.agent_id], label='Total reward', color='C2', marker='o', markersize=8, linewidth=3)

    # Labels
    ax.tick_params(axis='both', which='both', labelsize=20)
    # ax2.tick_params(axis='both', which='both', labelsize=14)
    ax.set_xlim(time_labels[48], time_labels[-1])
    # ax.set_ylim(bottom=0, top=15)
    ax.set_ylabel('Reward', fontsize=32)
    ax.legend(loc='upper left', fontsize=20)
    ax.set_xlabel('Time (h)', fontsize=32)
    ax.grid(which='major', linewidth=1)
    # ax.grid(which='minor', linewidth=0.5)
    # plt.title('Dissatisfaction' + '\n' + str(path))
    # plt.show()


def plot_schedule(plot_agent, time_labels, path):
    # ax, ax2 = init_plot()
    ax = init_plot(twinx=False)

    requests = env.requests_new[:, plot_agent.agent_id][:TIME_STEPS_TRAIN]
    actions = env.request_actions[:, plot_agent.agent_id][:TIME_STEPS_TRAIN]
    time_labels = time_labels[:TIME_STEPS_TRAIN]
    incentives = env.incentives[:TIME_STEPS_TRAIN]
    power_rates = env.power_rates[:, plot_agent.agent_id][:TIME_STEPS_TRAIN]
    ac_rates = env.ac_rates[:, plot_agent.agent_id][:TIME_STEPS_TRAIN]

    # Uncomment for hourly average incentives
    # incentives = np.mean(incentives.reshape(-1, 4), axis=1)

    for i, (time_label, req, act) in enumerate(zip(time_labels, requests, actions)):
        for j, (dev_name, dev_color, dev_req, dev_act) in enumerate(zip(['AC', 'EV', 'WM', 'DW', 'Dryer'], ['C4', 'C0', 'C1', 'C2', 'C3'], req, act)):
            height = 0.4
            request_bar = ax.barh(y=j + height, height=height, width=dev_req / 96, left=time_label, label=dev_name + ' original', color=dev_color, alpha=0.5, align='edge')
            if j == 0:
                height *= ac_rates[i]
            action_bar = ax.barh(y=dev_name, height=height, width=dev_act / 96, left=time_label, label=dev_name, color=dev_color, align='edge')

    # incentive_bar = ax.bar(time_labels, incentives, width=1 / 96, label='Incentive', align='edge', color='C1', alpha=0.4)
    # power_rate_bar = ax2.plot(time_labels, power_rates * 10, label='Power rates', color='C2')

    ax.tick_params(axis='both', which='both', labelsize=20)
    # ax2.tick_params(axis='both', which='both', labelsize=14)
    ax.set_xlim(time_labels[47], time_labels[-1])
    # ax2.set_xlim(time_labels[47], time_labels[-1])
    ax.set_ylim(bottom=0)
    # ax2.set_ylim(bottom=0)
    ax.grid(which='major', linewidth=1)
    # ax.grid(which='minor', linewidth=0.5)
    # ax.legend([request_bar, action_bar], ['Appliance requested', 'Appliance scheduled'], loc='upper left', fontsize=20)
    # ax2.legend([incentive_bar], ['Incentive'], loc='upper right', fontsize=16)
    # ax2.set_ylabel('Incentive (¢)', fontsize=16)
    ax.set_xlabel('Time (h)', fontsize=32)
    plt.setp(ax.get_yticklabels(), rotation=90, va="bottom", fontsize=20)
    # plt.title('Load schedule agent ' + str(plot_agent.data_id) + '\n' + str(path))
    # plt.show()


def init_plot(twinx=True):
    fig, ax = plt.subplots()

    day_locator = mdates.DayLocator()
    hour_locator = mdates.HourLocator(interval=1)
    minute_locator = mdates.MinuteLocator(interval=15)
    ax.xaxis.set_major_locator(hour_locator)
    ax.xaxis.set_minor_locator(minute_locator)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H'))
    # ax.xaxis.set_minor_formatter(mdates.DateFormatter('%H:%M'))

    if twinx:
        ax2 = ax.twinx()
        ax2.xaxis.set_major_locator(hour_locator)
        ax2.xaxis.set_minor_locator(minute_locator)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H'))
        # ax2.xaxis.set_minor_formatter(mdates.DateFormatter('%H:%M'))
        return ax, ax2

    return ax


main()
