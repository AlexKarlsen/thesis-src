import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

def draw_average_overhead_per_img(cfg, overhead_list):
    x = np.arange(len(cfg.scenario))
    width = 0.3

    mean_overhead = np.mean(overhead_list, axis=2)
    std_overhead = np.std(overhead_list, axis=2)

    for idx, size in enumerate(cfg.chunk_size):
        xx = x - width/2 if idx == 0 else x + width/2
        plt.bar(xx, mean_overhead[idx,:], width=width, yerr=std_overhead[idx,:])

    plt.ylabel('Average TCP overhead per image (frames)')
    plt.xlabel('Transmission scenarios')
    plt.xticks(x, cfg.scenario)
    #plt.title('Average {} transmission latency'.format(protocol))

    plt.legend(['chunk size 1024 bytes', 'chunk size 4096 bytes'])
    plt.show()

def draw_histogram_overhead_per_img(cfg, overhead_list):
    fig, ax = plt.subplots()
    for idx_size, (size, ls) in enumerate(zip(cfg.chunk_size, cfg.size_ls)):
        overhead = overhead_list[idx_size, :, :]
        for idx, (sc, color) in enumerate(zip(cfg.scenario, cfg.sc_color)):
            sns.distplot(overhead[idx], bins=100, hist=False, kde_kws={'linestyle': ls, 'color': color},
                         label=sc+' '+str(size))

    ax.set_ylabel('Density', fontsize=12)
    ax.set_xlabel('TCP overhead (frames)', fontsize=12)
    ax.set_title('Density plot of TCP overhead'.format(cfg.protocol), fontsize=12)
    ax.legend()
    plt.show()

def draw_max_retransmission_per_img(cfg, retran):
    x = np.arange(len(cfg.scenario))
    width = 0.3

    max_lr = np.max(retran, axis=2)

    for idx, size in enumerate(cfg.chunk_size):
        xx = x - width/2 if idx == 0 else x + width/2
        plt.bar(xx, max_lr[idx,:], width=width)

    plt.ylabel('Maximal number of retransmission per image', fontsize=12)
    plt.xlabel('Transmission scenarios', fontsize=12)
    plt.xticks(x, cfg.scenario, fontsize=12)
    #plt.title('Maximal {} packet loss rate per image'.format(protocol))

    plt.legend(['chunk size 1024 bytes', 'chunk size 4096 bytes'])
    plt.show()

def draw_average_retransmission_per_img(cfg, retran):
    x = np.arange(len(cfg.scenario))
    width = 0.3

    mean_lr = np.mean(retran, axis=2)
    nonzero_lr = np.count_nonzero(retran, axis=2)

    fig, ax1 = plt.subplots()

    for idx, size in enumerate(cfg.chunk_size):
        xx = x - width*2/3 if idx == 0 else x + width*2/3
        ax1.bar(xx, mean_lr[idx,:], width=width, log=True)
    ax1.set_ylabel('Average number of retransmission per image', fontsize=12)
    ax1.set_xlabel('Transmission scenarios', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(cfg.scenario, fontsize=12)
    ax1.legend(['chunk size 1024 bytes', 'chunk size 4096 bytes'])

    ax2 = ax1.twinx()
    for idx, size in enumerate(cfg.chunk_size):
        ax2.plot(x, nonzero_lr[idx, :], '-o')
    ax2.set_ylabel('The number of images with retransmission', fontsize=12)

    #plt.title('Average {} packet loss rate per image'.format(protocol))
    #ax2.legend(['chunk size 1024 bytes', 'chunk size 4096 bytes'])

    plt.show()

def draw_inference_accuracy(cfg, accuracy, topk):
    x = np.arange(len(cfg.scenario))
    width = 0.3

    for idx, size in enumerate(cfg.chunk_size):
        xx = x - width/2 if idx == 0 else x + width/2
        plt.bar(xx, accuracy[idx], width=width)

    plt.ylabel('Top-{} inference accuracy'.format(topk), fontsize=12)
    plt.xlabel('Transmission scenarios', fontsize=12)
    plt.xticks(x, cfg.scenario, fontsize=12)
    plt.title('Top-{} inference accuracy of received images via {}'.format(topk, cfg.protocol))

    plt.legend(['chunk size 1024 bytes', 'chunk size 4096 bytes'])
    plt.show()

def draw_max_lossrate_per_img(cfg, loss_rate):
    x = np.arange(len(cfg.scenario))
    width = 0.3

    max_lr = np.max(loss_rate, axis=2)

    for idx, size in enumerate(cfg.chunk_size):
        xx = x - width/2 if idx == 0 else x + width/2
        plt.bar(xx, max_lr[idx,:], width=width)

    plt.ylabel('Maximal packet loss rate per image', fontsize=12)
    plt.xlabel('Transmission scenarios', fontsize=12)
    plt.xticks(x, cfg.scenario, fontsize=12)
    #plt.title('Maximal {} packet loss rate per image'.format(protocol))

    plt.legend(['chunk size 1024 bytes', 'chunk size 4096 bytes'])
    plt.show()

def draw_average_lossrate_per_img(cfg, loss_rate):
    x = np.arange(len(cfg.scenario))
    width = 0.3

    mean_lr = np.mean(loss_rate, axis=2)
    std_lr = np.std(loss_rate, axis=2)

    nonzero_lr = np.count_nonzero(loss_rate, axis=2)

    fig, ax1 = plt.subplots()

    for idx, size in enumerate(cfg.chunk_size):
        xx = x - width*2/3 if idx == 0 else x + width*2/3
        ax1.bar(xx, mean_lr[idx,:], width=width, log=True)
    ax1.set_ylabel('Average packet loss rate per image (%)', fontsize=12)
    ax1.set_xlabel('Transmission scenarios', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(cfg.scenario, fontsize=12)
    ax1.legend(['chunk size 1024 bytes', 'chunk size 4096 bytes'])

    ax2 = ax1.twinx()
    for idx, size in enumerate(cfg.chunk_size):
        ax2.plot(x, nonzero_lr[idx, :], '-o')
    ax2.set_ylabel('The number of images with packet loss', fontsize=12)

    #plt.title('Average {} packet loss rate per image'.format(protocol))
    #ax2.legend(['chunk size 1024 bytes', 'chunk size 4096 bytes'])

    plt.show()

def draw_average_delay_per_img(cfg, delay_list):
    x = np.arange(len(cfg.scenario))
    width = 0.3

    mean_delay = np.mean(delay_list, axis=2)
    std_delay = np.std(delay_list, axis=2)

    for idx, size in enumerate(cfg.chunk_size):
        xx = x - width/2 if idx == 0 else x + width/2
        plt.bar(xx, mean_delay[idx,:], width=width, yerr=std_delay[idx,:])

    plt.ylabel('Average image transmission latency (ms)')
    plt.ylim(0, 170)
    plt.xlabel('Transmission scenarios')
    plt.xticks(x, cfg.scenario)
    plt.title('Average {} transmission latency'.format(cfg.protocol))

    plt.legend(['chunk size 1024 bytes', 'chunk size 4096 bytes'])
    plt.show()

def draw_histogram_delay_per_img(cfg, delay_list):
    fig, ax = plt.subplots()
    for idx_size, (size, ls) in enumerate(zip(cfg.chunk_size, cfg.size_ls)):
        delay = delay_list[idx_size, :, :]
        for idx, (sc, color) in enumerate(zip(cfg.scenario, cfg.sc_color)):
            sns.distplot(delay[idx], bins=100, hist=False, kde_kws={'clip': (0, 80), 'linestyle': ls, 'color': color},
                         label=sc+' '+str(size))

    ax.set_ylabel('Density', fontsize=12)
    ax.set_xlabel('Image transmission latency (ms)', fontsize=12)
    ax.set_title('Density plot of {} transmission latency'.format(cfg.protocol), fontsize=12)
    ax.legend()
    plt.show()