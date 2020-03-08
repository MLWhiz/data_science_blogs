import pickle, os, argparse
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--GPU', nargs='+', default=['gpu_name1', 'gpu_name2', 'gpu_name3'])
parser.add_argument('--method', nargs='+', default=['FP32', 'FP16', 'amp'])
parser.add_argument('--batch', nargs='+', default=[128, 256, 512, 1024, 2048], type=int)
parser.add_argument('--plot_std', action='store_true')
args = parser.parse_args()

print(args.GPU)
print(args.method)
print(args.batch)
print(args.plot_std)
color = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
line = ['-', '--', ':', '-.']
data = 'CIFAR'
Time = []
Memory = []
Accuracy = []
for g in args.GPU:
    for m in args.method:
        for b in args.batch:
            with open(os.path.join('result', g, data + '_' + m + '_' + str(b) + '_100_result.pkl'), 'rb') as f:
                r = pickle.load(f)
                Time.append(r['train_time'])
                Memory.append(r['train_mem'])
                Accuracy.append(r['test_acc'])

l = len(args.batch)
color = color[:len(args.GPU)]
line = line[:len(args.method)]

for gn, g in enumerate(args.GPU):
    for mn, m in enumerate(args.method):
        if args.plot_std:
            plt.errorbar(args.batch, np.mean(Time[l*(gn*len(args.method)) + l*mn:l*(gn*len(args.method)) + l*(mn+1)], 1), fmt=color[gn] + line[mn], yerr=np.std(Time[l*(gn*len(args.method)) + l*mn:l*(gn*len(args.method)) + l*(mn+1)], 1), marker='.', capsize=5, label=g + '_' + m)
        else:
            plt.plot(args.batch, np.mean(Time[l * (gn * len(args.method)) + l * mn:l * (gn * len(args.method)) + l * (mn + 1)], 1), color[gn] + line[mn], marker='.', label=g + '_' + m)

plt.title(data + ' - Time')
plt.xlabel('Batch size')
plt.ylabel('Time')
plt.xticks(args.batch, args.batch)
plt.legend()
plt.grid(linestyle='-')
if args.plot_std:
    plt.savefig(data + ' - Time (std).png')
else:
    plt.savefig(data + ' - Time.png')
plt.close()

for gn, g in enumerate(args.GPU):
    for mn, m in enumerate(args.method):
        if args.plot_std:
            plt.errorbar(args.batch, np.mean(Memory[l*(gn*len(args.method)) + l*mn:l*(gn*len(args.method)) + l*(mn+1)], 1), fmt=color[gn] + line[mn], yerr=np.std(Memory[l*(gn*len(args.method)) + l*mn:l*(gn*len(args.method)) + l*(mn+1)], 1), marker='.', capsize=5, label=g + '_' + m)
        else:
            plt.plot(args.batch, np.mean(Memory[l * (gn * len(args.method)) + l * mn:l * (gn * len(args.method)) + l * (mn + 1)], 1), color[gn] + line[mn], marker='.', label=g + '_' + m)

plt.title(data + ' - Memory')
plt.xlabel('Batch size')
plt.ylabel('Time')
plt.xticks(args.batch, args.batch)
plt.legend()
plt.grid(linestyle='-')
if args.plot_std:
    plt.savefig(data + ' - Memory (std).png')
else:
    plt.savefig(data + ' - Memory.png')
plt.close()

for gn, g in enumerate(args.GPU):
    for mn, m in enumerate(args.method):
        if args.plot_std:
            plt.errorbar(args.batch, np.mean(Accuracy[l*(gn*len(args.method)) + l*mn:l*(gn*len(args.method)) + l*(mn+1)], 1), fmt=color[gn] + line[mn], yerr=np.std(Accuracy[l*(gn*len(args.method)) + l*mn:l*(gn*len(args.method)) + l*(mn+1)], 1), marker='.', capsize=5, label=g + '_' + m)
        else:
            plt.plot(args.batch, np.mean(Accuracy[l * (gn * len(args.method)) + l * mn:l * (gn * len(args.method)) + l * (mn + 1)], 1), color[gn] + line[mn], marker='.', label=g + '_' + m)

plt.title(data + ' - Accuracy')
plt.xlabel('Batch size')
plt.ylabel('Accuracy')
plt.xticks(args.batch, args.batch)
plt.legend()
plt.grid(linestyle='-')
if args.plot_std:
    plt.savefig(data + ' - Accuracy (std).png')
else:
    plt.savefig(data + ' - Accuracy.png')
plt.close()
