import os
import psutil
from ast import literal_eval


def print_memory_usage():
    process = psutil.Process(os.getpid())
    print(f"{process.memory_info().rss / 1024 ** 2:.2f}MiB")


def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 ** 2


def plot_number(inputs):
    line = ""
    lines = []
    for p in inputs:
        line += ".░▒▓█"[round(p * 4)]
        if len(line) > 27:
            lines.append(line)
            line = ""
    return lines


def save_to_file(filename, *data):
    f = open(filename, "w")
    for line in data:
        f.write(str(line)+'\n')
    f.close()


def report(filename="report.csv", *data):
    f = open(filename, "a")
    for line in data:
        f.write(str(line)+',')
    f.write('\n')
    f.close()


def check_progress(initial_value, final_value):
    return float(initial_value-final_value)


def load_config(filename):
    with open(filename) as file:
        text = file.read()
    configs = text.strip().split("\n")

    weight_inputs_to_hiddens = literal_eval(configs[0])
    weight_hiddens_to_outputs = literal_eval(configs[1])
    biases_inputs_to_hiddens = literal_eval(configs[2])
    biases_hiddens_to_outputs = literal_eval(configs[3])

    return weight_inputs_to_hiddens, weight_hiddens_to_outputs, biases_inputs_to_hiddens, biases_hiddens_to_outputs
