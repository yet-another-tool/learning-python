from wtest import launch_test
from wutils import load_config
from os import path
import cv2
import fns as f
import numpy as np


def plot_number(inputs, h=27):
    line = ""
    for p in inputs:
        line += ".░▒▓█"[round(p * 4)]
        if len(line) > h:
            print(line)
            line = ""


# config_path = "config.txt"
config_path = "_data.bin"

if path.exists(config_path):
    weight_inputs_to_hiddens, weight_hiddens_to_outputs, biases_inputs_to_hiddens, biases_hiddens_to_outputs = load_config(
        config_path)
#     accuracy = launch_test(weight_inputs_to_hiddens, biases_inputs_to_hiddens,
#                            weight_hiddens_to_outputs, biases_hiddens_to_outputs)

#     print(f"Accuracy: {accuracy}")
n = 9
image = cv2.imread("./dataset/"+str(n)+".png", cv2.IMREAD_GRAYSCALE)
(h, w) = image.shape[:2]

# print(image)


plot_number([round(float(c/255), 2)
             for _ in cv2.flip(np.invert(image), 1) for c in _], h-1)


print("width: {} pixels".format(image.shape[1]))
print("height: {} pixels".format(image.shape[0]))

labels = []
targets = []
inputs = []
labels.append(n)
targets.append([int(0) if index != n else int(1)
                for index in range(10)])
inputs.append([round(float(c/255), 2)
               for _ in cv2.flip(np.invert(image), 1) for c in _])
#    for _ in np.invert(image) for c in _])


print("Testing Model")

# labels, targets, inputs = data.load_test_data()
print(f"Starting Test")
predictions_hiddens = [[sum([weight*activation for weight, activation in zip(weights, inp)]) +
                        bias for weights, bias in zip(weight_inputs_to_hiddens, biases_inputs_to_hiddens)] for inp in inputs]
activations_hiddens = [[max(0, prediction) for prediction in predictions]
                       for predictions in predictions_hiddens]

predictions_outputs = [[sum([weight*activation for weight, activation in zip(weights, activations)]) + bias for weights,
                        bias in zip(weight_hiddens_to_outputs, biases_hiddens_to_outputs)] for activations in activations_hiddens]
activations_outputs = [f.softmax(predictions)
                       for predictions in predictions_outputs]

correct = 0
for activation, target, inp in zip(activations_outputs, targets, inputs):
    _neuron = activation.index(max(activation))
    _target = target.index(max(target))
    if(_neuron == _target):
        correct += 1
    else:
        print(f"Digit: {_target}, guessed: {_neuron}")
        plot_number(inp)

print(f"Correct: {correct}/{len(inputs)}, ({correct/len(inputs):%})")

# wutils.print_memory_usage()
# return correct/len(inputs)
