import data
import wutils
import fns as f


def launch_test(weight_inputs_to_hiddens, biases_inputs_to_hiddens, weight_hiddens_to_outputs, biases_hiddens_to_outputs):
    print("Testing Model")

    labels, targets, inputs = data.load_test_data()
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
            # print(f"Digit: {_target}, guessed: {_neuron}")
            wutils.save_to_file("test_results.txt",
                                f"Digit: {_target}, guessed: {_neuron}", wutils.plot_number(inp))

    print(f"Correct: {correct}/{len(inputs)}, ({correct/len(inputs):%})")

    wutils.print_memory_usage()
    return correct/len(inputs)
