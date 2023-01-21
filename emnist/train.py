from learning_rate import dynamic_learning_rate
import data
from wtest import launch_test
import wutils
import random
import fns as f
import time
from datetime import datetime, timedelta
import signal
from os import path


def train(epochs, batch_size, learning_rate, shuffle, input_count, hidden_count, output_count, report_path, config_path):
    # Weights and Biases
    if path.exists(config_path):
        print("Load existing configurations")
        wutils.report(report_path,
                      f'# Started: {datetime.now()}, epochs: {epochs}, batch_size: {batch_size}, learning_rate: {learning_rate}, input_count: {input_count}, hidden_count: {hidden_count}, output_count: {output_count}, shuffle: {shuffle}',
                      )
        weight_inputs_to_hiddens, weight_hiddens_to_outputs, biases_inputs_to_hiddens, biases_hiddens_to_outputs = wutils.load_config(
            config_path)
    else:
        print("Initialize new configurations")
        wutils.save_to_file(report_path,
                            f'# Started: {datetime.now()}, epochs: {epochs}, batch_size: {batch_size}, learning_rate: {learning_rate}, input_count: {input_count}, hidden_count: {hidden_count}, output_count: {output_count}, shuffle: {shuffle}',
                            'timestamp,epoch_took,batch_took,epoch,cost,progress,progress_made,memory_usage,accuracy,training_took,learning_rate')
        weight_inputs_to_hiddens = [
            [random.random() - 0.5 for _ in range(input_count)] for _ in range(hidden_count)]
        weight_hiddens_to_outputs = [
            [random.random()-0.5 for _ in range(hidden_count)] for _ in range(output_count)]
        biases_inputs_to_hiddens = [0 for _ in range(hidden_count)]
        biases_hiddens_to_outputs = [0 for _ in range(output_count)]

        wutils.save_to_file(config_path, weight_inputs_to_hiddens, weight_hiddens_to_outputs,
                            biases_inputs_to_hiddens, biases_hiddens_to_outputs)

    # Handlers

    def signal_handler(signal, frame):
        print('You pressed Ctrl+C!')

        print("Saving current configurations")
        wutils.save_to_file(config_path, weight_inputs_to_hiddens, weight_hiddens_to_outputs,
                            biases_inputs_to_hiddens, biases_hiddens_to_outputs)
        wutils.report(report_path,
                      f'# Stopped: {datetime.now()}, epoch: {epoch}/{epochs}',
                      )
        exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    # Training
    last_cost = None
    training_start_time = time.monotonic()
    for epoch in range(epochs):
        start_time = time.monotonic()
        for labels, targets, inputs in data.load_training_data(batch_size, shuffle):
            wutils.print_memory_usage()
            batch_start_time = time.monotonic()
            predictions_hiddens = [[sum([weight*activation for weight, activation in zip(weights, inp)]) + bias for weights,
                                    bias in zip(weight_inputs_to_hiddens, biases_inputs_to_hiddens)] for inp in inputs]
            activations_hiddens = [[max(0, prediction) for prediction in predictions]
                                   for predictions in predictions_hiddens]
            predictions_outputs = [[sum([weight*activation for weight, activation in zip(weights, inp)]) + bias for weights,
                                    bias in zip(weight_hiddens_to_outputs, biases_hiddens_to_outputs)] for inp in activations_hiddens]
            activations_outputs = [f.softmax(predictions)
                                   for predictions in predictions_outputs]

            cost = sum([f.log_loss(activation, target) for activation, target in zip(
                activations_outputs, targets)]) / len(activations_outputs)
            if last_cost == None:
                last_cost = cost
            progress = wutils.check_progress(last_cost, cost)
            learning_rate = dynamic_learning_rate(
                learning_rate, progress, 0.05, 0.05)
            print(
                f"Epoch: {epoch} / {epochs}, Cost: {cost:.8f}, Change: {progress:.8f}")
            last_cost = cost

            # Back Propagation
            # Log Loss Error Derivative
            errors_derivative_outputs = [
                [a - t for a, t in zip(activation, target)] for activation, target in zip(activations_outputs, targets)]
            weight_hiddens_to_outputs_transpose = list(
                zip(*weight_hiddens_to_outputs))
            errors_derivative_hiddens = [[sum([delta*weight for delta, weight in zip(deltas, weights)]) * (0 if prediction <= 0 else 1) for weights, prediction in zip(
                weight_hiddens_to_outputs_transpose, predictions)] for deltas, predictions in zip(errors_derivative_outputs, predictions_hiddens)]

            # Gradient hidden -> output
            activations_hiddens_transpose = list(zip(*activations_hiddens))
            errors_derivative_outputs_transpose = list(
                zip(*errors_derivative_outputs))
            weight_hiddens_to_outputs_derivative = [[sum([delta*activation for delta, activation in zip(deltas, activations)])
                                                    for deltas in errors_derivative_outputs_transpose] for activations in activations_hiddens_transpose]
            biases_hiddens_to_outputs_derivative = [sum(
                [delta for delta in deltas]) for deltas in errors_derivative_outputs_transpose]

            # Gradient input -> hidden
            inputs_transpose = list(zip(*inputs))
            errors_derivative_hiddens_transpose = list(
                zip(*errors_derivative_hiddens))
            weight_inputs_to_hiddens_derivative = [[sum([delta * activation for delta, activation in zip(
                deltas, activations)]) for deltas in errors_derivative_hiddens_transpose] for activations in inputs_transpose]
            biases_inputs_to_hiddens_derivative = [sum(
                [delta for delta in deltas]) for deltas in errors_derivative_hiddens_transpose]

            # Update Weights and biases for all layers
            weight_hiddens_to_outputs_derivative_transpose = list(
                zip(*weight_hiddens_to_outputs_derivative))
            for y in range(output_count):
                for x in range(hidden_count):
                    weight_hiddens_to_outputs[y][x] -= learning_rate * \
                        weight_hiddens_to_outputs_derivative_transpose[y][x] / len(
                            inputs)
                biases_hiddens_to_outputs[y] -= learning_rate * \
                    biases_hiddens_to_outputs_derivative[y] / len(inputs)

            weight_inputs_to_hiddens_derivative_transpose = list(
                zip(*weight_inputs_to_hiddens_derivative))
            for y in range(hidden_count):
                for x in range(input_count):
                    weight_inputs_to_hiddens[y][x] -= learning_rate * \
                        weight_inputs_to_hiddens_derivative_transpose[y][x] / len(
                            inputs)
                biases_inputs_to_hiddens[y] -= learning_rate * \
                    biases_inputs_to_hiddens_derivative[y] / len(inputs)

            batch_end_time = time.monotonic()
            print(
                f"Batch Took: {timedelta(seconds=batch_end_time - batch_start_time)}")
            wutils.report(report_path, datetime.now(), None, timedelta(
                seconds=batch_end_time - batch_start_time), epoch, cost, progress, 1 if progress > 0 else 0, wutils.get_memory_usage(), None, None, learning_rate)

        end_time = time.monotonic()
        print(f"Epoch Took: {timedelta(seconds=end_time - start_time)}")
        print()
        wutils.save_to_file(config_path, weight_inputs_to_hiddens, weight_hiddens_to_outputs,
                            biases_inputs_to_hiddens, biases_hiddens_to_outputs)

        wutils.report(report_path, datetime.now(), timedelta(seconds=end_time - start_time), timedelta(
            seconds=batch_end_time - batch_start_time), epoch, cost, progress, 1 if progress > 0 else 0, wutils.get_memory_usage(), None, None, learning_rate)

        # Test each epoch to validate progression
        accuracy = launch_test(weight_inputs_to_hiddens, biases_inputs_to_hiddens,
                               weight_hiddens_to_outputs, biases_hiddens_to_outputs)
        wutils.report(report_path, datetime.now(), timedelta(seconds=end_time - start_time), timedelta(
            seconds=batch_end_time - batch_start_time), epoch, cost, progress, 1 if progress > 0 else 0, wutils.get_memory_usage(), accuracy, None, learning_rate)

    training_end_time = time.monotonic()
    print(
        f"Training Took: {timedelta(seconds=training_end_time - training_start_time)}")
    wutils.report(report_path, datetime.now(), None, None,
                  None, None, None, None, None, timedelta(seconds=training_end_time - training_start_time), learning_rate)

    # Final test after all epochs
    accuracy = launch_test(weight_inputs_to_hiddens, biases_inputs_to_hiddens,
                           weight_hiddens_to_outputs, biases_hiddens_to_outputs)
    wutils.report(report_path, datetime.now(), None, None, None,
                  None, None, None, wutils.get_memory_usage(), accuracy, None, learning_rate)
