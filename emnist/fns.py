import math


def softmax(predictions):
    m = max(predictions)
    temp = [math.exp(p-m) for p in predictions]
    total = sum(temp)
    return [t/total for t in temp]


def log_loss(activations, targets):
    losses = [-t * math.log(a) - (1-t) * math.log(1-a)
              for a, t, in zip(activations, targets)]
    return sum(losses)
