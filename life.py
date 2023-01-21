from enum import Enum
import math
import random
import hashlib

random.seed(36912369123691236912369123691236912369123691236912)


class Word(str, Enum):
    T = "WHY"
    A = "WHY NOT"
    C = "HUM"
    G = "SO"


def speak(say="WHY"):
    return say


def listen(hear):
    # Initial State
    if(hear == "WHY"):
        return hear + " " + "WHY NOT"
    else:
        return hear + " " + "WHY WHY NOT"


def think(state: str):
    wait_for = random.random()
    return state + ((int(round(wait_for+1))*(" HUM")))


def demand(state):
    return state + " " + "SO"


def reverse(str):
    return str[::-1]


def encode(str):
    encoding = ""
    str_arr = str.split()
    for index, word in enumerate(str_arr):
        before = "_"
        after = "_"

        if(index > 0):
            before = str_arr[index-1]
        if(index < len(str_arr)-1):
            after = str_arr[index+1]

        if(word == "WHY" and (after != "NOT" or after == "WHY")):
            encoding += "T"
        elif(before == "WHY" and word == "NOT"):
            encoding += "A"
        elif(word == "HUM"):
            encoding += 'C'
        elif(word == "SO"):
            encoding += "G"

    encoding = list(encoding)
    offset = 0
    for index, _encoding in enumerate([encoding[i:i+3] for i in range(0, len(encoding), 3)]):
        if(encoding[offset+index-3:offset+index] == _encoding):
            # If last 3 elements are a repetition, swap the next sequence to avoid repetition
            _e = encoding[offset+index:offset+index+3]
            for idx, _ in enumerate(encoding[offset+index:offset+index+3]):
                encoding[offset+index+idx] = _e[2-idx]

        offset += 2
    return [''.join(encoding)[i:i+3] for i in range(0, len(''.join(encoding)), 3)]


life = 0
say = "WHY"
history = []
iter = 0
life = 0.00001
lost, stories = math.modf(1/life)
print(f"Lost: {int(lost*100)}, Stories: {stories}")

for _ in range(min((33, (int(stories) + int(lost*100))))):
    print(f"#: {iter}, lost: {lost}")
    stuck = random.random()
    if stuck > 0.5:
        history = demand(think(''.join([listen(speak(say))
                                        for _ in range(int(stuck*3))])))
    else:
        history = demand(think(listen(speak(say))))
    say = ''.join(history)
    iter += 1
encoded = encode(history)

print()
print("Input: ")
print(history)
print()
print("Encoded: ")
print(''.join(encoded))
print()

print(f"Hash: {hashlib.sha256(''.join(encoded).encode('ascii')).hexdigest()}")
