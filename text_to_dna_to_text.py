# -*- coding: UTF-8 -*-

def ascii2Dec(letter):
    return ord(letter)


def numberToBase(n, b):
    if n == 0:
        return [0]*b
    digits = [0]*b
    i = 0
    while n:
        digits[i] = int(n % b)
        n //= b
        i += 1
    return digits[::-1]


def base4ToDNA(msg):
    dna = ""
    for b in [*msg]:
        if(b == '0'):
            dna += 'T'
        if(b == '1'):
            dna += 'A'
        if(b == '2'):
            dna += 'C'
        if(b == '3'):
            dna += 'G'
    return dna


def dna2ascii(dna):
    text = ""
    num = []
    for b in [*dna]:
        if(b == 'T'):
            num.append(0)
        if(b == 'A'):
            num.append(1)
        if(b == 'C'):
            num.append(2)
        if(b == 'G'):
            num.append(3)

    for chunk in [num[i:i+4] for i in range(0, len(num), 4)]:
        s = sum([(l*(4**(len(chunk)-index-1)))
                 for index, l in enumerate(chunk)])
        text += chr(s)
    return text


msg = "Is this thing encrypted, or at least encoded properly ? Will he be able to break it :scream:"
base = 4

dna = base4ToDNA(''.join(
    [''.join(str(s) for s in numberToBase(ascii2Dec(l), base)) for l in msg]))

print(dna)

print(dna2ascii(dna))
