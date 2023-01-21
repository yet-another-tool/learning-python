x = 256
y = 0

print(f"Inputs: x:{x}, y:{y}")
print("\/\/\/\/\/\/\/\/\/")

a = list(bin(x)[2:].zfill(32))
b = list(bin(y)[2:].zfill(32))

print(f"A: {''.join(a)} => {int(''.join(a), 2)}")
print(f"B: {''.join(b)} => {int(''.join(b), 2)}")

for i in range(len(a)):
    a.append(b[0])
    b.append(a[0])
    a.pop(0)
    b.pop(0)

print("Toggle..")
print(f"A: {''.join(a)} => {int(''.join(a), 2)}")
print(f"B: {''.join(b)} => {int(''.join(b), 2)}")

print("Toggle it back..")

c = a
a = b
b = c

print(f"A: {''.join(a)} => {int(''.join(a), 2)}")
print(f"B: {''.join(b)} => {int(''.join(b), 2)}")
