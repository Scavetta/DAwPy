# emirp numbers

def is_prime(x):
    if x <= 1:
        return False
    for i in range(2,x):
        if x % i == 0:
            return False
    return True

is_prime(23) 

def is_emirp(x):
    result = is_prime(int(str(x)[::-1])) & is_prime(x)
    print(f"{x} is {'' if result else 'not'} an emrip number")

is_emirp(13) # True
is_emirp(23) # False
is_emirp(32) # False
is_prime(18)
is_emirp(18)

vals = [6, 8, 4, 2, 5, 6, 7, 3, 5]
vals
vals + 10
import numpy as np
vals = np.array([6, 8, 4, 2, 5, 6, 7, 3, 5])
vals
vals + 10

pairs_1 = []

for num1 in range(0,2):
    for num2 in range(6,8):
        pairs_1.append((num1, num2))
        
pairs_1

[num + 10 for num in range(10)]