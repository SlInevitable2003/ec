def inverse(a: int, p: int) -> int:
    u, v = a, p
    x1, x2 = 1, 0
    while u != 1:
        q = v // u
        r = v - q * u
        x = x2 - q * x1
        v = u
        u = r
        x2 = x1
        x1 = x
    return x1 % p

a  = int("41f21efba9e3e1461f297ccd58bad7abdc515cff944a58ec456723c698694873", 16)
b  = int("c233d79fc4c9079a0f76255af92e7263231be9e8cde7438d007c62c2085427f8", 16)
p  = int("ffffffff00000001000000000000000000000000ffffffffffffffffffffffff", 16)
ai = int("438c21aa9e6900d50f4fcd4de29ecbef29dedd783c9a4ca99d9e0c665e063377", 16)
pp = int("ffffffff00000002000000000000000000000001000000000000000000000001", 16)

print(hex(2 ** 256 - inverse(p, 2 ** 256)))
print(((2 ** 256 - pp) * p ) % (2 ** 256))