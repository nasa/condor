"""
Creating Model Templates
=========================
"""

import functools

import numpy as np

import condor as co

rng = np.random.default_rng(123)


class instancemethod:  # noqa: N801
    def __init__(self, func):
        print("creating wrapper with func", func, "on", self)
        self.func = func

    def __get__(self, obj, cls):
        print("returning self", self, " with", self.func, obj, cls)
        if obj is None:
            return self
        else:
            return functools.partial(self, obj)

    def __call__(self, *args, **kwargs):
        print("calling self", self, " with", self.func, *args, **kwargs)
        return self.func(*args, **kwargs)


class Class:
    def __init__(self, x):
        self.x = x

    @instancemethod
    def test(self, y):
        return self.x + y


cls = Class(2.0)
print(cls.test(3.0))


class ComponentRaw(co.models.ModelTemplate):
    """Raw Component base"""

    input = co.FreeField(co.Direction.input)
    output = co.AssignedField(co.Direction.output)

    x = placeholder(default=2.0)
    y = placeholder(default=1.0)

    output.z = x**2 + y

    def hello(self):
        print("world", self.z, self.x, self.y, self.input, self.output)


class ComponentImplementation(co.implementations.ExplicitSystem):
    pass


co.implementations.ComponentRaw = ComponentImplementation


class ComponentAT(co.ExplicitSystem, as_template=True):
    """AT component base"""

    x = placeholder(default=2.0)
    y = placeholder(default=1.0)

    output.z = x**2 + y

    def hello(self):
        print("world", self.x, self.y, self.z)


class MyComponentR(ComponentRaw):
    """my component R"""

    u = input()
    output.w = z + u

    def hello2(self):
        print("world", self.z)


class MyComponentA(ComponentAT):
    """my component A"""

    u = input()
    output.w = z + u


assert MyComponentR(u=1.23).z == MyComponentA(u=1.23).z  # noqa

# comp = MyComponentA(u=1., z=5.)


class MyComponent1(ComponentRaw):
    pass


comp1 = MyComponent1()


class MyComponent2(ComponentAT):
    u = input()
    # output.xx = z+u
    output.x = u + 2.0
    # output.x = z+u # this should produce an error because it's overwriting x but didnt


comp2 = MyComponent2(u=1.0)


class MatSys(co.ExplicitSystem):
    A = input(shape=(3, 4))
    B = input(shape=(4, 2))
    output.C = A @ B


ms = MatSys(rng.random(size=(3, 4)), rng.random(size=(4, 2)))


class SymMatSys(co.ExplicitSystem):
    A = input(shape=(3, 3), symmetric=True)
    B = input(shape=(3, 3))
    output.C = A @ B + B.T @ A


a = rng.random(size=(3, 3))
sms = SymMatSys(a + a.T, rng.random(size=(3, 3)))


class Sys(co.ExplicitSystem):
    x = input()
    y = input()
    v = y**2
    output.w = x**2 + y**2
    output.z = x**2 + y


sys = Sys(1.2, 3.4)
print(sys, sys.output)


class Opt(co.OptimizationProblem):
    x = variable()
    y = variable()

    sys = Sys(x=x, y=y)

    objective = (sys.w - 1) ** 2 - sys.z


Opt.set_initial(x=3.0, y=4.0)
opt = Opt()
