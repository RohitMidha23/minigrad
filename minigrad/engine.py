import math
from utils import topological_sort


class Value:
    def __init__(self, data, children=(), _op="") -> None:
        self.data = data
        self._prev = set(children)
        self._op = _op
        self.grad = 0
        self._backward = lambda: None

    def __repr__(self) -> str:
        return f"Value: (data: {self.data})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        output = self.data + other.data

        def _backward():
            self.grad += 1 * output.grad
            other.grad += 1 * output.grad

        output._backward = _backward
        return Value(output, children=(self, other), _op="*")

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        output = self.data * other.data

        def _backward():
            self.grad += other.data * output.grad
            other.grad += self.data * output.grad

        output._backward = _backward
        return Value(output, children=(self, other), _op="*")

    def __rmul__(self, other):  # called if python can't do other * self (eg. 2 * a)
        return self * other

    def tanh(self):
        n = self.data
        t = (math.exp(2 * n) - 1) / (math.exp(2 * n) + 1)
        output = Value(t, (self,), _op="tan")

        def _backward():
            self.grad += (1 - (t**2)) * output.grad

        output._backward = _backward
        return output

    def __truediv__(self, other):  # self / other
        return self * (other ** (-1))

    def __pow__(self, k):
        assert isinstance(k, (int, float)), "only int and float supported"
        output = Value(self.data**k, (self,), f"**{k}")

        def _backward():
            self.grad += k * (self.data ** (k - 1)) * output.grad

        output._backward = _backward
        return output

    def __sub__(self, other):  # self - other
        return self + (-other)

    def __neg__(self):  # -self
        return self * (-1)

    def exp(self):
        x = self.data
        r = math.exp(x)

        output = Value(t, (self,), _op="exp")

        def _backward():
            self.grad += r * output.grad

        output._backward = _backward
        return output

    def backward(self):
        topo = topological_sort(self)
        self.grad = 1

        for node in reversed(topo):
            node._backward()
