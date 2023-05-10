import math


class Value:
    def __init__(self, data, children=(), _op="") -> None:
        self.data = data
        self._prev = set(children)
        self._op = _op
        self.grad = 0

    def __repr__(self) -> str:
        return f"Value: (data: {self.data})"

    def __add__(self, other):
        output = self.data + other.data
        return Value(output, children=(self, other), _op="*")

    def __mul__(self, other):
        output = self.data * other.data
        return Value(output, children=(self, other), _op="*")

    def tanh(self):
        n = self.data
        output = (math.exp(2 * n) - 1) / (math.exp(2 * n) + 1)
        return Value(output, (self,), _op="tan")
