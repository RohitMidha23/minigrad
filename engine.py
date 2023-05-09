class Value:
    def __init__(self, data, children=(), _op="") -> None:
        self.data = data
        self._prev = set(children)
        self._op = _op
        self.grad = None

    def __repr__(self) -> str:
        return f"Value: (data: {self.data})"

    def __add__(self, other):
        output = self.data + other.data
        return Value(output, children=(self, other), _op="*")

    def __mul__(self, other):
        output = self.data * other.data
        return Value(output, children=(self, other), _op="*")
