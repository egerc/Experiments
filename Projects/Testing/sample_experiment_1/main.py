from dataclasses import dataclass
from itertools import product
from typing import List
from exp_runner import Variable, runner, MetaData


@dataclass
class Input:
    a: int
    b: int


variables = [
    Variable(value=Input(a, b), metadata={"a": a, "b": b})
    for a, b in product([1, 2], [2, 3])
]


@runner()
def experiment(x: Input) -> List[MetaData]:
    return [{"id": id, "sum": x.a + x.b, "product": x.a * x.b} for id in range(10)]


def main():
    experiment(variables)


if __name__ == "__main__":
    main()
