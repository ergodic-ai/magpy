import logging
import pandas
from typing import Any


class BaseLogger(logging.Logger):
    def __init__(self, name: str, level: int = logging.NOTSET):
        super().__init__(name, level)

    def send_data(self, data: Any, metadata: dict):
        pass

    def graph(self, graph: pandas.DataFrame, metadata: dict):
        self.send_data(graph, metadata)

    def table(self, table: pandas.DataFrame, metadata: dict):
        self.send_data(table, metadata)

    def thought(self, thought: str, metadata: dict):
        self.send_data(thought, metadata)


def test():
    logging.basicConfig(level=logging.INFO)
    logger = BaseLogger("magpy")

    # set level to info
    logger.setLevel(logging.INFO)

    logger.error("Hello, world!")

    logger.graph(pandas.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}), {})
    logger.table(pandas.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}), {})
    logger.thought("Hello, world!", {})


if __name__ == "__main__":
    test()
