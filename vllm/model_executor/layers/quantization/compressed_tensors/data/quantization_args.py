from enum import Enum


class Strategy(Enum):
    TENSOR = "tensor"
    TOKEN = "token"
    BLOCK = "block"
    GROUP = "group"
