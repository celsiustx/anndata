from dataclasses import dataclass
from typing import Any


@dataclass
class Obj:
    '''Dict wrapper that supports getattr'''
    dict: Any
    default: Any = None

    def __getattr__(self, item):
        if item in self.dict:
            return self.dict[item]

        if self.default:
            return self.default[item]

        return self.dict[item]


