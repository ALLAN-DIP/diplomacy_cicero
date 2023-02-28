from abc import ABC, abstractmethod
from dataclasses import dataclass

from typing_extensions import get_args

from daidepp.grammar import create_daide_grammar
from daidepp.grammar.grammar import DAIDELevel

_grammar = create_daide_grammar(get_args(DAIDELevel)[-1], string_type="all")


@dataclass
class _DAIDEObject(ABC):
    @abstractmethod
    def __str__(self) -> str:
        pass
