from collections.abc import Iterator
from pathlib import Path
from types import TracebackType
from typing import Literal

class Page:
    def get_text(self, option: Literal["text"]) -> str: ...

class Document:
    def __enter__(self) -> Document: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None: ...
    def __iter__(self) -> Iterator[Page]: ...

def open(filename: str | Path) -> Document: ...
