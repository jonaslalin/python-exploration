import sys
from typing import TypeVar, Union, overload

if sys.version_info < (3, 9):
    from typing import Sequence
else:
    from collections.abc import Sequence


TItem = TypeVar("TItem")


class WindowedDataset(Sequence[Sequence[TItem]]):
    _items: Sequence[TItem]
    window_size: int

    def __init__(self, items: Sequence[TItem], window_size: int = 1) -> None:
        assert window_size > 0
        self._items = items
        self.window_size = window_size

    @overload
    def __getitem__(self, index: int) -> Sequence[TItem]: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[Sequence[TItem]]: ...

    def __getitem__(self, index: Union[int, slice]) -> Union[Sequence[TItem], Sequence[Sequence[TItem]]]:
        if isinstance(index, int):
            if index >= len(self) or index < -len(self):
                raise IndexError("Window index out of range")
            window_index = index % len(self)
            window = self._items[window_index * self.window_size : (window_index + 1) * self.window_size]
            return window
        elif isinstance(index, slice):
            window_start_index, window_stop_index, window_stride = index.indices(len(self))
            windows = tuple(
                self[window_index]
                for window_index in range(
                    window_start_index,
                    window_stop_index,
                    window_stride,
                )
            )
            return windows
        else:
            raise TypeError("Only int and slice types are supported")

    def __len__(self) -> int:
        return -(-len(self._items) // self.window_size)
