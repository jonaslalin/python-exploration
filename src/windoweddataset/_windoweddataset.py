import sys
from dataclasses import dataclass
from typing import TypeVar, Union, overload

if sys.version_info < (3, 9):
    from typing import Sequence
else:
    from collections.abc import Sequence

TItem = TypeVar("TItem")


@dataclass(init=False)
class WindowedDataset(Sequence[Sequence[TItem]]):
    dataset: Sequence[TItem]
    window_size: int

    def __init__(self, dataset: Sequence[TItem], window_size: int = 1) -> None:
        assert window_size > 0
        self.dataset = dataset
        self.window_size = window_size

    @overload
    def __getitem__(self, index: int) -> Sequence[TItem]: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[Sequence[TItem]]: ...

    def __getitem__(self, index: Union[int, slice]) -> Union[Sequence[TItem], Sequence[Sequence[TItem]]]:
        if isinstance(index, int):
            dataset_start, dataset_stop = index * self.window_size, (index + 1) * self.window_size
            window: Sequence[TItem] = self.dataset[dataset_start:dataset_stop]
            return window
        elif isinstance(index, slice):
            window_start, window_stop, window_step = index.start, index.stop, index.step
            windows_idx = range(window_start, window_stop, window_step)
            windows: Sequence[Sequence[TItem]] = tuple(self[window_idx] for window_idx in windows_idx)
            return windows
        else:
            raise TypeError(...)

    def __len__(self) -> int:
        return -(-len(self.dataset) // self.window_size)
