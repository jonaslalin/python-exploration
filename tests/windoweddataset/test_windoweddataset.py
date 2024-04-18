from typing import Tuple

import pytest

from windoweddataset import WindowedDataset


@pytest.fixture
def dataset() -> Tuple[int, ...]:
    return tuple(range(5))


@pytest.fixture
def windowed_dataset(dataset: Tuple[int, ...]) -> WindowedDataset[int]:
    return WindowedDataset(dataset, window_size=2)


@pytest.mark.parametrize(
    ("window", "expected"),
    [
        (0, (0, 1)),
        (1, (2, 3)),
        (2, (4,)),
    ],
)
def test_windowed_dataset(
    window: int,
    expected: Tuple[int, ...],
    windowed_dataset: WindowedDataset[int],
) -> None:
    assert windowed_dataset[window] == expected
