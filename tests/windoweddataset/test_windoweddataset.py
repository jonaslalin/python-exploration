import sys

import pytest

from windoweddataset import WindowedDataset

if sys.version_info < (3, 9):
    from typing import Sequence
else:
    from collections.abc import Sequence


@pytest.fixture
def sample_dataset() -> Sequence[int]:
    return tuple(range(20))


@pytest.fixture
def sample_windowed_dataset(sample_dataset: Sequence[int]) -> WindowedDataset[int]:
    return WindowedDataset(sample_dataset, window_size=3)


def test_len(sample_windowed_dataset: WindowedDataset[int]) -> None:
    assert len(sample_windowed_dataset) == 7


@pytest.mark.parametrize(
    ("window_index", "expected_window"),
    [
        (0, (0, 1, 2)),
        (1, (3, 4, 5)),
        (2, (6, 7, 8)),
        (3, (9, 10, 11)),
        (4, (12, 13, 14)),
        (5, (15, 16, 17)),
        (6, (18, 19)),
        (-1, (18, 19)),
        (-2, (15, 16, 17)),
        (-3, (12, 13, 14)),
        (-4, (9, 10, 11)),
        (-5, (6, 7, 8)),
        (-6, (3, 4, 5)),
        (-7, (0, 1, 2)),
    ],
)
def test_get_item_for_int(
    window_index: int,
    expected_window: Sequence[int],
    sample_windowed_dataset: WindowedDataset[int],
) -> None:
    assert sample_windowed_dataset[window_index] == expected_window


def test_git_item_for_int_raises_exceptions(sample_windowed_dataset: WindowedDataset[int]) -> None:
    with pytest.raises(IndexError):
        sample_windowed_dataset[7]
    with pytest.raises(IndexError):
        sample_windowed_dataset[-8]
    with pytest.raises(TypeError):
        sample_windowed_dataset[0:2, 5]  # type: ignore[call-overload]


@pytest.mark.parametrize(
    ("window_slice", "expected_windows"),
    [
        (slice(0, 2), ((0, 1, 2), (3, 4, 5))),
        (slice(0, 6, 2), ((0, 1, 2), (6, 7, 8), (12, 13, 14))),
        (slice(-1, 6), ()),
        (slice(-1, 7), ((18, 19),)),
        (slice(-1, 27), ((18, 19),)),
    ],
)
def test_get_item_for_slice(
    window_slice: slice,
    expected_windows: Sequence[Sequence[int]],
    sample_windowed_dataset: WindowedDataset[int],
) -> None:
    assert sample_windowed_dataset[window_slice] == expected_windows
