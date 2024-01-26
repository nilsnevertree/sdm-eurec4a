import numpy as np
import pandas as pd
import pytest
import xarray as xr

from sdm_eurec4a.identifications import (
    consecutive_events_np,
    consecutive_events_xr,
    match_clouds_and_cloudcomposite,
    match_clouds_and_dropsondes,
)


def test_consecutive_events_np_old():
    """Tests the consecutive_events_np function
    It handles the following cases:
    1. min_duration = 1
    2. min_duration = 3, axis = 1
    3. min_duration = 3, axis = 0
    4. min_duration = 3, axis = 0, mask transposed
    5. min_duration = 0

    The original mask has the following shape: (3,12)
    has cosecutive events of length 3 in
    - the middle of the array
    - both ends of the array

    """
    # Set up example array
    mask = np.array(
        [
            [1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0],
            [1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0],
            [0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1],
        ],
        dtype=bool,
    )

    # --------------
    # Test 1
    # min_duration = 1
    expected_result_1 = mask

    # Axis = 1
    result_1a = consecutive_events_np(
        mask=mask,
        min_duration=1,
        axis=1,
    )
    np.testing.assert_array_equal(result_1a, expected_result_1)
    # Axis = 0
    result_1b = consecutive_events_np(
        mask=mask,
        min_duration=1,
        axis=0,
    )
    np.testing.assert_array_equal(result_1a, expected_result_1)
    np.testing.assert_array_equal(result_1b, expected_result_1)

    # --------------
    # Test 2
    # min_duration = 3
    # Axis = 1

    result_2 = consecutive_events_np(
        mask=mask,
        min_duration=3,
        axis=1,
    )
    expected_result_2 = np.array(
        [
            [0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0],
            [1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
        ],
        dtype=bool,
    )
    np.testing.assert_array_equal(result_2, expected_result_2)

    # --------------
    # Test 3
    # min_duration = 3
    # Axis = 0 which is the shorter one in the arrays

    result_3 = consecutive_events_np(
        mask=mask,
        min_duration=3,
        axis=0,
    )
    expected_result_3 = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
        ],
        dtype=bool,
    )
    np.testing.assert_array_equal(result_3, expected_result_3)

    # Test 4
    # Same as Test 2 but with axis = 0 and tranpsosed mask
    result_4 = consecutive_events_np(mask=mask.T, min_duration=3, axis=0)
    np.testing.assert_array_equal(result_4, expected_result_2.T)

    # Test 5
    result_5 = consecutive_events_np(mask=mask, min_duration=0, axis=0)
    np.testing.assert_array_equal(result_5, np.zeros_like(mask))


@pytest.fixture
def mask():
    return np.array(
        [
            [1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0],
            [1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0],
            [0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1],
        ],
        dtype=bool,
    )


@pytest.mark.parametrize(
    "min_duration, axis, expected",
    [
        # Test 1
        # min_duration = 1
        #  a)
        # axis = 1, which is the longer one in the arrays
        (
            1,
            1,
            np.array(
                [
                    [1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0],
                    [1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0],
                    [0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1],
                ],
                dtype=bool,
            ),
        ),
        #  b)
        # axis = 0, which is the shorter one in the arrays
        (
            1,
            0,
            np.array(
                [
                    [1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0],
                    [1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0],
                    [0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1],
                ],
                dtype=bool,
            ),
        ),
        # Test 2
        # min_duration = 3
        # axis = 1
        (
            3,
            1,
            np.array(
                [
                    [0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0],
                    [1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
                ],
                dtype=bool,
            ),
        ),
        # Test 3
        # min_duration = 3
        # axis = 0
        (
            3,
            0,
            np.array(
                [
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
                ],
                dtype=bool,
            ),
        ),
        # Test 4
        # min_duration = 0
        # results should be zeros
        (0, 0, np.zeros_like(mask)),
        (0, 1, np.zeros_like(mask)),
    ],
)
def test_consecutive_events_np(mask, min_duration, axis, expected):
    print(mask, min_duration, axis, expected)
    result = consecutive_events_np(mask, min_duration=min_duration, axis=axis)
    np.testing.assert_array_equal(result, expected)


def test_consecutive_events_xr():
    """Tests the consecutive_events_xr function
    It handles the following cases:
    1. min_duration = 1
    2. min_duration = 3, axis = "space"
    3. min_duration = 3, axis = "time"
    4. min_duration = 0

    The original mask has the following shape: (3,12)
    has cosecutive events of length 3 in
    - the middle of the array
    - both ends of the array

    """
    # Set up example array
    mask_np = np.array(
        [
            [1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0],
            [1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0],
            [0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1],
        ],
        dtype=bool,
    )
    mask = xr.DataArray(
        mask_np,
        dims=("time", "space"),
        coords={"time": np.arange(0, 3), "space": np.arange(0, 12)},
    )

    # --------------
    # Test 1
    # Check for min_duration = 1

    expected_result_1 = xr.DataArray(
        mask_np,
        dims=("time", "space"),
        coords={"time": np.arange(0, 3), "space": np.arange(0, 12)},
    )

    # Dimension time
    result_1a = consecutive_events_xr(
        da_mask=mask,
        min_duration=1,
        axis="time",
    )
    np.testing.assert_array_equal(result_1a, expected_result_1)
    xr.testing.assert_identical(result_1a, expected_result_1)

    # Dimension space
    result_1b = consecutive_events_xr(
        da_mask=mask,
        min_duration=1,
        axis="space",
    )
    np.testing.assert_array_equal(result_1b, expected_result_1)
    xr.testing.assert_identical(result_1b, expected_result_1)

    # --------------
    # Test 2
    # min_duration = 3
    # Axis = "space"

    result_2 = consecutive_events_xr(
        da_mask=mask,
        min_duration=3,
        axis="space",
    )
    expected_result_2 = xr.DataArray(
        np.array(
            [
                [0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0],
                [1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
            ],
            dtype=bool,
        ),
        dims=("time", "space"),
        coords={"time": np.arange(0, 3), "space": np.arange(0, 12)},
    )
    np.testing.assert_array_equal(result_2, expected_result_2)
    xr.testing.assert_identical(result_2, expected_result_2)

    # --------------
    # Test 3
    # min_duration = 3
    # Axis = time which is the shorter one in the arrays

    result_3 = consecutive_events_xr(
        da_mask=mask,
        min_duration=3,
        axis="time",
    )

    expected_result_3 = xr.DataArray(
        np.array(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
            ],
            dtype=bool,
        ),
        dims=("time", "space"),
        coords={"time": np.arange(0, 3), "space": np.arange(0, 12)},
    )
    np.testing.assert_array_equal(result_3, expected_result_3)
    xr.testing.assert_identical(result_3, expected_result_3)

    # Test 4
    # ----------------
    # min_duration = 0

    result_4a = consecutive_events_xr(
        da_mask=mask,
        min_duration=0,
        axis="time",
    )

    result_4b = consecutive_events_xr(
        da_mask=mask,
        min_duration=0,
        axis="space",
    )

    expected_result_4 = xr.DataArray(
        np.zeros_like(mask_np),
        dims=("time", "space"),
        coords={"time": np.arange(0, 3), "space": np.arange(0, 12)},
    )
    np.testing.assert_array_equal(result_4a, expected_result_4)
    np.testing.assert_array_equal(result_4b, expected_result_4)
    xr.testing.assert_identical(result_4a, expected_result_4)
    xr.testing.assert_identical(result_4b, expected_result_4)
