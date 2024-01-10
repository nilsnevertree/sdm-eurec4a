import numpy as np

from sdm_eurec4a.identifications import consecutive_events_np


def test_consecutive_events_np():
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

    # Test 1
    # Check for min_duration = 1
    result = consecutive_events_np(
        mask=mask,
        min_duration=1,
        axis=1,
    )
    expected_result_2 = mask
    np.testing.assert_array_equal(result, expected_result_2)
    # Same for axis = 0
    result = consecutive_events_np(
        mask=mask,
        min_duration=1,
        axis=0,
    )
    expected_result_2 = mask
    np.testing.assert_array_equal(result, expected_result_2)

    # Test 2 Check along axis 1 which is the longer one in the arrays
    result = consecutive_events_np(
        mask=mask,
        min_duration=3,
        axis=1,
    )
    expected_result_1 = np.array(
        [
            [0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0],
            [1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
        ],
        dtype=bool,
    )
    np.testing.assert_array_equal(result, expected_result_1)

    # Test 3 Check along axis 0 which is the shorter one in the arrays
    min_duration = 3
    axis = 0
    result = consecutive_events_np(
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
    np.testing.assert_array_equal(result, expected_result_3)
    # Test 4
    # same as test 1 but with axis = 0 and tranpsosed mask
    result = consecutive_events_np(mask=mask.T, min_duration=3, axis=0)
    np.testing.assert_array_equal(result, expected_result_1.T)

    # Test 5
    result = consecutive_events_np(mask=mask, min_duration=0, axis=0)
    np.testing.assert_array_equal(result, np.zeros_like(mask))
