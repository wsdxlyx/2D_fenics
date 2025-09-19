import numpy as np


default_cycle = np.array(
    [1.0, 180.0, 3.0, 1500.0],
)

default_n_cycles = 50


def create_time_points(
    n_cycles: int, one_cycle: np.ndarray, output_division=1, skip_cold=False
) -> tuple[np.number, np.ndarray, np.ndarray]:
    period = one_cycle.sum()
    cycle_time_points = np.zeros(n_cycles * len(one_cycle) + 1)
    cycle_time_points[1:] = np.tile(one_cycle, n_cycles).cumsum()

    # Add a point in between all points from cycle_time_points
    if output_division >= 1:
        if skip_cold:
            output_division = output_division * np.ones_like(one_cycle, dtype=int)
            output_division[-1] = 1
        one_cycle_output = np.repeat(one_cycle / output_division, output_division)
    elif output_division == 0.5:
        one_cycle_output = one_cycle[::2] + one_cycle[1::2]
    elif output_division == 0.25:
        one_cycle_output = one_cycle.sum()[None]
    else:
        raise ValueError(
            "output_division must be greater than or equal to 1 or exactly equal to 0.25 or 0.5"
        )

    output_time_points = np.zeros(n_cycles * len(one_cycle_output) + 1)
    output_time_points[1:] = np.tile(one_cycle_output, n_cycles).cumsum()
    return period, cycle_time_points, output_time_points
