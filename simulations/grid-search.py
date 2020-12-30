from main import main

from typing import Any, Callable, Tuple

from time import time


def time_run(call):
    stime = time()
    call()
    return stime - time()


def optimize2(
    pmin: float, pmax: float, fn: Callable[[float], Any], depth=5
) -> Tuple[float, float]:
    center = (pmin + pmax) / 2
    root_time = time_run(lambda: fn(center))
    if not depth:
        return root_time, center
    else:
        l_res = optimize2(pmin, center, fn, depth=depth - 1)
        r_res = optimize2(center, pmax, fn, depth=depth - 1)

        def get_time(t: Tuple[float, float]) -> float:
            return t[0]

        data = [l_res, r_res, (root_time, center)]
        data.sort(key=get_time)
        print(f"\nDepth: {depth} - Space: {pmin} - {pmax}")
        print(f"Best time: {data[0][0]} - value: {data[0][1]}")
        return data[0]


total_iters = 2 ** 12


def get_param(x: float):
    return 2 ** x


optimize2(0, 8, fn=lambda v: main(int(total_iters), int(get_param(v))), depth=2)