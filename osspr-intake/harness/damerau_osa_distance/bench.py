"""Time DamerauOsa._distance on near-miss pairs from the frequency dictionary.

Same fixed two-edit dictionary pairs as the banded bench, but max_distance is
large enough that DamerauOsa.distance takes the unbounded _distance path
(remaining length is not greater than max_distance). That is the path used for
short words. No RNG. Cost buffers are reused the way DamerauOsa.distance
reuses them.
"""

from __future__ import annotations

import gc
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from importlib.resources import files

from symspellpy.editdistance import DamerauOsa
from symspellpy.helpers import prefix_suffix_prep

# Larger than any remaining slice of a dictionary word, so distance() uses
# _distance rather than _distance_max.
MAX_DISTANCE = 64
PASSES = 2
REPEATS = 9
WARMUP = 1


def load_words() -> list[str]:
    dictionary = files("symspellpy").joinpath("frequency_dictionary_en_82_765.txt")
    words: list[str] = []
    for line in dictionary.read_text(encoding="utf-8").splitlines():
        if line:
            words.append(line.split()[0])
    return words


def variant(word: str) -> str:
    chars = list(word)
    first = 1 if len(word) > 1 else 0
    last = len(chars) - 1
    chars[first] = "q" if chars[first] != "q" else "z"
    if last != first:
        chars[last] = "x" if chars[last] != "x" else "j"
    return "".join(chars)


def build_jobs(words: list[str]) -> list[tuple[str, str, int, int, int]]:
    comparer = DamerauOsa()
    jobs: list[tuple[str, str, int, int, int]] = []
    for word in words:
        if len(word) < 4:
            continue
        other = variant(word)
        expected = comparer.distance(word, other, MAX_DISTANCE)
        string_1, string_2 = word, other
        if len(string_1) > len(string_2):
            string_2, string_1 = string_1, string_2
        len_1, len_2, start = prefix_suffix_prep(string_1, string_2)
        if len_1 == 0 or MAX_DISTANCE < len_2:
            continue
        costs = comparer._base_char_1_costs
        prev = comparer._base_prev_char_1_costs
        if len_2 > len(costs):
            costs = [0 for _ in range(len_2)]
            prev = [0 for _ in range(len_2)]
            comparer._base_char_1_costs = costs
            comparer._base_prev_char_1_costs = prev
        direct = DamerauOsa._distance(
            string_1, string_2, len_1, len_2, start, costs, prev
        )
        if direct != expected:
            raise RuntimeError(
                f"direct _distance {direct} != distance() {expected} for {word!r}"
            )
        jobs.append((string_1, string_2, len_1, len_2, start))
    if not jobs:
        raise RuntimeError("no _distance jobs built from the frequency dictionary")
    return jobs


def run(
    jobs: list[tuple[str, str, int, int, int]],
    costs: list[int],
    prev: list[int],
) -> int:
    total = 0
    distance = DamerauOsa._distance
    for _ in range(PASSES):
        for string_1, string_2, len_1, len_2, start in jobs:
            total += distance(string_1, string_2, len_1, len_2, start, costs, prev)
    return total


def main() -> None:
    jobs = build_jobs(load_words())
    longest = max(job[3] for job in jobs)
    costs = [0 for _ in range(longest)]
    prev = [0 for _ in range(longest)]
    checksum = 0
    for _ in range(WARMUP):
        checksum = run(jobs, costs, prev)
    samples: list[float] = []
    for _ in range(REPEATS):
        gc.collect()
        started = time.perf_counter()
        checksum = run(jobs, costs, prev)
        samples.append((time.perf_counter() - started) * 1000.0)
    median_ms = statistics.median(samples)
    print(f"jobs={len(jobs)}")
    print(f"checksum={checksum}")
    print(f"median_ms={median_ms:.3f}")


if __name__ == "__main__":
    main()
