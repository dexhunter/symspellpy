"""Time Levenshtein._distance_max on near-miss pairs from the frequency dictionary.

Same fixed two-edit pairs as the Damerau bench: every bundled dictionary word
of length >= 4, substitutions chosen without RNG. Only pairs that
Levenshtein.distance sends through _distance_max are timed. Cost buffers are
reused the way Levenshtein.distance reuses them.
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

from symspellpy.editdistance import Levenshtein
from symspellpy.helpers import prefix_suffix_prep

MAX_DISTANCE = 2
# Two passes over the ~80k banded comparisons land near half a second.
PASSES = 2
REPEATS = 11
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
    first = 1
    last = len(chars) - 1
    chars[first] = "q" if chars[first] != "q" else "z"
    chars[last] = "x" if chars[last] != "x" else "j"
    return "".join(chars)


def build_jobs(words: list[str]) -> list[tuple[str, str, int, int, int]]:
    comparer = Levenshtein()
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
        if len_1 == 0 or MAX_DISTANCE >= len_2:
            continue
        costs = comparer._base_char_1_costs
        if len_2 > len(costs):
            costs = [0 for _ in range(len_2)]
            comparer._base_char_1_costs = costs
        direct = Levenshtein._distance_max(
            string_1, string_2, len_1, len_2, start, MAX_DISTANCE, costs
        )
        if direct != expected:
            raise RuntimeError(
                f"direct _distance_max {direct} != distance() {expected} for {word!r}"
            )
        jobs.append((string_1, string_2, len_1, len_2, start))
    if not jobs:
        raise RuntimeError("no _distance_max jobs built from the frequency dictionary")
    return jobs


def run(jobs: list[tuple[str, str, int, int, int]], costs: list[int]) -> int:
    total = 0
    distance_max = Levenshtein._distance_max
    for _ in range(PASSES):
        for string_1, string_2, len_1, len_2, start in jobs:
            total += distance_max(
                string_1, string_2, len_1, len_2, start, MAX_DISTANCE, costs
            )
    return total


def main() -> None:
    jobs = build_jobs(load_words())
    longest = max(job[3] for job in jobs)
    costs = [0 for _ in range(longest)]
    checksum = 0
    for _ in range(WARMUP):
        checksum = run(jobs, costs)
    samples: list[float] = []
    for _ in range(REPEATS):
        gc.collect()
        started = time.perf_counter()
        checksum = run(jobs, costs)
        samples.append((time.perf_counter() - started) * 1000.0)
    median_ms = statistics.median(samples)
    print(f"jobs={len(jobs)}")
    print(f"checksum={checksum}")
    print(f"median_ms={median_ms:.3f}")


if __name__ == "__main__":
    main()
