"""Time SymSpell._edits on every term in the bundled frequency dictionary.

The timed call matches SymSpell._edits_prefix: the term is clipped to
prefix_length, the seed delete set is built, then _edits fills it. Inputs are
the dictionary file in order. No RNG.
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

from symspellpy import SymSpell

REPEATS = 9
WARMUP = 1


def load_words() -> list[str]:
    dictionary = files("symspellpy").joinpath("frequency_dictionary_en_82_765.txt")
    words: list[str] = []
    for line in dictionary.read_text(encoding="utf-8").splitlines():
        if line:
            words.append(line.split()[0])
    return words


def run(sym: SymSpell, words: list[str]) -> int:
    """Return the total number of delete strings produced."""
    total = 0
    max_distance = sym._max_dictionary_edit_distance
    prefix_length = sym._prefix_length
    edits = sym._edits
    for key in words:
        delete_words: set[str] = set()
        if len(key) <= max_distance:
            delete_words.add("")
        if len(key) > prefix_length:
            key = key[:prefix_length]
        delete_words.add(key)
        total += len(edits(key, 0, delete_words))
    return total


def main() -> None:
    words = load_words()
    sym = SymSpell()
    checksum = 0
    for _ in range(WARMUP):
        checksum = run(sym, words)
    samples: list[float] = []
    for _ in range(REPEATS):
        gc.collect()
        started = time.perf_counter()
        checksum = run(sym, words)
        samples.append((time.perf_counter() - started) * 1000.0)
    median_ms = statistics.median(samples)
    print(f"checksum={checksum}")
    print(f"median_ms={median_ms:.3f}")


if __name__ == "__main__":
    main()
