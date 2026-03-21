"""Benchmark suite for event-jepa-cube.

Run:
    python tests/benchmark.py
    python tests/benchmark.py --dim 256 --max-seq 50000
"""

from __future__ import annotations

import argparse
import math
import random
import resource
import time
from dataclasses import dataclass
from typing import List

from event_jepa_cube import EmbeddingCube, Entity, EventJEPA, EventSequence
from event_jepa_cube.streaming import StreamBuffer, StreamingJEPA


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@dataclass
class BenchResult:
    name: str
    iterations: int
    total_sec: float
    peak_rss_mb: float

    @property
    def per_iter_ms(self) -> float:
        return (self.total_sec / self.iterations) * 1000

    @property
    def throughput(self) -> float:
        return self.iterations / self.total_sec if self.total_sec > 0 else float("inf")


def _rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # Linux: kB -> MB


def _rand_embedding(dim: int) -> List[float]:
    return [random.gauss(0, 1) for _ in range(dim)]


def _make_sequence(n: int, dim: int) -> EventSequence:
    return EventSequence(
        embeddings=[_rand_embedding(dim) for _ in range(n)],
        timestamps=[float(i) for i in range(n)],
    )


def _bench(name: str, fn, iterations: int) -> BenchResult:
    # warm-up
    fn()
    rss_before = _rss_mb()
    t0 = time.perf_counter()
    for _ in range(iterations):
        fn()
    elapsed = time.perf_counter() - t0
    rss_after = _rss_mb()
    return BenchResult(name, iterations, elapsed, rss_after - rss_before)


def _print_result(r: BenchResult) -> None:
    print(f"  {r.name:<45s}  {r.per_iter_ms:>9.3f} ms/iter   {r.throughput:>10.1f} iter/s   RSS delta {r.peak_rss_mb:+.1f} MB")


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------

def bench_process_scaling(dims: List[int], lengths: List[int]) -> None:
    """EventJEPA.process() across dims and sequence lengths."""
    print("\n== EventJEPA.process() scaling ==")
    for dim in dims:
        jepa = EventJEPA(embedding_dim=dim)
        for n in lengths:
            seq = _make_sequence(n, dim)
            iters = max(1, 500 // max(1, n // 100))
            r = _bench(f"dim={dim:<4d} n={n}", lambda: jepa.process(seq), iters)
            _print_result(r)


def bench_process_multilevel(dim: int, n: int) -> None:
    """process_multilevel() at various num_levels."""
    print(f"\n== EventJEPA.process_multilevel() dim={dim} n={n} ==")
    for levels in [1, 2, 4, 8]:
        jepa = EventJEPA(embedding_dim=dim, num_levels=levels)
        seq = _make_sequence(n, dim)
        r = _bench(f"levels={levels}", lambda: jepa.process_multilevel(seq), 200)
        _print_result(r)


def bench_predict_next(dim: int, n: int) -> None:
    """predict_next() varying num_steps."""
    print(f"\n== EventJEPA.predict_next() dim={dim} n={n} ==")
    jepa = EventJEPA(embedding_dim=dim)
    seq = _make_sequence(n, dim)
    for steps in [1, 5, 20]:
        r = _bench(f"steps={steps}", lambda: jepa.predict_next(seq, num_steps=steps), 200)
        _print_result(r)


def bench_dense_loss(dim: int) -> None:
    """compute_dense_loss() overhead."""
    print(f"\n== Dense loss vs regularized loss  dim={dim} ==")
    jepa = EventJEPA(embedding_dim=dim)
    n = 200
    embs = [_rand_embedding(dim) for _ in range(n)]
    ts = [float(i) for i in range(n)]
    mask_ts = [float(i) for i in range(0, n, 5)]
    pred_loss = 1.0

    r1 = _bench(
        "compute_regularized_loss",
        lambda: jepa.compute_regularized_loss(embs, pred_loss),
        500,
    )
    r2 = _bench(
        "compute_dense_loss (lambda=0.5)",
        lambda: jepa.compute_dense_loss(embs, ts, embs, mask_ts, pred_loss, 0.5),
        500,
    )
    _print_result(r1)
    _print_result(r2)
    if r1.total_sec > 0:
        print(f"  Dense overhead: {r2.per_iter_ms / r1.per_iter_ms:.2f}x")


def bench_streaming_throughput(dims: List[int], n: int) -> None:
    """StreamingJEPA.update() throughput (events/sec)."""
    print(f"\n== StreamingJEPA.update() throughput  n={n} ==")
    for dim in dims:
        sj = StreamingJEPA(embedding_dim=dim)
        embs = [_rand_embedding(dim) for _ in range(n)]
        ts = [float(i) for i in range(n)]

        def run():
            sj.reset()
            for e, t in zip(embs, ts):
                sj.update(e, t)

        r = _bench(f"dim={dim}", run, 5)
        evts_per_sec = n * r.iterations / r.total_sec
        print(f"  dim={dim:<4d}  {r.per_iter_ms:>9.3f} ms/iter   {evts_per_sec:>12,.0f} events/s   RSS delta {r.peak_rss_mb:+.1f} MB")


def bench_streaming_batch_vs_sequential(dim: int, n: int) -> None:
    """update_batch() vs sequential update()."""
    print(f"\n== Batch vs sequential  dim={dim} n={n} ==")
    embs = [_rand_embedding(dim) for _ in range(n)]
    ts = [float(i) for i in range(n)]

    sj = StreamingJEPA(embedding_dim=dim)

    def sequential():
        sj.reset()
        for e, t in zip(embs, ts):
            sj.update(e, t)

    def batch():
        sj.reset()
        sj.update_batch(embs, ts)

    r_seq = _bench("sequential", sequential, 10)
    r_bat = _bench("batch", batch, 10)
    _print_result(r_seq)
    _print_result(r_bat)
    if r_bat.per_iter_ms > 0:
        print(f"  Batch speedup: {r_seq.per_iter_ms / r_bat.per_iter_ms:.2f}x")


def bench_streaming_convergence(dim: int) -> None:
    """Measure how many events until streaming repr converges to batch."""
    print(f"\n== Streaming convergence  dim={dim} ==")
    n = 500
    seq = _make_sequence(n, dim)
    batch_jepa = EventJEPA(embedding_dim=dim)
    batch_repr = batch_jepa.process(seq)

    sj = StreamingJEPA(embedding_dim=dim)
    checkpoints = [10, 25, 50, 100, 200, 500]
    for i, (emb, ts) in enumerate(zip(seq.embeddings, seq.timestamps)):
        sj.update(emb, ts)
        if (i + 1) in checkpoints:
            sr = sj.get_representation()
            # cosine similarity
            dot = sum(a * b for a, b in zip(sr, batch_repr))
            norm_a = math.sqrt(sum(a * a for a in sr))
            norm_b = math.sqrt(sum(b * b for b in batch_repr))
            cos = dot / (norm_a * norm_b) if norm_a > 0 and norm_b > 0 else 0.0
            print(f"  after {i+1:>4d} events: cosine_sim={cos:.6f}")


def bench_embedding_cube(dim: int) -> None:
    """EmbeddingCube entity add + relationship discovery."""
    print(f"\n== EmbeddingCube  dim={dim} ==")
    cube = EmbeddingCube()
    entities = []
    for i in range(200):
        e = Entity(embeddings={"text": _rand_embedding(dim), "visual": _rand_embedding(dim)})
        entities.append(e)

    r_add = _bench("add 200 entities", lambda: [EmbeddingCube().add_entity(e) for e in entities], 20)
    _print_result(r_add)

    for e in entities:
        cube.add_entity(e)
    ids = [e.id for e in entities[:50]]
    r_disc = _bench("discover_relationships (50 ids)", lambda: cube.discover_relationships(ids, threshold=0.3), 20)
    _print_result(r_disc)


def bench_memory_large_sequence(dim: int, max_n: int) -> None:
    """Peak RSS for increasingly large sequences."""
    print(f"\n== Memory scaling  dim={dim} ==")
    for n in [1000, 5000, 10000, max_n]:
        if n > max_n:
            continue
        rss0 = _rss_mb()
        seq = _make_sequence(n, dim)
        jepa = EventJEPA(embedding_dim=dim)
        _ = jepa.process(seq)
        rss1 = _rss_mb()
        print(f"  n={n:<6d}  RSS delta {rss1 - rss0:+.1f} MB")
        del seq


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark event-jepa-cube")
    parser.add_argument("--dim", type=int, default=128, help="default embedding dim")
    parser.add_argument("--max-seq", type=int, default=10000, help="max sequence length for memory test")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    dim = args.dim

    print(f"Benchmarking event-jepa-cube  dim={dim}  seed={args.seed}")
    print("=" * 80)

    bench_process_scaling([64, dim, 512], [100, 500, 1000, 5000])
    bench_process_multilevel(dim, 1000)
    bench_predict_next(dim, 500)
    bench_dense_loss(dim)
    bench_streaming_throughput([64, dim, 512], 2000)
    bench_streaming_batch_vs_sequential(dim, 2000)
    bench_streaming_convergence(dim)
    bench_embedding_cube(dim)
    bench_memory_large_sequence(dim, args.max_seq)

    print("\n" + "=" * 80)
    print("Done.")


if __name__ == "__main__":
    main()
