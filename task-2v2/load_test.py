import argparse
import asyncio
import time
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Optional
from aiohttp import ClientSession

class TraceGenerator(ABC):
    """Abstract base class for trace generation."""

    @abstractmethod
    def generate(self) -> List[int]:
        """
        Generate a list of timestamps (ms) at which requests should be sent.

        Returns:
            List[int]: Sorted list of timestamps in milliseconds.

        Raises:
            ValueError: If inputs are invalid or generation fails.
        """
        pass


class SyntheticTraceGenerator(TraceGenerator):
    """Generate synthetic traces based on patterns."""

    def __init__(
        self, rps: int, pattern: str, duration: int, seed: Optional[int] = None
    ):
        """
        Initialize synthetic trace generator.

        Args:
            rps (int): Requests per second. Must be non-negative.
            pattern (str): Distribution pattern ('uniform', 'random', 'poisson', etc.).
            duration (int): Total duration in seconds. Must be non-negative.
            seed (int): Seed for reproducibility of 'poisson' and 'random' patterns
        """
        if not isinstance(rps, int) or rps < 0:
            raise ValueError("rps must be a non-negative integer")
        if not isinstance(duration, int) or duration < 0:
            raise ValueError("duration must be a non-negative integer")

        self.rps = rps
        self.pattern = pattern
        self.duration = duration
        if seed is not None:
            np.random.seed(seed)

    def generate(self) -> List[int]:
        total_requests = self.rps * self.duration
        total_duration_ms = self.duration * 1000
        timestamps = []

        if total_requests == 0:
            return timestamps

        if self.pattern == "uniform":
            # Distribute requests evenly across the duration
            interval = total_duration_ms / total_requests
            current_time = 0.0
            for _ in range(total_requests):
                timestamp = int(round(current_time))
                timestamp = min(timestamp, total_duration_ms - 1)
                timestamps.append(timestamp)
                current_time += interval
        elif self.pattern == "poisson":
            # Exponential distribution for intervals
            rate_ms = self.rps / 1000
            intervals = np.random.exponential(1 / rate_ms, total_requests)
            current_time = 0.0
            for i in range(total_requests):
                timestamp = int(round(current_time))
                timestamps.append(timestamp)
                current_time += intervals[i]
        elif self.pattern == "random":
            timestamps = np.random.randint(
                0, total_duration_ms, size=total_requests
            ).tolist()
        else:
            raise ValueError(f"Unknown pattern: {self.pattern}")

        return sorted(timestamps)


async def send_request(session, url, query, request_id):
    payload = {"query": query, "k": 2}
    start_time = time.time()
    try:
        async with session.post(url, json=payload) as response:
            response_text = await response.text()
            latency = time.time() - start_time
            return (request_id, latency, "success", response_text)
    except Exception as e:
        return (request_id, time.time() - start_time, f"error: {str(e)}", None)


async def load_test(url, pattern_config, test_queries):
    # Initialize trace generator
    generator = SyntheticTraceGenerator(
        rps=pattern_config["rps"],
        pattern=pattern_config["pattern"],
        duration=pattern_config["duration"],
        seed=pattern_config.get("seed")
    )

    timestamps_ms = generator.generate()
    if not timestamps_ms:
        return []

    async with ClientSession() as session:
        tasks = []
        test_start = time.time()
        
        for i, ts_ms in enumerate(timestamps_ms):
            # Schedule request at precise timestamp
            await asyncio.sleep(ts_ms/1000 - (time.time() - test_start))
            
            query = test_queries[i % len(test_queries)]
            tasks.append(
                send_request(session, url, query, i)
            )

        return await asyncio.gather(*tasks)


def analyze_results(results):
    latencies = [r[1] for r in results if r[2] == "success"]
    errors = [r for r in results if r[2] != "success"]
    error_rate = len(errors) / len(results) if results else 0
    
    print(f"\n{'='*40}")
    print(f"Total requests: {len(results)}")
    print(f"Successful: {len(latencies)}")
    print(f"Error rate: {error_rate:.2%}")
    
    if latencies:
        print("\nLatency metrics (seconds):")
        print(f"Average: {np.mean(latencies):.3f}")
        print(f"Median: {np.median(latencies):.3f}")
        print(f"95th percentile: {np.percentile(latencies, 95):.3f}")
        print(f"99th percentile: {np.percentile(latencies, 99):.3f}")
        print(f"Throughput: {len(latencies)/sum(latencies):.2f} req/s")
    
    # Bottleneck analysis
    print("\nBottleneck Indicators:")
    if latencies:
        if np.percentile(latencies, 95) > 2 * np.median(latencies):
            print("- High latency variance suggests queue buildup")
        if error_rate > 0.1:
            print("- High error rate indicates system overload")
        target_throughput = pattern_config["rps"]
        achieved_throughput = len(latencies)/sum(latencies)
        if achieved_throughput < target_throughput * 0.8:
            print(f"- Throughput ({achieved_throughput:.1f} req/s) below 80% of target ({target_throughput} req/s)")
    
    # Capacity warning
    if error_rate > 0.05:
        print("\n[WARNING] System capacity likely exceeded (error rate >5%)")
    
    # CSV reporting
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"load_test_{timestamp}.csv"
    with open(filename, "w", encoding="utf-8") as f:
        f.write("metric,value\n")
        f.write(f"total_requests,{len(results)}\n")
        f.write(f"success_rate,{(1 - error_rate):.4f}\n")
        f.write(f"avg_latency,{np.mean(latencies) if latencies else 0:.4f}\n")
        f.write(f"p95_latency,{np.percentile(latencies, 95) if latencies else 0:.4f}\n")
        f.write(f"p99_latency,{np.percentile(latencies, 99) if latencies else 0:.4f}\n")
        f.write(f"throughput,{len(latencies)/sum(latencies) if latencies else 0:.2f}\n")
        f.write(f"error_rate,{error_rate:.4f}\n")
    print(f"\nReport saved to {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RAG Service Load Tester')
    parser.add_argument('--url', type=str, default='http://localhost:8000/rag')
    parser.add_argument('--rps', type=int, default=100, help='Requests per second')
    parser.add_argument('--duration', type=int, default=10, help='Test duration in seconds')
    parser.add_argument('--pattern', choices=['uniform', 'poisson', 'random'], default='poisson')
    parser.add_argument('--seed', type=int, help='Random seed for reproducibility')

    args = parser.parse_args()
    
    # Sample queries
    test_queries = [
        "What are cats?",
        "Tell me about dogs",
        "How do hummingbirds fly?",
        "Describe pets",
        "Explain animal locomotion"
    ]
    
    pattern_config = {
        "rps": args.rps,
        "duration": args.duration,
        "pattern": args.pattern,
        "seed": args.seed,
    }

    print(f"Starting load test with pattern: {pattern_config}")
    start_time = time.time()
    
    results = asyncio.run(load_test(
        args.url,
        pattern_config,
        test_queries
    ))
    
    total_time = time.time() - start_time
    print(f"\nTotal test time: {total_time:.2f} seconds")
    analyze_results(results)
