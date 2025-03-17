import argparse
import asyncio
import time
import numpy as np
from aiohttp import ClientSession

async def send_request(session, url, query, request_id):
    payload = {"query": query, "k": 2}
    start_time = time.time()
    try:
        async with session.post(url, json=payload) as response:
            await response.text()
            latency = time.time() - start_time
            return (request_id, latency, "success")
    except Exception as e:
        return (request_id, time.time() - start_time, f"error: {str(e)}")

async def load_test(url, num_requests, concurrency, queries):
    async with ClientSession() as session:
        tasks = []
        for i in range(num_requests):
            query = queries[i % len(queries)]  # Cycle through sample queries
            tasks.append(send_request(session, url, query, i))
        
        results = []
        for chunk in chunks(tasks, concurrency):
            start = time.time()
            chunk_results = await asyncio.gather(*chunk)
            batch_time = time.time() - start
            results.extend(chunk_results)
            print(f"Processed {len(chunk)} requests in {batch_time:.2f}s")

        return results

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def analyze_results(results):
    latencies = [r[1] for r in results if r[2] == "success"]
    errors = [r for r in results if r[2] != "success"]
    
    print(f"\n{'='*40}")
    print(f"Total requests: {len(results)}")
    print(f"Successful: {len(latencies)}")
    print(f"Errors: {len(errors)}")
    
    if latencies:
        print("\nLatency metrics (seconds):")
        print(f"Average: {np.mean(latencies):.3f}")
        print(f"Median: {np.median(latencies):.3f}")
        print(f"95th percentile: {np.percentile(latencies, 95):.3f}")
        print(f"Throughput: {len(latencies)/sum(latencies):.2f} req/s")
    
    if errors:
        print("\nFirst 3 errors:")
        for e in errors[:3]:
            print(f"Request {e[0]}: {e[2]}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RAG Service Load Tester')
    parser.add_argument('--url', type=str, default='http://localhost:8000/rag')
    parser.add_argument('--num-requests', type=int, default=100)
    parser.add_argument('--concurrency', type=int, default=10)
    
    args = parser.parse_args()
    
    # Sample queries
    test_queries = [
        "What are cats?",
        "Tell me about dogs",
        "How do hummingbirds fly?",
        "Describe pets",
        "Explain animal locomotion"
    ]
    
    print(f"Starting load test with {args.num_requests} requests ({args.concurrency} concurrent)")
    start_time = time.time()
    
    results = asyncio.run(load_test(
        args.url,
        args.num_requests,
        args.concurrency,
        test_queries
    ))
    
    total_time = time.time() - start_time
    print(f"\nTotal test time: {total_time:.2f} seconds")
    analyze_results(results)