import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel, pipeline
from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
from threading import Thread
import queue
import time
from contextlib import asynccontextmanager

# Performance metrics tracking
class QueueMetrics:
    def __init__(self):
        self.total_requests = 0
        self.processing_times = []
        self.queue_sizes = []
    
    def record_request(self, processing_time, queue_size):
        self.total_requests += 1
        self.processing_times.append(processing_time)
        self.queue_sizes.append(queue_size)
    
    def get_stats(self):
        return {
            "total_processed": self.total_requests,
            "avg_processing_time": np.mean(self.processing_times) if self.processing_times else 0,
            "max_queue_size": max(self.queue_sizes) if self.queue_sizes else 0
        }

# Initialize metrics collection
metrics = QueueMetrics()

# Lifespan handler for FastAPI
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize application state
    app.state.request_queue = queue.Queue()
    app.state.stop_thread = False
    app.state.metrics = metrics
    
    # Background request processor
    def process_requests():
        while not app.state.stop_thread or not app.state.request_queue.empty():
            try:
                # Get single request from queue
                item = app.state.request_queue.get(timeout=0.01)
                payload, result_queue = item
                start_time = time.time()
                
                # Process request through existing RAG pipeline
                result = rag_pipeline(payload.query, payload.k)
                result_queue.put(result)
                
                # Record performance metrics
                processing_time = time.time() - start_time
                app.state.metrics.record_request(
                    processing_time=processing_time,
                    queue_size=app.state.request_queue.qsize()
                )
                
                app.state.request_queue.task_done()
                
            except queue.Empty:
                continue

    # Start processing thread
    processing_thread = Thread(target=process_requests, daemon=True)
    processing_thread.start()
    app.state.processing_thread = processing_thread
    
    yield  # App is running
    
    # Print final metrics on shutdown
    print("\n=== Queue Processing Metrics ===")
    stats = app.state.metrics.get_stats()
    print(f"Total requests processed: {stats['total_processed']}")
    print(f"Average processing time: {stats['avg_processing_time']:.2f}s")
    print(f"Maximum queue size: {stats['max_queue_size']}")

    # Cleanup on shutdown
    app.state.stop_thread = True
    processing_thread.join(timeout=5)

app = FastAPI(lifespan=lifespan)

# Example documents in memory
documents = [
    "Cats are small furry carnivores that are often kept as pets.",
    "Dogs are domesticated mammals, not natural wild animals.",
    "Hummingbirds can hover in mid-air by rapidly flapping their wings."
]

# 1. Load embedding model
EMBED_MODEL_NAME = "intfloat/multilingual-e5-large-instruct"
embed_tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_NAME)
embed_model = AutoModel.from_pretrained(EMBED_MODEL_NAME)

# Basic Chat LLM 
chat_pipeline = pipeline("text-generation", model="facebook/opt-125m")
# Note: try this 1.5B model if you got enough GPU memory
# chat_pipeline = pipeline("text-generation", model="Qwen/Qwen2.5-1.5B-Instruct")

## Hints:

### Step 3.1:
# 1. Initialize a request queue
# 2. Initialize a background thread to process the request (via calling the rag_pipeline function)
# 3. Modify the predict function to put the request in the queue, instead of processing it immediately

### Step 3.2:
# 1. Take up to MAX_BATCH_SIZE requests from the queue or wait until MAX_WAITING_TIME
# 2. Process the batched requests

def get_embedding(text: str) -> np.ndarray:
    """Compute a simple average-pool embedding."""
    inputs = embed_tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = embed_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).cpu().numpy()

# Precompute document embeddings
doc_embeddings = np.vstack([get_embedding(doc) for doc in documents])

def retrieve_top_k(query_emb: np.ndarray, k: int = 2) -> list:
    """Retrieve top-k docs via L2 distance."""
    # Convert numpy arrays to PyTorch tensors
    A_tensor = torch.tensor(doc_embeddings, dtype=torch.float32)
    X_tensor = torch.tensor(query_emb, dtype=torch.float32)

    # Compute L2 distances (shape: [num_docs])
    distances = torch.norm(A_tensor - X_tensor.squeeze(0), dim=1)
    
    # Get indices of smallest k distances
    top_k_indices = torch.argsort(distances)[:k].numpy()
    
    return [documents[i] for i in top_k_indices]

def rag_pipeline(query: str, k: int = 2) -> str:
    # Step 1: Input embedding
    query_emb = get_embedding(query)
    
    # Step 2: Retrieval
    retrieved_docs = retrieve_top_k(query_emb, k)
    
    # Construct the prompt from query + retrieved docs
    context = "\n".join(retrieved_docs)
    prompt = f"Question: {query}\nContext:\n{context}\nAnswer:"
    
    # Step 3: LLM Output
    generated = chat_pipeline(prompt, max_length=100, do_sample=True)[0]["generated_text"]
    return generated

# Define request model
class QueryRequest(BaseModel):
    query: str
    k: int = 2

@app.post("/rag")
def predict(payload: QueryRequest):
    result_queue = queue.Queue(maxsize=1)
    app.state.request_queue.put((payload, result_queue))
    result = result_queue.get()

    # Wait for processing
    return {
        "query": payload.query,
        "result": result
        }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)