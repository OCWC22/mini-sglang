# Mini-SGLang: Complete End-to-End Execution Visualization on Blackwell B200

This document provides a comprehensive visual walkthrough of how a prompt flows through Mini-SGLang, from user input to final response, mapped onto NVIDIA Blackwell B200 GPU hardware.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Our Concrete Example](#2-our-concrete-example)
3. [Phase 1: Request Arrival & Tokenization](#3-phase-1-request-arrival--tokenization)
4. [Phase 2: Scheduling & Memory Allocation](#4-phase-2-scheduling--memory-allocation)
5. [Phase 3: Prefill Execution on GPU](#5-phase-3-prefill-execution-on-gpu)
6. [Phase 4: Decode Loop](#6-phase-4-decode-loop)
7. [Phase 5: Detokenization & Response](#7-phase-5-detokenization--response)
8. [Blackwell B200 Hardware Mapping](#8-blackwell-b200-hardware-mapping)
9. [Latency & Bottleneck Analysis](#9-latency--bottleneck-analysis)
10. [Optimization Opportunities](#10-optimization-opportunities)

---

## 1. System Overview

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              USER REQUEST                                        │
│                    "What is the capital of France?"                              │
└─────────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           MINI-SGLANG SYSTEM                                     │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐       │
│  │  API Server │───▶│  Tokenizer  │───▶│  Scheduler  │───▶│ Detokenizer │       │
│  │  (FastAPI)  │    │   Worker    │    │   (Rank 0)  │    │   Worker    │       │
│  │             │    │             │    │      │      │    │             │       │
│  │   CPU       │    │    CPU      │    │   CPU+GPU   │    │    CPU      │       │
│  └─────────────┘    └─────────────┘    └──────┼──────┘    └─────────────┘       │
│         ▲                                     │                   │              │
│         │                                     ▼                   │              │
│         │                            ┌───────────────┐            │              │
│         │                            │    Engine     │            │              │
│         │                            │  (B200 GPU)   │            │              │
│         │                            │               │            │              │
│         │                            │ ┌───────────┐ │            │              │
│         │                            │ │  Model    │ │            │              │
│         │                            │ │  Weights  │ │            │              │
│         │                            │ ├───────────┤ │            │              │
│         │                            │ │ KV Cache  │ │            │              │
│         │                            │ ├───────────┤ │            │              │
│         │                            │ │  Radix    │ │            │              │
│         │                            │ │  Cache    │ │            │              │
│         │                            │ └───────────┘ │            │              │
│         │                            └───────────────┘            │              │
│         │                                                         │              │
│         └─────────────────────────────────────────────────────────┘              │
│                              Streaming Response                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Process Communication Map

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                        INTER-PROCESS COMMUNICATION                                │
│                                                                                   │
│   ┌─────────┐         ZMQ (IPC)          ┌─────────┐                             │
│   │   API   │ ◀──────────────────────────▶│Tokenizer│                             │
│   │ Server  │    zmq_tokenizer_addr       │ Worker  │                             │
│   │         │    zmq_frontend_addr        │         │                             │
│   └────┬────┘                             └────┬────┘                             │
│        │                                       │                                  │
│        │ HTTP/WebSocket                        │ ZMQ (IPC)                        │
│        │ (User)                                │ zmq_backend_addr                 │
│        │                                       │                                  │
│        ▼                                       ▼                                  │
│   ┌─────────┐                             ┌─────────┐      NCCL       ┌─────────┐│
│   │  User   │                             │Scheduler│◀───────────────▶│Scheduler││
│   │ Client  │                             │ Rank 0  │   (GPU-GPU)     │ Rank N  ││
│   └─────────┘                             └────┬────┘                 └────┬────┘│
│                                                │                           │     │
│                                                │ CUDA API                  │     │
│                                                ▼                           ▼     │
│                                           ┌─────────┐                 ┌─────────┐│
│                                           │  GPU 0  │                 │  GPU N  ││
│                                           │  B200   │                 │  B200   ││
│                                           └─────────┘                 └─────────┘│
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Our Concrete Example

We'll trace this exact request through the entire system:

```python
# User Request (via curl or OpenAI client)
{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [{"role": "user", "content": "What is the capital of France?"}],
    "max_tokens": 50,
    "stream": True
}
```

**Token Breakdown:**
```
Input (after chat template):
┌─────────────────────────────────────────────────────────────────────────────────┐
│ <|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n │
│     [151644] [872] [198] [3838] [374] [279] [6722] [315] [9625] [30] ...        │
│                                                                                  │
│ Total: ~15 tokens                                                                │
└─────────────────────────────────────────────────────────────────────────────────┘

Expected Output:
┌─────────────────────────────────────────────────────────────────────────────────┐
│ "The capital of France is Paris."                                                │
│                                                                                  │
│ ~10 tokens generated one at a time                                               │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Phase 1: Request Arrival & Tokenization

### Step-by-Step Flow

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         PHASE 1: REQUEST ARRIVAL                                 │
│                                                                                  │
│  TIME ──────────────────────────────────────────────────────────────────────▶   │
│                                                                                  │
│  t=0ms        t=0.1ms         t=0.5ms          t=1ms           t=1.5ms          │
│    │             │               │                │               │              │
│    ▼             ▼               ▼                ▼               ▼              │
│ ┌──────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐        │
│ │ HTTP │───▶│ FastAPI  │───▶│  Create  │───▶│   ZMQ    │───▶│Tokenizer │        │
│ │ POST │    │ Endpoint │    │TokenizeMsg│   │   Send   │    │  Recv    │        │
│ └──────┘    └──────────┘    └──────────┘    └──────────┘    └──────────┘        │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Code Path: API Server

```python
# python/minisgl/server/api_server.py

@app.post("/v1/chat/completions")
async def v1_completions(req: OpenAICompletionRequest):
    """
    Entry point for chat completions.
    
    Memory: ~1KB for request object
    CPU: ~0.1ms for parsing
    """
    state = get_global_state()  # FrontendManager singleton
    
    # Extract messages
    prompt = [msg.model_dump() for msg in req.messages]
    # prompt = [{"role": "user", "content": "What is the capital of France?"}]
    
    # Create unique user ID and tracking structures
    uid = state.new_user()  # uid = 0
    # state.ack_map[0] = []
    # state.event_map[0] = asyncio.Event()
    
    # Send to tokenizer via ZMQ
    await state.send_one(
        TokenizeMsg(
            uid=0,
            text=prompt,
            sampling_params=SamplingParams(max_tokens=50),
        )
    )
    
    # Return streaming response immediately
    return StreamingResponse(
        state.stream_chat_completions(uid),
        media_type="text/event-stream",
    )
```

### Data Structure: TokenizeMsg

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           TokenizeMsg Structure                                  │
│                                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │ {                                                                        │    │
│  │   "__type__": "TokenizeMsg",                                             │    │
│  │   "uid": 0,                                                              │    │
│  │   "text": [{"role": "user", "content": "What is the capital..."}],       │    │
│  │   "sampling_params": {                                                   │    │
│  │     "__type__": "SamplingParams",                                        │    │
│  │     "top_k": 1,                                                          │    │
│  │     "temperature": 0.0,                                                  │    │
│  │     "max_tokens": 50,                                                    │    │
│  │     "ignore_eos": false                                                  │    │
│  │   }                                                                      │    │
│  │ }                                                                        │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                  │
│  Serialized via msgpack: ~200 bytes                                              │
│  Transport: ZMQ IPC socket (ipc:///tmp/minisgl_4.pid=XXXX)                       │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Code Path: Tokenizer Worker

```python
# python/minisgl/tokenizer/server.py

@torch.inference_mode()
def tokenize_worker(...):
    """
    Runs in separate process.
    Handles both tokenization (text→tokens) and detokenization (tokens→text).
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tokenize_manager = TokenizeManager(tokenizer)
    
    while True:
        # Receive message from API server
        pending_msg = recv_listener.get()  # Blocking
        
        # Process tokenization requests
        tokenize_msg = [m for m in pending_msg if isinstance(m, TokenizeMsg)]
        
        if len(tokenize_msg) > 0:
            # Convert text to tokens
            tensors = tokenize_manager.tokenize(tokenize_msg)
            
            # Send to scheduler
            send_backend.put(BatchBackendMsg(data=[
                UserMsg(uid=msg.uid, input_ids=t, sampling_params=msg.sampling_params)
                for msg, t in zip(tokenize_msg, tensors)
            ]))
```

### Tokenization Process

```python
# python/minisgl/tokenizer/tokenize.py

class TokenizeManager:
    def tokenize(self, msgs: List[TokenizeMsg]) -> List[torch.Tensor]:
        for msg in msgs:
            # Apply chat template
            prompt = self.tokenizer.apply_chat_template(
                msg.text,  # [{"role": "user", "content": "What is..."}]
                tokenize=False,
                add_generation_prompt=True,
            )
            # Result: "<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n"
            
            # Tokenize
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
            # Result: tensor([151644, 872, 198, 3838, 374, 279, 6722, 315, 9625, 30, 151645, 198, 151644, 77091, 198])
            
            return input_ids.view(-1).to(torch.int32)
```

### Tokenization Memory Layout

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        TOKENIZATION MEMORY (CPU)                                 │
│                                                                                  │
│  Input String (Python heap):                                                     │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │ "What is the capital of France?" (32 bytes UTF-8)                        │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                      │                                           │
│                                      ▼ tokenizer.encode()                        │
│                                                                                  │
│  Token IDs (torch.Tensor, CPU, int32):                                           │
│  ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐
│  │151644│ 872 │ 198 │3838 │ 374 │ 279 │6722 │ 315 │9625 │  30 │151645│ 198 │151644│77091│ 198 │
│  └─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘
│    │                                                                              │
│    └── 15 tokens × 4 bytes = 60 bytes                                             │
│                                                                                  │
│  Total CPU memory for this request: ~1KB (including Python objects)              │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Phase 2: Scheduling & Memory Allocation

### Scheduler Receives Request

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                      PHASE 2: SCHEDULING & ALLOCATION                            │
│                                                                                  │
│  TIME ──────────────────────────────────────────────────────────────────────▶   │
│                                                                                  │
│  t=1.5ms      t=2ms           t=2.5ms          t=3ms           t=3.5ms          │
│    │             │               │                │               │              │
│    ▼             ▼               ▼                ▼               ▼              │
│ ┌──────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐        │
│ │ ZMQ  │───▶│ Process  │───▶│  Radix   │───▶│ Allocate │───▶│ Prepare  │        │
│ │ Recv │    │  Msg     │    │  Match   │    │ KV Cache │    │ Metadata │        │
│ └──────┘    └──────────┘    └──────────┘    └──────────┘    └──────────┘        │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Code Path: Scheduler Main Loop

```python
# python/minisgl/scheduler/scheduler.py

class Scheduler(SchedulerIOMixin):
    def overlap_loop(self, last_data: ForwardData | None) -> ForwardData | None:
        """
        Main scheduling loop with overlap optimization.
        
        Key insight: Process last batch's results while scheduling next batch.
        This hides CPU latency behind GPU compute.
        """
        
        # 1. Receive new messages (non-blocking if we have work)
        blocking = not (last_data or self.prefill_manager.runnable or self.decode_manager.runnable)
        for msg in self.receive_msg(blocking=blocking):
            self._process_one_msg(msg)
        
        # 2. Schedule next batch
        forward_input = self._schedule_next_batch()
        
        # 3. Execute on GPU (in engine's stream)
        ongoing_data = None
        if forward_input is not None:
            with self.engine_stream_ctx:
                self.engine.stream.wait_stream(self.stream)
                ongoing_data = (forward_input, self._forward(forward_input))
        
        # 4. Process results from LAST batch (overlap!)
        self._process_last_data(last_data, ongoing_data)
        
        return ongoing_data
```

### Overlap Scheduling Visualization

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         OVERLAP SCHEDULING TIMELINE                              │
│                                                                                  │
│  Without Overlap:                                                                │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │ CPU: [Schedule B1][        ][Schedule B2][        ][Schedule B3]        │    │
│  │ GPU: [           ][Compute1][           ][Compute2][           ][Comp3] │    │
│  │                                                                          │    │
│  │ Total time: CPU + GPU + CPU + GPU + ...                                  │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                  │
│  With Overlap:                                                                   │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │ CPU: [Schedule B1][Process B0 + Schedule B2][Process B1 + Schedule B3]  │    │
│  │ GPU: [           ][Compute B1              ][Compute B2              ]  │    │
│  │                   ▲                         ▲                            │    │
│  │                   └─ CPU work hidden ───────┘                            │    │
│  │                                                                          │    │
│  │ Total time: max(CPU, GPU) per iteration                                  │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                  │
│  Savings: ~1ms per batch (CPU scheduling overhead hidden)                        │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Radix Cache Lookup

```python
# python/minisgl/kvcache/radix_manager.py

class RadixCacheManager(BaseCacheManager):
    def match_prefix(self, input_ids: torch.Tensor) -> Tuple[RadixCacheHandle, torch.Tensor]:
        """
        Walk the radix tree to find longest matching prefix.
        
        For our example (first request): No match, returns empty handle.
        For subsequent requests with same system prompt: Returns cached KV indices.
        """
        node, prefix_len = self._walk(input_ids)
        
        if prefix_len == 0:
            return RadixCacheHandle(0, node), self.empty_tensor
        
        # Collect cached indices by walking back up the tree
        value_list = []
        while not node.is_root():
            value_list.append(node.value)
            node = node.parent
        value_list.reverse()
        
        return RadixCacheHandle(prefix_len, node), torch.cat(value_list)
```

### Radix Tree Structure

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           RADIX TREE STRUCTURE                                   │
│                                                                                  │
│  Initial State (empty):                                                          │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │                              ROOT                                        │    │
│  │                           (ref_count=1)                                  │    │
│  │                                │                                         │    │
│  │                               (empty)                                    │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                  │
│  After first request completes:                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │                              ROOT                                        │    │
│  │                           (ref_count=1)                                  │    │
│  │                                │                                         │    │
│  │                                ▼                                         │    │
│  │                    ┌───────────────────────┐                             │    │
│  │                    │ key: [151644, 872...] │                             │    │
│  │                    │ value: [0, 1, 2, ...] │ ◀── KV cache page indices   │    │
│  │                    │ ref_count: 0          │                             │    │
│  │                    │ timestamp: t1         │                             │    │
│  │                    └───────────────────────┘                             │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                  │
│  After second request with same prefix:                                          │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │                              ROOT                                        │    │
│  │                                │                                         │    │
│  │                                ▼                                         │    │
│  │                    ┌───────────────────────┐                             │    │
│  │                    │ key: [151644, 872...] │                             │    │
│  │                    │ value: [0, 1, 2, ...] │ ◀── REUSED! No recompute    │    │
│  │                    │ ref_count: 1          │ ◀── Locked during use       │    │
│  │                    └───────────────────────┘                             │    │
│  │                          │           │                                   │    │
│  │                          ▼           ▼                                   │    │
│  │                    ┌─────────┐ ┌─────────┐                               │    │
│  │                    │ "Paris" │ │ "Lyon"  │  ◀── Different continuations  │    │
│  │                    └─────────┘ └─────────┘                               │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### KV Cache Allocation

```python
# python/minisgl/scheduler/prefill.py

@dataclass
class PrefillAdder:
    def _try_allocate_one(self, req: PendingReq) -> Tuple[BaseCacheHandle, int] | None:
        """
        Allocate resources for a prefill request.
        
        Steps:
        1. Check radix cache for prefix match
        2. Estimate memory needed
        3. Lock cache handle (prevents eviction)
        4. Allocate table slot
        5. Copy cached KV indices to page table
        """
        # Check radix cache
        handle, match_indices = self.cache_manager.match_req(req)
        cached_len = handle.cached_len  # 0 for first request
        
        # Estimate memory: tokens to process + expected output
        extend_len = req.input_len - cached_len  # 15 - 0 = 15
        estimated_len = extend_len + req.output_len  # 15 + 50 = 65
        
        # Check available cache space
        if estimated_len > self.cache_manager.available_size:
            return None  # Can't fit, will be queued
        
        # Lock the handle (prevents eviction during use)
        self.cache_manager.lock(handle)
        
        # Allocate table slot (index into page_table)
        table_idx = self.table_manager.allocate()  # Returns 0
        
        return handle, table_idx
```

### Memory Allocation Visualization

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        KV CACHE MEMORY ALLOCATION                                │
│                                                                                  │
│  Page Pool (GPU HBM):                                                            │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │ Page 0 │ Page 1 │ Page 2 │ ... │ Page 14 │ Page 15 │ ... │ Page N │     │    │
│  │ (free) │ (free) │ (free) │     │ (free)  │ (free)  │     │ (free) │     │    │
│  └────┬───┴────┬───┴────┬───┴─────┴────┬────┴────┬────┴─────┴────────┘     │    │
│       │        │        │              │         │                          │    │
│       └────────┴────────┴──────────────┴─────────┘                          │    │
│                         │                                                    │    │
│                         ▼ Allocate 15 pages for our request                  │    │
│                                                                              │    │
│  After allocation:                                                           │    │
│  ┌─────────────────────────────────────────────────────────────────────────┐│    │
│  │ Page 0 │ Page 1 │ Page 2 │ ... │ Page 14 │ Page 15 │ ... │ Page N │     ││    │
│  │ [REQ0] │ [REQ0] │ [REQ0] │     │ [REQ0]  │ (free)  │     │ (free) │     ││    │
│  └────────┴────────┴────────┴─────┴─────────┴─────────┴─────┴────────┘     ││    │
│                                                                              │    │
│  Page Table (maps request → pages):                                          │    │
│  ┌─────────────────────────────────────────────────────────────────────────┐│    │
│  │ Request 0: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, ...]      ││    │
│  │ Request 1: [_, _, _, _, _, _, _, _, _, _, __, __, __, __, __, ...]      ││    │
│  │ ...                                                                      ││    │
│  └─────────────────────────────────────────────────────────────────────────┘│    │
│                                                                              │    │
│  Each page stores K and V for one token position:                            │    │
│  ┌─────────────────────────────────────────────────────────────────────────┐│    │
│  │ Page Layout (for Qwen3-0.6B):                                            ││    │
│  │   K: [num_kv_heads=2, head_dim=64] × dtype=bf16 = 256 bytes              ││    │
│  │   V: [num_kv_heads=2, head_dim=64] × dtype=bf16 = 256 bytes              ││    │
│  │   Total per page: 512 bytes                                              ││    │
│  │   × 28 layers = 14,336 bytes per token position                          ││    │
│  └─────────────────────────────────────────────────────────────────────────┘│    │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Prepare Batch Metadata

```python
# python/minisgl/scheduler/scheduler.py

def _prepare_batch(self, batch: Batch) -> ForwardInput:
    """
    Prepare all metadata needed for GPU execution.
    """
    # Allocate output locations in KV cache
    needed_size = sum(r.extend_len for r in batch.reqs)  # 15 tokens
    batch.out_loc = self.cache_manager.allocate(needed_size)
    # out_loc = tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
    
    # Prepare 2D indices for loading token IDs
    load_indices = _make_2d_indices(
        self.token_pool,
        [(r.table_idx, r.cached_len, r.device_len) for r in batch.padded_reqs]
    )
    # load_indices = tensor([0, 1, 2, ..., 14])  # Positions to load from
    
    # Write output locations to page table
    self.page_table.view(-1)[load_indices] = batch.out_loc
    
    # Prepare attention metadata (backend-specific)