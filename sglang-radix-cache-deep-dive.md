# SGLang RadixAttention: A Code-First Deep Dive

*Why SGLang exists, what problem radix cache solves, and how it's actually implemented*

---

## Core Questions Answered First

### 1) What inference problem does SGLang solve that vLLM/TensorRT do not?

**The Problem: Repeated Prefill Computation**

When you serve an LLM, most requests share common prefixes:
- System prompts: "You are a helpful AI assistant..."
- Few-shot examples: "Here are some examples..."
- Document context: The same document queried multiple times
- Agent tool descriptions: Same tools listed for every agent call

**Without SGLang:**
```
Request 1: [System: 500 tokens] + [User: 50 tokens] → Compute 550 tokens
Request 2: [System: 500 tokens] + [User: 60 tokens] → Compute 560 tokens (500 WASTED!)
Request 3: [System: 500 tokens] + [User: 40 tokens] → Compute 540 tokens (500 WASTED!)
```

**vLLM's PagedAttention** solves memory fragmentation but does NOT solve this compute waste. It still recomputes the KV cache for the system prompt every time.

**SGLang's RadixAttention** automatically detects and reuses the KV cache for shared prefixes:
```
Request 1: Compute 550 tokens, CACHE [System: 500 tokens]
Request 2: REUSE 500 tokens, Compute 60 tokens only
Request 3: REUSE 500 tokens, Compute 40 tokens only
```

**The key insight:** SGLang caches at the semantic level (token prefixes), not the hardware level (memory pages).

---

### 2) Why are shared prompts and agent-style workloads fundamentally different?

**Traditional Serving Assumption:** Each request is independent.

**Reality in Production:**

| Workload | Shared Prefix Pattern | Prefix Overlap |
|----------|----------------------|----------------|
| **Chatbot** | Same system prompt for all users | 90%+ |
| **Multi-turn Chat** | Full history up to current turn | 80-95% |
| **Agent/Tool Calls** | Tool descriptions + memory | 70-90% |
| **RAG** | Retrieved documents + instructions | 50-80% |
| **Batch Processing** | Same template, different inputs | Variable |

**Why this matters:** If 90% of your tokens are repeated, you're paying 10x more compute than necessary.

---

### 3) Why is recomputing prefill for repeated prefixes wasteful, and why paging alone does not solve it?

**The Compute Cost:**

```
Prefill Cost = O(seq_len² × num_layers × hidden_dim)

For Llama-7B (32 layers, 4096 hidden):
  500 token prefix ≈ 500² × 32 × 4096 = 32 billion multiply-adds
  
  1000 requests with same prefix:
    Without caching: 32 trillion operations
    With caching:    32 billion operations (1000x savings!)
```

**Why Paging (vLLM) Doesn't Help:**

```
┌─────────────────────────────────────────────────────────────────────┐
│                    vLLM PagedAttention                               │
│                                                                      │
│  Memory Layout:                                                      │
│  ┌────┐┌────┐┌────┐┌────┐┌────┐┌────┐                               │
│  │ P0 ││ P1 ││ P2 ││ P3 ││ P4 ││ P5 │  (Physical Pages)            │
│  └────┘└────┘└────┘└────┘└────┘└────┘                               │
│                                                                      │
│  Request A: P0 → P1 → P2  (System prompt tokens 0-47)               │
│  Request B: P3 → P4 → P5  (SAME system prompt, DIFFERENT pages!)    │
│                                                                      │
│  Problem: Each request RECOMPUTES the same KV values                 │
│           and stores them in SEPARATE pages.                         │
│                                                                      │
│  Paging solves: Memory fragmentation (variable sequence lengths)     │
│  Paging does NOT solve: Redundant computation for shared prefixes    │
└─────────────────────────────────────────────────────────────────────┘
```

**What vLLM's Automatic Prefix Caching (APC) adds:**

vLLM later added APC, which uses block-level hashing:
```
Block 0 (tokens 0-15): hash → check if cached
Block 1 (tokens 16-31): hash → check if cached
...
```

**Limitation:** Only works at block boundaries (16 tokens). A 37-token prefix caches 32 tokens, recomputes 5.

---

### 4) What is a radix-tree KV cache, and why is token-level prefix reuse powerful?

**A Radix Tree (Patricia Trie) for Tokens:**

```
                              ROOT
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                ▼
      ┌───────────────┐ ┌───────────┐ ┌───────────────┐
      │"You are a     │ │"Translate"│ │"Summarize the"│
      │ helpful       │ │           │ │               │
      │ assistant."   │ │           │ │               │
      │               │ │           │ │               │
      │ KV: [0-99]    │ │ KV:[100-] │ │ KV: [200-]    │
      │ ref_count: 2  │ │ ref: 0    │ │ ref: 1        │
      └───────┬───────┘ └───────────┘ └───────────────┘
              │
      ┌───────┴───────┐
      ▼               ▼
┌───────────┐   ┌───────────┐
│"What is   │   │"Please    │
│ the       │   │ explain"  │
│ capital"  │   │           │
│           │   │           │
│KV:[100-]  │   │KV:[150-]  │
│ref: 1     │   │ref: 0     │
└───────────┘   └───────────┘
```

**Why Token-Level Beats Block-Level:**

| Prefix Length | Radix (token) | Block-Hash (16-token) | Radix Advantage |
|---------------|---------------|----------------------|-----------------|
| 37 tokens | 37 cached | 32 cached | 5 fewer computes |
| 100 tokens | 100 cached | 96 cached | 4 fewer computes |
| 1000 tokens | 1000 cached | 992 cached | 8 fewer computes |
| 17 tokens | 17 cached | 16 cached | 1 fewer compute |

**The real win:** Partial matches at ANY granularity.

---

### 5) When does SGLang win vs vLLM, and when does it lose?

| Scenario | SGLang Wins | vLLM Wins | Why |
|----------|-------------|-----------|-----|
| **Chatbot with system prompt** | ✅ | - | 90%+ prefix reuse |
| **Multi-turn conversations** | ✅ | - | Each turn builds on history |
| **Agent tool loops** | ✅ | - | Tool descriptions cached |
| **RAG with shared docs** | ✅ | - | Document context cached |
| **Batch of unrelated queries** | - | ✅ | No prefixes to share |
| **High request diversity** | - | ✅ | Cache thrashing |
| **Memory-constrained setup** | - | ✅ | vLLM's paging is more efficient |
| **Maximum raw throughput** | Tie | Tie | Both well-optimized |

**Decision Rule:**
- If `prefix_reuse_rate > 30%`: SGLang will likely help
- If `prefix_reuse_rate < 10%`: vLLM may be simpler to operate

---

### 6) What trade-offs does radix caching introduce?

| Trade-off | Impact | Mitigation |
|-----------|--------|------------|
| **Memory overhead** | Tree nodes consume RAM (~100 bytes/node) | <1% of KV cache typically |
| **CPU overhead** | Tree walk per request (~10-50μs) | Hidden by overlap scheduling |
| **Cache churn** | Low-overlap workloads thrash cache | Size cache appropriately |
| **Eviction latency** | LRU heap operations (~50-200μs) | Only when cache is full |
| **LoRA complexity** | Same tokens + different adapters = different KV | Must key on (adapter, tokens) |
| **Debugging** | Cache behavior is implicit | Instrument hit rate metrics |

---

## A) WHY SGLang Exists (CEO-Level)

### The Problem of Repeated Prompts and Agent Loops

**The Hidden Tax on LLM Serving:**

Every time your LLM processes "You are a helpful AI assistant...", it performs billions of floating-point operations. If 100 users have the same system prompt, you're paying that tax 100 times.

```
┌─────────────────────────────────────────────────────────────────────┐
│                    THE REPEATED PROMPT TAX                           │
│                                                                      │
│  Your Customer Support Bot:                                          │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │ System: "You are a customer support agent for Acme Corp.    │    │
│  │          You help users with billing, technical issues,     │    │
│  │          and product questions. Be helpful and concise.     │    │
│  │          Our products include: Widget Pro, Widget Lite..."  │    │
│  │                                                             │    │
│  │          [500 tokens of instructions and context]           │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  100,000 queries/day × 500 tokens = 50,000,000 redundant tokens/day │
│                                                                      │
│  At $0.01 per 1K tokens:                                            │
│    Without caching: $500/day × 365 = $182,500/year                  │
│    With caching:    $0.005/day × 365 = $1.83/year                   │
│                                                                      │
│  Savings: $182,498/year (on system prompts alone!)                  │
└─────────────────────────────────────────────────────────────────────┘
```

### Why Radix Caching Changes Cost, Latency, and Throughput

**Impact on Key Metrics:**

| Metric | Before Radix | After Radix | Improvement |
|--------|--------------|-------------|-------------|
| **Prefill Latency** | 200-500ms | 20-50ms | **10x faster** first token |
| **Cost per Query** | $0.01 | $0.001-0.003 | **3-10x cheaper** |
| **Throughput** | X req/s | 2-5X req/s | **2-5x more users per GPU** |
| **GPU Utilization** | 30-50% | 70-90% | **Better ROI on hardware** |

### Decision Table: Choose SGLang vs vLLM

| Your Situation | Choose | Reason |
|----------------|--------|--------|
| Building a chatbot/assistant | **SGLang** | High prefix reuse |
| Running agent workflows | **SGLang** | Tool descriptions cached |
| RAG with repeated documents | **SGLang** | Document context cached |
| Batch processing unrelated text | **vLLM** | No shared prefixes |
| Need widest model support | **vLLM** | Larger ecosystem |
| Memory-constrained deployment | **vLLM** | Better paging |
| Both patterns in one system | **Both** | Route by workload type |

---

## B) Foundational Concepts (Intern-Friendly)

### Prefill vs Decode Recap

```
┌─────────────────────────────────────────────────────────────────────┐
│                    LLM INFERENCE PHASES                              │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                      PREFILL PHASE                           │    │
│  │                                                              │    │
│  │  Input: "What is the capital of France?"                     │    │
│  │  Tokens: [What, is, the, capital, of, France, ?]             │    │
│  │                                                              │    │
│  │  For EACH token, at EACH layer:                              │    │
│  │    1. Compute Query, Key, Value projections                  │    │
│  │    2. Store K, V in KV cache                                 │    │
│  │    3. Compute attention output                               │    │
│  │                                                              │    │
│  │  Result: KV cache filled, logits for last position          │    │
│  │  Cost: O(n² × layers) - EXPENSIVE for long inputs           │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                              │                                       │
│                              ▼                                       │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                      DECODE PHASE                            │    │
│  │                                                              │    │
│  │  For each output token:                                      │    │
│  │    1. Take last generated token as input                     │    │
│  │    2. Compute Q, K, V for ONE token                          │    │
│  │    3. Attend to ALL cached K, V                              │    │
│  │    4. Sample next token                                      │    │
│  │    5. Append new K, V to cache                               │    │
│  │    6. Repeat until EOS                                       │    │
│  │                                                              │    │
│  │  Result: Generated tokens one by one                         │    │
│  │  Cost: O(n × layers) per token - memory-bound                │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  KEY INSIGHT: Prefill is EXPENSIVE and often REDUNDANT!             │
└─────────────────────────────────────────────────────────────────────┘
```

### What "Shared Prefix" Really Means at the Token Level

```
Request 1: "You are helpful. What is 2+2?"
Tokens:    [151644, 872, 527, 11190, 13, 3555, 374, 220, 17, 10, 17, 30]
            │       │    │    │      │   │     │    │   │   │   │   │
            └───────┴────┴────┴──────┴───┴─────┴────┘   │   │   │   │
                      SHARED PREFIX (8 tokens)          └───┴───┴───┘
                                                         UNIQUE (4)

Request 2: "You are helpful. What is 3+3?"
Tokens:    [151644, 872, 527, 11190, 13, 3555, 374, 220, 18, 10, 18, 30]
            │       │    │    │      │   │     │    │   │   │   │   │
            └───────┴────┴────┴──────┴───┴─────┴────┘   │   │   │   │
                      SHARED PREFIX (8 tokens)          └───┴───┴───┘
                                                         UNIQUE (4)

The SAME token IDs mean the SAME KV cache values!
→ Compute once, reuse for all matching requests
```

### Why Token-Level Reuse Beats Sequence-Level Reuse

| Approach | Granularity | Matches |
|----------|-------------|---------|
| **Exact match** | Full sequence | Only identical prompts |
| **Block hash** | 16 tokens | At block boundaries only |
| **Radix tree** | 1 token | ANY prefix length |

**Example:**
```
Prefix A: "You are a helpful assistant. Please answer:"  (10 tokens)
Prefix B: "You are a helpful assistant. Please explain:" (10 tokens)
                                                 ↑
                                        Differs at token 9

Exact match: MISS (different prompts)
Block hash:  MISS (block 0 matches, but incomplete)
Radix tree:  HIT for 8 tokens! Only recompute 2.
```

### What a Radix Tree Is (Conceptually)

**A compressed trie where:**
1. Each node stores a **sequence** of tokens (not just one)
2. Nodes are **split** only when prefixes diverge
3. Each node points to **KV cache indices** for its tokens
4. Reference counts prevent **eviction** of in-use nodes

```
┌─────────────────────────────────────────────────────────────────────┐
│                    RADIX TREE INTUITION                              │
│                                                                      │
│  Think of it like a DICTIONARY organized by WORD BEGINNINGS:         │
│                                                                      │
│  Words: "car", "cart", "card", "care", "cat", "cattle"               │
│                                                                      │
│  Trie (one letter per node):        Radix Tree (compressed):         │
│                                                                      │
│       c                                      c                        │
│       │                                      │                        │
│       a                                      a                        │
│      / \                                    / \                       │
│     r   t                               "r"   "t"                     │
│    /|\   \                              /|\    |                      │
│   t d e   t                           t d e  "tle"                    │
│           |                                                           │
│           l                                                           │
│           |                                                           │
│           e                                                           │
│                                                                       │
│  "car" is a prefix of "cart", "card", "care"                         │
│  → Store "car" ONCE, branch at divergence points                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## C) Repo Map (Engineer-Level, Code-Driven)

### Directory Structure (Mini-SGLang Implementation)

```
python/minisgl/
├── kvcache/
│   ├── __init__.py          # Factory: create_cache_manager()
│   ├── base.py              # Abstract interfaces
│   ├── radix_manager.py     # ⭐ CORE: RadixTreeNode, RadixCacheManager
│   ├── naive_manager.py     # Baseline: no caching
│   └── mha_pool.py          # KV tensor storage
│
├── scheduler/
│   ├── scheduler.py         # ⭐ Main scheduling loop
│   ├── prefill.py           # ⭐ PrefillAdder with cache lookup
│   ├── cache.py             # CacheManager wrapper
│   ├── decode.py            # DecodeManager
│   └── table.py             # Page table management
│
├── engine/
│   ├── engine.py            # Model execution
│   └── graph.py             # CUDA graph capture
│
├── attention/
│   ├── fi.py                # FlashInfer backend
│   └── fa3.py               # FlashAttention3 backend
│
├── kernel/
│   ├── radix.py             # ⭐ fast_compare_key() wrapper
│   └── csrc/src/radix.cpp   # ⭐ SIMD-optimized token comparison
│
└── core.py                  # Req, Batch, Context definitions
```

### 1) Radix Tree / Prefix Cache

**File:** `python/minisgl/kvcache/radix_manager.py`

**Class `RadixTreeNode`** - The tree node structure:

```python
class RadixTreeNode:
    counter: int = 0  # Global node ID counter

    def __init__(self, tic: int | None = None) -> None:
        self.children: Dict[int, RadixTreeNode] = {}  # first_token_id → child
        self._parent: RadixTreeNode | None = None
        self.ref_count: int = 0          # Active request count (prevents eviction)
        self.uuid = RadixTreeNode.counter  # Unique ID for debugging
        RadixTreeNode.counter += 1
        self.timestamp = tic or time.monotonic_ns()  # LRU timestamp

        # Set by set_key_value():
        self._key: torch.Tensor    # Token IDs this node represents
        self._value: torch.Tensor  # KV cache indices for these tokens
        self._length: int          # len(_key) == len(_value)
```

**Why each field exists:**
- `children`: Navigate to child nodes by first token ID
- `ref_count`: Prevent eviction while request is using this prefix
- `timestamp`: LRU eviction—oldest unused leaves evicted first
- `_key`: The token sequence this node covers
- `_value`: Indices into the GPU KV cache pool

**Class `RadixCacheManager`** - The main manager:

```python
class RadixCacheManager(BaseCacheManager):
    def __init__(self, device: torch.device):
        self.device = device
        self.empty_tensor = torch.empty(0, dtype=torch.int32, device=device)
        self.root_node = RadixTreeNode()
        self.root_node.ref_count = 1  # Root is always protected
        self.evictable_size = 0       # Tokens that CAN be evicted
        self.protected_size = 0       # Tokens in use (ref_count > 0)
```

### 2) Cache Lookup + Insertion

**Method `_walk()`** - Find longest matching prefix:

```python
def _walk(self, input_ids: torch.Tensor) -> Tuple[RadixTreeNode, int]:
    """
    Walk the tree to find the longest matching prefix.
    
    Returns:
        node: The deepest node we reached
        prefix_len: How many tokens matched
    """
    prefix_len = 0
    node = self.root_node
    tic = time.monotonic_ns()  # For timestamp update

    while prefix_len < len(input_ids):
        # Check if there's a child starting with the next token
        this_id = int(input_ids[prefix_len].item())
        if this_id not in node.children:
            return node, prefix_len  # No match, stop here

        # Move to child node
        node = node.children[this_id]

        # Compare this node's key with our input
        # Uses SIMD-optimized C++ kernel for speed
        match_len = node.get_match_len(input_ids[prefix_len:])
        prefix_len += match_len

        # If we didn't match the entire node's key, SPLIT the node
        if match_len != node.length:
            node = node._split_at(match_len)
            return node, prefix_len

        # Update timestamp for LRU (this node was accessed)
        node.timestamp = tic

    return node, prefix_len
```

**Method `match_prefix()`** - Get cached KV indices:

```python
def match_prefix(self, input_ids: torch.Tensor) -> Tuple[RadixCacheHandle, torch.Tensor]:
    """
    Find longest matching prefix and return KV cache indices.
    """
    node, prefix_len = self._walk(input_ids)
    
    if prefix_len == 0:
        # No match at all
        return RadixCacheHandle(cached_len=0, node=self.root_node), self.empty_tensor
    
    # Collect KV indices by walking BACK UP the tree
    value_list: List[torch.Tensor] = []
    walk_node = node
    while not walk_node.is_root():
        value_list.append(walk_node.value)  # KV cache indices for this node
        walk_node = walk_node.parent
    value_list.reverse()  # Root-to-leaf order
    
    return RadixCacheHandle(prefix_len, node), torch.cat(value_list)
```

**Method `insert_prefix()`** - Add new prefix after request completes:

```python
def insert_prefix(self, input_ids: torch.Tensor, indices: torch.Tensor) -> int:
    """
    After a request completes, add its prefix to the tree for future reuse.
    
    Returns:
        int: How many tokens were ALREADY cached (caller can free duplicates)
    """
    node, prefix_len = self._walk(input_ids)
    
    if prefix_len < len(input_ids):
        # Create new node for the unmatched suffix
        new_node = RadixTreeNode()
        new_node.set_key_value(
            input_ids[prefix_len:],   # Unmatched tokens
            indices[prefix_len:]      # Their KV cache indices
        )
        new_node.set_parent(node)
        self.evictable_size += new_node.length  # New tokens are evictable
    
    return prefix_len  # Caller frees indices[old_cached_len:prefix_len]
```

### 3) Fast Token Comparison (C++ Kernel)

**File:** `python/minisgl/kernel/csrc/src/radix.cpp`

```cpp
// Uses std::mismatch for SIMD-optimized comparison
auto fast_compare_key(const tvm::ffi::TensorView a,
                      const tvm::ffi::TensorView b) -> size_t {
  const auto common_len = std::min(a.size(0), b.size(0));
  
  if (a.dtype().bits == 64) {
    const auto a_ptr = static_cast<const int64_t*>(a.data_ptr());
    const auto b_ptr = static_cast<const int64_t*>(b.data_ptr());
    // std::mismatch is highly optimized by compilers (uses SIMD)
    const auto diff_pos = std::mismatch(a_ptr, a_ptr + common_len, b_ptr);
    return static_cast<size_t>(diff_pos.first - a_ptr);
  } else {
    const auto a_ptr = static_cast<const int32_t*>(a.data_ptr());
    const auto b_ptr = static_cast<const int32_t*>(b.data_ptr());
    const auto diff_pos = std::mismatch(a_ptr, a_ptr + common_len, b_ptr);
    return static_cast<size_t>(diff_pos.first - a_ptr);
  }
}
```

**Why this is fast:**
- `std::mismatch` is vectorized by modern compilers (AVX2/AVX-512)
- Operates on contiguous memory (cache-friendly)
- Early termination on first mismatch
- No memory allocation

### 4) Execution/Scheduling Logic

**File:** `python/minisgl/scheduler/scheduler.py`

**The main scheduling loop with overlap:**

```python
def overlap_loop(self, last_data: ForwardData | None) -> ForwardData | None:
    """
    Overlap CPU scheduling with GPU computation.
    
    Timeline:
    GPU: [Batch N-1 compute ][Batch N compute    ][Batch N+1 compute ]
    CPU:        [Process N-2][Schedule N         ][Process N-1       ]
                              ↑ Cache lookup happens here
    """
    # 1. Receive new requests (non-blocking if we have work)
    blocking = not (last_data or self.prefill_manager.runnable 
                    or self.decode_manager.runnable)
    for msg in self.receive_msg(blocking=blocking):
        self._process_one_msg(msg)

    # 2. Schedule next batch (INCLUDES RADIX CACHE LOOKUP)
    forward_input = self._schedule_next_batch()
    
    ongoing_data = None
    if forward_input is not None:
        # 3. Execute on GPU (in engine's separate stream)
        with self.engine_stream_ctx:
            self.engine.stream.wait_stream(self.stream)
            ongoing_data = (forward_input, self._forward(forward_input))

    # 4. Process results from LAST batch (overlap!)
    self._process_last_data(last_data, ongoing_data)
    
    return ongoing_data
```

**File:** `python/minisgl/scheduler/prefill.py`

**Where cache lookup happens:**

```python
class PrefillAdder:
    def _try_allocate_one(self, req: PendingReq) -> Tuple[BaseCacheHandle, int] | None:
        """Attempt to allocate resources for one prefill request."""
        
        if self.table_manager.available_size == 0:
            return None  # No table slots
        
        # ⭐ CACHE LOOKUP HAPPENS HERE
        handle, match_indices = self.cache_manager.match_req(req)
        cached_len = handle.cached_len
        
        # Calculate how much we actually need to compute
        extend_len = req.input_len - cached_len  # Only uncached tokens!
        estimated_len = extend_len + req.output_len
        
        # Check if we have enough cache space
        if estimated_len + self.reserved_size > self.cache_manager.available_size:
            return None  # Can't fit, queue for later
        
        # ⭐ LOCK HANDLE (prevent eviction during use)
        self.cache_manager.lock(handle)
        
        # Double-check after locking (race condition protection)
        if estimated_len + self.reserved_size > self.cache_manager.available_size:
            self.cache_manager.unlock(handle)
            return None
        
        # Allocate table slot
        table_idx = self.table_manager.allocate()
        
        # ⭐ COPY CACHED KV INDICES TO PAGE TABLE
        if cached_len > 0:
            device_ids = self.table_manager.token_pool[table_idx][:cached_len]
            page_entry = self.table_manager.page_table[table_idx][:cached_len]
            device_ids.copy_(req.input_ids[:cached_len].pin_memory(), non_blocking=True)
            page_entry.copy_(match_indices)  # Point to existing KV cache!
        
        return handle, table_idx
```

### 5) Model Execution + Kernel Calls

**File:** `python/minisgl/engine/engine.py`

```python
def forward_batch(self, batch: Batch, args: BatchSamplingArgs) -> ForwardOutput:
    """Execute a forward pass on a batch."""
    with self.ctx.forward_batch(batch):
        if self.graph_runner.can_use_cuda_graph(batch):
            logits = self.graph_runner.replay(batch)  # CUDA graph replay (fast)
        else:
            logits = self.model.forward()  # Full forward pass
    
    # Mark tokens as processed
    for req in batch.reqs:
        req.complete_one()
    
    # Sample next tokens
    next_tokens_gpu = self.sampler.sample(logits[: batch.size], args).to(torch.int32)
    next_tokens_cpu = next_tokens_gpu.to("cpu", non_blocking=True)
    copy_done_event = torch.cuda.Event()
    copy_done_event.record()
    
    return ForwardOutput(next_tokens_gpu, next_tokens_cpu, copy_done_event)
```

**File:** `python/minisgl/attention/fi.py` (FlashInfer backend)

```python
def forward(
    self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, 
    layer_id: int, batch: Batch
) -> torch.Tensor:
    """Execute attention with KV cache."""
    metadata = batch.attn_metadata
    
    # ⭐ STORE NEW K,V AT out_loc INDICES
    # Only the NEW tokens get stored; cached tokens are already there
    self.kvcache.store_kv(k, v, batch.out_loc, layer_id)
    
    # Run attention over ALL K,V (cached + new)
    kv_cache = (self.kvcache.k_cache(layer_id), self.kvcache.v_cache(layer_id))
    return metadata.wrapper.run(q=q, paged_kv_cache=kv_cache)
```

---

## D) One Request Walkthrough

### Trace: Request with Partial Prefix Match

**Setup:**
- Tree already contains: "You are a helpful assistant." (10 tokens) → KV indices [0-9]
- New request: "You are a helpful assistant. What is 2+2?" (15 tokens)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         REQUEST LIFECYCLE                                    │
│                                                                              │
│  STEP 1: Request Arrives                                                     │
│  ────────────────────────────────────────────────────────────────────────── │
│                                                                              │
│  UserMsg:                                                                    │
│    input_ids: [151644, 872, 527, 11190, 13, 3555, 374, 220, 17, 10, 17, 30,  │
│                18, 10, 18]                                                   │
│    uid: 42                                                                   │
│    sampling_params: SamplingParams(max_tokens=100)                          │
│                                                                              │
│  Scheduler._process_one_msg() → prefill_manager.add_one_req()               │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 2: Schedule Prefill (Cache Lookup Happens Here)                        │
│  ────────────────────────────────────────────────────────────────────────── │
│                                                                              │
│  PrefillAdder._try_allocate_one(req):                                        │
│                                                                              │
│    # 2a. Cache lookup                                                        │
│    handle, match_indices = cache_manager.match_req(req)                      │
│    #                                                                         │
│    # RadixCacheManager._walk() executes:                                     │
│    #   prefix_len = 0, node = ROOT                                           │
│    #   token 151644 → found child "You are a helpful assistant."             │
│    #   fast_compare_key([151644, 872, ...], node._key) → 10 tokens match    │
│    #   check children for token at position 10 (3555 = "What") → NOT FOUND  │
│    #   return (node, prefix_len=10)                                          │
│    #                                                                         │
│    # match_prefix() collects indices:                                        │
│    #   walk back up: node.value = [0,1,2,3,4,5,6,7,8,9]                      │
│    #   return (handle(cached_len=10), indices=[0-9])                         │
│                                                                              │
│    cached_len = 10                                                           │
│    match_indices = tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])                    │
│                                                                              │
│    # 2b. Calculate compute needed                                            │
│    extend_len = 15 - 10 = 5  # Only 5 tokens to compute!                     │
│                                                                              │
│    # 2c. Lock handle (prevents eviction)                                     │
│    cache_manager.lock(handle)                                                │
│    #   → node.ref_count incremented from 0 to 1                              │
│    #   → evictable_size decreased by 10                                      │
│                                                                              │
│    # 2d. Copy cached indices to page table                                   │
│    page_table[table_idx, 0:10] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]             │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 3: Prepare Batch                                                       │
│  ────────────────────────────────────────────────────────────────────────── │
│                                                                              │
│  Scheduler._prepare_batch():                                                 │
│                                                                              │
│    # Allocate KV slots for NEW tokens only                                   │
│    needed_size = sum(r.extend_len for r in batch.reqs) = 5                   │
│    batch.out_loc = cache_manager.allocate(5)  # e.g., [100, 101, 102, 103, 104]│
│                                                                              │
│    # Update page table with new indices                                      │
│    page_table[table_idx] now = [0,1,2,3,4,5,6,7,8,9,100,101,102,103,104]    │
│                                  └──── CACHED ────────┘└──── NEW ──────┘     │
│                                                                              │
│    # Prepare attention metadata                                              │
│    attn_backend.prepare_metadata(batch)                                      │
│                                                                              │
│  Req state:                                                                  │
│    cached_len = 10                                                           │
│    device_len = 15  (total tokens to attend to)                              │
│    extend_len = 5   (tokens to compute KV for)                               │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 4: GPU Forward Pass                                                    │
│  ────────────────────────────────────────────────────────────────────────── │
│                                                                              │
│  engine.forward_batch(batch):                                                │
│                                                                              │
│    # Load token IDs for the EXTEND portion only                              │
│    batch.input_ids = [3555, 374, 220, 17, 10]  # "What is 2+2"              │
│                       (tokens 10-14, the uncached part)                      │
│                                                                              │
│    # Forward pass through model                                              │
│    For each transformer layer:                                               │
│      # Compute Q, K, V for the 5 new tokens                                  │
│      qkv = linear(hidden_states)                                             │
│                                                                              │
│      # In attention layer:                                                   │
│      AttentionLayer.forward():                                               │
│        # Store NEW K,V at out_loc indices [100-104]                         │
│        kvcache.store_kv(k, v, batch.out_loc, layer_id)                      │
│                                                                              │
│        # Attention reads ALL 15 positions:                                   │
│        #   positions 0-9: from KV cache at indices [0-9] (CACHED!)          │
│        #   positions 10-14: from KV cache at indices [100-104] (just stored)│
│        output = flashinfer.run(q, kv_cache)                                 │
│                                                                              │
│    # Sample next token                                                       │
│    next_token = sampler.sample(logits[-1])  # → 6314 ("Paris" maybe)        │
│                                                                              │
│  SPEEDUP: Only 5 tokens computed instead of 15 = 3x faster prefill!         │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 5: Decode Loop (generate remaining tokens)                             │
│  ────────────────────────────────────────────────────────────────────────── │
│                                                                              │
│  For each decode step:                                                       │
│    # Single token forward pass                                               │
│    # Attends to all cached K,V (now 16 positions: 0-9, 100-104, + new)      │
│    next_token = model.forward(last_token)                                    │
│    # Store new K,V, sample, repeat...                                        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 6: Request Completes (Insert into Cache)                               │
│  ────────────────────────────────────────────────────────────────────────── │
│                                                                              │
│  Scheduler._process_last_data():                                             │
│                                                                              │
│    cache_manager.free_and_cache_finished_req(                                │
│        req.cache_handle,                                                     │
│        req.host_ids[:req.cached_len],    # Input token IDs                   │
│        page_table[req.table_idx, :req.cached_len]  # KV indices              │
│    )                                                                         │
│                                                                              │
│    # Inside RadixCacheManager.insert_prefix():                               │
│    #   Walk tree again, find that tokens 0-9 already cached                  │
│    #   Create NEW node for "What is 2+2?" with KV indices [100-104]         │
│    #   Tree now has:                                                         │
│    #     ROOT                                                                │
│    #      └── "You are a helpful assistant." [0-9]                          │
│    #           └── "What is 2+2?" [100-104]  ← NEW!                         │
│    #                                                                         │
│    # Unlock handle (decrement ref_count)                                     │
│    #   → node becomes evictable again if ref_count hits 0                    │
│                                                                              │
│  Future requests with "You are a helpful assistant. What is 2+2?"            │
│  will now match ALL 15 tokens!                                               │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## E) ASCII Diagrams

### Diagram 1: Inference WITHOUT Radix Cache (Repeated Prefill Waste)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    WITHOUT RADIX CACHE                                       │
│                                                                              │
│  Request 1: "You are helpful. What is 2+2?"                                 │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ PREFILL: ████████████████████████████████████████████ (15 tokens)      │ │
│  │          Compute K,V for ALL 15 tokens                                 │ │
│  │          Store in KV slots [0-14]                                      │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ DECODE: Generate response...                                           │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│  [Request completes, KV slots [0-14] FREED]                                 │
│                                                                              │
│  Request 2: "You are helpful. What is 3+3?"                                 │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ PREFILL: ████████████████████████████████████████████ (15 tokens)      │ │
│  │          RECOMPUTE K,V for "You are helpful. What is" AGAIN!          │ │
│  │          Store in KV slots [15-29]                                     │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ DECODE: Generate response...                                           │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  Request 3: "You are helpful. What is 4+4?"                                 │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ PREFILL: ████████████████████████████████████████████ (15 tokens)      │ │
│  │          RECOMPUTE K,V for "You are helpful. What is" AGAIN!          │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  TOTAL COMPUTE: 15 + 15 + 15 = 45 tokens                                    │
│  WASTED:        10 + 10 + 10 = 30 tokens (same prefix computed 3x!)         │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Diagram 2: Radix Tree Structure with Shared Token Prefixes

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         RADIX TREE STRUCTURE                                 │
│                                                                              │
│  After processing requests:                                                  │
│    1. "You are helpful. What is 2+2?"                                       │
│    2. "You are helpful. What is 3+3?"                                       │
│    3. "You are helpful. Please explain quantum physics"                     │
│    4. "Translate the following to French: Hello"                            │
│                                                                              │
│                                                                              │
│                              ┌────────────┐                                  │
│                              │    ROOT    │                                  │
│                              │ ref_cnt: 1 │ (always protected)               │
│                              └─────┬──────┘                                  │
│                                    │                                         │
│              ┌─────────────────────┴─────────────────────┐                  │
│              │                                           │                   │
│              ▼                                           ▼                   │
│    ┌─────────────────────────┐             ┌─────────────────────────┐      │
│    │ "You are helpful."      │             │ "Translate the          │      │
│    │                         │             │  following to           │      │
│    │ key: [151644,872,527,   │             │  French: Hello"         │      │
│    │       11190,13]         │             │                         │      │
│    │ value: [0,1,2,3,4]      │             │ key: [...]              │      │
│    │ ref_cnt: 0              │             │ value: [50,51,...,60]   │      │
│    │ timestamp: 1000         │             │ ref_cnt: 0              │      │
│    └───────────┬─────────────┘             └─────────────────────────┘      │
│                │                                                             │
│      ┌─────────┴─────────┐                                                  │
│      │                   │                                                   │
│      ▼                   ▼                                                   │
│ ┌───────────────┐  ┌───────────────┐                                        │
│ │ "What is"     │  │ "Please       │                                        │
│ │               │  │  explain      │                                        │
│ │ key:[3555,374]│  │  quantum      │                                        │
│ │ value:[5,6]   │  │  physics"     │                                        │
│ │ ref_cnt: 0    │  │               │                                        │
│ └───────┬───────┘  │ key:[...]     │                                        │
│         │          │ value:[20-35] │                                        │
│    ┌────┴────┐     │ ref_cnt: 0    │                                        │
│    │         │     └───────────────┘                                        │
│    ▼         ▼                                                              │
│ ┌──────┐ ┌──────┐                                                           │
│ │"2+2?"│ │"3+3?"│                                                           │
│ │      │ │      │                                                           │
│ │key:  │ │key:  │                                                           │
│ │[17,..│ │[18,..│                                                           │
│ │val:  │ │val:  │                                                           │
│ │[7-10]│ │[11-14│                                                           │
│ └──────┘ └──────┘                                                           │
│                                                                              │
│  MEMORY SHARING:                                                             │
│  "You are helpful." stored ONCE, reused by 3 request patterns               │
│  "What is" stored ONCE, reused by 2 patterns                                │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Diagram 3: Partial Prefix Match and Reuse Path

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PARTIAL PREFIX MATCH                                      │
│                                                                              │
│  New Request: "You are helpful. How do I cook pasta?"                       │
│  Tokens: [151644, 872, 527, 11190, 13, 3838, 653, 358, 4394, 46846, 30]     │
│          │       │    │    │      │   │                                     │
│          └───────┴────┴────┴──────┴───┘                                     │
│                   matches tree         ↑ diverges here                       │
│                                                                              │
│  Tree Walk:                                                                  │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  1. Start at ROOT, check for child with token 151644                  │   │
│  │     → Found: "You are helpful." node                                  │   │
│  │                                                                       │   │
│  │  2. Compare input[0:5] with node key [151644,872,527,11190,13]       │   │
│  │     → fast_compare_key returns 5 (all match!)                         │   │
│  │                                                                       │   │
│  │  3. Check children for token 3838 ("How")                             │   │
│  │     → Children are: 3555 ("What"), "Please"                          │   │
│  │     → 3838 NOT FOUND                                                  │   │
│  │                                                                       │   │
│  │  4. Return (node="You are helpful.", prefix_len=5)                    │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  Result:                                                                     │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  cached_len = 5 tokens                                                │   │
│  │  match_indices = [0, 1, 2, 3, 4]                                      │   │
│  │                                                                       │   │
│  │  extend_len = 11 - 5 = 6 tokens to compute                            │   │
│  │  "How do I cook pasta?" needs forward pass                            │   │
│  │                                                                       │   │
│  │  Speedup: 11 tokens → 6 tokens = 1.8x faster prefill                  │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  After request completes, tree is updated:                                   │
│                                                                              │
│    ┌─────────────────────────┐                                              │
│    │ "You are helpful."      │                                              │
│    └───────────┬─────────────┘                                              │
│                │                                                             │
│      ┌─────────┼─────────────────────┐                                      │
│      ▼         ▼                     ▼                                       │
│ ┌─────────┐ ┌─────────┐ ┌─────────────────────────┐                         │
│ │"What is"│ │"Please" │ │"How do I cook pasta?"   │ ← NEW NODE              │
│ └─────────┘ └─────────┘ │ key: [3838,653,358,...] │                         │
│                         │ value: [100,101,...]    │                         │
│                         └─────────────────────────┘                         │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Diagram 4: Execution Timeline: Cache Hit vs Cache Miss

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    EXECUTION TIMELINE COMPARISON                             │
│                                                                              │
│  REQUEST: "System prompt (100 tokens). User query (20 tokens)"               │
│                                                                              │
│  ═══════════════════════════════════════════════════════════════════════════│
│  CACHE MISS (First request with this system prompt)                          │
│  ═══════════════════════════════════════════════════════════════════════════│
│                                                                              │
│  Time ───────────────────────────────────────────────────────────────────►  │
│                                                                              │
│  CPU:  │ Recv │ Lookup │ Alloc 120 │ Prepare │     Idle      │ Process │    │
│        │ Msg  │ (miss) │ KV slots  │ Batch   │               │ Output  │    │
│        └──────┴────────┴───────────┴─────────┴───────────────┴─────────┘    │
│                                                                              │
│  GPU:  │                           │████████████████████████│ Decode...│    │
│        │         Waiting           │  PREFILL 120 tokens    │          │    │
│        │                           │  (FULL computation)    │          │    │
│        └───────────────────────────┴────────────────────────┴──────────┘    │
│                                                                              │
│  Prefill time: ████████████████████████ (120 tokens × T_per_token)          │
│                                                                              │
│  ═══════════════════════════════════════════════════════════════════════════│
│  CACHE HIT (Second request with same system prompt)                          │
│  ═══════════════════════════════════════════════════════════════════════════│
│                                                                              │
│  Time ───────────────────────────────────────────────────────────────────►  │
│                                                                              │
│  CPU:  │ Recv │ Lookup │ Alloc 20 │ Prepare │  Idle  │ Process │            │
│        │ Msg  │ (HIT!) │ KV slots │ Batch   │        │ Output  │            │
│        │      │ 100 ms │ (20 only)│         │        │         │            │
│        └──────┴────────┴──────────┴─────────┴────────┴─────────┘            │
│                                                                              │
│  GPU:  │                │██████│ Decode...│                                  │
│        │    Waiting     │PREFIL│          │                                  │
│        │                │ 20tk │          │                                  │
│        └────────────────┴──────┴──────────┘                                  │
│                                                                              │
│  Prefill time: ██████ (20 tokens × T_per_token)                              │
│                                                                              │
│  SPEEDUP: 120/20 = 6x FASTER PREFILL!                                        │
│                                                                              │
│  ═══════════════════════════════════════════════════════════════════════════│
│  OVERLAP SCHEDULING (CPU hidden behind GPU)                                  │
│  ═══════════════════════════════════════════════════════════════════════════│
│                                                                              │
│  GPU:  │  Batch N-1 Decode  │  Batch N Prefill  │  Batch N Decode  │        │
│        └────────────────────┴───────────────────┴──────────────────┘        │
│                                                                              │
│  CPU:  │ Process N-2 Output │ Recv + Lookup + Prepare N │ Process N-1 │      │
│        │                    │                           │             │      │
│        └────────────────────┴───────────────────────────┴─────────────┘      │
│                              ↑                                               │
│                              └─ CPU work HIDDEN behind GPU compute           │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Diagram 5: vLLM (Paged KV) vs SGLang (Radix KV)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    vLLM vs SGLang COMPARISON                                 │
│                                                                              │
│  ═══════════════════════════════════════════════════════════════════════════│
│  vLLM: PAGED ATTENTION (Memory Optimization)                                 │
│  ═══════════════════════════════════════════════════════════════════════════│
│                                                                              │
│  Physical Page Pool:                                                         │
│  ┌────┐┌────┐┌────┐┌────┐┌────┐┌────┐┌────┐┌────┐┌────┐┌────┐              │
│  │ P0 ││ P1 ││ P2 ││ P3 ││ P4 ││ P5 ││ P6 ││ P7 ││ P8 ││ P9 │ ...         │
│  │16tk││16tk││16tk││16tk││16tk││16tk││16tk││16tk││16tk││16tk│              │
│  └────┘└────┘└────┘└────┘└────┘└────┘└────┘└────┘└────┘└────┘              │
│                                                                              │
│  Request A: "System prompt. Query A"   Page Table: P0 → P1 → P2             │
│  Request B: "System prompt. Query B"   Page Table: P3 → P4 → P5             │
│                                         ↑                                    │
│                         SAME CONTENT, DIFFERENT PAGES!                       │
│                         System prompt RECOMPUTED for each request            │
│                                                                              │
│  Prefix Caching (vLLM APC - Added Later):                                    │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  Block hash table:                                                    │   │
│  │    hash(tokens[0:16])  → P0  (if matches exactly)                     │   │
│  │    hash(tokens[16:32]) → P1  (if matches exactly)                     │   │
│  │                                                                       │   │
│  │  Limitation: Only 16-token block granularity                          │   │
│  │              37-token prefix → 32 cached, 5 recomputed                │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ═══════════════════════════════════════════════════════════════════════════│
│  SGLang: RADIX ATTENTION (Compute Optimization)                              │
│  ═══════════════════════════════════════════════════════════════════════════│
│                                                                              │
│  Token Slot Pool (page_size = 1):                                            │
│  ┌──┐┌──┐┌──┐┌──┐┌──┐┌──┐┌──┐┌──┐┌──┐┌──┐┌──┐┌──┐┌──┐┌──┐┌──┐...          │
│  │S0││S1││S2││S3││S4││S5││S6││S7││S8││S9││10││11││12││13││14│              │
│  └──┘└──┘└──┘└──┘└──┘└──┘└──┘└──┘└──┘└──┘└──┘└──┘└──┘└──┘└──┘              │
│                                                                              │
│  Radix Tree:                                                                 │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │               ROOT                                                      │ │
│  │                │                                                        │ │
│  │                ▼                                                        │ │
│  │    ┌───────────────────────────┐                                        │ │
│  │    │ "System prompt."          │                                        │ │
│  │    │ KV indices: [0-19]        │ ← STORED ONCE                          │ │
│  │    └───────────┬───────────────┘                                        │ │
│  │                │                                                        │ │
│  │       ┌────────┴────────┐                                               │ │
│  │       ▼                 ▼                                               │ │
│  │  ┌──────────┐     ┌──────────┐                                          │ │
│  │  │"Query A" │     │"Query B" │                                          │ │
│  │  │KV:[20-24]│     │KV:[25-29]│                                          │ │
│  │  └──────────┘     └──────────┘                                          │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  Request A: Uses indices [0-19] (cached) + [20-24] (new)                    │
│  Request B: Uses indices [0-19] (SAME cached!) + [25-29] (new)              │
│                              ↑                                               │
│              SAME KV VALUES REUSED, NO RECOMPUTATION!                        │
│                                                                              │
│  ═══════════════════════════════════════════════════════════════════════════│
│                           COMPARISON SUMMARY                                 │
│  ═══════════════════════════════════════════════════════════════════════════│
│                                                                              │
│  ┌─────────────────────┬────────────────────┬────────────────────────────┐  │
│  │ Aspect              │ vLLM               │ SGLang                     │  │
│  ├─────────────────────┼────────────────────┼────────────────────────────┤  │
│  │ Primary goal        │ Memory efficiency  │ Compute efficiency         │  │
│  │ Cache granularity   │ 16-token blocks    │ 1 token (any prefix len)   │  │
│  │ Matching method     │ Block hash         │ Token-by-token tree walk   │  │
│  │ Partial matches     │ Block boundaries   │ Any position               │  │
│  │ Shared prefix reuse │ With APC (limited) │ Native (full)              │  │
│  │ Best for            │ Diverse workloads  │ Prefix-heavy workloads     │  │
│  └─────────────────────┴────────────────────┴────────────────────────────┘  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## F) Practical Takeaways

### Strengths and Weaknesses of Radix Caching

| Strength | Why It Matters |
|----------|----------------|
| **Token-level granularity** | Any prefix length benefits, not just block multiples |
| **Automatic detection** | No manual cache key specification needed |
| **LRU eviction** | Adapts to workload naturally |
| **Low overhead** | ~50μs lookup, hidden by overlap scheduling |
| **Compound benefits** | More users = more cache hits = better efficiency |

| Weakness | Impact | Mitigation |
|----------|--------|------------|
| **Memory overhead** | ~100 bytes per tree node | Minimal vs KV cache size |
| **CPU complexity** | Tree operations on critical path | Overlap scheduling hides it |
| **Cache churn** | Low-overlap workloads thrash | Size cache appropriately |
| **LoRA complexity** | Same tokens ≠ same KV with adapters | Key on (adapter, tokens) |
| **Debugging** | Cache behavior is implicit | Instrument hit rate |

### Common Failure Modes

| Mode | Symptoms | Root Cause | Fix |
|------|----------|------------|-----|
| **Cache thrashing** | Low hit rate, high eviction | Working set > cache size | Increase cache or reduce diversity |
| **No hits despite similar prompts** | 0% hit rate | Different tokenization | Standardize tokenizer |
| **OOM during prefill** | CUDA OOM | Too many locked handles | Reduce max_running_requests |
| **Slow first token** | High TTFT | Cold cache | Warm up with common prefixes |
| **Stale cache with LoRA** | Wrong outputs | Same tokens, different adapters | Separate caches per adapter |

### "If You See X Workload Pattern, Radix Cache Helps/Hurts"

| Pattern | Radix Cache Effect | Recommendation |
|---------|-------------------|----------------|
| **Same system prompt, many users** | ✅ Massive win (10-50x) | Use SGLang |
| **Multi-turn conversations** | ✅ Strong win (2-5x) | Use SGLang |
| **Agent tool loops** | ✅ Strong win | Use SGLang |
| **RAG with shared documents** | ✅ Good win | Use SGLang |
| **Few-shot with shared examples** | ✅ Good win | Use SGLang |
| **Batch of unique documents** | ⚠️ No benefit | Use either |
| **Very short prompts** | ⚠️ Minimal benefit | Use either |
| **High request diversity** | ❌ Cache thrashing | Consider vLLM |
| **Memory-constrained** | ❌ Tree overhead hurts | Consider vLLM |

### Top Tuning Knobs

| Knob | Config | Effect | Recommendation |
|------|--------|--------|----------------|
| **Cache type** | `--cache radix` or `--cache naive` | Enable/disable radix cache | Radix for prefix-heavy workloads |
| **Max running requests** | `--max-running-requests N` | Concurrent request limit | Start at 128, tune for memory |
| **Prefill chunk size** | `--max-prefill-length N` | Split long prefills | 8192 default, reduce if OOM |
| **GPU memory ratio** | `--mem-fraction-static` | KV cache size | 0.85-0.90 for most setups |
| **Overlap scheduling** | `DISABLE_OVERLAP_SCHEDULING=1` | Hide CPU latency | Keep enabled (default) |

### When to Combine SGLang with vLLM (Hybrid Architectures)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    HYBRID ARCHITECTURE                                       │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                        LOAD BALANCER                                     ││
│  └─────────────────────────────┬───────────────────────────────────────────┘│
│                                │                                             │
│         ┌──────────────────────┴──────────────────────┐                     │
│         │                                             │                      │
│         ▼                                             ▼                      │
│  ┌─────────────────┐                         ┌─────────────────┐            │
│  │    SGLang       │                         │    vLLM         │            │
│  │    Cluster      │                         │    Cluster      │            │
│  │                 │                         │                 │            │
│  │ Route here for: │                         │ Route here for: │            │
│  │ • Chat sessions │                         │ • Batch jobs    │            │
│  │ • Agent calls   │                         │ • One-off API   │            │
│  │ • RAG queries   │                         │ • Embeddings    │            │
│  │ • Multi-turn    │                         │ • Diverse input │            │
│  └─────────────────┘                         └─────────────────┘            │
│                                                                              │
│  Routing logic:                                                              │
│  • session_id exists → SGLang (multi-turn benefits)                         │
│  • system_prompt in hot_set → SGLang (prefix cached)                        │
│  • batch_job flag → vLLM (throughput focus)                                 │
│  • default → either (load balance)                                          │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Summary

**SGLang exists because:**
1. Most production LLM workloads have significant prefix overlap
2. Recomputing KV cache for shared prefixes wastes 2-10x compute
3. vLLM's PagedAttention solves memory fragmentation, not compute redundancy
4. Token-level radix caching automatically detects and reuses any shared prefix

**The core innovation:**
- A radix tree indexes the KV cache by token sequences
- Partial prefix matches work at any granularity (1 token)
- LRU eviction on leaves adapts to workload patterns
- Overlap scheduling hides CPU overhead behind GPU compute

**When to use SGLang:**
- Chatbots, agents, RAG, multi-turn conversations
- Any workload with >30% prefix reuse potential

**When to use vLLM:**
- Batch processing of unrelated documents
- High request diversity with minimal prefix overlap
- Memory-constrained deployments

**Trade-offs to watch:**
- Cache sizing vs. working set size
- LoRA adapter isolation
- Cache churn under high diversity

---

*This document is derived from analysis of the Mini-SGLang implementation and the SGLang paper (arXiv:2312.07104). For the full SGLang implementation, see https://github.com/sgl-project/sglang*
