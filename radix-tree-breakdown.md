# Radix Tree Prefix Cache: How It Actually Works in Practice

This document explains the practical implementation of Radix Tree–based prefix caching in Mini-SGLang, comparing it to traditional caching approaches and clarifying exactly what is cached, when, and how.

---

## Table of Contents

1. [What Is Actually Being Cached?](#1-what-is-actually-being-cached)
2. [How the Radix Tree Works at Runtime](#2-how-the-radix-tree-works-at-runtime)
3. [Code Implementation Walkthrough](#3-code-implementation-walkthrough)
4. [Cache Hit vs Cache Miss](#4-cache-hit-vs-cache-miss)
5. [Comparison to Other Caching Approaches](#5-comparison-to-other-caching-approaches)
6. [Latency and Overhead Analysis](#6-latency-and-overhead-analysis)
7. [Why This Is Simpler Than Embedding Retrieval](#7-why-this-is-simpler-than-embedding-retrieval)
8. [When to Use Radix Tree Caching](#8-when-to-use-radix-tree-caching)

---

## 1. What Is Actually Being Cached?

### The Core Insight

Radix Tree prefix caching does **NOT** cache:
- Embeddings
- Model weights
- Intermediate activations
- The model's "understanding" of text

It **DOES** cache:
- **KV cache tensors** (Key and Value projections from attention layers)
- **Indexed by token sequences** (the actual token IDs)

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        WHAT IS CACHED: KV TENSORS                                │
│                                                                                  │
│  When the model processes "You are a helpful assistant":                         │
│                                                                                  │
│  Token: "You"      → Compute K₁, V₁ for all 28 layers → Store at index 0        │
│  Token: "are"      → Compute K₂, V₂ for all 28 layers → Store at index 1        │
│  Token: "a"        → Compute K₃, V₃ for all 28 layers → Store at index 2        │
│  Token: "helpful"  → Compute K₄, V₄ for all 28 layers → Store at index 3        │
│  Token: "assistant"→ Compute K₅, V₅ for all 28 layers → Store at index 4        │
│                                                                                  │
│  The Radix Tree stores:                                                          │
│    key:   [token_id("You"), token_id("are"), ...]  (the token sequence)         │
│    value: [0, 1, 2, 3, 4]                          (indices into KV cache pool)  │
│                                                                                  │
│  The KV Cache Pool (GPU memory) stores:                                          │
│    index 0: K₁, V₁ for all layers                                                │
│    index 1: K₂, V₂ for all layers                                                │
│    ...                                                                           │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Why KV Cache Specifically?

During transformer inference, the attention mechanism needs to attend to all previous tokens:

```python
# Simplified attention computation
def attention(query, key, value):
    # query: current token's Q projection
    # key, value: ALL previous tokens' K, V projections
    scores = query @ key.transpose(-2, -1)  # Attend to all previous
    weights = softmax(scores)
    output = weights @ value
    return output
```

**The expensive part**: Computing K and V for each token requires a full forward pass through the model up to that layer. If we've already computed K, V for a prefix, we can skip that computation entirely.

---

## 2. How the Radix Tree Works at Runtime

### Data Structure Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         RADIX TREE STRUCTURE                                     │
│                                                                                  │
│  A Radix Tree (also called Patricia Trie) is a compressed trie where:            │
│  - Each node stores a SEQUENCE of tokens (not just one)                          │
│  - Nodes are split only when prefixes diverge                                    │
│  - Each node points to KV cache indices for its token sequence                   │
│                                                                                  │
│  Example after processing 3 requests:                                            │
│    Request 1: "System prompt. What is 2+2?"                                      │
│    Request 2: "System prompt. What is 3+3?"                                      │
│    Request 3: "Different prompt. Hello!"                                         │
│                                                                                  │
│                              ROOT                                                │
│                               │                                                  │
│              ┌────────────────┴────────────────┐                                 │
│              ▼                                 ▼                                 │
│    ┌──────────────────┐              ┌──────────────────┐                        │
│    │ "System prompt.  │              │ "Different       │                        │
│    │  What is"        │              │  prompt. Hello!" │                        │
│    │ KV: [0,1,2,3,4,5]│              │ KV: [20,21,22,   │                        │
│    │ ref_count: 0     │              │      23,24,25]   │                        │
│    └────────┬─────────┘              └──────────────────┘                        │
│             │                                                                    │
│    ┌────────┴────────┐                                                           │
│    ▼                 ▼                                                           │
│  ┌───────┐       ┌───────┐                                                       │
│  │"2+2?" │       │"3+3?" │                                                       │
│  │KV:[6-9]│      │KV:[10-13]│                                                    │
│  └───────┘       └───────┘                                                       │
│                                                                                  │
│  Key insight: "System prompt. What is" is stored ONCE, reused by both branches  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Node Structure (from code)

```python
# python/minisgl/kvcache/radix_manager.py

class RadixTreeNode:
    def __init__(self, tic: int | None = None) -> None:
        self.children: Dict[int, RadixTreeNode] = {}  # first_token_id → child
        self._parent: RadixTreeNode | None = None
        self.ref_count: int = 0          # How many active requests use this
        self.timestamp = tic or time.monotonic_ns()  # For LRU eviction
        
        # The actual cached data
        self._key: torch.Tensor    # Token IDs (e.g., [151644, 872, 198, ...])
        self._value: torch.Tensor  # KV cache indices (e.g., [0, 1, 2, ...])
        self._length: int          # len(_key) == len(_value)
```

**What each field means:**
- `_key`: The actual token IDs this node represents
- `_value`: Indices into the GPU KV cache pool where K,V tensors are stored
- `children`: Map from first token of child sequences to child nodes
- `ref_count`: Prevents eviction while requests are using this prefix
- `timestamp`: For LRU eviction when memory is full

---

## 3. Code Implementation Walkthrough

### Step 1: Matching a Prefix

When a new request arrives, we walk the tree to find the longest matching prefix:

```python
# python/minisgl/kvcache/radix_manager.py

def _walk(self, input_ids: torch.Tensor) -> Tuple[RadixTreeNode, int]:
    """
    Walk the tree to find the longest matching prefix.
    
    Returns:
        node: The deepest node we reached
        prefix_len: How many tokens matched
    """
    prefix_len = 0
    indice_len = len(input_ids)
    node = self.root_node
    tic = time.monotonic_ns()

    while prefix_len < indice_len:
        # Check if there's a child starting with the next token
        this_id = int(input_ids[prefix_len].item())
        if this_id not in node.children:
            return node, prefix_len  # No match, stop here

        node = node.children[this_id]

        # Compare this node's key with our input
        # Uses fast C++ kernel for efficiency
        match_len = node.get_match_len(input_ids[prefix_len:])
        prefix_len += match_len

        # If we didn't match the entire node, we need to split
        if match_len != node.length:
            node = node._split_at(match_len)
            return node, prefix_len

        # Update timestamp for LRU
        node.timestamp = tic

    return node, prefix_len
```

**Visual walkthrough:**

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         TREE WALK EXAMPLE                                        │
│                                                                                  │
│  Input: [A, B, C, D, E, F, G]                                                    │
│                                                                                  │
│  Tree:                                                                           │
│                    ROOT                                                          │
│                     │                                                            │
│                     ▼                                                            │
│              ┌─────────────┐                                                     │
│              │ key: [A,B,C]│                                                     │
│              │ val: [0,1,2]│                                                     │
│              └──────┬──────┘                                                     │
│                     │                                                            │
│           ┌─────────┴─────────┐                                                  │
│           ▼                   ▼                                                  │
│    ┌─────────────┐     ┌─────────────┐                                           │
│    │ key: [D,E]  │     │ key: [X,Y]  │                                           │
│    │ val: [3,4]  │     │ val: [5,6]  │                                           │
│    └─────────────┘     └─────────────┘                                           │
│                                                                                  │
│  Walk steps:                                                                     │
│  1. At ROOT, input[0]=A, check children → found child starting with A            │
│  2. At [A,B,C], compare input[0:3] with [A,B,C] → full match (3 tokens)          │
│  3. prefix_len = 3, check children for input[3]=D → found child                  │
│  4. At [D,E], compare input[3:5] with [D,E] → full match (2 tokens)              │
│  5. prefix_len = 5, check children for input[5]=F → NOT FOUND                    │
│  6. Return (node=[D,E], prefix_len=5)                                            │
│                                                                                  │
│  Result: 5 tokens cached, only need to compute [F, G]                            │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Step 2: Using the Match Result

```python
# python/minisgl/kvcache/radix_manager.py

def match_prefix(self, input_ids: torch.Tensor) -> Tuple[RadixCacheHandle, torch.Tensor]:
    """
    Find longest matching prefix and return KV cache indices.
    """
    node, prefix_len = self._walk(input_ids)
    
    if prefix_len == 0:
        # No match at all
        return RadixCacheHandle(0, node), self.empty_tensor
    
    # Collect KV indices by walking back up the tree
    value_list: List[torch.Tensor] = []
    while not node.is_root():
        value_list.append(node.value)  # KV cache indices for this node
        node = node.parent
    value_list.reverse()
    
    # Concatenate all indices
    return RadixCacheHandle(prefix_len, node), torch.cat(value_list)
```

**What this returns:**
- `RadixCacheHandle(prefix_len, node)`: How many tokens matched, and which node
- `torch.Tensor`: The KV cache indices for all matched tokens

### Step 3: Using Cached KV During Inference

```python
# python/minisgl/scheduler/prefill.py

class PrefillAdder:
    def _try_allocate_one(self, req: PendingReq) -> Tuple[BaseCacheHandle, int] | None:
        # 1. Check radix cache for prefix match
        handle, match_indices = self.cache_manager.match_req(req)
        cached_len = handle.cached_len
        
        # 2. Calculate how much we actually need to compute
        extend_len = req.input_len - cached_len  # Only uncached tokens!
        
        # 3. Lock the handle (prevents eviction during use)
        self.cache_manager.lock(handle)
        
        # 4. Copy cached KV indices to this request's page table
        if cached_len > 0:
            page_entry = self.table_manager.page_table[table_idx][:cached_len]
            page_entry.copy_(match_indices)  # Point to existing KV cache!
        
        return handle, table_idx
```

**The key insight**: We don't copy the KV tensors themselves. We just copy the *indices* that point to where those tensors already exist in GPU memory.

### Step 4: Inserting New Prefixes After Completion

```python
# python/minisgl/kvcache/radix_manager.py

def insert_prefix(self, input_ids: torch.Tensor, indices: torch.Tensor) -> int:
    """
    After a request completes, add its prefix to the tree for future reuse.
    """
    node, prefix_len = self._walk(input_ids)
    
    # If we didn't match everything, create a new node
    if prefix_len < len(input_ids):
        new_node = RadixTreeNode()
        new_node.set_key_value(
            input_ids[prefix_len:],   # Unmatched tokens
            indices[prefix_len:]      # Their KV cache indices
        )
        new_node.set_parent(node)
        self.evictable_size += new_node.length
    
    return prefix_len  # How much was already cached (caller can free these)
```

---

## 4. Cache Hit vs Cache Miss

### Cache Hit Scenario

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              CACHE HIT                                           │
│                                                                                  │
│  Request: "You are a helpful assistant. What is the capital of France?"          │
│  Tokens: [A, B, C, D, E, F, G, H, I, J, K, L, M, N, O]  (15 tokens)              │
│                                                                                  │
│  Tree already has:                                                               │
│    "You are a helpful assistant. What is" → KV indices [0-11]                    │
│                                                                                  │
│  match_prefix() returns:                                                         │
│    cached_len = 12                                                               │
│    indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]                              │
│                                                                                  │
│  What happens:                                                                   │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │ Tokens 0-11: SKIP COMPUTATION                                            │    │
│  │   - Page table points to existing KV cache indices [0-11]                │    │
│  │   - Attention can read K,V directly from these locations                 │    │
│  │                                                                          │    │
│  │ Tokens 12-14: COMPUTE                                                    │    │
│  │   - Allocate new KV cache indices [100, 101, 102]                        │    │
│  │   - Run forward pass for only these 3 tokens                             │    │
│  │   - Store new K,V at indices [100, 101, 102]                             │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                  │
│  Speedup: 15 tokens → 3 tokens = 5x faster prefill!                              │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Cache Miss Scenario

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              CACHE MISS                                          │
│                                                                                  │
│  Request: "Translate this to French: Hello world"                                │
│  Tokens: [X, Y, Z, W, V, U, T]  (7 tokens)                                       │
│                                                                                  │
│  Tree has nothing starting with token X                                          │
│                                                                                  │
│  match_prefix() returns:                                                         │
│    cached_len = 0                                                                │
│    indices = []  (empty)                                                         │
│                                                                                  │
│  What happens:                                                                   │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │ All 7 tokens: COMPUTE                                                    │    │
│  │   - Allocate new KV cache indices [200, 201, 202, 203, 204, 205, 206]   │    │
│  │   - Run forward pass for all 7 tokens                                    │    │
│  │   - Store K,V at these indices                                           │    │
│  │                                                                          │    │
│  │ After completion: INSERT into tree                                       │    │
│  │   - New node: key=[X,Y,Z,W,V,U,T], value=[200-206]                       │    │
│  │   - Future requests with this prefix will hit cache                      │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                  │
│  No speedup for this request, but enables speedup for future similar requests    │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Partial Match Scenario

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           PARTIAL MATCH                                          │
│                                                                                  │
│  Tree has: "You are a helpful assistant. What is 2+2?"                           │
│  Request:  "You are a helpful assistant. What is 3+3?"                           │
│                                                                                  │
│  match_prefix() walks:                                                           │
│    "You" ✓ "are" ✓ "a" ✓ "helpful" ✓ "assistant" ✓ "." ✓                        │
│    "What" ✓ "is" ✓ "2" ✗ (mismatch!)                                            │
│                                                                                  │
│  Result: cached_len = 8 (matched "You are a helpful assistant. What is")         │
│                                                                                  │
│  Tree SPLITS:                                                                    │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │  Before:                          After:                                 │    │
│  │  ┌──────────────────┐            ┌──────────────────┐                   │    │
│  │  │"You are...2+2?" │            │"You are...What is"│ ◀── shared       │    │
│  │  │ KV: [0-14]      │            │ KV: [0-7]         │                   │    │
│  │  └──────────────────┘            └────────┬─────────┘                   │    │
│  │                                           │                              │    │
│  │                                  ┌────────┴────────┐                     │    │
│  │                                  ▼                 ▼                     │    │
│  │                            ┌──────────┐     ┌──────────┐                 │    │
│  │                            │ "2+2?"   │     │ "3+3?"   │                 │    │
│  │                            │ KV:[8-14]│     │ KV:[new] │                 │    │
│  │                            └──────────┘     └──────────┘                 │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                  │
│  Both "2+2?" and "3+3?" now share the common prefix!                             │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Comparison to Other Caching Approaches

### Approach 1: No Caching (Baseline)

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           NO CACHING                                             │
│                                                                                  │
│  Every request recomputes everything:                                            │
│                                                                                  │
│  Request 1: "System prompt. Question A" → Compute 20 tokens                      │
│  Request 2: "System prompt. Question B" → Compute 20 tokens                      │
│  Request 3: "System prompt. Question C" → Compute 20 tokens                      │
│                                                                                  │
│  Total: 60 token computations                                                    │
│                                                                                  │
│  Memory: Only current request's KV cache                                         │
│  Compute: Maximum (no reuse)                                                     │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Approach 2: Exact Match Hash Cache

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        EXACT MATCH HASH CACHE                                    │
│                                                                                  │
│  Cache: hash(full_prompt) → KV tensors                                           │
│                                                                                  │
│  Request 1: "System prompt. Question A" → hash1 → MISS → Compute 20              │
│  Request 2: "System prompt. Question B" → hash2 → MISS → Compute 20              │
│  Request 3: "System prompt. Question A" → hash1 → HIT  → Compute 0               │
│                                                                                  │
│  Total: 40 token computations                                                    │
│                                                                                  │
│  Problem: "System prompt" is computed twice (requests 1 and 2)                   │
│  No partial matching - must be EXACT same prompt                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Approach 3: Radix Tree Prefix Cache

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        RADIX TREE PREFIX CACHE                                   │
│                                                                                  │
│  Cache: token_sequence → KV indices (with prefix sharing)                        │
│                                                                                  │
│  Request 1: "System prompt. Question A"                                          │
│    → match_prefix: 0 tokens                                                      │
│    → Compute 20 tokens                                                           │
│    → Insert: "System prompt. Question A" → [0-19]                                │
│                                                                                  │
│  Request 2: "System prompt. Question B"                                          │
│    → match_prefix: 15 tokens ("System prompt. ")                                 │
│    → Compute 5 tokens ("Question B")                                             │
│    → Tree splits, shares prefix                                                  │
│                                                                                  │
│  Request 3: "System prompt. Question C"                                          │
│    → match_prefix: 15 tokens                                                     │
│    → Compute 5 tokens                                                            │
│                                                                                  │
│  Total: 20 + 5 + 5 = 30 token computations (vs 60 baseline, vs 40 hash)          │
│                                                                                  │
│  Advantage: Automatic prefix sharing, partial matches work                       │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Approach 4: Embedding-Based Retrieval (NOT what Radix does)

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    EMBEDDING-BASED RETRIEVAL (Different approach)                │
│                                                                                  │
│  This is what RAG systems do - NOT what Radix Cache does:                        │
│                                                                                  │
│  1. Embed the query: embed("What is the capital?") → vector [0.1, 0.3, ...]     │
│  2. Search vector DB for similar embeddings                                      │
│  3. Retrieve relevant documents                                                  │
│  4. Concatenate to prompt and run inference                                      │
│                                                                                  │
│  This is SEMANTIC similarity, not EXACT prefix matching                          │
│                                                                                  │
│  Radix Cache is much simpler:                                                    │
│  - No embeddings computed                                                        │
│  - No vector similarity search                                                   │
│  - Just exact token sequence matching                                            │
│  - O(prefix_length) lookup, not O(log N) or O(N) similarity search               │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Latency and Overhead Analysis

### Where Overhead Occurs

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         OVERHEAD BREAKDOWN                                       │
│                                                                                  │
│  OPERATION              │ TIME      │ WHERE      │ NOTES                         │
│  ───────────────────────┼───────────┼────────────┼─────────────────────────────  │
│  Tree walk (_walk)      │ ~10-50 μs │ CPU        │ O(prefix_len), uses C++ kernel│
│  Token comparison       │ ~1-5 μs   │ CPU        │ fast_compare_key (C++)        │
│  Index concatenation    │ ~5-20 μs  │ CPU→GPU    │ torch.cat for KV indices      │
│  Lock/unlock handle     │ ~1 μs     │ CPU        │ Just ref_count increment      │
│  Insert new prefix      │ ~10-30 μs │ CPU        │ Tree modification             │
│  ───────────────────────┼───────────┼────────────┼─────────────────────────────  │
│  TOTAL OVERHEAD         │ ~30-100 μs│            │ Per request                   │
│                                                                                  │
│  COMPARISON:                                                                     │
│  - Prefill 100 tokens:  ~10-50 ms (GPU)                                          │
│  - Radix overhead:      ~0.1 ms (CPU)                                            │
│  - Overhead ratio:      ~0.2-1% of prefill time                                  │
│                                                                                  │
│  SAVINGS when cache hits:                                                        │
│  - Skip 100 tokens:     ~10-50 ms saved                                          │
│  - Net benefit:         ~10-50 ms - 0.1 ms = ~10-50 ms                           │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### The Fast Comparison Kernel

```cpp
// python/minisgl/kernel/csrc/src/radix.cpp

auto fast_compare_key(const tvm::ffi::TensorView a,
                      const tvm::ffi::TensorView b) -> size_t {
  // Both tensors must be 1D CPU int tensors
  const auto common_len = std::min(a.size(0), b.size(0));
  
  if (a.dtype().bits == 64) {
    const auto a_ptr = static_cast<const int64_t*>(a.data_ptr());
    const auto b_ptr = static_cast<const int64_t*>(b.data_ptr());
    // std::mismatch is highly optimized, often uses SIMD
    const auto diff_pos = std::mismatch(a_ptr, a_ptr + common_len, b_ptr);
    return static_cast<size_t>(diff_pos.first - a_ptr);
  } else {
    // Same for int32
    const auto a_ptr = static_cast<const int32_t*>(a.data_ptr());
    const auto b_ptr = static_cast<const int32_t*>(b.data_ptr());
    const auto diff_pos = std::mismatch(a_ptr, a_ptr + common_len, b_ptr);
    return static_cast<size_t>(diff_pos.first - a_ptr);
  }
}
```

**Why this is fast:**
- `std::mismatch` is optimized by compilers (often uses SIMD)
- Operates on contiguous memory (cache-friendly)
- Early termination on first mismatch
- No memory allocation

### Overlap Scheduling Hides Overhead

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                      OVERLAP SCHEDULING                                          │
│                                                                                  │
│  Without overlap:                                                                │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │ CPU: [Radix lookup][Schedule][        ][Radix lookup][Schedule]         │    │
│  │ GPU: [              ][Compute Batch 1 ][            ][Compute Batch 2 ] │    │
│  │                                                                          │    │
│  │ Total: CPU_time + GPU_time per batch                                     │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                  │
│  With overlap (Mini-SGLang default):                                             │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │ CPU: [Radix + Schedule B1][Process B0 + Radix + Schedule B2][...]       │    │
│  │ GPU: [                    ][Compute B1                      ][Compute B2]│    │
│  │                            ▲                                             │    │
│  │                            └─ CPU work completely hidden!                │    │
│  │                                                                          │    │
│  │ Total: max(CPU_time, GPU_time) per batch ≈ GPU_time                      │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                  │
│  From scheduler.py:                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │  def overlap_loop(self, last_data: ForwardData | None) -> ForwardData:   │    │
│  │      # 1. Receive new messages (non-blocking if we have work)            │    │
│  │      for msg in self.receive_msg(blocking=blocking):                     │    │
│  │          self._process_one_msg(msg)                                      │    │
│  │                                                                          │    │
│  │      # 2. Schedule next batch (includes radix lookup)                    │    │
│  │      forward_input = self._schedule_next_batch()                         │    │
│  │                                                                          │    │
│  │      # 3. Execute on GPU (in engine's stream)                            │    │
│  │      if forward_input is not None:                                       │    │
│  │          with self.engine_stream_ctx:                                    │    │
│  │              ongoing_data = (forward_input, self._forward(forward_input))│    │
│  │                                                                          │    │
│  │      # 4. Process results from LAST batch (overlap!)                     │    │
│  │      self._process_last_data(last_data, ongoing_data)                    │    │
│  │                                                                          │    │
│  │      return ongoing_data                                                 │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Where Overhead Actually Occurs

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    DETAILED OVERHEAD BREAKDOWN                                   │
│                                                                                  │
│  OPERATION                    │ TIME      │ LOCATION │ NOTES                     │
│  ────────────────────────────┼───────────┼──────────┼─────────────────────────  │
│                               │           │          │                           │
│  1. Tree Walk (_walk)         │ ~10-50 μs │ CPU      │ O(prefix_depth)           │
│     ├─ Dict lookup per node   │ ~0.1 μs   │          │ Python dict is fast       │
│     ├─ Token comparison       │ ~1-5 μs   │          │ C++ kernel                │
│     └─ Node split (if needed) │ ~5 μs     │          │ Rare case                 │
│                               │           │          │                           │
│  2. Index Collection          │ ~5-20 μs  │ CPU→GPU  │ torch.cat for KV indices  │
│     ├─ Walk back up tree      │ ~2 μs     │ CPU      │ Collect value tensors     │
│     └─ Concatenate tensors    │ ~10 μs    │ CPU      │ torch.cat                 │
│                               │           │          │                           │
│  3. Lock/Unlock Handle        │ ~1 μs     │ CPU      │ ref_count increment       │
│     └─ Walk to root           │ ~0.5 μs   │          │ Update ref_count          │
│                               │           │          │                           │
│  4. Insert New Prefix         │ ~10-30 μs │ CPU      │ After request completes   │
│     ├─ Tree walk              │ ~10 μs    │          │ Find insertion point      │
│     ├─ Create new node        │ ~5 μs     │          │ Python object creation    │
│     └─ Update parent links    │ ~2 μs     │          │ Dict insertion            │
│                               │           │          │                           │
│  5. Eviction (when needed)    │ ~50-200 μs│ CPU      │ LRU heap operations       │
│     ├─ Collect leaves         │ ~20 μs    │          │ Tree traversal            │
│     ├─ Heapify                │ ~10 μs    │          │ O(n log n)                │
│     └─ Pop and delete         │ ~5 μs/node│          │ Per evicted node          │
│                               │           │          │                           │
│  ────────────────────────────┼───────────┼──────────┼─────────────────────────  │
│  TOTAL PER REQUEST            │ ~30-100 μs│          │ Typical case              │
│  TOTAL WITH EVICTION          │ ~100-300μs│          │ When cache is full        │
│                               │           │          │                           │
│  COMPARISON TO GPU COMPUTE:                                                      │
│  ────────────────────────────────────────────────────────────────────────────   │
│  Prefill 100 tokens:          ~10-50 ms (GPU)                                    │
│  Decode 1 token:              ~5-10 ms (GPU)                                     │
│  Radix overhead:              ~0.03-0.3 ms (CPU)                                 │
│                                                                                  │
│  Overhead ratio: 0.3% - 3% of GPU time (hidden by overlap scheduling)            │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 7. Why This Is Simpler Than Embedding Retrieval

### Comparison: Radix Cache vs Embedding-Based Systems

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    RADIX CACHE vs EMBEDDING RETRIEVAL                            │
│                                                                                  │
│  ┌─────────────────────────────────┬─────────────────────────────────────────┐  │
│  │       RADIX CACHE               │       EMBEDDING RETRIEVAL               │  │
│  │       (Mini-SGLang)             │       (RAG Systems)                     │  │
│  ├─────────────────────────────────┼─────────────────────────────────────────┤  │
│  │                                 │                                         │  │
│  │  WHAT IT DOES:                  │  WHAT IT DOES:                          │  │
│  │  Exact token sequence matching  │  Semantic similarity search             │  │
│  │                                 │                                         │  │
│  │  INPUT:                         │  INPUT:                                 │  │
│  │  Token IDs [151644, 872, ...]   │  Text "What is the capital?"            │  │
│  │                                 │                                         │  │
│  │  LOOKUP:                        │  LOOKUP:                                │  │
│  │  Tree walk O(prefix_len)        │  Vector similarity O(log N) or O(N)     │  │
│  │                                 │                                         │  │
│  │  MATCH TYPE:                    │  MATCH TYPE:                            │  │
│  │  Exact prefix match             │  Approximate semantic match             │  │
│  │                                 │                                         │  │
│  │  WHAT'S CACHED:                 │  WHAT'S CACHED:                         │  │
│  │  KV tensors (already computed)  │  Document embeddings                    │  │
│  │                                 │                                         │  │
│  │  REUSE BENEFIT:                 │  REUSE BENEFIT:                         │  │
│  │  Skip KV computation entirely   │  Find relevant context                  │  │
│  │                                 │                                         │  │
│  └─────────────────────────────────┴─────────────────────────────────────────┘  │
│                                                                                  │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │                                                                           │   │
│  │  WHY RADIX IS SIMPLER:                                                    │   │
│  │                                                                           │   │
│  │  1. NO EMBEDDING MODEL NEEDED                                             │   │
│  │     Radix: Just compare token IDs (integers)                              │   │
│  │     RAG: Need to run embedding model on query                             │   │
│  │                                                                           │   │
│  │  2. NO VECTOR DATABASE NEEDED                                             │   │
│  │     Radix: Simple tree structure in memory                                │   │
│  │     RAG: Need FAISS, Pinecone, Milvus, etc.                               │   │
│  │                                                                           │   │
│  │  3. NO APPROXIMATE MATCHING                                               │   │
│  │     Radix: Exact match or no match (deterministic)                        │   │
│  │     RAG: Similarity threshold, top-k selection (probabilistic)            │   │
│  │                                                                           │   │
│  │  4. NO RERANKING NEEDED                                                   │   │
│  │     Radix: Longest prefix is always the best match                        │   │
│  │     RAG: May need reranking for relevance                                 │   │
│  │                                                                           │   │
│  │  5. ZERO ADDITIONAL LATENCY FOR LOOKUP                                    │   │
│  │     Radix: ~50 μs tree walk (hidden by overlap)                           │   │
│  │     RAG: ~10-100 ms embedding + search                                    │   │
│  │                                                                           │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### What Radix Cache Does NOT Do

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    RADIX CACHE LIMITATIONS                                       │
│                                                                                  │
│  Radix Cache is NOT:                                                             │
│                                                                                  │
│  ✗ A semantic search system                                                      │
│    - "What is Paris?" won't match "What is the capital of France?"               │
│    - Only exact token sequences match                                            │
│                                                                                  │
│  ✗ A document retrieval system                                                   │
│    - Can't find "relevant" documents                                             │
│    - Only reuses previously computed KV cache                                    │
│                                                                                  │
│  ✗ A knowledge base                                                              │
│    - Doesn't store facts or information                                          │
│    - Only stores intermediate computation results                                │
│                                                                                  │
│  ✗ A compression system                                                          │
│    - Doesn't reduce model size                                                   │
│    - Only avoids redundant computation                                           │
│                                                                                  │
│  ═══════════════════════════════════════════════════════════════════════════    │
│                                                                                  │
│  Radix Cache IS:                                                                 │
│                                                                                  │
│  ✓ A computation cache                                                           │
│    - Stores KV tensors indexed by token sequences                                │
│    - Enables skipping redundant forward passes                                   │
│                                                                                  │
│  ✓ An automatic optimization                                                     │
│    - No code changes needed                                                      │
│    - Works transparently for any workload                                        │
│                                                                                  │
│  ✓ A memory-efficient deduplication system                                       │
│    - Shared prefixes stored once                                                 │
│    - Multiple requests reference same KV cache                                   │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 8. When to Use Radix Tree Caching

### Decision Matrix

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         WHEN TO USE RADIX CACHE                                  │
│                                                                                  │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │                                                                           │   │
│  │  HIGH VALUE (Use Radix Cache):                                            │   │
│  │                                                                           │   │
│  │  ✓ Chatbots with system prompts                                           │   │
│  │    - Same "You are a helpful assistant..." for all users                  │   │
│  │    - Savings: 90%+ on system prompt computation                           │   │
│  │                                                                           │   │
│  │  ✓ Multi-turn conversations                                               │   │
│  │    - Each turn builds on previous context                                 │   │
│  │    - Turn N reuses KV from turns 1 to N-1                                 │   │
│  │                                                                           │   │
│  │  ✓ RAG with shared document context                                       │   │
│  │    - Same document, multiple questions                                    │   │
│  │    - Document KV computed once, reused for all questions                  │   │
│  │                                                                           │   │
│  │  ✓ Few-shot learning                                                      │   │
│  │    - Same examples for all queries                                        │   │
│  │    - Example KV cached and reused                                         │   │
│  │                                                                           │   │
│  │  ✓ Code completion with file context                                      │   │
│  │    - Same file content for multiple completions                           │   │
│  │    - File context KV cached                                               │   │
│  │                                                                           │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │                                                                           │   │
│  │  LOW VALUE (Radix Cache helps less):                                      │   │
│  │                                                                           │   │
│  │  ✗ Batch processing of unrelated documents                                │   │
│  │    - Each document is unique                                              │   │
│  │    - No shared prefixes to cache                                          │   │
│  │                                                                           │   │
│  │  ✗ One-off API queries                                                    │   │
│  │    - Each query is independent                                            │   │
│  │    - No prefix reuse opportunity                                          │   │
│  │                                                                           │   │
│  │  ✗ Very short prompts                                                     │   │
│  │    - Little computation to save                                           │   │
│  │    - Overhead may not be worth it                                         │   │
│  │                                                                           │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Quantifying the Benefit

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         BENEFIT CALCULATION                                      │
│                                                                                  │
│  Formula:                                                                        │
│                                                                                  │
│    Speedup = Total_Tokens / Tokens_After_Cache_Hit                               │
│                                                                                  │
│  Example 1: Chatbot with 500-token system prompt                                 │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │                                                                           │   │
│  │  Request: [System: 500 tokens] + [User: 50 tokens]                        │   │
│  │                                                                           │   │
│  │  First request (cold):                                                    │   │
│  │    Tokens computed: 500 + 50 = 550                                        │   │
│  │                                                                           │   │
│  │  Subsequent requests (warm):                                              │   │
│  │    Tokens computed: 50 (system prompt cached!)                            │   │
│  │                                                                           │   │
│  │  Speedup: 550 / 50 = 11x faster prefill!                                  │   │
│  │                                                                           │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
│  Example 2: Multi-turn conversation                                              │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │                                                                           │   │
│  │  Turn 1: [System: 100] + [User1: 50] + [Asst1: 100] = 250 tokens          │   │
│  │  Turn 2: [Turn1: 250] + [User2: 50] + [Asst2: 100] = 400 tokens           │   │
│  │  Turn 3: [Turn2: 400] + [User3: 50] + [Asst3: 100] = 550 tokens           │   │
│  │                                                                           │   │
│  │  Without cache:                                                           │   │
│  │    Turn 1: 250 tokens                                                     │   │
│  │    Turn 2: 400 tokens                                                     │   │
│  │    Turn 3: 550 tokens                                                     │   │
│  │    Total: 1200 tokens                                                     │   │
│  │                                                                           │   │
│  │  With cache:                                                              │   │
│  │    Turn 1: 250 tokens (cold)                                              │   │
│  │    Turn 2: 150 tokens (250 cached)                                        │   │
│  │    Turn 3: 150 tokens (400 cached)                                        │   │
│  │    Total: 550 tokens                                                      │   │
│  │                                                                           │   │
│  │  Speedup: 1200 / 550 = 2.2x overall                                       │   │
│  │                                                                           │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
│  Example 3: RAG with shared context                                              │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │                                                                           │   │
│  │  Context: 2000 tokens (document)                                          │   │
│  │  Questions: 10 questions, 50 tokens each                                  │   │
│  │                                                                           │   │
│  │  Without cache:                                                           │   │
│  │    Per question: 2000 + 50 = 2050 tokens                                  │   │
│  │    Total: 2050 × 10 = 20,500 tokens                                       │   │
│  │                                                                           │   │
│  │  With cache:                                                              │   │
│  │    First question: 2050 tokens (cold)                                     │   │
│  │    Subsequent: 50 tokens each (context cached)                            │   │
│  │    Total: 2050 + 50 × 9 = 2500 tokens                                     │   │
│  │                                                                           │   │
│  │  Speedup: 20,500 / 2,500 = 8.2x overall                                   │   │
│  │                                                                           │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 9. Mini-SGLang Implementation Reference

### Key Files and Their Roles

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    MINI-SGLANG CODE REFERENCE                                    │
│                                                                                  │
│  RADIX CACHE CORE:                                                               │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │                                                                           │   │
│  │  python/minisgl/kvcache/radix_manager.py                                  │   │
│  │  ├── RadixTreeNode          # Tree node with key, value, ref_count        │   │
│  │  ├── RadixCacheHandle       # Handle returned by match_prefix             │   │
│  │  └── RadixCacheManager      # Main manager class                          │   │
│  │      ├── match_prefix()     # Find longest matching prefix                │   │
│  │      ├── insert_prefix()    # Add new prefix to tree                      │   │
│  │      ├── lock_handle()      # Prevent eviction during use                 │   │
│  │      ├── evict()            # LRU eviction of unused leaves               │   │
│  │      └── _walk()            # Internal tree traversal                     │   │
│  │                                                                           │   │
│  │  python/minisgl/kvcache/base.py                                           │   │
│  │  ├── BaseCacheManager       # Abstract interface                          │   │
│  │  ├── BaseCacheHandle        # Abstract handle                             │   │
│  │  └── SizeInfo               # Evictable/protected size tracking           │   │
│  │                                                                           │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
│  SCHEDULER INTEGRATION:                                                          │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │                                                                           │   │
│  │  python/minisgl/scheduler/cache.py                                        │   │
│  │  └── CacheManager           # Combines free slots + radix/naive manager   │   │
│  │      ├── match_req()        # Match prefix for pending request            │   │
│  │      ├── allocate()         # Allocate KV slots (with eviction)           │   │
│  │      ├── lock() / unlock()  # Delegate to radix manager                   │   │
│  │      └── free_and_cache_finished_req()  # Cache completed request         │   │
│  │                                                                           │   │
│  │  python/minisgl/scheduler/prefill.py                                      │   │
│  │  └── PrefillAdder._try_allocate_one()                                     │   │
│  │      # Checks radix cache, locks handle, allocates table slot             │   │
│  │                                                                           │   │
│  │  python/minisgl/scheduler/scheduler.py                                    │   │
│  │  └── Scheduler._process_last_data()                                       │   │
│  │      # Calls free_and_cache_finished_req() for completed requests         │   │
│  │                                                                           │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
│  FAST PREFIX COMPARISON (C++ kernel):                                            │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │                                                                           │   │
│  │  python/minisgl/kernel/csrc/src/radix.cpp                                 │   │
│  │  └── fast_compare_key()     # Optimized token sequence comparison         │   │
│  │      # Uses std::mismatch for SIMD-optimized comparison                   │   │
│  │                                                                           │   │
│  │  python/minisgl/kernel/radix.py                                           │   │
│  │  └── fast_compare_key()     # Python wrapper                              │   │
│  │                                                                           │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```
# Radix Tree Prefix Cache: How It Actually Works in Practice

*(Continued from previous section)*

---

## 9. Mini-SGLang Implementation Reference (Continued)

### Request Flow Through Radix Cache (Continued)

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    REQUEST FLOW THROUGH RADIX CACHE                              │
│                                                                                  │
│  7. Request completes, insert into cache                                         │
│     ┌─────────────────────────────────────────────────────────────────────┐     │
│     │  self.cache_manager.free_and_cache_finished_req(                     │     │
│     │      req.cache_handle,                                               │     │
│     │      req.host_ids[:req.cached_len],  # Token IDs                     │     │
│     │      self.page_table[req.table_idx, :req.cached_len]  # KV indices   │     │
│     │  )                                                                   │     │
│     │  # Inserts new prefix into tree for future reuse                     │     │
│     │  # Unlocks handle (decrements ref_count)                             │     │
│     │  # Frees any duplicate KV slots                                      │     │
│     └─────────────────────────────────────────────────────────────────────┘     │
│                                      │                                           │
│                                      ▼                                           │
│  8. Future requests with same prefix get cache hit                               │
│     ┌─────────────────────────────────────────────────────────────────────┐     │
│     │  # Next request: "You are helpful. What is 3+3?"                     │     │
│     │  # match_prefix() finds "You are helpful. What is" in tree           │     │
│     │  # Returns cached KV indices, skips computation for those tokens     │     │
│     └─────────────────────────────────────────────────────────────────────┘     │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Complete Code Path: From Request to Cache Hit

Let's trace through the actual code files:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    COMPLETE CODE PATH                                            │
│                                                                                  │
│  FILE: python/minisgl/scheduler/scheduler.py                                     │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │                                                                           │   │
│  │  class Scheduler:                                                         │   │
│  │      def __init__(self, config):                                          │   │
│  │          # Create cache manager (radix or naive based on config)          │   │
│  │          self.cache_manager = CacheManager(                               │   │
│  │              self.device,                                                 │   │
│  │              self.engine.num_pages,                                       │   │
│  │              config.cache_type  # "radix" or "naive"                      │   │
│  │          )                                                                │   │
│  │                                                                           │   │
│  │      def _process_one_msg(self, msg):                                     │   │
│  │          if isinstance(msg, UserMsg):                                     │   │
│  │              # Add to prefill queue                                       │   │
│  │              self.prefill_manager.add_one_req(msg)                        │   │
│  │                                                                           │   │
│  │      def _schedule_next_batch(self):                                      │   │
│  │          # Try prefill first, then decode                                 │   │
│  │          batch = self.prefill_manager.schedule_next_batch(budget)         │   │
│  │          if batch is None:                                                │   │
│  │              batch = self.decode_manager.schedule_next_batch()            │   │
│  │          return self._prepare_batch(batch)                                │   │
│  │                                                                           │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
│  FILE: python/minisgl/scheduler/prefill.py                                       │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │                                                                           │   │
│  │  class PrefillAdder:                                                      │   │
│  │      def _try_allocate_one(self, req: PendingReq):                        │   │
│  │          """This is where radix cache lookup happens!"""                  │   │
│  │                                                                           │   │
│  │          # 1. Check radix cache for prefix match                          │   │
│  │          handle, match_indices = self.cache_manager.match_req(req)        │   │
│  │          cached_len = handle.cached_len                                   │   │
│  │                                                                           │   │
│  │          # 2. Calculate tokens to compute                                 │   │
│  │          extend_len = req.input_len - cached_len                          │   │
│  │                                                                           │   │
│  │          # 3. Check if we have enough cache space                         │   │
│  │          estimated_len = extend_len + req.output_len                      │   │
│  │          if estimated_len > self.cache_manager.available_size:            │   │
│  │              return None  # Can't fit, queue for later                    │   │
│  │                                                                           │   │
│  │          # 4. Lock handle (prevents eviction)                             │   │
│  │          self.cache_manager.lock(handle)                                  │   │
│  │                                                                           │   │
│  │          # 5. Copy cached KV indices to page table                        │   │
│  │          if cached_len > 0:                                               │   │
│  │              page_entry[:cached_len].copy_(match_indices)                 │   │
│  │                                                                           │   │
│  │          return handle, table_idx                                         │   │
│  │                                                                           │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
│  FILE: python/minisgl/scheduler/cache.py                                         │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │                                                                           │   │
│  │  class CacheManager:                                                      │   │
│  │      def __init__(self, device, num_pages, cache_type):                   │   │
│  │          # Choose radix or naive manager                                  │   │
│  │          if cache_type == "radix":                                        │   │
│  │              self.manager = RadixCacheManager()                           │   │
│  │          else:                                                            │   │
│  │              self.manager = NaiveCacheManager()                           │   │
│  │                                                                           │   │
│  │          # Free slot tracking                                             │   │
│  │          self._free_slots = torch.arange(num_pages, device=device)        │   │
│  │                                                                           │   │
│  │      def match_req(self, req: PendingReq):                                │   │
│  │          """Delegate to radix manager for prefix matching."""             │   │
│  │          input_len = req.input_len                                        │   │
│  │          # Match all but last token                                       │   │
│  │          return self.manager.match_prefix(req.input_ids[:input_len-1])    │   │
│  │                                                                           │   │
│  │      def allocate(self, needed_len: int):                                 │   │
│  │          """Allocate KV slots, evicting if necessary."""                  │   │
│  │          if needed_len <= len(self._free_slots):                          │   │
│  │              # Fast path: enough free slots                               │   │
│  │              allocated = self._free_slots[:needed_len]                    │   │
│  │              self._free_slots = self._free_slots[needed_len:]             │   │
│  │              return allocated                                             │   │
│  │                                                                           │   │
│  │          # Slow path: need to evict                                       │   │
│  │          evicted = self.manager.evict(needed_len - free_len)              │   │
│  │          # ... merge and return                                           │   │
│  │                                                                           │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
│  FILE: python/minisgl/kvcache/radix_manager.py                                   │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │                                                                           │   │
│  │  class RadixCacheManager(BaseCacheManager):                               │   │
│  │      def __init__(self):                                                  │   │
│  │          self.root_node = RadixTreeNode()                                 │   │
│  │          self.root_node.ref_count = 1  # Root never evicted               │   │
│  │                                                                           │   │
│  │      def match_prefix(self, input_ids: torch.Tensor):                     │   │
│  │          """Find longest matching prefix in tree."""                      │   │
│  │          node, prefix_len = self._walk(input_ids)                         │   │
│  │                                                                           │   │
│  │          if prefix_len == 0:                                              │   │
│  │              return RadixCacheHandle(0, node), self.empty_tensor          │   │
│  │                                                                           │   │
│  │          # Collect KV indices by walking back up                          │   │
│  │          value_list = []                                                  │   │
│  │          while not node.is_root():                                        │   │
│  │              value_list.append(node.value)                                │   │
│  │              node = node.parent                                           │   │
│  │          value_list.reverse()                                             │   │
│  │                                                                           │   │
│  │          return RadixCacheHandle(prefix_len, node), torch.cat(value_list) │   │
│  │                                                                           │   │
│  │      def _walk(self, input_ids: torch.Tensor):                            │   │
│  │          """Walk tree to find longest match."""                           │   │
│  │          prefix_len = 0                                                   │   │
│  │          node = self.root_node                                            │   │
│  │                                                                           │   │
│  │          while prefix_len < len(input_ids):                               │   │
│  │              this_id = int(input_ids[prefix_len].item())                  │   │
│  │              if this_id not in node.children:                             │   │
│  │                  return node, prefix_len  # No match                      │   │
│  │                                                                           │   │
│  │              node = node.children[this_id]                                │   │
│  │              match_len = node.get_match_len(input_ids[prefix_len:])       │   │
│  │              prefix_len += match_len                                      │   │
│  │                                                                           │   │
│  │              if match_len != node.length:                                 │   │
│  │                  node = node._split_at(match_len)                         │   │
│  │                  return node, prefix_len                                  │   │
│  │                                                                           │   │
│  │          return node, prefix_len                                          │   │
│  │                                                                           │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
│  FILE: python/minisgl/kernel/csrc/src/radix.cpp                                  │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │                                                                           │   │
│  │  // Fast token sequence comparison using std::mismatch                    │   │
│  │  auto fast_compare_key(const TensorView a, const TensorView b) -> size_t {│   │
│  │      const auto common_len = std::min(a.size(0), b.size(0));              │   │
│  │      const auto a_ptr = static_cast<const int32_t*>(a.data_ptr());        │   │
│  │      const auto b_ptr = static_cast<const int32_t*>(b.data_ptr());        │   │
│  │      const auto diff_pos = std::mismatch(a_ptr, a_ptr + common_len, b_ptr);│  │
│  │      return static_cast<size_t>(diff_pos.first - a_ptr);                  │   │
│  │  }                                                                        │   │
│  │                                                                           │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 10. Summary: Key Takeaways

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           KEY TAKEAWAYS                                          │
│                                                                                  │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │                                                                           │   │
│  │  WHAT IS CACHED:                                                          │   │
│  │  ─────────────────                                                        │   │
│  │  • KV tensors (Key and Value projections from attention layers)           │   │
│  │  • Indexed by exact token sequences (not embeddings, not semantics)       │   │
│  │  • Stored in GPU memory, referenced by indices in radix tree              │   │
│  │                                                                           │   │
│  │  HOW LOOKUP WORKS:                                                        │   │
│  │  ─────────────────                                                        │   │
│  │  • Tree walk: O(prefix_depth) time, ~10-50 μs                             │   │
│  │  • Token-by-token comparison using fast C++ kernel                        │   │
│  │  • Returns: (cached_length, KV_indices) for matched prefix                │   │
│  │                                                                           │   │
│  │  WHEN CACHE HITS:                                                         │   │
│  │  ─────────────────                                                        │   │
│  │  • Exact token sequence match (not semantic similarity)                   │   │
│  │  • Partial matches work (longest prefix is used)                          │   │
│  │  • Locked during use to prevent eviction                                  │   │
│  │                                                                           │   │
│  │  WHY IT'S SIMPLE:                                                         │   │
│  │  ─────────────────                                                        │   │
│  │  • No embedding model needed                                              │   │
│  │  • No vector database needed                                              │   │
│  │  • No approximate matching or reranking                                   │   │
│  │  • Just exact integer comparison                                          │   │
│  │                                                                           │   │
│  │  OVERHEAD:                                                                │   │
│  │  ─────────────────                                                        │   │
│  │  • ~30-100 μs per request (CPU)                                           │   │
│  │  • Hidden by overlap scheduling                                           │   │
│  │  • <1% of typical GPU compute time                                        │   │
│  │                                                                           │   │
│  │  WHEN TO USE:                                                             │   │
│  │  ─────────────────                                                        │   │
│  │  • Chatbots with system prompts (90%+ savings)                            │   │
│  │  • Multi-turn conversations (incremental caching)                         │   │
│  │  • RAG with shared document context                                       │   │
│  │  • Any workload with repeated prefixes                                    │   │
│  │                                                                           │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │                                                                           │   │
│  │  COMPARISON TO OTHER APPROACHES:                                          │   │
│  │                                                                           │   │
│  │  ┌─────────────────┬─────────────────┬─────────────────┬───────────────┐ │   │
│  │  │   Approach      │  Lookup Time    │  Match Type     │  Complexity   │ │   │
│  │  ├─────────────────┼─────────────────┼─────────────────┼───────────────┤ │   │
│  │  │ No caching      │ N/A             │ N/A             │ None          │ │   │
│  │  │ Hash cache      │ O(1)            │ Exact only      │ Low           │ │   │
│  │  │ Radix tree      │ O(prefix_len)   │ Prefix match    │ Medium        │ │   │
│  │  │ Embedding RAG   │ O(log N)        │ Semantic        │ High          │ │   │
│  │  └─────────────────┴─────────────────┴─────────────────┴───────────────┘ │   │
│  │                                                                           │   │
│  │  Radix tree is the sweet spot:                                            │   │
│  │  • More flexible than hash (partial matches)                              │   │
│  │  • Much simpler than embedding retrieval                                  │   │
│  │  • Perfect for LLM serving where exact prefixes are common                │   │
│  │                                                                           │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 11. Visual Summary: The Complete Picture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    RADIX CACHE: THE COMPLETE PICTURE                             │
│                                                                                  │
│                                                                                  │
│  USER REQUEST                                                                    │
│  "You are helpful. What is 2+2?"                                                 │
│       │                                                                          │
│       ▼                                                                          │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │                         TOKENIZATION                                     │    │
│  │  [151644, 872, 198, 3838, 374, 279, 220, 17, 10, 17, 30]                 │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│       │                                                                          │
│       ▼                                                                          │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │                      RADIX TREE LOOKUP                                   │    │
│  │                                                                          │    │
│  │  Tree:           ROOT                                                    │    │
│  │                   │                                                      │    │
│  │                   ▼                                                      │    │
│  │        ┌─────────────────────┐                                           │    │
│  │        │ "You are helpful."  │ ◀── MATCH! (8 tokens)                     │    │
│  │        │ KV indices: [0-7]   │                                           │    │
│  │        └──────────┬──────────┘                                           │    │
│  │                   │                                                      │    │
│  │                   ▼                                                      │    │
│  │        (no child matches "What")                                         │    │
│  │                                                                          │    │
│  │  Result: cached_len=8, indices=[0,1,2,3,4,5,6,7]                         │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│       │                                                                          │
│       ▼                                                                          │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │                      PAGE TABLE SETUP                                    │    │
│  │                                                                          │    │
│  │  Page Table for this request:                                            │    │
│  │  ┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┐                          │    │
│  │  │ 0 │ 1 │ 2 │ 3 │ 4 │ 5 │ 6 │ 7 │100│101│102│                          │    │
│  │  └───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┘                          │    │
│  │  ◄────── cached (reused) ──────►◄─ new (allocated) ─►                   │    │
│  │                                                                          │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│       │                                                                          │
│       ▼                                                                          │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │                      GPU FORWARD PASS                                    │    │
│  │                                                                          │    │
│  │  For each layer:                                                         │    │
│  │    • Load K,V from indices [0-7] (already computed!)                     │    │
│  │    • Compute K,V for tokens 8-10 ("What is 2+2?")                        │    │
│  │    • Store new K,V at indices [100,101,102]                              │    │
│  │    • Run attention over all 11 K,V pairs                                 │    │
│  │                                                                          │    │
│  │  Compute saved: 8 tokens × 28 layers = 224 layer computations!           │    │
│  │                                                                          │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│       │                                                                          │
│       ▼                                                                          │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │                      INSERT INTO TREE                                    │    │
│  │                                                                          │    │
│  │  After completion, tree is updated:                                      │    │
│  │                                                                          │    │
│  │  Tree:           ROOT                                                    │    │
│  │                   │                                                      │    │
│  │                   ▼                                                      │    │
│  │        ┌─────────────────────┐                                           │    │
│  │        │ "You are helpful."  │                                           │    │
│  │        │ KV indices: [0-7]   │                                           │    │
│  │        └──────────┬──────────┘                                           │    │
│  │                   │                                                      │    │
│  │                   ▼                                                      │    │
│  │        ┌─────────────────────┐                                           │    │
│  │        │ "What is 2+2?"      │ ◀── NEW NODE                              │    │
│  │        │ KV indices:[100-102]│                                           │    │
│  │        └─────────────────────┘                                           │    │
│  │                                                                          │    │
│  │  Future requests with this prefix will get cache hit!                    │    │
│  │                                                                          │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│       │                                                                          │
│       ▼                                                                          │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │                      RESPONSE                                            │    │
│  │                                                                          │    │
│  │  "4"                                                                     │    │
│  │                                                                          │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                  │
│                                                                                  │
│  TOTAL BENEFIT:                                                                  │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │                                                                           │   │
│  │  Without cache: Compute 11 tokens                                         │   │
│  │  With cache:    Compute 3 tokens (8 cached)                               │   │
│  │  Speedup:       11/3 = 3.7x faster prefill                                │   │
│  │                                                                           │   │
│  │  For longer system prompts (500+ tokens), speedup can be 10-50x!          │   │
│  │                                                                           │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

This completes the explanation of how Radix Tree prefix caching works in practice. The key insight is that it's a **simple, exact-match computation cache** that avoids the complexity of embedding-based retrieval while providing significant speedups for workloads with shared prefixes.

<chatName>Radix Tree Prefix Cache Deep Dive</chatName>