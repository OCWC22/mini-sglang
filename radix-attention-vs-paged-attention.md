# RadixAttention vs PagedAttention: A Complete Technical Comparison

*Comprehensive guide for CEOs, engineers, and interns on KV cache optimization in LLM inference*

---

## Table of Contents

1. [CEO Explanation](#1-ceo-explanation)
2. [Intern Onboarding Explanation](#2-intern-onboarding-explanation)
3. [Engineering Deep Dive](#3-engineering-deep-dive)
4. [Practical Playbook](#4-practical-playbook)
5. [Starter Snippets](#5-starter-snippets)
6. [What I Would Do Next Week](#6-what-i-would-do-next-week)

---

## 1. CEO Explanation

### What's Happening When You Ask an LLM a Question

Think of running an LLM like a restaurant with a very particular chef:

**The Restaurant Analogy:**
- **Prefill** (Reading the order): The chef must read your entire order word-by-word before cooking anything. For a 500-word instruction + 50-word question, they read all 550 words.
- **Decode** (Cooking and serving): The chef prepares one dish at a time, remembering what they've cooked so far.
- **KV Cache** (The chef's notes): To avoid re-reading their notes from scratch for every dish, they keep a running list of what they've done.

**The Problem:** If 100 customers all have the same 500-word instruction ("You are a helpful assistant who..."), the chef reads that identical instruction 100 times. That's 50,000 redundant word-reads.

### What KV Cache Reuse Means in Business Terms

| Metric | Without Cache Reuse | With Cache Reuse | Business Impact |
|--------|---------------------|------------------|-----------------|
| **Latency (TTFT)** | 500-2000ms | 50-200ms | Users see responses 10x faster |
| **Throughput** | X requests/second | 2-10X requests/second | Same hardware serves more users |
| **Cost per query** | $Y | $0.1-0.5Y | Dramatic compute savings |
| **GPU utilization** | 30-50% | 70-90% | Better return on GPU investment |

**Real ROI Example** (Source: SGLang paper, arXiv:2312.07104):
- Customer support chatbot with 500-token system prompt
- 100,000 queries/day
- Without caching: 50M tokens processed for system prompts alone
- With caching: 500 tokens processed once, reused 99,999 times
- **Annual savings: ~$180,000** on system prompts alone

### "Radix Tree" Intuition (No Math Required)

Imagine a filing cabinet for the chef's notes:

```
Traditional Filing (Hash Table):
┌─────────────────────────────────────────────────────┐
│ Drawer 1: "You are a helpful assistant. What is 2+2?" → Notes for this exact question │
│ Drawer 2: "You are a helpful assistant. What is 3+3?" → Separate notes (duplicate!)   │
│ Drawer 3: "You are a helpful assistant. Translate X" → More duplicates!               │
└─────────────────────────────────────────────────────┘
Problem: "You are a helpful assistant" is stored 3 times!

Radix Tree Filing (Smart Organization):
┌─────────────────────────────────────────────────────┐
│ Main Folder: "You are a helpful assistant."         │
│   ├── Subfolder: "What is 2+2?" → Additional notes  │
│   ├── Subfolder: "What is 3+3?" → Additional notes  │
│   └── Subfolder: "Translate X"  → Additional notes  │
└─────────────────────────────────────────────────────┘
Benefit: Common prefix stored ONCE, shared by all!
```

**The Key Insight:** A radix tree automatically discovers and shares common beginnings (prefixes) of requests. You don't need to manually specify what to cache—it happens automatically.

### Decision Table: When to Choose SGLang vs vLLM

| Use Case | Choose SGLang | Choose vLLM | Why |
|----------|---------------|-------------|-----|
| **Chatbot with system prompt** | ✅ | ○ | System prompt cached, 10-50x speedup on prefill |
| **Multi-turn conversations** | ✅ | ○ | Each turn builds on cached history |
| **RAG with shared documents** | ✅ | ○ | Document context computed once |
| **Agent loops (tool calling)** | ✅ | ○ | Tool descriptions cached across calls |
| **Batch of unrelated queries** | ○ | ✅ | No shared prefixes to exploit |
| **Maximum raw throughput** | ○ | ✅ | vLLM has more mature optimizations |
| **Widest model support** | ○ | ✅ | vLLM supports more architectures |

**Strategic Recommendation:**
- **For interactive applications** (chatbots, assistants, agents): SGLang's radix cache provides significant latency and cost benefits
- **For batch processing** (embeddings, one-off queries): vLLM's memory efficiency matters more
- **For production stability**: Both are production-ready; vLLM has a larger community

---

## 2. Intern Onboarding Explanation

### Step-by-Step: How LLM Inference Works

#### Step 1: Tokenization
```
User Input: "What is the capital of France?"
     │
     ▼
Tokenizer: [151644, 3838, 374, 279, 6864, 315, 9822, 30]
           ("What", "is", "the", "capital", "of", "France", "?")
```

#### Step 2: Prefill Phase (Processing the Input)
```
┌─────────────────────────────────────────────────────────────────────┐
│                         PREFILL PHASE                                │
│                                                                      │
│  Input tokens: [What, is, the, capital, of, France, ?]               │
│                                                                      │
│  For EACH token, at EACH layer (e.g., 32 layers for Llama-7B):       │
│                                                                      │
│  Token 1 "What":                                                     │
│    Layer 1: hidden → Key₁¹, Value₁¹ → store in KV cache              │
│    Layer 2: hidden → Key₁², Value₁² → store in KV cache              │
│    ...                                                               │
│    Layer 32: hidden → Key₁³², Value₁³² → store in KV cache           │
│                                                                      │
│  Token 2 "is":                                                       │
│    (same process for all 32 layers)                                  │
│    ...                                                               │
│                                                                      │
│  Result: KV cache now contains K,V for all 7 tokens × 32 layers      │
│          = 224 (K,V) pairs stored                                    │
└─────────────────────────────────────────────────────────────────────┘
```

#### Step 3: What's Stored in the KV Cache

```python
# Simplified representation of KV cache structure
# From: minisgl/kvcache/mha_pool.py

class MHAKVCache:
    def __init__(self, num_layers, num_pages, num_kv_heads, head_dim, device, dtype):
        # Shape: [2, num_layers, num_pages, num_kv_heads, head_dim]
        # 2 = key + value
        self._kv_buffer = torch.empty(
            (2, num_layers, num_pages, num_kv_heads, head_dim),
            device=device, dtype=dtype
        )
    
    # Example for Llama-7B:
    # - 32 layers
    # - 32 KV heads
    # - 128 head dimension
    # - 2 bytes per element (float16)
    # Memory per token: 2 × 32 × 32 × 128 × 2 = 512 KB per token!
```

**Why the KV cache is expensive to recompute:**
- Each token requires a full forward pass through all layers
- For a 7B model, this means ~7 billion multiply-add operations per token
- A 500-token system prompt = 3.5 trillion operations
- With caching: do this once, reuse forever

#### Step 4: Decode Phase (Generating Output)

```
┌─────────────────────────────────────────────────────────────────────┐
│                         DECODE PHASE                                 │
│                                                                      │
│  KV Cache: [What, is, the, capital, of, France, ?]                   │
│                                                                      │
│  Generate token 8:                                                   │
│    For each layer:                                                   │
│      1. Compute Query for position 8                                 │
│      2. Attend to ALL cached Keys (positions 1-7)                    │
│      3. Weight ALL cached Values                                     │
│      4. Compute new Key₈, Value₈                                     │
│      5. Add to KV cache                                              │
│    Sample next token → "Paris"                                       │
│                                                                      │
│  Generate token 9:                                                   │
│    Now attending to positions 1-8 (including "Paris")                │
│    Sample → "."                                                      │
│                                                                      │
│  ... continue until EOS or max length                                │
└─────────────────────────────────────────────────────────────────────┘
```

### What "Shared Prefix" Means in Real Workloads

#### Example 1: Chatbot with System Prompt
```
Request 1: [SYSTEM: You are a helpful AI assistant...] + [USER: What is 2+2?]
Request 2: [SYSTEM: You are a helpful AI assistant...] + [USER: Explain Python]
Request 3: [SYSTEM: You are a helpful AI assistant...] + [USER: Write a poem]
           └──────────── SHARED PREFIX ──────────────┘   └─── UNIQUE ────┘
           
Without caching: Process system prompt 3 times
With caching: Process system prompt 1 time, reuse for all 3
```

#### Example 2: Multi-Turn Conversation
```
Turn 1: [System] [User1: Hi!] [Assistant1: Hello!]
Turn 2: [System] [User1: Hi!] [Assistant1: Hello!] [User2: What's your name?]
        └──────────────── CACHED FROM TURN 1 ────────────────┘
Turn 3: [System] [User1] [Asst1] [User2] [Asst2] [User3: Tell me a joke]
        └──────────── ALL CACHED FROM TURN 2 ──────────────┘
```

#### Example 3: Agent Tool Calls
```
Call 1: [Tools: {search}, {calculator}, {weather}...] + [Query: search for X]
Call 2: [Tools: {search}, {calculator}, {weather}...] + [Query: calculate Y]
Call 3: [Tools: {search}, {calculator}, {weather}...] + [Query: weather in Z]
        └─────────── TOOL DESCRIPTIONS CACHED ──────────────┘
```

### ASCII Diagrams

#### Diagram A: A Small Radix Tree of Token Prefixes

```
                                ┌──────────┐
                                │   ROOT   │
                                │  (empty) │
                                └────┬─────┘
                                     │
              ┌──────────────────────┼──────────────────────┐
              │                      │                      │
              ▼                      ▼                      ▼
      ┌───────────────┐    ┌───────────────┐    ┌───────────────┐
      │ "You are a    │    │ "Translate    │    │ "Summarize    │
      │  helpful      │    │  the          │    │  the          │
      │  assistant."  │    │  following:"  │    │  following:"  │
      │               │    │               │    │               │
      │ KV: indices   │    │ KV: indices   │    │ KV: indices   │
      │ [0, 1, ..., 9]│    │ [50, 51, ..., │    │ [100, 101,    │
      │               │    │  59]          │    │  ..., 109]    │
      │ ref_count: 2  │    │ ref_count: 0  │    │ ref_count: 1  │
      └───────┬───────┘    └───────────────┘    └───────────────┘
              │
      ┌───────┴───────┐
      │               │
      ▼               ▼
┌───────────┐   ┌───────────┐
│ "What is  │   │ "Please   │
│  the      │   │  explain  │
│  capital" │   │  quantum" │
│           │   │           │
│ KV: [10,  │   │ KV: [20,  │
│  ..., 14] │   │  ..., 24] │
│ ref_count:│   │ ref_count:│
│  1        │   │  0        │
└───────────┘   └───────────┘

Legend:
- Each node stores: token sequence (key) + KV cache indices (value)
- ref_count > 0: Currently in use by active requests (cannot be evicted)
- ref_count = 0: Can be evicted if memory is needed (LRU order)
```

#### Diagram B: KV Cache Reuse Flow When a New Request Partially Matches

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    NEW REQUEST ARRIVES                                       │
│                                                                              │
│  Request: "You are a helpful assistant. What is the capital of France?"     │
│  Tokens:  [You, are, a, helpful, assistant, ., What, is, the, capital, ...] │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    STEP 1: RADIX TREE WALK                                   │
│                                                                              │
│  Start at ROOT                                                               │
│    │                                                                         │
│    ├─► Check children for first token "You"                                 │
│    │   Found! Child node: "You are a helpful assistant."                    │
│    │                                                                         │
│    └─► Walk into this node                                                  │
│        Compare token-by-token:                                               │
│          Input[0] "You" == Node[0] "You" ✓                                  │
│          Input[1] "are" == Node[1] "are" ✓                                  │
│          ...                                                                 │
│          Input[9] "." == Node[9] "." ✓                                      │
│          All 10 tokens MATCH!                                               │
│                                                                              │
│    Continue walking...                                                       │
│        Check children for "What" → Found child "What is the capital"        │
│          Input[10-14] all match!                                            │
│        Check children for "of" → NOT FOUND                                  │
│        STOP HERE                                                            │
│                                                                              │
│  Result: 15 tokens cached, need to compute remaining tokens                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    STEP 2: PAGE TABLE SETUP                                  │
│                                                                              │
│  Page Table for this request:                                                │
│  ┌────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬───┐│
│  │  0 │  1 │  2 │  3 │  4 │  5 │  6 │  7 │  8 │  9 │ 10 │ 11 │ 12 │ 13 │14 ││
│  └────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴───┘│
│  ◄─────────────── CACHED (copy from radix tree) ──────────────►│            │
│                                                                              │
│  Then allocate new slots for remaining tokens:                               │
│  ┌────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬───┐│
│  │  0 │  1 │  2 │...│ 14 │200 │201 │202 │203 │    │    │    │    │    │   ││
│  └────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴───┘│
│                       └─► NEW (allocated for "of France?...")               │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    STEP 3: GPU FORWARD PASS                                  │
│                                                                              │
│  For each transformer layer:                                                 │
│    ┌────────────────────────────────────────────────────────────────┐       │
│    │ Load K,V from indices [0-14] ← Already computed! (CACHED)      │       │
│    │ Compute K,V for tokens [15-18] "of France?" (NEW)              │       │
│    │ Store new K,V at indices [200-203]                             │       │
│    │ Run attention over ALL 19 K,V pairs                            │       │
│    └────────────────────────────────────────────────────────────────┘       │
│                                                                              │
│  Speedup: 19 tokens → only 4 computed = 4.75x faster prefill!               │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    STEP 4: INSERT INTO RADIX TREE                            │
│                                                                              │
│  After request completes:                                                    │
│    - Unlock the cached handle (decrement ref_count)                         │
│    - Insert new suffix into tree for future reuse                           │
│                                                                              │
│  Tree before:                           Tree after:                          │
│  ┌─────────────────┐                    ┌─────────────────┐                 │
│  │ "You are a      │                    │ "You are a      │                 │
│  │  helpful        │                    │  helpful        │                 │
│  │  assistant."    │                    │  assistant."    │                 │
│  └────────┬────────┘                    └────────┬────────┘                 │
│           │                                      │                          │
│           ▼                                      ▼                          │
│  ┌─────────────────┐                    ┌─────────────────┐                 │
│  │ "What is the    │                    │ "What is the    │                 │
│  │  capital"       │                    │  capital"       │                 │
│  └─────────────────┘                    └────────┬────────┘                 │
│                                                  │                          │
│                                                  ▼                          │
│                                         ┌─────────────────┐                 │
│                                         │ "of France?"    │ ◄── NEW!       │
│                                         │ KV: [200-203]   │                 │
│                                         └─────────────────┘                 │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### Diagram C: Block-Based Caching (vLLM) vs Token/Prefix Tree Reuse (SGLang)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                 vLLM: BLOCK-BASED CACHING (PagedAttention + APC)            │
│                                                                              │
│  Memory organized as FIXED-SIZE BLOCKS (e.g., 16 tokens per block):         │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────┐     │
│  │                    PHYSICAL BLOCK POOL                              │     │
│  │  ┌────┐┌────┐┌────┐┌────┐┌────┐┌────┐┌────┐┌────┐┌────┐┌────┐      │     │
│  │  │ B0 ││ B1 ││ B2 ││ B3 ││ B4 ││ B5 ││ B6 ││ B7 ││ B8 ││ B9 │ ... │     │
│  │  │16tk││16tk││16tk││16tk││16tk││16tk││16tk││16tk││16tk││16tk│      │     │
│  │  └────┘└────┘└────┘└────┘└────┘└────┘└────┘└────┘└────┘└────┘      │     │
│  └────────────────────────────────────────────────────────────────────┘     │
│                                                                              │
│  Prefix Caching via BLOCK HASH:                                              │
│  ┌──────────────────────────────────────────────────────────────────┐       │
│  │ hash(tokens[0:16])   → Block B0                                   │       │
│  │ hash(tokens[16:32])  → Block B1  (if hash matches existing block) │       │
│  │ hash(tokens[32:48])  → Block B2  (if hash matches)                │       │
│  └──────────────────────────────────────────────────────────────────┘       │
│                                                                              │
│  GRANULARITY: Block-level (16 tokens)                                        │
│  MATCHING: Hash-based (must match entire block exactly)                      │
│                                                                              │
│  Example: 37-token prefix                                                    │
│    Block 0 (tokens 0-15):  hash → CACHED ✓                                  │
│    Block 1 (tokens 16-31): hash → CACHED ✓                                  │
│    Block 2 (tokens 32-36): hash → MISS ✗ (only 5 tokens, not 16)           │
│                                                                              │
│  Result: 32 tokens cached, 5 tokens must be recomputed                      │
│  (Even though all 37 tokens were seen before!)                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                 SGLang: TOKEN-LEVEL RADIX TREE CACHING                       │
│                                                                              │
│  Memory as TOKEN-LEVEL SLOTS (1 token per slot):                            │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────┐     │
│  │                    TOKEN SLOT POOL                                  │     │
│  │  ┌──┐┌──┐┌──┐┌──┐┌──┐┌──┐┌──┐┌──┐┌──┐┌──┐┌──┐┌──┐┌──┐┌──┐ ...     │     │
│  │  │S0││S1││S2││S3││S4││S5││S6││S7││S8││S9││10││11││12││13│         │     │
│  │  └──┘└──┘└──┘└──┘└──┘└──┘└──┘└──┘└──┘└──┘└──┘└──┘└──┘└──┘         │     │
│  └────────────────────────────────────────────────────────────────────┘     │
│                                                                              │
│  Prefix Caching via RADIX TREE:                                              │
│  ┌──────────────────────────────────────────────────────────────────┐       │
│  │                        ROOT                                       │       │
│  │                          │                                        │       │
│  │              ┌───────────┴───────────┐                            │       │
│  │              ▼                       ▼                            │       │
│  │     ┌────────────────┐     ┌────────────────┐                     │       │
│  │     │ "You are a..." │     │ "Translate..." │                     │       │
│  │     │ slots: [0-9]   │     │ slots: [50-59] │                     │       │
│  │     └───────┬────────┘     └────────────────┘                     │       │
│  │             │                                                     │       │
│  │             ▼                                                     │       │
│  │     ┌────────────────┐                                            │       │
│  │     │ "What is the"  │                                            │       │
│  │     │ slots: [10-14] │                                            │       │
│  │     └────────────────┘                                            │       │
│  └──────────────────────────────────────────────────────────────────┘       │
│                                                                              │
│  GRANULARITY: Token-level (1 token)                                          │
│  MATCHING: Prefix tree walk (partial matches work!)                          │
│                                                                              │
│  Example: 37-token prefix (same as above)                                    │
│    Tree walk: Match tokens 0-36 exactly → ALL 37 CACHED ✓                   │
│                                                                              │
│  Result: All 37 tokens cached, 0 tokens recomputed!                         │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                    COMPARISON SUMMARY                                        │
│                                                                              │
│  ┌─────────────────────┬───────────────────────┬──────────────────────────┐ │
│  │ Aspect              │ vLLM (Block-Based)    │ SGLang (Radix Tree)      │ │
│  ├─────────────────────┼───────────────────────┼──────────────────────────┤ │
│  │ Granularity         │ 16 tokens per block   │ 1 token (exact)          │ │
│  │ Matching method     │ Hash of entire block  │ Token-by-token tree walk │ │
│  │ Partial matches     │ Block boundaries only │ Any prefix length        │ │
│  │ Memory overhead     │ Lower (block headers) │ Higher (tree nodes)      │ │
│  │ Lookup complexity   │ O(1) hash lookup      │ O(prefix_length)         │ │
│  │ Best for            │ Exact block matches   │ Partial prefix sharing   │ │
│  └─────────────────────┴───────────────────────┴──────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Engineering Deep Dive

### 3.1 Matching Granularity: Token-Level vs Block-Hash

#### SGLang Radix Tree (Token-Level)

From `minisgl/kvcache/radix_manager.py`:

```python
class RadixTreeNode:
    def __init__(self, tic: int | None = None) -> None:
        self.children: Dict[int, RadixTreeNode] = {}  # first_token_id → child
        self._parent: RadixTreeNode | None = None
        self.ref_count: int = 0          # Active request count
        self.timestamp = tic or time.monotonic_ns()  # For LRU eviction
        
        # The cached data
        self._key: torch.Tensor    # Token IDs (e.g., [151644, 872, 198, ...])
        self._value: torch.Tensor  # KV cache indices (e.g., [0, 1, 2, ...])

    def get_match_len(self, input_ids: torch.Tensor) -> int:
        # Uses fast C++ kernel for SIMD-optimized comparison
        from minisgl.kernel import fast_compare_key
        return fast_compare_key(self._key, input_ids)
```

**Fast comparison kernel** (from `minisgl/kernel/csrc/src/radix.cpp`):

```cpp
// Uses std::mismatch for SIMD-optimized comparison
auto fast_compare_key(const TensorView a, const TensorView b) -> size_t {
    const auto common_len = std::min(a.size(0), b.size(0));
    const auto a_ptr = static_cast<const int32_t*>(a.data_ptr());
    const auto b_ptr = static_cast<const int32_t*>(b.data_ptr());
    const auto diff_pos = std::mismatch(a_ptr, a_ptr + common_len, b_ptr);
    return static_cast<size_t>(diff_pos.first - a_ptr);
}
```

#### vLLM Block-Hash Caching

From vLLM's design doc (https://docs.vllm.ai/en/v0.9.0/design/automatic_prefix_caching.html):

```python
# Conceptual representation
class BlockHashCache:
    def __init__(self, block_size: int = 16):
        self.block_size = block_size
        self.hash_to_block: Dict[int, PhysicalBlock] = {}
    
    def get_cached_blocks(self, tokens: List[int]) -> List[PhysicalBlock]:
        cached_blocks = []
        for i in range(0, len(tokens), self.block_size):
            block_tokens = tokens[i:i + self.block_size]
            if len(block_tokens) < self.block_size:
                break  # Incomplete block, can't cache
            block_hash = hash(tuple(block_tokens))
            if block_hash in self.hash_to_block:
                cached_blocks.append(self.hash_to_block[block_hash])
            else:
                break  # Cache miss, stop here
        return cached_blocks
```

**Key Differences:**

| Aspect | SGLang Radix | vLLM Block-Hash |
|--------|--------------|-----------------|
| **Minimum reuse unit** | 1 token | 16 tokens (block_size) |
| **Partial prefix** | ✅ Any length | ❌ Only at block boundaries |
| **Memory for metadata** | Higher (tree nodes) | Lower (hash table) |
| **Lookup cost** | O(prefix_depth × compare_cost) | O(num_blocks × hash_cost) |
| **Insertion** | Tree walk + node creation | Hash computation + table insert |

### 3.2 Insert / Lookup / Eviction

#### Lookup (`match_prefix`)

From `minisgl/kvcache/radix_manager.py`:

```python
def _walk(self, input_ids: torch.Tensor) -> Tuple[RadixTreeNode, int]:
    """Walk the tree to find the longest matching prefix."""
    prefix_len = 0
    node = self.root_node
    tic = time.monotonic_ns()

    while prefix_len < len(input_ids):
        this_id = int(input_ids[prefix_len].item())
        if this_id not in node.children:
            return node, prefix_len  # No child matches

        node = node.children[this_id]
        
        # Compare this node's key with input
        match_len = node.get_match_len(input_ids[prefix_len:])
        prefix_len += match_len

        # If partial match, split the node
        if match_len != node.length:
            node = node._split_at(match_len)
            return node, prefix_len

        # Update timestamp for LRU
        node.timestamp = tic

    return node, prefix_len
```

**Complexity:** O(prefix_depth) where depth is typically O(log N) for balanced workloads

#### Insertion (`insert_prefix`)

```python
def insert_prefix(self, input_ids: torch.Tensor, indices: torch.Tensor) -> int:
    node, prefix_len = self._walk(input_ids)
    
    if prefix_len < len(input_ids):
        # Create new node for unmatched suffix
        new_node = RadixTreeNode()
        new_node.set_key_value(
            input_ids[prefix_len:],   # Unmatched tokens
            indices[prefix_len:]      # Their KV cache indices
        )
        new_node.set_parent(node)
        self.evictable_size += new_node.length
    
    return prefix_len  # Caller frees duplicate indices
```

#### Eviction (LRU on Leaves)

```python
def evict(self, size: int) -> torch.Tensor:
    """Evict least-recently-used leaves until we free 'size' tokens."""
    leave_nodes = self._collect_leave_nodes_for_evict()  # ref_count == 0
    heapq.heapify(leave_nodes)  # Min-heap by timestamp
    
    evicted_indices = []
    evicted_size = 0

    while evicted_size < size:
        node = heapq.heappop(leave_nodes)
        assert node.ref_count == 0 and node.is_leaf()
        
        evicted_size += node.length
        evicted_indices.append(node.value)
        self.evictable_size -= node.length
        
        # Remove from parent's children
        parent = node.parent
        del parent.children[int(node._key[0].item())]
        
        # If parent becomes a leaf with ref_count == 0, add to heap
        if parent.is_leaf() and parent.ref_count == 0:
            heapq.heappush(leave_nodes, parent)

    return torch.cat(evicted_indices)
```

**Eviction Strategy:**
- LRU (Least Recently Used) based on timestamp
- Only evicts leaves (preserves shared prefixes)
- Protected nodes (ref_count > 0) are never evicted
- Parent becomes evictable only when all children are evicted

### 3.3 Memory Layout and Paged KV Cache

From `minisgl/kvcache/mha_pool.py`:

```python
class MHAKVCache(BaseKVCache):
    def __init__(self, num_layers, num_pages, local_kv_heads, head_dim, device, dtype):
        # Shape: [2, num_layers, num_pages, num_kv_heads, head_dim]
        self._kv_buffer = torch.empty(
            (2, num_layers, num_pages, local_kv_heads, head_dim),
            device=device, dtype=dtype
        )
    
    def k_cache(self, layer_idx: int) -> torch.Tensor:
        return self._kv_buffer[0, layer_idx]  # [num_pages, num_heads, head_dim]
    
    def v_cache(self, layer_idx: int) -> torch.Tensor:
        return self._kv_buffer[1, layer_idx]  # [num_pages, num_heads, head_dim]
```

**Integration with Radix Tree:**
- Radix tree stores **indices** into the KV buffer, not the actual tensors
- This enables efficient sharing: multiple requests point to same physical KV slots
- Memory layout is contiguous for GPU efficiency

### 3.4 Scheduler Interactions

From `minisgl/scheduler/scheduler.py`:

```python
def overlap_loop(self, last_data: ForwardData | None) -> ForwardData | None:
    """
    Overlap CPU scheduling with GPU computation.
    
    Timeline:
    GPU: [Batch N-1 compute ][Batch N compute    ][Batch N+1 compute ]
    CPU:        [Process N-2][Schedule N         ][Process N-1       ]
                              ▲                    ▲
                              └─ Hidden behind GPU └─ Hidden behind GPU
    """
    # 1. Receive new requests (non-blocking if we have work)
    blocking = not (last_data or self.prefill_manager.runnable or self.decode_manager.runnable)
    for msg in self.receive_msg(blocking=blocking):
        self._process_one_msg(msg)

    # 2. Schedule next batch (includes radix lookup!)
    forward_input = self._schedule_next_batch()
    
    ongoing_data = None
    if forward_input is not None:
        # 3. Execute on GPU (in engine's stream)
        with self.engine_stream_ctx:
            self.engine.stream.wait_stream(self.stream)
            ongoing_data = (forward_input, self._forward(forward_input))

    # 4. Process results from LAST batch (overlap!)
    self._process_last_data(last_data, ongoing_data)
    
    return ongoing_data
```

**Cache-Aware Scheduling** (from `minisgl/scheduler/prefill.py`):

```python
class PrefillAdder:
    def _try_allocate_one(self, req: PendingReq) -> Tuple[BaseCacheHandle, int] | None:
        # 1. Check radix cache for prefix match
        handle, match_indices = self.cache_manager.match_req(req)
        cached_len = handle.cached_len
        
        # 2. Calculate actual memory needed (only for new tokens)
        extend_len = req.input_len - cached_len
        estimated_len = extend_len + req.output_len
        
        # 3. Check if we have enough cache space
        if estimated_len + self.reserved_size > self.cache_manager.available_size:
            return None  # Queue for later
        
        # 4. Lock handle (prevents eviction during use)
        self.cache_manager.lock(handle)
        
        # 5. Copy cached KV indices to page table
        if cached_len > 0:
            page_entry[:cached_len].copy_(match_indices)
        
        return handle, table_idx
```

### 3.5 Strengths and Weaknesses

#### Best-Case Speedups

**Scenario: 500-token system prompt, 50-token user query**

| Metric | Without Radix | With Radix | Improvement |
|--------|---------------|------------|-------------|
| Tokens to compute (1st request) | 550 | 550 | 0% |
| Tokens to compute (2nd+ request) | 550 | 50 | **11x** |
| Memory for 10 identical prompts | 10 × 550 | 1 × 550 | **10x** |

**Observed Speedups** (from SGLang paper):
- Multi-turn chat: **1.5-5x** throughput improvement
- RAG with shared context: **3-10x** prefill speedup
- Few-shot prompting: **2-8x** depending on example length

#### Worst-Case Overheads

**Scenario: All unique, unrelated queries**

| Overhead Type | Cost | When It Matters |
|---------------|------|-----------------|
| Tree walk per request | ~10-50 μs | Negligible vs GPU compute |
| Node creation | ~5-10 μs | Negligible |
| Memory for tree structure | ~100 bytes/node | <1% of KV cache memory |
| Eviction overhead | ~50-200 μs | Only when cache is full |

**When Radix Cache Hurts:**
- Cache churn: Rapid eviction/insertion cycles waste CPU time
- Low reuse: Tree overhead without matching benefit
- Memory pressure: Tree metadata competes with KV cache

#### Fragmentation / Churn Scenarios

```
┌─────────────────────────────────────────────────────────────────┐
│                 PROBLEMATIC PATTERN: CACHE CHURN                 │
│                                                                  │
│ Time 0: Cache fills with prefixes A, B, C, D, E                 │
│ Time 1: New prefix F arrives → evict A (LRU)                    │
│ Time 2: Prefix A requested again → evict B, recompute A         │
│ Time 3: Prefix B requested again → evict C, recompute B         │
│ ... cycle continues ...                                         │
│                                                                  │
│ Symptoms:                                                        │
│   - Low cache hit rate despite repeated prefixes                 │
│   - High eviction count                                         │
│   - CPU overhead from constant tree modifications               │
│                                                                  │
│ Fixes:                                                           │
│   1. Increase cache size (--num-gpu-blocks-override)            │
│   2. Reduce concurrent request diversity                         │
│   3. Pin frequently-used prefixes (if supported)                │
└─────────────────────────────────────────────────────────────────┘
```

#### Multi-Tenant Pitfalls

**Problem:** Different tenants with different system prompts compete for cache

```
Tenant A: "You are a customer support agent for Company A..."
Tenant B: "You are a sales assistant for Company B..."
Tenant C: "You are a technical support agent for Company C..."

Each tenant's prefix evicts the others → low hit rate for all
```

**Solutions:**
1. **Per-tenant cache partitioning** (not in base SGLang)
2. **Weighted eviction** based on tenant priority
3. **Separate instances** for high-value tenants

#### Correctness / Isolation Concerns

**LoRA/Adapter Compatibility:**
```
┌─────────────────────────────────────────────────────────────────┐
│                    LORA COMPATIBILITY ISSUE                      │
│                                                                  │
│ Problem:                                                         │
│   Same token sequence with different LoRA adapters               │
│   produces DIFFERENT KV values!                                  │
│                                                                  │
│   Request 1: [tokens] + LoRA_A → KV_A                           │
│   Request 2: [tokens] + LoRA_B → Should get KV_B, not KV_A!     │
│                                                                  │
│ Current behavior in base SGLang:                                 │
│   ⚠️ May incorrectly share KV across different adapters          │
│                                                                  │
│ Solution:                                                        │
│   Include adapter ID in cache key (e.g., (adapter_id, tokens))  │
│   SGLang's full implementation handles this                     │
└─────────────────────────────────────────────────────────────────┘
```

### 3.6 Common Failure Modes + Symptoms + Fixes

| Failure Mode | Symptoms | Root Cause | Fix |
|--------------|----------|------------|-----|
| **Cache thrashing** | Low hit rate, high eviction rate, CPU spikes | Cache too small for working set | Increase cache size or reduce concurrency |
| **Memory OOM** | CUDA OOM during allocation | Too many locked handles | Reduce max_running_requests |
| **Slow prefill** | High TTFT despite expected cache hits | Tokens don't match (tokenizer difference) | Ensure consistent tokenization |
| **Stale cache** | Wrong outputs | LoRA/adapter mismatch | Include adapter in cache key |
| **Tree imbalance** | Slow lookup times | Single long prefix chain | Usually self-correcting; monitor depth |

---

## 4. Practical Playbook

### 4.1 Checklist for Maximizing Cache Hit Rate

#### Prompt Design

- [ ] **Use consistent system prompts** across requests
- [ ] **Place static content first** (system prompt, examples, context)
- [ ] **Place dynamic content last** (user query, specific parameters)
- [ ] **Avoid random elements in prompts** (timestamps, UUIDs at the start)
- [ ] **Batch similar requests together** to maximize sharing window

```
❌ BAD (timestamp breaks prefix):
"[2024-01-15 10:30:00] You are a helpful assistant. What is 2+2?"

✅ GOOD (static prefix first):
"You are a helpful assistant. [Current time: 2024-01-15] What is 2+2?"
```

#### Serving Patterns

- [ ] **Group requests by system prompt** (route similar prompts to same instance)
- [ ] **Warm up cache** with common prefixes at startup
- [ ] **Use sticky sessions** for multi-turn conversations
- [ ] **Size cache appropriately** for your working set

### 4.2 Metrics to Instrument

```python
# Key metrics to track
metrics = {
    # Cache effectiveness
    "cache_hit_rate": "matched_tokens / total_input_tokens",
    "cache_hit_rate_by_prefix": "per system prompt breakdown",
    
    # Latency
    "ttft_p50_p99": "Time to first token (prefill latency)",
    "tpot_p50_p99": "Time per output token (decode latency)",
    "e2e_latency_p50_p99": "End-to-end request latency",
    
    # Throughput
    "tokens_per_second": "total output tokens / time",
    "requests_per_second": "completed requests / time",
    
    # Memory
    "kv_cache_utilization": "used_slots / total_slots",
    "eviction_rate": "evictions / second",
    "tree_node_count": "number of nodes in radix tree",
    
    # Resource
    "gpu_utilization": "percentage GPU active",
    "gpu_memory_used": "bytes allocated",
    "cpu_scheduling_time": "time spent in scheduler",
}
```

### 4.3 Benchmark Plan

#### Microbenchmarks

**1. Prefix Match Speed**
```bash
# Measure radix tree lookup latency for various prefix lengths
python -c "
import torch
import time
from minisgl.kvcache import RadixCacheManager

manager = RadixCacheManager(torch.device('cpu'))

# Insert prefixes of various lengths
for length in [100, 500, 1000, 5000]:
    tokens = torch.arange(length, dtype=torch.int32)
    indices = torch.arange(length, dtype=torch.int32)
    manager.insert_prefix(tokens, indices)
    
    # Measure lookup time
    start = time.perf_counter_ns()
    for _ in range(1000):
        manager.match_prefix(tokens)
    elapsed = (time.perf_counter_ns() - start) / 1000
    print(f'Prefix length {length}: {elapsed:.2f} ns per lookup')
"
```

**2. Eviction Performance**
```bash
# Measure eviction overhead under pressure
python -c "
import torch
import time
from minisgl.kvcache import RadixCacheManager

manager = RadixCacheManager(torch.device('cpu'))

# Fill cache with many small prefixes
for i in range(10000):
    tokens = torch.tensor([i * 100 + j for j in range(10)], dtype=torch.int32)
    indices = torch.tensor([i * 10 + j for j in range(10)], dtype=torch.int32)
    manager.insert_prefix(tokens, indices)

# Measure eviction time
start = time.perf_counter_ns()
evicted = manager.evict(50000)  # Evict half
elapsed = (time.perf_counter_ns() - start) / 1e6
print(f'Evicted {len(evicted)} slots in {elapsed:.2f} ms')
"
```

**3. Insertion/Lookup Under Load**
```python
# Simulate realistic request pattern
import concurrent.futures
import random

def simulate_request(manager, system_prompt, query):
    tokens = torch.tensor(system_prompt + query, dtype=torch.int32)
    handle, indices = manager.match_prefix(tokens)
    # ... use cache ...
    manager.unlock(handle)

# Run with multiple threads
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
    futures = []
    for _ in range(1000):
        system = [1, 2, 3, 4, 5]  # Fixed system prompt
        query = [random.randint(100, 200) for _ in range(10)]  # Random query
        futures.append(executor.submit(simulate_request, manager, system, query))
```

#### Realistic Workloads

**1. Multi-Turn Chat Benchmark**
```bash
# Simulate multi-turn conversation workload
python benchmark/online/bench_simple.py \
    --model "Qwen/Qwen3-0.6B" \
    --num-conversations 100 \
    --turns-per-conversation 5 \
    --system-prompt-length 500 \
    --user-message-length 50 \
    --output-file chat_benchmark.json
```

**2. Agent/Tool Calling Benchmark**
```bash
# Simulate agent loop with tool descriptions
python benchmark/online/bench_simple.py \
    --model "Qwen/Qwen3-0.6B" \
    --num-requests 500 \
    --shared-prefix-length 1000 \  # Tool descriptions
    --unique-suffix-length 100 \   # Specific queries
    --output-file agent_benchmark.json
```

### 4.4 Tuning Knobs

| Knob | Flag/Config | Effect | Recommendation |
|------|-------------|--------|----------------|
| **Cache type** | `--cache radix` or `--cache naive` | Radix enables prefix sharing; naive is simpler | Use radix for shared prefixes |
| **Max running requests** | `--max-running-requests N` | Limits concurrent requests (protects memory) | Start at 256, tune based on OOM |
| **Prefill chunk size** | `--max-prefill-length N` | Splits long prefills into chunks | 8192 default; reduce if OOM |
| **CUDA graph** | `--cuda-graph-max-bs N` | Captures decode kernels for replay | Enable (default); set to 0 to disable |
| **Overlap scheduling** | `DISABLE_OVERLAP_SCHEDULING=1` | Disables CPU/GPU overlap | Keep enabled (default) |

---

## 5. Starter Snippets

### 5.1 SGLang Server with Radix Cache

```bash
# Start SGLang server (radix cache enabled by default)
python -m minisgl \
    --model "Qwen/Qwen3-0.6B" \
    --host 0.0.0.0 \
    --port 8000 \
    --cache radix \
    --max-running-requests 128 \
    --max-prefill-length 8192

# Or with the full SGLang package:
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --host 0.0.0.0 \
    --port 30000 \
    --enable-prefix-caching
```

### 5.2 vLLM Server with Prefix Caching

```bash
# Start vLLM server with automatic prefix caching
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --enable-prefix-caching \
    --block-size 16 \
    --max-model-len 8192

# Or via Python API
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    enable_prefix_caching=True,  # Enable APC
    block_size=16,               # Block size for caching
)
```

### 5.3 Request Flow with Caching Decisions (Pseudo-Code)

```python
# Simplified request flow showing where caching decisions happen

class LLMServer:
    def __init__(self):
        self.radix_cache = RadixCacheManager()
        self.kv_pool = KVCachePool(num_pages=10000)
        self.scheduler = Scheduler()
    
    def handle_request(self, tokens: List[int]) -> str:
        """
        Complete request flow from input to output.
        """
        # ═══════════════════════════════════════════════════════════
        # STEP 1: CACHE LOOKUP
        # ═══════════════════════════════════════════════════════════
        # Time: ~10-50 μs (CPU)
        # Where: radix_cache.match_prefix()
        
        handle, cached_indices = self.radix_cache.match_prefix(tokens[:-1])
        cached_len = handle.cached_len
        
        # ═══════════════════════════════════════════════════════════
        # STEP 2: LOCK HANDLE (prevent eviction during use)
        # ═══════════════════════════════════════════════════════════
        # Time: ~1 μs (CPU)
        # Where: radix_cache.lock_handle()
        
        self.radix_cache.lock_handle(handle)
        
        # ═══════════════════════════════════════════════════════════
        # STEP 3: ALLOCATE KV SLOTS FOR NEW TOKENS
        # ═══════════════════════════════════════════════════════════
        # Time: ~5-50 μs (CPU), may trigger eviction
        # Where: kv_pool.allocate()
        
        new_len = len(tokens) - cached_len
        new_indices = self.kv_pool.allocate(new_len)
        
        # ═══════════════════════════════════════════════════════════
        # STEP 4: BUILD PAGE TABLE (combine cached + new indices)
        # ═══════════════════════════════════════════════════════════
        # Time: ~10 μs (CPU→GPU copy)
        
        page_table = torch.cat([cached_indices, new_indices])
        
        # ═══════════════════════════════════════════════════════════
        # STEP 5: GPU PREFILL (only uncached tokens!)
        # ═══════════════════════════════════════════════════════════
        # Time: ~10-100 ms (GPU) - THIS IS THE BIG WIN
        # Note: Only computing KV for tokens[cached_len:], not all tokens!
        
        self.model.prefill(
            tokens=tokens[cached_len:],  # Only new tokens!
            kv_indices=new_indices,
            cached_kv_indices=cached_indices,  # Attention reads from here
        )
        
        # ═══════════════════════════════════════════════════════════
        # STEP 6: GPU DECODE (generate tokens one by one)
        # ═══════════════════════════════════════════════════════════
        # Time: ~5-10 ms per token (GPU)
        
        output_tokens = []
        while not done:
            next_token = self.model.decode_one(page_table)
            output_tokens.append(next_token)
            # ... update page_table with new token's KV slot ...
        
        # ═══════════════════════════════════════════════════════════
        # STEP 7: INSERT NEW PREFIX INTO CACHE
        # ═══════════════════════════════════════════════════════════
        # Time: ~10-30 μs (CPU)
        # Where: radix_cache.insert_prefix()
        # Note: Only inserts the newly computed part
        
        self.radix_cache.insert_prefix(tokens, page_table)
        
        # ═══════════════════════════════════════════════════════════
        # STEP 8: UNLOCK HANDLE
        # ═══════════════════════════════════════════════════════════
        # Time: ~1 μs (CPU)
        
        self.radix_cache.unlock_handle(handle)
        
        return self.tokenizer.decode(output_tokens)


# ═══════════════════════════════════════════════════════════════════
# CACHE HIT EXAMPLE
# ═══════════════════════════════════════════════════════════════════

# Request 1: "You are helpful. What is 2+2?"
#   cached_len = 0 (cold start)
#   compute_len = 12 tokens
#   After: Tree contains "You are helpful. What is 2+2?"

# Request 2: "You are helpful. What is 3+3?"
#   cached_len = 9 ("You are helpful. What is")
#   compute_len = 3 ("3+3?")
#   Speedup: 12/3 = 4x faster prefill!
```

---

## 6. What I Would Do Next Week

### Day 1-2: Baseline Measurement

```bash
# 1. Set up a reproducible benchmark environment
git clone https://github.com/sgl-project/sglang.git
cd sglang && pip install -e .

# 2. Run baseline throughput test without caching
python -m sglang.launch_server \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --disable-prefix-caching \
    --port 30000 &

python benchmark/offline/bench.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --num-prompts 1000 \
    --shared-prefix-len 0 \
    --output baseline_no_cache.json

# 3. Run with caching enabled
# (restart server with --enable-prefix-caching)
python benchmark/offline/bench.py \
    --output baseline_with_cache.json
```

### Day 3-4: Workload Analysis

```python
# Analyze your actual production workload
from collections import Counter

def analyze_prefix_sharing(requests: List[str], tokenizer):
    """Measure potential cache hit rate for your workload."""
    all_tokens = [tokenizer.encode(r) for r in requests]
    
    # Build a trie to find common prefixes
    prefix_counts = Counter()
    for tokens in all_tokens:
        for length in range(1, len(tokens) + 1):
            prefix_counts[tuple(tokens[:length])] += 1
    
    # Calculate potential savings
    total_tokens = sum(len(t) for t in all_tokens)
    unique_tokens = len(set(tuple(t) for t in all_tokens))
    
    # Find prefixes that appear multiple times
    reusable_prefixes = {k: v for k, v in prefix_counts.items() if v > 1}
    reusable_tokens = sum(len(k) * (v - 1) for k, v in reusable_prefixes.items())
    
    print(f"Total tokens: {total_tokens}")
    print(f"Potentially reusable: {reusable_tokens} ({100*reusable_tokens/total_tokens:.1f}%)")
    print(f"Most common prefixes:")
    for prefix, count in sorted(reusable_prefixes.items(), key=lambda x: -x[1])[:10]:
        print(f"  {count}x: '{tokenizer.decode(list(prefix)[:20])}...'")
```

### Day 5: Optimization Experiments

```bash
# Experiment 1: Prompt restructuring
# Move system prompt to beginning for better caching

# Experiment 2: Cache size tuning
for cache_fraction in 0.1 0.2 0.4 0.6 0.8; do
    python benchmark/offline/bench.py \
        --gpu-memory-utilization $cache_fraction \
        --output "cache_fraction_${cache_fraction}.json"
done

# Experiment 3: Compare SGLang vs vLLM on YOUR workload
python benchmark/online/bench_simple.py \
    --backend sglang \
    --your-custom-workload \
    --output sglang_results.json

python benchmark/online/bench_simple.py \
    --backend vllm \
    --your-custom-workload \
    --output vllm_results.json
```

### Week 2+: Production Rollout

1. **Instrument cache metrics** in your serving infrastructure
2. **A/B test** cached vs non-cached serving
3. **Monitor for regressions** (correctness, latency outliers)
4. **Document optimal prompt patterns** for your use cases
5. **Set up alerts** for cache churn, eviction rate spikes

---

## References

1. **SGLang Paper**: Zheng et al., "SGLang: Efficient Execution of Structured Language Model Programs", arXiv:2312.07104, 2023
2. **LMSYS Blog on SGLang/RadixAttention**: https://lmsys.org/blog/2024-01-17-sglang/
3. **vLLM Automatic Prefix Caching Design Doc**: https://docs.vllm.ai/en/v0.9.0/design/automatic_prefix_caching.html
4. **SGLang GitHub Repository**: https://github.com/sgl-project/sglang
5. **PagedAttention Paper**: Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention", 2023
6. **Mini-SGLang Implementation** (this codebase): `/workspace/python/minisgl/kvcache/radix_manager.py`

---

*Document generated from analysis of Mini-SGLang implementation and external sources. For production deployments, always benchmark on your specific workload.*
