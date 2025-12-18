# Radix Tree Prefix Cache vs Paged Attention: A Visual Comparison

This document provides a comprehensive visual comparison between **Radix Tree Prefix Cache** (used in SGLang/Mini-SGLang) and **Paged Attention** (used in vLLM), explaining how each approach manages KV cache for LLM inference.

---

## Table of Contents

1. [The Core Problem: KV Cache Management](#1-the-core-problem-kv-cache-management)
2. [Approach 1: Paged Attention (vLLM)](#2-approach-1-paged-attention-vllm)
3. [Approach 2: Radix Tree Prefix Cache (SGLang)](#3-approach-2-radix-tree-prefix-cache-sglang)
4. [Side-by-Side Comparison](#4-side-by-side-comparison)
5. [Step-by-Step Request Serving](#5-step-by-step-request-serving)
6. [Performance & Cost Trade-offs](#6-performance--cost-trade-offs)
7. [When to Use Which](#7-when-to-use-which)

---

## 1. The Core Problem: KV Cache Management

### What is KV Cache?

When an LLM processes text, it computes **Key** and **Value** tensors for each token at each layer. These must be stored and reused during generation.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           THE KV CACHE PROBLEM                                   │
│                                                                                  │
│  Input: "The capital of France is"                                               │
│                                                                                  │
│  Token 1: "The"     → Compute K₁, V₁ → Store in cache                           │
│  Token 2: "capital" → Compute K₂, V₂ → Store in cache                           │
│  Token 3: "of"      → Compute K₃, V₃ → Store in cache                           │
│  Token 4: "France"  → Compute K₄, V₄ → Store in cache                           │
│  Token 5: "is"      → Compute K₅, V₅ → Store in cache                           │
│                                                                                  │
│  Generate Token 6: "Paris"                                                       │
│    → Attention needs K₁,K₂,K₃,K₄,K₅ and V₁,V₂,V₃,V₄,V₅                          │
│    → Without cache: Recompute all K,V (SLOW!)                                    │
│    → With cache: Just look them up (FAST!)                                       │
│                                                                                  │
│  Memory per token per layer (Llama-70B):                                         │
│    K: [128 heads × 128 dim] × 2 bytes = 32 KB                                    │
│    V: [128 heads × 128 dim] × 2 bytes = 32 KB                                    │
│    Total: 64 KB × 80 layers = 5.12 MB per token!                                 │
│                                                                                  │
│  For 100 concurrent users with 4K context each:                                  │
│    100 × 4096 × 5.12 MB = 2 TB of KV cache needed!                               │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### The Two Key Challenges

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           TWO OPTIMIZATION TARGETS                               │
│                                                                                  │
│  ┌─────────────────────────────────┐  ┌─────────────────────────────────────┐   │
│  │     MEMORY EFFICIENCY           │  │     COMPUTE EFFICIENCY              │   │
│  │     (vLLM's Focus)              │  │     (SGLang's Focus)                │   │
│  ├─────────────────────────────────┤  ├─────────────────────────────────────┤   │
│  │                                 │  │                                     │   │
│  │  Problem: Variable-length       │  │  Problem: Redundant computation     │   │
│  │  sequences waste memory         │  │  for shared prefixes                │   │
│  │                                 │  │                                     │   │
│  │  Traditional allocation:        │  │  10 users with same system prompt:  │   │
│  │  ┌────────────────────────┐     │  │                                     │   │
│  │  │ Seq1: ████████░░░░░░░░ │     │  │  "You are a helpful assistant..."   │   │
│  │  │ Seq2: ████░░░░░░░░░░░░ │     │  │                                     │   │
│  │  │ Seq3: ██████████████░░ │     │  │  Without prefix cache:              │   │
│  │  └────────────────────────┘     │  │    → Process 500 tokens × 10        │   │
│  │       ░ = wasted memory         │  │    → 5000 token computations        │   │
│  │                                 │  │                                     │   │
│  │  Solution: Page-based           │  │  With prefix cache:                 │   │
│  │  allocation (like OS memory)    │  │    → Process 500 tokens × 1         │   │
│  │                                 │  │    → Reuse for other 9 users        │   │
│  │                                 │  │    → 500 token computations         │   │
│  └─────────────────────────────────┘  └─────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Approach 1: Paged Attention (vLLM)

### Core Concept

Paged Attention treats GPU memory like an operating system treats RAM: divide it into fixed-size **pages** and allocate them on-demand.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         PAGED ATTENTION ARCHITECTURE                             │
│                                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │                        PHYSICAL PAGE POOL (GPU HBM)                      │    │
│  │                                                                          │    │
│  │  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐       │    │
│  │  │ P0  │ │ P1  │ │ P2  │ │ P3  │ │ P4  │ │ P5  │ │ P6  │ │ P7  │ ...   │    │
│  │  │     │ │     │ │     │ │     │ │     │ │     │ │     │ │     │       │    │
│  │  │ K,V │ │ K,V │ │ K,V │ │ K,V │ │ K,V │ │ K,V │ │ K,V │ │ K,V │       │    │
│  │  │ for │ │ for │ │ for │ │ for │ │ for │ │ for │ │ for │ │ for │       │    │
│  │  │ 16  │ │ 16  │ │ 16  │ │ 16  │ │ 16  │ │ 16  │ │ 16  │ │ 16  │       │    │
│  │  │tokens│ │tokens│ │tokens│ │tokens│ │tokens│ │tokens│ │tokens│ │tokens│       │    │
│  │  └─────┘ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘       │    │
│  │     ▲       ▲       ▲       ▲       ▲       ▲       ▲       ▲          │    │
│  └─────┼───────┼───────┼───────┼───────┼───────┼───────┼───────┼──────────┘    │
│        │       │       │       │       │       │       │       │               │
│  ┌─────┴───────┴───────┴───────┴───────┴───────┴───────┴───────┴──────────┐    │
│  │                         PAGE TABLES (per sequence)                      │    │
│  │                                                                          │    │
│  │  Sequence A (45 tokens):     Sequence B (28 tokens):                     │    │
│  │  ┌───────────────────────┐   ┌───────────────────────┐                   │    │
│  │  │ Logical │ Physical    │   │ Logical │ Physical    │                   │    │
│  │  │ Block   │ Page        │   │ Block   │ Page        │                   │    │
│  │  ├─────────┼─────────────┤   ├─────────┼─────────────┤                   │    │
│  │  │   0     │   P0        │   │   0     │   P3        │                   │    │
│  │  │   1     │   P2        │   │   1     │   P5        │                   │    │
│  │  │   2     │   P6        │   └─────────┴─────────────┘                   │    │
│  │  └─────────┴─────────────┘                                               │    │
│  │                                                                          │    │
│  │  Key insight: Pages are allocated non-contiguously!                      │    │
│  │  No wasted space from pre-allocation.                                    │    │
│  └──────────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Memory Layout Comparison

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    TRADITIONAL vs PAGED MEMORY ALLOCATION                        │
│                                                                                  │
│  TRADITIONAL (Contiguous):                                                       │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │                                                                           │   │
│  │  Seq A: ████████████████████████████████████████████░░░░░░░░░░░░░░░░░░░  │   │
│  │         ◄──────── 45 tokens used ────────►◄─── 19 tokens wasted ───►     │   │
│  │                                                                           │   │
│  │  Seq B: ████████████████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  │   │
│  │         ◄──── 28 tokens used ────►◄────── 36 tokens wasted ──────►       │   │
│  │                                                                           │   │
│  │  Seq C: ████████████████████████████████████████████████████████████████  │   │
│  │         ◄──────────────────── 64 tokens used ────────────────────►       │   │
│  │                                                                           │   │
│  │  Total: 192 slots allocated, 137 used, 55 wasted (28.6% waste)           │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
│  PAGED (Non-contiguous):                                                         │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │                                                                           │   │
│  │  Page Pool: [P0][P1][P2][P3][P4][P5][P6][P7][P8][░░][░░][░░]...          │   │
│  │              ▲   ▲   ▲   ▲   ▲   ▲   ▲   ▲   ▲                           │   │
│  │              │   │   │   │   │   │   │   │   │                           │   │
│  │  Seq A: ─────┴───┼───┴───┼───┼───┼───┴───┼───┼─── (P0,P2,P6 = 3 pages)  │   │
│  │  Seq B: ─────────┼───────┴───┼───┴───────┼───┼─── (P3,P5 = 2 pages)     │   │
│  │  Seq C: ─────────┴───────────┴───────────┴───┴─── (P1,P4,P7,P8 = 4 pgs) │   │
│  │                                                                           │   │
│  │  Total: 9 pages allocated (144 slots), 137 used, 7 wasted (4.9% waste)   │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
│  Memory savings: 28.6% → 4.9% waste = ~24% more efficient!                       │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### How Paged Attention Handles Requests

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    PAGED ATTENTION: REQUEST LIFECYCLE                            │
│                                                                                  │
│  TIME ═══════════════════════════════════════════════════════════════════════►  │
│                                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │ t=0: Request arrives: "What is the capital of France?"                   │    │
│  │                                                                          │    │
│  │      Page Pool: [░░][░░][░░][░░][░░][░░][░░][░░]  (all free)             │    │
│  │      Page Table: (empty)                                                 │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                              │                                                   │
│                              ▼                                                   │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │ t=1: Prefill - Process all 8 input tokens                                │    │
│  │                                                                          │    │
│  │      Allocate 1 page (page size = 16 tokens)                             │    │
│  │      Compute K,V for all 8 tokens                                        │    │
│  │      Store in Page 0                                                     │    │
│  │                                                                          │    │
│  │      Page Pool: [██][░░][░░][░░][░░][░░][░░][░░]                         │    │
│  │                  P0                                                      │    │
│  │      Page Table: [0] → P0                                                │    │
│  │                                                                          │    │
│  │      P0: [What][is][the][capital][of][France][?][░░]...[░░]              │    │
│  │           ◄────────── 8 tokens ──────────►◄── 8 free ──►                 │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                              │                                                   │
│                              ▼                                                   │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │ t=2 to t=9: Decode - Generate 8 tokens one by one                        │    │
│  │                                                                          │    │
│  │      Each step:                                                          │    │
│  │        1. Attention over all cached K,V                                  │    │
│  │        2. Generate next token                                            │    │
│  │        3. Compute K,V for new token                                      │    │
│  │        4. Append to current page (or allocate new if full)               │    │
│  │                                                                          │    │
│  │      After generating "The capital of France is Paris.":                 │    │
│  │                                                                          │    │
│  │      Page Pool: [██][░░][░░][░░][░░][░░][░░][░░]                         │    │
│  │                  P0                                                      │    │
│  │      Page Table: [0] → P0                                                │    │
│  │                                                                          │    │
│  │      P0: [What][is][the][capital][of][France][?][The][capital]...        │    │
│  │           ◄────────────────── 16 tokens (full) ──────────────────►       │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                              │                                                   │
│                              ▼                                                   │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │ t=10: Page full, allocate new page                                       │    │
│  │                                                                          │    │
│  │      Page Pool: [██][██][░░][░░][░░][░░][░░][░░]                         │    │
│  │                  P0  P1                                                  │    │
│  │      Page Table: [0] → P0, [1] → P1                                      │    │
│  │                                                                          │    │
│  │      Continue generating into P1...                                      │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                              │                                                   │
│                              ▼                                                   │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │ t=end: Request complete, free pages                                      │    │
│  │                                                                          │    │
│  │      Page Pool: [░░][░░][░░][░░][░░][░░][░░][░░]  (all free again)       │    │
│  │      Page Table: (empty)                                                 │    │
│  │                                                                          │    │
│  │      Pages P0, P1 returned to free pool for next request                 │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Approach 2: Radix Tree Prefix Cache (SGLang)

### Core Concept

Radix Tree Prefix Cache organizes cached KV data by **token sequences**, enabling automatic detection and reuse of shared prefixes.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                      RADIX TREE PREFIX CACHE ARCHITECTURE                        │
│                                                                                  │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │                           RADIX TREE STRUCTURE                            │   │
│  │                                                                           │   │
│  │                              ┌──────────┐                                 │   │
│  │                              │   ROOT   │                                 │   │
│  │                              │ (empty)  │                                 │   │
│  │                              └────┬─────┘                                 │   │
│  │                                   │                                       │   │
│  │                    ┌──────────────┼──────────────┐                        │   │
│  │                    │              │              │                        │   │
│  │                    ▼              ▼              ▼                        │   │
│  │            ┌───────────┐  ┌───────────┐  ┌───────────┐                    │   │
│  │            │"You are a"│  │"Translate"│  │"Summarize"│                    │   │
│  │            │ helpful   │  │   the     │  │   the     │                    │   │
│  │            │assistant" │  │ following │  │ following │                    │   │
│  │            │           │  │           │  │           │                    │   │
│  │            │ KV:[0-99] │  │KV:[100-149]│ │KV:[150-199]│                   │   │
│  │            └─────┬─────┘  └───────────┘  └───────────┘                    │   │
│  │                  │                                                        │   │
│  │       ┌──────────┼──────────┐                                             │   │
│  │       │          │          │                                             │   │
│  │       ▼          ▼          ▼                                             │   │
│  │  ┌─────────┐┌─────────┐┌─────────┐                                        │   │
│  │  │"What is"││"How do" ││"Please" │                                        │   │
│  │  │   the   ││   I     ││ explain │                                        │   │
│  │  │         ││         ││         │                                        │   │
│  │  │KV:[200- ││KV:[250- ││KV:[300- │                                        │   │
│  │  │   249]  ││   299]  ││   349]  │                                        │   │
│  │  └─────────┘└─────────┘└─────────┘                                        │   │
│  │                                                                           │   │
│  │  Each node stores:                                                        │   │
│  │    - Token sequence (the "key")                                           │   │
│  │    - Indices into KV cache pool (the "value")                             │   │
│  │    - Reference count (for eviction)                                       │   │
│  │    - Timestamp (for LRU eviction)                                         │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │                         KV CACHE POOL (GPU HBM)                           │   │
│  │                                                                           │   │
│  │  ┌────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┐ │   │
│  │  │ 0  │ 1  │ 2  │... │ 99 │100 │... │149 │150 │... │199 │200 │... │349 │ │   │
│  │  └────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┘ │   │
│  │  ◄─── "You are a helpful..." ──►◄─ "Translate" ─►◄─ "Summarize" ─►       │   │
│  │                                                                           │   │
│  │  Unlike Paged Attention: indices are SEMANTIC (by prefix), not PHYSICAL   │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### How Radix Cache Handles Prefix Matching

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         RADIX TREE: PREFIX MATCHING                              │
│                                                                                  │
│  Query: "You are a helpful assistant. What is the capital of France?"            │
│                                                                                  │
│  Step 1: Walk the tree from root                                                 │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │                                                                           │   │
│  │  Input tokens: [You][are][a][helpful][assistant][.][What][is][the]...     │   │
│  │                                                                           │   │
│  │  Tree walk:                                                               │   │
│  │                                                                           │   │
│  │  ROOT ──► Check children for "You"                                        │   │
│  │           │                                                               │   │
│  │           ▼                                                               │   │
│  │       ┌───────────────────┐                                               │   │
│  │       │ "You are a        │ ◄── MATCH! First 100 tokens match             │   │
│  │       │  helpful          │                                               │   │
│  │       │  assistant..."    │     Compare token-by-token:                   │   │
│  │       │                   │     [You] == [You] ✓                          │   │
│  │       │  KV: [0-99]       │     [are] == [are] ✓                          │   │
│  │       └─────────┬─────────┘     [a] == [a] ✓                              │   │
│  │                 │               ... all 100 tokens match!                 │   │
│  │                 ▼                                                         │   │
│  │       ┌───────────────────┐                                               │   │
│  │       │ "What is the"     │ ◄── MATCH! Next 50 tokens match               │   │
│  │       │                   │                                               │   │
│  │       │  KV: [200-249]    │                                               │   │
│  │       └─────────┬─────────┘                                               │   │
│  │                 │                                                         │   │
│  │                 ▼                                                         │   │
│  │            (no more children match "capital")                             │   │
│  │                                                                           │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
│  Result: 150 tokens matched (cached), only need to compute remaining tokens!     │
│                                                                                  │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │                                                                           │   │
│  │  Total input: 158 tokens                                                  │   │
│  │  Cached:      150 tokens (from radix tree)                                │   │
│  │  To compute:    8 tokens ("capital of France?")                           │   │
│  │                                                                           │   │
│  │  Speedup: 158/8 = 19.75x faster prefill!                                  │   │
│  │                                                                           │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Radix Tree Operations (from Mini-SGLang code)

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                      RADIX TREE OPERATIONS (Code Reference)                      │
│                                                                                  │
│  From: python/minisgl/kvcache/radix_manager.py                                   │
│                                                                                  │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │ OPERATION 1: match_prefix(input_ids)                                      │   │
│  │                                                                           │   │
│  │   def match_prefix(self, input_ids: torch.Tensor):                        │   │
│  │       node, prefix_len = self._walk(input_ids)  # Walk tree               │   │
│  │       # Collect KV indices by walking back up                             │   │
│  │       value_list = []                                                     │   │
│  │       while not node.is_root():                                           │   │
│  │           value_list.append(node.value)  # KV cache indices               │   │
│  │           node = node.parent                                              │   │
│  │       return RadixCacheHandle(prefix_len, node), torch.cat(value_list)    │   │
│  │                                                                           │   │
│  │   Returns: (handle, indices_tensor)                                       │   │
│  │     - handle.cached_len = number of tokens already cached                 │   │
│  │     - indices_tensor = KV cache locations for cached tokens               │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │ OPERATION 2: insert_prefix(input_ids, indices)                            │   │
│  │                                                                           │   │
│  │   def insert_prefix(self, input_ids: torch.Tensor, indices: torch.Tensor):│   │
│  │       node, prefix_len = self._walk(input_ids)                            │   │
│  │       if prefix_len < len(input_ids):                                     │   │
│  │           # Create new node for unmatched suffix                          │   │
│  │           new_node = RadixTreeNode()                                      │   │
│  │           new_node.set_key_value(                                         │   │
│  │               input_ids[prefix_len:],   # Token sequence                  │   │
│  │               indices[prefix_len:]      # KV cache indices                │   │
│  │           )                                                               │   │
│  │           new_node.set_parent(node)                                       │   │
│  │       return prefix_len  # How much was already cached                    │   │
│  │                                                                           │   │
│  │   After request completes, new prefix is added to tree for future reuse   │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘     

# Radix Tree Prefix Cache vs Paged Attention: A Visual Comparison

*(Continued from previous section)*

---

### Radix Tree Operations (from Mini-SGLang code)

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                      RADIX TREE OPERATIONS (Code Reference)                      │
│                                                                                  │
│  From: python/minisgl/kvcache/radix_manager.py                                   │
│                                                                                  │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │ OPERATION 1: match_prefix(input_ids)                                      │   │
│  │                                                                           │   │
│  │   def match_prefix(self, input_ids: torch.Tensor):                        │   │
│  │       node, prefix_len = self._walk(input_ids)  # Walk tree               │   │
│  │       # Collect KV indices by walking back up                             │   │
│  │       value_list = []                                                     │   │
│  │       while not node.is_root():                                           │   │
│  │           value_list.append(node.value)  # KV cache indices               │   │
│  │           node = node.parent                                              │   │
│  │       return RadixCacheHandle(prefix_len, node), torch.cat(value_list)    │   │
│  │                                                                           │   │
│  │   Returns: (handle, indices_tensor)                                       │   │
│  │     - handle.cached_len = number of tokens already cached                 │   │
│  │     - indices_tensor = KV cache locations for cached tokens               │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │ OPERATION 2: insert_prefix(input_ids, indices)                            │   │
│  │                                                                           │   │
│  │   def insert_prefix(self, input_ids: torch.Tensor, indices: torch.Tensor):│   │
│  │       node, prefix_len = self._walk(input_ids)                            │   │
│  │       if prefix_len < len(input_ids):                                     │   │
│  │           # Create new node for unmatched suffix                          │   │
│  │           new_node = RadixTreeNode()                                      │   │
│  │           new_node.set_key_value(                                         │   │
│  │               input_ids[prefix_len:],   # Token sequence                  │   │
│  │               indices[prefix_len:]      # KV cache indices                │   │
│  │           )                                                               │   │
│  │           new_node.set_parent(node)                                       │   │
│  │       return prefix_len  # How much was already cached                    │   │
│  │                                                                           │   │
│  │   After request completes, new prefix is added to tree for future reuse   │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │ OPERATION 3: evict(size)                                                  │   │
│  │                                                                           │   │
│  │   def evict(self, size: int) -> torch.Tensor:                             │   │
│  │       # Collect all leaf nodes with ref_count == 0                        │   │
│  │       leave_nodes = self._collect_leave_nodes_for_evict()                 │   │
│  │       heapq.heapify(leave_nodes)  # Min-heap by timestamp (LRU)           │   │
│  │                                                                           │   │
│  │       evicted_indices = []                                                │   │
│  │       while evicted_size < size:                                          │   │
│  │           node = heapq.heappop(leave_nodes)  # Oldest leaf                │   │
│  │           evicted_indices.append(node.value)                              │   │
│  │           # Remove node from tree                                         │   │
│  │           del parent.children[node.first_token]                           │   │
│  │           # If parent becomes leaf and unreferenced, add to heap          │   │
│  │           if parent.is_leaf() and parent.ref_count == 0:                  │   │
│  │               heapq.heappush(leave_nodes, parent)                         │   │
│  │                                                                           │   │
│  │       return torch.cat(evicted_indices)                                   │   │
│  │                                                                           │   │
│  │   LRU eviction: oldest unused leaves are evicted first                    │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │ OPERATION 4: lock_handle / unlock_handle                                  │   │
│  │                                                                           │   │
│  │   def lock_handle(self, handle, unlock=False):                            │   │
│  │       node = handle.node                                                  │   │
│  │       while not node.is_root():                                           │   │
│  │           node = node.parent                                              │   │
│  │           if unlock:                                                      │   │
│  │               node.ref_count -= 1                                         │   │
│  │               if node.ref_count == 0:                                     │   │
│  │                   self.evictable_size += node.length                      │   │
│  │           else:                                                           │   │
│  │               if node.ref_count == 0:                                     │   │
│  │                   self.evictable_size -= node.length                      │   │
│  │               node.ref_count += 1                                         │   │
│  │                                                                           │   │
│  │   Locking prevents eviction while request is using cached KV              │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### RadixTreeNode Data Structure

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         RADIX TREE NODE STRUCTURE                                │
│                                                                                  │
│  From: python/minisgl/kvcache/radix_manager.py                                   │
│                                                                                  │
│  class RadixTreeNode:                                                            │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │                                                                           │   │
│  │   ┌─────────────────────────────────────────────────────────────────┐    │   │
│  │   │                         NODE FIELDS                              │    │   │
│  │   ├─────────────────────────────────────────────────────────────────┤    │   │
│  │   │  children: Dict[int, RadixTreeNode]                              │    │   │
│  │   │     └── Maps first token ID → child node                         │    │   │
│  │   │                                                                  │    │   │
│  │   │  _parent: RadixTreeNode | None                                   │    │   │
│  │   │     └── Parent node (None for root)                              │    │   │
│  │   │                                                                  │    │   │
│  │   │  ref_count: int                                                  │    │   │
│  │   │     └── Number of active requests using this node                │    │   │
│  │   │     └── If 0, node can be evicted                                │    │   │
│  │   │                                                                  │    │   │
│  │   │  timestamp: int                                                  │    │   │
│  │   │     └── Last access time (for LRU eviction)                      │    │   │
│  │   │                                                                  │    │   │
│  │   │  _key: torch.Tensor                                              │    │   │
│  │   │     └── Token IDs stored in this node                            │    │   │
│  │   │                                                                  │    │   │
│  │   │  _value: torch.Tensor                                            │    │   │
│  │   │     └── KV cache indices for these tokens                        │    │   │
│  │   │                                                                  │    │   │
│  │   │  _length: int                                                    │    │   │
│  │   │     └── Number of tokens in this node                            │    │   │
│  │   └─────────────────────────────────────────────────────────────────┘    │   │
│  │                                                                           │   │
│  │   Example Node:                                                           │   │
│  │   ┌─────────────────────────────────────────────────────────────────┐    │   │
│  │   │  _key:    [151644, 872, 198, 3838, 374]  (5 token IDs)           │    │   │
│  │   │  _value:  [0, 1, 2, 3, 4]                (KV cache indices)      │    │   │
│  │   │  _length: 5                                                      │    │   │
│  │   │  ref_count: 2  (2 requests currently using this prefix)          │    │   │
│  │   │  timestamp: 1234567890                                           │    │   │
│  │   │  children: {279: <child_node>}  (token 279 leads to child)       │    │   │
│  │   └─────────────────────────────────────────────────────────────────┘    │   │
│  │                                                                           │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Tree Walk Algorithm (_walk)

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           TREE WALK ALGORITHM                                    │
│                                                                                  │
│  def _walk(self, input_ids: torch.Tensor) -> Tuple[RadixTreeNode, int]:          │
│                                                                                  │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │  Example: input_ids = [A, B, C, D, E, F, G, H]                             │   │
│  │                                                                           │   │
│  │  Tree State:                                                              │   │
│  │                    ROOT                                                   │   │
│  │                     │                                                     │   │
│  │                     ▼                                                     │   │
│  │              ┌─────────────┐                                              │   │
│  │              │ key: [A,B,C]│                                              │   │
│  │              │ len: 3      │                                              │   │
│  │              └──────┬──────┘                                              │   │
│  │                     │                                                     │   │
│  │           ┌─────────┴─────────┐                                           │   │
│  │           ▼                   ▼                                           │   │
│  │    ┌─────────────┐     ┌─────────────┐                                    │   │
│  │    │ key: [D,E]  │     │ key: [X,Y]  │                                    │   │
│  │    │ len: 2      │     │ len: 2      │                                    │   │
│  │    └─────────────┘     └─────────────┘                                    │   │
│  │                                                                           │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │  Walk Steps:                                                              │   │
│  │                                                                           │   │
│  │  Step 1: Start at ROOT, prefix_len = 0                                    │   │
│  │          Check: input_ids[0] = A in children? YES                         │   │
│  │          Move to child node [A,B,C]                                       │   │
│  │                                                                           │   │
│  │  Step 2: At node [A,B,C]                                                  │   │
│  │          Compare: input_ids[0:3] vs node._key                             │   │
│  │          [A,B,C] == [A,B,C] ✓ (full match)                                │   │
│  │          prefix_len = 3                                                   │   │
│  │          Check: input_ids[3] = D in children? YES                         │   │
│  │          Move to child node [D,E]                                         │   │
│  │                                                                           │   │
│  │  Step 3: At node [D,E]                                                    │   │
│  │          Compare: input_ids[3:5] vs node._key                             │   │
│  │          [D,E] == [D,E] ✓ (full match)                                    │   │
│  │          prefix_len = 5                                                   │   │
│  │          Check: input_ids[5] = F in children? NO                          │   │
│  │          STOP - no more matches                                           │   │
│  │                                                                           │   │
│  │  Result: (node=[D,E], prefix_len=5)                                       │   │
│  │          Tokens [A,B,C,D,E] are cached, [F,G,H] need computation          │   │
│  │                                                                           │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │  Partial Match Case: input_ids = [A, B, X, Y, Z]                          │   │
│  │                                                                           │   │
│  │  Step 1: At ROOT, move to child [A,B,C]                                   │   │
│  │                                                                           │   │
│  │  Step 2: At node [A,B,C]                                                  │   │
│  │          Compare: input_ids[0:3] vs [A,B,C]                               │   │
│  │          [A,B,X] vs [A,B,C]                                               │   │
│  │          Match only 2 tokens! (A,B match, X≠C)                            │   │
│  │                                                                           │   │
│  │          SPLIT the node:                                                  │   │
│  │          ┌─────────────┐         ┌─────────────┐                          │   │
│  │          │ key: [A,B,C]│   →     │ key: [A,B]  │ (new parent)             │   │
│  │          │ len: 3      │         └──────┬──────┘                          │   │
│  │          └─────────────┘                │                                 │   │
│  │                                         ▼                                 │   │
│  │                                  ┌─────────────┐                          │   │
│  │                                  │ key: [C]    │ (old node, shortened)    │   │
│  │                                  │ len: 1      │                          │   │
│  │                                  └─────────────┘                          │   │
│  │                                                                           │   │
│  │  Result: (node=[A,B], prefix_len=2)                                       │   │
│  │                                                                           │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### How Radix Cache Handles Requests

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                      RADIX CACHE: REQUEST LIFECYCLE                              │
│                                                                                  │
│  TIME ═══════════════════════════════════════════════════════════════════════►  │
│                                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │ t=0: Request 1 arrives: "You are a helpful assistant. What is 2+2?"      │    │
│  │                                                                          │    │
│  │      Radix Tree: ROOT (empty)                                            │    │
│  │      KV Pool: [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]                 │    │
│  │                                                                          │    │
│  │      match_prefix() → (handle, [])  # No match, empty indices            │    │
│  │      cached_len = 0                                                      │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                              │                                                   │
│                              ▼                                                   │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │ t=1: Process Request 1 (full prefill)                                    │    │
│  │                                                                          │    │
│  │      Allocate KV indices: [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]           │    │
│  │      Compute KV for all 15 tokens                                        │    │
│  │      Generate response: "4"                                              │    │
│  │                                                                          │    │
│  │      KV Pool: [██████████████████████████████░░░░░░░░░░]                 │    │
│  │                ◄─── 15 tokens for Request 1 ───►                         │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                              │                                                   │
│                              ▼                                                   │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │ t=2: Request 1 completes, insert into radix tree                         │    │
│  │                                                                          │    │
│  │      insert_prefix(input_ids, indices)                                   │    │
│  │                                                                          │    │
│  │      Radix Tree:                                                         │    │
│  │                    ROOT                                                  │    │
│  │                     │                                                    │    │
│  │                     ▼                                                    │    │
│  │      ┌────────────────────────────────────┐                              │    │
│  │      │ key: "You are a helpful assistant. │                              │    │
│  │      │       What is 2+2?"                │                              │    │
│  │      │ value: [0,1,2,...,14]              │                              │    │
│  │      │ ref_count: 0 (request done)        │                              │    │
│  │      └────────────────────────────────────┘                              │    │
│  │                                                                          │    │
│  │      KV Pool: [██████████████████████████████░░░░░░░░░░]                 │    │
│  │                ◄─── cached, reusable ───►                                │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                              │                                                   │
│                              ▼                                                   │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │ t=3: Request 2 arrives: "You are a helpful assistant. What is 3+3?"      │    │
│  │                                                                          │    │
│  │      match_prefix() walks tree:                                          │    │
│  │        - "You are a helpful assistant. What is" MATCHES (12 tokens)      │    │
│  │        - "2+2?" does NOT match "3+3?"                                    │    │
│  │                                                                          │    │
│  │      Result: cached_len = 12, indices = [0,1,2,...,11]                   │    │
│  │                                                                          │    │
│  │      Tree splits:                                                        │    │
│  │                    ROOT                                                  │    │
│  │                     │                                                    │    │
│  │                     ▼                                                    │    │
│  │      ┌────────────────────────────────────┐                              │    │
│  │      │ key: "You are a helpful assistant. │                              │    │
│  │      │       What is"                     │  ◄── SHARED PREFIX           │    │
│  │      │ value: [0,1,2,...,11]              │                              │    │
│  │      │ ref_count: 1 (locked by Req 2)     │                              │    │
│  │      └─────────────┬──────────────────────┘                              │    │
│  │                    │                                                     │    │
│  │                    ▼                                                     │    │
│  │      ┌────────────────────────────────────┐                              │    │
│  │      │ key: "2+2?"                        │                              │    │
│  │      │ value: [12,13,14]                  │                              │    │
│  │      │ ref_count: 0                       │                              │    │
│  │      └────────────────────────────────────┘                              │    │
│  │                                                                          │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                              │                                                   │
│                              ▼                                                   │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │ t=4: Process Request 2 (partial prefill - only 3 new tokens!)            │    │
│  │                                                                          │    │
│  │      Reuse KV for tokens 0-11 (from cache)                               │    │
│  │      Allocate new indices: [15,16,17] for "3+3?"                         │    │
│  │      Compute KV for only 3 tokens!                                       │    │
│  │                                                                          │    │
│  │      Speedup: 15 tokens → 3 tokens = 5x faster prefill!                  │    │
│  │                                                                          │    │
│  │      KV Pool: [██████████████████████████████████████░░]                 │    │
│  │                ◄─── shared ───►◄─ R1 ─►◄─ R2 ─►                          │    │
│  │                     12 tok      3 tok   3 tok                            │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                              │                                                   │
│                              ▼                                                   │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │ t=5: Request 2 completes, tree updated                                   │    │
│  │                                                                          │    │
│  │      Radix Tree:                                                         │    │
│  │                    ROOT                                                  │    │
│  │                     │                                                    │    │
│  │                     ▼                                                    │    │
│  │      ┌────────────────────────────────────┐                              │    │
│  │      │ key: "You are a helpful assistant. │                              │    │
│  │      │       What is"                     │                              │    │
│  │      │ value: [0,1,2,...,11]              │                              │    │
│  │      │ ref_count: 0                       │                              │    │
│  │      └─────────────┬──────────────────────┘                              │    │
│  │                    │                                                     │    │
│  │          ┌─────────┴─────────┐                                           │    │
│  │          ▼                   ▼                                           │    │
│  │   ┌─────────────┐     ┌─────────────┐                                    │    │
│  │   │ key: "2+2?" │     │ key: "3+3?" │                                    │    │
│  │   │ val:[12-14] │     │ val:[15-17] │                                    │    │
│  │   └─────────────┘     └─────────────┘                                    │    │
│  │                                                                          │    │
│  │      Both "2+2?" and "3+3?" branches share the common prefix!            │    │
│  │                                                                          │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Side-by-Side Comparison

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    PAGED ATTENTION vs RADIX CACHE: COMPARISON                    │
│                                                                                  │
│  ┌─────────────────────────────────┬─────────────────────────────────────────┐  │
│  │       PAGED ATTENTION           │         RADIX CACHE                     │  │
│  │           (vLLM)                │         (SGLang)                        │  │
│  ├─────────────────────────────────┼─────────────────────────────────────────┤  │
│  │                                 │                                         │  │
│  │  PRIMARY GOAL:                  │  PRIMARY GOAL:                          │  │
│  │  Memory efficiency              │  Compute efficiency                     │  │
│  │  (reduce fragmentation)         │  (reduce redundant computation)         │  │
│  │                                 │                                         │  │
│  ├─────────────────────────────────┼─────────────────────────────────────────┤  │
│  │                                 │                                         │  │
│  │  DATA STRUCTURE:                │  DATA STRUCTURE:                        │  │
│  │  Page Table (per sequence)      │  Radix Tree (global, shared)            │  │
│  │                                 │                                         │  │
│  │  ┌─────────────────────┐        │  ┌─────────────────────┐                │  │
│  │  │ Seq → [P0,P2,P5]    │        │  │      ROOT           │                │  │
│  │  │ Seq → [P1,P3]       │        │  │       │             │                │  │
│  │  └─────────────────────┘        │  │    ┌──┴──┐          │                │  │
│  │                                 │  │    ▼     ▼          │                │  │
│  │  Physical page mapping          │  │  [A,B] [X,Y]        │                │  │
│  │                                 │  └─────────────────────┘                │  │
│  │                                 │  Token sequence tree                    │  │
│  │                                 │                                         │  │
│  ├─────────────────────────────────┼─────────────────────────────────────────┤  │
│  │                                 │                                         │  │
│  │  ALLOCATION UNIT:               │  ALLOCATION UNIT:                       │  │
│  │  Fixed-size pages               │  Variable-length token sequences        │  │
│  │  (e.g., 16 tokens/page)         │  (matches actual prefix lengths)        │  │
│  │                                 │                                         │  │
│  ├─────────────────────────────────┼─────────────────────────────────────────┤  │
│  │                                 │                                         │  │
│  │  PREFIX SHARING:                │  PREFIX SHARING:                        │  │
│  │  Manual / Limited               │  Automatic / Full                       │  │
│  │                                 │                                         │  │
│  │  Must explicitly enable         │  Automatically detects and              │  │
│  │  prefix caching, exact          │  reuses any shared prefix,              │  │
│  │  match required                 │  partial matches supported              │  │
│  │                                 │                                         │  │
│  ├─────────────────────────────────┼─────────────────────────────────────────┤  │
│  │                                 │                                         │  │
│  │  EVICTION STRATEGY:             │  EVICTION STRATEGY:                     │  │
│  │  Per-sequence                   │  LRU on tree leaves                     │  │
│  │                                 │                                         │  │
│  │  When sequence ends,            │  Unused prefixes evicted                │  │
│  │  all pages freed                │  based on access time                   │  │
│  │                                 │                                         │  │
│  ├─────────────────────────────────┼─────────────────────────────────────────┤  │
│  │                                 │                                         │  │
│  │  MEMORY OVERHEAD:               │  MEMORY OVERHEAD:                       │  │
│  │  Low (just page tables)         │  Medium (tree structure)                │  │
│  │                                 │                                         │  │
│  │  O(num_sequences × max_pages)   │  O(unique_prefixes × avg_length)        │  │
│  │                                 │                                         │  │
│  │                                 │                                         │  │
│  └─────────────────────────────────┴─────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────────────┘


# Radix Tree Prefix Cache vs Paged Attention: A Visual Comparison

*(Continued from previous section)*

---

## 4. Side-by-Side Comparison (Continued)

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    PAGED ATTENTION vs RADIX CACHE: COMPARISON                    │
│                                                                                  │
│  ┌─────────────────────────────────────┬─────────────────────────────────────┐  │
│  │       PAGED ATTENTION               │         RADIX CACHE                 │  │
│  │           (vLLM)                    │         (SGLang)                    │  │
│  ├─────────────────────────────────────┼─────────────────────────────────────┤  │
│  │                                     │                                     │  │
│  │  BEST FOR:                          │  BEST FOR:                          │  │
│  │  • Diverse, unrelated queries       │  • Shared system prompts            │  │
│  │  • Maximum memory utilization       │  • Multi-turn conversations         │  │
│  │  • High-throughput batch serving    │  • RAG with shared context          │  │
│  │                                     │  • Few-shot learning                │  │
│  │                                     │                                     │  │
│  ├─────────────────────────────────────┼─────────────────────────────────────┤  │
│  │                                     │                                     │  │
│  │  COMPUTE SAVINGS:                   │  COMPUTE SAVINGS:                   │  │
│  │  None (always recompute)            │  Up to 95%+ for shared prefixes     │  │
│  │                                     │                                     │  │
│  │  Example: 10 requests with          │  Example: 10 requests with          │  │
│  │  500-token system prompt            │  500-token system prompt            │  │
│  │                                     │                                     │  │
│  │  Compute: 5000 tokens               │  Compute: 500 tokens (first req)    │  │
│  │           (500 × 10)                │           + ~0 tokens (reuse)       │  │
│  │                                     │                                     │  │
│  ├─────────────────────────────────────┼─────────────────────────────────────┤  │
│  │                                     │                                     │  │
│  │  MEMORY SAVINGS:                    │  MEMORY SAVINGS:                    │  │
│  │  ~25% (reduced fragmentation)       │  Variable (depends on prefix        │  │
│  │                                     │  sharing patterns)                  │  │
│  │                                     │                                     │  │
│  └─────────────────────────────────────┴─────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Visual: How Each Handles the Same Workload

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    SAME WORKLOAD, DIFFERENT APPROACHES                           │
│                                                                                  │
│  Workload: 3 requests with shared system prompt                                  │
│                                                                                  │
│  Request 1: "You are helpful. What is 2+2?"                                      │
│  Request 2: "You are helpful. What is 3+3?"                                      │
│  Request 3: "You are helpful. What is 4+4?"                                      │
│                                                                                  │
│  ═══════════════════════════════════════════════════════════════════════════    │
│                                                                                  │
│  PAGED ATTENTION (vLLM):                                                         │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │                                                                           │   │
│  │  Page Pool:                                                               │   │
│  │  ┌────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┐           │   │
│  │  │ P0 │ P1 │ P2 │ P3 │ P4 │ P5 │ P6 │ P7 │ P8 │ P9 │P10 │P11 │           │   │
│  │  │R1  │R1  │R1  │R2  │R2  │R2  │R3  │R3  │R3  │free│free│free│           │   │
│  │  └────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┘           │   │
│  │                                                                           │   │
│  │  Each request gets its own pages:                                         │   │
│  │  • Request 1: P0, P1, P2 (computed "You are helpful. What is 2+2?")       │   │
│  │  • Request 2: P3, P4, P5 (computed "You are helpful. What is 3+3?")       │   │
│  │  • Request 3: P6, P7, P8 (computed "You are helpful. What is 4+4?")       │   │
│  │                                                                           │   │
│  │  Total KV computed: 3 × full_length = 3 × 10 = 30 tokens                  │   │
│  │  Total pages used: 9 pages                                                │   │
│  │                                                                           │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
│  ═══════════════════════════════════════════════════════════════════════════    │
│                                                                                  │
│  RADIX CACHE (SGLang):                                                           │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │                                                                           │   │
│  │  Radix Tree:                                                              │   │
│  │                         ROOT                                              │   │
│  │                          │                                                │   │
│  │                          ▼                                                │   │
│  │              ┌───────────────────────┐                                    │   │
│  │              │ "You are helpful."    │ ◀── SHARED (computed once)         │   │
│  │              │ KV indices: [0-5]     │                                    │   │
│  │              └───────────┬───────────┘                                    │   │
│  │                          │                                                │   │
│  │           ┌──────────────┼──────────────┐                                 │   │
│  │           ▼              ▼              ▼                                 │   │
│  │     ┌──────────┐  ┌──────────┐  ┌──────────┐                              │   │
│  │     │"What is  │  │"What is  │  │"What is  │                              │   │
│  │     │ 2+2?"    │  │ 3+3?"    │  │ 4+4?"    │                              │   │
│  │     │KV:[6-9]  │  │KV:[10-13]│  │KV:[14-17]│                              │   │
│  │     └──────────┘  └──────────┘  └──────────┘                              │   │
│  │                                                                           │   │
│  │  KV Cache Pool:                                                           │   │
│  │  ┌────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┐ │   │
│  │  │ 0  │ 1  │ 2  │ 3  │ 4  │ 5  │ 6  │ 7  │ 8  │ 9  │ 10 │ 11 │... │    │ │   │
│  │  │◄── shared prefix ──────────►│◄─R1─►│◄─R2─►│◄─R3─►│    │ │   │
│  │  └────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┘ │   │
│  │                                                                           │   │
│  │  Total KV computed: 6 (prefix) + 4×3 (unique) = 18 tokens                 │   │
│  │  Total cache slots used: 18 slots                                         │   │
│  │                                                                           │   │
│  │  SAVINGS: 30 - 18 = 12 tokens (40% compute reduction!)                    │   │
│  │                                                                           │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Step-by-Step Request Serving

### Scenario: Multi-Turn Conversation

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    MULTI-TURN CONVERSATION EXAMPLE                               │
│                                                                                  │
│  Turn 1: User asks "What is Python?"                                             │
│  Turn 2: User asks "How do I install it?"                                        │
│  Turn 3: User asks "Show me a hello world example"                               │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Paged Attention Approach

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    PAGED ATTENTION: MULTI-TURN HANDLING                          │
│                                                                                  │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │ TURN 1: "System: You are helpful. User: What is Python?"                  │   │
│  │                                                                           │   │
│  │   Allocate pages: P0, P1, P2                                              │   │
│  │   Compute KV for all 20 tokens                                            │   │
│  │   Generate response: "Python is a programming language..."                │   │
│  │   Append response KV to pages: P2, P3, P4                                 │   │
│  │                                                                           │   │
│  │   Page Table: [P0, P1, P2, P3, P4]                                        │   │
│  │   Total tokens in cache: 50                                               │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│                              │                                                   │
│                              ▼                                                   │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │ TURN 2: "... User: How do I install it?"                                  │   │
│  │                                                                           │   │
│  │   Full context = Turn 1 + new question                                    │   │
│  │                                                                           │   │
│  │   Option A (No prefix caching - default):                                 │   │
│  │     • Recompute ALL 50 + 10 = 60 tokens                                   │   │
│  │     • Allocate new pages: P5, P6, P7, P8                                  │   │
│  │                                                                           │   │
│  │   Option B (With explicit prefix caching):                                │   │
│  │     • Reuse pages P0-P4 (if exact match)                                  │   │
│  │     • Only compute new 10 tokens                                          │   │
│  │     • Requires explicit API call to enable                                │   │
│  │                                                                           │   │
│  │   Page Table: [P0, P1, P2, P3, P4, P5, P6, P7, P8]                        │   │
│  │   Total tokens in cache: 90                                               │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│                              │                                                   │
│                              ▼                                                   │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │ TURN 3: "... User: Show me a hello world example"                         │   │
│  │                                                                           │   │
│  │   Full context = Turn 1 + Turn 2 + new question                           │   │
│  │                                                                           │   │
│  │   Without prefix caching: Recompute ALL 90 + 15 = 105 tokens              │   │
│  │   With prefix caching: Only compute 15 new tokens (if exact match)        │   │
│  │                                                                           │   │
│  │   Total compute over 3 turns (no caching): 20 + 60 + 105 = 185 tokens     │   │
│  │   Total compute over 3 turns (with caching): 20 + 10 + 15 = 45 tokens     │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Radix Cache Approach

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    RADIX CACHE: MULTI-TURN HANDLING                              │
│                                                                                  │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │ TURN 1: "System: You are helpful. User: What is Python?"                  │   │
│  │                                                                           │   │
│  │   1. match_prefix() → No match (empty tree)                               │   │
│  │   2. Compute KV for all 20 tokens                                         │   │
│  │   3. Generate response (30 tokens)                                        │   │
│  │   4. insert_prefix() → Add to tree                                        │   │
│  │                                                                           │   │
│  │   Radix Tree:                                                             │   │
│  │                    ROOT                                                   │   │
│  │                     │                                                     │   │
│  │                     ▼                                                     │   │
│  │   ┌─────────────────────────────────────────────────────┐                 │   │
│  │   │ "System: You are helpful. User: What is Python?     │                 │   │
│  │   │  Assistant: Python is a programming language..."    │                 │   │
│  │   │ KV indices: [0-49]                                  │                 │   │
│  │   └─────────────────────────────────────────────────────┘                 │   │
│  │                                                                           │   │
│  │   Total compute: 20 tokens (input) + 30 tokens (output) = 50 tokens       │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│                              │                                                   │
│                              ▼                                                   │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │ TURN 2: "... User: How do I install it?"                                  │   │
│  │                                                                           │   │
│  │   1. match_prefix() → MATCH! 50 tokens cached                             │   │
│  │   2. Only compute KV for 10 new tokens                                    │   │
│  │   3. Generate response (25 tokens)                                        │   │
│  │   4. insert_prefix() → Extend tree                                        │   │
│  │                                                                           │   │
│  │   Radix Tree:                                                             │   │
│  │                    ROOT                                                   │   │
│  │                     │                                                     │   │
│  │                     ▼                                                     │   │
│  │   ┌─────────────────────────────────────────────────────┐                 │   │
│  │   │ "System: ... Python is a programming language..."   │                 │   │
│  │   │ KV indices: [0-49]                                  │                 │   │
│  │   └────────────────────────┬────────────────────────────┘                 │   │
│  │                            │                                              │   │
│  │                            ▼                                              │   │
│  │   ┌─────────────────────────────────────────────────────┐                 │   │
│  │   │ "User: How do I install it? Assistant: You can..."  │                 │   │
│  │   │ KV indices: [50-84]                                 │                 │   │
│  │   └─────────────────────────────────────────────────────┘                 │   │
│  │                                                                           │   │
│  │   Total compute: 10 tokens (new input) + 25 tokens (output) = 35 tokens   │   │
│  │   SAVED: 50 tokens (reused from cache)                                    │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│                              │                                                   │
│                              ▼                                                   │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │ TURN 3: "... User: Show me a hello world example"                         │   │
│  │                                                                           │   │
│  │   1. match_prefix() → MATCH! 85 tokens cached                             │   │
│  │   2. Only compute KV for 15 new tokens                                    │   │
│  │   3. Generate response (40 tokens)                                        │   │
│  │   4. insert_prefix() → Extend tree                                        │   │
│  │                                                                           │   │
│  │   Total compute: 15 tokens (new input) + 40 tokens (output) = 55 tokens   │   │
│  │   SAVED: 85 tokens (reused from cache)                                    │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
│  ═══════════════════════════════════════════════════════════════════════════    │
│                                                                                  │
│  TOTAL COMPARISON (3 turns):                                                     │
│                                                                                  │
│  │ Metric              │ Paged (no cache) │ Paged (w/ cache) │ Radix Cache │    │
│  ├─────────────────────┼──────────────────┼──────────────────┼─────────────┤    │
│  │ Turn 1 compute      │ 50 tokens        │ 50 tokens        │ 50 tokens   │    │
│  │ Turn 2 compute      │ 85 tokens        │ 35 tokens        │ 35 tokens   │    │
│  │ Turn 3 compute      │ 140 tokens       │ 55 tokens        │ 55 tokens   │    │
│  │ TOTAL               │ 275 tokens       │ 140 tokens       │ 140 tokens  │    │
│  ├─────────────────────┼──────────────────┼──────────────────┼─────────────┤    │
│  │ Automatic?          │ N/A              │ No (explicit)    │ YES         │    │
│  │ Partial match?      │ N/A              │ No               │ YES         │    │
│  └─────────────────────┴──────────────────┴──────────────────┴─────────────┘    │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Performance & Cost Trade-offs

### Compute Efficiency

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         COMPUTE EFFICIENCY COMPARISON                            │
│                                                                                  │
│  Scenario: 1000 requests/hour with 500-token system prompt                       │
│                                                                                  │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │                                                                           │   │
│  │  PAGED ATTENTION (no prefix caching):                                     │   │
│  │  ┌────────────────────────────────────────────────────────────────────┐  │   │
│  │  │                                                                     │  │   │
│  │  │  Tokens computed per request: 500 (system) + 100 (user) = 600      │  │   │
│  │  │  Total tokens/hour: 600 × 1000 = 600,000 tokens                    │  │   │
│  │  │                                                                     │  │   │
│  │  │  ████████████████████████████████████████████████████████████████  │  │   │
│  │  │  ◄──────────────── 600,000 tokens ────────────────────────────►    │  │   │
│  │  │                                                                     │  │   │
│  │  └────────────────────────────────────────────────────────────────────┘  │   │
│  │                                                                           │   │
│  │  RADIX CACHE:                                                             │   │
│  │  ┌────────────────────────────────────────────────────────────────────┐  │   │
│  │  │                                                                     │  │   │
│  │  │  First request: 500 (system) + 100 (user) = 600 tokens             │  │   │
│  │  │  Subsequent 999 requests: 100 (user only) × 999 = 99,900 tokens    │  │   │
│  │  │  Total tokens/hour: 600 + 99,900 = 100,500 tokens                  │  │   │
│  │  │                                                                     │  │   │
│  │  │  ██████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  │  │   │
│  │  │  ◄─ 100,500 ─►                                                     │  │   │
│  │  │                                                                     │  │   │
│  │  │  SAVINGS: 600,000 - 100,500 = 499,500 tokens (83% reduction!)      │  │   │
│  │  │                                                                     │  │   │
│  │  └────────────────────────────────────────────────────────────────────┘  │   │
│  │                                                                           │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Memory Efficiency

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         MEMORY EFFICIENCY COMPARISON                             │
│                                                                                  │
│  Scenario: 100 concurrent requests, variable lengths (100-2000 tokens)           │
│                                                                                  │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │                                                                           │   │
│  │  TRADITIONAL (Contiguous allocation):                                     │   │
│  │                                                                           │   │
│  │  Request 1:  ████████████████████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░  │   │
│  │  Request 2:  ████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  │   │
│  │  Request 3:  ████████████████████████████████████████████████████████░░  │   │
│  │  ...                                                                      │   │
│  │                                                                           │   │
│  │  Pre-allocate max_seq_len for each request                                │   │
│  │  Memory waste: ~40% (average)                                             │   │
│  │                                                                           │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │                                                                           │   │
│  │  PAGED ATTENTION:                                                         │   │
│  │                                                                           │   │
│  │  Page Pool: [██][██][██][██][██][██][██][██][░░][░░][░░][░░]...           │   │
│  │              R1  R1  R2  R3  R3  R3  R1  R2                               │   │
│  │                                                                           │   │
│  │  Allocate pages on-demand, non-contiguous                                 │   │
│  │  Memory waste: ~5% (page-level fragmentation only)                        │   │
│  │                                                                           │   │
│  │  IMPROVEMENT: 40% → 5% = 35% more efficient                               │   │
│  │                                                                           │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │                                                                           │   │
│  │  RADIX CACHE:                                                             │   │
│  │                                                                           │   │
│  │  KV Pool: [██][██][██][██][██][██][██][██][██][░░][░░][░░]...             │   │
│  │            ◄─ shared ─►  R1  R2  R3  R1  R2  R3                           │   │
│  │                                                                           │   │
│  │  Shared prefixes stored once, referenced by multiple requests             │   │
│  │  Memory waste: Variable (depends on sharing patterns)                     │   │
│  │                                                                           │   │
│  │  With 50% prefix sharing: Additional ~25% memory savings                  │   │
│  │                                                                           │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Latency Impact

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                            LATENCY COMPARISON                                    │
│                                                                                  │
│  Metric: Time to First Token (TTFT) for request with 500-token system prompt     │
│                                                                                  │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │                                                                           │   │
│  │  PAGED ATTENTION (cold start):                                            │   │
│  │                                                                           │   │
│  │  ┌─────────────────────────────────────────────────────────────────────┐ │   │
│  │  │ Tokenize │ Allocate │    Prefill 500 tokens    │ Decode │ First tok │ │   │
│  │  │  1ms     │  0.5ms   │         50ms             │  5ms   │           │ │   │
│  │  └─────────────────────────────────────────────────────────────────────┘ │   │
│  │                                                                           │   │
│  │  TTFT: ~56.5ms                                                            │   │
│  │                                                                           │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │                                                                           │   │
│  │  RADIX CACHE (cache hit):                                                 │   │
│  │                                                                           │   │
│  │  ┌─────────────────────────────────────────────────────────────────────┐ │   │
│  │  │ Tokenize │ Match │ Prefill 50 tokens │ Decode │ First token         │ │   │
│  │  │  1ms     │ 0.1ms │      5ms          │  5ms   │                     │ │   │
│  │  └─────────────────────────────────────────────────────────────────────┘ │   │
│  │                                                                           │   │
│  │  TTFT: ~11.1ms                                                            │   │
│  │                                                                           │   │
│  │  IMPROVEMENT: 56.5ms → 11.1ms = 5x faster TTFT!                           │   │
│  │                                                                           │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```
# Radix Tree Prefix Cache vs Paged Attention: A Visual Comparison

*(Continued from previous section)*

---

## 6. Performance & Cost Trade-offs (Continued)

### Cost Analysis

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              COST ANALYSIS                                       │
│                                                                                  │
│  Assumptions:                                                                    │
│  • H100 GPU: $3/hour                                                             │
│  • Throughput without optimization: 1000 tokens/second                           │
│  • Workload: 1M requests/day with 500-token system prompt                        │
│                                                                                  │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │                                                                           │   │
│  │  PAGED ATTENTION (no prefix caching):                                     │   │
│  │                                                                           │   │
│  │  Tokens per request: 500 (system) + 100 (user) + 200 (output) = 800      │   │
│  │  Total tokens/day: 800 × 1,000,000 = 800M tokens                          │   │
│  │  GPU hours needed: 800M / (1000 × 3600) = 222 GPU-hours                   │   │
│  │  Daily cost: 222 × $3 = $666/day                                          │   │
│  │  Monthly cost: $666 × 30 = $19,980/month                                  │   │
│  │                                                                           │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │                                                                           │   │
│  │  RADIX CACHE:                                                             │   │
│  │                                                                           │   │
│  │  First request: 500 + 100 + 200 = 800 tokens                              │   │
│  │  Subsequent requests: 100 + 200 = 300 tokens (system prompt cached)       │   │
│  │                                                                           │   │
│  │  Total tokens/day: 800 + (300 × 999,999) ≈ 300M tokens                    │   │
│  │  GPU hours needed: 300M / (1000 × 3600) = 83 GPU-hours                    │   │
│  │  Daily cost: 83 × $3 = $249/day                                           │   │
│  │  Monthly cost: $249 × 30 = $7,470/month                                   │   │
│  │                                                                           │   │
│  │  SAVINGS: $19,980 - $7,470 = $12,510/month (62.6% reduction!)             │   │
│  │                                                                           │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Summary: Trade-off Matrix

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           TRADE-OFF SUMMARY MATRIX                               │
│                                                                                  │
│  ┌────────────────────┬─────────────────────┬─────────────────────────────────┐ │
│  │     Dimension      │   Paged Attention   │        Radix Cache              │ │
│  ├────────────────────┼─────────────────────┼─────────────────────────────────┤ │
│  │                    │                     │                                 │ │
│  │ Memory Efficiency  │ ★★★★★ (Excellent)   │ ★★★☆☆ (Good)                    │ │
│  │                    │ ~5% fragmentation   │ Variable, depends on sharing    │ │
│  │                    │                     │                                 │ │
│  ├────────────────────┼─────────────────────┼─────────────────────────────────┤ │
│  │                    │                     │                                 │ │
│  │ Compute Efficiency │ ★★☆☆☆ (Basic)       │ ★★★★★ (Excellent)               │ │
│  │                    │ No prefix reuse     │ Up to 95%+ savings              │ │
│  │                    │                     │                                 │ │
│  ├────────────────────┼─────────────────────┼─────────────────────────────────┤ │
│  │                    │                     │                                 │ │
│  │ Implementation     │ ★★★★☆ (Simple)      │ ★★★☆☆ (Moderate)                │ │
│  │ Complexity         │ Page table only     │ Tree structure + LRU eviction   │ │
│  │                    │                     │                                 │ │
│  ├────────────────────┼─────────────────────┼─────────────────────────────────┤ │
│  │                    │                     │                                 │ │
│  │ Latency (TTFT)     │ ★★☆☆☆ (Baseline)    │ ★★★★★ (Excellent)               │ │
│  │                    │ Full prefill always │ Cached prefixes skip prefill    │ │
│  │                    │                     │                                 │ │
│  ├────────────────────┼─────────────────────┼─────────────────────────────────┤ │
│  │                    │                     │                                 │ │
│  │ Throughput         │ ★★★★☆ (Good)        │ ★★★★★ (Excellent)               │ │
│  │ (shared prefixes)  │ Limited by compute  │ Compute savings → more capacity │ │
│  │                    │                     │                                 │ │
│  ├────────────────────┼─────────────────────┼─────────────────────────────────┤ │
│  │                    │                     │                                 │ │
│  │ Throughput         │ ★★★★★ (Excellent)   │ ★★★★☆ (Good)                    │ │
│  │ (diverse queries)  │ Memory-optimized    │ Tree overhead, no cache hits    │ │
│  │                    │                     │                                 │ │
│  ├────────────────────┼─────────────────────┼─────────────────────────────────┤ │
│  │                    │                     │                                 │ │
│  │ Best Use Case      │ Batch processing    │ Chatbots, RAG, agents           │ │
│  │                    │ Diverse queries     │ Shared system prompts           │ │
│  │                    │                     │                                 │ │
│  └────────────────────┴─────────────────────┴─────────────────────────────────┘ │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 7. When to Use Which

### Decision Tree

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              DECISION TREE                                       │
│                                                                                  │
│                        ┌─────────────────────────┐                               │
│                        │  What's your workload?  │                               │
│                        └───────────┬─────────────┘                               │
│                                    │                                             │
│              ┌─────────────────────┼─────────────────────┐                       │
│              │                     │                     │                       │
│              ▼                     ▼                     ▼                       │
│     ┌────────────────┐   ┌────────────────┐   ┌────────────────┐                 │
│     │   Chatbot /    │   │  Batch API /   │   │   Mixed /      │                 │
│     │   Assistant    │   │  Diverse       │   │   Unknown      │                 │
│     └───────┬────────┘   └───────┬────────┘   └───────┬────────┘                 │
│             │                    │                    │                          │
│             ▼                    ▼                    ▼                          │
│     ┌────────────────┐   ┌────────────────┐   ┌────────────────┐                 │
│     │ Do you have    │   │ Is memory the  │   │ Start with     │                 │
│     │ system prompts │   │ bottleneck?    │   │ Radix Cache    │                 │
│     │ or shared      │   │                │   │ (more flexible)│                 │
│     │ context?       │   │                │   │                │                 │
│     └───────┬────────┘   └───────┬────────┘   └────────────────┘                 │
│             │                    │                                               │
│      ┌──────┴──────┐      ┌──────┴──────┐                                        │
│      │             │      │             │                                        │
│      ▼             ▼      ▼             ▼                                        │
│   ┌──────┐     ┌──────┐ ┌──────┐    ┌──────┐                                     │
│   │ YES  │     │  NO  │ │ YES  │    │  NO  │                                     │
│   └──┬───┘     └──┬───┘ └──┬───┘    └──┬───┘                                     │
│      │            │        │           │                                         │
│      ▼            ▼        ▼           ▼                                         │
│  ┌─────────┐  ┌─────────┐ ┌─────────┐ ┌─────────┐                                │
│  │ RADIX   │  │ Either  │ │ PAGED   │ │ RADIX   │                                │
│  │ CACHE   │  │ works   │ │ ATTN    │ │ CACHE   │                                │
│  │         │  │         │ │         │ │         │                                │
│  │ ✓ Best  │  │ ✓ Paged │ │ ✓ Best  │ │ ✓ Saves │                                │
│  │   for   │  │   for   │ │   for   │ │   comp- │                                │
│  │   prefix│  │   memory│ │   memory│ │   ute   │                                │
│  │   reuse │  │   effic.│ │   bound │ │         │                                │
│  └─────────┘  └─────────┘ └─────────┘ └─────────┘                                │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Workload Examples

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           WORKLOAD EXAMPLES                                      │
│                                                                                  │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │                                                                           │   │
│  │  USE RADIX CACHE WHEN:                                                    │   │
│  │                                                                           │   │
│  │  ✓ Customer Support Chatbot                                               │   │
│  │    - Same system prompt: "You are a helpful customer service agent..."    │   │
│  │    - 1000s of users, all share the same prefix                            │   │
│  │    - Savings: 90%+ on system prompt computation                           │   │
│  │                                                                           │   │
│  │  ✓ RAG (Retrieval-Augmented Generation)                                   │   │
│  │    - Shared document context across queries                               │   │
│  │    - "Given the following document: [10K tokens]... Answer: ..."          │   │
│  │    - Multiple questions about same document                               │   │
│  │                                                                           │   │
│  │  ✓ Multi-turn Conversations                                               │   │
│  │    - Each turn builds on previous context                                 │   │
│  │    - Turn 5 reuses KV cache from turns 1-4                                │   │
│  │                                                                           │   │
│  │  ✓ Few-shot Learning                                                      │   │
│  │    - Same examples prefix for all queries                                 │   │
│  │    - "Example 1: ... Example 2: ... Now answer: ..."                      │   │
│  │                                                                           │   │
│  │  ✓ Code Completion with Context                                           │   │
│  │    - Same codebase context for multiple completions                       │   │
│  │    - IDE sends same file context repeatedly                               │   │
│  │                                                                           │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │                                                                           │   │
│  │  USE PAGED ATTENTION WHEN:                                                │   │
│  │                                                                           │   │
│  │  ✓ Batch Translation Service                                              │   │
│  │    - Each request is independent                                          │   │
│  │    - No shared prefixes between requests                                  │   │
│  │    - Memory efficiency is critical                                        │   │
│  │                                                                           │   │
│  │  ✓ One-off API Queries                                                    │   │
│  │    - Users send unique, unrelated prompts                                 │   │
│  │    - No system prompt or minimal shared context                           │   │
│  │                                                                           │   │
│  │  ✓ Very Long Context (128K+ tokens)                                       │   │
│  │    - Memory is the primary constraint                                     │   │
│  │    - Need to maximize sequences per GPU                                   │   │
│  │                                                                           │   │
│  │  ✓ High-throughput Batch Processing                                       │   │
│  │    - Processing millions of diverse documents                             │   │
│  │    - No prefix sharing patterns                                           │   │
│  │                                                                           │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 8. Mini-SGLang Implementation Details

Now let's look at how Mini-SGLang implements the Radix Cache, with references to the actual codebase.

### Radix Tree Node Structure

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    RADIX TREE NODE (from radix_manager.py)                       │
│                                                                                  │
│  class RadixTreeNode:                                                            │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │                                                                           │   │
│  │   ┌─────────────────────────────────────────────────────────────────┐    │   │
│  │   │                         FIELDS                                   │    │   │
│  │   ├─────────────────────────────────────────────────────────────────┤    │   │
│  │   │                                                                  │    │   │
│  │   │  children: Dict[int, RadixTreeNode]                              │    │   │
│  │   │     └── Maps first token ID of child → child node                │    │   │
│  │   │     └── Enables O(1) lookup for next token                       │    │   │
│  │   │                                                                  │    │   │
│  │   │  _parent: RadixTreeNode | None                                   │    │   │
│  │   │     └── Back-pointer for tree traversal                          │    │   │
│  │   │     └── Used when collecting KV indices                          │    │   │
│  │   │                                                                  │    │   │
│  │   │  ref_count: int                                                  │    │   │
│  │   │     └── Number of active requests using this node                │    │   │
│  │   │     └── If 0, node can be evicted                                │    │   │
│  │   │     └── Root always has ref_count = 1 (never evicted)            │    │   │
│  │   │                                                                  │    │   │
│  │   │  timestamp: int                                                  │    │   │
│  │   │     └── Last access time (monotonic nanoseconds)                 │    │   │
│  │   │     └── Used for LRU eviction ordering                           │    │   │
│  │   │                                                                  │    │   │
│  │   │  _key: torch.Tensor                                              │    │   │
│  │   │     └── Token IDs stored in this node                            │    │   │
│  │   │     └── Variable length (can be 1 token or many)                 │    │   │
│  │   │                                                                  │    │   │
│  │   │  _value: torch.Tensor                                            │    │   │
│  │   │     └── KV cache indices for these tokens                        │    │   │
│  │   │     └── Same length as _key                                      │    │   │
│  │   │                                                                  │    │   │
│  │   │  _length: int                                                    │    │   │
│  │   │     └── len(_key) = len(_value)                                  │    │   │
│  │   │                                                                  │    │   │
│  │   └─────────────────────────────────────────────────────────────────┘    │   │
│  │                                                                           │   │
│  │   Key Methods:                                                            │   │
│  │   ┌─────────────────────────────────────────────────────────────────┐    │   │
│  │   │                                                                  │    │   │
│  │   │  set_key_value(key, value)                                       │    │   │
│  │   │     └── Initialize node with token IDs and KV indices            │    │   │
│  │   │                                                                  │    │   │
│  │   │  set_parent(parent)                                              │    │   │
│  │   │     └── Link to parent and register in parent's children         │    │   │
│  │   │                                                                  │    │   │
│  │   │  get_match_len(input_ids)                                        │    │   │
│  │   │     └── Compare with input, return matching prefix length        │    │   │
│  │   │     └── Uses fast C++ kernel (fast_compare_key)                  │    │   │
│  │   │                                                                  │    │   │
│  │   │  _split_at(pos)                                                  │    │   │
│  │   │     └── Split node at position for partial matches               │    │   │
│  │   │     └── Creates new parent node with shared prefix               │    │   │
│  │   │                                                                  │    │   │
│  │   └─────────────────────────────────────────────────────────────────┘    │   │
│  │                                                                           │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Cache Manager Integration

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    CACHE MANAGER (from cache.py)                                 │
│                                                                                  │
│  class CacheManager:                                                             │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │                                                                           │   │
│  │   Combines:                                                               │   │
│  │   • Free slot tracking (_free_slots)                                      │   │
│  │   • Radix/Naive cache manager (manager)                                   │   │
│  │   • Allocation and eviction logic                                         │   │
│  │                                                                           │   │
│  │   ┌─────────────────────────────────────────────────────────────────┐    │   │
│  │   │                                                                  │    │   │
│  │   │  def match_req(self, req: PendingReq):                           │    │   │
│  │   │      """Match prefix for a pending request."""                   │    │   │
│  │   │      input_len = req.input_len                                   │    │   │
│  │   │      # Match all but last token (last token needs computation)   │    │   │
│  │   │      return self.manager.match_prefix(req.input_ids[:input_len-1])│   │   │
│  │   │                                                                  │    │   │
│  │   │  def allocate(self, needed_len: int) -> torch.Tensor:            │    │   │
│  │   │      """Allocate KV cache slots, evicting if necessary."""       │    │   │
│  │   │      if needed_len <= len(self._free_slots):                     │    │   │
│  │   │          # Fast path: enough free slots                          │    │   │
│  │   │          allocated = self._free_slots[:needed_len]               │    │   │
│  │   │          self._free_slots = self._free_slots[needed_len:]        │    │   │
│  │   │          return allocated                                        │    │   │
│  │   │                                                                  │    │   │
│  │   │      # Slow path: need to evict                                  │    │   │
│  │   │      evicted = self.manager.evict(needed_len - free_len)         │    │   │
│  │   │      merged = torch.cat([self._free_slots, evicted])             │    │   │
│  │   │      allocated = merged[:needed_len]                             │    │   │
│  │   │      self._free_slots = merged[needed_len:]                      │    │   │
│  │   │      return allocated                                            │    │   │
│  │   │                                                                  │    │   │
│  │   │  def free_and_cache_finished_req(self, handle, input_ids, indices):│  │   │
│  │   │      """Cache completed request and free unused slots."""        │    │   │
│  │   │      in_cache_len = self.manager.insert_prefix(input_ids, indices)│   │   │
│  │   │      # Free slots that are now in cache (deduplicated)           │    │   │
│  │   │      self._free(indices[handle.cached_len:in_cache_len])         │    │   │
│  │   │      self.unlock(handle)                                         │    │   │
│  │   │                                                                  │    │   │
│  │   └─────────────────────────────────────────────────────────────────┘    │   │
│  │                                                                           │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Request Flow Through Radix Cache

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    REQUEST FLOW (from prefill.py and scheduler.py)               │
│                                                                                  │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │                                                                           │   │
│  │  STEP 1: Request arrives at PrefillAdder._try_allocate_one()              │   │
│  │  ┌─────────────────────────────────────────────────────────────────┐     │   │
│  │  │                                                                  │     │   │
│  │  │  # Check radix cache for prefix match                           │     │   │
│  │  │  handle, match_indices = self.cache_manager.match_req(req)       │     │   │
│  │  │  cached_len = handle.cached_len                                  │     │   │
│  │  │                                                                  │     │   │
│  │  │  # Calculate how much we need to compute                         │     │   │
│  │  │  extend_len = req.input_len - cached_len                         │     │   │
│  │  │  estimated_len = extend_len + req.output_len                     │     │   │
│  │  │                                                                  │     │   │
│  │  │  # Check if we have enough cache space                           │     │   │
│  │  │  if estimated_len > self.cache_manager.available_size:           │     │   │
│  │  │      return None  # Can't fit, queue for later                   │     │   │
│  │  │                                                                  │     │   │
│  │  │  # Lock the handle to prevent eviction                           │     │   │
│  │  │  self.cache_manager.lock(handle)                                 │     │   │
│  │  │                                                                  │     │   │
│  │  │  # Copy cached KV indices to page table                          │     │   │
│  │  │  if cached_len > 0:                                              │     │   │
│  │  │      page_entry[:cached_len].copy_(match_indices)                │     │   │
│  │  │                                                                  │     │   │
│  │  └─────────────────────────────────────────────────────────────────┘     │   │
│  │                                                                           │   │
│  │  STEP 2: Scheduler._prepare_batch() allocates new KV slots               │   │
│  │  ┌─────────────────────────────────────────────────────────────────┐     │   │
│  │  │                                                                  │     │   │
│  │  │  # Allocate slots for new tokens (extend_len per request)        │     │   │
│  │  │  needed_size = sum(r.extend_len for r in batch.reqs)             │     │   │
│  │  │  batch.out_loc = self.cache_manager.allocate(needed_size)        │     │   │
│  │  │                                                                  │     │   │
│  │  │  # Write new slot indices to page table                          │     │   │
│  │  │  self.page_table.view(-1)[load_indices] = batch.out_loc          │     │   │
│  │  │                                                                  │     │   │
│  │  └─────────────────────────────────────────────────────────────────┘     │   │
│  │                                                                           │   │
│  │  STEP 3: Engine.forward_batch() computes KV for new tokens               │   │
│  │  ┌─────────────────────────────────────────────────────────────────┐     │   │
│  │  │                                                                  │     │   │
│  │  │  # Only compute attention for extend_len tokens                  │     │   │
│  │  │  # Cached tokens are already in KV cache, just referenced        │     │   │
│  │  │  logits = self.model.forward()                                   │     │   │
│  │  │                                                                  │     │   │
│  │  └─────────────────────────────────────────────────────────────────┘     │   │
│  │                                                                           │   │
│  │  STEP 4: Request completes, insert into radix tree                       │   │
│  │  ┌─────────────────────────────────────────────────────────────────┐     │   │
│  │  │                                                                  │     │   │
│  │  │  # Cache the completed request for future reuse                  │     │   │
│  │  │  self.cache_manager.free_and_cache_finished_req(                 │     │   │
│  │  │      req.cache_handle,                                           │     │   │
│  │  │      req.host_ids[:req.cached_len],  # Token IDs                 │     │   │
│  │  │      self.page_table[req.table_idx, :req.cached_len]  # KV locs  │     │   │
│  │  │  )                                                               │     │   │
│  │  │                                                                  │     │   │
│  │  └─────────────────────────────────────────────────────────────────┘     │   │
│  │                                                                           │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

# Radix Tree Prefix Cache vs Paged Attention: A Visual Comparison

*(Continued from previous section)*

---

## 8. Mini-SGLang Implementation Details (Continued)

### Eviction Process

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    LRU EVICTION (from radix_manager.py)                          │
│                                                                                  │
│  def evict(self, size: int) -> torch.Tensor:                                     │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │                                                                           │   │
│  │  # Step 1: Collect all evictable leaf nodes                               │   │
│  │  leave_nodes = self._collect_leave_nodes_for_evict()                      │   │
│  │  # Only nodes with ref_count == 0 and is_leaf() are evictable             │   │
│  │                                                                           │   │
│  │  # Step 2: Build min-heap by timestamp (oldest first)                     │   │
│  │  heapq.heapify(leave_nodes)                                               │   │
│  │                                                                           │   │
│  │  # Step 3: Evict nodes until we have enough space                         │   │
│  │  evicted_indices = []                                                     │   │
│  │  evicted_size = 0                                                         │   │
│  │  while evicted_size < size:                                               │   │
│  │      node = heapq.heappop(leave_nodes)  # Get oldest leaf                 │   │
│  │      evicted_size += node.length                                          │   │
│  │      evicted_indices.append(node.value)  # KV cache indices               │   │
│  │      self.evictable_size -= node.length                                   │   │
│  │                                                                           │   │
│  │      # Remove node from tree                                              │   │
│  │      parent = node.parent                                                 │   │
│  │      del parent.children[int(node._key[0].item())]                        │   │
│  │                                                                           │   │
│  │      # If parent becomes evictable leaf, add to heap                      │   │
│  │      if parent.is_leaf() and parent.ref_count == 0:                       │   │
│  │          heapq.heappush(leave_nodes, parent)                              │   │
│  │                                                                           │   │
│  │  return torch.cat(evicted_indices)                                        │   │
│  │                                                                           │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Eviction Visualization

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         LRU EVICTION EXAMPLE                                     │
│                                                                                  │
│  Initial State (need to evict 50 tokens):                                        │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │                                                                           │   │
│  │                              ROOT (ref=1)                                 │   │
│  │                                │                                          │   │
│  │                                ▼                                          │   │
│  │              ┌─────────────────────────────────────┐                      │   │
│  │              │ "System prompt..." (100 tokens)     │                      │   │
│  │              │ ref_count: 0, timestamp: t=100      │                      │   │
│  │              └─────────────────┬───────────────────┘                      │   │
│  │                                │                                          │   │
│  │           ┌────────────────────┼────────────────────┐                     │   │
│  │           ▼                    ▼                    ▼                     │   │
│  │   ┌─────────────┐      ┌─────────────┐      ┌─────────────┐              │   │
│  │   │ "Query A"   │      │ "Query B"   │      │ "Query C"   │              │   │
│  │   │ 30 tokens   │      │ 40 tokens   │      │ 25 tokens   │              │   │
│  │   │ ref=0, t=50 │      │ ref=0, t=80 │      │ ref=1, t=90 │ ◀── LOCKED   │   │
│  │   └─────────────┘      └─────────────┘      └─────────────┘              │   │
│  │                                                                           │   │
│  │   Evictable leaves: [Query A (t=50), Query B (t=80)]                      │   │
│  │   Query C is locked (ref=1), cannot evict                                 │   │
│  │                                                                           │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
│  Step 1: Evict "Query A" (oldest, t=50)                                          │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │                                                                           │   │
│  │                              ROOT (ref=1)                                 │   │
│  │                                │                                          │   │
│  │                                ▼                                          │   │
│  │              ┌─────────────────────────────────────┐                      │   │
│  │              │ "System prompt..." (100 tokens)     │                      │   │
│  │              │ ref_count: 0, timestamp: t=100      │                      │   │
│  │              └─────────────────┬───────────────────┘                      │   │
│  │                                │                                          │   │
│  │                     ┌──────────┴──────────┐                               │   │
│  │                     ▼                     ▼                               │   │
│  │             ┌─────────────┐       ┌─────────────┐                         │   │
│  │             │ "Query B"   │       │ "Query C"   │                         │   │
│  │             │ 40 tokens   │       │ 25 tokens   │                         │   │
│  │             │ ref=0, t=80 │       │ ref=1, t=90 │                         │   │
│  │             └─────────────┘       └─────────────┘                         │   │
│  │                                                                           │   │
│  │   Evicted: 30 tokens (need 20 more)                                       │   │
│  │   Freed KV indices: [indices for Query A]                                 │   │
│  │                                                                           │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
│  Step 2: Evict "Query B" (next oldest, t=80)                                     │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │                                                                           │   │
│  │                              ROOT (ref=1)                                 │   │
│  │                                │                                          │   │
│  │                                ▼                                          │   │
│  │              ┌─────────────────────────────────────┐                      │   │
│  │              │ "System prompt..." (100 tokens)     │                      │   │
│  │              │ ref_count: 0, timestamp: t=100      │                      │   │
│  │              └─────────────────┬───────────────────┘                      │   │
│  │                                │                                          │   │
│  │                                ▼                                          │   │
│  │                        ┌─────────────┐                                    │   │
│  │                        │ "Query C"   │                                    │   │
│  │                        │ 25 tokens   │                                    │   │
│  │                        │ ref=1, t=90 │                                    │   │
│  │                        └─────────────┘                                    │   │
│  │                                                                           │   │
│  │   Evicted: 30 + 40 = 70 tokens (done! needed 50)                          │   │
│  │   Freed KV indices: [indices for Query A] + [indices for Query B]         │   │
│  │                                                                           │   │
│  │   Note: "System prompt" is NOT evicted because:                           │   │
│  │     - It's not a leaf (has child "Query C")                               │   │
│  │     - Even if it were a leaf, Query C's lock protects the path            │   │
│  │                                                                           │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 9. Complete Request Flow Comparison

### Paged Attention Flow

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    PAGED ATTENTION: COMPLETE REQUEST FLOW                        │
│                                                                                  │
│  Request: "You are helpful. What is 2+2?"                                        │
│                                                                                  │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │ STEP 1: Request Arrives                                                   │   │
│  │                                                                           │   │
│  │   Input tokens: [You, are, helpful, ., What, is, 2, +, 2, ?]              │   │
│  │   Token count: 10                                                         │   │
│  │                                                                           │   │
│  │   Page Pool: [░░][░░][░░][░░][░░][░░][░░][░░]  (all free)                 │   │
│  │   Page Table for this request: (empty)                                    │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│                              │                                                   │
│                              ▼                                                   │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │ STEP 2: Allocate Pages                                                    │   │
│  │                                                                           │   │
│  │   Need: ceil(10 / 16) = 1 page (page_size = 16)                           │   │
│  │   Allocate: Page 0                                                        │   │
│  │                                                                           │   │
│  │   Page Pool: [██][░░][░░][░░][░░][░░][░░][░░]                             │   │
│  │               P0                                                          │   │
│  │   Page Table: [0] → P0                                                    │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│                              │                                                   │
│                              ▼                                                   │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │ STEP 3: Prefill (Compute KV for all input tokens)                         │   │
│  │                                                                           │   │
│  │   For each layer:                                                         │   │
│  │     Q = W_q @ hidden_states  # [10, num_heads, head_dim]                  │   │
│  │     K = W_k @ hidden_states  # [10, num_kv_heads, head_dim]               │   │
│  │     V = W_v @ hidden_states  # [10, num_kv_heads, head_dim]               │   │
│  │                                                                           │   │
│  │     # Store K, V in page 0                                                │   │
│  │     kv_cache[P0, 0:10] = (K, V)                                           │   │
│  │                                                                           │   │
│  │     # Compute attention                                                   │   │
│  │     attn_output = attention(Q, K, V)                                      │   │
│  │                                                                           │   │
│  │   Compute: 10 tokens × all layers                                         │   │
│  │   Time: ~10ms (depends on model size)                                     │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│                              │                                                   │
│                              ▼                                                   │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │ STEP 4: Decode (Generate tokens one by one)                               │   │
│  │                                                                           │   │
│  │   Token 11: "4"                                                           │   │
│  │     - Load K, V from P0[0:10]                                             │   │
│  │     - Compute new K, V for token 11                                       │   │
│  │     - Store in P0[10]                                                     │   │
│  │     - Attention over all 11 K, V                                          │   │
│  │     - Sample next token                                                   │   │
│  │                                                                           │   │
│  │   Token 12: "."                                                           │   │
│  │     - Load K, V from P0[0:11]                                             │   │
│  │     - Compute new K, V for token 12                                       │   │
│  │     - Store in P0[11]                                                     │   │
│  │     - Attention over all 12 K, V                                          │   │
│  │     - Sample next token → EOS                                             │   │
│  │                                                                           │   │
│  │   Page Pool: [██][░░][░░][░░][░░][░░][░░][░░]                             │   │
│  │               P0 (12 tokens used)                                         │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│                              │                                                   │
│                              ▼                                                   │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │ STEP 5: Request Complete - Free Pages                                     │   │
│  │                                                                           │   │
│  │   Free Page 0                                                             │   │
│  │   Page Pool: [░░][░░][░░][░░][░░][░░][░░][░░]  (all free again)           │   │
│  │                                                                           │   │
│  │   Total compute: 10 (prefill) + 2 (decode) = 12 token computations        │   │
│  │   KV cache NOT preserved for future requests                              │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Radix Cache Flow

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    RADIX CACHE: COMPLETE REQUEST FLOW                            │
│                                                                                  │
│  Request 1: "You are helpful. What is 2+2?"                                      │
│                                                                                  │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │ STEP 1: Request Arrives                                                   │   │
│  │                                                                           │   │
│  │   Input tokens: [You, are, helpful, ., What, is, 2, +, 2, ?]              │   │
│  │   Token count: 10                                                         │   │
│  │                                                                           │   │
│  │   Radix Tree: ROOT (empty)                                                │   │
│  │   KV Pool: [░░][░░][░░][░░][░░][░░][░░][░░][░░][░░]...                    │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│                              │                                                   │
│                              ▼                                                   │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │ STEP 2: Match Prefix (match_prefix)                                       │   │
│  │                                                                           │   │
│  │   Walk radix tree with input tokens...                                    │   │
│  │   Result: No match (tree is empty)                                        │   │
│  │   cached_len = 0                                                          │   │
│  │   extend_len = 10 (all tokens need computation)                           │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│                              │                                                   │
│                              ▼                                                   │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │ STEP 3: Allocate KV Slots                                                 │   │
│  │                                                                           │   │
│  │   Allocate 10 slots from free pool: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]        │   │
│  │                                                                           │   │
│  │   KV Pool: [██][██][██][██][██][██][██][██][██][██][░░][░░]...            │   │
│  │             0   1   2   3   4   5   6   7   8   9                         │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│                              │                                                   │
│                              ▼                                                   │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │ STEP 4: Prefill (Same as Paged Attention)                                 │   │
│  │                                                                           │   │
│  │   Compute K, V for all 10 tokens                                          │   │
│  │   Store in KV pool at indices [0-9]                                       │   │
│  │   Compute: 10 tokens × all layers                                         │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│                              │                                                   │
│                              ▼                                                   │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │ STEP 5: Decode (Same as Paged Attention)                                  │   │
│  │                                                                           │   │
│  │   Generate "4." (2 tokens)                                                │   │
│  │   Allocate slots [10, 11]                                                 │   │
│  │                                                                           │   │
│  │   KV Pool: [██][██][██][██][██][██][██][██][██][██][██][██][░░]...        │   │
│  │             0   1   2   3   4   5   6   7   8   9  10  11                 │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│                              │                                                   │
│                              ▼                                                   │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │ STEP 6: Request Complete - INSERT INTO RADIX TREE                         │   │
│  │                                                                           │   │
│  │   insert_prefix(input_ids, indices)                                       │   │
│  │                                                                           │   │
│  │   Radix Tree:                                                             │   │
│  │                    ROOT                                                   │   │
│  │                     │                                                     │   │
│  │                     ▼                                                     │   │
│  │   ┌─────────────────────────────────────────────────────┐                 │   │
│  │   │ key: [You, are, helpful, ., What, is, 2, +, 2, ?]   │                 │   │
│  │   │ value: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]               │                 │   │
│  │   │ ref_count: 0 (request done)                         │                 │   │
│  │   │ timestamp: t1                                       │                 │   │
│  │   └─────────────────────────────────────────────────────┘                 │   │
│  │                                                                           │   │
│  │   KV cache PRESERVED for future requests!                                 │   │
│  │   Slots [10, 11] freed (output tokens not cached)                         │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
│  ════════════════════════════════════════════════════════════════════════════   │
│                                                                                  │
│  Request 2: "You are helpful. What is 3+3?"                                      │
│                                                                                  │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │ STEP 1: Request Arrives                                                   │   │
│  │                                                                           │   │
│  │   Input tokens: [You, are, helpful, ., What, is, 3, +, 3, ?]              │   │
│  │   Token count: 10                                                         │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│                              │                                                   │
│                              ▼                                                   │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │ STEP 2: Match Prefix (match_prefix)                                       │   │
│  │                                                                           │   │
│  │   Walk radix tree:                                                        │   │
│  │     [You] matches ✓                                                       │   │
│  │     [are] matches ✓                                                       │   │
│  │     [helpful] matches ✓                                                   │   │
│  │     [.] matches ✓                                                         │   │
│  │     [What] matches ✓                                                      │   │
│  │     [is] matches ✓                                                        │   │
│  │     [3] ≠ [2] ✗ STOP                                                      │   │
│  │                                                                           │   │
│  │   Result: 6 tokens matched!                                               │   │
│  │   cached_len = 6                                                          │   │
│  │   extend_len = 4 (only [3, +, 3, ?] need computation)                     │   │
│  │   cached_indices = [0, 1, 2, 3, 4, 5]                                     │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│                              │                                                   │
│                              ▼                                                   │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │ STEP 3: Lock Handle & Allocate                                            │   │
│  │                                                                           │   │
│  │   Lock the matched node (ref_count: 0 → 1)                                │   │
│  │   Allocate 4 new slots: [10, 11, 12, 13]                                  │   │
│  │                                                                           │   │
│  │   Page table for this request:                                            │   │
│  │   [0, 1, 2, 3, 4, 5, 10, 11, 12, 13]                                      │   │
│  │    ◄─── cached ────►◄─── new ────►                                        │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│                              │                                                   │
│                              ▼                                                   │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │ STEP 4: Prefill (ONLY 4 TOKENS!)                                          │   │
│  │                                                                           │   │
│  │   Compute K, V for only [3, +, 3, ?]                                      │   │
│  │   Store in KV pool at indices [10, 11, 12, 13]                            │   │
│  │                                                                           │   │
│  │   Attention uses:                                                         │   │
│  │     - Cached K, V from indices [0-5] (no recompute!)                      │   │
│  │     - New K, V from indices [10-13]                                       │   │
│  │                                                                           │   │
│  │   Compute: 4 tokens × all layers (vs 10 without cache!)                   │   │
│  │   SPEEDUP: 10/4 = 2.5x faster prefill!                                    │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│                              │                                                   │
│                              ▼                                                   │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │ STEP 5: Decode & Complete                                                 │   │
│  │                                                                           │   │
│  │   Generate "6." (2 tokens)                                                │   │
│  │   Unlock handle (ref_count: 1 → 0)                                        │   │
│  │   Insert new prefix into tree (splits existing node)                      │   │
│  │                                                                           │   │
│  │   Radix Tree (after):                                                     │   │
│  │                    ROOT                                                   │   │
│  │                     │                                                     │   │
│  │                     ▼                                                     │   │
│  │   ┌─────────────────────────────────────────────────────┐                 │   │
│  │   │ key: [You, are, helpful, ., What, is]               │ ◀── SHARED      │   │
│  │   │ value: [0, 1, 2, 3, 4, 5]                           │                 │   │
│  │   │ ref_count: 0                                        │                 │   │
│  │   └─────────────────────┬───────────────────────────────┘                 │   │
│  │                         │                                                 │   │
│  │              ┌──────────┴──────────┐                                      │   │
│  │              ▼                     ▼                                      │   │
│  │   ┌─────────────────┐   ┌─────────────────┐                               │   │
│  │   │ key: [2,+,2,?]  │   │ key: [3,+,3,?]  │                               │   │
│  │   │ value: [6,7,8,9]│   │ value:[10-13]   │                               │   │
│  │   │ ref_count: 0    │   │ ref_count: 0    │                               │   │
│  │   └─────────────────┘   └─────────────────┘                               │   │
│  │                                                                           │   │
│  │   Both "2+2?" and "3+3?" share the common prefix!                         │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```
# Radix Tree Prefix Cache vs Paged Attention: A Visual Comparison

*(Continued from previous section)*

---

## 10. Summary: Key Differences

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           FINAL SUMMARY                                          │
│                                                                                  │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │                                                                           │   │
│  │  PAGED ATTENTION (vLLM)                                                   │   │
│  │  ════════════════════════                                                 │   │
│  │                                                                           │   │
│  │  Core Idea: Treat GPU memory like OS virtual memory                       │   │
│  │                                                                           │   │
│  │  ┌─────────────────────────────────────────────────────────────────┐     │   │
│  │  │  Physical Pages: [P0][P1][P2][P3][P4][P5][P6][P7]...            │     │   │
│  │  │                   ▲   ▲   ▲                                     │     │   │
│  │  │  Seq A: ──────────┴───┼───┴                                     │     │   │
│  │  │  Seq B: ──────────────┴                                         │     │   │
│  │  │                                                                  │     │   │
│  │  │  • Non-contiguous allocation                                    │     │   │
│  │  │  • Minimal memory fragmentation                                 │     │   │
│  │  │  • Each request gets independent pages                          │     │   │
│  │  │  • Pages freed when request completes                           │     │   │
│  │  └─────────────────────────────────────────────────────────────────┘     │   │
│  │                                                                           │   │
│  │  Strengths:                                                               │   │
│  │    ✓ Excellent memory efficiency (~5% fragmentation)                      │   │
│  │    ✓ Simple implementation                                                │   │
│  │    ✓ Works well for diverse, unrelated queries                            │   │
│  │    ✓ Predictable memory behavior                                          │   │
│  │                                                                           │   │
│  │  Weaknesses:                                                              │   │
│  │    ✗ No automatic prefix sharing                                          │   │
│  │    ✗ Redundant computation for shared prefixes                            │   │
│  │    ✗ Higher latency for repeated system prompts                           │   │
│  │                                                                           │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │                                                                           │   │
│  │  RADIX CACHE (SGLang / Mini-SGLang)                                       │   │
│  │  ══════════════════════════════════                                       │   │
│  │                                                                           │   │
│  │  Core Idea: Organize KV cache by token sequences for automatic reuse      │   │
│  │                                                                           │   │
│  │  ┌─────────────────────────────────────────────────────────────────┐     │   │
│  │  │                         ROOT                                     │     │   │
│  │  │                          │                                       │     │   │
│  │  │              ┌───────────┴───────────┐                           │     │   │
│  │  │              ▼                       ▼                           │     │   │
│  │  │     ┌───────────────┐       ┌───────────────┐                    │     │   │
│  │  │     │ "System..."   │       │ "Translate..."│                    │     │   │
│  │  │     │ KV: [0-99]    │       │ KV: [100-149] │                    │     │   │
│  │  │     └───────┬───────┘       └───────────────┘                    │     │   │
│  │  │             │                                                    │     │   │
│  │  │      ┌──────┴──────┐                                             │     │   │
│  │  │      ▼             ▼                                             │     │   │
│  │  │  ┌───────┐    ┌───────┐                                          │     │   │
│  │  │  │"Q: A" │    │"Q: B" │  ◀── Different queries share prefix      │     │   │
│  │  │  └───────┘    └───────┘                                          │     │   │
│  │  │                                                                  │     │   │
│  │  │  • Semantic organization by token sequence                       │     │   │
│  │  │  • Automatic prefix detection and reuse                          │     │   │
│  │  │  • LRU eviction on tree leaves                                   │     │   │
│  │  └─────────────────────────────────────────────────────────────────┘     │   │
│  │                                                                           │   │
│  │  Strengths:                                                               │   │
│  │    ✓ Automatic prefix sharing (no code changes needed)                    │   │
│  │    ✓ Up to 95%+ compute savings for shared prefixes                       │   │
│  │    ✓ Dramatically lower TTFT for cached prefixes                          │   │
│  │    ✓ Excellent for chatbots, RAG, multi-turn conversations                │   │
│  │                                                                           │   │
│  │  Weaknesses:                                                              │   │
│  │    ✗ Tree structure overhead                                              │   │
│  │    ✗ More complex implementation                                          │   │
│  │    ✗ Less benefit for diverse, unrelated queries                          │   │
│  │                                                                           │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 11. Decision Framework

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         CHOOSING THE RIGHT APPROACH                              │
│                                                                                  │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │                                                                           │   │
│  │  ASK YOURSELF THESE QUESTIONS:                                            │   │
│  │                                                                           │   │
│  │  1. Do your requests share common prefixes?                               │   │
│  │     ┌─────────────────────────────────────────────────────────────────┐  │   │
│  │     │ YES: System prompts, few-shot examples, shared context          │  │   │
│  │     │      → RADIX CACHE (SGLang)                                     │  │   │
│  │     │                                                                  │  │   │
│  │     │ NO:  Each request is completely independent                     │  │   │
│  │     │      → PAGED ATTENTION (vLLM)                                   │  │   │
│  │     └─────────────────────────────────────────────────────────────────┘  │   │
│  │                                                                           │   │
│  │  2. What's your primary constraint?                                       │   │
│  │     ┌─────────────────────────────────────────────────────────────────┐  │   │
│  │     │ MEMORY: Need to serve many concurrent users                     │  │   │
│  │     │         → PAGED ATTENTION (better memory packing)               │  │   │
│  │     │                                                                  │  │   │
│  │     │ COMPUTE: Need to minimize redundant work                        │  │   │
│  │     │          → RADIX CACHE (prefix reuse)                           │  │   │
│  │     │                                                                  │  │   │
│  │     │ LATENCY: Need fast time-to-first-token                          │  │   │
│  │     │          → RADIX CACHE (skip prefill for cached prefixes)       │  │   │
│  │     └─────────────────────────────────────────────────────────────────┘  │   │
│  │                                                                           │   │
│  │  3. What's your workload pattern?                                         │   │
│  │     ┌─────────────────────────────────────────────────────────────────┐  │   │
│  │     │ CHATBOT / ASSISTANT:                                            │  │   │
│  │     │   - Same system prompt for all users                            │  │   │
│  │     │   - Multi-turn conversations                                    │  │   │
│  │     │   → RADIX CACHE                                                 │  │   │
│  │     │                                                                  │  │   │
│  │     │ RAG (Retrieval-Augmented Generation):                           │  │   │
│  │     │   - Shared document context                                     │  │   │
│  │     │   - Multiple questions per document                             │  │   │
│  │     │   → RADIX CACHE                                                 │  │   │
│  │     │                                                                  │  │   │
│  │     │ BATCH TRANSLATION / SUMMARIZATION:                              │  │   │
│  │     │   - Each document is independent                                │  │   │
│  │     │   - No shared context                                           │  │   │
│  │     │   → PAGED ATTENTION                                             │  │   │
│  │     │                                                                  │  │   │
│  │     │ CODE COMPLETION:                                                │  │   │
│  │     │   - Same file context for multiple completions                  │  │   │
│  │     │   → RADIX CACHE                                                 │  │   │
│  │     └─────────────────────────────────────────────────────────────────┘  │   │
│  │                                                                           │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 12. Mini-SGLang Code Reference Summary

For developers working with Mini-SGLang, here's a quick reference to the key files:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                      MINI-SGLANG CODE REFERENCE                                  │
│                                                                                  │
│  RADIX CACHE IMPLEMENTATION:                                                     │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │                                                                           │   │
│  │  python/minisgl/kvcache/radix_manager.py                                  │   │
│  │  ├── RadixTreeNode          # Tree node with key, value, ref_count        │   │
│  │  ├── RadixCacheHandle       # Handle returned by match_prefix             │   │
│  │  └── RadixCacheManager      # Main manager class                          │   │
│  │      ├── match_prefix()     # Find longest matching prefix                │   │
│  │      ├── insert_prefix()    # Add new prefix to tree                      │   │
│  │      ├── lock_handle()      # Prevent eviction during use                 │   │
│  │      └── evict()            # LRU eviction of unused leaves               │   │
│  │                                                                           │   │
│  │  python/minisgl/kvcache/base.py                                           │   │
│  │  ├── BaseCacheManager       # Abstract interface                          │   │
│  │  ├── BaseCacheHandle        # Abstract handle                             │   │
│  │  └── SizeInfo               # Evictable/protected size tracking           │   │
│  │                                                                           │   │
│  │  python/minisgl/scheduler/cache.py                                        │   │
│  │  └── CacheManager           # Combines free slots + radix/naive manager   │   │
│  │      ├── match_req()        # Match prefix for pending request            │   │
│  │      ├── allocate()         # Allocate KV slots (with eviction)           │   │
│  │      └── free_and_cache_finished_req()  # Cache completed request         │   │
│  │                                                                           │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
│  SCHEDULER INTEGRATION:                                                          │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
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
│  │                                                                           │   │
│  │  python/minisgl/kernel/radix.py                                           │   │
│  │  └── fast_compare_key()     # Python wrapper                              │   │
│  │                                                                           │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 13. Conclusion

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              KEY TAKEAWAYS                                       │
│                                                                                  │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │                                                                           │   │
│  │  FOR BEGINNERS:                                                           │   │
│  │  ─────────────                                                            │   │
│  │  • Paged Attention = Smart memory organization (like OS virtual memory)   │   │
│  │  • Radix Cache = Smart computation reuse (like a library's master copy)   │   │
│  │  • Both make LLMs faster, but in different ways                           │   │
│  │                                                                           │   │
│  │  FOR ENGINEERS:                                                           │   │
│  │  ─────────────                                                            │   │
│  │  • Paged Attention: O(1) page allocation, ~5% fragmentation               │   │
│  │  • Radix Cache: O(prefix_len) lookup, automatic prefix deduplication      │   │
│  │  • Choose based on workload: shared prefixes → Radix, diverse → Paged     │   │
│  │                                                                           │   │
│  │  FOR CTOs:                                                                │   │
│  │  ─────────────                                                            │   │
│  │  • Radix Cache can reduce costs 2-10x for chatbot/RAG workloads           │   │
│  │  • Paged Attention is better for batch processing diverse documents       │   │
│  │  • Consider hybrid approaches for mixed workloads                         │   │
│  │                                                                           │   │
│  │  THE FUTURE:                                                              │   │
│  │  ─────────────                                                            │   │
│  │  • These approaches are complementary, not mutually exclusive             │   │
│  │  • Future systems may combine both for optimal performance                │   │
│  │  • The caching primitive generalizes to multimodal models                 │   │
│  │                                                                           │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │                                                                           │   │
│  │  QUICK REFERENCE:                                                         │   │
│  │                                                                           │   │
│  │  ┌─────────────────────┬─────────────────┬─────────────────────────────┐ │   │
│  │  │     Workload        │  Best Approach  │         Why                 │ │   │
│  │  ├─────────────────────┼─────────────────┼─────────────────────────────┤ │   │
│  │  │ Chatbot             │ Radix Cache     │ Shared system prompt        │ │   │
│  │  │ RAG                 │ Radix Cache     │ Shared document context     │ │   │
│  │  │ Multi-turn chat     │ Radix Cache     │ Incremental history         │ │   │
│  │  │ Batch translation   │ Paged Attention │ Independent documents       │ │   │
│  │  │ Long context (128K) │ Paged Attention │ Memory efficiency critical  │ │   │
│  │  │ Code completion     │ Radix Cache     │ Shared file context         │ │   │
│  │  │ Mixed workload      │ Radix Cache     │ More flexible, auto-detects │ │   │
│  │  └─────────────────────┴─────────────────┴─────────────────────────────┘ │   │
│  │                                                                           │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

This concludes the comprehensive comparison between Radix Tree Prefix Cache and Paged Attention. The key insight is that these are **complementary optimizations** targeting different bottlenecks:

- **Paged Attention** optimizes **memory efficiency** (how to store KV cache)
- **Radix Cache** optimizes **compute efficiency** (what computation to skip)

For most production LLM serving scenarios with shared prefixes (chatbots, RAG, agents), **Radix Cache provides significant benefits** with automatic prefix detection and reuse. For batch processing of diverse, unrelated documents, **Paged Attention's memory efficiency** may be more important.

<chatName="Radix Cache vs Paged Attention Visual Comparison"/>