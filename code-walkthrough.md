# Mini-SGLang: Complete End-to-End Code Walkthrough

This document provides a comprehensive, step-by-step walkthrough of the Mini-SGLang codebase, tracing a real request from user input to final output. We'll use a concrete example throughout to make the flow tangible.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Our Concrete Example](#2-our-concrete-example)
3. [Phase 1: Server Startup](#3-phase-1-server-startup)
4. [Phase 2: User Request Arrives](#4-phase-2-user-request-arrives)
5. [Phase 3: Tokenization](#5-phase-3-tokenization)
6. [Phase 4: Scheduling](#6-phase-4-scheduling)
7. [Phase 5: Model Inference](#7-phase-5-model-inference)
8. [Phase 6: Detokenization](#8-phase-6-detokenization)
9. [Phase 7: Response Streaming](#9-phase-7-response-streaming)
10. [Deep Dive: Radix Cache](#10-deep-dive-radix-cache)
11. [Deep Dive: Attention Backends](#11-deep-dive-attention-backends)
12. [Summary](#12-summary)

---

## 1. System Overview

Mini-SGLang is a **multi-process LLM inference system** with the following architecture:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              User Request                                │
│                     "What is the capital of France?"                     │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           API Server (FastAPI)                           │
│  • Receives HTTP requests                                                │
│  • Manages async request/response lifecycle                              │
│  • Streams responses back to user                                        │
└─────────────────────────────────────────────────────────────────────────┘
                                    │ ZMQ
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          Tokenizer Worker                                │
│  • Converts text → token IDs                                             │
│  • Converts token IDs → text (detokenization)                            │
└─────────────────────────────────────────────────────────────────────────┘
                                    │ ZMQ
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    Scheduler (Rank 0) ◄──NCCL──► Scheduler (Rank N)     │
│  • Manages request queue                                                 │
│  • Allocates KV cache                                                    │
│  • Batches requests                                                      │
│  • Coordinates multi-GPU execution                                       │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                              Engine                                      │
│  • Loads model weights                                                   │
│  • Manages KV cache                                                      │
│  • Executes forward pass                                                 │
│  • Handles CUDA graphs                                                   │
└─────────────────────────────────────────────────────────────────────────┘
```

**Key Design Decisions:**

| Component | Why It Exists |
|-----------|---------------|
| Separate processes | Isolates CPU-bound tokenization from GPU-bound inference |
| ZMQ messaging | Fast, async inter-process communication |
| Radix Cache | Reuses KV cache for shared prefixes |
| CUDA Graphs | Eliminates CPU launch overhead for decode |
| Overlap Scheduling | Hides CPU scheduling latency behind GPU compute |

---

## 2. Our Concrete Example

Throughout this walkthrough, we'll trace this exact request:

```python
# User sends this via curl or OpenAI client
{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [{"role": "user", "content": "What is the capital of France?"}],
    "max_tokens": 50,
    "stream": True
}
```

**Expected flow:**
1. Text "What is the capital of France?" → tokenized to ~8 tokens
2. Model generates ~10 tokens: "The capital of France is Paris."
3. Each token is streamed back as it's generated

---

## 3. Phase 1: Server Startup

### 3.1 Entry Point

When you run `python -m minisgl --model "Qwen/Qwen3-0.6B"`, execution starts here:

```python
# python/minisgl/__main__.py
from .server import launch_server

assert __name__ == "__main__"
launch_server()
```

**What this does:** Simply calls `launch_server()` from the server module.

### 3.2 Argument Parsing

```python
# python/minisgl/server/launch.py
def launch_server(run_shell: bool = False) -> None:
    from .api_server import run_api_server
    from .args import parse_args

    server_args, run_shell = parse_args(sys.argv[1:], run_shell)
    # ...
```

The `parse_args` function in `args.py` does several things:

```python
# python/minisgl/server/args.py
def parse_args(args: List[str], run_shell: bool = False) -> Tuple[ServerArgs, bool]:
    parser = argparse.ArgumentParser(description="MiniSGL Server Arguments")
    
    # Key arguments:
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--tp-size", type=int, default=1)  # Tensor parallelism
    parser.add_argument("--max-running-requests", type=int, default=256)
    parser.add_argument("--cache-type", type=str, default="radix", choices=["naive", "radix"])
    # ... more arguments
    
    kwargs = parser.parse_args(args).__dict__.copy()
    
    # Resolve dtype from model config
    if kwargs["dtype"] == "auto":
        dtype_or_str = cached_load_hf_config(kwargs["model_path"]).dtype
        kwargs["dtype"] = DTYPE_MAP.get(dtype_or_str, dtype_or_str)
    
    # Create DistributedInfo for tensor parallelism
    kwargs["tp_info"] = DistributedInfo(0, kwargs["tensor_parallel_size"])
    
    return ServerArgs(**kwargs), run_shell
```

**For our example:**
- `model_path = "Qwen/Qwen3-0.6B"`
- `tp_info = DistributedInfo(rank=0, size=1)` (single GPU)
- `cache_type = "radix"` (default)
- `dtype = torch.bfloat16` (from model config)

### 3.3 ServerArgs Configuration

The `ServerArgs` dataclass holds all configuration:

```python
# python/minisgl/server/args.py
@dataclass(frozen=True)
class ServerArgs(SchedulerConfig):
    server_host: str = "127.0.0.1"
    server_port: int = 1919
    num_tokenizer: int = 0  # 0 means shared tokenizer/detokenizer
    
    @property
    def zmq_frontend_addr(self) -> str:
        return "ipc:///tmp/minisgl_3" + self._unique_suffix
    
    @property
    def zmq_tokenizer_addr(self) -> str:
        # When num_tokenizer=0, tokenizer and detokenizer share the same address
        if self.share_tokenizer:
            return self.zmq_detokenizer_addr
        return "ipc:///tmp/minisgl_4" + self._unique_suffix
```

**Why these addresses matter:**
- `zmq_backend_addr`: Tokenizer → Scheduler communication
- `zmq_detokenizer_addr`: Scheduler → Detokenizer communication
- `zmq_frontend_addr`: Detokenizer → API Server communication

### 3.4 Spawning Subprocesses

```python
# python/minisgl/server/launch.py
def launch_server(run_shell: bool = False) -> None:
    # ...
    def start_subprocess() -> None:
        mp.set_start_method("spawn", force=True)
        ack_queue: mp.Queue[str] = mp.Queue()
        
        # 1. Start Scheduler processes (one per GPU)
        for i in range(world_size):
            new_args = replace(server_args, tp_info=DistributedInfo(i, world_size))
            mp.Process(
                target=_run_scheduler,
                args=(new_args, ack_queue),
                name=f"minisgl-TP{i}-scheduler",
            ).start()
        
        # 2. Start Tokenizer/Detokenizer worker
        mp.Process(
            target=tokenize_worker,
            kwargs={
                "tokenizer_path": server_args.model_path,
                "addr": server_args.zmq_detokenizer_addr,
                "backend_addr": server_args.zmq_backend_addr,
                "frontend_addr": server_args.zmq_frontend_addr,
                # ...
            },
            name="minisgl-detokenizer-0",
        ).start()
        
        # 3. Wait for all processes to be ready
        for _ in range(num_tokenizers + 2):
            logger.info(ack_queue.get())
    
    # 4. Start API server (runs in main process)
    run_api_server(server_args, start_subprocess, run_shell=run_shell)
```

**Process hierarchy for our single-GPU example:**
```
Main Process (API Server)
├── Scheduler Process (Rank 0)
│   └── Engine (GPU 0)
└── Tokenizer/Detokenizer Process
```

### 3.5 Scheduler Initialization

Each scheduler process runs `_run_scheduler`:

```python
# python/minisgl/server/launch.py
def _run_scheduler(args: ServerArgs, ack_queue: mp.Queue[str]) -> None:
    import torch
    from minisgl.scheduler import Scheduler

    with torch.inference_mode():
        scheduler = Scheduler(args)
        scheduler.sync_all_ranks()  # Barrier to ensure all ranks are ready
        
        if args.tp_info.is_primary():
            ack_queue.put("Scheduler is ready")
        
        scheduler.run_forever()  # Main loop
```

The `Scheduler.__init__` is where the heavy lifting happens:

```python
# python/minisgl/scheduler/scheduler.py
class Scheduler(SchedulerIOMixin):
    def __init__(self, config: SchedulerConfig):
        from minisgl.engine import Engine
        
        # 1. Create the Engine (loads model, allocates KV cache)
        self.engine = Engine(config)
        
        # 2. Initialize I/O (ZMQ connections)
        super().__init__(config, self.engine.tp_cpu_group)
        
        # 3. Create CUDA streams for overlap scheduling
        self.device = self.engine.device
        self.stream = torch.cuda.Stream(device=self.device)
        self.engine_stream_ctx = torch.cuda.stream(self.engine.stream)
        
        # 4. Initialize managers
        self.table_manager = TableManager(config.max_running_req, self.engine.page_table)
        self.cache_manager = CacheManager(self.device, self.engine.num_pages, config.cache_type)
        self.decode_manager = DecodeManager()
        self.prefill_manager = PrefillManager(
            self.cache_manager, self.table_manager, self.decode_manager
        )
        
        # 5. Load tokenizer for EOS detection
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_path)
        self.eos_token_id = self.tokenizer.eos_token_id
```

### 3.6 Engine Initialization

The `Engine` class is responsible for the actual model and GPU resources:

```python
# python/minisgl/engine/engine.py
class Engine:
    def __init__(self, config: EngineConfig):
        # 1. Set up GPU device
        set_tp_info(rank=config.tp_info.rank, size=config.tp_info.size)
        self.device = torch.device(f"cuda:{config.tp_info.rank}")
        torch.cuda.set_device(self.device)
        self.stream = torch.cuda.Stream()
        
        # 2. Initialize distributed communication
        self.tp_cpu_group = self._init_communication(config)
        
        # 3. Load model on meta device first (no memory allocation)
        set_rope_device(self.device)
        with torch.device("meta"), torch_dtype(config.dtype):
            self.model = create_model(config.model_path, config.model_config)
        
        # 4. Load actual weights
        self.model.load_state_dict(self._load_weight_state_dict(config))
        
        # 5. Determine KV cache size based on available memory
        self.num_pages = self._determine_num_pages(init_free_memory, config)
        
        # 6. Create KV cache
        self.kv_cache = create_kvcache(
            num_layers=self.model_config.num_layers,
            num_kv_heads=self.model_config.num_kv_heads,
            num_pages=self.num_pages + 1,  # +1 for dummy page
            head_dim=self.model_config.head_dim,
            device=self.device,
            dtype=self.dtype,
        )
        
        # 7. Create page table for KV cache management
        self.page_table = create_page_table(
            (config.max_running_req + 1, self.max_seq_len),
            device=self.device,
        )
        
        # 8. Create attention backend
        self.attn_backend = create_attention_backend(
            config.model_config, self.kv_cache, config.attention_backend, self.page_table
        )
        
        # 9. Set up global context
        self.ctx = Context(
            page_size=1,
            kv_cache=self.kv_cache,
            attn_backend=self.attn_backend,
            page_table=self.page_table,
        )
        set_global_ctx(self.ctx)
        
        # 10. Capture CUDA graphs for decode
        self.graph_runner = GraphRunner(...)
```

**Memory layout after initialization (for Qwen3-0.6B on H100):**

```
GPU Memory (~80GB total):
├── Model Weights: ~1.2 GB
├── KV Cache: ~60 GB (millions of pages)
├── CUDA Graph Memory: ~2 GB
├── Attention Workspace: ~128 MB
└── Free: ~16 GB (for activations during forward pass)
```

### 3.7 Model Architecture

Let's look at how the model is constructed:

```python
# python/minisgl/models/qwen3.py
class Qwen3ForCausalLM(BaseLLMModel):
    def __init__(self, config: ModelConfig):
        self.model = Qwen3Model(config)
        self.lm_head = ParallelLMHead(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            tie_word_embeddings=config.tie_word_embeddings,
            tied_embedding=self.model.embed_tokens if config.tie_word_embeddings else None,
        )

class Qwen3Model(BaseOP):
    def __init__(self, config: ModelConfig):
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
        )
        self.layers = OPList([
            Qwen3DecoderLayer(config, layer_id) 
            for layer_id in range(config.num_layers)
        ])
        self.norm = RMSNormFused(size=config.hidden_size, eps=config.rms_norm_eps)

class Qwen3DecoderLayer(BaseOP):
    def __init__(self, config: ModelConfig, layer_id: int):
        self.self_attn = RopeAttn(config, layer_id, has_qk_norm=True)
        self.mlp = GatedMLP(config)
        self.input_layernorm = RMSNormFused(size=config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNormFused(size=config.hidden_size, eps=config.rms_norm_eps)
```

**Layer structure for Qwen3-0.6B:**
- 28 decoder layers
- Hidden size: 1024
- 16 attention heads (Q), 2 KV heads (GQA)
- Head dimension: 64
- Intermediate size: 2816

---

## 4. Phase 2: User Request Arrives

### 4.1 FastAPI Endpoint

When our curl request arrives:

```python
# python/minisgl/server/api_server.py
@app.post("/v1/chat/completions")
async def v1_completions(req: OpenAICompletionRequest):
    state = get_global_state()
    
    # 1. Extract messages from request
    if req.messages:
        prompt = [msg.model_dump() for msg in req.messages]
    else:
        prompt = req.prompt
    
    # 2. Create unique user ID
    uid = state.new_user()
    
    # 3. Send tokenization request
    await state.send_one(
        TokenizeMsg(
            uid=uid,
            text=prompt,  # [{"role": "user", "content": "What is the capital of France?"}]
            sampling_params=SamplingParams(
                ignore_eos=req.ignore_eos,
                max_tokens=req.max_tokens,  # 50
            ),
        )
    )
    
    # 4. Return streaming response
    return StreamingResponse(
        state.stream_chat_completions(uid),
        media_type="text/event-stream",
    )
```

### 4.2 FrontendManager State

The `FrontendManager` tracks all in-flight requests:

```python
# python/minisgl/server/api_server.py
@dataclass
class FrontendManager:
    config: ServerArgs
    send_tokenizer: ZmqAsyncPushQueue[BaseTokenizerMsg]
    recv_tokenizer: ZmqAsyncPullQueue[BaseFrontendMsg]
    uid_counter: int = 0
    ack_map: Dict[int, List[UserReply]] = field(default_factory=dict)
    event_map: Dict[int, asyncio.Event] = field(default_factory=dict)
    
    def new_user(self) -> int:
        uid = self.uid_counter
        self.uid_counter += 1
        self.ack_map[uid] = []  # Will store responses
        self.event_map[uid] = asyncio.Event()  # For async notification
        return uid
```

**For our example:**
- `uid = 0` (first request)
- `ack_map[0] = []` (empty, waiting for responses)
- `event_map[0] = asyncio.Event()` (not set yet)

### 4.3 Message Serialization

The `TokenizeMsg` is serialized for ZMQ transport:

```python
# python/minisgl/message/tokenizer.py
@dataclass
class TokenizeMsg(BaseTokenizerMsg):
    uid: int
    text: str | List[Dict[str, str]]
    sampling_params: SamplingParams

# python/minisgl/message/utils.py
def serialize_type(self) -> Dict:
    serialized = {"__type__": self.__class__.__name__}
    for k, v in self.__dict__.items():
        serialized[k] = _serialize_any(v)
    return serialized
```

**Serialized message:**
```python
{
    "__type__": "TokenizeMsg",
    "uid": 0,
    "text": [{"role": "user", "content": "What is the capital of France?"}],
    "sampling_params": {
        "__type__": "SamplingParams",
        "top_k": 1,
        "ignore_eos": False,
        "temperature": 0.0,
        "max_tokens": 50
    }
}
```

---

## 5. Phase 3: Tokenization

### 5.1 Tokenizer Worker

The tokenizer worker runs in a separate process:

```python
# python/minisgl/tokenizer/server.py
@torch.inference_mode()
def tokenize_worker(
    *,
    tokenizer_path: str,
    addr: str,
    backend_addr: str,
    frontend_addr: str,
    # ...
) -> None:
    # 1. Set up ZMQ connections
    send_backend = ZmqPushQueue(backend_addr, create=False, encoder=BaseBackendMsg.encoder)
    send_frontend = ZmqPushQueue(frontend_addr, create=False, encoder=BaseFrontendMsg.encoder)
    recv_listener = ZmqPullQueue(addr, create=create, decoder=BatchTokenizerMsg.decoder)
    
    # 2. Load tokenizer
    tokenizer: LlamaTokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
    
    # 3. Create managers
    tokenize_manager = TokenizeManager(tokenizer)
    detokenize_manager = DetokenizeManager(tokenizer)
    
    # 4. Main loop
    while True:
        pending_msg = _unwrap_msg(recv_listener.get())
        
        # Separate tokenize and detokenize messages
        tokenize_msg = [m for m in pending_msg if isinstance(m, TokenizeMsg)]
        detokenize_msg = [m for m in pending_msg if isinstance(m, DetokenizeMsg)]
        
        # Handle tokenization
        if len(tokenize_msg) > 0:
            tensors = tokenize_manager.tokenize(tokenize_msg)
            batch_output = BatchBackendMsg(data=[
                UserMsg(uid=msg.uid, input_ids=t, sampling_params=msg.sampling_params)
                for msg, t in zip(tokenize_msg, tensors)
            ])
            send_backend.put(batch_output)
        
        # Handle detokenization (we'll see this later)
        if len(detokenize_msg) > 0:
            # ...
```

### 5.2 TokenizeManager

```python
# python/minisgl/tokenizer/tokenize.py
class TokenizeManager:
    def __init__(self, tokenizer: LlamaTokenizer) -> None:
        self.tokenizer = tokenizer

    def tokenize(self, msgs: List[TokenizeMsg]) -> List[torch.Tensor]:
        results: List[torch.Tensor] = []
        for msg in msgs:
            if isinstance(msg.text, list):
                # Apply chat template for message list
                prompt = self.tokenizer.apply_chat_template(
                    msg.text,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            else:
                prompt = msg.text
            
            input_ids: torch.Tensor = self.tokenizer.encode(prompt, return_tensors="pt")
            results.append(input_ids.view(-1).to(torch.int32))
        return results
```

**For our example:**

Input:
```python
[{"role": "user", "content": "What is the capital of France?"}]
```

After `apply_chat_template`:
```
<|im_start|>user
What is the capital of France?<|im_end|>
<|im_start|>assistant
```

After tokenization:
```python
torch.tensor([151644, 872, 198, 3838, 374, 279, 6722, 315, 9625, 30, 151645, 198, 151644, 77091, 198], dtype=torch.int32)
# 15 tokens
```

### 5.3 UserMsg Creation

The tokenized result is wrapped in a `UserMsg`:

```python
# python/minisgl/message/backend.py
@dataclass
class UserMsg(BaseBackendMsg):
    uid: int
    input_ids: torch.Tensor  # CPU 1D int32 tensor
    sampling_params: SamplingParams
```

**Our UserMsg:**
```python
UserMsg(
    uid=0,
    input_ids=tensor([151644, 872, 198, ...]),  # 15 tokens
    sampling_params=SamplingParams(max_tokens=50, temperature=0.0, ...)
)
```

---

## 6. Phase 4: Scheduling

### 6.1 Scheduler Main Loop

The scheduler runs an infinite loop processing messages:

```python
# python/minisgl/scheduler/scheduler.py
@torch.inference_mode()
def run_forever(self) -> NoReturn:
    if ENV.DISABLE_OVERLAP_SCHEDULING:
        # Simple mode: process one batch at a time
        while True:
            self.normal_loop()
    else:
        # Overlap mode: process current batch while scheduling next
        data = None
        while True:
            data = self.overlap_loop(data)
```

### 6.2 Overlap Loop (Default)

```python
# python/minisgl/scheduler/scheduler.py
def overlap_loop(self, last_data: ForwardData | None) -> ForwardData | None:
    """
    The main loop of overlapping scheduling and execution.
    
    Timeline:
    GPU: [Batch N compute    ][Batch N+1 compute  ]
    CPU:        [Process N-1 ][Schedule N+1       ]
    """
    # 1. Receive new messages (non-blocking if we have work)
    blocking = not (
        last_data or 
        self.prefill_manager.runnable or 
        self.decode_manager.runnable
    )
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

**Why overlap scheduling?**
- GPU compute takes ~10ms for a batch
- CPU scheduling takes ~1ms
- Without overlap: 11ms total
- With overlap: 10ms total (CPU work hidden behind GPU)

### 6.3 Processing Incoming Messages

```python
# python/minisgl/scheduler/scheduler.py
def _process_one_msg(self, msg: BaseBackendMsg) -> None:
    if isinstance(msg, BatchBackendMsg):
        for msg in msg.data:
            self._process_one_msg(msg)
    elif isinstance(msg, ExitMsg):
        raise KeyboardInterrupt
    elif isinstance(msg, UserMsg):
        logger.debug_rank0("Received user msg: %s", msg)
        
        # Validate input length
        input_len, max_seq_len = len(msg.input_ids), self.engine.max_seq_len
        if input_len >= max_seq_len:
            logger.warning_rank0(f"Input too long, dropping request {msg.uid}")
            return
        
        # Adjust max_tokens if needed
        max_output_len = max_seq_len - input_len
        if msg.sampling_params.max_tokens > max_output_len:
            msg.sampling_params.max_tokens = max_output_len
        
        # Add to prefill queue
        self.prefill_manager.add_one_req(msg)
```

### 6.4 PrefillManager

```python
# python/minisgl/scheduler/prefill.py
@dataclass
class PrefillManager:
    cache_manager: CacheManager
    table_manager: TableManager
    decode_manager: DecodeManager
    pending_list: List[PendingReq] = field(default_factory=list)

    def add_one_req(self, req: UserMsg) -> None:
        self.pending_list.append(PendingReq(req.uid, req.input_ids, req.sampling_params))
```

**PendingReq for our example:**
```python
PendingReq(
    uid=0,
    input_ids=tensor([151644, 872, 198, ...]),  # 15 tokens
    sampling_params=SamplingParams(max_tokens=50),
    chunked_req=None  # Not chunked
)
```

### 6.5 Scheduling Next Batch

```python
# python/minisgl/scheduler/scheduler.py
def _schedule_next_batch(self) -> ForwardInput | None:
    # Priority: Prefill first, then Decode
    batch = (
        self.prefill_manager.schedule_next_batch(self.prefill_budget)
        or self.decode_manager.schedule_next_batch()
    )
    return self._prepare_batch(batch) if batch else None
```

### 6.6 PrefillAdder - Resource Allocation

```python
# python/minisgl/scheduler/prefill.py
@dataclass
class PrefillAdder:
    token_budget: int
    reserved_size: int
    cache_manager: CacheManager
    table_manager: TableManager

    def _try_allocate_one(self, req: PendingReq) -> Tuple[BaseCacheHandle, int] | None:
        # 1. Check if we have a free table slot
        if self.table_manager.available_size == 0:
            return None
        
        # 2. Check radix cache for prefix match
        handle, match_indices = self.cache_manager.match_req(req)
        cached_len = handle.cached_len
        
        # 3. Estimate memory needed
        extend_len = req.input_len - cached_len
        estimated_len = extend_len + req.output_len
        
        # 4. Check if we have enough cache space
        if estimated_len + self.reserved_size > self.cache_manager.available_size:
            return None
        
        # 5. Lock the cache handle (prevents eviction)
        self.cache_manager.lock(handle)
        
        # 6. Allocate table slot
        table_idx = self.table_manager.allocate()
        
        # 7. Copy cached KV indices to page table
        if cached_len > 0:
            device_ids = self.table_manager.token_pool[table_idx][:cached_len]
            page_entry = self.table_manager.page_table[table_idx][:cached_len]
            device_ids.copy_(req.input_ids[:cached_len].pin_memory(), non_blocking=True)
            page_entry.copy_(match_indices)
        
        return handle, table_idx
```

**For our first request (no cache hit):**
- `cached_len = 0` (nothing in cache yet)
- `extend_len = 15` (all