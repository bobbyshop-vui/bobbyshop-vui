# 👋 Hello! Xin chào! Tôi là bobbydeveloper2014 và bobbyshop-vui tên là Lê Đình Quốc Hưng
# 🔗 Liên kết
- 🎥 [Kênh YouTube: Hưng MC](https://www.youtube.com/@hungmc4636)
- 🎵 [TikTok: @bobbydeveloper](https://tiktok.com/@bobbydeveloper)
## 🧑‍💻 About Me | Giới thiệu
- 🧒 I'm 11 years old. Tôi 11 tuổi.
- 💻 Passionate about coding apps, websites, and low-level stuff! Đam mê lập trình ứng dụng, web, cả lập trình cấp thấp!
- 🌱 Exploring every corner of the coding world. Luôn tò mò khám phá công nghệ.

## 🛠️ Skills | Kỹ năng
- Python, JavaScript, C,bybylang (https://github.com/bobbyshop-vui/bybylang)
- Web: HTML, CSS, JavaScript, bybywebscript(commingsoon)
- App: Python, JavaScript, bybylang (https://github.com/bobbyshop-vui/bybylang)
- Low-level: C, đang học Assembly, bybylang (https://github.com/bobbyshop-vui/bybylang) 

## 🚀 Projects | Dự án nổi bật
- Personal learning website | Website học tập cá nhân
- **BybyLang**(https://github.com/bobbyshop-vui/bybylang): My own programming language! | Ngôn ngữ lập trình tự phát triển!

## 🎯 Slogan
> Code is a game, every line is a discovery!  
> Lập trình như một trò chơi, mỗi dòng là một khám phá!

---
# Nimformer Framework ([https://github.com/bobbyshop-vui/nimformers-framework](https://github.com/bobbyshop-vui/nimformers-framework))

A pure **Nim** port of a small transformer (char-level LM) with real
forward/backward passes running on the **Metal GPU** (macOS), plus a custom
quantization system (int8 / int4 / fp8 / "APF" auto-adaptive per tensor) for
compressing checkpoints.

No dependency on Python, PyTorch, or tinygrad — every tensor op, matmul,
attention, LayerNorm, Adam optimizer, and quantization routine is
hand-written in Nim + Metal Shading Language.

---

## 1. Table of contents

- [2. Project layout](#2-project-layout)
- [3. Requirements & setup](#3-requirements--setup)
- [4. Build](#4-build)
- [5. Using each library module](#5-using-each-library-module)
  - [5.1. `customfloat.nim` — CustomFloat & APF](#51-customfloatnim--customfloat--apf)
  - [5.2. `quant.nim` — Tensor quantization](#52-quantnim--tensor-quantization)
  - [5.3. `metal_ai.nim` — MetalContext & GPU kernels](#53-metal_ainim--metalcontext--gpu-kernels)
  - [5.4. `nimformer.nim` — Model, forward/backward, optimizer](#54-nimformernim--model-forwardbackward-optimizer)
- [6. Full end-to-end example](#6-full-end-to-end-example)
- [7. Using `main.nim` — the CLI training driver](#7-using-mainnim--the-cli-training-driver)
- [8. The `.nimq` file format](#8-the-nimq-file-format)
- [9. Important notes / known issues](#9-important-notes--known-issues)

---

## 2. Project layout

```
customfloat.nim     -- CustomFloat + APF, pure Nim, RUNS ON ANY OS (no GPU needed)
quant.nim            -- Quantization: int8/int4/fp8_e4m3/fp8_e5m2/custom/auto(APF)
                        + reading/writing a compressed state dict to a .nimq binary file
nimformer.nim        -- Tensor + Linear/LayerNorm/Attention/FeedForward/Embedding
                        + TransformerBlock + NimformerModel + ApfAdam optimizer
                        (REAL forward/backward, running on GPU via metal_ai)
metal_bridge.h       -- Generic C header so Nim can call into Metal via {.importc.}
metal_bridge.m       -- Objective-C implementation of the bridge (compiled alongside Nim)
metal_kernels.metal  -- Real Metal kernels: add, matmul, relu/sigmoid/tanh,
                        softmax, layernorm, embedding_lookup
metal_ai.nim         -- MetalContext (device/queue/pipeline cache/buffer pool)
                        + Nim wrappers for ALL the kernels above
main.nim             -- CLI driver: tokenizer, data loading, training loop,
                        checkpoint + resume, quantization on save
tokenizer.json       -- Pre-saved byte-level tokenizer (vocab_size=198)
databricks-dolly-15k.jsonl -- Sample dataset for test training (instruction/response format)
Makefile             -- make / make build / make run / make metal / make clean
```

`customfloat.nim`, `quant.nim`, `nimformer.nim`, `metal_ai.nim` are **PURE
LIBRARIES** — no `when isMainModule`, meant only to be `import`ed. `main.nim`
is the actual CLI entry point used to run training.

> ⚠️ **Note on the current contents of the repo:** at the time this README
> was written, `test_nimformer.nim` and `metal_bridge.h` in the uploaded
> folder both **mistakenly contain the same content** (the actual content of
> `main.nim`) instead of their own correct content (`test_nimformer.nim`
> should be a small, CLI-free test file; `metal_bridge.h` should be a plain C
> header). Before pushing/building, make sure these two files are replaced
> with their correct content, otherwise `make build` will fail to compile
> with clang. The Git repo should contain: `main.nim` (correct content),
> `metal_bridge.h` (the C header), and (recommended) a lightweight
> `test_nimformer.nim` that just builds+forwards+backwards+quantizes as a
> quick demo, with no CLI/external data required.

---

## 3. Requirements & setup

- **macOS** with a Metal-capable GPU — required for the GPU part. Only
  `customfloat.nim` is CPU-only and works on Linux/Windows.
- Xcode Command Line Tools (needed for `clang` to compile the Objective-C
  code and link the `Metal`/`Foundation` frameworks):
  ```bash
  xcode-select --install
  ```
- Nim >= 2.0:
  ```bash
  brew install nim
  # or via choosenim: https://github.com/dom96/choosenim
  ```

---

## 4. Build

Using the provided `Makefile`:

```bash
make            # build test_nimformer + metal_ai, then run test_nimformer
make build      # build only, don't run
make run        # build + run test_nimformer
make metal      # build + run metal_ai on its own (standalone GPU kernel demo)
make clean      # remove binaries, nimcache/, and any model_*.nimq checkpoints
```

Or build manually, piece by piece:

```bash
# CPU-only part, runs on any OS
nim c -r customfloat.nim

# GPU part (macOS required) — Nim automatically invokes clang to compile metal_bridge.m
nim c -d:release -r metal_ai.nim

# Build main.nim (the real CLI training driver)
nim c -d:release -o:main main.nim
```

Manual step-by-step build if you need to debug:
```bash
clang -fobjc-arc -c metal_bridge.m -o metal_bridge.o \
  -framework Metal -framework Foundation

nim c -d:release --passL:"metal_bridge.o -framework Metal -framework Foundation" metal_ai.nim
```

---

## 5. Using each library module

### 5.1. `customfloat.nim` — CustomFloat & APF

`CustomFloat` describes a custom floating-point type with a configurable
number of exponent/mantissa bits (like fp8, fp16, bfloat16... except you
define the bit widths yourself). "APF" (Adaptive Precision Float)
automatically figures out how many bits are actually needed based on the
values in a tensor.

```nim
import customfloat

# Define a custom dtype: 5 exponent bits, 5 mantissa bits (fp11)
let cf = newCustomFloat(exponentBits = 5, mantissaBits = 5, name = "fp11")
echo cf.totalBits    # 11 (1 sign + 5 exp + 5 mant)
echo cf.itemSize     # 2 (bytes needed to store 1 element, rounded up)

# Encode/decode a float32 array
let data = @[1.5'f32, -0.001, 3.14159, 100000.0]
let packed: seq[uint8] = encodeArray(data, cf)
let restored: seq[float32] = decodeArray(packed, cf)

# Ready-made presets (equivalent to FP8_E4M3 ... FP64 in the Python version)
echo FP8_E4M3.totalBits   # 8
echo FP8_E5M2.totalBits   # 8
echo FP16C.totalBits      # 16
echo FP32C.totalBits      # 32
```

**APF — auto-build a dtype for a tensor** (no need to pick exponent/mantissa
bits yourself):

```nim
# Automatically compute the exponent/mantissa bits needed for THIS tensor
let cf2 = buildCustomDtypeForTensor(data)
echo cf2.name   # e.g. "auto_e4m6"

# Or pass both (weight, gradient) — the gradient helps APF figure out how
# many extra mantissa bits are needed so Adam updates don't underflow
let grad = @[0.0001'f32, -0.0002, 0.0003, 0.0001]
let cf3 = buildCustomDtypeForTensor(data, grad)

# Convenience helper combining encode + dtype-build in one call (used in the
# training loop)
let (encoded, cfUsed) = apfCastForTraining(data, grad)
let decoded = apfDecodeForTraining(encoded, cfUsed)
```

APF configuration constants you can tune when calling (all have sensible
defaults):
```nim
const
  APF_DEFAULT_REL_ERROR_TOL = 1e-3   # max acceptable relative error
  APF_MIN_MANTISSA_BITS = 2
  APF_MAX_MANTISSA_BITS = 23
  APF_EXP_MARGIN_BITS = 1
```

### 5.2. `quant.nim` — Tensor quantization

Wraps `customfloat.nim` into a single unified quantization API, supporting
7 kinds via the `QuantKind` enum:

```nim
type QuantKind* = enum
  qkFp32Raw   # no compression — used for bias/LayerNorm (error-sensitive)
  qkInt8      # symmetric int8, scale = max(|x|)/127
  qkInt4      # symmetric int4, 2 values/byte, scale = max(|x|)/7
  qkFp8E4M3   # CustomFloat(4,3)
  qkFp8E5M2   # CustomFloat(5,2)
  qkCustom    # any CustomFloat you declare yourself (any bit width)
  qkAuto      # APF — auto-builds a CustomFloat for EACH tensor
```

Use the general dispatcher API (recommended, instead of calling
`quantizeInt8`/`quantizeInt4`/... individually):

```nim
import quant

let w = @[0.5'f32, -1.2, 3.7, -0.001, 2.2]
let shape = @[5]

# --- int8 ---
let q8 = quantizeTensor(w, shape, qkInt8)
let back8 = dequantizeTensor(q8)

# --- int4 ---
let q4 = quantizeTensor(w, shape, qkInt4)

# --- fp8 (2 variants) ---
let qf8a = quantizeTensor(w, shape, qkFp8E4M3)
let qf8b = quantizeTensor(w, shape, qkFp8E5M2)

# --- custom dtype of your choosing (e.g. fp6: 3 exponent bits + 2 mantissa bits) ---
let myDtype = newCustomFloat(3, 2, "fp6")
let qCustom = quantizeTensor(w, shape, qkCustom, customCf = myDtype)

# --- auto (APF), pass the gradient too if you have it for a better estimate ---
let grad = @[0.001'f32, -0.002, 0.0007, 0.0001, 0.0003]
let qAuto = quantizeTensor(w, shape, qkAuto, gradArr = grad)

# --- no compression (raw fp32), used for bias/gamma/beta ---
let qRaw = quantizeTensor(w, shape, qkFp32Raw)
```

Each result is a `QuantTensor` object; decompress any of them back with a
single `dequantizeTensor(qt)` call for EVERY kind (it dispatches internally
based on `qt.kind`).

**Saving / loading a compressed state dict to a `.nimq` binary file:**

```nim
# arch: 5 integers describing the architecture — a convention you choose,
# e.g. [vocab, embedDim, nHeads, nLayers, ffMult]
let arch = [1000, 64, 4, 2, 4]
var sd: seq[(string, QuantTensor)] = @[
  ("outProj.weight", q8),
  ("outProj.bias", qRaw)
]
saveQuantStateDict("checkpoint.nimq", arch, sd)

let (loadedArch, loadedSd) = loadQuantStateDict("checkpoint.nimq")
```

### 5.3. `metal_ai.nim` — MetalContext & GPU kernels

Create the GPU context once and reuse it for the entire lifetime of the
program:

```nim
import metal_ai

let ctx = newMetalContext()   # raises IOError if the machine has no Metal GPU

# Available kernels — every function takes flat seq[float32] arrays
let a = @[1.0'f32, 2, 3, 4]
let b = @[10.0'f32, 20, 30, 40]
let c = ctx.metalAdd(a, b)                      # elementwise add

let m = ctx.metalMatmul(a, 2, 2, b, 2, 2)       # [M,K] x [K,N] -> [M,N]

let r1 = ctx.metalRelu(a)
let r2 = ctx.metalSigmoid(a)
let r3 = ctx.metalTanh(a)

let sm = ctx.metalSoftmax(a, rows = 1, cols = 4)          # row-wise softmax
let ln = ctx.metalLayernorm(a, gamma = @[1'f32,1,1,1],
                             beta = @[0'f32,0,0,0], rows = 1, cols = 4)

let table = @[0.1'f32, 0.2, 0.3, 0.4, 0.5, 0.6]  # [vocab=3, dim=2]
let ids = @[int32(0), 2]
let emb = ctx.metalEmbeddingLookup(table, vocab = 3, dim = 2, indices = ids)

# Encode/decode CustomFloat directly on the GPU (instead of on the CPU as in customfloat.nim)
import customfloat
let cf = buildCustomDtypeForTensor(a)
let packed = ctx.customfloatEncodeGpu(a, cf)
let restored = ctx.customfloatDecodeGpu(packed, cf)

# Clean up the buffer pool before the program exits (not mandatory, process
# exit reclaims memory too, but tidier if the context lives for a long time)
closeMetalContext(ctx)
```

Internally, `MetalContext` automatically:
- **Caches pipelines** by kernel name (compiles each kernel exactly once).
- **Pools GPU buffers** by size (bytes) to reuse them instead of repeatedly
  allocating/deallocating on every dispatch — this prevents the process's
  RAM footprint from growing gradually with every training step.

You'll rarely need to touch the lower-level C functions
(`mtl_dispatch`, `mtl_command_buffer_*`, `mtl_encoder_*`) directly — they
live inside `metal_ai.nim` for adding new kernels, without needing to touch
`metal_bridge.h/.m`.

### 5.4. `nimformer.nim` — Model, forward/backward, optimizer

**Basic Tensor:**
```nim
import nimformer

var t = newTensor(@[2, 3])              # all-zero tensor, shape [2,3]
var t2 = randnTensor(@[2, 3], scale=0.02'f32)  # random gaussian * scale
```

**Building a full transformer model:**
```nim
import nimformer, metal_ai

let ctx = newMetalContext()

var model = newNimformerModel(
  vocab = 64,      # tokenizer vocabulary size
  embedDim = 32,   # embedding dimension
  nHeads = 4,      # number of attention heads
  nLayers = 2,     # number of transformer blocks
  ffMult = 4       # feed-forward expansion factor (hidden = embedDim * ffMult)
)
```

**Forward — a single sequence or a whole batch:**
```nim
# A single sequence (ids: seq[int])
let logits = model.forward(@[1, 5, 9, 2], ctx)     # shape [T, vocab]

# A batch of B sequences of the same length T (recommended — much faster
# because the GPU gets a single matmul with M=B*T rows instead of B
# sequential calls)
let idsBatch = @[@[1, 5, 9, 2], @[3, 3, 1, 0]]
let logitsBatch = model.forwardBatch(idsBatch, ctx) # shape [B, T, vocab]
```

**Backward — needs the gradient of the loss w.r.t. logits (`dLoss`):**
```nim
# Single sequence:
let grads = model.backward(ids, dLoss, ctx)          # seq[Tensor], one gradient per parameter

# Batch (recommended):
let gradsBatch = model.backwardBatch(idsBatch, dLossBatch, ctx)
```
Order of the returned `grads`: `outProj.dW, outProj.dB`, then each block
(from the LAST block back to the FIRST) in the order
`attn.qkv.{W,B}, attn.proj.{W,B}, ff.fc1.{W,B}, ff.fc2.{W,B},
ln1.{gamma,beta}, ln2.{gamma,beta}`, and finally `embed.weight`.

**ApfAdam optimizer** — regular Adam plus re-quantizing the parameter with
APF every `requantizeEvery` steps:
```nim
var state = newApfAdamState(paramLen = model.outProj.weight.data.len)

# Every training step, for EACH parameter and its corresponding gradient:
let cfUsed = apfAdamStep(model.outProj.weight, gradOutProjW, state,
                          lr = 3e-3'f32, requantizeEvery = 50)
echo cfUsed.name   # the APF dtype just rebuilt for this parameter (if the requantize milestone was hit)
```
`requantizeEvery = 1` means APF re-quantizes the parameter after EVERY Adam
step; set it higher (e.g. 50) to reduce overhead, since building a dtype is
computationally expensive.

---

## 6. Full end-to-end example

```nim
import metal_ai, nimformer, quant, customfloat, math

let ctx = newMetalContext()

# 1. Build a small model
var model = newNimformerModel(vocab = 64, embedDim = 32, nHeads = 4,
                                nLayers = 2, ffMult = 4)

# 2. Forward a fake batch
let idsBatch = @[@[1, 5, 9, 2], @[3, 3, 1, 0]]
let logits = model.forwardBatch(idsBatch, ctx)      # [B=2, T=4, vocab=64]

# 3. Fake a random loss gradient and backward (in practice: cross-entropy)
var dLoss = randnTensor(logits.shape, scale = 0.01'f32)
let grads = model.backwardBatch(idsBatch, dLoss, ctx)

# 4. One Adam step on the first parameter (outProj.weight) as an example
var state = newApfAdamState(model.outProj.weight.data.len)
discard apfAdamStep(model.outProj.weight, grads[0], state, lr = 3e-3'f32)

# 5. Quantize + save (int8, int4, fp8, auto/APF) and reload
let q8 = quantizeTensor(model.outProj.weight.data, model.outProj.weight.shape, qkInt8)
let restored = dequantizeTensor(q8)

var maxErr = 0'f32
for i in 0 ..< restored.len:
  maxErr = max(maxErr, abs(restored[i] - model.outProj.weight.data[i]))
echo "Max error after int8 compression: ", maxErr

closeMetalContext(ctx)
```

---

## 8. The `.nimq` file format

A self-defined binary format, header `"NIMQ1"`, layout:

```
"NIMQ1" (5 bytes)
arch: 5 x int32           -- e.g. [vocab, embedDim, nHeads, nLayers, ffMult]
n_tensors: int32
repeated n_tensors times:
  name: (int32 len) + bytes
  QuantTensor:
    kind: 1 byte (QuantKind)
    ndims: int32 + shape[i]: int32 x ndims
    scale: float32          -- used for int8/int4
    exponentBits: int32, mantissaBits: int32   -- used for fp8/custom/auto
    data_len: int32 + data bytes
```

Convention when used through `main.nim`: **weights** (Linear/Embedding
weight) use whatever dtype you chose via `--quant`; **bias and LayerNorm
(gamma/beta)** always stay `qkFp32Raw` (uncompressed) because they are small
and far more sensitive to error than large weight matrices.

---

## 9. Important notes / known issues

- **`metal_bridge.h`/`test_nimformer.nim` content mix-up** — see the warning
  in [section 2](#2-project-layout). Double-check these two files before
  building.
- **Buffer pool in `metal_ai.nim`**: if you write new kernels/wrappers
  yourself, remember to get buffers via `ctx.poolGet(length)` and return
  them via `releaseBufs(ctx, bufs)` (or `ctx.poolPut(buf)` for a single
  buffer) instead of calling `mtlNewBuffer`/`mtlRelease` directly — otherwise
  the process's RAM will grow gradually with every step as Metal keeps
  allocating new memory regions.
- **`@autoreleasepool`**: every function in `metal_bridge.m` that touches an
  Objective-C object (even ones we don't retain ourselves) should be wrapped
  in `@autoreleasepool { ... }` — a Nim program has no run loop to drain the
  autorelease pool the way a real Cocoa app does.
- **Pipeline cache**: do NOT call `mtl_get_pipeline` directly on every
  dispatch — always go through `ctx.pipeline(name)` (which caches it) to
  avoid recompiling the pipeline over and over.
- **Bias/LayerNorm precision**: always keep these as `qkFp32Raw`, don't
  compress them — error in these small parameters has a disproportionate
  effect on training/inference stability compared to large weight matrices.
- Some parts of the original Python transformer/Adam/attention code still
  had unfinished `PLACEHOLDER_*` sections — the ported version here uses the
  standard, correct math (multi-head causal self-attention, Post-LN,
  standard Adam bias-correction) rather than a line-by-line port of the
  original's unfinished placeholders.
# 🧠 Tổng quan BybyLang | BybyLang Overview (https://github.com/bobbyshop-vui/bybylang)

A Nim-based DSL, AOT-compiled to Nim source then built into a native binary. Its core is a set of `gpu ...` commands for running tensor ops on CUDA / Metal / OpenCL / TSIC-IR / CPU, plus generic control flow and a set of low-level hardware-simulation commands.

## Build & run

```bash
make build #build library don't run that test code
make test
# equivalent to:
nim c -d:release -o:bybylang bybylang.nim
./bybylang demo/demo_gpu.bybylang --aot=demo/demo_gpu_out
./demo/demo_gpu_out
```

`--aot=<path>` generates `<path>.nim`, compiles it with Nim, and produces the binary `<path>`.

---

## 1. General syntax

- One command per line in a `.bybylang` file. Indentation (spaces/tabs) determines the block for `if/elif/else/while/for`.
- `#` at the start of a line is a comment.
- `import name` or `import "path/name.bybylang"` — recursive import, `.bybylang` extension auto-appended, cycle-protected via absolute path.
- `print <expr>` → generates `echo <expr>`.
- `function NAME` ... a line containing only `NAME` closes the definition; call it with `call NAME`.
- `if cond:` / `elif cond:` / `else:` / `while cond:` / `for x in range(a, b):` — translated directly to the equivalent Nim construct, arbitrarily nestable.
- `mode is N` (N = 1..4) → prints "Mode 1: Low-level" / "Mode 2: Mid-level" / "Mode 3: High-level" / "Mode 4: Web-level" (any other N → "Unknown mode").

## 2. `gpu ...` commands

### Select backend

```
gpu backend is "auto"      # auto | cpu | cuda | metal | opencl | tsic
```

`auto` probes in order: CUDA (NVIDIA) → Metal (macOS) → OpenCL → plain CPU.
By default, **silent CPU fallback is forbidden** (`gForbidCpuFallback = true`): if a specific GPU backend is requested and it's unavailable or fails, the program raises instead of silently falling back to CPU (so you never accidentally train on CPU without noticing).

### Declare an array

```
gpu array A = [1, 2, 3, 4]
```

Generates `var A: seq[float32] = @[1, 2, 3, 4].mapIt(it.float32)`. **Must be declared before use** — codegen has no hoisting; it translates lines strictly in file order.

### Basic binary ops

```
gpu add A, B -> C
gpu sub A, B -> C
gpu mul A, B -> C
gpu div A, B -> C
```

A trailing `size N` is descriptive only and doesn't affect codegen.

### Matmul

```
gpu matmul A, B -> C m M k K n N
```
A: [M×K], B: [K×N] → C: [M×N].

```
gpu matmul2 A1, B1, A2, B2 -> C1, C2 m1 M k1 K n1 N m2 M k2 K n2 N
```
Two independent matmuls fused into a single call.

### Activations

```
gpu relu X -> Y
gpu sigmoid X -> Y
gpu tanh X -> Y
gpu apflu X -> Y alpha A beta B      # defaults: alpha=0.1, beta=1.0 if omitted
gpu apflu_backward X, DY -> DX alpha A beta B   # alpha/beta optional, same defaults
```

### Fused add + activation

```
gpu fused_add_act A, B -> C act "relu"    # "relu" | "sigmoid" | "tanh" | "none"
```

### Softmax / LayerNorm

```
gpu softmax X -> Y rows R cols C
gpu layernorm X, GAMMA, BETA -> Y rows R cols C eps E
gpu layernorm_backward DY, X, GAMMA, BETA -> DX, DGAMMA, DBETA rows R cols C eps E
```

### Embedding

```
gpu embedding TABLE, INDICES -> Y vocab V dim D
```
`INDICES` is auto-cast to `int32`.

### Attention (fused)

```
gpu attention Q, K, V -> O, S_MATRIX B H S D scale SCALE
gpu attention_backward Q, K, V, S_MATRIX, DY -> DQ, DK, DV B H S D scale SCALE
```
B=batch, H=heads, S=seq_len, D=head_dim.

### Generic op (fallback)

```
gpu <other_op> A, B -> C
```
Generates `gpuOp("<other_op>", backend, A, B)` — requires a matching branch inside `gpuOp` to actually do something.

---

## 3. GPU Resident Tensor API (CUDA) — called directly from Nim

Defined at the end of `gpubackend.nim`. **Not** exposed through `.bybylang` syntax — call it from plain Nim code to avoid a CPU↔GPU round trip on every op:

```nim
let ta = cuUpload(dataA)          # upload seq[float32] to the GPU once
let tb = cuUpload(dataB)
let tc = cuAddR(ta, tb)           # result STAYS on the GPU
let out = cuDownload(tc)          # only download when you actually need the result
cuFree(ta); cuFree(tb); cuFree(tc)
```

Available: `cuUpload`, `cuUploadIndices`, `cuDownload`, `cuFree`, `cuMatmulR`, `cuAddR`, `cuSubR`, `cuMulR`, `cuDivR`, `cuReluR`, `cuSigmoidR`, `cuTanhR`, `cuSoftmaxR`, `cuLayernormR`, `cuEmbeddingLookupR`.

> ⚠️ **Note:** these resident-tensor calls exist but the forward pass generated from `.bybylang` (`genGpuLine`) does not call them — every `gpu ...` DSL command still uploads/downloads through `seq[float32]` on each call (per-op round trip), not the resident chain.

---

## 4. Low-level hardware-simulation commands

These **do** have real DSL syntax, parsed in `genBlock` (`bybylang.nim` ~line 668–714) and lowered to calls that operate on a simulated RAM (1024 ints) / BUS (seq[string]) / 32 Pins:

```
apu tran "chip1" with 101010          # -> apuTran("chip1", 101010)
apu mem write RAM0 with 42            # -> apuMem("write", "RAM0", "42")
apu mem read RAM0 with 0              # -> apuMem("read", "RAM0", "0")
apu core run                          # -> apuCore(1, "run")  (mode is always hardcoded to 1)
apu pin 3 is high                     # -> apuPin(3, "high")
bit send 1010                         # -> bitSend("1010")
bit recv                              # -> bitRecv()
mem map "device0"                     # -> memMap("device0")
mem push RAM0 with 99                 # -> memPush("RAM0", "99")
tran pulse pin 3 width 10ns           # -> tranPulse(3, "10ns")
```

Parser quirks worth knowing:
- `apu mem <action> <target> with <value>` — `target` is read positionally as the 4th word (`left[3]`) and quotes are stripped; `action` should be `write` or `read`.
- `apu core ...` ignores everything after `apu core` — it always emits `apuCore(1, "run")`.
- `apu pin <n> is <state>` reads `n` from word index 2 and `state` from word index 4 — extra or missing words will misparse silently.
- `tran pulse pin <n> width <w>` reads `n` from word index 3 and `w` as the **last** word on the line.

`apuTran`, `apuMem`, `apuCore`, `apuPin`, `bitSend`, `bitRecv`, `memMap`, `memPush`, `tranPulse` show up as "declared but not used" warnings only when compiling `bybylang.nim` itself (the compiler for the DSL) — that warning is irrelevant to whether the DSL syntax works. The AOT-generated output (`--aot=...nim`) re-declares its own copies of these same procs (see `bybylang.nim` ~line 775 onward) so the compiled program can actually call them at runtime.

---

## 5. Backends

| Backend | File | Notes |
|---|---|---|
| CUDA | `backends/cuda/cuda_driver.nim`, `cuda_runtime.nim` | Direct Driver API + cuBLAS, PTX JIT via `cuModuleLoadDataEx`, persistent context/module cache |
| Metal | `backends/metal/metal_backend.nim` (+ `.metal` kernels) | macOS GPU, buffer/pipeline cache |
| OpenCL | `backends/opencl/opencl_api.nim` | any other GPU/CPU |
| TSIC-IR | `tsic_ir.nim` | intermediate IR, lowerable to PTX / MSL / OpenCL C / GLSL |
| CPU | inside `gpubackend.nim` (`cpuRelu`, `cpuMatmul`, ...) | pure-Nim fallback, always correct but slow |

## 6. Minimal example

```
gpu backend is "tsic"
gpu array A = [1, 2, 3, 4]
gpu array B = [5, 6, 7, 8]
gpu add A, B -> C
print C
```
## 📫 Contact | Liên hệ
- Email: akirasumeragi699@gmail.com
- 🎥 [Kênh YouTube: Hưng MC](https://www.youtube.com/@hungmc4636)
- 🎵 [TikTok: @bobbydeveloper](https://tiktok.com/@bobbydeveloper)
- Zalo: 0862562514
- Minecraft: bobbydeveloper hoặc bobby_developer
---

**Thanks for visiting my hub!  
Cảm ơn bạn đã ghé thăm! Nếu bạn cũng đam mê lập trình, hãy kết nối với mình!**
