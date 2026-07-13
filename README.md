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
# Nimformer Framework (https://github.com/bobbyshop-vui/nimformers-framework)

Nimformer is a pure **Nim** transformer framework that supports multiple backends (CPU, Metal, CUDA), adaptive quantization (int8/int4/fp8/APF), and compressed `.nimq` checkpoints.

## 1. Project Structure

```
nimformers-framework/
├── backend.nim
├── customfloat.nim
├── databricks-dolly-15k.jsonl
├── export_to_nimq.py
├── finetune.nimq.ckpt
├── harness_attn.nim
├── harness_attn2.nim
├── LICENSE
├── Makefile
├── nim_inference.nim
├── nimformer.nim
├── quant.nim
├── readme.md
├── test_nimformer.nim
├── tokenizer.json
└── vendor/
    └── bybylang/
        ├── LICENSE
        ├── backends/
        │   ├── cuda/
        │   │   ├── cuda_driver.nim
        │   │   ├── cuda_runtime.nim
        │   │   └── kernels/
        │   │       └── vecop.ptx
        │   ├── metal/
        │   │   ├── kernels/
        │   │   │   └── vecop_matmul.metal
        │   │   ├── metal_backend.nim
        │   │   ├── metal_shim.h
        │   │   └── metal_shim.m
        │   └── opencl/
        │       ├── kernels/
        │       │   └── vecop_matmul.cl
        │       └── opencl_api.nim
        ├── bybylang.nim
        ├── demo/
        │   ├── demo_gpu.bybylang
        │   └── demo_gpu_out.nim
        ├── gpubackend.nim
        ├── makefile
        ├── readme.md
        └── tsic_ir.nim
```

---

## 2. Backend API – Available Operations

`backend.nim` provides a unified interface for CPU, Metal, and CUDA. All functions below are available through the `Backend` object.

### 2.1. `beMatmul` – Matrix Multiplication

```
proc beMatmul(ctx: Backend, a: seq[float32], M, K: int,
              b: seq[float32], K2, N: int): seq[float32]
```

Multiplies matrix `A (M x K)` by `B (K x N)` and returns a flattened `C (M x N)`.

### 2.2. `beMatmul2` – Two Independent Matrix Multiplications in a Single Call

```
proc beMatmul2(ctx: Backend,
               a1: seq[float32], M1, K1: int, b1: seq[float32], K1b, N1: int,
               a2: seq[float32], M2, K2: int, b2: seq[float32], K2b, N2: int):
               tuple[y1, y2: seq[float32]]
```

Performs two separate matrix multiplications within the same command buffer (Metal/CUDA) to reduce overhead.

### 2.3. `beAdd` – Vector Addition

```
proc beAdd(ctx: Backend, a, b: seq[float32]): seq[float32]
```

### 2.4. `beSub` – Vector Subtraction

```
proc beSub(ctx: Backend, a, b: seq[float32]): seq[float32]
```

Subtracts each element: `a - b`.

### 2.5. `beMul` – Element-wise Multiplication

```
proc beMul(ctx: Backend, a, b: seq[float32]): seq[float32]
```

### 2.6. `beDiv` – Element-wise Division

```
proc beDiv(ctx: Backend, a, b: seq[float32]): seq[float32]
```

### 2.7. `beRelu` – ReLU Activation

```
proc beRelu(ctx: Backend, x: seq[float32]): seq[float32]
```

### 2.8. `beSigmoid` – Sigmoid Activation

```
proc beSigmoid(ctx: Backend, x: seq[float32]): seq[float32]
```

### 2.9. `beTanh` – Tanh Activation

```
proc beTanh(ctx: Backend, x: seq[float32]): seq[float32]
```

### 2.10. `beApflu` – APF Linear Unit Activation

```
proc beApflu(ctx: Backend, x: seq[float32],
             alpha: float32 = 0.1, beta: float32 = 0.1): seq[float32]
```

A customizable activation with `alpha`/`beta` parameters (default 0.1).

### 2.11. `beApfluBackward` – Backward Pass for APFLU

```
proc beApfluBackward(ctx: Backend, x, dy: seq[float32],
                      alpha: float32 = 0.1, beta: float32 = 0.1): seq[float32]
```

Computes the backward gradient for `beApflu`.

### 2.12. `beSoftmax` – Row-wise Softmax

```
proc beSoftmax(ctx: Backend, x: seq[float32], rows, cols: int): seq[float32]
```

Applies Softmax to each row of a `rows x cols` matrix.

### 2.13. `beLayernorm` – Layer Normalization

```
proc beLayernorm(ctx: Backend, x: seq[float32], gamma, beta: seq[float32],
                 rows, cols: int, eps: float32 = 1e-5): seq[float32]
```

### 2.14. `beLayernormBackward` – Backward Pass for Layer Normalization

```
proc beLayernormBackward(ctx: Backend, dy, x, gamma, beta: seq[float32],
                          rows, cols: int, eps: float32):
                          tuple[dx, dgamma, dbeta: seq[float32]]
```

Computes the gradients of the input, `gamma`, and `beta` from `dy` (the output gradient).

### 2.15. `beEmbeddingLookup` – Embedding Lookup

```
proc beEmbeddingLookup(ctx: Backend, table: seq[float32], vocab, dim: int,
                        indices: seq[int32]): seq[float32]
```

Looks up rows in `table (vocab x dim)` using `indices`, returning `(len(indices) x dim)`.

### 2.16. `beAttentionFused` – Fused Attention (Forward)

```
proc beAttentionFused(ctx: Backend, q, k, v, mask: seq[float32],
                       B, H, S, D: int, scale: float32):
                       tuple[o, s_matrix: seq[float32]]
```

Computes attention (Q, K, V, mask) fused into a single kernel call, reducing overhead compared to separate matmul + softmax operations. `B` = batch size, `H` = number of heads, `S` = sequence length, `D` = head dimension. Returns the output `o` and the attention score matrix `s_matrix` (reused during backward).

### 2.17. `beAttentionFusedBackward` – Fused Attention (Backward)

```
proc beAttentionFusedBackward(ctx: Backend, q, k, v, s_matrix, dy: seq[float32],
                               B, H, S, D: int, scale: float32):
                               tuple[dq, dk, dv: seq[float32]]
```

Computes gradients `dq`, `dk`, and `dv` for fused attention using the `s_matrix` saved during the forward pass.

---

## 2b. CustomFloat / APF Helpers (`backend.nim`)

These functions support encoding/decoding and automatic CustomFloat format selection for compressing data at lower precision than standard float32.

### 2b.1. `beCustomfloatEncode` – Encode float32 Array to Bytes

```
proc beCustomfloatEncode(ctx: Backend, arr: seq[float32], cf: CustomFloat): seq[uint8]
```

### 2b.2. `beCustomfloatDecode` – Decode Bytes Back to float32

```
proc beCustomfloatDecode(ctx: Backend, buf: seq[uint8], cf: CustomFloat): seq[float32]
```

### 2b.3. `beApfCastForTraining` – Automatically Select the Appropriate CustomFloat Format for Training

```
proc beApfCastForTraining(ctx: Backend, arr: seq[float32], gradArr: seq[float32] = [],
                           relErrorTol = APF_DEFAULT_REL_ERROR_TOL,
                           expMargin = APF_EXP_MARGIN_BITS):
                           tuple[data: seq[uint8], cf: CustomFloat]
```

Automatically chooses the optimal CustomFloat (APF) bit width/format based on `arr` (and `gradArr`, if provided), balancing the allowed relative error (`relErrorTol`) and exponent margin (`expMargin`). Returns the encoded data and the selected `CustomFloat` format.

---

## 3. Selecting a Backend

In `main.nim` or your own code:

```
import backend

# Automatically select: Metal if available, otherwise CPU
newBackend("auto")

# Explicit selection
newBackend("cpu")
newBackend("metal")
newBackend("cuda")
newBackend("tsic")
```

Or via build flags (Makefile):

```
make run-metal   # use Metal
make run-cpu     # use CPU
make run-cuda    # use CUDA (requires building cuda_kernels.cu first)
```

---

## 4. Quantization (`quant.nim`)

### 4.1. Quantization Types

```
type QuantKind* = enum
  qkFp32Raw   # no compression
  qkInt8      # symmetric int8
  qkInt4      # symmetric int4 (2 values per byte)
  qkFp8E4M3   # fp8 (4 exponent, 3 mantissa)
  qkFp8E5M2   # fp8 (5 exponent, 2 mantissa)
  qkCustom    # arbitrary CustomFloat
  qkAuto      # APF – automatically choose bit width
```

### 4.2. Quantization and Dequantization

```
import quant

let qt = quantizeTensor(data, shape, qkInt8)
let raw = dequantizeTensor(qt)

# Using APF (automatic)
let qtAuto = quantizeTensor(data, shape, qkAuto, gradArr = grad)
```

### 4.3. Reading/Writing `.nimq` Checkpoints

```
saveQuantStateDict("model.nimq", [vocab, embedDim, heads, layers, ffMult], sd)
let (arch, sd) = loadQuantStateDict("model.nimq")
```

---

## 5. Model (`nimformer.nim`)

### 5.1. Creating a Model

```
import nimformer, backend

var model = newNimformerModel(
  vocab = 64,
  embedDim = 32,
  nHeads = 4,
  nLayers = 2,
  ffMult = 4
)
```

### 5.2. Forward – Batch

```
let idsBatch = @[@[1, 2, 3], @[4, 5, 6]]   # [B=2, T=3]
let logits = model.forwardBatch(idsBatch, ctx)   # shape [2, 3, vocab]
```

### 5.3. Backward – Batch

```
let dLoss = randnTensor(@[2, 3, vocab], 0.01)   # dummy gradient
let grads = model.backwardBatch(idsBatch, dLoss, ctx)
# grads is seq[Tensor] in the following order: outProj.W, outProj.B,
# blocks from last to first (each block contains 12 gradients),
# and finally embed.weight.
```

### 5.4. ApfAdam Optimizer

```
var state = newApfAdamState(param.data.len)
let cf = apfAdamStep(param, grad, state, lr = 3e-3, requantizeEvery = 50)
```

---

## 6. Checkpoints (Save and Resume)

### 6.1. Save a Complete Checkpoint (Weights + Optimizer State + Step)

```
saveCheckpoint(model, states, stepNo, "ckpt.nimq.ckpt",
               weightKind = qkAuto,
               embedDim, nHeads, nLayers, ffMult)
```

### 6.2. Load a Complete Checkpoint

```
let (model, states, stepNo) = loadCheckpointFull("ckpt.nimq.ckpt")
# Continue training from stepNo
```

---

## 7. Export from Hugging Face → `.nimq`

```
python export_to_nimq.py --model TheBloke/deepseek-coder-6.7B-instruct-GPTQ --output model.nimq --int8
```

Supports both standard models and GPTQ models.

---

## 8. Inference with `nim_inference.nim` (for LLaMA)

```
import nim_inference, metal_ai

let ctx = newMetalContext()
let config = newLlamaConfig("path/to/hf/model")
let tokenizer = newHFTokenizer("path/to/hf/model")
var model = loadLlamaModel("model.nimq", config)

let output = generate(model, tokenizer, ctx, "def fib(n):", max_new_tokens = 64)
echo output
```

---

## 9. `.nimq` Format

- Header: `NIMQ1`
- Architecture: 5 integers (`vocab`, `embedDim`, `nHeads`, `nLayers`, `ffMult`)
- List of `(name, QuantTensor)` entries
- Convention: weights use the selected quantization type; biases and LayerNorm (`gamma`/`beta`) always use `qkFp32Raw`

---

## 10. Important Notes

- **Biases and LayerNorm** are always kept in fp32 to avoid numerical errors.
- **Buffer pool** in Metal: use `poolGet`/`poolPut` to avoid RAM leaks.
- **Pipeline cache**: compile kernels only once.
- **`@autoreleasepool`** in `metal_bridge.m`: required to clean up autorelease objects.
- **Backend `"auto"`**: if no GPU backend (CUDA/Metal/OpenCL/TSIC) is detected, it raises an error instead of silently falling back to CPU (`gForbidCpuFallback = true`). To allow CPU execution when no GPU is available, call `setForbidCpuFallback(false)` before `newBackend()`, or explicitly use `newBackend("cpu")`.
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
