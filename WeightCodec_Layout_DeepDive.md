# Ethos-U65 Weight Codec & IFM/OFM Layout — Deep Dive

> Companion to `Notes.md`. Focus: byte-level codec walk-through, physical memory layout, and
> optimization opportunities. All examples use **int8** weights/activations, **U65-256** config
> (1 core, 256 MACs/cycle, ublock = 2W × 2H × 8C).

---

## Table of Contents

1. [MLW Weight Codec — Step-by-Step](#1-mlw-weight-codec--step-by-step)
   - 1.1 [Overview of the 5-stage pipeline](#11-overview-of-the-5-stage-pipeline)
   - 1.2 [Stage 0 — Zero-point subtraction](#12-stage-0--zero-point-subtraction)
   - 1.3 [Stage 1 — Weight reorder (OHWI NPU traversal)](#13-stage-1--weight-reorder-ohwi-npu-traversal)
   - 1.4 [Stage 2 — Palette construction](#14-stage-2--palette-construction)
   - 1.5 [Stage 3 — Index mapping](#15-stage-3--index-mapping)
   - 1.6 [Stage 4 — GRC (Golomb-Rice) entropy coding](#16-stage-4--grc-golomb-rice-entropy-coding)
   - 1.7 [Stage 5 — Slice bitstream layout](#17-stage-5--slice-bitstream-layout)
   - 1.8 [Full concrete example: 8 weights → bitstream](#18-full-concrete-example-8-weights--bitstream)
2. [IFM / OFM Physical Memory Layout](#2-ifm--ofm-physical-memory-layout)
   - 2.1 [What is the 2×2 ublock? H×W confirmed](#21-what-is-the-22-ublock-hw-confirmed)
   - 2.2 [NHWC layout — stride formulas + concrete example](#22-nhwc-layout--stride-formulas--concrete-example)
   - 2.3 [NHCWB16 layout — stride formulas + concrete example](#23-nhcwb16-layout--stride-formulas--concrete-example)
   - 2.4 [Side-by-side comparison: NHWC vs NHCWB16](#24-side-by-side-comparison-nhwc-vs-nhcwb16)
   - 2.5 [How strides are pushed to NPU registers](#25-how-strides-are-pushed-to-npu-registers)
3. [Optimization Opportunities](#3-optimization-opportunities)

---

## 1. MLW Weight Codec — Step-by-Step

### 1.1 Overview of the 5-stage pipeline

```
int8 weights (TFLite)
       │
       ▼ (Python)  weight_compressor.py  L362-365
Stage 0: Zero-point subtraction  →  int16 weights
       │
       ▼ (C)  mlw_encode.c  reorder()  L969
Stage 1: Reorder into NPU OHWI traversal order
       │
       ▼ (C)  mlw_encode.c  find_palette() → create_palette()  L328 / L222
Stage 2: Palette construction (up to 32 entries, frequency-sorted)
       │
       ▼ (C)  mlw_encode.c  create_inverse_palette()  L368
Stage 3: Index mapping  weight → palette_index  (or direct sign-mag index)
       │
       ▼ (C)  mlw_encode.c  search_grc_params()  L403
Stage 4: Viterbi search for optimal GRC divisor per slice
       │
       ▼ (C)  mlw_encode.c  encode_slice()  L560
Stage 5: Write packed bitstream (slice header + interleaved GRC chunks)
       │
       ▼
Compressed weight blob (bytes) — DMA'd to SRAM, NPU decompresses on-the-fly
```

> **Source**: [ethosu/vela/weight_compressor.py](ethos-u-vela/ethosu/vela/weight_compressor.py#L321) — top-level Python orchestration  
> **Source**: [ethosu/mlw_codec/mlw_encode.c](ethos-u-vela/ethosu/mlw_codec/mlw_encode.c#L1121) — public entry `mlw_reorder_encode()`

---

### 1.2 Stage 0 — Zero-point subtraction

Before any C code runs, Python corrects the quantization zero-point:

```python
# weight_compressor.py  L362-365
quant_buf  = weight_tens.values.astype(np.int16)   # int8 → int16 (widen)
zero_point = weight_tens.quantization.zero_point    # scalar or array, int16
weights    = quant_buf - zero_point                 # signed int16 symmetric
```

**Why?**  The NPU accumulator assumes weights are zero-centered (symmetric quantization).
The TFLite weight zero-point is subtracted once here rather than at every MAC.

**Example** (int8, zero_point = 2):

| Original int8 | After subtraction (int16) |
|:---:|:---:|
| `0x05` = 5 | 5 - 2 = **3** |
| `0x01` = 1 | 1 - 2 = **-1** |
| `0x04` = 4 | 4 - 2 = **2** |
| `0x02` = 2 | 2 - 2 = **0** |
| `0x00` = 0 | 0 - 2 = **-2** |
| `0x03` = 3 | 3 - 2 = **1** |

> **Source**: [ethosu/vela/weight_compressor.py](ethos-u-vela/ethosu/vela/weight_compressor.py#L362)

---

### 1.3 Stage 1 — Weight reorder (OHWI NPU traversal)

The weights in the TFLite flatbuffer are stored in **OHWI** order
`[OFM_depth][Kernel_H][Kernel_W][IFM_depth]`.
The NPU needs them in a specific nested micro-block order for its DPU pipeline.

#### Traversal loop order (innermost = fastest-changing)

```c
// mlw_encode.c  reorder()  L1003-1085  — DEPTH_FIRST mode (most convolutions)

for ofm_block_z in [0 .. ofm_depth)  step ofm_block_depth        // OFM macro-blocks
  for ifm_block_z in [0 .. ifm_depth)  step ifm_block_depth       // IFM macro-blocks (32 for int8)
    for subkernel_y in [0 .. kernel_H)  step decomp_h              // kernel H tile (max 8)
      for subkernel_x in [0 .. kernel_W)  step decomp_w            // kernel W tile (max 8)
        // ifm_ublk_outer = 0 only (DEPTH_FIRST)
        for ofm_ublk in [0 .. ofm_block_depth)  step 8             // OFM micro-blocks (ublock=8)
          for element in [0 .. sub_h*sub_w)                        // kernel spatial, row-major
            for ifm_ublk_inner in [0 .. ifm_block_depth)  step 8  // IFM micro-blocks (ublock=8)
              for ofm_ublock_z in [0 .. 8)                         // 8 OFM channels (one ublock)
                for ifm_ublock_z in [0 .. 8)                       // 8 IFM channels (one ublock)
                  emit weights[ofm_z, ky, kx, ifm_z]              // one int16 value
```

Key sizes:
- `ofm_ublock_depth = 8` (from `architecture_features.py` L222)
- `ifm_ublock_depth = 8`
- `ifm_block_depth  = 32` for int8, `16` for int16 (set at `mlw_encode.c` L1004)
- `decomp_h / decomp_w` ≤ 8 (sub-kernel splitting, `SubKernelMax = Block(8,8,…)` at `architecture_features.py` L243)

#### Concrete example — 1×1 kernel, IFM=8, OFM=16 (2 OFM ublocks)

Weights in TFLite order: `W[ofm][ky=0][kx=0][ifm]`

After reorder, the byte stream is:

```
Position  Content
──────────────────────────────────────────────────────────────────────
0.. 63  OFM_block=0 / IFM_block=0 / subkernel=(0,0) / OFM_ublk=0
          → element=0 / IFM_ublk=0
            ofm_ublock_z=0, ifm_ublock_z=0..7 : W[0,0,0,0..7]
            ofm_ublock_z=1, ifm_ublock_z=0..7 : W[1,0,0,0..7]
            ofm_ublock_z=2, ifm_ublock_z=0..7 : W[2,0,0,0..7]
            ...
            ofm_ublock_z=7, ifm_ublock_z=0..7 : W[7,0,0,0..7]
                                                  = 8×8 = 64 int16 values
64..127 OFM_block=0 / IFM_block=0 / subkernel=(0,0) / OFM_ublk=1
          → W[8..15, 0, 0, 0..7]   (second OFM micro-block)
128..191  OFM_block=8 (if ofm_depth>8) ...
```

For a **3×3 kernel**, IFM=8, OFM=8 (single ublock):

```
Subkernel (0,0) covers elements (ky,kx) = (0,0),(0,1),(0,2),(1,0),...,(2,2)
element=0: ky=0, kx=0 → emit W[0..7, 0, 0, 0..7]  (64 values)
element=1: ky=0, kx=1 → emit W[0..7, 0, 1, 0..7]  (64 values)
...
element=8: ky=2, kx=2 → emit W[0..7, 2, 2, 0..7]  (64 values)
Total: 9 elements × 64 values = 576 int16 values before encoding
```

> **Source**: [ethosu/mlw_codec/mlw_encode.c](ethos-u-vela/ethosu/mlw_codec/mlw_encode.c#L969) — `reorder()` function

---

### 1.4 Stage 2 — Palette construction

After reorder, the codec decides on a compression palette for each "section" of the weight stream.

#### Sign-magnitude encoding

MLW uses an internal **sign-magnitude** representation for palette values:

```c
// mlw_encode.c  create_palette()  L251
int sign   = i < 0;
int mag    = abs(i);
int palval = (mag << 1) | sign;
```

| Weight | mag | sign | palval (hex) |
|:---:|:---:|:---:|:---:|
| 0 | 0 | 0 | 0x00 |
| +1 | 1 | 0 | 0x02 |
| -1 | 1 | 1 | 0x03 |
| +2 | 2 | 0 | 0x04 |
| -2 | 2 | 1 | 0x05 |
| +3 | 3 | 0 | 0x06 |
| -3 | 3 | 1 | 0x07 |

#### Zero-run mode decision

```c
// mlw_encode.c  find_palette()  L328 — ZERO_RUN_THRES = 4  (L34)
int use_zero_runs = (most_common_val[0] == 0)
                 && (most_common_freq[0] > 4 * most_common_freq[1]);
```

If zero appears **> 4×** more often than the second most common value → **alternating mode**
(zeros coded separately as run-lengths, non-zeros coded normally).
This exploits sparsity in ReLU-activated weight tensors.

#### Palette selection heuristic

```c
// mlw_encode.c  create_palette()  L302-307
// Use palette only if > half of all weights are in it
if (pal_cnt > all_cnt / 2)
    p->palsize = i;   // up to 32 entries
else
    p->palsize = 0;   // no palette, use direct sign-mag encoding
```

Max 32 palette entries, sorted by **descending frequency**. Most frequent weight → index 0.

#### Palette bits (palbits)

```c
// mlw_encode.c  L315-319
int palbits = 2;
while ((1 << palbits) <= palbits_val)
    palbits++;
// palbits in [2..9]
```

Number of bits needed to store each palette entry value (the *palval*, not the index).
Minimum 2 bits.

> **Source**: [ethosu/mlw_codec/mlw_encode.c](ethos-u-vela/ethosu/mlw_codec/mlw_encode.c#L222) — `create_palette()`  
> **Source**: [ethosu/mlw_codec/mlw_encode.c](ethos-u-vela/ethosu/mlw_codec/mlw_encode.c#L328) — `find_palette()`

---

### 1.5 Stage 3 — Index mapping

The inverse palette maps each weight to its compressed index:

```c
// mlw_encode.c  create_inverse_palette()  L368
// If weight is in the palette → index = palette position (0..palsize-1)
// Otherwise → index = palval + palsize - direct_offset   (direct mode)
```

`direct_offset` = number of unused palvals near zero. Subtracting it shifts the direct range
down, keeping all indices < 512.

---

### 1.6 Stage 4 — GRC (Golomb-Rice) entropy coding

A Viterbi-like dynamic-programming search picks the best GRC divisor `WDIV` for each
"slice" of the index stream.

```c
// mlw_encode.c  L398-400  — available GRC parameter configs
// Format: (trunc<<4) | div
static const uint8_t w_grc_params[] = {
    0x00, 0x01, 0x02, 0x03, 0x04, 0x05,  // div=0..5, no truncation
    0x10, 0x11, 0x12, 0x13, 0x14, 0x15,  // div=0..5, truncated unary
    0x20                                   // uncompressed (WDIV_UNCOMPRESSED=7)
};
static const uint8_t z_grc_params[] = { 0x00, 0x01, 0x02, 0x03, 0x04 };
```

#### GRC encoding of a single index value `v` with divisor `WDIV=d`

```
quotient  q = v >> d          (i.e., v / 2^d)
remainder r = v & ((1<<d)-1)  (i.e., v mod 2^d)

Bits written: [unary(q)] [binary(r, d bits)]

Unary(q) is a modified 2-bit-per-symbol unary for the weight stream
(see WUNARY0 / WUNARY1 interleaving in encode_slice()).
```

#### Example: WDIV=1

| Index | q=v>>1 | r=v&1 | Unary bits | Remainder bits | Total bits |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 0 | 0 | 0 | `0` (1 bit) | `0` | 2 |
| 1 | 0 | 1 | `0` | `1` | 2 |
| 2 | 1 | 0 | `10` (2 bits) | `0` | 3 |
| 3 | 1 | 1 | `10` | `1` | 3 |
| 4 | 2 | 0 | `110` | `0` | 4 |
| 5 | 2 | 1 | `110` | `1` | 4 |

Compared to raw binary with palbits=3 (3 bits/value), WDIV=1 saves bits for small indices
(0,1 → 2 bits each) at the cost of extra bits for large indices (4,5 → 4 bits each).
The Viterbi picks the divisor that minimizes total bits across the whole slice.

> **Source**: [ethosu/mlw_codec/mlw_encode.c](ethos-u-vela/ethosu/mlw_codec/mlw_encode.c#L403) — `search_grc_params()`  
> **Source**: [ethosu/mlw_codec/mlw_common.h](ethos-u-vela/ethosu/mlw_codec/mlw_common.h#L25) — `ZDIV_DISABLE=6`, `WDIV_UNCOMPRESSED=7`

---

### 1.7 Stage 5 — Slice bitstream layout

Each "slice" is a header + data chunk written by `encode_slice()`:

```
┌──────────────────────────────────────────────────────────┐
│                    SLICE HEADER                          │
│  ZDIV      [3 bits]  zero-run GRC divisor                │
│                      (6 = ZDIV_DISABLE, no zero runs)    │
│  SLICELEN  [15 bits] number of weight values − 1        │
│  WDIV      [3 bits]  weight GRC divisor                  │
│                      (7 = WDIV_UNCOMPRESSED)             │
│  WTRUNC    [1 bit]   truncated unary flag                │
│  NEWPAL    [1 bit]   new palette follows                  │
│  ──── if NEWPAL=1 ───────────────────────────────────── │
│  DIROFS    [5 bits]  direct_offset                       │
│  PALSIZE   [5 bits]  palette size − 1  (0..31)           │
│  PALBITS   [3 bits]  palbits − 2       (0..7 → 2..9 bits)│
│  PALETTE   [palbits × palsize bits]  palette entries     │
│            each entry = palval in sign-magnitude form    │
└──────────────────────────────────────────────────────────┘
┌──────────────────────────────────────────────────────────┐
│                    SLICE DATA (interleaved chunks)       │
│  Per chunk (12 weight symbols at a time):                │
│    WUNARY0  [12 bits] — low unary bits for 12 weights    │
│    ZUNARY   [8 or 12 bits] — zero-run unary              │
│    WUNARY1  [variable] — high unary bits                 │
│    WREMAIN  [WDIV bits × prev_nsymbols] — remainders     │
│    ZREMAIN  [ZDIV bits × prev_z_nsymbols]                │
│  Chunks repeat until all nvalues are coded               │
└──────────────────────────────────────────────────────────┘
```

Header fixed overhead (no palette): **3+15+3+1+1 = 23 bits**  
Header with palette (palsize=6, palbits=3): **23+5+5+3+6×3 = 54 bits**

> **Source**: [ethosu/mlw_codec/mlw_encode.c](ethos-u-vela/ethosu/mlw_codec/mlw_encode.c#L560) — `encode_slice()`  
> **Source**: [ethosu/mlw_codec/mlw_encode.c](ethos-u-vela/ethosu/mlw_codec/mlw_encode.c#L602-620) — header `bitbuf_put` calls

---

### 1.8 Full concrete example: 8 weights → bitstream

**Setup**: tiny 1×1 conv, IFM=8, OFM=8. Original int8 weights with zero_point=2:

```
Original int8:  [5, 3, 4, 2, 0, 4, 2, 5]
ZP corrected:   [3,-1, 2, 0,-2, 2, 0, 3]  (int16)
```

#### Stage 1: Reorder

For 1×1 kernel, IFM=8, OFM=8 — single ublock, single kernel element:

```
Reordered stream (ofm_ublock_z=0..7, ifm_ublock_z=0..7):
  Only one OFM ublock, one IFM ublock, element=0 →
  weights[0..7] in order = [3, -1, 2, 0, -2, 2, 0, 3]
  (already in order for this trivial case)
```

#### Stage 2: Sign-magnitude + palette

Frequencies:
- 0 → 2×, +3 → 2×, +2 → 2×, -1 → 1×, -2 → 1×

Sign-mag palvals:
- 0 → palval=0 (freq=2), +3 → palval=6 (freq=2), +2 → palval=4 (freq=2),
  -1 → palval=3 (freq=1), -2 → palval=5 (freq=1)

Zero-run check: freq(0)=2, second best=2. `2 > 4×2`? No → **no zero-run mode**.

Sorted descending by `(freq<<16)|palval`:
```
(2<<16)|6 = 131078  → palval=6 → weight +3  →  p->lut[0]=6
(2<<16)|4 = 131076  → palval=4 → weight +2  →  p->lut[1]=4
(2<<16)|0 = 131072  → palval=0 → weight  0  →  p->lut[2]=0
(1<<16)|5 =  65541  → palval=5 → weight -2  →  p->lut[3]=5
(1<<16)|3 =  65539  → palval=3 → weight -1  →  p->lut[4]=3
```

`palsize=5`, `palbits=3` (since `2^3=8 > max palval=6`). All 8 weights fit in palette.

`direct_offset`: count unused palvals from 0: palval=1 is unused, palval=2 unused → direct_offset=2.

Palette LUT:
```
Index 0 → palval 6 → weight +3
Index 1 → palval 4 → weight +2
Index 2 → palval 0 → weight  0
Index 3 → palval 5 → weight -2
Index 4 → palval 3 → weight -1
```

#### Stage 3: Index mapping

```
Weight:  [ +3,  -1,  +2,   0,  -2,  +2,   0,  +3 ]
Index:   [  0,   4,   1,   2,   3,   1,   2,   0 ]
```

#### Stage 4: GRC encoding (WDIV=1, chosen by Viterbi)

`uncompressed_bits = ceil(log2(5)) = 3` (palette-only mode)

With WDIV=1, costs vs uncompressed (3 bits/sym):

| Weight | Index | q | r | Bits(WDIV=1) | Bits(raw) |
|:---:|:---:|:---:|:---:|:---:|:---:|
| +3 | 0 | 0 | 0 | 2 | 3 |
| -1 | 4 | 2 | 0 | 4 | 3 |
| +2 | 1 | 0 | 1 | 2 | 3 |
| 0  | 2 | 1 | 0 | 3 | 3 |
| -2 | 3 | 1 | 1 | 3 | 3 |
| +2 | 1 | 0 | 1 | 2 | 3 |
| 0  | 2 | 1 | 0 | 3 | 3 |
| +3 | 0 | 0 | 0 | 2 | 3 |
| **Total** | | | | **21 bits** | **24 bits** |

WDIV=1 saves 3 bits over raw. The Viterbi would confirm this or pick a better divisor.

#### Stage 5: Slice header bits

```
ZDIV     = 6 (ZDIV_DISABLE)   → 3 bits  = 0b110
SLICELEN = 7 (8-1)             → 15 bits = 0b000000000000111
WDIV     = 1                   → 3 bits  = 0b001
WTRUNC   = 0                   → 1 bit   = 0b0
NEWPAL   = 1                   → 1 bit   = 0b1
DIROFS   = 2                   → 5 bits  = 0b00010
PALSIZE  = 4 (5-1)             → 5 bits  = 0b00100
PALBITS  = 1 (3-2)             → 3 bits  = 0b001
PALETTE[0]=6 (+3)              → 3 bits  = 0b110
PALETTE[1]=4 (+2)              → 3 bits  = 0b100
PALETTE[2]=0 ( 0)              → 3 bits  = 0b000
PALETTE[3]=5 (-2)              → 3 bits  = 0b101
PALETTE[4]=3 (-1)              → 3 bits  = 0b011
─────────────────────────────────────────────────────
Header total: 3+15+3+1+1+5+5+3+(3×5) = 51 bits ≈ 7 bytes
+ GRC data:   21 bits ≈ 3 bytes
Grand total:  ~72 bits = 9 bytes  (vs 8 bytes raw int8 = same size for 8 values)
```

> For real networks with hundreds of weights per slice, the palette amortizes over more values
> and compression ratios of 2–4× are typical.

---

## 2. IFM / OFM Physical Memory Layout

### 2.1 What is the 2×2 ublock? H×W confirmed

```python
# architecture_features.py  L218-221
# Block(w, h, d)  ← constructor signature
class Block:
    def __init__(self, w=0, h=0, d=0):  # L49
        self.width = w; self.height = h; self.depth = d

# U65-256 config  L222-226:
Accelerator.Ethos_U65_256: ArchitectureConfig(
    256, 1, Block(2, 2, 8), Block(2, 2, 8), ...
)          #  macs  cores  ofm_ublock  ifm_ublock
```

**`Block(2, 2, 8)` → `width=2, height=2, depth=8`**

| Dimension | Size | Meaning |
|:---:|:---:|:---|
| width | 2 | 2 columns (W axis) |
| height | 2 | 2 rows (H axis) |
| depth | 8 | 8 output channels per DPU cycle |

So the **2×2 in the ublock is 2 columns (W) × 2 rows (H)** — 4 spatial positions, each
across 8 channels simultaneously. One DPU tick computes `2W × 2H × 8C = 32` output values,
each requiring `ifm_depth` MACs.

> **Source**: [ethosu/vela/architecture_features.py](ethos-u-vela/ethosu/vela/architecture_features.py#L48) — `Block.__init__(w,h,d)`  
> **Source**: [ethosu/vela/architecture_features.py](ethos-u-vela/ethosu/vela/architecture_features.py#L222) — U65-256 ublock config

---

### 2.2 NHWC layout — stride formulas + concrete example

#### Stride formulas (int8, elem_size=1)

```python
# register_command_stream_util.py  get_strides()  L155-159
stride_c = elem_size                            # = 1
stride_x = fm.shape.depth * stride_c            # = depth
stride_y = fm.shape.width * stride_x            # = width × depth
```

```python
# npu_performance.py  _strides_for_shape()  L199-203
strides[3] = elem_size                          # +Z (channel)
strides[2] = depth * elem_size                  # +X (column)
strides[1] = depth * width * elem_size          # +Y (row)
strides[0] = depth * width * height * elem_size # +N (batch)
```

Address of element `[h, w, c]`:
```
addr = BASE + h × stride_y + w × stride_x + c × stride_c
     = BASE + h × (W×C) + w × C + c
```

#### Concrete example: 4×4×8 feature map, NHWC, int8

`stride_c=1, stride_x=8, stride_y=32`

```
Memory (1 byte = 1 cell):

Offset  Content
──────  ──────────────────────────────────────────
  0     [h=0, w=0, c=0]  ┐
  1     [h=0, w=0, c=1]  │  row 0, col 0,
  2     [h=0, w=0, c=2]  │  all 8 channels
  ...                     │  (8 bytes contiguous)
  7     [h=0, w=0, c=7]  ┘
  8     [h=0, w=1, c=0]  ┐
  ...                     │  row 0, col 1
 15     [h=0, w=1, c=7]  ┘
 16     [h=0, w=2, c=0]  ─  row 0, col 2
 24     [h=0, w=3, c=0]  ─  row 0, col 3
 32     [h=1, w=0, c=0]  ─  row 1, col 0
 40     [h=1, w=1, c=0]  ─  row 1, col 1
 ...
127     [h=3, w=3, c=7]     total = 4×4×8 = 128 bytes
```

DPU reads the **2×2×8 ublock** at `[h=0,w=0]`:
```
Position (h,w,c)   Offset    Bytes
(0,0,0..7)         0.. 7   ← contiguous ✓  (1 AXI burst = 16 B covers this + col1)
(0,1,0..7)         8..15   ← contiguous ✓
(1,0,0..7)        32..39   ← gap! (bytes 16..31 skipped)  requires new burst
(1,1,0..7)        40..47   ← contiguous with (1,0) ✓ (covered by same burst if burst_len≥16)
```

With AXI burst_len=16 bytes: 2 bursts needed (one per row). With 8 channels the two columns
within a row fit in one 16-byte burst — efficient.

NPU registers set by `generate_ifm()`:
```
NPU_SET_IFM_BASE0   = BASE         (absolute address)
NPU_SET_IFM_STRIDE_C = 1           (bytes per channel step)
NPU_SET_IFM_STRIDE_X = 8           (bytes per column step)
NPU_SET_IFM_STRIDE_Y = 32          (bytes per row step)
NPU_SET_IFM_DEPTH_M1 = 7           (depth − 1)
```

> **Source**: [ethosu/vela/register_command_stream_util.py](ethos-u-vela/ethosu/vela/register_command_stream_util.py#L152) — `get_strides()` NHWC branch  
> **Source**: [ethosu/vela/register_command_stream_generator.py](ethos-u-vela/ethosu/vela/register_command_stream_generator.py#L443) — `generate_ifm()`

---

### 2.3 NHCWB16 layout — stride formulas + concrete example

NHCWB16 stands for **N H (C/16) W B16** — channels are grouped in blocks of 16 and the
16-channel block (`B16`) is the *innermost* dimension inside each (H, C_block, W) position.

#### Stride formulas (int8, elem_size=1)

```python
# register_command_stream_util.py  get_strides()  L161-165
stride_x = 16 * elem_size                               # = 16  (fixed width of one B16 block)
stride_c = stride_x * fm.shape.width                    # = 16 × W  (one C_block row)
stride_y = elem_size * fm.shape.width * round_up(depth,16)  # = W × ceil(C/16)×16
```

Address of element `[h, w, c]`:
```
addr = BASE + h × stride_y
             + (c // 16) × stride_c
             + w × stride_x          ← always 16 bytes, regardless of actual depth
             + (c %  16) × elem_size
```

> **Source**: [ethosu/vela/register_command_stream_util.py](ethos-u-vela/ethosu/vela/register_command_stream_util.py#L161) — `get_strides()` NHCWB16 branch  
> **Source**: [ethosu/vela/register_command_stream_util.py](ethos-u-vela/ethosu/vela/register_command_stream_util.py#L174) — `get_address()` formula

#### Concrete example: 4×4×8 feature map, NHCWB16, int8

`ceil(8/16)=1` channel block. `stride_x=16, stride_c=64, stride_y=64`

```
Memory (1 byte = 1 cell):

Offset  Content
──────  ──────────────────────────────────────────────────────────
  0     [h=0, C_block=0, w=0, b16=0..7]   = channels 0..7  ┐
  8     [h=0, C_block=0, w=0, b16=8..15]  = PADDING         │  h=0, w=0
                                                             │  16 bytes total
 15     end                                                  ┘
 16     [h=0, C_block=0, w=1, b16=0..15]  = h=0, w=1, all 16 B16 slots (8 valid + 8 pad)
 32     [h=0, C_block=0, w=2, ...]
 48     [h=0, C_block=0, w=3, ...]
 64     [h=1, C_block=0, w=0, ...]        = h=1, w=0
 80     [h=1, C_block=0, w=1, ...]
...
252     [h=3, C_block=0, w=3, b16=0..15]   total = 4×1×4×16 = 256 bytes (padded, vs 128 for NHWC)
```

DPU reads the **2×2×8 ublock** at `[h=0,w=0]`:
```
Position (h,w,c)   Offset    Burst
(0,0,c=0..7)       0.. 7   ─ part of 16-byte block at offset 0..15  ✓ (16-byte aligned burst)
(0,1,c=0..7)      16..23   ─ part of 16-byte block at offset 16..31 ✓
(1,0,c=0..7)      64..71   ─ part of 16-byte block at offset 64..79 ✓
(1,1,c=0..7)      80..87   ─ part of 16-byte block at offset 80..95 ✓
```

Every access starts at a 16-byte-aligned address → **always one clean AXI burst per (h,w)
position**, regardless of actual channel count. 4 bursts for the 2×2 spatial ublock.

NPU registers set by `generate_ifm()`:
```
NPU_SET_IFM_BASE0    = BASE
NPU_SET_IFM_STRIDE_C = 64   (= 16 × W = 16 × 4)
NPU_SET_IFM_STRIDE_X = 16   (= 16 bytes, fixed)
NPU_SET_IFM_STRIDE_Y = 64   (= W × round_up(C,16) = 4 × 16)
NPU_SET_IFM_DEPTH_M1 = 7
```

---

### 2.4 Side-by-side comparison: NHWC vs NHCWB16

#### Example: 4×4×32 feature map, int8

**NHWC**: `stride_c=1, stride_x=32, stride_y=128`

```
Row 0: [w=0,c=0..31] [w=1,c=0..31] [w=2,c=0..31] [w=3,c=0..31]
       ─────32────── ─────32────── ─────32────── ─────32──────
       Offset 0..31   32..63        64..95        96..127
Row 1: offset 128..255
```

**NHCWB16**: `stride_x=16, stride_c=16×4=64, stride_y=4×32=128`

```
Row 0, C_block=0 (c=0..15):  [w=0,b16=0..15] [w=1] [w=2] [w=3]
                               ────16────      ──16─  ──16─  ──16─
                               Offset 0..15    16..31 32..47 48..63
Row 0, C_block=1 (c=16..31): offset 64..127
Row 1, C_block=0:            offset 128..191
Row 1, C_block=1:            offset 192..255
```

| Feature | NHWC (C=32) | NHCWB16 (C=32) |
|:---|:---:|:---:|
| Stride_X (per column step) | 32 bytes | **16 bytes** (fixed) |
| 16-byte alignment of each (h,w) access? | ✗ (stride_x=32 → misaligned c=16 group) | ✓ (stride_x always 16) |
| Bytes to store 4×4×32 | 512 | **512** (no waste when C%16=0) |
| Bytes to store 4×4×8 | 128 | **256** (8 channels padded to 16) |
| AXI bursts for 2×2×16 ublock | 4 (contiguous c=0..15 per (h,w)) | 4 (same, each 16-byte aligned) |
| AXI bursts for 2×2×8 ublock | 4 (each 8-byte sub-burst) | 4 (each 16-byte burst, 8 valid) |

**When depth < 16**: NHCWB16 wastes memory (padding to 16) but keeps burst addresses aligned.  
**When depth ≥ 16 and depth%16 = 0**: NHCWB16 = no waste, always 16-byte aligned bursts.  
**When depth%32 = 0**: NHWC is equally efficient (stride_x is a multiple of 16).

> **Source**: [ethosu/vela/npu_performance.py](ethos-u-vela/ethosu/vela/npu_performance.py#L197) — `_strides_for_shape()`  
> **Source**: [ethosu/vela/npu_performance.py](ethos-u-vela/ethosu/vela/npu_performance.py#L213) — burst efficiency calculation

---

### 2.5 How strides are pushed to NPU registers

The command stream generator writes addresses and strides into NPU CMD1 registers:

```python
# register_command_stream_generator.py  generate_ifm()  L443
emit.cmd0_with_param(cmd0.NPU_SET_IFM_REGION, ifm.region)
# BASE addresses (up to 4 tiles)
generate_addresses(emit, [NPU_SET_IFM_BASE0..BASE3], ifm.tiles.addresses, ...)
# Dimensions
emit.cmd0_with_param(cmd0.NPU_SET_IFM_DEPTH_M1, ifm.shape.depth - 1)
# Strides — computed by get_strides()
generate_strides(emit, ifm, NPU_SET_IFM_STRIDE_C, NPU_SET_IFM_STRIDE_Y, NPU_SET_IFM_STRIDE_X)
emit.cmd0_with_param(cmd0.NPU_SET_IFM_ZERO_POINT, ...)
```

All stride values must be **16-byte aligned**:

```python
# register_command_stream_util.py  check_alignment()  L59
def check_alignment(payload, required_alignment):
    if payload % required_alignment != 0:
        raise ByteAlignmentError(...)
```

> **Source**: [ethosu/vela/register_command_stream_generator.py](ethos-u-vela/ethosu/vela/register_command_stream_generator.py#L443) — `generate_ifm()`  
> **Source**: [ethosu/vela/register_command_stream_generator.py](ethos-u-vela/ethosu/vela/register_command_stream_generator.py#L481) — `generate_ofm()`  
> **Source**: [ethosu/vela/register_command_stream_util.py](ethos-u-vela/ethosu/vela/register_command_stream_util.py#L59) — 16-byte alignment check

---

## 3. Optimization Opportunities

### 3.1 Weight codec optimizations

| Strategy | Effect | How |
|:---|:---|:---|
| **Center weights around zero** | More zero-runs, smaller indices → fewer GRC bits | Ensure quantization zero_point is accurate; retrain with symmetric quantization |
| **Use ReLU / sparsity** | Triggers zero-run (alternating) mode when `freq(0) > 4× second_most` | Post-activation weight pruning or sparse training |
| **Keep unique values < 32** | All weights fit in palette → uses direct GRC on small indices | Quantize to fewer bits (e.g., 4-bit with 16 palette entries), or cluster weights |
| **OFM depth multiple of 8** | No padding zeros in weight reorder stream | Pad OFM channels to next multiple of 8 at model design time |
| **IFM depth multiple of 32** (int8) | No partial IFM blocks; no padding zeros | Pad or group IFM channels; avoids `clipped_ifm_block_depth < 32` |
| **Kernel dimensions ≤ 8** | Single sub-kernel; no decomposition overhead | Standard 3×3, 5×5 kernels; avoid 9×9+ or use depthwise decomposition |

> **Source**: [ethosu/mlw_codec/mlw_encode.c](ethos-u-vela/ethosu/mlw_codec/mlw_encode.c#L34) — `ZERO_RUN_THRES=4`  
> **Source**: [ethosu/vela/weight_compressor.py](ethos-u-vela/ethosu/vela/weight_compressor.py#L383) — block traversal selection  
> **Source**: [ethosu/mlw_codec/mlw_encode.c](ethos-u-vela/ethosu/mlw_codec/mlw_encode.c#L1004) — `ifm_block_depth` (32 for int8)

### 3.2 IFM / OFM layout optimizations

| Strategy | Effect | How |
|:---|:---|:---|
| **Use NHCWB16 for feature maps** | Each (h,w) access is exactly 16 bytes, always 16-byte aligned | Select `NpuLayout.NHCWB16` in the API or let vela choose; set in `generate_ifm()` |
| **Channel count multiple of 16** | No NHCWB16 padding waste; perfect burst utilization | Design layers with C = 16, 32, 48, … |
| **Channel count multiple of 32** | Also satisfies NHWC alignment (stride_x = 32-byte aligned) | For models that must use NHWC |
| **Height multiple of 2** | No partial OFM ublock rows (ublock H=2); no padding cycles | Pad spatial dims to even numbers |
| **Width multiple of 2** | Same, for ublock W=2 | Pad spatial dims to even numbers |
| **Keep feature maps in SRAM** | Avoid DRAM latency (DRAM latency ≫ SRAM); SRAM bandwidth = 16 B/cycle | Size model to fit activations in SRAM (48 KB SHRAM + system SRAM) |
| **Align base addresses to 16 bytes** | Hardware requires 16-byte alignment on all BASE/STRIDE registers | Use 16-byte aligned allocators; vela enforces this via `check_alignment()` |

> **Source**: [ethosu/vela/architecture_features.py](ethos-u-vela/ethosu/vela/architecture_features.py#L293) — AXI width=128 bits=16 bytes/cycle  
> **Source**: [ethosu/vela/architecture_features.py](ethos-u-vela/ethosu/vela/architecture_features.py#L496) — `calc_ifm_block_depth()` (32 int8, 16 int16)  
> **Source**: [ethosu/vela/register_command_stream_util.py](ethos-u-vela/ethosu/vela/register_command_stream_util.py#L59) — alignment enforcement

### 3.3 Block traversal: DEPTH_FIRST vs PART_KERNEL_FIRST

```python
# weight_compressor.py  L383-391
depth_utilization     = ifm_depth / round_up(ifm_depth, 32)
part_kernel_util      = (ifm_depth / round_up(ifm_depth, 8))
                      * (kernel_size / round_up(kernel_size, 4))
if part_kernel_util >= depth_utilization or ifm_depth <= 8:
    traversal = PART_KERNEL_FIRST
```

- **DEPTH_FIRST** (default): loops IFM-channels faster; best when IFM depth is a multiple of 32.
- **PART_KERNEL_FIRST**: loops kernel elements faster; better for small IFM depths (≤8) or
  kernels where `kernel_H × kernel_W` utilization improves more than IFM depth utilization.

Vela chooses automatically, but you can force better hardware utilization by ensuring
`ifm_depth % 32 == 0` (for int8) to always prefer depth-first without DPU under-utilization.

> **Source**: [ethosu/vela/weight_compressor.py](ethos-u-vela/ethosu/vela/weight_compressor.py#L383) — traversal selection logic  
> **Source**: [ethosu/mlw_codec/mlw_encode.c](ethos-u-vela/ethosu/mlw_codec/mlw_encode.c#L1004) — `ifm_block_depth` definition in `reorder()`
