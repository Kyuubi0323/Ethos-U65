# Ethos-U65 Architecture & Weight/Bias Encoding Notes

---

## 1. Hardware Configurations

> **Source**: [`ethosu/vela/architecture_features.py`](ethosu/vela/architecture_features.py#L222-L230) — `ArchitectureFeatures.accelerator_configs` dict

| Variant | MACs/cycle | Cores | OFM ublock (H×W×D) | IFM ublock | SHRAM banks | AXI bus width | Address space |
|---|---|---|---|---|---|---|---|
| Ethos-U65-256 | 256 | 1 | 2×2×8 | 2×2×8 | 48 | 128-bit (16 B/cycle) | 40-bit (1 TB) |
| Ethos-U65-512 | 512 (2×256) | 2 | 2×2×8 | 2×2×8 | 48 | 128-bit (16 B/cycle) | 40-bit (1 TB) |

- **SHRAM total size**: 48 banks × 1024 bytes = **48 KB** — [`architecture_features.py:435`](ethosu/vela/architecture_features.py#L435) `self.shram_bank_size = 1024`
- **Max outstanding DMA transactions**: 2 (vs 1 for U55) — [`architecture_features.py:294`](ethosu/vela/architecture_features.py#L294) `self.max_outstanding_dma = 2`
- **Max sub-kernel size**: 8×8 — [`architecture_features.py:243`](ethosu/vela/architecture_features.py#L243) `SubKernelMax = Block(8, 8, 65536)`
- **Max outstanding kernel blocks (double buffering)**: 2 — [`architecture_features.py:363`](ethosu/vela/architecture_features.py#L363) `self.max_outstanding_kernels = 2`

---

## 2. What the DPU Computes Per Clock Cycle

### MACs per cycle breakdown

> **Source**: [`ethosu/vela/architecture_features.py`](ethosu/vela/architecture_features.py#L370-L376) — `ArchitectureFeatures.__init__()`, DPU sizing variables

The U65 DPU is organized around a **micro-block (ublock)** of `2 × 2 × 8`:

```
OFM ublock = 2 (H) × 2 (W) × 8 (depth channels)
```

Each clock cycle the DPU processes:
- 2 × 2 = 4 output spatial positions (H×W)
- 8 output channels (OFM depth)
- 8 weight multiply-accumulates per OFM channel (dot product width = 8)

Total: `2 × 2 × 8 × 8 = 256 MACs/cycle` for U65-256.
U65-512 doubles this with 2 cores: `2 × 256 = 512 MACs/cycle`.

```python
# architecture_features.py:370-376
dpu_min_height = accel_config.ofm_ublock.height          # 2
dpu_min_width = accel_config.ofm_ublock.width            # 2
dpu_dot_product_width = 8
dpu_min_ofm_channels = accel_config.ofm_ublock.depth     # 8
self.num_macs_per_cycle = dpu_min_height * dpu_min_width * dpu_dot_product_width * dpu_min_ofm_channels
# = 2 * 2 * 8 * 8 = 256
```

### DPU cycles for one OFM block (Conv2D, depth-first mode)

> **Source**: [`ethosu/vela/npu_performance.py`](ethosu/vela/npu_performance.py#L314-L435) — `_estimate_conv_cycles()`

```
cycles_wb = 32 × ofm_ublock_depth / 8 = 32 cycles   (write-back pipeline drain)
# npu_performance.py:350  cycles_wb = 32 * ofm_ublock.depth // 8

num_ublk_x  = ceil(OFM_block_W / 2)
num_ublk_y  = ceil(OFM_block_H / 2)
num_ublk_xy = num_ublk_x × num_ublk_y
num_ublk_z  = ceil(OFM_block_D / 8)

# per sub-kernel (the 8×8 tile):
cycles_per_subkernel = max(cycles_wb, 4 × num_ublk_xy) × kernel_H × kernel_W × num_ublk_z

# over all IFM depth blocks (int8 → depth block = 32):
cycles_dpu_block = cycles_per_subkernel × ceil(IFM_depth / 32)

# divide by number of cores for U65-512:
cycles_dpu_block /= ncores
# npu_performance.py:416  cycles_dpu_blk /= arch.ncores
```

### DPU cycles for Part-Kernel-First mode (small IFM depth ≤ 8)

> **Source**: [`ethosu/vela/npu_performance.py`](ethosu/vela/npu_performance.py#L385-L396) — `_estimate_conv_cycles()`, `is_partkernel` branch

IFM block depth is always 16 in this mode. Kernel steps are rounded up to 4 (int8) or 2 (int16):

```
num_kernel_steps = ceil(kernel_H × kernel_W / 4)    (int8)
cycles = max(cycles_wb, 4 × num_ublk_xy) × num_kernel_steps × ceil(IFM_depth/8) × num_ublk_z
```

### DPU cycles for Depthwise Conv

> **Source**: [`ethosu/vela/npu_performance.py`](ethosu/vela/npu_performance.py#L368-L377) — `_estimate_conv_cycles()`, `ConvolutionDepthWise` branch

```
cycles = 4 × num_ublk_xy  (×2 if int16)
num_kernel_steps = ceil(kernel_elements / 4)
cycles_dpu_block = max(cycles_wb, cycles) × num_kernel_steps × num_ublk_z
```

### Minimum cycles per OFM element — output stage

> **Source**: [`ethosu/vela/npu_performance.py`](ethosu/vela/npu_performance.py#L263-L311) — `_estimate_output_cycles_per_element()`
> **Source**: [`ethosu/vela/architecture_features.py`](ethosu/vela/architecture_features.py#L484-L492) — `_generate_output_perf_tables()`

After the DPU, the write-out / activation / bias-scale stage has its own throughput:

| Operation | U65-256 cycles/elem | U65-512 cycles/elem |
|---|---|---|
| Conv (int32 acc) | 0.625 | 0.3125 |
| Depthwise / Pooling (int40 acc) | 0.375 | 0.1875 |
| MUL (int32 output) | 0.500 | 0.250 |
| Add/Sub (advanced) | 0.750 | 0.375 |
| MaxPool | 0.125 | 0.0625 |
| AvgPool / other | 0.250 | 0.125 |
| Sigmoid/Tanh (LUT) activation overhead | 1.000 | 0.500 |
| ReLU activation overhead | 0.250 | 0.125 |

```python
# architecture_features.py:488-492
# U65-256 (same table as U55-256):
self.output_cycles_per_elem = (0.625, 1.125, 0.5, 0.375, 0.5, 0.75, 0.125, 0.25)
self.activation_cycles_per_elem = (1.0, 0.25, 0.0)
# U65-512:
self.output_cycles_per_elem = (0.3125, 0.5625, 0.25, 0.1875, 0.25, 0.375, 0.0625, 0.125)
self.activation_cycles_per_elem = (0.5, 0.125, 0.0)
```

Total time per OFM block = `max(DPU_cycles, output_stage_cycles)`.

---

## 3. Memory Bandwidth Per Cycle

> **Source**: [`ethosu/vela/architecture_features.py`](ethosu/vela/architecture_features.py#L293-L311) — AXI port widths and `memory_bandwidths_per_cycle`
> **Source**: [`ethosu/vela/npu_performance.py`](ethosu/vela/npu_performance.py#L227-L260) — `_estimate_minimum_memory_cycles()`

U65 AXI bus = **128 bits = 16 bytes per cycle** for all memory regions.

```python
# architecture_features.py:293-311
elif self.is_ethos_u65_system:
    self.max_outstanding_dma = 2
    axi_port_address_width = 40
    axi_port_data_width = [128 for i in range(MemArea.Size)]   # 128-bit = 16 bytes/cycle
self.memory_bandwidths_per_cycle = (
    np.array([a * b for a, b in zip(axi_port_data_width, self.memory_clock_scales)]) / 8
)
```

| Transfer type | Bytes moved per cycle |
|---|---|
| IFM read | 16 bytes/cycle |
| OFM write | 16 bytes/cycle |
| Weight stream fetch | 16 bytes/cycle |
| Scale/bias fetch | 16 bytes/cycle |
| DMA (max outstanding=2) | up to 32 bytes/cycle overlapped |

### IFM block bytes fetched per OFM block (Conv2D example)

> **Source**: [`ethosu/vela/architecture_features.py`](ethosu/vela/architecture_features.py#L496-L530) — `calc_ifm_block_depth()` and `get_ifm_block_size()`

```
IFM_block_H = ceil((OFM_block_H − 1) × stride_H + kernel_H)  [aligned to ifm_ublock=2]
IFM_block_W = ceil((OFM_block_W − 1) × stride_W + kernel_W)  [aligned to ifm_ublock=2]
IFM_block_D = 32 (int8) or 16 (int16)
# architecture_features.py:496-502
# max_block_depth = 8 * 32 // ifm_bits  → 32 for int8, 16 for int16

IFM_bytes_per_block = IFM_block_H × IFM_block_W × IFM_block_D × (bitdepth/8)
```

### Weight bytes per OFM block (compressed, worst case uncompressed)

> **Source**: [`ethosu/vela/weight_compressor.py`](ethosu/vela/weight_compressor.py#L321-L490) — `encode_weight_and_scale_tensor()`

```
Weight elements = kernel_H × kernel_W × IFM_depth × OFM_block_D
Weight bytes (uncompressed int8) = kernel_H × kernel_W × IFM_depth × OFM_block_D bytes

Typical MLW compression ratio: 2×–5× depending on weight distribution
After compression, padded to 16-byte boundary per sub-stream
```

### Bias/scale bytes

> **Source**: [`ethosu/vela/weight_compressor.py`](ethosu/vela/weight_compressor.py#L202-L232) — `encode_bias()`

```
10 bytes per output channel (80-bit packed: 40b bias + 32b scale + 6b shift + 2b reserved)
Total for one OFM block = OFM_block_D × 10 bytes, padded to 16 bytes
```

---

## 4. SHRAM Buffer Layout (48 KB)

> **Source**: [`ethosu/vela/architecture_features.py`](ethosu/vela/architecture_features.py#L435-L448) — SHRAM constants in `__init__()`
> **Source**: [`ethosu/vela/architecture_features.py`](ethosu/vela/architecture_features.py#L462-L479) — `generate_block_config()`

SHRAM is divided into:

| Section | Size | Description |
|---|---|---|
| Reserved output banks | 2 banks = 2 KB | Output pipeline buffering |
| Reserved unused banks | 2 banks = 2 KB (if banks>16) | Arm internal reserved |
| Available IFM + Acc banks | 44 banks = 44 KB | IFM buffer + accumulator (double-buffered) |
| LUT area | 2 KB (last 2 banks) | Activation lookup table (Sigmoid/Tanh) |

```python
# architecture_features.py:435-448
self.shram_bank_size = 1024                              # bytes per bank
self.shram_reserved_output_banks = 2
self.shram_reserved_unused_banks = 2 if accel_config.shram_banks > 16 else 0  # 2 for 48-bank U65
self.shram_total_banks = accel_config.shram_banks - self.shram_reserved_unused_banks
# = 48 - 2 = 46 total, minus 2 output = 44 KB available
self.shram_lut_size = 2048                               # 2 KB for activation LUT
```

The IFM buffer and accumulator share the 44 available KB, **double-buffered**:
- Half is used for the current block being computed
- Half is filled by DMA with the next block

#### SHRAM requirement for a block

> **Source**: [`ethosu/vela/architecture_features.py`](ethosu/vela/architecture_features.py#L462-L479) — `generate_block_config()`

```python
# architecture_features.py:462-479
def generate_block_config(self, width, height, depth):
    d1 = round_up(depth, SHRAMElements.PreAlign)         # PreAlign = 8
    d2 = round_up(d1 × SHRAMElements.ByteSizes, SHRAMElements.PostAlign)
    size_bytes = height * width * d2
    size_banks = round_up_divide(size_bytes, self.shram_bank_size)
    size_banks *= 2                                       # ×2 double buffering
    required_banks = round_up(size_banks, self.shram_bank_granules)
```

Granule per data type (U65-256/512):

> **Source**: [`ethosu/vela/architecture_features.py`](ethosu/vela/architecture_features.py#L222-L230) — `shram_granules` field in `ArchitectureConfig`

| Data | Granule (banks) |
|---|---|
| IFM int8 | 8 |
| IFM int16 | 8 |
| IFM int32 | 8 |
| Acc int16 | 16 |
| Acc int32 | 8 |
| Acc int40 | 16 |
| Elementwise int8 | 8 |
| Elementwise int16 | 20 |

```python
# architecture_features.py:222-226
# shram_granules = [IFM8, IFM16, IFM32, Acc16, Acc32, Acc40, EW_IFM8, EW_IFM16]
Accelerator.Ethos_U65_512: ArchitectureConfig(
    256, 2, Block(2, 2, 8), Block(2, 2, 8), 48, [8, 8, 8, 8, 16, 8, 16, 20], 8
),
```

---

## 5. Weight Encoding Pipeline (Step by Step)

### Step 1 — Zero-point correction

> **Source**: [`ethosu/vela/weight_compressor.py`](ethosu/vela/weight_compressor.py#L362-L365) — inside `encode_weight_and_scale_tensor()`

```python
# weight_compressor.py:362-365
quant_buf = weight_tens.values.astype(np.int16)
zero_point = weight_tens.quantization.zero_point.astype(np.int16)
weights = quant_buf - zero_point
# Result: signed weights centered at zero, range ≈ −255..+255
```

### Step 2 — Block traversal reordering

> **Source**: [`ethosu/mlw_codec/mlw_encode.c`](ethosu/mlw_codec/mlw_encode.c#L969-L1119) — `reorder()` static function
> **Source**: [`ethosu/mlw_codec/mlw_encode.c`](ethosu/mlw_codec/mlw_encode.c#L1121) — `mlw_reorder_encode()` public entry point
> **Source**: [`ethosu/vela/weight_compressor.py`](ethosu/vela/weight_compressor.py#L141-L196) — `encode_weights()` Python wrapper

Weights are physically reordered in memory to match NPU traversal order **before** compression.

**Traversal order (nested loops)**:
```
OFM blocks (depth slices of ofm_block_depth)
  IFM blocks (depth slices of 32 for int8, 16 for int16)
    Sub-kernel H (tiles of ≤8 rows)
      Sub-kernel W (tiles of ≤8 cols)
        OFM ublock (groups of 8 OFM channels)
          Kernel element (H×W position)
            IFM ublock (groups of 8 IFM channels)
              OFM ublock element (ofm_ublock_depth = 8)
                IFM ublock element (ifm_ublock_depth = 8)
```

**Traversal mode selection**:

> **Source**: [`ethosu/vela/weight_compressor.py`](ethosu/vela/weight_compressor.py#L383-L391) — `encode_weight_and_scale_tensor()`, block traversal selection

| Mode | When | IFM depth block |
|---|---|---|
| DEPTH_FIRST | Default, large IFM depth | 32 (int8) / 16 (int16) |
| PART_KERNEL_FIRST | ifm_depth ≤ 8, or better utilization | 16 always |
| DEPTHWISE | Depthwise convolution | ifm_ublock_depth = 8, only 1 IFM channel |

```python
# weight_compressor.py:383-391
depth_utilization = weights.shape[2] / round_up(weights.shape[2], 32 if ifm_bitdepth == 8 else 16)
part_kernel_utilization = (weights.shape[2] / round_up(weights.shape[2], 8)) * (
    kernel_size / round_up(kernel_size, 4 if ifm_bitdepth == 8 else 2)
)
if part_kernel_utilization >= depth_utilization or ifm_depth <= 8:
    npu_tensor.hw_traversal = NpuBlockTraversal.PART_KERNEL_FIRST
```

**For U65-512 with 2 cores**, weights are deinterleaved across cores before reordering:

> **Source**: [`ethosu/vela/weight_compressor.py`](ethosu/vela/weight_compressor.py#L234-L237) — `core_deinterleave()`

```python
# weight_compressor.py:234-237
def core_deinterleave(hwio, core, ncores):
    ohwi = np.transpose(hwio, (3, 0, 1, 2))
    return ohwi[core : ohwi.shape[0] : ncores]
# Core 0 → OFM channels 0, 2, 4, ...
# Core 1 → OFM channels 1, 3, 5, ...
```

### Step 3 — MLW compression (`mlw_encode()`)

> **Source**: [`ethosu/mlw_codec/mlw_encode.c`](ethosu/mlw_codec/mlw_encode.c#L872-L968) — `mlw_encode()`

The reordered weight stream is compressed with Arm's MLW codec:

#### a) Palette sections

> **Source**: [`ethosu/mlw_codec/mlw_encode.c`](ethosu/mlw_codec/mlw_encode.c#L80-L221) — `search_palette_sections()`
> **Source**: [`ethosu/mlw_codec/mlw_encode.c`](ethosu/mlw_codec/mlw_encode.c#L222-L327) — `create_palette()`
> **Source**: [`ethosu/mlw_codec/mlw_encode.c`](ethosu/mlw_codec/mlw_encode.c#L328-L398) — `find_palette()`

- Frequency histogram of all weight values (−255 to +255) computed
- If zero is >4× more frequent than the next value → **zero-run mode** activated (zeros coded as run-lengths separately)
- Top-32 most frequent non-zero values form the **palette** (up to 32 entries)
- The palette is re-created at "restart" boundaries for efficiency

```c
// mlw_encode.c:34 — threshold for activating zero-run mode
#define ZERO_RUN_THRES  4
// mlw_encode.c:328-330
int use_zero_runs = most_common_val[0]==0
                  && most_common_freq[0] > ZERO_RUN_THRES * most_common_freq[1];
```

#### b) Value mapping

- Palette members → palette index (0..31)
- Non-palette values → `palette_size + direct_sign_magnitude_index − direct_offset`

#### c) GRC entropy coding (Golomb-Rice-like)

> **Source**: [`ethosu/mlw_codec/mlw_encode.c`](ethosu/mlw_codec/mlw_encode.c#L403-L557) — `search_grc_params()`

- Weight indices are entropy-coded with adaptive GRC divisor (WDIV 0–5) and truncation flag
- A **Viterbi-like search** finds optimal GRC parameter transitions across sections
- Zero runs have their own GRC stream (ZDIV 0–3)
- Uncompressed fallback: raw fixed-width bits if smaller

```c
// mlw_encode.c:396-397 — available GRC parameter configs
// (trunc<<4) | div, 0x20 means uncompressed
static const uint8_t w_grc_params[] = { 0x00, 0x01, 0x02, 0x03, 0x04, 0x05,
                                         0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x20 };
static const uint8_t z_grc_params[] = { 0x00, 0x01, 0x02, 0x03, 0x04 };
```

#### d) Bitstream slice format

> **Source**: [`ethosu/mlw_codec/mlw_encode.c`](ethosu/mlw_codec/mlw_encode.c#L560-L727) — `encode_slice()`
> **Source**: [`ethosu/mlw_codec/mlw_encode.c`](ethosu/mlw_codec/mlw_encode.c#L728-L871) — `encode_section()`

```c
// encode_slice() — mlw_encode.c:608-625
bitbuf_put( bb, "ZDIV",     3,  zdiv);
bitbuf_put( bb, "SLICELEN", 15, nvalues-1 );
bitbuf_put( bb, "WDIV",     3,  wdiv);
bitbuf_put( bb, "WTRUNC",   1,  w_grc_trunc );
bitbuf_put( bb, "NEWPAL",   1,  new_palette );
// If NEWPAL:
bitbuf_put( bb, "DIROFS",   5,  p->direct_offset );
bitbuf_put( bb, "PALSIZE",  5,  max(0, p->palsize-1));
bitbuf_put( bb, "PALBITS",  3,  p->palbits-2 );
// palette_entries[palsize × palbits]
// Then interleaved: WUNARY0 | ZUNARY | WUNARY1 | WREMAIN | ZREMAIN ...
```

End of stream: `ZDIV=7 (EOS marker)` + byte-align + 0xFF pad to 128-bit boundary.

```c
// mlw_encode.c:915-920
bitbuf_put( bb, "ZDIV", 3, ZDIV_EOS);
bitbuf_put( bb, "BYTEALIGN", (8-(bb->pos&7))&7, 0xff );
while( bb->pos & 127 )
    bitbuf_put( bb, "PAD", 8, 0xff );
```

All weight sub-streams are padded to **16-byte (128-bit) alignment**.

### Step 4 — Bias/scale packing

> **Source**: [`ethosu/vela/weight_compressor.py`](ethosu/vela/weight_compressor.py#L202-L232) — `encode_bias()`
> **Source**: [`ethosu/vela/weight_compressor.py`](ethosu/vela/weight_compressor.py#L254-L320) — `_prepare_scale_and_bias()`
> **Source**: [`ethosu/vela/scaling.py`](ethosu/vela/scaling.py#L33-L57) — `quantise_scale()` and `reduced_quantise_scale()`

Each output channel gets a **10-byte (80-bit)** packed entry:

```
Bytes [0..4]  → 40-bit signed bias (little-endian)
Bytes [5..8]  → 32-bit unsigned scale multiplier Q31 (little-endian)
Byte  [9]     → 6-bit right-shift (bits [5:0]), bits [7:6] = 0
```

```python
# weight_compressor.py:202-232 — encode_bias()
data[0] = (bias >> (0 * 8)) & 0xFF   # bias byte 0
data[1] = (bias >> (1 * 8)) & 0xFF
data[2] = (bias >> (2 * 8)) & 0xFF
data[3] = (bias >> (3 * 8)) & 0xFF
data[4] = (bias >> (4 * 8)) & 0xFF   # bias byte 4 (40-bit total)
data[5] = (scale >> (0 * 8)) & 0xFF  # scale byte 0
data[6] = (scale >> (1 * 8)) & 0xFF
data[7] = (scale >> (2 * 8)) & 0xFF
data[8] = (scale >> (3 * 8)) & 0xFF  # scale byte 3 (32-bit Q31 total)
data[9] = shift & 0x3F               # 6-bit shift
```

Scale derivation:
```python
# scaling.py:33-45 — quantise_scale()
scale_fp = (ifm_scale × weight_scale) / ofm_scale
significand, exponent = math.frexp(scale_fp)
scale_q31 = round(significand × 2^31)   # 32-bit integer
shift      = -(exponent - 31)            # 6-bit, range 0..63
```

For int16 input with int64 bias: **reduced precision** (16-bit multiplier, adjusted shift).

```python
# scaling.py:47-57 — reduced_quantise_scale()
# weight_compressor.py:300-302
if ifm_dtype == DataType.int16 and bias_tens.dtype == DataType.int64:
    quantised_scales = [reduced_quantise_scale(scale) for scale in scales]
```

The bias/scale stream is written **before** the compressed weights for each core, then padded to 16 bytes.

```python
# weight_compressor.py:443-453
scale_stream.extend(encode_bias(np.int64(core_bias), *core_scales[j]))
weight_range.scale_bytes = len(scale_stream)
encoded_stream.extend(scale_stream)
# Align to 16 for start of next substream
remainder = len(encoded_stream) % 16
if remainder > 0:
    encoded_stream.extend(bytearray(16 - remainder))
```

### Complete per-core stream layout

> **Source**: [`ethosu/vela/weight_compressor.py`](ethosu/vela/weight_compressor.py#L430-L470) — inner loop of `encode_weight_and_scale_tensor()`

```
[ Bias/scale for core N: OFM_channels × 10 bytes, padded to 16B ]
[ Compressed weight stream for core N, 16B aligned              ]
```

---

## 6. Command Stream Structure

### Command word formats

> **Source**: [`ethosu/vela/register_command_stream_generator.py`](ethosu/vela/register_command_stream_generator.py#L123-L225) — `CommandStreamEmitter` class

```python
# register_command_stream_generator.py:100-122
class CmdMode(IntEnum):
    NoPayload = 0x0000
    Payload32 = 0x4000
    Mask      = 0xC000
    CmdOpMask = 0x03FF

# cmd0 (no payload) — 4 bytes:  register_command_stream_generator.py:184-196
#   bits[15:0]  = opcode
#   bits[31:16] = parameter
def cmd0_with_param(self, cmd: cmd0, param):
    command = cmd.value | (param << 16)
    self.cmd_stream.append((command,))         # single 32-bit word

# cmd1 (with payload) — 8 bytes:  register_command_stream_generator.py:198-210
#   word0[15:0]  = opcode | 0x4000
#   word0[31:16] = high parameter
#   word1[31:0]  = 32-bit payload (address or value)
def cmd1_with_offset(self, cmd: cmd1, offset, param=0x0):
    command = cmd.value | CmdMode.Payload32.value | (param << 16)
    self.cmd_stream.append((command, offset))  # two 32-bit words
```

Redundant register writes are suppressed — a register is only written if the value changed.

### Register sequence for Conv2D (one block)

> **Source**: [`ethosu/vela/register_command_stream_generator.py`](ethosu/vela/register_command_stream_generator.py#L443-L500) — `generate_ifm()`, `generate_ofm()`
> **Source**: [`ethosu/vela/register_command_stream_generator.py`](ethosu/vela/register_command_stream_generator.py#L521-L570) — `generate_weights()`, `generate_biases()`
> **Source**: [`ethosu/vela/register_command_stream_generator.py`](ethosu/vela/register_command_stream_generator.py#L502-L520) — `generate_kernel()`
> **Source**: [`ethosu/vela/register_command_stream_generator.py`](ethosu/vela/register_command_stream_generator.py#L315-L321) — `generate_padding()`
> **Source**: [`ethosu/vela/register_command_stream_generator.py`](ethosu/vela/register_command_stream_generator.py#L573-L597) — `generate_shram_registers()`
> **Source**: [`ethosu/vela/register_command_stream_generator.py`](ethosu/vela/register_command_stream_generator.py#L323-L382) — `generate_activation()`

```
# IFM — generate_ifm(), register_command_stream_generator.py:443
NPU_SET_IFM_REGION          ← memory region index
NPU_SET_IFM_BASE0..3        ← IFM tile base addresses (supports 4-tile tiling)
NPU_SET_IFM_HEIGHT0/1_M1, WIDTH0_M1   ← tile dimensions
NPU_SET_IFM_DEPTH_M1        ← IFM channels − 1
NPU_SET_IFM_STRIDE_C/Y/X    ← byte strides
NPU_SET_IFM_ZERO_POINT      ← quantization zero point
NPU_SET_IFM_PRECISION       ← data type, layout (NHWC/NHCWB16), signed

# OFM — generate_ofm(), register_command_stream_generator.py:481
NPU_SET_OFM_REGION, BASE0..3, HEIGHT/WIDTH/DEPTH, STRIDES, ZERO_POINT, PRECISION

# Weights — generate_weights(), register_command_stream_generator.py:521
NPU_SET_WEIGHT_REGION
NPU_SET_WEIGHT_BASE         ← core 0: pointer to bias/scale + weight stream (16B aligned)
NPU_SET_WEIGHT_LENGTH       ← byte length (multiple of 16)
NPU_SET_WEIGHT1_BASE        ← core 1 (U65-512 only)
NPU_SET_WEIGHT1_LENGTH

# Scale/bias — generate_biases(), register_command_stream_generator.py:545
NPU_SET_SCALE_REGION
NPU_SET_SCALE_BASE          ← core 0: pointer to start of bias/scale section
NPU_SET_SCALE_LENGTH
NPU_SET_SCALE1_BASE/LENGTH  ← core 1 (U65-512 only)

# Kernel shape — generate_kernel(), register_command_stream_generator.py:502
NPU_SET_KERNEL_HEIGHT_M1    ← (kernel_H − 1) × dilation_Y
NPU_SET_KERNEL_WIDTH_M1     ← (kernel_W − 1) × dilation_X
NPU_SET_KERNEL_STRIDE       ← stride, dilation, part-kernel flag packed in 12 bits

# Padding — generate_padding(), register_command_stream_generator.py:315
NPU_SET_IFM_PAD_TOP/LEFT/BOTTOM/RIGHT

# OFM processing block shape
NPU_SET_OFM_BLK_HEIGHT_M1
NPU_SET_OFM_BLK_WIDTH_M1
NPU_SET_OFM_BLK_DEPTH_M1

# SHRAM layout — generate_shram_registers(), register_command_stream_generator.py:573
NPU_SET_IFM_IB_END          ← end bank of IFM buffer
NPU_SET_AB_START            ← start bank of accumulator
NPU_SET_IFM2_IB_START       ← (elementwise ops only)
NPU_SET_ACC_FORMAT           ← INT_32BIT or INT_40BIT

# Activation — generate_activation(), register_command_stream_generator.py:323
NPU_SET_ACTIVATION           ← NONE / TANH / SIGMOID / TABLE_LOOKUP
NPU_SET_ACTIVATION_MIN/MAX   ← quantized clamp values

# KICK — starts the DPU
NPU_OP_CONV / NPU_OP_DEPTHWISE_CONV / NPU_OP_POOL / NPU_OP_ELEMENTWISE
```

### Alignment enforcement

> **Source**: [`ethosu/vela/register_command_stream_util.py`](ethosu/vela/register_command_stream_util.py#L59) — `check_alignment()`
> **Source**: [`ethosu/vela/register_command_stream_generator.py`](ethosu/vela/register_command_stream_generator.py#L530-L543) — alignment checks inside `generate_weights()`

```python
# register_command_stream_generator.py:530-543
for core, (addr, length) in enumerate(...):
    check_alignment(weights[core].address, 16)   # must be 16-byte aligned
    check_length(weights[core].length, 16)        # must be multiple of 16
```

### DMA wait/dependency commands

> **Source**: [`ethosu/vela/register_command_stream_generator.py`](ethosu/vela/register_command_stream_generator.py#L213-L220) — `cmd_wait()`

Between blocks, the command stream inserts `NPU_KERNEL_WAIT` and `NPU_DMA_WAIT` to synchronize:
- `NPU_KERNEL_WAIT(channel, outstanding)` — stall until pending kernel blocks ≤ `outstanding`
- `NPU_DMA_WAIT(channel, outstanding)` — stall until pending DMA transactions ≤ `outstanding`

Max outstanding kernels = 2 (pipeline depth), max outstanding DMA = 2 for U65.

---

## 7. Practical Throughput Summary

### How many bytes per cycle for Conv2D (int8, U65-256)?

> **Source**: [`ethosu/vela/architecture_features.py`](ethosu/vela/architecture_features.py#L293-L311) — AXI bus width and bandwidth
> **Source**: [`ethosu/vela/npu_performance.py`](ethosu/vela/npu_performance.py#L314-L435) — `_estimate_conv_cycles()`

| Item | Per cycle |
|---|---|
| DPU throughput | 256 MACs = 256 weight×input products |
| Weight fetch from memory | 16 bytes = 16 int8 weights |
| IFM fetch | 16 bytes = 16 int8 activations |
| OFM write | 16 bytes = 16 int8 outputs |
| Scale/bias | 16 bytes = 1.6 output channels (10B each) |

The DPU is **compute-bound** when weight bandwidth cannot keep up:  
- To sustain 256 MACs/cycle you need 256 weight bytes/cycle, but AXI delivers only 16 bytes/cycle  
- This means the DPU only runs at peak when the IFM block depth (32 channels) and sub-kernel reuse amortize the weight fetch across many IFM positions  
- Effective weight bandwidth requirement: `256 MACs / IFM_block_depth = 256/32 = 8 weight bytes/cycle` → fits in the 16 B/cycle AXI budget with headroom for IFM reads

### Minimum cycles for one OFM element (U65-256, Conv int8)

> **Source**: [`ethosu/vela/npu_performance.py`](ethosu/vela/npu_performance.py#L419-L435) — final cycle calculation in `_estimate_conv_cycles()`

```
DPU: 1 MAC / 256 MACs_per_cycle = 0.0039 cycles/MAC
     = 1 / (kernel_H × kernel_W × IFM_depth) cycles/output element
Output stage: 0.625 cycles/output element (dominant for small kernels)
```

So for a 1×1 Conv with 256 IFM channels:
- DPU: 256 MACs / 256 MACs_per_cycle = 1.0 DPU cycle per output element
- Output stage: 0.625 cycles/elem
- Bottleneck: DPU at 1.0 cycles/elem

For a 3×3 Conv with 64 IFM channels:
- DPU: 9 × 64 / 256 = 2.25 DPU cycles per output element
- Output stage: 0.625 cycles/elem
- Bottleneck: DPU at 2.25 cycles/elem

---

## 8. Key Constraints and Alignment Rules

> **Source**: [`ethosu/vela/architecture_features.py`](ethosu/vela/architecture_features.py#L365) — `ofm_block_max`
> **Source**: [`ethosu/vela/architecture_features.py`](ethosu/vela/architecture_features.py#L400-L411) — `min_block_sizes`
> **Source**: [`ethosu/vela/architecture_features.py`](ethosu/vela/architecture_features.py#L243) — `SubKernelMax`
> **Source**: [`ethosu/vela/architecture_features.py`](ethosu/vela/architecture_features.py#L496-L502) — `calc_ifm_block_depth()`
> **Source**: [`ethosu/vela/register_command_stream_util.py`](ethosu/vela/register_command_stream_util.py#L59) — `check_alignment()`
> **Source**: [`ethosu/vela/weight_compressor.py`](ethosu/vela/weight_compressor.py#L202-L232) — `encode_bias()` (10-byte format, 6-bit shift, Q31 scale)

| Rule | Value | Source line |
|---|---|---|
| Weight base address alignment | 16 bytes | `register_command_stream_generator.py:530` |
| Weight stream length alignment | 16 bytes | `register_command_stream_generator.py:531` |
| Bias/scale stream padding | 16 bytes | `weight_compressor.py:449-452` |
| SHRAM bank size | 1024 bytes | `architecture_features.py:435` |
| Max kernel size per sub-kernel tile | 8×8 | `architecture_features.py:243` |
| Max OFM block size | 64 (H) × 32 (W) × 128 (D) | `architecture_features.py:365` |
| Min OFM block size (Conv) | 2 (H) × 2 (W) | `architecture_features.py:401` |
| IFM block depth (int8, depth-first) | 32 channels | `architecture_features.py:499` |
| IFM block depth (int16, depth-first) | 16 channels | `architecture_features.py:499` |
| IFM block depth (part-kernel-first) | 16 channels always | `architecture_features.py:498` |
| OFM ublock depth | 8 channels | `architecture_features.py:222-226` |
| Max dilation | 2 (in X or Y) | `weight_compressor.py:167-168` |
| MLW palette size | up to 32 entries | `mlw_encode.c:222` (`create_palette`) |
| Bias packed size | 10 bytes per channel | `weight_compressor.py:202` |
| Max shift in bias scale | 6 bits (0..63) | `weight_compressor.py:229` |
| Scale multiplier precision | Q31, 32-bit unsigned | `scaling.py:39` |
| Address space | 40-bit (up to 1 TB) | `architecture_features.py:295` |
