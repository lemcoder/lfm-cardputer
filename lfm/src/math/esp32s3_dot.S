// ESP32-S3 PIE (SIMD) kernels for the integer dot products: per 32-weight
// block, the int32 sum of weight * activation. The C wrapper in esp32s3.cpp
// applies the block scales. Activations (xq) must be 16-byte aligned; weight
// rows can sit at any address (blocks are 18 / 34 bytes): unaligned loads use
// EE.LD.128.USAR (load the aligned 16 bytes, remember the offset in
// SAR_BYTE) + EE.SRC.Q (shift two such loads into the 16 bytes we want).
// Loads may read up to 16 bytes past the last block: buffers are padded.
//
// Xtensa windowed ABI: args in a2..a5, `entry` / `retw.n`.

    .text

// void lfm_s3_q4_q8_sums(const uint8_t* row, const int8_t* xq, int32_t* sums, int nb)
//
// Q4_0 block: f16 scale (2 bytes), then 16 bytes; byte j holds element j in
// its low nibble and element j+16 in its high nibble; weight = nibble - 8.
    .align  4
    .global lfm_s3_q4_q8_sums
    .type   lfm_s3_q4_q8_sums, @function
lfm_s3_q4_q8_sums:
    entry   a1, 32
    // q7 = 0x0F in every byte, q6 = 8 in every byte (broadcast from the stack)
    movi    a9, 0x0F
    s8i     a9, a1, 0
    movi    a9, 8
    s8i     a9, a1, 1
    mov     a10, a1
    ee.vldbc.8.ip   q7, a10, 1
    ee.vldbc.8.ip   q6, a10, 0
    ssai    4                           // SAR = 4 for ee.vsr.32

    loopnez a5, .Lq4_end
        addi    a8, a2, 2               // this block's nibbles
        ee.ld.128.usar.ip   q0, a8, 16
        ee.vld.128.ip       q1, a8, 0
        ee.src.q            q0, q0, q1  // q0 = the 16 nibble bytes
        ee.andq             q1, q0, q7  // low nibbles: elements 0..15
        ee.vsr.32           q2, q0      // high nibbles down by 4 ...
        ee.andq             q2, q2, q7  // ... masked: elements 16..31
        ee.vsubs.s8         q1, q1, q6  // 0..15 -> -8..7
        ee.vsubs.s8         q2, q2, q6
        ee.vld.128.ip       q3, a3, 16  // activations 0..15
        ee.vld.128.ip       q4, a3, 16  // activations 16..31
        ee.zero.accx
        ee.vmulas.s8.accx   q1, q3
        ee.vmulas.s8.accx   q2, q4
        rur.accx_0          a9
        s32i    a9, a4, 0
        addi    a4, a4, 4
        addi    a2, a2, 18
.Lq4_end:
    retw.n
    .size   lfm_s3_q4_q8_sums, . - lfm_s3_q4_q8_sums

// void lfm_s3_q8_q8_sums(const uint8_t* row, const int8_t* xq, int32_t* sums, int nb)
//
// Q8_0 block: f16 scale (2 bytes), then 32 int8 weights.
    .align  4
    .global lfm_s3_q8_q8_sums
    .type   lfm_s3_q8_q8_sums, @function
lfm_s3_q8_q8_sums:
    entry   a1, 32
    loopnez a5, .Lq8_end
        addi    a8, a2, 2               // this block's weights
        ee.ld.128.usar.ip   q0, a8, 16
        ee.vld.128.ip       q1, a8, 16
        ee.vld.128.ip       q2, a8, 0
        ee.src.q            q0, q0, q1  // weights 0..15
        ee.src.q            q1, q1, q2  // weights 16..31
        ee.vld.128.ip       q3, a3, 16
        ee.vld.128.ip       q4, a3, 16
        ee.zero.accx
        ee.vmulas.s8.accx   q0, q3
        ee.vmulas.s8.accx   q1, q4
        rur.accx_0          a9
        s32i    a9, a4, 0
        addi    a4, a4, 4
        addi    a2, a2, 34
.Lq8_end:
    retw.n
    .size   lfm_s3_q8_q8_sums, . - lfm_s3_q8_q8_sums

// void lfm_s3_q4_q8_rows(const uint8_t* blocks, const int8_t* xq, int32_t* sums, int rows, int nb)
//
// Many short Q4_0 rows (nb blocks each, stored back to back, e.g. a [vocab, 128]
// table) against one activation vector of nb*32 int8: sums[r*nb + b]. One call
// for a whole chunk of rows instead of one per row, whose fixed cost dominated
// for 4-block rows.
    .align  4
    .global lfm_s3_q4_q8_rows
    .type   lfm_s3_q4_q8_rows, @function
lfm_s3_q4_q8_rows:
    entry   a1, 32
    movi    a9, 0x0F
    s8i     a9, a1, 0
    movi    a9, 8
    s8i     a9, a1, 1
    mov     a10, a1
    ee.vldbc.8.ip   q7, a10, 1
    ee.vldbc.8.ip   q6, a10, 0
    ssai    4
.Lrows_row:
    beqz    a5, .Lrows_end
    mov     a7, a3                      // activations restart for every row
    loopnez a6, .Lrows_blk_end
        addi    a8, a2, 2
        ee.ld.128.usar.ip   q0, a8, 16
        ee.vld.128.ip       q1, a8, 0
        ee.src.q            q0, q0, q1
        ee.andq             q1, q0, q7
        ee.vsr.32           q2, q0
        ee.andq             q2, q2, q7
        ee.vsubs.s8         q1, q1, q6
        ee.vsubs.s8         q2, q2, q6
        ee.vld.128.ip       q3, a7, 16
        ee.vld.128.ip       q4, a7, 16
        ee.zero.accx
        ee.vmulas.s8.accx   q1, q3
        ee.vmulas.s8.accx   q2, q4
        rur.accx_0          a9
        s32i    a9, a4, 0
        addi    a4, a4, 4
        addi    a2, a2, 18
.Lrows_blk_end:
    addi    a5, a5, -1
    j       .Lrows_row
.Lrows_end:
    retw.n
    .size   lfm_s3_q4_q8_rows, . - lfm_s3_q4_q8_rows

// void lfm_s3_q4_q8_multi(const MultiArgs* a)
//
// One Q4_0 row against n_tok (1..8) token inputs, finished in float:
//   out[t] = sum over blocks b of (d[b] * xs[t][b]) * sum_b(t)
// with d[b] the block scales (converted by the caller), sum_b(t) the int
// dot product of block b with token t's int8 inputs. Each block's nibbles
// are unpacked once for all tokens; each token's accumulator lives in a
// float register (f0..f7). Same operations and order as the C version
// (mul.s, then madd.s), so the results are bit-identical.
//
// struct MultiArgs { row, xq, d, xs, out, nb, n_tok, xq_stride_minus_16, xs_stride_bytes } (offsets 0..32)
    .macro TOK fk
        lsi     f9, a11, 0                  // the scale product first: ready by the madd
        add     a11, a11, a9
        mul.s   f9, f8, f9                  // d * xs
        ee.vld.128.ip       q3, a10, 16
        ee.vld.128.xp       q4, a10, a13    // a10 -> this block of the next token
        ee.zero.accx
        ee.vmulas.s8.accx   q1, q3
        ee.vmulas.s8.accx   q2, q4
        rur.accx_0          a12
        float.s f10, a12, 0
        madd.s  \fk, f9, f10               // acc += (d * xs) * sum
    .endm

    .align  4
    .global lfm_s3_q4_q8_multi
    .type   lfm_s3_q4_q8_multi, @function
lfm_s3_q4_q8_multi:
    entry   a1, 48
    l32i    a14, a2, 16                 // out
    s32i    a14, a1, 4
    l32i    a6, a2, 20                  // nb
    l32i    a7, a2, 24                  // n_tok
    l32i    a13, a2, 28                 // xq stride - 16
    l32i    a9, a2, 32                  // xs stride, bytes
    l32i    a3, a2, 4                   // xq (token 0, block 0)
    l32i    a4, a2, 8                   // d
    l32i    a5, a2, 12                  // xs (token 0, block 0)
    l32i    a2, a2, 0                   // row
    movi    a12, 0
    wfr     f0, a12
    wfr     f1, a12
    wfr     f2, a12
    wfr     f3, a12
    wfr     f4, a12
    wfr     f5, a12
    wfr     f6, a12
    wfr     f7, a12
    movi    a12, 0x0F
    s8i     a12, a1, 0
    movi    a12, 8
    s8i     a12, a1, 1
    mov     a10, a1
    ee.vldbc.8.ip   q7, a10, 1
    ee.vldbc.8.ip   q6, a10, 0
    ssai    4
.Lmulti_blk:
    beqz    a6, .Lmulti_store
    addi    a8, a2, 2
    ee.ld.128.usar.ip   q0, a8, 16
    ee.vld.128.ip       q1, a8, 0
    ee.src.q            q0, q0, q1
    ee.andq             q1, q0, q7
    ee.vsr.32           q2, q0
    ee.andq             q2, q2, q7
    ee.vsubs.s8         q1, q1, q6
    ee.vsubs.s8         q2, q2, q6
    lsi     f8, a4, 0                   // d[b]
    mov     a10, a3
    mov     a11, a5
    TOK f0
    beqi    a7, 1, .Lmulti_next
    TOK f1
    beqi    a7, 2, .Lmulti_next
    TOK f2
    beqi    a7, 3, .Lmulti_next
    TOK f3
    beqi    a7, 4, .Lmulti_next
    TOK f4
    beqi    a7, 5, .Lmulti_next
    TOK f5
    beqi    a7, 6, .Lmulti_next
    TOK f6
    beqi    a7, 7, .Lmulti_next
    TOK f7
.Lmulti_next:
    addi    a3, a3, 32
    addi    a4, a4, 4
    addi    a5, a5, 4
    addi    a2, a2, 18
    addi    a6, a6, -1
    j       .Lmulti_blk
.Lmulti_store:
    l32i    a14, a1, 4
    ssi     f0, a14, 0
    beqi    a7, 1, .Lmulti_end
    ssi     f1, a14, 4
    beqi    a7, 2, .Lmulti_end
    ssi     f2, a14, 8
    beqi    a7, 3, .Lmulti_end
    ssi     f3, a14, 12
    beqi    a7, 4, .Lmulti_end
    ssi     f4, a14, 16
    beqi    a7, 5, .Lmulti_end
    ssi     f5, a14, 20
    beqi    a7, 6, .Lmulti_end
    ssi     f6, a14, 24
    beqi    a7, 7, .Lmulti_end
    ssi     f7, a14, 28
.Lmulti_end:
    retw.n
    .size   lfm_s3_q4_q8_multi, . - lfm_s3_q4_q8_multi
