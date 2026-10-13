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
