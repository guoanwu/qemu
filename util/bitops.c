/*
 * Copyright (C) 2004 Red Hat, Inc. All Rights Reserved.
 * Written by David Howells (dhowells@redhat.com)
 * Copyright (C) 2008 IBM Corporation
 * Written by Rusty Russell <rusty@rustcorp.com.au>
 * (Inspired by David Howell's find_next_bit implementation)
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version
 * 2 of the License, or (at your option) any later version.
 */

#include "qemu/osdep.h"
#include "qemu/bitops.h"
#include <stddef.h>
#include <stdint.h>

/*
 * Find the next set bit in a memory region.
 */
unsigned long find_next_bit(const unsigned long *addr, unsigned long size,
                            unsigned long offset)
{
    const unsigned long *p = addr + BIT_WORD(offset);
    unsigned long result = offset & ~(BITS_PER_LONG-1);
    unsigned long tmp;

    if (offset >= size) {
        return size;
    }
    size -= result;
    offset %= BITS_PER_LONG;
    if (offset) {
        tmp = *(p++);
        tmp &= (~0UL << offset);
        if (size < BITS_PER_LONG) {
            goto found_first;
        }
        if (tmp) {
            goto found_middle;
        }
        size -= BITS_PER_LONG;
        result += BITS_PER_LONG;
    }
    while (size >= 4*BITS_PER_LONG) {
        unsigned long d1, d2, d3;
        tmp = *p;
        d1 = *(p+1);
        d2 = *(p+2);
        d3 = *(p+3);
        if (tmp) {
            goto found_middle;
        }
        if (d1 | d2 | d3) {
            break;
        }
        p += 4;
        result += 4*BITS_PER_LONG;
        size -= 4*BITS_PER_LONG;
    }
    while (size >= BITS_PER_LONG) {
        if ((tmp = *(p++))) {
            goto found_middle;
        }
        result += BITS_PER_LONG;
        size -= BITS_PER_LONG;
    }
    if (!size) {
        return result;
    }
    tmp = *p;

found_first:
    tmp &= (~0UL >> (BITS_PER_LONG - size));
    if (tmp == 0UL) {		/* Are any bits set? */
        return result + size;	/* Nope. */
    }
found_middle:
    return result + ctzl(tmp);
}

/*
 * This implementation of find_{first,next}_zero_bit was stolen from
 * Linus' asm-alpha/bitops.h.
 */
unsigned long find_next_zero_bit(const unsigned long *addr, unsigned long size,
                                 unsigned long offset)
{
    const unsigned long *p = addr + BIT_WORD(offset);
    unsigned long result = offset & ~(BITS_PER_LONG-1);
    unsigned long tmp;

    if (offset >= size) {
        return size;
    }
    size -= result;
    offset %= BITS_PER_LONG;
    if (offset) {
        tmp = *(p++);
        tmp |= ~0UL >> (BITS_PER_LONG - offset);
        if (size < BITS_PER_LONG) {
            goto found_first;
        }
        if (~tmp) {
            goto found_middle;
        }
        size -= BITS_PER_LONG;
        result += BITS_PER_LONG;
    }
    while (size & ~(BITS_PER_LONG-1)) {
        if (~(tmp = *(p++))) {
            goto found_middle;
        }
        result += BITS_PER_LONG;
        size -= BITS_PER_LONG;
    }
    if (!size) {
        return result;
    }
    tmp = *p;

found_first:
    tmp |= ~0UL << size;
    if (tmp == ~0UL) {	/* Are any bits zero? */
        return result + size;	/* Nope. */
    }
found_middle:
    return result + ctzl(~tmp);
}

unsigned long find_last_bit(const unsigned long *addr, unsigned long size)
{
    unsigned long words;
    unsigned long tmp;

    /* Start at final word. */
    words = size / BITS_PER_LONG;

    /* Partial final word? */
    if (size & (BITS_PER_LONG-1)) {
        tmp = (addr[words] & (~0UL >> (BITS_PER_LONG
                                       - (size & (BITS_PER_LONG-1)))));
        if (tmp) {
            goto found;
        }
    }

    while (words) {
        tmp = addr[--words];
        if (tmp) {
        found:
            return words * BITS_PER_LONG + BITS_PER_LONG - 1 - clzl(tmp);
        }
    }

    /* Not found */
    return size;
}

#if defined(CONFIG_AVX512F_OPT) && defined(CONFIG_AVX512BW_OPT) \
    && defined(CONFIG_BMI2_OPT)
#pragma GCC push_options
#pragma GCC target("avx512f", "avx512bw", "bmi2")
#include <immintrin.h>

enum ID_FOR_HALF { half32, half16, half8, half4, half2, half1 };

static const uint64_t HalfVals[] = {
  0xffffffff00000000,
  0xffff0000ffff0000,
  0xff00ff00ff00ff00,
  0xf0f0f0f0f0f0f0f0,
  0xcccccccccccccccc,
  0xaaaaaaaaaaaaaaaa
};
static __m512i weightFor0thBits;
static __m512i weightFor1thBits;
static __m512i weightFor2thBits;
static __m512i weightFor3thBits;
static __m512i weightFor4thBits;
static __m512i weightFor5thBits;
static __m512i allZero512;

void init_avx512_consts_for_bitmap_scan(void)
{
    weightFor0thBits = _mm512_set1_epi8(1);
    weightFor1thBits = _mm512_set1_epi8(2);
    weightFor2thBits = _mm512_set1_epi8(4);
    weightFor3thBits = _mm512_set1_epi8(8);
    weightFor4thBits = _mm512_set1_epi8(16);
    weightFor5thBits = _mm512_set1_epi8(32);
    allZero512       = _mm512_set1_epi8(0);
}

inline __attribute__((always_inline))
int64_t find_all_bits_in_bitmap64_avx512(const uint64_t *bitmap, size_t offset,
                                               uint8_t *res) {
  uint64_t bmp64 = bitmap[offset];

  uint64_t bits5sForAllIndexs = _pext_u64(HalfVals[half32], bmp64);
  uint64_t bits4sForAllIndexs = _pext_u64(HalfVals[half16], bmp64);
  uint64_t bits3sForAllIndexs = _pext_u64(HalfVals[half8], bmp64);
  uint64_t bit2sForAllIndexs  = _pext_u64(HalfVals[half4], bmp64);
  uint64_t bit1sForAllIndexs  = _pext_u64(HalfVals[half2], bmp64);
  uint64_t bit0sForAllIndexs  = _pext_u64(HalfVals[half1], bmp64);

  __m512i tmp;

  tmp = _mm512_maskz_add_epi8(bits5sForAllIndexs, weightFor5thBits, allZero512);
  tmp = _mm512_mask_add_epi8(tmp, bits4sForAllIndexs, weightFor4thBits, tmp);
  tmp = _mm512_mask_add_epi8(tmp, bits3sForAllIndexs, weightFor3thBits, tmp);
  tmp = _mm512_mask_add_epi8(tmp, bit2sForAllIndexs, weightFor2thBits, tmp);
  tmp = _mm512_mask_add_epi8(tmp, bit1sForAllIndexs, weightFor1thBits, tmp);
  tmp = _mm512_mask_add_epi8(tmp, bit0sForAllIndexs, weightFor0thBits, tmp);
  _mm512_mask_storeu_epi8(res, 0xffffffffffffffff, tmp);

  return __builtin_popcountll(bmp64);
}
#pragma GCC pop_options
#endif


