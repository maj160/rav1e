// Copyright (c) 2019-2022, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

pub use self::cdef_dist::*;
pub use self::sse::*;
use crate::cpu_features::CpuFeatureLevel;
use crate::dist::*;
use crate::partition::BlockSize;
use crate::tiling::*;
use crate::util::*;

mod cdef_dist;
mod sse;

type SadFn = unsafe extern fn(
  src: *const u8,
  src_stride: isize,
  dst: *const u8,
  dst_stride: isize,
) -> u32;

type SatdFn = SadFn;

type SadHBDFn = unsafe extern fn(
  src: *const u16,
  src_stride: isize,
  dst: *const u16,
  dst_stride: isize,
) -> u32;

type SatdHBDFn = unsafe extern fn(
  src: *const u16,
  src_stride: isize,
  dst: *const u16,
  dst_stride: isize,
  bdmax: u32,
) -> u32;

macro_rules! declare_asm_dist_fn {
  ($(($name: ident, $T: ident)),+) => (
    $(
      extern { fn $name (
        src: *const $T, src_stride: isize, dst: *const $T, dst_stride: isize
      ) -> u32; }
    )+
  )
}

macro_rules! declare_asm_satd_hbd_fn {
  ($($name: ident),+) => (
    $(
      extern { pub(crate) fn $name (
        src: *const u16, src_stride: isize, dst: *const u16, dst_stride: isize, bdmax: u32
      ) -> u32; }
    )+
  )
}

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[target_feature(enable = "avx2")]
#[inline]
unsafe fn mm_sum_i32(ymm : __m256i) -> i32 {
  // We split the vector in half and then add 2 + 3 -> 0 + 1, and finally 0 + 1.
  let m1 = _mm256_extracti128_si256(ymm, 1);
  let m2 = _mm256_castsi256_si128(ymm);
  let m2 = _mm_add_epi32(m2, m1);
  let m1 = _mm_shuffle_epi32(m2, 0b11_10_11_10);
  let m2 = _mm_add_epi32(m2, m1);
  let m1 = _mm_shuffle_epi32(m2, 0b01_01_01_01);
  let m2 = _mm_add_epi32(m2, m1);
  _mm_cvtsi128_si32(m2)
}

/// Performs a restricted 16-bit SAD over up to 256 elements.
/// This is only valid if the inputs are 12-bit at most.
/// Because we can accumulate 16 such values per AVX2 lane.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn rav1e_sad_KxN_hbd_avx2_inner<const K : usize>(src: *const u16, src_stride: isize, dst: *const u16, dst_stride: isize, n_rows : usize) -> __m256i {

  const LOADS_PER_REGISTER : isize = 32 / core::mem::size_of::<u16>() as isize; // 16
  assert!(K * n_rows <= 256);
  assert_eq!(K % LOADS_PER_REGISTER as usize, 0);
  
  let zero = _mm256_setzero_si256();
  let src = src as *const i16;
  let dst = dst as *const i16;
  let mut sum16 = _mm256_setzero_si256();
  for y in 0..n_rows as isize {
    for x in 0..(K/16) as isize {
      let a = _mm256_abs_epi16(_mm256_sub_epi16(
        _mm256_loadu_si256(src.offset(x * LOADS_PER_REGISTER + y * src_stride / 2) as *const _),
        _mm256_loadu_si256(dst.offset(x * LOADS_PER_REGISTER + y * dst_stride / 2) as *const _)
      ));
      sum16 = _mm256_add_epi16(sum16, a);
    }
  }

  let b = _mm256_unpackhi_epi16(sum16, zero);
  let c = _mm256_unpacklo_epi16(sum16, zero);

  _mm256_add_epi32(b, c)
}

#[target_feature(enable = "avx2")]
#[inline]
unsafe fn rav1e_sad_KxN_hbd_avx2<const K : usize>(src: *const u16, src_stride: isize, dst: *const u16, dst_stride: isize, n_rows : usize) -> u32 {
  if (K * n_rows <= 256) {
    return mm_sum_i32(rav1e_sad_KxN_hbd_avx2_inner::<K>(src, src_stride, dst, dst_stride, row_step)) as u32;
  }

  // IF LPR = 16 and k =16, rows % 4
  // If LPR = 16 and k = 64, rows % 1
  const MAX_ELEMS : usize = 256;
  let mut sum = _mm256_setzero_si256();
  let col_step = K.min(MAX_ELEMS);
  let row_step = n_rows.min(MAX_ELEMS / col_step);
  assert!(row_step * col_step <= MAX_ELEMS);
  assert_eq!(K % col_step, 0);
  assert_eq!(n_rows % row_step, 0);
  for w in 0..(n_rows / row_step) as isize {
    for h in 0..(K / col_step) as isize {
      let src = src.offset(w * src_stride / 2 * row_step as isize + h * col_step as isize);
      let dst = dst.offset(w * dst_stride / 2 * row_step as isize + h * col_step as isize);
      let res = if K > MAX_ELEMS {
        rav1e_sad_KxN_hbd_avx2_inner::<MAX_ELEMS>(src, src_stride, dst, dst_stride, row_step)
       } else {
        rav1e_sad_KxN_hbd_avx2_inner::<K>(src, src_stride, dst, dst_stride, row_step)
       };
      sum = _mm256_add_epi32(sum, res);
    }
  }
  mm_sum_i32(sum) as u32
}

declare_asm_dist_fn![
  // SSSE3
  (rav1e_sad_4x4_hbd_ssse3, u16),
  (rav1e_sad_16x16_hbd_ssse3, u16),
  (rav1e_satd_8x8_ssse3, u8),
  // SSE2
  (rav1e_sad4x4_sse2, u8),
  (rav1e_sad4x8_sse2, u8),
  (rav1e_sad4x16_sse2, u8),
  (rav1e_sad8x4_sse2, u8),
  (rav1e_sad8x8_sse2, u8),
  (rav1e_sad8x16_sse2, u8),
  (rav1e_sad8x32_sse2, u8),
  (rav1e_sad16x4_sse2, u8),
  (rav1e_sad16x8_sse2, u8),
  (rav1e_sad16x16_sse2, u8),
  (rav1e_sad16x32_sse2, u8),
  (rav1e_sad16x64_sse2, u8),
  (rav1e_sad32x8_sse2, u8),
  (rav1e_sad32x16_sse2, u8),
  (rav1e_sad32x32_sse2, u8),
  (rav1e_sad32x64_sse2, u8),
  (rav1e_sad64x16_sse2, u8),
  (rav1e_sad64x32_sse2, u8),
  (rav1e_sad64x64_sse2, u8),
  (rav1e_sad64x128_sse2, u8),
  (rav1e_sad128x64_sse2, u8),
  (rav1e_sad128x128_sse2, u8),
  // SSE4
  (rav1e_satd_4x4_sse4, u8),
  // AVX

  (rav1e_sad32x8_avx2, u8),
  (rav1e_sad32x16_avx2, u8),
  (rav1e_sad32x32_avx2, u8),
  (rav1e_sad32x64_avx2, u8),
  (rav1e_sad64x16_avx2, u8),
  (rav1e_sad64x32_avx2, u8),
  (rav1e_sad64x64_avx2, u8),
  (rav1e_sad64x128_avx2, u8),
  (rav1e_sad128x64_avx2, u8),
  (rav1e_sad128x128_avx2, u8),
  (rav1e_satd_4x4_avx2, u8),
  (rav1e_satd_8x8_avx2, u8),
  (rav1e_satd_16x16_avx2, u8),
  (rav1e_satd_32x32_avx2, u8),
  (rav1e_satd_64x64_avx2, u8),
  (rav1e_satd_128x128_avx2, u8),
  (rav1e_satd_4x8_avx2, u8),
  (rav1e_satd_8x4_avx2, u8),
  (rav1e_satd_8x16_avx2, u8),
  (rav1e_satd_16x8_avx2, u8),
  (rav1e_satd_16x32_avx2, u8),
  (rav1e_satd_32x16_avx2, u8),
  (rav1e_satd_32x64_avx2, u8),
  (rav1e_satd_64x32_avx2, u8),
  (rav1e_satd_64x128_avx2, u8),
  (rav1e_satd_128x64_avx2, u8),
  (rav1e_satd_4x16_avx2, u8),
  (rav1e_satd_16x4_avx2, u8),
  (rav1e_satd_8x32_avx2, u8),
  (rav1e_satd_32x8_avx2, u8),
  (rav1e_satd_16x64_avx2, u8),
  (rav1e_satd_64x16_avx2, u8)
];

declare_asm_satd_hbd_fn![
  rav1e_satd_4x4_hbd_avx2,
  rav1e_satd_8x4_hbd_avx2,
  rav1e_satd_4x8_hbd_avx2,
  rav1e_satd_8x8_hbd_avx2,
  rav1e_satd_16x8_hbd_avx2,
  rav1e_satd_16x16_hbd_avx2,
  rav1e_satd_32x32_hbd_avx2,
  rav1e_satd_64x64_hbd_avx2,
  rav1e_satd_128x128_hbd_avx2,
  rav1e_satd_16x32_hbd_avx2,
  rav1e_satd_16x64_hbd_avx2,
  rav1e_satd_32x16_hbd_avx2,
  rav1e_satd_32x64_hbd_avx2,
  rav1e_satd_64x16_hbd_avx2,
  rav1e_satd_64x32_hbd_avx2,
  rav1e_satd_64x128_hbd_avx2,
  rav1e_satd_128x64_hbd_avx2,
  rav1e_satd_32x8_hbd_avx2,
  rav1e_satd_8x16_hbd_avx2,
  rav1e_satd_8x32_hbd_avx2,
  rav1e_satd_16x4_hbd_avx2,
  rav1e_satd_4x16_hbd_avx2
];

// BlockSize::BLOCK_SIZES.next_power_of_two();
pub(crate) const DIST_FNS_LENGTH: usize = 32;

#[inline]
pub(crate) const fn to_index(bsize: BlockSize) -> usize {
  bsize as usize & (DIST_FNS_LENGTH - 1)
}

/// # Panics
///
/// - If in `check_asm` mode, panics on mismatch between native and ASM results.
#[inline(always)]
#[allow(clippy::let_and_return)]
pub fn get_sad<T: Pixel>(
  src: &PlaneRegion<'_, T>, dst: &PlaneRegion<'_, T>,
  bit_depth: usize, cpu: CpuFeatureLevel,
) -> u32 {
  let bsize_opt = BlockSize::from_width_and_height_opt(src.rect().width, src.rect().height);

  let call_rust = || -> u32 { rust::get_sad(dst, src, bit_depth, cpu) };

  #[cfg(feature = "check_asm")]
  let ref_dist = call_rust();

  let dist = match (bsize_opt, T::type_enum()) {
    (Err(_), _) => call_rust(),
    (Ok(bsize), PixelType::U8) => {
      match SAD_FNS[cpu.as_index()][to_index(bsize)] {
        // SAFETY: Calls Assembly code.
        Some(func) => unsafe {
          (func)(
            src.data_ptr() as *const _,
            T::to_asm_stride(src.plane_cfg.stride),
            dst.data_ptr() as *const _,
            T::to_asm_stride(dst.plane_cfg.stride),
          )
        },
        None => call_rust(),
      }
    }
    (Ok(bsize), PixelType::U16) => {
      match SAD_HBD_FNS[cpu.as_index()][to_index(bsize)] {
        // SAFETY: Calls Assembly code.
        Some(func) => unsafe {
          (func)(
            src.data_ptr() as *const _,
            T::to_asm_stride(src.plane_cfg.stride),
            dst.data_ptr() as *const _,
            T::to_asm_stride(dst.plane_cfg.stride),
          )
        },
        None => call_rust(),
      }
    }
  };

  #[cfg(feature = "check_asm")]
  assert_eq!(dist, ref_dist);

  dist
}

/// # Panics
///
/// - If in `check_asm` mode, panics on mismatch between native and ASM results.
#[inline(always)]
#[allow(clippy::let_and_return)]
pub fn get_satd<T: Pixel>(
  src: &PlaneRegion<'_, T>, dst: &PlaneRegion<'_, T>,
  bit_depth: usize, cpu: CpuFeatureLevel,
) -> u32 {
  let w = src.rect().width;
  let h = src.rect().height;
  let bsize_opt = BlockSize::from_width_and_height_opt(w, h);

  let call_rust = || -> u32 { rust::get_satd(dst, src, bit_depth, cpu) };

  #[cfg(feature = "check_asm")]
  let ref_dist = call_rust();

  let dist = match (bsize_opt, T::type_enum()) {
    (Err(_), _) => call_rust(),
    (Ok(bsize), PixelType::U8) => {
      match SATD_FNS[cpu.as_index()][to_index(bsize)] {
        // SAFETY: Calls Assembly code.
        Some(func) => unsafe {
          (func)(
            src.data_ptr() as *const _,
            T::to_asm_stride(src.plane_cfg.stride),
            dst.data_ptr() as *const _,
            T::to_asm_stride(dst.plane_cfg.stride),
          )
        },
        None => call_rust(),
      }
    }
    (Ok(bsize), PixelType::U16) => {
      match SATD_HBD_FNS[cpu.as_index()][to_index(bsize)] {
        // SAFETY: Calls Assembly code.
        Some(func) => unsafe {
          (func)(
            src.data_ptr() as *const _,
            T::to_asm_stride(src.plane_cfg.stride),
            dst.data_ptr() as *const _,
            T::to_asm_stride(dst.plane_cfg.stride),
            (1 << bit_depth) - 1,
          )
        },
        None => call_rust(),
      }
    }
  };

  #[cfg(feature = "check_asm")]
  assert_eq!(dist, ref_dist);

  dist
}

// We have hand-written ASM for 4x4 and 16x16 HBD blocks,
// so we can use those for other block sizes as well.
macro_rules! get_sad_hbd_ssse3 {
  ($(($W:expr, $H:expr, $BS:expr)),*) => {
    $(
      paste::item! {
        #[target_feature(enable = "ssse3")]
        unsafe extern fn [<rav1e_sad_ $W x $H _hbd_ssse3>](
          src: *const u16, src_stride: isize, dst: *const u16, dst_stride: isize,
        ) -> u32 {
          let mut sum = 0;
          for w in (0..$W).step_by($BS) {
            for h in (0..$H).step_by($BS) {
              sum += [<rav1e_sad_ $BS x $BS _hbd_ssse3>](
                src.offset(w + h * src_stride / 2),
                src_stride,
                dst.offset(w + h * dst_stride / 2),
                dst_stride
              );
            }
          }
          sum
        }
      }
    )*
  }
}

get_sad_hbd_ssse3!(
  // 4x4 base
  (8, 8, 4),
  (4, 8, 4),
  (8, 4, 4),
  (8, 16, 4),
  (16, 8, 4),
  (4, 16, 4),
  (16, 4, 4),
  (8, 32, 4),
  (32, 8, 4),
  (32, 32, 16),
  (64, 64, 16),
  (128, 128, 16),
  (16, 32, 16),
  (32, 16, 16),
  (32, 64, 16),
  (64, 32, 16),
  (64, 128, 16),
  (128, 64, 16),
  (16, 64, 16),
  (64, 16, 16)
);

macro_rules! get_sad_hbd_avx2_WxN {
  ($(($W:expr, $H:expr)),*) => {
    $(
      paste::item! {
        #[target_feature(enable = "avx2")]
        unsafe extern fn [<rav1e_sad_ $W x $H _hbd_avx2>](
          src: *const u16, src_stride: isize, dst: *const u16, dst_stride: isize,
        ) -> u32 {
          rav1e_sad_KxN_hbd_avx2::<[<$W>]>(
                src,
                src_stride,
                dst,
                dst_stride,
                $H,
              )
        }
      }
    )*
  }
}

get_sad_hbd_avx2_WxN!(
  (16,4),
  (16,8),
  (16,16),
  (16,32),
  (16,64),
  (32,8),
  (32,16),
  (32,32),
  (32,64),
  (64,16),
  (64,32),
  (64,64),
  (64,128),
  (128, 64),
  (128, 128)
);

static SAD_FNS_SSE2: [Option<SadFn>; DIST_FNS_LENGTH] = {
  let mut out: [Option<SadFn>; DIST_FNS_LENGTH] = [None; DIST_FNS_LENGTH];

  use BlockSize::*;

  out[BLOCK_4X4 as usize] = Some(rav1e_sad4x4_sse2);
  out[BLOCK_4X8 as usize] = Some(rav1e_sad4x8_sse2);
  out[BLOCK_4X16 as usize] = Some(rav1e_sad4x16_sse2);

  out[BLOCK_8X4 as usize] = Some(rav1e_sad8x4_sse2);
  out[BLOCK_8X8 as usize] = Some(rav1e_sad8x8_sse2);
  out[BLOCK_8X16 as usize] = Some(rav1e_sad8x16_sse2);
  out[BLOCK_8X32 as usize] = Some(rav1e_sad8x32_sse2);

  out[BLOCK_16X4 as usize] = Some(rav1e_sad16x4_sse2);
  out[BLOCK_16X8 as usize] = Some(rav1e_sad16x8_sse2);
  out[BLOCK_16X16 as usize] = Some(rav1e_sad16x16_sse2);
  out[BLOCK_16X32 as usize] = Some(rav1e_sad16x32_sse2);
  out[BLOCK_16X64 as usize] = Some(rav1e_sad16x64_sse2);

  out[BLOCK_32X8 as usize] = Some(rav1e_sad32x8_sse2);
  out[BLOCK_32X16 as usize] = Some(rav1e_sad32x16_sse2);
  out[BLOCK_32X32 as usize] = Some(rav1e_sad32x32_sse2);
  out[BLOCK_32X64 as usize] = Some(rav1e_sad32x64_sse2);

  out[BLOCK_64X16 as usize] = Some(rav1e_sad64x16_sse2);
  out[BLOCK_64X32 as usize] = Some(rav1e_sad64x32_sse2);
  out[BLOCK_64X64 as usize] = Some(rav1e_sad64x64_sse2);
  out[BLOCK_64X128 as usize] = Some(rav1e_sad64x128_sse2);

  out[BLOCK_128X64 as usize] = Some(rav1e_sad128x64_sse2);
  out[BLOCK_128X128 as usize] = Some(rav1e_sad128x128_sse2);

  out
};

static SAD_FNS_AVX2: [Option<SadFn>; DIST_FNS_LENGTH] = {
  let mut out: [Option<SadFn>; DIST_FNS_LENGTH] = [None; DIST_FNS_LENGTH];

  use BlockSize::*;

  out[BLOCK_4X4 as usize] = Some(rav1e_sad4x4_sse2);
  out[BLOCK_4X8 as usize] = Some(rav1e_sad4x8_sse2);
  out[BLOCK_4X16 as usize] = Some(rav1e_sad4x16_sse2);

  out[BLOCK_8X4 as usize] = Some(rav1e_sad8x4_sse2);
  out[BLOCK_8X8 as usize] = Some(rav1e_sad8x8_sse2);
  out[BLOCK_8X16 as usize] = Some(rav1e_sad8x16_sse2);
  out[BLOCK_8X32 as usize] = Some(rav1e_sad8x32_sse2);

  out[BLOCK_16X4 as usize] = Some(rav1e_sad16x4_sse2);
  out[BLOCK_16X8 as usize] = Some(rav1e_sad16x8_sse2);
  out[BLOCK_16X16 as usize] = Some(rav1e_sad16x16_sse2);
  out[BLOCK_16X32 as usize] = Some(rav1e_sad16x32_sse2);
  out[BLOCK_16X64 as usize] = Some(rav1e_sad16x64_sse2);

  out[BLOCK_32X8 as usize] = Some(rav1e_sad32x8_avx2);
  out[BLOCK_32X16 as usize] = Some(rav1e_sad32x16_avx2);
  out[BLOCK_32X32 as usize] = Some(rav1e_sad32x32_avx2);
  out[BLOCK_32X64 as usize] = Some(rav1e_sad32x64_avx2);

  out[BLOCK_64X16 as usize] = Some(rav1e_sad64x16_avx2);
  out[BLOCK_64X32 as usize] = Some(rav1e_sad64x32_avx2);
  out[BLOCK_64X64 as usize] = Some(rav1e_sad64x64_avx2);
  out[BLOCK_64X128 as usize] = Some(rav1e_sad64x128_avx2);

  out[BLOCK_128X64 as usize] = Some(rav1e_sad128x64_avx2);
  out[BLOCK_128X128 as usize] = Some(rav1e_sad128x128_avx2);

  out
};

cpu_function_lookup_table!(
  SAD_FNS: [[Option<SadFn>; DIST_FNS_LENGTH]],
  default: [None; DIST_FNS_LENGTH],
  [SSE2, AVX2]
);

static SAD_HBD_FNS_SSSE3: [Option<SadHBDFn>; DIST_FNS_LENGTH] = {
  let mut out: [Option<SadHBDFn>; DIST_FNS_LENGTH] = [None; DIST_FNS_LENGTH];

  use BlockSize::*;

  out[BLOCK_4X4 as usize] = Some(rav1e_sad_4x4_hbd_ssse3);
  out[BLOCK_8X8 as usize] = Some(rav1e_sad_8x8_hbd_ssse3);
  out[BLOCK_16X16 as usize] = Some(rav1e_sad_16x16_hbd_ssse3);
  out[BLOCK_32X32 as usize] = Some(rav1e_sad_32x32_hbd_ssse3);
  out[BLOCK_64X64 as usize] = Some(rav1e_sad_64x64_hbd_ssse3);
  out[BLOCK_128X128 as usize] = Some(rav1e_sad_128x128_hbd_ssse3);

  out[BLOCK_4X8 as usize] = Some(rav1e_sad_4x8_hbd_ssse3);
  out[BLOCK_8X4 as usize] = Some(rav1e_sad_8x4_hbd_ssse3);
  out[BLOCK_8X16 as usize] = Some(rav1e_sad_8x16_hbd_ssse3);
  out[BLOCK_16X8 as usize] = Some(rav1e_sad_16x8_hbd_ssse3);
  out[BLOCK_16X32 as usize] = Some(rav1e_sad_16x32_hbd_ssse3);
  out[BLOCK_32X16 as usize] = Some(rav1e_sad_32x16_hbd_ssse3);
  out[BLOCK_32X64 as usize] = Some(rav1e_sad_32x64_hbd_ssse3);
  out[BLOCK_64X32 as usize] = Some(rav1e_sad_64x32_hbd_ssse3);
  out[BLOCK_64X128 as usize] = Some(rav1e_sad_64x128_hbd_ssse3);
  out[BLOCK_128X64 as usize] = Some(rav1e_sad_128x64_hbd_ssse3);

  out[BLOCK_4X16 as usize] = Some(rav1e_sad_4x16_hbd_ssse3);
  out[BLOCK_16X4 as usize] = Some(rav1e_sad_16x4_hbd_ssse3);
  out[BLOCK_8X32 as usize] = Some(rav1e_sad_8x32_hbd_ssse3);
  out[BLOCK_32X8 as usize] = Some(rav1e_sad_32x8_hbd_ssse3);
  out[BLOCK_16X64 as usize] = Some(rav1e_sad_16x64_hbd_ssse3);
  out[BLOCK_64X16 as usize] = Some(rav1e_sad_64x16_hbd_ssse3);

  out
};

static SAD_HBD_FNS_AVX2: [Option<SadHBDFn>; DIST_FNS_LENGTH] = {
  let mut out: [Option<SadHBDFn>; DIST_FNS_LENGTH] = [None; DIST_FNS_LENGTH];

  use BlockSize::*;

  out[BLOCK_16X16 as usize] = Some(rav1e_sad_16x16_hbd_avx2);
  out[BLOCK_32X32 as usize] = Some(rav1e_sad_32x32_hbd_avx2);
  out[BLOCK_64X64 as usize] = Some(rav1e_sad_64x64_hbd_avx2);
  out[BLOCK_128X128 as usize] = Some(rav1e_sad_128x128_hbd_avx2);

  out[BLOCK_16X8 as usize] = Some(rav1e_sad_16x8_hbd_avx2);
  out[BLOCK_16X32 as usize] = Some(rav1e_sad_16x32_hbd_avx2);
  out[BLOCK_32X16 as usize] = Some(rav1e_sad_32x16_hbd_avx2);
  out[BLOCK_32X64 as usize] = Some(rav1e_sad_32x64_hbd_avx2);
  out[BLOCK_64X32 as usize] = Some(rav1e_sad_64x32_hbd_avx2);
  out[BLOCK_64X128 as usize] = Some(rav1e_sad_64x128_hbd_avx2);
  out[BLOCK_128X64 as usize] = Some(rav1e_sad_128x64_hbd_avx2);

  out[BLOCK_16X4 as usize] = Some(rav1e_sad_16x4_hbd_avx2);
  out[BLOCK_32X8 as usize] = Some(rav1e_sad_32x8_hbd_avx2);
  out[BLOCK_16X64 as usize] = Some(rav1e_sad_16x64_hbd_avx2);
  out[BLOCK_64X16 as usize] = Some(rav1e_sad_64x16_hbd_avx2);

  out
};

cpu_function_lookup_table!(
  SAD_HBD_FNS: [[Option<SadHBDFn>; DIST_FNS_LENGTH]],
  default: [None; DIST_FNS_LENGTH],
  [SSSE3, AVX2]
);

static SATD_FNS_SSSE3: [Option<SatdFn>; DIST_FNS_LENGTH] = {
  let mut out: [Option<SatdFn>; DIST_FNS_LENGTH] = [None; DIST_FNS_LENGTH];

  use BlockSize::*;

  out[BLOCK_8X8 as usize] = Some(rav1e_satd_8x8_ssse3);

  out
};

static SATD_FNS_SSE4_1: [Option<SatdFn>; DIST_FNS_LENGTH] = {
  let mut out: [Option<SatdFn>; DIST_FNS_LENGTH] = [None; DIST_FNS_LENGTH];

  use BlockSize::*;

  out[BLOCK_4X4 as usize] = Some(rav1e_satd_4x4_sse4);
  out[BLOCK_8X8 as usize] = Some(rav1e_satd_8x8_ssse3);

  out
};

static SATD_FNS_AVX2: [Option<SatdFn>; DIST_FNS_LENGTH] = {
  let mut out: [Option<SatdFn>; DIST_FNS_LENGTH] = [None; DIST_FNS_LENGTH];

  use BlockSize::*;

  out[BLOCK_4X4 as usize] = Some(rav1e_satd_4x4_avx2);
  out[BLOCK_8X8 as usize] = Some(rav1e_satd_8x8_avx2);
  out[BLOCK_16X16 as usize] = Some(rav1e_satd_16x16_avx2);
  out[BLOCK_32X32 as usize] = Some(rav1e_satd_32x32_avx2);
  out[BLOCK_64X64 as usize] = Some(rav1e_satd_64x64_avx2);
  out[BLOCK_128X128 as usize] = Some(rav1e_satd_128x128_avx2);

  out[BLOCK_4X8 as usize] = Some(rav1e_satd_4x8_avx2);
  out[BLOCK_8X4 as usize] = Some(rav1e_satd_8x4_avx2);
  out[BLOCK_8X16 as usize] = Some(rav1e_satd_8x16_avx2);
  out[BLOCK_16X8 as usize] = Some(rav1e_satd_16x8_avx2);
  out[BLOCK_16X32 as usize] = Some(rav1e_satd_16x32_avx2);
  out[BLOCK_32X16 as usize] = Some(rav1e_satd_32x16_avx2);
  out[BLOCK_32X64 as usize] = Some(rav1e_satd_32x64_avx2);
  out[BLOCK_64X32 as usize] = Some(rav1e_satd_64x32_avx2);
  out[BLOCK_64X128 as usize] = Some(rav1e_satd_64x128_avx2);
  out[BLOCK_128X64 as usize] = Some(rav1e_satd_128x64_avx2);

  out[BLOCK_4X16 as usize] = Some(rav1e_satd_4x16_avx2);
  out[BLOCK_16X4 as usize] = Some(rav1e_satd_16x4_avx2);
  out[BLOCK_8X32 as usize] = Some(rav1e_satd_8x32_avx2);
  out[BLOCK_32X8 as usize] = Some(rav1e_satd_32x8_avx2);
  out[BLOCK_16X64 as usize] = Some(rav1e_satd_16x64_avx2);
  out[BLOCK_64X16 as usize] = Some(rav1e_satd_64x16_avx2);

  out
};

cpu_function_lookup_table!(
  SATD_FNS: [[Option<SatdFn>; DIST_FNS_LENGTH]],
  default: [None; DIST_FNS_LENGTH],
  [SSSE3, SSE4_1, AVX2]
);

static SATD_HBD_FNS_AVX2: [Option<SatdHBDFn>; DIST_FNS_LENGTH] = {
  let mut out: [Option<SatdHBDFn>; DIST_FNS_LENGTH] = [None; DIST_FNS_LENGTH];

  use BlockSize::*;

  out[BLOCK_4X4 as usize] = Some(rav1e_satd_4x4_hbd_avx2);
  out[BLOCK_8X8 as usize] = Some(rav1e_satd_8x8_hbd_avx2);
  out[BLOCK_16X16 as usize] = Some(rav1e_satd_16x16_hbd_avx2);
  out[BLOCK_32X32 as usize] = Some(rav1e_satd_32x32_hbd_avx2);
  out[BLOCK_64X64 as usize] = Some(rav1e_satd_64x64_hbd_avx2);
  out[BLOCK_128X128 as usize] = Some(rav1e_satd_128x128_hbd_avx2);

  out[BLOCK_4X8 as usize] = Some(rav1e_satd_4x8_hbd_avx2);
  out[BLOCK_8X4 as usize] = Some(rav1e_satd_8x4_hbd_avx2);
  out[BLOCK_8X16 as usize] = Some(rav1e_satd_8x16_hbd_avx2);
  out[BLOCK_16X8 as usize] = Some(rav1e_satd_16x8_hbd_avx2);
  out[BLOCK_16X32 as usize] = Some(rav1e_satd_16x32_hbd_avx2);
  out[BLOCK_32X16 as usize] = Some(rav1e_satd_32x16_hbd_avx2);
  out[BLOCK_32X64 as usize] = Some(rav1e_satd_32x64_hbd_avx2);
  out[BLOCK_64X32 as usize] = Some(rav1e_satd_64x32_hbd_avx2);
  out[BLOCK_64X128 as usize] = Some(rav1e_satd_64x128_hbd_avx2);
  out[BLOCK_128X64 as usize] = Some(rav1e_satd_128x64_hbd_avx2);

  out[BLOCK_4X16 as usize] = Some(rav1e_satd_4x16_hbd_avx2);
  out[BLOCK_16X4 as usize] = Some(rav1e_satd_16x4_hbd_avx2);
  out[BLOCK_8X32 as usize] = Some(rav1e_satd_8x32_hbd_avx2);
  out[BLOCK_32X8 as usize] = Some(rav1e_satd_32x8_hbd_avx2);
  out[BLOCK_16X64 as usize] = Some(rav1e_satd_16x64_hbd_avx2);
  out[BLOCK_64X16 as usize] = Some(rav1e_satd_64x16_hbd_avx2);

  out
};

cpu_function_lookup_table!(
  SATD_HBD_FNS: [[Option<SatdHBDFn>; DIST_FNS_LENGTH]],
  default: [None; DIST_FNS_LENGTH],
  [AVX2]
);

#[cfg(test)]
mod test {
  use super::*;
  use crate::frame::{AsRegion, Plane};
  use rand::random;
  use std::str::FromStr;

  macro_rules! test_dist_fns {
    ($(($W:expr, $H:expr)),*, $DIST_TY:ident, $BD:expr, $OPT:ident, $OPTLIT:tt) => {
      $(
        paste::item! {
          #[test]
          fn [<get_ $DIST_TY _ $W x $H _bd_ $BD _ $OPT>]() {
            if !is_x86_feature_detected!($OPTLIT) {
              eprintln!("Ignoring {} test, not supported on this machine!", $OPTLIT);
              return;
            }

            if $BD > 8 {
              // dynamic allocation: test
              let mut src = Plane::from_slice(&[0u16; $W * $H], $W);
              // dynamic allocation: test
              let mut dst = Plane::from_slice(&[0u16; $W * $H], $W);
              for (s, d) in src.data.iter_mut().zip(dst.data.iter_mut()) {
                *s = random::<u8>() as u16 * $BD / 8;
                *d = random::<u8>() as u16 * $BD / 8;
              }
              let result = [<get_ $DIST_TY>](&src.as_region(), &dst.as_region(), $BD, CpuFeatureLevel::from_str($OPTLIT).unwrap());
              let rust_result = [<get_ $DIST_TY>](&src.as_region(), &dst.as_region(), $BD, CpuFeatureLevel::RUST);

              assert_eq!(rust_result, result);
            } else {
              // dynamic allocation: test
              let mut src = Plane::from_slice(&[0u8; $W * $H], $W);
              // dynamic allocation: test
              let mut dst = Plane::from_slice(&[0u8; $W * $H], $W);
              for (s, d) in src.data.iter_mut().zip(dst.data.iter_mut()) {
                *s = random::<u8>();
                *d = random::<u8>();
              }
              let result = [<get_ $DIST_TY>](&src.as_region(), &dst.as_region(), $BD, CpuFeatureLevel::from_str($OPTLIT).unwrap());
              let rust_result = [<get_ $DIST_TY>](&src.as_region(), &dst.as_region(),  $BD, CpuFeatureLevel::RUST);

              assert_eq!(rust_result, result);
            }
          }
        }
      )*
    }
  }

  test_dist_fns!(
    (4, 4),
    (16, 16),
    (8, 8),
    (4, 8),
    (8, 4),
    (8, 16),
    (16, 8),
    (4, 16),
    (16, 4),
    (8, 32),
    (32, 8),
    (32, 32),
    (64, 64),
    (128, 128),
    (16, 32),
    (32, 16),
    (32, 64),
    (64, 32),
    (64, 128),
    (128, 64),
    (16, 64),
    (64, 16),
    sad,
    10,
    ssse3,
    "ssse3"
  );

  test_dist_fns!(
    (4, 4),
    (16, 16),
    (8, 8),
    (4, 8),
    (8, 4),
    (8, 16),
    (16, 8),
    (4, 16),
    (16, 4),
    (8, 32),
    (32, 8),
    (32, 32),
    (64, 64),
    (128, 128),
    (16, 32),
    (32, 16),
    (32, 64),
    (64, 32),
    (64, 128),
    (128, 64),
    (16, 64),
    (64, 16),
    sad,
    10,
    avx2,
    "avx2"
  );

  test_dist_fns!(
    (4, 4),
    (4, 8),
    (4, 16),
    (8, 4),
    (8, 8),
    (8, 16),
    (8, 32),
    (16, 16),
    (32, 32),
    (64, 64),
    (128, 128),
    sad,
    8,
    sse2,
    "sse2"
  );

  test_dist_fns!(
    (16, 4),
    (16, 8),
    (16, 16),
    (16, 32),
    (16, 64),
    (32, 8),
    (32, 16),
    (32, 32),
    (32, 64),
    (64, 16),
    (64, 32),
    (64, 64),
    (64, 128),
    (128, 64),
    (128, 128),
    sad,
    8,
    avx2,
    "avx2"
  );

  test_dist_fns!((8, 8), satd, 8, ssse3, "ssse3");

  test_dist_fns!((4, 4), satd, 8, sse4, "sse4.1");

  test_dist_fns!(
    (4, 4),
    (8, 8),
    (16, 16),
    (32, 32),
    (64, 64),
    (128, 128),
    (4, 8),
    (8, 4),
    (8, 16),
    (16, 8),
    (16, 32),
    (32, 16),
    (32, 64),
    (64, 32),
    (64, 128),
    (128, 64),
    (4, 16),
    (16, 4),
    (8, 32),
    (32, 8),
    (16, 64),
    (64, 16),
    satd,
    8,
    avx2,
    "avx2"
  );

  test_dist_fns!(
    (4, 4),
    (8, 8),
    (16, 16),
    (32, 32),
    (64, 64),
    (128, 128),
    (4, 8),
    (8, 4),
    (8, 16),
    (16, 8),
    (16, 32),
    (32, 16),
    (32, 64),
    (64, 32),
    (64, 128),
    (128, 64),
    (4, 16),
    (16, 4),
    (8, 32),
    (32, 8),
    (16, 64),
    (64, 16),
    satd,
    10,
    avx2,
    "avx2"
  );

  test_dist_fns!(
    (4, 4),
    (8, 8),
    (16, 16),
    (32, 32),
    (64, 64),
    (128, 128),
    (4, 8),
    (8, 4),
    (8, 16),
    (16, 8),
    (16, 32),
    (32, 16),
    (32, 64),
    (64, 32),
    (64, 128),
    (128, 64),
    (4, 16),
    (16, 4),
    (8, 32),
    (32, 8),
    (16, 64),
    (64, 16),
    satd,
    12,
    avx2,
    "avx2"
  );
}
