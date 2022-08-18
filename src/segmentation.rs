// Copyright (c) 2018-2021, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

use std::cmp::Ordering;

use crate::context::*;
use crate::header::PRIMARY_REF_NONE;
use crate::partition::BlockSize;
use crate::rdo::spatiotemporal_scale;
use crate::rdo::DistortionScale;
use crate::tiling::TileStateMut;
use crate::util::Pixel;
use crate::FrameInvariants;
use crate::FrameState;

pub const MAX_SEGMENTS: usize = 8;

// This value was obtained through testing over multiple runs in AWCY.
const BASE_AQ_MULT: f64 = -9.0;

pub fn segmentation_optimize<T: Pixel>(
  fi: &FrameInvariants<T>, fs: &mut FrameState<T>,
) {
  assert!(fi.enable_segmentation);
  fs.segmentation.enabled = true;

  if fs.segmentation.enabled {
    fs.segmentation.update_map = true;

    // We don't change the values between frames.
    fs.segmentation.update_data = fi.primary_ref_frame == PRIMARY_REF_NONE;

    // Avoid going into lossless mode by never bringing qidx below 1.
    // Because base_q_idx changes more frequently than the segmentation
    // data, it is still possible for a segment to enter lossless, so
    // enforcement elsewhere is needed.
    let offset_lower_limit = 1 - fi.base_q_idx as i16;

    if !fs.segmentation.update_data {
      let mut min_segment = MAX_SEGMENTS;
      for i in 0..MAX_SEGMENTS {
        if fs.segmentation.features[i][SegLvl::SEG_LVL_ALT_Q as usize]
          && fs.segmentation.data[i][SegLvl::SEG_LVL_ALT_Q as usize]
            >= offset_lower_limit
        {
          min_segment = i;
          break;
        }
      }
      assert_ne!(min_segment, MAX_SEGMENTS);
      fs.segmentation.min_segment = min_segment as u8;
      fs.segmentation.update_threshold(fi.base_q_idx, fi.config.bit_depth);
      return;
    }

    segmentation_optimize_inner(fi, fs, offset_lower_limit);

    /* Figure out parameters */
    fs.segmentation.preskip = false;
    fs.segmentation.last_active_segid = 0;
    for i in 0..MAX_SEGMENTS {
      for j in 0..SegLvl::SEG_LVL_MAX as usize {
        if fs.segmentation.features[i][j] {
          fs.segmentation.last_active_segid = i as u8;
          if j >= SegLvl::SEG_LVL_REF_FRAME as usize {
            fs.segmentation.preskip = true;
          }
        }
      }
    }
  }
}

fn segmentation_optimize_inner<T: Pixel>(
  fi: &FrameInvariants<T>, fs: &mut FrameState<T>, offset_lower_limit: i16,
) {
  const AVG_SEG: f64 = 3.0;

  let coded_data = fi.coded_frame_data.as_ref().unwrap();
  let segments: Box<[_]> = coded_data
    .spatiotemporal_scores
    .iter()
    .map(|&s| segment_idx_from_distortion(&INIT_THRESHOLD, s))
    .collect();
  let mut seg_counts = [0usize; MAX_SEGMENTS];
  segments.iter().for_each(|&seg| {
    seg_counts[seg as usize % 8] += 1;
  });

  let mut num_neg = 0usize;
  let mut num_pos = 0usize;
  let mut tmp_delta = [0f64; MAX_SEGMENTS];
  for i in 0..MAX_SEGMENTS {
    tmp_delta[i] = (AVG_SEG - i as f64) * BASE_AQ_MULT * fi.config.aq_strength;
    if tmp_delta[i] > 0f64 {
      num_pos += 1;
    } else if tmp_delta[i] < 0f64 {
      num_neg += 1;
    }
  }

  // We want at least 1/12 of the blocks in a segment in order to code it.
  // In most cases this results in 3-4 segments being coded, which is optimal.
  let threshold = segments.len() / 12;

  let mut remap_segment_tab: [usize; MAX_SEGMENTS] = [0, 1, 2, 3, 4, 5, 6, 7];
  let mut num_segments = MAX_SEGMENTS;

  loop {
    let mut changed = false;

    if num_segments < 4 {
      break;
    }

    for i in (0..MAX_SEGMENTS).rev() {
      if seg_counts[remap_segment_tab[i]] >= threshold {
        continue;
      };
      if seg_counts[remap_segment_tab[i]] == 0 {
        continue;
      }; /* Already eliminated */

      let prev_id = remap_segment_tab[i];

      #[derive(Debug, Default, Clone, Copy)]
      struct ScoreTab {
        idx: usize,
        score: f64,
      }
      let mut s_array =
        [ScoreTab { idx: usize::max_value(), score: std::f64::MAX };
          MAX_SEGMENTS];

      for j in 0..MAX_SEGMENTS {
        s_array[j].idx = remap_segment_tab[j];
        if (remap_segment_tab[j] == prev_id)
          || (seg_counts[remap_segment_tab[j]] == 0)
          || (((num_neg < 2) || (num_pos < 2))
            && (tmp_delta[remap_segment_tab[j]].signum()
              != tmp_delta[prev_id].signum()))
        {
          s_array[j].score = std::f64::MAX;
        } else {
          s_array[j].score =
            (tmp_delta[remap_segment_tab[j]] - tmp_delta[prev_id]).abs();
        }
      }

      s_array.sort_by(|a, b| {
        (a.score).partial_cmp(&b.score).unwrap_or(Ordering::Less)
      });

      if s_array[0].score == std::f64::MAX {
        continue;
      }

      /* Remap any old mappings to the current segment as well */
      for j in 0..MAX_SEGMENTS {
        if remap_segment_tab[j] == prev_id {
          remap_segment_tab[j] = s_array[0].idx;
        }
      }

      let num_2bins = seg_counts[remap_segment_tab[i]] + seg_counts[prev_id];
      let mut ratio_new =
        (seg_counts[remap_segment_tab[i]] as f64) / (num_2bins as f64);
      let mut ratio_old = (seg_counts[prev_id] as f64) / (num_2bins as f64);

      ratio_new *= tmp_delta[remap_segment_tab[i]];
      ratio_old *= tmp_delta[prev_id];

      num_pos -= (tmp_delta[prev_id] > 0f64) as usize;
      num_neg -= (tmp_delta[prev_id] < 0f64) as usize;

      tmp_delta[remap_segment_tab[i]] = ratio_new + ratio_old;
      tmp_delta[prev_id] = std::f64::MAX;

      seg_counts[remap_segment_tab[i]] += seg_counts[prev_id];
      seg_counts[prev_id] = 0;

      num_segments -= 1;

      changed = true;
      break;
    }

    if !changed {
      break;
    }
  }

  tmp_delta.iter_mut().for_each(|delta| {
    if *delta > 0.0 {
      // We want the strength of the bitrate reduction to be
      // less than the strength of the bitrate increase.
      *delta *= 0.8;
    }
  });

  /* Get all unique values in the intentionally unsorted array (its a LUT) */
  let mut uniq_array = [0usize; MAX_SEGMENTS];
  let mut num_segments = 0;
  for i in 0..MAX_SEGMENTS {
    let mut seen_match = false;
    for j in 0..num_segments {
      if remap_segment_tab[i] == uniq_array[j] {
        seen_match = true;
      }
    }
    if !seen_match {
      uniq_array[num_segments] = remap_segment_tab[i];
      num_segments += 1;
    }
  }

  let mut seg_delta = [0f64; MAX_SEGMENTS];
  for i in 0..num_segments {
    /* Collect all used segment deltas into the actual segment map */
    seg_delta[i] = tmp_delta[uniq_array[i]];

    /* Remap the LUT to make it match the layout of the seg deltaq map */
    for j in 0..MAX_SEGMENTS {
      if remap_segment_tab[j] == uniq_array[i] {
        remap_segment_tab[j] = i;
      }
    }
  }

  fs.segmentation.max_segment = num_segments as u8 - 1;
  for i in 0..num_segments {
    fs.segmentation.features[i][SegLvl::SEG_LVL_ALT_Q as usize] = true;
    fs.segmentation.data[i][SegLvl::SEG_LVL_ALT_Q as usize] =
      (seg_delta[i].round() as i16).max(offset_lower_limit);
  }

  fs.segmentation.update_threshold(fi.base_q_idx, fi.config.bit_depth);
}

pub fn select_segment<T: Pixel>(
  fi: &FrameInvariants<T>, ts: &TileStateMut<'_, T>, tile_bo: TileBlockOffset,
  bsize: BlockSize, skip: bool,
) -> std::ops::RangeInclusive<u8> {
  // If skip is true or segmentation is turned off, sidx is not coded.
  if skip || !fi.enable_segmentation {
    return 0..=0;
  }

  use crate::api::SegmentationLevel;
  if fi.config.speed_settings.segmentation == SegmentationLevel::Full {
    return ts.segmentation.min_segment..=ts.segmentation.max_segment;
  }

  let frame_bo = ts.to_frame_block_offset(tile_bo);
  let scale = spatiotemporal_scale(fi, frame_bo, bsize);

  let sidx = segment_idx_from_distortion(&ts.segmentation.threshold, scale);

  // Avoid going into lossless mode by never bringing qidx below 1.
  let sidx = sidx.max(ts.segmentation.min_segment);

  sidx..=sidx
}

const fn d3(num: u64) -> DistortionScale {
  DistortionScale::new(num, 1000)
}

// These values were obtained based on thorough testing
// around the idea that the resulting encode filesize
// with dynamic segment maps should be, on average, the same as
// with the static segment maps.
// i.e. These values were found to improve perceptual quality
// without significantly impacting filesize.
static INIT_THRESHOLD: [DistortionScale; MAX_SEGMENTS - 1] =
  [d3(2_000), d3(1_500), d3(1_150), d3(900), d3(800), d3(700), d3(625)];

fn segment_idx_from_distortion(
  threshold: &[DistortionScale; MAX_SEGMENTS - 1], s: DistortionScale,
) -> u8 {
  threshold.partition_point(|&t| s.0 < t.0) as u8
}
