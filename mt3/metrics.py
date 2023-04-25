# Copyright 2023 The MT3 Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Transcription metrics."""

import collections
import copy
import functools
from typing import Any, Iterable, Mapping, Optional, Sequence

import mir_eval

from mt3 import event_codec
from mt3 import metrics_utils
from mt3 import note_sequences
from mt3 import spectrograms
from mt3 import summaries
from mt3 import vocabularies

import note_seq
import numpy as np
import seqio


def _program_aware_note_scores(
    ref_ns: note_seq.NoteSequence,
    est_ns: note_seq.NoteSequence,
    granularity_type: str
) -> Mapping[str, float]:
  """Compute precision/recall/F1 for notes taking program into account.

  For non-drum tracks, uses onsets and offsets. For drum tracks, uses onsets
  only. Applies MIDI program map of specified granularity type.

  Args:
    ref_ns: Reference NoteSequence with ground truth labels.
    est_ns: Estimated NoteSequence.
    granularity_type: String key in vocabularies.PROGRAM_GRANULARITIES dict.

  Returns:
    A dictionary containing precision, recall, and F1 score.
  """
  program_map_fn = vocabularies.PROGRAM_GRANULARITIES[
      granularity_type].program_map_fn

  ref_ns = copy.deepcopy(ref_ns)
  for note in ref_ns.notes:
    if not note.is_drum:
      note.program = program_map_fn(note.program)

  est_ns = copy.deepcopy(est_ns)
  for note in est_ns.notes:
    if not note.is_drum:
      note.program = program_map_fn(note.program)

  program_and_is_drum_tuples = (
      set((note.program, note.is_drum) for note in ref_ns.notes) |
      set((note.program, note.is_drum) for note in est_ns.notes)
  )

  drum_precision_sum = 0.0
  drum_precision_count = 0
  drum_recall_sum = 0.0
  drum_recall_count = 0

  nondrum_precision_sum = 0.0
  nondrum_precision_count = 0
  nondrum_recall_sum = 0.0
  nondrum_recall_count = 0

  for program, is_drum in program_and_is_drum_tuples:
    est_track = note_sequences.extract_track(est_ns, program, is_drum)
    ref_track = note_sequences.extract_track(ref_ns, program, is_drum)

    est_intervals, est_pitches, unused_est_velocities = (
        note_seq.sequences_lib.sequence_to_valued_intervals(est_track))
    ref_intervals, ref_pitches, unused_ref_velocities = (
        note_seq.sequences_lib.sequence_to_valued_intervals(ref_track))

    args = {
        'ref_intervals': ref_intervals, 'ref_pitches': ref_pitches,
        'est_intervals': est_intervals, 'est_pitches': est_pitches
    }
    if is_drum:
      args['offset_ratio'] = None

    precision, recall, unused_f_measure, unused_avg_overlap_ratio = (
        mir_eval.transcription.precision_recall_f1_overlap(**args))

    if is_drum:
      drum_precision_sum += precision * len(est_intervals)
      drum_precision_count += len(est_intervals)
      drum_recall_sum += recall * len(ref_intervals)
      drum_recall_count += len(ref_intervals)
    else:
      nondrum_precision_sum += precision * len(est_intervals)
      nondrum_precision_count += len(est_intervals)
      nondrum_recall_sum += recall * len(ref_intervals)
      nondrum_recall_count += len(ref_intervals)

  precision_sum = drum_precision_sum + nondrum_precision_sum
  precision_count = drum_precision_count + nondrum_precision_count
  recall_sum = drum_recall_sum + nondrum_recall_sum
  recall_count = drum_recall_count + nondrum_recall_count

  precision = (precision_sum / precision_count) if precision_count else 0
  recall = (recall_sum / recall_count) if recall_count else 0
  f_measure = mir_eval.util.f_measure(precision, recall)

  drum_precision = ((drum_precision_sum / drum_precision_count)
                    if drum_precision_count else 0)
  drum_recall = ((drum_recall_sum / drum_recall_count)
                 if drum_recall_count else 0)
  drum_f_measure = mir_eval.util.f_measure(drum_precision, drum_recall)

  nondrum_precision = ((nondrum_precision_sum / nondrum_precision_count)
                       if nondrum_precision_count else 0)
  nondrum_recall = ((nondrum_recall_sum / nondrum_recall_count)
                    if nondrum_recall_count else 0)
  nondrum_f_measure = mir_eval.util.f_measure(nondrum_precision, nondrum_recall)

  return {
      f'Onset + offset + program precision ({granularity_type})': precision,
      f'Onset + offset + program recall ({granularity_type})': recall,
      f'Onset + offset + program F1 ({granularity_type})': f_measure,
      f'Drum onset precision ({granularity_type})': drum_precision,
      f'Drum onset recall ({granularity_type})': drum_recall,
      f'Drum onset F1 ({granularity_type})': drum_f_measure,
      f'Nondrum onset + offset + program precision ({granularity_type})':
          nondrum_precision,
      f'Nondrum onset + offset + program recall ({granularity_type})':
          nondrum_recall,
      f'Nondrum onset + offset + program F1 ({granularity_type})':
          nondrum_f_measure
  }


def _note_onset_tolerance_sweep(
    ref_ns: note_seq.NoteSequence, est_ns: note_seq.NoteSequence,
    tolerances: Iterable[float] = (0.01, 0.02, 0.05, 0.1, 0.2, 0.5)
) -> Mapping[str, float]:
  """Compute note precision/recall/F1 across a range of tolerances."""
  est_intervals, est_pitches, unused_est_velocities = (
      note_seq.sequences_lib.sequence_to_valued_intervals(est_ns))
  ref_intervals, ref_pitches, unused_ref_velocities = (
      note_seq.sequences_lib.sequence_to_valued_intervals(ref_ns))

  scores = {}

  for tol in tolerances:
    precision, recall, f_measure, _ = (
        mir_eval.transcription.precision_recall_f1_overlap(
            ref_intervals=ref_intervals, ref_pitches=ref_pitches,
            est_intervals=est_intervals, est_pitches=est_pitches,
            onset_tolerance=tol, offset_min_tolerance=tol))

    scores[f'Onset + offset precision ({tol})'] = precision
    scores[f'Onset + offset recall ({tol})'] = recall
    scores[f'Onset + offset F1 ({tol})'] = f_measure

  return scores


def transcription_metrics(
    targets: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
    codec: event_codec.Codec,
    spectrogram_config: spectrograms.SpectrogramConfig,
    onsets_only: bool,
    use_ties: bool,
    track_specs: Optional[Sequence[note_sequences.TrackSpec]] = None,
    num_summary_examples: int = 5,
    frame_fps: float = 62.5,
    frame_velocity_threshold: int = 30,
) -> Mapping[str, seqio.metrics.MetricValue]:
  """Compute mir_eval transcription metrics."""
  if onsets_only and use_ties:
    raise ValueError('Ties not compatible with onset-only transcription.')
  if onsets_only:
    encoding_spec = note_sequences.NoteOnsetEncodingSpec
  elif not use_ties:
    encoding_spec = note_sequences.NoteEncodingSpec
  else:
    encoding_spec = note_sequences.NoteEncodingWithTiesSpec

  # The first target for each full example contains the NoteSequence; just
  # organize by ID.
  full_targets = {}
  for target in targets:
    if target['ref_ns']:
      full_targets[target['unique_id']] = {'ref_ns': target['ref_ns']}

  # Gather all predictions for the same ID and concatenate them in time order,
  # to construct full-length predictions.
  full_predictions = metrics_utils.combine_predictions_by_id(
      predictions=predictions,
      combine_predictions_fn=functools.partial(
          metrics_utils.event_predictions_to_ns,
          codec=codec,
          encoding_spec=encoding_spec))

  assert sorted(full_targets.keys()) == sorted(full_predictions.keys())

  full_target_prediction_pairs = [
      (full_targets[id], full_predictions[id])
      for id in sorted(full_targets.keys())
  ]

  scores = collections.defaultdict(list)
  all_track_pianorolls = collections.defaultdict(list)
  for target, prediction in full_target_prediction_pairs:
    scores['Invalid events'].append(prediction['est_invalid_events'])
    scores['Dropped events'].append(prediction['est_dropped_events'])

    def remove_drums(ns):
      ns_drumless = note_seq.NoteSequence()
      ns_drumless.CopyFrom(ns)
      del ns_drumless.notes[:]
      ns_drumless.notes.extend([note for note in ns.notes if not note.is_drum])
      return ns_drumless

    est_ns_drumless = remove_drums(prediction['est_ns'])
    ref_ns_drumless = remove_drums(target['ref_ns'])

    # Whether or not there are separate tracks, compute metrics for the full
    # NoteSequence minus drums.
    est_tracks = [est_ns_drumless]
    ref_tracks = [ref_ns_drumless]
    use_track_offsets = [not onsets_only]
    use_track_velocities = [not onsets_only]
    track_instrument_names = ['']

    if track_specs is not None:
      # Compute transcription metrics separately for each track.
      for spec in track_specs:
        est_tracks.append(note_sequences.extract_track(
            prediction['est_ns'], spec.program, spec.is_drum))
        ref_tracks.append(note_sequences.extract_track(
            target['ref_ns'], spec.program, spec.is_drum))
        use_track_offsets.append(not onsets_only and not spec.is_drum)
        use_track_velocities.append(not onsets_only)
        track_instrument_names.append(spec.name)

    for est_ns, ref_ns, use_offsets, use_velocities, instrument_name in zip(
        est_tracks, ref_tracks, use_track_offsets, use_track_velocities,
        track_instrument_names):
      track_scores = {}

      est_intervals, est_pitches, est_velocities = (
          note_seq.sequences_lib.sequence_to_valued_intervals(est_ns))

      ref_intervals, ref_pitches, ref_velocities = (
          note_seq.sequences_lib.sequence_to_valued_intervals(ref_ns))

      # Precision / recall / F1 using onsets (and pitches) only.
      precision, recall, f_measure, avg_overlap_ratio = (
          mir_eval.transcription.precision_recall_f1_overlap(
              ref_intervals=ref_intervals,
              ref_pitches=ref_pitches,
              est_intervals=est_intervals,
              est_pitches=est_pitches,
              offset_ratio=None))
      del avg_overlap_ratio
      track_scores['Onset precision'] = precision
      track_scores['Onset recall'] = recall
      track_scores['Onset F1'] = f_measure

      if use_offsets:
        # Precision / recall / F1 using onsets and offsets.
        precision, recall, f_measure, avg_overlap_ratio = (
            mir_eval.transcription.precision_recall_f1_overlap(
                ref_intervals=ref_intervals,
                ref_pitches=ref_pitches,
                est_intervals=est_intervals,
                est_pitches=est_pitches))
        del avg_overlap_ratio
        track_scores['Onset + offset precision'] = precision
        track_scores['Onset + offset recall'] = recall
        track_scores['Onset + offset F1'] = f_measure

      if use_velocities:
        # Precision / recall / F1 using onsets and velocities (no offsets).
        precision, recall, f_measure, avg_overlap_ratio = (
            mir_eval.transcription_velocity.precision_recall_f1_overlap(
                ref_intervals=ref_intervals,
                ref_pitches=ref_pitches,
                ref_velocities=ref_velocities,
                est_intervals=est_intervals,
                est_pitches=est_pitches,
                est_velocities=est_velocities,
                offset_ratio=None))
        track_scores['Onset + velocity precision'] = precision
        track_scores['Onset + velocity recall'] = recall
        track_scores['Onset + velocity F1'] = f_measure

      if use_offsets and use_velocities:
        # Precision / recall / F1 using onsets, offsets, and velocities.
        precision, recall, f_measure, avg_overlap_ratio = (
            mir_eval.transcription_velocity.precision_recall_f1_overlap(
                ref_intervals=ref_intervals,
                ref_pitches=ref_pitches,
                ref_velocities=ref_velocities,
                est_intervals=est_intervals,
                est_pitches=est_pitches,
                est_velocities=est_velocities))
        track_scores['Onset + offset + velocity precision'] = precision
        track_scores['Onset + offset + velocity recall'] = recall
        track_scores['Onset + offset + velocity F1'] = f_measure

      # Calculate framewise metrics.
      is_drum = all([n.is_drum for n in ref_ns.notes])
      ref_pr = metrics_utils.get_prettymidi_pianoroll(
          ref_ns, frame_fps, is_drum=is_drum)
      est_pr = metrics_utils.get_prettymidi_pianoroll(
          est_ns, frame_fps, is_drum=is_drum)
      all_track_pianorolls[instrument_name].append((est_pr, ref_pr))
      frame_precision, frame_recall, frame_f1 = metrics_utils.frame_metrics(
          ref_pr, est_pr, velocity_threshold=frame_velocity_threshold)
      track_scores['Frame Precision'] = frame_precision
      track_scores['Frame Recall'] = frame_recall
      track_scores['Frame F1'] = frame_f1

      for metric_name, metric_value in track_scores.items():
        if instrument_name:
          scores[f'{instrument_name}/{metric_name}'].append(metric_value)
        else:
          scores[metric_name].append(metric_value)

    # Add program-aware note metrics for all program granularities.
    # Note that this interacts with the training program granularity; in
    # particular granularities *higher* than the training granularity are likely
    # to have poor metrics.
    for granularity_type in vocabularies.PROGRAM_GRANULARITIES:
      for name, score in _program_aware_note_scores(
          target['ref_ns'], prediction['est_ns'],
          granularity_type=granularity_type).items():
        scores[name].append(score)

    # Add (non-program-aware) note metrics across a range of onset/offset
    # tolerances.
    for name, score in _note_onset_tolerance_sweep(
        ref_ns=ref_ns_drumless, est_ns=est_ns_drumless).items():
      scores[name].append(score)

  mean_scores = {k: np.mean(v) for k, v in scores.items()}

  score_histograms = {'%s (hist)' % k: seqio.metrics.Histogram(np.array(v))
                      for k, v in scores.items()}

  # Pick several examples to summarize.
  targets_to_summarize, predictions_to_summarize = zip(
      *full_target_prediction_pairs[:num_summary_examples])

  # Compute audio summaries.
  audio_summaries = summaries.audio_summaries(
      targets=targets_to_summarize,
      predictions=predictions_to_summarize,
      spectrogram_config=spectrogram_config)

  # Compute transcription summaries.
  transcription_summaries = summaries.transcription_summaries(
      targets=targets_to_summarize,
      predictions=predictions_to_summarize,
      spectrogram_config=spectrogram_config,
      ns_feature_suffix='ns',
      track_specs=track_specs)

  pianorolls_to_summarize = {
      k: v[:num_summary_examples] for k, v in all_track_pianorolls.items()
  }

  prettymidi_pianoroll_summaries = summaries.prettymidi_pianoroll(
      pianorolls_to_summarize, fps=frame_fps)

  return {
      **mean_scores,
      **score_histograms,
      **audio_summaries,
      **transcription_summaries,
      **prettymidi_pianoroll_summaries,
  }
