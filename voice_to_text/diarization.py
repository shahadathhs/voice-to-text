"""Speaker diarization: SpeechBrain embeddings, sliding-window sub-segments, temporal smoothing."""

import os
from collections import Counter
from typing import Any

import numpy as np

from voice_to_text.config import (
    DIARIZE_SAMPLE_RATE,
    MIN_CHUNK_MS,
    MIN_SEGMENT_MS,
    SMOOTHING_MAX_DURATION_S,
    SUBSEGMENT_MIN_DURATION_S,
    SUBSEGMENT_STRIDE_S,
    SUBSEGMENT_WINDOW_S,
)


def overlap(s1: float, e1: float, s2: float, e2: float) -> float:
    """Overlap duration between intervals [s1,e1] and [s2,e2]."""
    return max(0.0, min(e1, e2) - max(s1, s2))


def assign_speaker_by_overlap(
    seg_start: float,
    seg_end: float,
    diarized_segments: list[dict[str, Any]],
) -> str:
    """Assign speaker to a segment by time overlap with diarized segments."""
    speaker_overlap: dict[str, float] = {}
    for d in diarized_segments:
        ov = overlap(seg_start, seg_end, d["start"], d["end"])
        if ov > 0:
            sp = d["speaker"]
            speaker_overlap[sp] = speaker_overlap.get(sp, 0) + ov
    if not speaker_overlap:
        return "SPEAKER_00"
    return max(speaker_overlap, key=speaker_overlap.get)


def _build_diarization_chunks(segments: list[dict[str, Any]]) -> list[tuple[float, float, int]]:
    """Build (start_s, end_s, segment_idx) for embedding extraction. Long segments use sliding windows."""
    chunks = []
    for i, seg in enumerate(segments):
        start_s, end_s = seg["start"], seg["end"]
        dur = end_s - start_s
        if dur >= SUBSEGMENT_MIN_DURATION_S:
            t = start_s
            while t + SUBSEGMENT_WINDOW_S <= end_s:
                chunks.append((t, t + SUBSEGMENT_WINDOW_S, i))
                t += SUBSEGMENT_STRIDE_S
            if t < end_s and (end_s - t) * 1000 >= MIN_CHUNK_MS:
                chunks.append((max(t, end_s - SUBSEGMENT_WINDOW_S), end_s, i))
        else:
            if dur * 1000 >= MIN_SEGMENT_MS:
                chunks.append((start_s, end_s, i))
    return chunks


def _temporal_smooth_labels(
    segments: list[dict[str, Any]],
    seg_label: dict[int, int],
    max_duration_s: float = SMOOTHING_MAX_DURATION_S,
) -> dict[int, int]:
    """Flip short segments that differ from both neighbors to reduce flickering."""
    n = len(segments)
    changed = True
    while changed:
        changed = False
        for i in range(n):
            if i not in seg_label:
                continue
            dur = segments[i]["end"] - segments[i]["start"]
            if dur > max_duration_s:
                continue
            prev_l = seg_label.get(i - 1) if i > 0 else None
            next_l = seg_label.get(i + 1) if i < n - 1 else None
            cur_l = seg_label[i]
            if prev_l is not None and next_l is not None and prev_l == next_l and cur_l != prev_l:
                seg_label[i] = prev_l
                changed = True
    return seg_label


def perform_diarization(
    audio_path: str,
    segments: list[dict[str, Any]],
    device: str,
    classifier: Any = None,
    distance_threshold: float = 0.35,
    max_speakers: int | None = None,
    use_silhouette: bool = False,
) -> list[dict[str, Any]] | None:
    """
    Diarization using SpeechBrain ECAPA embeddings and clustering.
    Long segments use sliding-window sub-segments and temporal smoothing.
    """
    print("[*] Extracting speaker embeddings for diarization...")
    try:
        import torch
        from pydub import AudioSegment
        from sklearn.cluster import AgglomerativeClustering
        from speechbrain.inference.speaker import EncoderClassifier
    except Exception as e:
        print(f"[!] Error loading diarization dependencies: {e}")
        return None

    if classifier is None:
        classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": device},
            savedir=os.path.join(os.path.expanduser("~"), ".cache", "speechbrain"),
        )

    audio = AudioSegment.from_file(audio_path)
    if audio.frame_rate != DIARIZE_SAMPLE_RATE:
        audio = audio.set_frame_rate(DIARIZE_SAMPLE_RATE)
    if audio.channels > 1:
        audio = audio.set_channels(1)

    chunks = _build_diarization_chunks(segments)
    embeddings = []
    chunk_meta = []

    for start_s, end_s, seg_idx in chunks:
        start_ms = int(start_s * 1000)
        end_ms = int(end_s * 1000)
        seg_audio = audio[start_ms:end_ms]
        if len(seg_audio) < MIN_CHUNK_MS:
            continue
        samples = np.array(seg_audio.get_array_of_samples()).astype(np.float32) / (2**15)
        signal = torch.from_numpy(samples).to(device)
        with torch.no_grad():
            emb = classifier.encode_batch(signal.unsqueeze(0))
            embeddings.append(emb.squeeze().cpu().numpy())
            chunk_meta.append((start_s, end_s, seg_idx))

    if not embeddings:
        return None

    embeddings = np.array(embeddings)
    n_emb = len(embeddings)

    if max_speakers is not None and max_speakers >= 1:
        n_clusters = min(max_speakers, n_emb)
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric="cosine",
            linkage="average",
        )
    elif use_silhouette and n_emb >= 4:
        from sklearn.metrics import silhouette_score
        best_k, best_score = 2, -1.0
        for k in range(2, min(11, n_emb)):
            c = AgglomerativeClustering(n_clusters=k, metric="cosine", linkage="average")
            labels = c.fit_predict(embeddings)
            if len(set(labels)) < k:
                continue
            sc = silhouette_score(embeddings, labels, metric="cosine")
            if sc > best_score:
                best_score = sc
                best_k = k
        clustering = AgglomerativeClustering(
            n_clusters=best_k,
            metric="cosine",
            linkage="average",
        )
    else:
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=distance_threshold,
            metric="cosine",
            linkage="average",
        )

    chunk_labels = clustering.fit_predict(embeddings)

    seg_label: dict[int, list[int]] = {}
    for (_, _, seg_idx), label in zip(chunk_meta, chunk_labels):
        if seg_idx not in seg_label:
            seg_label[seg_idx] = []
        seg_label[seg_idx].append(label)
    for seg_idx in seg_label:
        votes = seg_label[seg_idx]
        seg_label[seg_idx] = Counter(votes).most_common(1)[0][0]

    chunk_centers = [
        ((s + e) / 2, seg_idx, lab) for (s, e, seg_idx), lab in zip(chunk_meta, chunk_labels)
    ]
    for i in range(len(segments)):
        if i in seg_label:
            continue
        seg = segments[i]
        center_i = (seg["start"] + seg["end"]) / 2
        best_label = 0
        best_dist = float("inf")
        for (center_j, _, label) in chunk_centers:
            d = abs(center_i - center_j)
            if d < best_dist:
                best_dist = d
                best_label = label
        seg_label[i] = best_label

    seg_label = _temporal_smooth_labels(segments, seg_label)

    num_speakers = max(seg_label.values()) + 1
    diarization_map = []
    for i, seg in enumerate(segments):
        label = seg_label.get(i, 0)
        diarization_map.append({
            "start": seg["start"],
            "end": seg["end"],
            "speaker": f"SPEAKER_{label:02d}",
            "text": seg["text"].strip(),
        })

    print(f"[*] Diarization: {num_speakers} speaker(s) across {len(segments)} segments")
    return diarization_map
