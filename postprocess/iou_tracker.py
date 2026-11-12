"""The simple IoU tracker that ObjectTracker uses (and falls back to when ByteTrack/SORT cannot be created).

Greedy association, in track-creation order: each live track takes the unmatched detection of the same class with the
highest IoU against its last box, if that IoU is above match_thresh. Unmatched detections start new tracks. A track is
live while (frame_count - last_frame) <= track_buffer. Ported to the web front end as track_engine.js.
"""


def box_iou(box1, box2):
    """Intersection over union of two [x1, y1, x2, y2] boxes (0.0 when they do not overlap)."""
    x1_1, y1_1, x2_1, y2_1 = box1[:4]
    x1_2, y1_2, x2_2, y2_2 = box2[:4]
    x_left, y_top = max(x1_1, x1_2), max(y1_1, y1_2)
    x_right, y_bottom = min(x2_1, x2_2), min(y2_1, y2_2)
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = box1_area + box2_area - intersection_area
    return intersection_area / union_area if union_area > 0 else 0.0


def associate_simple(tracks, boxes, scores, classes, frame_count, next_id, track_buffer, match_thresh):
    """Update `tracks` ({id: {boxes, class, first_frame, last_frame, ...}}) in place with one frame of detections.

    Returns (active_track_ids, next_id).
    """
    if len(boxes) == 0:
        return [], next_id
    active_tracks = []
    matched_detections = set()
    for track_id, track in tracks.items():
        if (frame_count - track['last_frame']) > track_buffer or not track['boxes']:
            continue
        last_box = track['boxes'][-1]
        best_iou, best_idx = match_thresh, -1
        for i, box in enumerate(boxes):
            if i in matched_detections or track['class'] != classes[i]:
                continue
            iou = box_iou(last_box, box)
            if iou > best_iou:
                best_iou, best_idx = iou, i
        if best_idx >= 0:
            matched_detections.add(best_idx)
            track['boxes'].append(boxes[best_idx])
            track['last_frame'] = frame_count
            active_tracks.append(track_id)
    for i, box in enumerate(boxes):
        if i not in matched_detections:
            tracks[next_id] = {
                'boxes': [box],
                'class': int(classes[i]),
                'score': float(scores[i]),
                'first_frame': frame_count,
                'last_frame': frame_count,
            }
            active_tracks.append(next_id)
            next_id += 1
    return active_tracks, next_id
