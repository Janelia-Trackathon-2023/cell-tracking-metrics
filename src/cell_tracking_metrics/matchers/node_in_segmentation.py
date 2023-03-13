import numpy as np

from cell_tracking_metrics import TrackingData


def nearest_pred_in_segmentation(gt, pred):
    """Identifies pairs of cells between gt and pred where the locations
    in the graph lie within the same ground truth segmentation.
    Then matches the gt node to the one prediction nearest in euclidean distance to the
    gt graph node.


    Args:
        gt (TrackingData): Tracking data object containing graph and segmentations
        pred (TrackingData): Tracking data object containing a graph

    Returns:
        list[(gt_node, pred_node)]: list of tuples where each tuple contains a gt node and pred node

    Raises:
        ValueError: gt and pred must be a TrackingData object
        ValueError: pred node locations must be inside gt segmentation shape
    """
    if not isinstance(gt, TrackingData) or not isinstance(pred, TrackingData):
        raise ValueError(
            "Input data must be a TrackingData object with a graph and segmentations"
        )

    gt_graph, mask_gt = gt.tracking_graph, gt.segmentation
    pred_graph, mask_pred = pred.tracking_graph, pred.segmentation

    matches = []
    # Get overlaps for each frame
    for i, t in enumerate(
        range(gt.tracking_graph.start_frame, gt.tracking_graph.end_frame)
    ):
        seg = mask_gt[i]
        print(seg.shape)
        pred_nodes = pred_graph.get_nodes_in_frame(t)
        seg_to_pred = {}
        for node in pred_nodes:
            loc = pred_graph.get_location(node)
            print(loc)
            seg_id = seg[tuple(loc)]
            print(seg_id)
            if seg_id not in seg_to_pred:
                seg_to_pred[seg_id] = []
            seg_to_pred[seg_id].append(node)
        gt_nodes = gt_graph.get_nodes_in_frame(t)
        seg_to_gt = {seg[tuple(gt_graph.get_location(node))]: node for node in gt_nodes}
        for seg_id, gt_node in seg_to_gt.items():
            gt_loc = gt_graph.get_location(gt_node)
            if seg_id in seg_to_pred:
                min_dist = float("inf")
                closest_node = None
                for pred_node in seg_to_pred[seg_id]:
                    pred_loc = pred_graph.get_location(pred_node)
                    distance = np.linalg.norm(gt_loc, pred_loc)
                    if distance < min_dist:
                        min_dist = distance
                        closest_node = pred_node
                matches.append((gt_node, closest_node))
    return matches
