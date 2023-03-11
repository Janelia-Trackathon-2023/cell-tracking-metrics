import networkx as nx
import numpy as np
from skimage.measure import regionprops
from tqdm import tqdm

from cell_tracking_metrics.matchers.compute_overlap import compute_overlap
from cell_tracking_metrics.tracking_data import TrackingData

def match_ctc(gt, pred, label_key="segmentation_id"):
    """Match graph nodes based on measure used in cell tracking challenge benchmarking.

    A computed marker (segmentation) is matched to a reference marker if the computed 
    marker covers a majority of the reference marker.

    Each reference marker can therefore only be matched to one computed marker, but 
    multiple reference markers can be assigned to a single computed marker.

    See https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0144959
    for complete details.

    Args:
        gt (TrackingData): Tracking data object containing graph and segmentations
        pred (TrackingData): Tracking data object containing graph and segmentations
        label_key (str, optional): Key for the segmentation label attribute on each node. 
        Defaults to "segmentation_id".

    Returns:
        list[(gt_node, pred_node)]: list of tuples where each tuple contains a gt node and pred node

    Raises:
        ValueError: gt and pred must be a TrackingData object
        ValueError: GT and pred segmentations must be the same shape
    """
    if not isinstance(gt, TrackingData) or not isinstance(pred, TrackingData):
        raise ValueError(
            "Input data must be a TrackingData object with a graph and segmentations"
        )

    mapper = []

    G_gt, mask_gt = gt.tracking_graph, gt.segmentation
    G_pred, mask_pred = pred.tracking_graph, pred.segmentation

    if mask_gt.shape != mask_pred.shape:
        raise ValueError("Segmentation shapes must match between gt and pred")

    det_matrices = {}
    # Get overlaps for each frame
    for i, t in enumerate(
        tqdm(range(gt.tracking_graph.start_frame, gt.tracking_graph.end_frame), desc='Matching frames')
    ):
        gt_frame = mask_gt[i]
        res_frame = mask_pred[i]
        gt_frame_nodes = gt.tracking_graph.nodes_by_frame[t]
        pred_frame_nodes = pred.tracking_graph.nodes_by_frame[t]

        # get the labels for this frame
        gt_labels = dict(filter(lambda item: item[0] in gt_frame_nodes, nx.get_node_attributes(G_gt.graph, label_key).items()))
        pred_labels = dict(filter(lambda item: item[0] in pred_frame_nodes, nx.get_node_attributes(G_pred.graph, label_key).items()))
        
        # make dictionary from label to ID so we know where in matrix to assign matches
        gt_label_to_id = {v: (k, i) for i, (k, v) in enumerate(gt_labels.items())}
        pred_label_to_id = {v: (k, i) for i, (k, v) in enumerate(pred_labels.items())}
        frame_det_matrix = np.zeros((len(pred_frame_nodes), len(gt_frame_nodes)), dtype=np.uint8)
        overlapping_gt_labels, overlapping_res_labels = get_overlapping_bounding_boxes(gt_frame, res_frame)
        populate_det_matrix(frame_det_matrix, gt_frame, res_frame, overlapping_gt_labels, overlapping_res_labels, gt_label_to_id, pred_label_to_id)
        
        ordered_gt_node_ids = [v[0] for v in sorted(gt_label_to_id.values(), key = lambda x: x[1])]
        ordered_comp_node_ids = [v[0] for v in sorted(pred_label_to_id.values(), key = lambda x: x[1])]

        det_matrices[t] = {
            "det": frame_det_matrix,
            "comp_ids": ordered_comp_node_ids,
            "gt_ids": ordered_gt_node_ids,
        }
    pred._det_matrices = det_matrices
    matching = get_node_matching_map(det_matrices)
    return matching

def populate_det_matrix(frame_matrix, gt_frame, pred_frame, gt_labels, res_labels, gt_label_to_id, res_label_to_id):
    for i in range(len(gt_labels)):
        gt_label = gt_labels[i]
        res_label = res_labels[i]
        gt_blob_mask = gt_frame == gt_label
        comp_blob_mask = pred_frame == res_label
        is_match = int(detection_test(gt_blob_mask, comp_blob_mask))
        if is_match:
            pred_idx = res_label_to_id[res_label][1]
            gt_idx = gt_label_to_id[gt_label][1]
            frame_matrix[pred_idx, gt_idx] = is_match

def get_node_matching_map(detection_matrices: "Dict"):
    """Return list of tuples of (gt_id, comp_id) for all matched nodes

    Parameters
    ----------
    detection_matrices : Dict
        Dictionary indexed by t holding `det`, `comp_ids` and `gt_ids`

    Returns
    -------
    matched_nodes: List[Tuple[int, int]]
        List of tuples (gt_node_id, comp_node_id) denoting matched nodes
        between reference graph and computed graph
    """
    matched_nodes = []
    for m_dict in detection_matrices.values():
        matrix = m_dict["det"]
        comp_nodes = np.asarray(m_dict["comp_ids"])
        gt_nodes = np.asarray(m_dict["gt_ids"])
        row_idx, col_idx = np.nonzero(matrix)
        comp_node_ids = comp_nodes[row_idx]
        gt_node_ids = gt_nodes[col_idx]
        matched_nodes.extend(list(zip(gt_node_ids, comp_node_ids)))
    return matched_nodes

def detection_test(gt_blob: "np.ndarray", comp_blob: "np.ndarray") -> bool:
    """Check if computed marker overlaps majority of the reference marker.

    Given a reference marker and computer marker in original coordinates,
    return True if the computed marker overlaps strictly more than half
    of the reference marker's pixels, otherwise False.

    Parameters
    ----------
    gt_blob : np.ndarray
        2D or 3D boolean mask representing the pixels of the ground truth
        marker
    comp_blob : np.ndarray
        2D or 3D boolean mask representing the pixels of the computed
        marker

    Returns
    -------
    bool
        True if computed marker majority overlaps reference marker, else False.
    """
    n_gt_pixels = np.sum(gt_blob)
    intersection = np.logical_and(gt_blob, comp_blob)
    comp_blob_matches_gt_blob = int(np.sum(intersection) > 0.5 * n_gt_pixels)
    return comp_blob_matches_gt_blob

def get_overlapping_bounding_boxes(gt_frame, res_frame):
    gt_props = regionprops(gt_frame.astype(np.uint16))
    gt_boxes = [np.array(gt_prop.bbox) for gt_prop in gt_props]
    gt_boxes = np.array(gt_boxes).astype(np.float64)
    gt_box_labels = np.asarray([int(gt_prop.label) for gt_prop in gt_props], dtype=np.uint16)

    res_props = regionprops(res_frame.astype(np.uint16))
    res_boxes = [np.array(res_prop.bbox) for res_prop in res_props]
    res_boxes = np.array(res_boxes).astype(np.float64)
    res_box_labels = np.asarray([int(res_prop.label) for res_prop in res_props], dtype=np.uint16)

    overlaps = compute_overlap(gt_boxes, res_boxes)  # has the form [gt_bbox, res_bbox]

    # Find the bboxes that have overlap at all (ind_ corresponds to box number - starting at 0)
    ind_gt, ind_res = np.nonzero(overlaps)
    ind_gt = np.asarray(ind_gt, dtype=np.uint16)
    ind_res = np.asarray(ind_res, dtype=np.uint16)
    overlapping_gt_labels = gt_box_labels[ind_gt]
    overlapping_res_labels = res_box_labels[ind_res]
    return overlapping_gt_labels, overlapping_res_labels



if __name__ == '__main__':
    from cell_tracking_metrics.loaders.ctc import load_ctc_data
    gt_dir = "/home/draga/PhD/data/cell_tracking_challenge/Fluo-N2DL-HeLa/01_GT/TRA"
    gt_track_pth = "/home/draga/PhD/data/cell_tracking_challenge/Fluo-N2DL-HeLa/01_GT/TRA/man_track.txt"
    res_dir = "/home/draga/PhD/data/cell_tracking_challenge/Fluo-N2DL-HeLa/01_RES"
    res_track_pth = "/home/draga/PhD/data/cell_tracking_challenge/Fluo-N2DL-HeLa/01_RES/res_track.txt"
    gt_data = load_ctc_data(gt_dir, gt_track_pth)
    res_data = load_ctc_data(res_dir, res_track_pth)
    mapping = match_ctc(gt_data, res_data)