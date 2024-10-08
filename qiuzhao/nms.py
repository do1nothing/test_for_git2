import numpy as np

def iou(box1,boxes):
    xx1 = np.maximum(box1[0], boxes[:,0])
    yy1 = np.maximum(box1[1], boxes[:,1])
    
    xx2 = np.minimum(box1[2], boxes[:,2])
    yy2 = np.minimum(box1[3], boxes[:,3])
    inter_arena = np.abs(xx2 - xx1) * np.abs(yy2 - yy1)
    box1_arena = np.abs(box1[2] - box1[0]) * np.abs(box1[3] - box1[1])
    boxes_arena = np.abs(boxes[:,2] - boxes[:,0]) * np.abs(boxes[:,3] - boxes[:,1])
    return inter_arena / (box1_arena + boxes_arena - inter_arena)

def nms(boxes, scores, iou_thred):
    scores_index = np.argsort(scores)[::-1]
    keep = []
    while len(scores_index) >= 1:
        # scores_cur = scores[scores_index[0]]
        keep.append(scores_index[0])
        if len(scores_index) == 1:
            break
        # scores_other = scores[scores[1:]]
        iou_scores = iou(boxes[scores_index[0]], boxes[scores_index][1:])
        scores_index = scores_index[1:][iou_scores <= iou_thred]
    return keep
    
if __name__ == "__main__":
    boxes = np.array([[1, 1, 3, 3],
                      [2, 2, 4, 4],
                      [2.5, 2.5, 5, 5]])
    scores = np.array([0.8,0.6,0.7])
    iou_thred = 0.1
    # iou_score = iou(boxes[0], boxes[1:])
    # print(iou_score)
    index = nms(boxes, scores, iou_thred)
    print(index)