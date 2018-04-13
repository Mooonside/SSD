import matplotlib.pyplot as plt
from numpy import random


def plt_bboxes(img, scores, bboxes, figsize=(10, 10), palette=None, linewidth=1.5, ignore_labels=[0]):
    """Visualize bounding boxes. Largely inspired by SSD-MXNET!
    """
    fig = plt.figure(figsize=figsize)
    plt.imshow(img)
    height = img.shape[0]
    width = img.shape[1]
    colors = dict()

    for cls_id in scores.keys():
        if cls_id in ignore_labels:
            continue
        else:
            try:
                cls_score = scores[cls_id][0]
                cls_bboxes = bboxes[cls_id][0]
            except:
                continue

            if palette is None:
                cls_color = (random.random(), random.random(), random.random())
            else:
                cls_color = tuple(palette[cls_id, :] / 255.0)

            for bbox_idx, bbox_score in enumerate(cls_score):
                ymin = int(cls_bboxes[bbox_idx][0] * height)
                xmin = int(cls_bboxes[bbox_idx][1] * width)
                ymax = int(cls_bboxes[bbox_idx][2] * height)
                xmax = int(cls_bboxes[bbox_idx][3] * width)
                #                 print("Class:{}, Score:{:.3f}, Bboxes:{}" .format(cls_id, bbox_score, cls_bboxes[bbox_idx]))

                rect = plt.Rectangle((xmin, ymin), xmax - xmin,
                                     ymax - ymin, fill=False,
                                     edgecolor=cls_color,
                                     linewidth=linewidth)
                plt.gca().add_patch(rect)
                class_name = str(cls_id)
                plt.gca().text(xmin, ymin - 2,
                               '{:s} | {:.3f}'.format(class_name, bbox_score),
                               bbox=dict(facecolor=cls_color, alpha=0.5),
                               fontsize=12, color='white')
    plt.show()