import cv2
import numpy as np
from time import time


def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)

def calculate_pixel_scores(ref_hsv, mask):
    ref_hsc_shape = ref_hsv.shape[:2]
    h_scores = np.empty(ref_hsc_shape)
    s_scores = np.empty(ref_hsc_shape)
    v_scores = np.empty(ref_hsc_shape)
    for y in range(ref_hsv.shape[0]):
        for x in range(ref_hsv.shape[1]):
            if mask[y,x]:
                h_scores[y] = ref_hsv[y, x, 0]
                s_scores[y] = ref_hsv[y, x, 1]
                v_scores[y] = ref_hsv[y, x, 2]
            else:
                h_scores[y] = 0
                s_scores[y] = 0
                v_scores[y] = 0
    return h_scores, s_scores, v_scores

def calculate_heatmaps(ref_scores, target_hsv, mask):
    heatmap_h = np.zeros((target_hsv.shape[0], target_hsv.shape[1]), dtype=np.float16)
    heatmap_s = np.zeros((target_hsv.shape[0], target_hsv.shape[1]), dtype=np.float16)
    heatmap_v = np.zeros((target_hsv.shape[0], target_hsv.shape[1]), dtype=np.float16)

    mask_height, mask_width = mask.shape
    ref_h, ref_s, ref_v = ref_scores
    """ 
    accuracy is the ratio between how close each pixel is to the reference 
    confidence is how close all pixel averages are to the reference average
    
    for example, if the object you're looking for is cut in 2, but is almost identical, then the precision will be high, but the confidence will be low, 
    because there won't be many pixels. 
    
    If it's something completely different, the precision will be low, but the confidence higher."""

    accuracy_weight = 0.5
    confidence_weight = 0.5
    

    for y in range(target_hsv.shape[0]):
        for x in range(target_hsv.shape[1]):
            y_start = max(0, y - mask_height // 2)
            y_end = min(target_hsv.shape[0], y + mask_height // 2 + 1)
            x_start = max(0, x - mask_width // 2)
            x_end = min(target_hsv.shape[1], x + mask_width // 2 + 1)

            patch = target_hsv[y_start:y_end, x_start:x_end]
           
            valid_mask_height = min(y_end - y_start, mask_height)
            valid_mask_width = min(x_end - x_start, mask_width)
            valid_y_start = y_start + valid_mask_height
            valid_y_end = valid_y_start + valid_mask_height
            valid_x_start = x_start + valid_mask_width
            valid_x_end = valid_x_start + valid_mask_width
            print(valid_y_start, valid_y_end, valid_x_start, valid_x_end)
            valid_mask = mask[valid_y_start:valid_y_end, valid_x_start:valid_x_end]

            patch_h = patch[:valid_mask_height, :valid_mask_width, 0] * valid_mask
            patch_s = patch[:valid_mask_height, :valid_mask_width, 1] * valid_mask
            patch_v = patch[:valid_mask_height, :valid_mask_width, 2] * valid_mask
            
            score_h = np.mean(np.abs(patch_h - ref_h[:valid_mask_height, :valid_mask_width]))
            score_s = np.mean(np.abs(patch_s - ref_s[:valid_mask_height, :valid_mask_width]))
            score_v = np.mean(np.abs(patch_v - ref_v[:valid_mask_height, :valid_mask_width]))

            heatmap_h[y, x] = score_h
            heatmap_s[y, x] = score_s
            heatmap_v[y, x] = score_v

    return heatmap_h, heatmap_s, heatmap_v


def create_combined_heatmap(heatmap_h, heatmap_s, heatmap_v):
    combined_heatmap = (heatmap_h + heatmap_s + heatmap_v) / 3.0
    combined_heatmap = cv2.normalize(combined_heatmap.astype('float32'), None, 0, 1, cv2.NORM_MINMAX)
    return combined_heatmap

def create_color_heatmap(ref_image_path, target_image_path, threshold=.4):
    ref_image = cv2.imread(ref_image_path, cv2.IMREAD_UNCHANGED)
    target_image = cv2.imread(target_image_path)

    if ref_image.shape[2] == 4:
        alpha_channel = ref_image[:, :, 3]
        mask = alpha_channel > 0
        ref_image = cv2.cvtColor(ref_image, cv2.COLOR_BGRA2BGR)
    else:
        mask = np.ones((ref_image.shape[0], ref_image.shape[1]), dtype=bool)
    ref_hsv = cv2.cvtColor(ref_image, cv2.COLOR_BGR2HSV)
    target_hsv = cv2.cvtColor(target_image, cv2.COLOR_BGR2HSV)

    ref_scores = calculate_pixel_scores(ref_hsv, mask)
    heatmap_h, heatmap_s, heatmap_v = calculate_heatmaps(ref_scores, target_hsv, mask)
    print(heatmap_h.shape, target_hsv.shape)
    combined_heatmap = create_combined_heatmap(heatmap_h, heatmap_s, heatmap_v)

    heatmap_colored = cv2.applyColorMap((combined_heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
    
    
    thresholded_heatmap = np.zeros(combined_heatmap.shape, dtype=np.uint8)
    thresholded_heatmap[combined_heatmap >= threshold] = 255
    print(np.max(combined_heatmap))
    overlay_threshold = cv2.addWeighted(target_image, 0.7, cv2.cvtColor(thresholded_heatmap, cv2.COLOR_GRAY2BGR), 0.8, 0)
    
    
    overlay = cv2.addWeighted(target_image, 0.7, heatmap_colored, 0.3, 0)

    return overlay, heatmap_colored, thresholded_heatmap, overlay_threshold


ref_image_path = "available_logo.png"
target_image_path = "screenshot.jpg"

ot = time()
overlay, heatmap_colored,thresholded_heatmap, overlay_threshold = create_color_heatmap(ref_image_path, target_image_path)
et = time() - ot
print(f"Elapsed time: {et} s")
cv2.imshow("Heatmap Overlay", ResizeWithAspectRatio(overlay, width=250))
cv2.waitKey(0)
cv2.destroyAllWindows()