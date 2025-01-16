import cv2
import numpy as np
from time import time
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from statistics import *

offset_accuracy = 0
offset_deficiency = 0
imgplot = None
ref_image_path = "available_logo.png"
target_image_path = "screenshot.jpg"
threshold = 0
overlay = None
combined_heatmap = None

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

def apply_normalized_tunable_sigmoid_function(value, offset_accuracy):
    """
    This function is crucial for the calculation of the score of the pixels.
    It's a sigmoid function with x and k value, x  is the current score and k for adjusting curve steepness.
    
    The usage of this function is to amplify nice score and reduce bad score.
    
    s/o https://dhemery.github.io/DHE-Modules/technical/sigmoid/
    """
    return (value - value * offset_accuracy) / (offset_accuracy - np.abs(value) * 2 * offset_accuracy + 1)


def calculate_heatmaps(ref_scores, target_hsv, mask, offset_deficiency, offset_accuracy):
    heatmap_h = np.zeros((target_hsv.shape[0], target_hsv.shape[1]), dtype=np.float16)
    heatmap_s = np.zeros((target_hsv.shape[0], target_hsv.shape[1]), dtype=np.float16)
    heatmap_v = np.zeros((target_hsv.shape[0], target_hsv.shape[1]), dtype=np.float16)

    mask_height, mask_width = mask.shape
    ref_h, ref_s, ref_v = ref_scores

    #max_effectif_pixels_within_mask is the max number of pixels that can be affected by the mask when the mask x y value is true
    max_effectifs_pixels_within_mask = np.sum(mask)
    
    for y in range(target_hsv.shape[0]):
        for x in range(target_hsv.shape[1]):
            # Determine the patch size and location to compare with the reference image
            patch_x1y1 = (max(0, x - mask_width // 2), max(0, y - mask_height // 2))
            patch_x2y2 = (min(target_hsv.shape[1], x + mask_width // 2 + 1), min(target_hsv.shape[0], y + mask_height // 2 + 1))
            patch = target_hsv[patch_x1y1[1]:patch_x2y2[1], patch_x1y1[0]:patch_x2y2[0]]
            
            # Determine the valid mask region, it should be within the patch
            valid_mask_height = min(patch_x2y2[1] - patch_x1y1[1], mask_height)
            valid_mask_width = min(patch_x2y2[0] - patch_x1y1[0], mask_width)
            valid_x = mask_width // 2
            valid_y = mask_height // 2
            valid_mask_x1y1 = (valid_x - valid_mask_width // 2, valid_y - valid_mask_height // 2)
            valid_mask_x2y2 = (valid_x + valid_mask_width // 2 + valid_mask_width %2, valid_y + valid_mask_height // 2 + valid_mask_height %2)
            valid_mask = mask[valid_mask_x1y1[1]:valid_mask_x2y2[1], valid_mask_x1y1[0]:valid_mask_x2y2[0]]

            # Calculate the score for each channel
            patch_h = patch[:valid_mask_height, :valid_mask_width, 0] * valid_mask
            patch_s = patch[:valid_mask_height, :valid_mask_width, 1] * valid_mask
            patch_v = patch[:valid_mask_height, :valid_mask_width, 2] * valid_mask
            
            ref_patch_h = ref_h[:valid_mask_height, :valid_mask_width]
            ref_patch_s = ref_s[:valid_mask_height, :valid_mask_width]
            ref_patch_v = ref_v[:valid_mask_height, :valid_mask_width]
            
            effective_pixels = np.sum(valid_mask)
            
            """  
            The normalization factor is used to normalize the score of the pixels that are not masked out
            
            The normalization factor is calculated as follows:
            normalization_factor = effective pixels in the mask / max effectifs pixels in the mask
            
            """
            normalization_factor = effective_pixels / max_effectifs_pixels_within_mask
            

            score_h = np.abs(patch_h - ref_patch_h) / 360
            score_s = np.abs(patch_s - ref_patch_s) / 100
            score_v = np.abs(patch_v - ref_patch_v) / 100
            
            """
            0.8, 0.1, 0.1 are the weights of the channels h, s, v respectively
            For this case I considered the h channel to be more important than the other channels
            """

            score_h = apply_normalized_tunable_sigmoid_function(score_h, offset_accuracy) * normalization_factor * 0.8
            score_s = apply_normalized_tunable_sigmoid_function(score_s, offset_accuracy) * normalization_factor * 0.1
            score_v = apply_normalized_tunable_sigmoid_function(score_v, offset_accuracy) * normalization_factor * 0.1
            
            score_h = np.mean(score_h)
            score_s = np.mean(score_s) 
            score_v = np.mean(score_v) 
                        
            if x == 0 and y == 0:
                print(f"score_h: {score_h}, score_s: {score_s}, score_v: {score_v}")
                print(f'normalization_factor: {normalization_factor}')
                print(f'accacy_factor: {offset_accuracy}')
            
            heatmap_h[y, x] = score_h
            heatmap_s[y, x] = score_s
            heatmap_v[y, x] = score_v

    return heatmap_h, heatmap_s, heatmap_v


def create_combined_heatmap(heatmap_h, heatmap_s, heatmap_v):
    combined_heatmap = (heatmap_h + heatmap_s + heatmap_v) / 3.0
    combined_heatmap = cv2.normalize(combined_heatmap.astype('float32'), None, 0, 1, cv2.NORM_MINMAX)
    return combined_heatmap

def create_color_heatmap(ref_image_path, target_image_path, threshold, offset_deficiency, offset_accuracy):
    #print(ref_image_path, target_image_path, threshold, offset_accuracy, offset_deficiency)
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
    heatmap_h, heatmap_s, heatmap_v = calculate_heatmaps(ref_scores, target_hsv, mask, offset_deficiency, offset_accuracy)
    global combined_heatmap
    combined_heatmap = create_combined_heatmap(heatmap_h, heatmap_s, heatmap_v)
    # print max value of combined_heatmap and its position
    #print(np.max(combined_heatmap), np.unravel_index(np.argmax(combined_heatmap, axis=None), combined_heatmap.shape))
    heatmap_colored = cv2.applyColorMap((1 - combined_heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
    # Aplly the threshold
    heatmap_colored[combined_heatmap < threshold] = 0
    
    overlay = cv2.addWeighted(target_image, 0.5, heatmap_colored, 0.5, 0)
    return overlay


def display_pixel_score(event):
    if event.inaxes == ax:
        x, y = int(event.xdata), int(event.ydata)
        if 0 <= y < combined_heatmap.shape[0] and 0 <= x < combined_heatmap.shape[1]:
            # pixel value of the combined heatmap
            pixel_score = combined_heatmap[y, x] 
            fig.canvas.manager.set_window_title(f"Pixel ({x}, {y}) - Score: {pixel_score:.2f}")
        else:
            fig.canvas.manager.set_window_title("")

def calculate():
    """
    Calcule l'overlay et retourne l'image mise à jour.
    """
    ot = time()
    global overlay
    overlay= create_color_heatmap(ref_image_path, target_image_path, threshold, offset_deficiency, offset_accuracy)
    et = time() - ot
    print(f"Elapsed time: {et} s")

def update_image():
    global imgplot
    global overlay
    calculate()
    if imgplot is None:
        imgplot = ax.imshow(overlay)
    else:
        imgplot.set_data(overlay)
    fig.canvas.draw_idle()
    

def update_accuracy(val):
    global offset_accuracy
    offset_accuracy = val

def update_deficiency_factor(val):
    global offset_deficiency
    offset_deficiency = val

def update_threshold(val):
    global threshold
    threshold = val

fig, ax = plt.subplots(figsize=(15, 5))
plt.subplots_adjust(left=0.1, bottom=0.25, right=0.9, top=0.99, wspace=0.2, hspace=0.2)
axcolor = 'lightgoldenrodyellow'

ax_slider_accuracy = plt.axes([0.25, 0.2, 0.65, 0.03], facecolor=axcolor)
ax_slider_deficiency = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)
ax_slider_threshold = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)

slider_accuracy = Slider(ax_slider_accuracy, 'Accuracy', -1, 1, valinit=offset_accuracy, valstep=0.01)
slider_deficiency = Slider(ax_slider_deficiency, 'Deficiency', 0, .1, valinit=offset_deficiency, valstep=0.01)
slider_threshold = Slider(ax_slider_threshold, 'Threshold', 0, 1, valinit=threshold, valstep=0.01)

ax_button = plt.axes([0.8, 0.025, 0.1, 0.04])
button = Button(ax_button, 'Calculer', color=axcolor, hovercolor='0.975')

button.on_clicked(lambda event: update_image())
slider_accuracy.on_changed(update_accuracy)
slider_deficiency.on_changed(update_deficiency_factor)
slider_threshold.on_changed(update_threshold)

update_image()

fig.canvas.mpl_connect('motion_notify_event', display_pixel_score)

plt.show()