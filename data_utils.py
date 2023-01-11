import cv2
import numpy as np
import random

def get_statistics(data):
    """
    Return statistics results of data.
    bounds: the boundary of data, non-normalized
    storke_num:  the number of strokes
    abs_stroke_3: the absolute coordinate data
    """

    abs_data = np.cumsum(data, axis=0)

    min_x = np.min(abs_data[:, 0])
    min_y = np.min(abs_data[:, 1])
    max_x = np.max(abs_data[:, 0])
    max_y = np.max(abs_data[:, 1])

    # adjust non-negative coordinates
    if min_x < 0:
        abs_data[:, 0] += -min_x # ensure all x-axis is non-negative
    if min_y < 0:
        abs_data[:, 1] += -min_y # y-axis

    # update bounds
    min_x = np.min(abs_data[:, 0])
    min_y = np.min(abs_data[:, 1])
    max_x = np.max(abs_data[:, 0])
    max_y = np.max(abs_data[:, 1])

    bounds = (min_x, max_x, min_y, max_y)

    stroke_num = abs_data[-1][-1]

    abs_stroke_3 = abs_data
    abs_stroke_3[:, -1] = data[:, -1]

    return bounds, stroke_num, abs_stroke_3

def rescale(abs_data, bounds, img_size=None):
    """
    :param abs_data: absolute coordinate for each point
    :param img_size: the mininum item of the image size eg: min(img_w, img_h)
    :param bounds: normalize to [0, 1]
    :return:
    """

    # normalize factor
    # first normalize to [0, 1]
    normalize_factor = max(bounds[1], bounds[3])
    norm_data = np.zeros(abs_data.shape)

    norm_data[:, 0:2] = abs_data[:, 0:2] / normalize_factor
    norm_data[:, -1] = abs_data[:, -1]

    if img_size is not None:
        img_data = np.zeros(abs_data.shape)
        img_data[:, 0:2] = norm_data[:, 0:2] * img_size
        img_data[:, -1] = norm_data[:, -1]

        return img_data.astype("int16")

    else:
        return norm_data

def draw_three(img_data, save=False, img_size=256):
    """
    :param img_data: stroke_3 data, scaled to img_size
    :param padding: leave a blank space
    :param save: save the skecth image
    :return: None
    """

    thickness = 5 #int(img_size * 0.025)

    img_data[:, 0:2] += thickness
    start_x, start_y = img_data[0, 0],  img_data[0, 1]

    # initialize canvas
    color = (0, 0, 0) # black strokes
    canvas_color = 255 # white background
    canvas = np.ones((img_size + 3 * (thickness + 1), img_size + 3 * (thickness + 1), 3), dtype='uint8') * 255

    pen_start = np.array([start_x, start_y])
    first_zero = False # control the stop between strokes
    #segment_count = 0
    for stroke in img_data:
        #segment_count += 1
        state = stroke[2:]
        pen_end = stroke[:-1]
        if first_zero:
            pen_start = pen_end
            first_zero = False
            continue
        cv2.line(canvas, tuple(pen_start), tuple(pen_end), color, thickness=thickness)
        # test_file_name = './rasterize/'+str(segment_count)+'.png'
        # cv2.imwrite(test_file_name, canvas)
        if int(state) == 1:  # next stroke
            first_zero = True
        pen_start = pen_end

    if save:

        cv2.imwrite(f"./test.png", canvas)

    return cv2.resize(canvas, (img_size, img_size))

def draw_five(img_data, save=False, img_size=256):
    """
    :param img_data: stroke_3 data, scaled to img_size
    :param padding: leave a blank space
    :param save: save the skecth image
    :return: None
    """

    thickness = int(img_size * 0.025)
    img_data[:, :2] = img_data[:, :2] * img_size
    img_data = img_data.astype('uint8')

    img_data[:, 0:2] += thickness
    start_x, start_y = img_data[0, 0],  img_data[0, 1]

    # initialize canvas
    color = (0, 0, 0) # black strokes
    canvas_color = 255 # white background
    canvas = np.ones((img_size + 3 * (thickness + 1), img_size + 3 * (thickness + 1), 3), dtype='uint8') * 255

    pen_start = np.array([start_x, start_y])
    first_zero = False # control the stop between strokes
    #segment_count = 0
    for stroke in img_data:
        if (stroke[2:] == np.array((0, 1, 0))).astype('uint8').all():
            continue
        #segment_count += 1
        state = stroke[3]
        pen_end = stroke[:2]
        if first_zero:
            pen_start = pen_end
            first_zero = False
            continue
        cv2.line(canvas, tuple(pen_start), tuple(pen_end), color, thickness=thickness)
        # test_file_name = './rasterize/'+str(segment_count)+'.png'
        # cv2.imwrite(test_file_name, canvas)
        if int(state) == 1:  # next stroke
            first_zero = True
        pen_start = pen_end

    if save:

        cv2.imwrite(f"./test.png", canvas)

    return cv2.resize(canvas, (img_size, img_size))

if __name__=="__main__":

    test_data = np.load('../mini_data/apple.npz', encoding='latin1', allow_pickle=True)['train'][0]
    bounds, stroke_num, abs_s3 = get_statistics(test_data)

    a = rescale(abs_s3, bounds, img_size=256)

    draw_three(a, save=True, img_size=256)