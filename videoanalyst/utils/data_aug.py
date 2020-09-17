"""
补充实验中，需要验证第一帧是噪声时，对结果的影响。
我们将噪声分为三个部分：
1. 图像模糊
2. 图像亮度
3. 边框抖动
"""
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


def compute_iou(rec1, rec2):
    """
    computing IoU
    :param rec1: (y0, x0, y1, x1), which reflects
            (top, left, bottom, right)
    :param rec2: (y0, x0, y1, x1)
    :return: scala value of IoU
    """
    rec1[2] += rec1[0]
    rec1[3] += rec1[1]
    rec2[2] += rec2[0]
    rec2[3] += rec2[1]
    # computing area of each rectangles
    S_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
    S_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])

    # computing the sum_area
    sum_area = S_rec1 + S_rec2

    # find the each edge of intersect rectangle
    left_line = max(rec1[1], rec2[1])
    right_line = min(rec1[3], rec2[3])
    top_line = max(rec1[0], rec2[0])
    bottom_line = min(rec1[2], rec2[2])

    # judge if there is an intersect
    if left_line >= right_line or top_line >= bottom_line:
        return 0
    else:
        intersect = (right_line - left_line) * (bottom_line - top_line)
        return (intersect / (sum_area - intersect)) * 1.0


def blur1(image, param):
    blur_image = cv2.blur(image, (param, param))
    return blur_image


def blur(image, param):
    # enhancer = ImageEnhance.Sharpness(image)
    # blur_image = enhancer.enhance(param)
    blur_image = image.filter(ImageFilter.GaussianBlur(radius = param))
    return blur_image


def illum(image, param):
    enhancer = ImageEnhance.Brightness(image)
    illum_image = enhancer.enhance(param)
    return illum_image


def change_box(box, param):
    # 确定OTB的边框的格式：是xywh
    x, y, w, h = box
    # 处理边框越界->感觉不处理也没关系
    dx = w * param
    dy = h * param
    x += dx
    y += dy
    return [x, y, w, h]


def pil_to_cv2(image):
    image = np.asarray(image)
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    elif len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        assert False
    return image


if __name__ == '__main__':
    # '''START: 图像模糊(opencv)'''
    # im = cv2.imread('/data/zhbli/Dataset/OTB2015/Basketball/img/0001.jpg')
    # Blur_param = 7
    # Blur_image = blur1(im, Blur_param)
    # cv2.imwrite('/tmp/blur_image_{}.jpg'.format(Blur_param), Blur_image)
    # '''END: 图像模糊(opencv)'''

    '''START: 图像模糊'''
    blur_param = 0
    im = Image.open('/data/zhbli/Dataset/OTB2015/Basketball/img/0001.jpg')
    Illum_image = blur(im, blur_param)
    Illum_image.save('/tmp/blur_image_{}.jpg'.format(blur_param))
    '''END: 图像模糊'''

    '''START: 修改亮度'''
    illum_param = 0.5
    im = Image.open('/data/zhbli/Dataset/OTB2015/Basketball/img/0001.jpg')
    Illum_image = illum(im, illum_param)
    Illum_image.save('/tmp/illum_image_{}.jpg'.format(illum_param))
    '''END: 修改亮度'''

    '''计算IOU'''
    box_param = 0.07
    # box = [31, 41, 15, 926]
    box = [1,2,3,4]
    new_box = change_box(box, box_param)
    print(compute_iou(box, new_box))
    '''计算IOU'''

    '''检查pil_to_cv2是否正确'''
    im_pil = Image.open('/data/zhbli/Dataset/OTB2015/Basketball/img/0002.jpg')
    im_cv = cv2.imread('/data/zhbli/Dataset/OTB2015/Basketball/img/0002.jpg', cv2.IMREAD_COLOR)
    im_cv_new = pil_to_cv2(im_pil)
    print((im_cv == im_cv_new).all())
    delta = im_cv-im_cv_new
    delta[delta>240] = 0
    print(np.sum(delta))
    cv2.imwrite('/tmp/1.jpg', im_cv)
    cv2.imwrite('/tmp/2.jpg', im_cv_new)
    im_pil = Image.open('/data/zhbli/Dataset/OTB2015/Freeman3/img/0001.jpg')
    im_cv = cv2.imread('/data/zhbli/Dataset/OTB2015/Freeman3/img/0001.jpg', cv2.IMREAD_COLOR)
    im_cv_new = pil_to_cv2(im_pil)
    print((im_cv == im_cv_new).all())
    '''END: 检查pil_to_cv2是否正确'''