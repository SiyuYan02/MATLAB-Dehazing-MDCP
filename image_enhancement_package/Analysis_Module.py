from contrast_image import quantitation
from pylab import *
import skimage.metrics as sk
import numpy as np
import cv2


quant = quantitation.Quantitation()


# 获取亮度增加值
def get_AMBE(img_before, img_after):

    ambe = quant.AMBE(img_before, img_after)
    print("AMBE = ", ambe)

    return ambe


# 获取结构相似性
def get_SSIM(img_before, img_after):

    ssim = sk.structural_similarity(img_before, img_after)
    print("SSIM = ", ssim)

    return ssim


# 获取峰值信噪比
def get_PSNR(img_before, img_after):

    psnr = sk.peak_signal_noise_ratio(img_before, img_after)
    print("PSNR = ", psnr)

    return psnr


# 获取均方根误差
def get_MSE(img_before, img_after):

    mse = sk.mean_squared_error(img_before, img_after)
    print("MSE = ", mse)

    return mse


# 获取标准差
def get_SD(img_in):
    p = [0 for i in range(256)]
    a, b = shape(img_in)
    point_num = a * b

    m = 0
    s = 0

    for i in range(a):
        for j in range(b):
            p[img_in[i][j]] += 1
    for i in range(256):
        p[i] /= 256
        m += i * p[i]
    for i in range(256):
        s += (i - m) * (i - m) * p[i]
    sd = sqrt(s) / point_num

    print("SD = ", sd)

    return sd


# 获取平均梯度
def get_AVD(img_in):

    grad_x = cv2.Sobel(img_in, -1, 1, 0)
    grad_y = cv2.Sobel(img_in, -1, 0, 1)
    grad_xy = cv2.add(grad_x, grad_y)
    a, b = shape(grad_xy)
    point_num = a * b
    sum = 0
    for i in range(a):
        for j in range(b):
            sum += grad_xy[i][j]
    avd = sum / point_num

    print("AVD = ", avd)

    return avd


# 获取信息熵
def get_IE(img_in):

    k = 0
    entropy = 0

    img = np.array(img_in)
    tmp = [0 for i in range(256)]

    for i in range(len(img)):
        for j in range(len(img[i])):
            val = img[i][j]
            tmp[val] = float(tmp[val] + 1)
            k = float(k + 1)
    for i in range(len(tmp)):
        tmp[i] = float(tmp[i] / k)
    for i in range(len(tmp)):
        if tmp[i] == 0:
            entropy = entropy
        else:
            entropy = float(entropy - tmp[i] * (math.log(tmp[i]) / math.log(2.0)))

    print("Entropy(IE) = ", entropy)

    return entropy


# 获取全部信息
def get_ANALYSIS_self(img):
    sd = get_SD(img)
    avd = get_AVD(img)
    entropy = get_IE(img)

    return sd, avd, entropy


def get_ANALYSIS_contrast(img_in, img_out):
    ambe = get_AMBE(img_in, img_out)
    psnr = get_PSNR(img_in, img_out)
    ssis = get_SSIM(img_in, img_out)
    mse = get_MSE(img_in, img_out)

    return ambe, psnr, ssis, mse

