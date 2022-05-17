from pylab import *
import numpy as np
import random
import cv2


# 高斯噪声/白噪声（加性）
def noise_gasuss(image, mean=0, var=0.01):
    '''
        添加高斯噪声
        mean : 均值
        var : 方差
    '''
    image = np.array(image/255, dtype=float)
    noise = np.random.normal(mean, var ** 0.5, image.shape)
    img_out = image + noise
    img_out = np.clip(img_out, 0.0, 1.0)
    img_out = np.uint8(img_out * 255)
    return img_out


# 椒盐噪声
def noise_sp(image, prob=0.05):
    '''
        添加椒盐噪声
        prob:噪声比例
    '''
    output = np.zeros(image.shape, np.uint8)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output


# 获取直方图和梯度直方图
def get_histogram(img_in):
    a, b = shape(img_in)
    histogram1 = [0 for i in range(256)]  # 灰度直方图
    histogram2 = [0 for i in range(256)]  # 累加直方图

    for i in range(a):
        for j in range(b):
            histogram1[img_in[i][j]] += 1

    for k in range(256):
        for l in range(k + 1):
            histogram2[k] += histogram1[l]

    return histogram1, histogram2


# 获取梯度图(Soble)
def get_grad(img_in):
    a, b = shape(img_in)
    im_out = zeros_like(img_in)

    im_grad_x = cv2.Sobel(img_in, cv2.CV_64F, 1, 0)
    im_grad_x = cv2.convertScaleAbs(im_grad_x)
    im_grad_y = cv2.Sobel(img_in, cv2.CV_64F, 0, 1)
    im_grad_y = cv2.convertScaleAbs(im_grad_y)
    im_grad_x = im_grad_x / 2
    im_grad_y = im_grad_y / 2
    '''
    for i in range(a):
        for j in range(b):
            im_out[i][j] = sqrt(im_grad_x[i][j] ** 2 + im_grad_y[i][j] ** 2)
    '''
    im_out = im_grad_x + im_grad_y

    return uint8(im_out)


# 获取单位梯度的散度图
def get_div(img_in):
    img_in = uint8(img_in)
    a, b = shape(img_in)

    grad_x = cv2.Sobel(img_in, -1, 1, 0)
    grad_y = cv2.Sobel(img_in, -1, 0, 1)

    grad_xy = grad_x + grad_y

    grad = sqrt(grad_x ** 2 + grad_y ** 2 + 0.01)
    div = grad_xy / grad

    for i in range(a):
        for j in range(b):
            grad[i][j] = sqrt(grad_x[i][j] ** 2 + grad_y[i][j] ** 2)
            if grad[i][j] > 0:
                div[i][j] = grad_xy[i][j] / grad[i][j]

    return uint8(div)


# 获取CANNY边缘图 非自适应 其质量与设置的阈值有非常大的关系
def get_edge(img_in, threshold_low=20, threshold_high=120):

    img_in = cv2.GaussianBlur(uint8(img_in), (7, 7), 1)
    edge = cv2.Canny(img_in, threshold_low, threshold_high)

    return edge


def get_important_point(img_in):
    im_out = zeros_like(img_in)
    a, b = shape(img_in)
    point_num = a * b

    img_temp = cv2.medianBlur(uint8(img_in), 3)
    img_temp = cv2.GaussianBlur(uint8(img_temp), (5, 5), 1)
    img_grad = get_grad(img_temp)

    avd = np.mean(img_grad)
    for i in range(a):
        for j in range(b):
            if img_grad[i][j] > avd:  # 阈值应当自适应，简单粗暴地使用avd是有风险的
                im_out[i][j] = img_in[i][j]

    return im_out


def filter_min(src, r=7):
    '''最小值滤波，r是滤波器半径'''
    '''if r <= 0:
        return src
    h, w = src.shape[:2]
    I = src
    res = np.minimum(I  , I[[0]+range(h-1)  , :])
    res = np.minimum(res, I[range(1,h)+[h-1], :])
    I = res
    res = np.minimum(I  , I[:, [0]+range(w-1)])
    res = np.minimum(res, I[:, range(1,w)+[w-1]])
    return zmMinFilterGray(res, r-1)'''
    return cv2.erode(src, np.ones((2 * r + 1, 2 * r + 1)))  # 使用opencv的erode函数更高效


def filter_guide(I, p, r, eps):
    '''引导滤波，直接参考网上的matlab代码'''
    height, width = I.shape
    m_I = cv2.boxFilter(I, -1, (r, r))
    m_p = cv2.boxFilter(p, -1, (r, r))
    m_Ip = cv2.boxFilter(I * p, -1, (r, r))
    cov_Ip = m_Ip - m_I * m_p

    m_II = cv2.boxFilter(I * I, -1, (r, r))
    var_I = m_II - m_I * m_I

    a = cov_Ip / (var_I + eps)
    b = m_p - a * m_I

    m_a = cv2.boxFilter(a, -1, (r, r))
    m_b = cv2.boxFilter(b, -1, (r, r))
    return m_a * I + m_b


def getV1(m, r, eps, w, maxV1):  # 输入图像，值范围[0,1]
    # 计算大气遮罩图像V1和光照值A, V1 = （1-t）A

    # 得到暗通道图像
    V1 = np.min(m)
    m1 = ones_like(m) * V1
    m1 = filter_guide(m1, filter_min(m, 7), r, eps)  # 使用引导滤波优化
    print(V1)
    print(m1)
    bins = 2000
    ht = np.histogram(m1, bins)  # 计算大气光照A
    d = np.cumsum(ht[0]) / float(m1.size)
    print(ht)
    print(d)
    print(d[1])

    for lmax in range(bins - 1, 0, -1):
        if d[lmax] <= 0.99:
            break

    sum = 0
    num = 0
    a, b = shape(m1)
    for i in range(a):
        for j in range(b):
            if m[i][j] >= ht[1][lmax]:
                sum = sum + m[i][j]
                num = num + 1

    A = sum / num

    V1 = np.minimum(V1 * w, maxV1)  # 对值范围进行限制

    print(lmax)
    print(ht[0][lmax])
    print('v1 = ', V1)
    print('A = ', A)

    return V1, A
