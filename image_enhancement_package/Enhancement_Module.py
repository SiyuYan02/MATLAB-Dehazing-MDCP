from pylab import *
import Basic_Module as basic
import numpy as np
import pywt
import cv2


class unit:

    def __init__(self, img_in):
        self.img = img_in

        self.a, self.b = shape(self.img)
        self.point_num_all = self.a * self.b

    # 直方图均衡化 极易过曝
    def HE(self):
        trans = [0 for i in range(256)]  # 查找表
        im_out = zeros_like(self.img)

        histogram1, histogram2 = basic.get_histogram(self.img)

        for i in range(256):
            trans[i] = round(i * histogram2[i] / self.point_num_all)

        for i in range(self.a):
            for j in range(self.b):
                im_out[i][j] = trans[self.img[i][j]]

        return im_out

    # 自适应直方图均衡化 该方法效果较好
    def CLAHE(self):
        # 第一步：实例化自适应直方图均衡化函数
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        # 第二步：进行自适应直方图均衡化
        clahe = clahe.apply(self.img)
        img_out = uint8(clahe)

        return img_out

    # 线性变换 基于累加直方图 极易过曝
    def LINEAR(self):
        im_out = zeros_like(self.img)
        trans = [0 for i in range(256)]  # 查找表

        histogram1, histogram2 = basic.get_histogram(self.img)

        for i in range(256):
            trans[i] = round(225 * histogram2[i] / self.point_num_all)

        for i in range(self.a):
            for j in range(self.b):
                im_out[i][j] = trans[self.img[i][j]]

        return uint8(im_out)

    # 伽马变换 非自适应 gamma>1时调亮 gamma<1时调暗 极易过曝
    def GAMMA(self, gamma=1.5):
        invgamma = 1 / gamma
        im_out = np.array(np.power((self.img / 255), invgamma) * 255, dtype=np.uint8)

        return uint8(im_out)

    # 小波变换 该方法降噪效果好
    def WT(self):

        # 对img进行haar小波变换,变量分别是低频，水平高频，垂直高频，对角线高频
        cA, (cH, cV, cD) = pywt.dwt2(self.img, "haar")

        # 根据小波系数重构的图像
        img_out = pywt.idwt2((cA, (cH, cV, cD)), "haar")
        img_out = np.uint8(img_out)

        return img_out

    # PDE 基于PM模型 该方法降噪效果好
    def PDE(self, iter=50, dt=0.25, k_m=20, alpha=17, option=2):
        a, b = shape(self.img)

        img_in = float64(self.img)
        img_temp = img_in
        img_out = img_in

        for i in range(iter):
            k = k_m
            k = k_m * exp(-alpha / (1 + i * dt))
            im_grad_x = cv2.Sobel(img_out, -1, 1, 0)
            im_grad_y = cv2.Sobel(img_out, -1, 0, 1)

            if option == 1:
                c_x = exp(-im_grad_x / k ** 2)
                c_y = exp(-im_grad_y / k ** 2)

            if option == 2:
                c_x = 1.0 / (1 + (im_grad_x / k) ** 2)
                c_y = 1.0 / (1 + (im_grad_y / k) ** 2)

            img_temp += dt * (c_x * im_grad_x + c_y * im_grad_y)
            img_out = img_temp

        for i in range(a):
            for j in range(b):
                if img_out[i][j] > 255:
                    img_out[i][j] = 255
                if img_out[i][j] < 0:
                    img_out[i][j] = 0

        img_out = uint8(img_out)

        return img_out

    # 单尺度Retinex
    def SSR(self, sigma=80):

        img = np.clip(self.img, 1, 254)
        img_out = img - np.log10(cv2.GaussianBlur(img, (0, 0), sigma))

        '''
        img_loged = uint8(log10(self.img + 1.0))
        img_log_gaussed = cv2.GaussianBlur(img_loged, (0, 0), sigma)
        img_out = self.img - img_log_gaussed
        '''

        return uint8(img_out)

    # 多尺度Retinex
    def MSR(self, sigma_list=[15, 80, 200]):
        length = size(sigma_list)
        img = np.clip(self.img, 1, 254)

        img_temp = [0 for i in range(length)]
        img_out = zeros_like(self.img)

        for i in range(length):
            img_temp[i] = img - np.log10(cv2.GaussianBlur(img, (0, 0), sigma_list[i]))
            img_out = img_out + float64(img_temp[i]) / length

        img_out = uint8(img_out)

        return img_out

    # 自适应 多尺度retinex 颜色修复
    def Automated_MSRCR(self, sigma_list=[15, 80, 200], low=0.05, high=0.95):

        img_retinex = self.MSR(sigma_list)

        unique, count = np.unique(np.int32(img_retinex * 100), return_counts=True)

        for u, c in zip(unique, count):
            if u == 0:
                zero_count = c
                break

        low_val = unique[0] / 100.0
        high_val = unique[-1] / 100.0
        low_num = self.point_num_all * low
        high_num = self.point_num_all * high
        sum = 0

        for u, c in zip(unique, count):
            sum = sum + c
            if u < 0 and c <= zero_count * 0.1 and sum <= low_num:
                low_val = u / 100.0
            if u > 0 and c < zero_count * 0.1 and sum >= high_num:
                high_val = u / 100.0
                break

        img_retinex = np.maximum(np.minimum(img_retinex, 100 * high_val), 100 * low_val)
        img_retinex = (img_retinex - np.min(img_retinex)) / (np.max(img_retinex) - np.min(img_retinex)) * 255
        img_retinex = np.uint8(img_retinex)

        return img_retinex

    # Edit this for a new method
    def TEST_ATD(self, r=81, eps=0.001, w=0.95, maxV1=0.80, bGamma=False):
        # 当前 基于暗通道先验的图像去雾

        V1, A = basic.getV1(self.img, r, eps, w, maxV1)  # 得到遮罩图像和大气光照

        V1 = 0
        Y = (self.img - V1) / (1 - V1 / A) / 255  # 颜色校正
        Y = np.clip(Y, 0, 1)

        if bGamma:
            Y = Y ** (np.log(0.5) / np.log(Y.mean()))  # gamma校正,默认不进行该操作

        return uint8(Y * 255)

    def TEST_HE(self, low_threshold=0.02, high_threshold=0.02):
        img_out = zeros_like(self.img)
        img_information = basic.get_important_point(self.img)
        unique1, count1 = np.unique(np.int32(self.img * 100), return_counts=True)
        unique2, count2 = np.unique(np.int32(self.img * 100), return_counts=True)
        count = copy(count1)
        if unique1[0] == 0:
            for i in range(size(unique2)):
                count[i] = count[i] - count2[i]
        else:
            for i in unique2:
                count[i] = count[i] - count2[i]
        return 0

    def TEST(self):
        img_out = basic.get_important_point(self.img)
        return img_out
