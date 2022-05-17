import Basic_Module as basic
import Analysis_Module as analysis
import Enhancement_Module as enhance
import Exhibition_Module as exhibit
from pylab import *
import time
import cv2
import os

# 可能的报错处理办法:
# 调用image_enhancement库报错 需要修改histogram.py第39行 hitogram 为 histigram
# 调用quantitation库报错 需要将第18行调用MSE前添加 self.


# 用户接口函数
def img_getanalysis(filepath_read_before, filepath_read_after):

    filename_list1 = os.listdir(filepath_read_before)  # 对照组文件夹，仅放一个文件

    img_before = cv2.imread(filepath_read_before + '/' + filename_list1[0], 0)
    print('---------------------------------')
    print('*** information of ORIGIN PICTURE ' + filename_list1[0])
    print('shape of img is ', shape(img_before))
    analysis.get_ANALYSIS_self(img_before)

    filename_list2 = os.listdir(filepath_read_after)  # 实验组文件夹，允许多个文件

    for filename in filename_list2:
        img_after = cv2.imread(filepath_read_after + '/' + filename, 0)
        a, b = shape(img_before)
        img_after = cv2.resize(img_after, (b, a))
        print('---------------------------------')
        print('*** information of', filename)
        analysis.get_ANALYSIS_self(img_after)
        print('- - - - - - - - - - - - - - - - -')
        analysis.get_ANALYSIS_contrast(img_before, img_after)

    return 0


# 用户接口函数
def img_enhancement(filepath_read, filepath_write=0, method_list='HE', show_information=0, show_img=0):

    '''
    :param filepath_read: 读取图片的文件夹路径
    :param filepath_write: 默认0，取值0时，不保存图像；取值1时，保存到当前文件夹；取值为文件路径时，保存到文件路径
    :param method_list: 默认'HE'
                       1图像增强方法,部分含降噪能力,可选'HE', 'CLAHE', 'LINEAR', 'GAMMA', 'WT', 'PDE', 'SSR', 'MSR', 'MSRCR'
                                                  'TEST_ATD'
                       2传统加噪声方法，可选'GAUSS_NOISE', 'SP_NOISE'
                       3传统降噪声方法，可选'GAUSS_DENOISE', 'MEDIAN_DENOISE'
                       4实验中的方法，可选'TEST_ATD', 'TEST'，仅用于测试新方法
                       以上各方法具体释义见对应的Module文件
    :param show_information: 取值0时，不计算评价指标；取值1时，计算
    :param show_img: 取值0时，不显示图像和直方图；取值1时，显示对比信息，取值2时显示全部信息
    :return: 处理一幅图像的平均耗时, 计算一幅图像的信息的平均耗时
    '''

    filename_list = os.listdir(filepath_read)
    file_quantity = size(filename_list)
    method_quantity = size(method_list)
    time_cost_all1 = 0
    time_cost_all2 = 0
    time_cost1_ave = 0
    time_cost2_ave = 0

    for filename in filename_list:
        time_start = time.time()

        # 读取图像
        img_in = cv2.imread(filepath_read + '/' + filename, 0)
        img_list = [img_in]

        # 构建对象
        img_unit = enhance.unit(img_in)

        # 图像处理
        for method in method_list:
            if method == 'TEST':
                img_list.append(img_unit.TEST())
            if method == 'TEST_ATD':
                img_list.append(img_unit.TEST_ATD())
            if method == 'GRAD':
                img_list.append(basic.get_grad(img_in))
            if method == 'DIV':
                img_list.append(basic.get_div(img_in))
            if method == 'CANNY':
                img_list.append(basic.get_edge(img_in))

            if method == 'HE':
                img_list.append(img_unit.HE())
            if method == 'CLAHE':
                img_list.append(img_unit.CLAHE())
            if method == 'LINEAR':
                img_list.append(img_unit.LINEAR())
            if method == 'GAMMA':
                img_list.append(img_unit.GAMMA())
            if method == 'WT':
                img_list.append(img_unit.WT())
            if method == 'PDE':
                img_list.append(img_unit.PDE())
            if method == 'SSR':
                img_list.append(img_unit.SSR())
            if method == 'MSR':
                img_list.append(img_unit.MSR())
            if method == 'MSRCR':
                img_list.append(img_unit.Automated_MSRCR())

            if method == 'GAUSS_NOISE':
                img_list.append(basic.noise_gasuss(img_in))
            if method == 'SP_NOISE':
                img_list.append(basic.noise_sp(img_in))

            if method == 'GAUSS_DENOISE':
                img_list.append(cv2.GaussianBlur(uint8(img_in), (7, 7), 1))
            if method == 'MEDIAN_DENOISE':
                img_list.append(cv2.medianBlur(uint8(img_in), 3))

        # 显示处理耗时
        time_end1 = time.time()
        time_cost1 = time_end1 - time_start
        time_cost_all1 = time_cost_all1 + time_cost1
        print('=====================================================')
        print('image_dealing: cost ', time_cost1, 's')

        # 显示评价指标
        if show_information >= 1:
            print('---------------------------------')
            print('*** information of ORIGIN PICTURE ' + filename)
            print('shape of img is ', shape(img_in))
            analysis.get_ANALYSIS_self(img_in)

            for i in range(method_quantity):
                print('---------------------------------')
                print('*** information of', method_list[i])
                if show_information == 2:
                    analysis.get_ANALYSIS_self(img_list[i + 1])
                    print('- - - - - - - - - - - - - - - - -')
                analysis.get_ANALYSIS_contrast(img_in, img_list[i + 1])

        # 显示总耗时
        time_end2 = time.time()
        time_cost2 = time_end2 - time_end1
        time_cost_all2 = time_cost_all2 + time_cost2
        print('---------------------------------')
        print('information_calculation: cost ', time_cost2, 's')

        # 显示图像
        if show_img == 1:

            exhibit.img_show(img_list, method_list)

        # 保存图像
        if filepath_write != 0:

            # 保存原图的灰度图
            save_name = 'GREY result from ' + filename
            if filepath_write == 1:
                cv2.imwrite(save_name, img_in)
            else:
                cv2.imwrite(filepath_write + '/' + save_name, img_in)

            # 处理后的图像
            for i in range(method_quantity):
                save_name = method_list[i] + ' result from ' + filename
                if filepath_write == 1:
                    cv2.imwrite(save_name, img_list[i + 1])
                else:
                    cv2.imwrite(filepath_write + '/' + save_name, img_list[i + 1])

        # 计算平均耗时
        time_cost1_ave = time_cost_all1 / file_quantity
        time_cost2_ave = time_cost_all2 / file_quantity

    return time_cost1_ave, time_cost2_ave


if __name__ == '__main__':
    # filepath cannot contain chinese words and space, or image could not be successfully read
    filepath_read = r'D:\58\infrared\before'
    filepath_write = r'D:\58\infrared\after'
    # method for choice:
    method_all = ['TEST_ATD', 'TEST', 'GRAD', 'DIV', 'CANNY'] + \
                 ['HE', 'CLAHE', 'LINEAR', 'GAMMA', 'WT', 'PDE', 'SSR', 'MSR', 'MSRCR'] + \
                 ['GAUSS_NOISE', 'SP_NOISE'] + \
                 ['GAUSS_DENOISE', 'MEDIAN_DENOISE']

    method_list = ['CLAHE', 'TEST', 'GRAD']  # 效果较好的算法：'GRAD', 'CLAHE', 'WT', 'PDE', 'MSRCR'

    '''
    time_ave1, time_ave2 = img_enhancement(filepath_read, filepath_write, method_list, show_information=2, show_img=1)
    print('=====================================================')
    print('time_ave1(for each file)= ', time_ave1, 's')
    print('---------------------------------')
    print('time_ave2(for each file)= ', time_ave2, 's')'''

    # AMBE 绝对平均亮度误差 越小越好
    # PSNR 峰值信噪比 越大越好
    # SSIM 结构相似性 越大越好
    # MSE 均方误差 越小越好

    filepath_read_before = r'D:\58\infrared\before'
    filepath_read_after = r'D:\58\infrared\after'
    img_getanalysis(filepath_read_before, filepath_read_after)
