import Basic_Module as basic
from pylab import *
from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r"c:\windows\fonts\SimSun.ttc", size=14)  # 加载字体


def img_show(img_array, method_list):
    length, a, b = shape(img_array)
    point_num_all = a * b

    # 显示
    figure()

    for i in range(length):

        if i == 0:
            title_img = 'origin image'
            title_histogram = 'histogram'
        else:
            title_img = method_list[i - 1]
            title_histogram = 'histogram'

        # 显示图像和直方图
        subplot(2, length, i + 1)
        axis('off')
        gray()
        title(title_img, fontproperties=font)
        imshow(img_array[i])

        subplot(2, length, length + i + 1)
        hist(img_array[i].flatten(), 128)
        title(title_histogram, fontproperties=font)
        plt.xlim([0, 255])
        plt.ylim([0, point_num_all / 10])

        img_information = basic.get_important_point(img_array[i])
        hist(img_information.flatten(), 128)
        title(title_histogram, fontproperties=font)
        plt.xlim([0, 255])
        plt.ylim([0, point_num_all / 10])

    show()

    return 0

