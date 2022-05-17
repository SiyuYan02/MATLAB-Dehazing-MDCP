### ENVIRONMENT 

```python
conda python 3.8
opencv-python 4.5.3.56
```

#### 1、Enhancement_Module

```
直方图方法 HE CLAHE 
小波变换方法 WT
偏微分方程方法 PDE
Retinex方法 SSR MSR MSRCR
线性变换方法
伽马变换方法
```



#### 2、Basic_Module

    求梯度图
    求直方图
    求CANNY边缘
    加噪声函数（高斯噪声，椒盐噪声）
    降噪声函数（高斯降噪，中值降噪）
#### 3、Analysis_Module

    标准差  SD
    梯度均值  AVD
    信息熵  IE
    亮度增加值  AMBE
    峰值信噪比  PSNR
    结构相似性  SSIM
    均方根误差  MSE
#### 4、Exhibition_Module

    自动实现分图排版
    显示图片及直方图


### RUN：

```python
main.py
```

 NOTE:

1、 图像文件夹路径中不能出现中文字符，也不能出现空格和非法字符

2、图像大小建议为 偶数*偶数，否则图像很可能无法读入

3、初次使用需要配置几个库文件，其中有库文件有漏洞，修改措施已写于，main.py第10行

4、 初次使用需修改读图和存图文件夹路径，建议使用单斜杠 / 表示路径，不建议使用双反斜杠 \\

5、后续使用中，原则上无需修改main.py以外的内容，主函数和接口函数即可实现绝大部分功能



**The project was produced by the cooperator Zongyuan LI**