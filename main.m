clear
clc
file_path ='D:\lin_dataset\new\';% 图像文件夹路径
save_path='D:\58\infrared\ours';
img_path_list = dir(strcat(file_path,'*.jpg'));
img_num = length(img_path_list);%获取图像总数量
if img_num > 0 %有满足条件的图像
    for j = 1:img_num %逐一读取图像
        image_name  = img_path_list(j).name;% 图像名
        image=imread(strcat(file_path,image_name));
        mysize=size(image);
        if numel(mysize)>2
            image=rgb2gray(image); %将彩色图像转换为灰度图像
        end
        defogging_image = Defogg(image);
        %调整亮度，若需要可使用，K调整提亮程度
        K=0.2;
        defogging_image=defogging_image+(1-defogging_image).*defogging_image.*K;

        filepath=pwd; %保存当前工作目录
        cd(save_path) %把当前工作目录切换到图片存储文件夹
        imwrite(defogging_image,image_name)
        cd(filepath) %切回原工作目录
     end
end