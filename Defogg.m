function defogging_image = Defogg(J)
%去雾函数

%读进来是unot8类型，且后面的滤波等操作都要用到浮点运算
J = double(J);
 %归一化
% J = J ./255 ;   

[q,v,~] = size(J);

%求暗通道图像
Jdark = Idark(J);
Jdark=mat2gray(Jdark);
filepath=pwd; %保存当前工作目录
cd('D:\lin_dataset\old') %把当前工作目录切换到图片存储文件夹
imwrite(Jdark,'6.jpg')
cd(filepath) %切回原工作目录


 %天空亮度
A=imguidedfilter(Jdark,Jdark);
Ac=double(max(max(Jdark))); 
A=Ac*0.95+A*0.05;

% 采用梯度导向滤波方法对得到的粗透射率Jdark进行细化,可以加快运算速度，增加透射图细节
 %图片导向滤波后得到新的图片
Jdark = gradient_guidedfilter(Jdark,Jdark,2);  

% 大气物理模型 J = I*t + A*(1-t)  【直接衰减项】+【大气光照】
% 透射率 t与深度的关系 t=exp(-a*depth)
for i = 1:1:q
    for j = 1:1:v
        Jt(i,j)= (A(i,j) - Jdark(i,j))/(180- A(i,j));   %求解透射率
    end
end

% Jt=mat2gray(Jt);
% filepath=pwd; %保存当前工作目录
% cd('D:\58\infrared\ours') %把当前工作目录切换到图片存储文件夹
% imwrite(Jt,'Jt.jpg')
% cd(filepath) %切回原工作目录

Jt=mat2gray(Jt);

Jt = adapthisteq(Jt, 'NumTiles',[16 16] ,'ClipLimit',0.01);



% 求解清晰的图像
% 根据 J = I*t + A*(1-t)   I = (J-A)/Jt + A
t0=0.1;
defogging_image = zeros(q,v);
for i = 1:1:q
    for j = 1:1:v
        defogging_image(i,j) = (J(i,j)-A(i,j)) ./ max(Jt(i,j),t0) + A(i,j);%J为原图，Jt为透射率
    end
end
%归一化
defogging_image=mat2gray(defogging_image);
%后处理
defogging_image = adapthisteq(defogging_image,'NumTiles',[8 16] ,'ClipLimit',0.0018);
end
