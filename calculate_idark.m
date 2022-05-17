function Jdark = tongji_idark( I )
% output： Jdark = min(min(r),min(g),min(b));
Wnd = 15;           %窗口直径，为奇数，半径为此数减一除以二

% 图像拓展，采用补上半径减1的像素方法来处理边缘
[m,n,~] = size(I); 
I_temp = zeros(m+Wnd-1, n+Wnd-1);   %存放全图的矩阵，分RGB三个量，这里不除以二是因为相当于上下都补一个半径的长度；例如直径是5，图大小为4，则图片是从2到5
I_temp((Wnd-1)/2 : m+(Wnd-1)/2-1 , (Wnd-1)/2 : n+(Wnd-1)/2-1 ) = I;%这里是因为相当于新图从0开始计数所以要减一

% 暗通道
for i=1:1:m   %设置行遍历，从1到m，步进为1
    for j=1:1:n  %设置列遍历，从1到m，步进为1
        Imin = max(max (I_temp(i:i+Wnd-1, j:j+Wnd-1) ));  %每次比较的范围都是一个窗的大小例如i=1，wnd=5，则遍历1到5个像素，正好是窗的大小
          %第一次最小求的是列（j）最小的像素点
         %第二次最小求的是已求出每行最小像素点里，行（i）方向上最小的像素点
        Jdark(i,j) = Imin;          %嵌套比大小，比较当前窗里三个分量中最小的
        %这里都是对(i:i+Wnd-1, j:j+Wnd-1)这个区域内的RGB比大小，
        %比完的结果把三个分量中最小的像素值存到Jdark里，为第（i，j）处的值
    end
end
 
end
