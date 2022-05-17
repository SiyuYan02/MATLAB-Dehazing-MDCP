function q = gradient_guidedfilter(I, p, eps)  
%   GUIDEDFILTER   O(1) time implementation of guided filter.  
%  
%   - guidance image: I (should be a gray-scale/single channel image)  
%   - filtering input image: p (should be a gray-scale/single channel image)  
%   - regularization parameter: eps    %eps为正则化系数，即可求代价函数的系数，加入惩罚项，用来控制最小二乘法的贡献不大的参数 
                                       %模式识别的思想！
r=32;                             %窗口半径
[hei, wid] = size(I);  
N = boxfilter(ones(hei, wid), r); % 每个窗的大小为N=(2r+1)^2，权值为1，实际上只用到最中间的那个值作为除数平均
  
mean_I = boxfilter(I, r) ./ N;    %均值滤波器，注意这里的盒状滤波与一般的均值滤波不相同， ./ N处理后才是真正的均值
mean_p = boxfilter(p, r) ./ N;  
mean_Ip = boxfilter(I.*p, r) ./ N;   %互相关
cov_Ip = mean_Ip - mean_I .* mean_p; % 协方差 
mean_II = boxfilter(I.*I, r) ./ N;   %自相关
var_I = mean_II - mean_I .* mean_I;  %方差
  
%权重
epsilon=(0.001*(max(p(:))-min(p(:))))^2;  
r1=1;                                   %新的窗口大小
  
N1 = boxfilter(ones(hei, wid), r1); %权值为1的新窗口
mean_I1 = boxfilter(I, r1) ./ N1;   %原图均值滤波
mean_II1 = boxfilter(I.*I, r1) ./ N1;  %自相关
var_I1 = mean_II1 - mean_I1 .* mean_I1;  %方差
  
chi_I=sqrt(abs(var_I1.*var_I));     %标准差 
weight=(chi_I+epsilon)/(mean(chi_I(:))+epsilon); %计算每个像素点的值与平均值相比的大小即为权重  

log_edge_I = edge(I,'log');
log_edge_P = edge(p,'log');
N2 = boxfilter(ones(hei, wid), 3); %权值为1的新窗口
mean_edge1 = boxfilter(I, 3) ./ N2;   %原图均值滤波
mean_edge2 = boxfilter(p, 3) ./ N2;   %原图均值滤波
mean_edgeedge1 = boxfilter(I.*I, 3) ./ N2;  %自相关
mean_edgeedge2 = boxfilter(p.*p, 3) ./ N2;  %自相关
var_edge1 = mean_edgeedge1 - mean_edge1 .* mean_edge1;  %方差
var_edge2 = mean_edgeedge2 - mean_edge2 .* mean_edge2;  %方差
L = log2(max(max(p)) - min(min(p)));
zhengzehua = 0.001*L;
g = (log_edge_I.*var_edge1 + zhengzehua)./(log_edge_P.*var_edge2 + zhengzehua)./p;


  
gamma = (4/(mean(chi_I(:))-min(chi_I(:))))*(chi_I-mean(chi_I(:)));  %惩罚项系数，标准差越大，此处的系数越小
gamma = 1 - 1./(1 + exp(gamma));  %此处用岭回归代替线性回归
%其中gamma称为正则化参数，如果gamma选取过大，会把所有参数θ均最小化，造成线性方程欠拟合
%如果gamma选取过小，会导致对过拟合问题解决不当，因此gamma的选取是一个技术活。 
%称之为正则项，这一项可以看成是对A的各个元素，即各个特征的权的总体的平衡程度，也就是权之间的方差。

gamma = gamma./g;

a = (cov_Ip + (eps./weight).*gamma) ./ (var_I + (eps./weight));  %计算得系数a 注意eps越大惩罚项越大
b = mean_p - a .* mean_I;   %计算得系数b
  
mean_a = boxfilter(a, r) ./ N;  %对a中值滤波
mean_b = boxfilter(b, r) ./ N;  %对b中值滤波
  
q = mean_a .* I + mean_b;       %由计算精细导向图
end  
