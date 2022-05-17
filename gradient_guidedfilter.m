function q = gradient_guidedfilter(I, p, eps)  
%   GUIDEDFILTER   O(1) time implementation of guided filter.  
%  
%   - guidance image: I (should be a gray-scale/single channel image)  
%   - filtering input image: p (should be a gray-scale/single channel image)  
%   - regularization parameter: eps    %epsΪ����ϵ������������ۺ�����ϵ��������ͷ������������С���˷��Ĺ��ײ���Ĳ��� 
                                       %ģʽʶ���˼�룡
r=32;                             %���ڰ뾶
[hei, wid] = size(I);  
N = boxfilter(ones(hei, wid), r); % ÿ�����Ĵ�СΪN=(2r+1)^2��ȨֵΪ1��ʵ����ֻ�õ����м���Ǹ�ֵ��Ϊ����ƽ��
  
mean_I = boxfilter(I, r) ./ N;    %��ֵ�˲�����ע������ĺ�״�˲���һ��ľ�ֵ�˲�����ͬ�� ./ N�������������ľ�ֵ
mean_p = boxfilter(p, r) ./ N;  
mean_Ip = boxfilter(I.*p, r) ./ N;   %�����
cov_Ip = mean_Ip - mean_I .* mean_p; % Э���� 
mean_II = boxfilter(I.*I, r) ./ N;   %�����
var_I = mean_II - mean_I .* mean_I;  %����
  
%Ȩ��
epsilon=(0.001*(max(p(:))-min(p(:))))^2;  
r1=1;                                   %�µĴ��ڴ�С
  
N1 = boxfilter(ones(hei, wid), r1); %ȨֵΪ1���´���
mean_I1 = boxfilter(I, r1) ./ N1;   %ԭͼ��ֵ�˲�
mean_II1 = boxfilter(I.*I, r1) ./ N1;  %�����
var_I1 = mean_II1 - mean_I1 .* mean_I1;  %����
  
chi_I=sqrt(abs(var_I1.*var_I));     %��׼�� 
weight=(chi_I+epsilon)/(mean(chi_I(:))+epsilon); %����ÿ�����ص��ֵ��ƽ��ֵ��ȵĴ�С��ΪȨ��  

log_edge_I = edge(I,'log');
log_edge_P = edge(p,'log');
N2 = boxfilter(ones(hei, wid), 3); %ȨֵΪ1���´���
mean_edge1 = boxfilter(I, 3) ./ N2;   %ԭͼ��ֵ�˲�
mean_edge2 = boxfilter(p, 3) ./ N2;   %ԭͼ��ֵ�˲�
mean_edgeedge1 = boxfilter(I.*I, 3) ./ N2;  %�����
mean_edgeedge2 = boxfilter(p.*p, 3) ./ N2;  %�����
var_edge1 = mean_edgeedge1 - mean_edge1 .* mean_edge1;  %����
var_edge2 = mean_edgeedge2 - mean_edge2 .* mean_edge2;  %����
L = log2(max(max(p)) - min(min(p)));
zhengzehua = 0.001*L;
g = (log_edge_I.*var_edge1 + zhengzehua)./(log_edge_P.*var_edge2 + zhengzehua)./p;


  
gamma = (4/(mean(chi_I(:))-min(chi_I(:))))*(chi_I-mean(chi_I(:)));  %�ͷ���ϵ������׼��Խ�󣬴˴���ϵ��ԽС
gamma = 1 - 1./(1 + exp(gamma));  %�˴�����ع�������Իع�
%����gamma��Ϊ���򻯲��������gammaѡȡ���󣬻�����в����Ⱦ���С����������Է���Ƿ���
%���gammaѡȡ��С���ᵼ�¶Թ�������������������gamma��ѡȡ��һ������� 
%��֮Ϊ�������һ����Կ����Ƕ�A�ĸ���Ԫ�أ�������������Ȩ�������ƽ��̶ȣ�Ҳ����Ȩ֮��ķ��

gamma = gamma./g;

a = (cov_Ip + (eps./weight).*gamma) ./ (var_I + (eps./weight));  %�����ϵ��a ע��epsԽ��ͷ���Խ��
b = mean_p - a .* mean_I;   %�����ϵ��b
  
mean_a = boxfilter(a, r) ./ N;  %��a��ֵ�˲�
mean_b = boxfilter(b, r) ./ N;  %��b��ֵ�˲�
  
q = mean_a .* I + mean_b;       %�ɼ��㾫ϸ����ͼ
end  
