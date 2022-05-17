file_path ='D:\lin_dataset\new\';% 图像文件夹路径
save_path='D:\lin_dataset\old';
img_path_list = dir(strcat(file_path,'*.png'));
img_num = length(img_path_list);%获取图像总数量

num = [];

if img_num > 0 %有满足条件的图像
    for j = 1:img_num %逐一读取图像
        disp(j)
        image_name  = img_path_list(j).name;% 图像名
        image=imread(strcat(file_path,image_name));
        mysize=size(image);
%         if numel(mysize)>2
%             image=rgb2gray(image); %将彩色图像转换为灰度图像
%         end
        image = image./255;
        jdark = tongji_idark(image);
        jdark=mat2gray(jdark);
        filepath=pwd; %保存当前工作目录
        cd(save_path) %把当前工作目录切换到图片存储文件夹
        imwrite(jdark,image_name)
        cd(filepath) %切回原工作目录
    end
end
final(:)
N=histogram(final,[0,25,50,75,100,125,150,175,200,225,250]);
% % print(N)
% 
% % data = [0,0,0,0,0,0.1324,0.1133,0.6794,0,0,0];
% % b = bar(data,0.9);
% % % ch = get(b);
% % set(b,'Facecolor','k')
% % % set(b,width,2)
% % % color_matrix = [0,0,0]; 
% % set(gca,'XTickLabel',{'0','','50','','100','','150','','200','','250'})
% % 
% % xlabel('\fontname{Times New Roman}\fontsize{13}intensity'); 
% % ylabel('\fontname{Times New Roman}\fontsize{13}probability'); 
% 
% x=0:25:250;%x轴上的数据，第一个值代表数据开始，第二个值代表间隔，第三个值代表终止
% y=[0,0,0,0,0.01,0.1324,0.2457,0.9251,1,1,1]; %a数据y值
% a=linspace(min(x),max(x)); %插值后将散点连线平滑化
% b=interp1(x,y,a,'cubic');
% %标记点选取还需改进，现在的方法太麻烦
% plot(a,b,'Color',[0 0 0],'LineWidth',1.5)%画ab对应曲线，颜色，标记类型，标记填充颜色，粗细，选取的标记点
% 
% % plot(x,y,'-k'); %线性，颜色，标记
% % axis([0,1000,0,1])  %确定x轴与y轴框图大小
% set(gca,'XTickLabel',{'0','','50','','100','','150','','200','','250'})
% set(gca,'YTick',[0:0.1:1]) %y轴范围0-700，间隔100
% % legend('Neo4j','MongoDB');   %右上角标注
% xlabel('\fontname{Times New Roman}\fontsize{13}intensity'); 
% ylabel('\fontname{Times New Roman}\fontsize{13}cumulative probability'); 


