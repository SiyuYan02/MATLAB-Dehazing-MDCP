function imDst = boxfilter(imSrc, r)  
  
%   BOXFILTER   O(1) time box filtering using cumulative sum  
%  
%   - Definition imDst(x, y)=sum(sum(imSrc(x-r:x+r,y-r:y+r)));  
%   - Running time independent of r;   
%   - Equivalent to the function: colfilt(imSrc, [2*r+1, 2*r+1], 'sliding', @sum);  
%   - But much faster.  
  
[hei, wid] = size(imSrc);   %����ͼ���С
imDst = zeros(size(imSrc));  %����ͼ��ȴ����
  
%cumulative sum over Y axis  
imCum = cumsum(imSrc, 1);  %������е��ۼ�ֵ
%difference over Y axis  ���ڻ���ͼ�ĵ������ۻ����ô���ֻ����һ��
imDst(1:r+1, :) = imCum(1+r:2*r+1, :);  %��һ�������������ۼ�
imDst(r+2:hei-r, :) = imCum(2*r+2:hei, :) - imCum(1:hei-2*r-1, :);  %�м�Ĵ�Ҫ��ȥ������Ե��Ӱ��
imDst(hei-r+1:hei, :) = repmat(imCum(hei, :), [r, 1]) - imCum(hei-2*r:hei-r-1, :);  %�ֿ����
%�﷨��B = repmat(imCum,m,n)��������imCum���� r��1 �飬���� imCum��Ϊ B ��Ԫ�أ�B �� r��1 ��imCumƽ�̶��ɡ�
%��ȥǰ����м��ʣ������
%cumulative sum over X axis  
imCum = cumsum(imDst, 2);  %������е��ۼ�ֵ
%difference over X axis  
imDst(:, 1:r+1) = imCum(:, 1+r:2*r+1);  
imDst(:, r+2:wid-r) = imCum(:, 2*r+2:wid) - imCum(:, 1:wid-2*r-1);  
imDst(:, wid-r+1:wid) = repmat(imCum(:, wid), [1, r]) - imCum(:, wid-2*r:wid-r-1);  
end  
