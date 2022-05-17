function imDst = boxfilter(imSrc, r)  
  
%   BOXFILTER   O(1) time box filtering using cumulative sum  
%  
%   - Definition imDst(x, y)=sum(sum(imSrc(x-r:x+r,y-r:y+r)));  
%   - Running time independent of r;   
%   - Equivalent to the function: colfilt(imSrc, [2*r+1, 2*r+1], 'sliding', @sum);  
%   - But much faster.  
  
[hei, wid] = size(imSrc);   %计算图像大小
imDst = zeros(size(imSrc));  %创建图像等大矩阵
  
%cumulative sum over Y axis  
imCum = cumsum(imSrc, 1);  %计算各行的累加值
%difference over Y axis  基于积分图的迭代求累积，好处是只遍历一遍
imDst(1:r+1, :) = imCum(1+r:2*r+1, :);  %第一个窗就是正常累加
imDst(r+2:hei-r, :) = imCum(2*r+2:hei, :) - imCum(1:hei-2*r-1, :);  %中间的窗要减去两个边缘的影响
imDst(hei-r+1:hei, :) = repmat(imCum(hei, :), [r, 1]) - imCum(hei-2*r:hei-r-1, :);  %分块计算
%语法有B = repmat(imCum,m,n)，将矩阵imCum复制 r×1 块，即把 imCum作为 B 的元素，B 由 r×1 个imCum平铺而成。
%减去前面和中间的剩下最后的
%cumulative sum over X axis  
imCum = cumsum(imDst, 2);  %计算各列的累加值
%difference over X axis  
imDst(:, 1:r+1) = imCum(:, 1+r:2*r+1);  
imDst(:, r+2:wid-r) = imCum(:, 2*r+2:wid) - imCum(:, 1:wid-2*r-1);  
imDst(:, wid-r+1:wid) = repmat(imCum(:, wid), [1, r]) - imCum(:, wid-2*r:wid-r-1);  
end  
