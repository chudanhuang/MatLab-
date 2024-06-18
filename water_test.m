function restored_image = water_test(image_path)
    % 读取图像
    I = im2double(imread(image_path));
    
    % 参数设置
    t0 = 0.5; % 最小透射率
    
    % 估计水体光 A
    A = estimate_airlight(I);
    
    % 打印估计的水下光 A
    disp('Estimated underwater light:');
    disp(A);
    
    % 计算绿色和蓝色通道相对衰减系数
    lambda_G = A(2) / (1 - A(1));
    lambda_B = A(3) / (1 - A(1));
    
    % 估计透射率
    t_R = estimate_transmission(I, A);
    
    % 计算绿色和蓝色通道透射率
    t_G = t_R .^ lambda_G;
    t_B = t_R .^ lambda_B;
    
    % 精细化透射率图
    t_R_refined = guided_filter(I, t_R, 15, 1e-6);
    t_G_refined = guided_filter(I, t_G, 15, 1e-6);
    t_B_refined = guided_filter(I, t_B, 15, 1e-6);
    
    % 色彩校正
    J_R = (I(:,:,1) - A(1)) ./ max(t_R_refined, t0) + (1 - A(1)) * A(1);
    J_G = (I(:,:,2) - A(2)) ./ max(t_G_refined, t0) + (1 - A(2)) * A(2);
    J_B = (I(:,:,3) - A(3)) ./ max(t_B_refined, t0) + (1 - A(3)) * A(3);
    
    % 确保结果是实数
    J_R = real(J_R);
    J_G = real(J_G);
    J_B = real(J_B);
    
    restored_image = cat(3, J_R, J_G, J_B);
    
    % 色彩平衡
    restored_image = SimplestColorBalance(restored_image);
    
    % 显示结果
    figure;
    subplot(1, 2, 1), imshow(I), title('Original Underwater Image');
    subplot(1, 2, 2), imshow(restored_image), title('Restored Image');
end


function A = estimate_airlight(I)
    % 选择红色通道中最亮的10%像素
    R = I(:,:,1);
    num_pixels = numel(R);
    num_bright_pixels = round(0.4 * num_pixels);
    [sorted_R, ~] = sort(R(:), 'descend');
    threshold = sorted_R(num_bright_pixels);
    bright_pixels = R >= threshold;
    
    % 从中选择红色分量最低的像素
    A = zeros(1, 3);
    for c = 1:3
        channel = I(:,:,c);
        A(c) = mean(channel(bright_pixels));
    end
end

function t = estimate_transmission(I, A)
    % 计算透射率
    R = I(:,:,1);
    G = I(:,:,2);
    B = I(:,:,3);
    t = 1 - min(min((1 - R) / (1 - A(1)), G / A(2)), B / A(3));
end

function q = guided_filter(I, p, r, eps)
        %   GUIDEDFILTER_COLOR   O(1) time implementation of guided filter using a color image as the guidance.
    %
    %   - guidance image: I (should be a color (RGB) image)
    %   - filtering input image: p (should be a gray-scale/single channel image)
    %   - local window radius: r
    %   - regularization parameter: eps
    
    [hei, wid] = size(p);
    N = boxfilter(ones(hei, wid), r); % the size of each local patch; N=(2r+1)^2 except for boundary pixels.
    
    mean_I_r = boxfilter(I(:, :, 1), r) ./ N;
    mean_I_g = boxfilter(I(:, :, 2), r) ./ N;
    mean_I_b = boxfilter(I(:, :, 3), r) ./ N;
    
    mean_p = boxfilter(p, r) ./ N;
    
    mean_Ip_r = boxfilter(I(:, :, 1).*p, r) ./ N;
    mean_Ip_g = boxfilter(I(:, :, 2).*p, r) ./ N;
    mean_Ip_b = boxfilter(I(:, :, 3).*p, r) ./ N;
    
    % covariance of (I, p) in each local patch.
    cov_Ip_r = mean_Ip_r - mean_I_r .* mean_p;
    cov_Ip_g = mean_Ip_g - mean_I_g .* mean_p;
    cov_Ip_b = mean_Ip_b - mean_I_b .* mean_p;
    % Note the variance in each local patch is a 3x3 symmetric matrix:
    %           rr, rg, rb
    %   Sigma = rg, gg, gb
    %           rb, gb, bb
    var_I_rr = boxfilter(I(:, :, 1).*I(:, :, 1), r) ./ N - mean_I_r .*  mean_I_r; 
    var_I_rg = boxfilter(I(:, :, 1).*I(:, :, 2), r) ./ N - mean_I_r .*  mean_I_g; 
    var_I_rb = boxfilter(I(:, :, 1).*I(:, :, 3), r) ./ N - mean_I_r .*  mean_I_b; 
    var_I_gg = boxfilter(I(:, :, 2).*I(:, :, 2), r) ./ N - mean_I_g .*  mean_I_g; 
    var_I_gb = boxfilter(I(:, :, 2).*I(:, :, 3), r) ./ N - mean_I_g .*  mean_I_b; 
    var_I_bb = boxfilter(I(:, :, 3).*I(:, :, 3), r) ./ N - mean_I_b .*  mean_I_b; 
    
    a = zeros(hei, wid, 3);
    for y=1:hei
        for x=1:wid        
            Sigma = [var_I_rr(y, x), var_I_rg(y, x), var_I_rb(y, x);
                var_I_rg(y, x), var_I_gg(y, x), var_I_gb(y, x);
                var_I_rb(y, x), var_I_gb(y, x), var_I_bb(y, x)];
            Sigma = Sigma + eps * eye(3);
            
            cov_Ip = [cov_Ip_r(y, x), cov_Ip_g(y, x), cov_Ip_b(y, x)];        
            
            a(y, x, :) = cov_Ip * ((Sigma + eps) \ eye(3)); 
        end
    end
    
    b = mean_p - a(:, :, 1) .* mean_I_r - a(:, :, 2) .* mean_I_g - a(:, :, 3) .* mean_I_b; 
    
    q = (boxfilter(a(:, :, 1), r).* I(:, :, 1)...
    + boxfilter(a(:, :, 2), r).* I(:, :, 2)...
    + boxfilter(a(:, :, 3), r).* I(:, :, 3)...
    + boxfilter(b, r)) ./ N;  
end
function imDst = boxfilter(imSrc, r)

    %   BOXFILTER   O(1) time box filtering using cumulative sum
    %
    %   - Definition imDst(x, y)=sum(sum(imSrc(x-r:x+r,y-r:y+r)));
    %   - Running time independent of r; 
    %   - Equivalent to the function: colfilt(imSrc, [2*r+1, 2*r+1], 'sliding', @sum);
    %   - But much faster.
    
    [hei, wid] = size(imSrc);
    imDst = zeros(size(imSrc));
    
    %cumulative sum over Y axis
    imCum = cumsum(imSrc, 1);
    %difference over Y axis
    imDst(1:r+1, :) = imCum(1+r:2*r+1, :);
    imDst(r+2:hei-r, :) = imCum(2*r+2:hei, :) - imCum(1:hei-2*r-1, :);
    imDst(hei-r+1:hei, :) = repmat(imCum(hei, :), [r, 1]) - imCum(hei-2*r:hei-r-1, :);
    
    %cumulative sum over X axis
    imCum = cumsum(imDst, 2);
    %difference over Y axis
    imDst(:, 1:r+1) = imCum(:, 1+r:2*r+1);
    imDst(:, r+2:wid-r) = imCum(:, 2*r+2:wid) - imCum(:, 1:wid-2*r-1);
    imDst(:, wid-r+1:wid) = repmat(imCum(:, wid), [1, r]) - imCum(:, wid-2*r:wid-r-1);
end

function outval = SimplestColorBalance(im_org)
    num = 255;
    
    if ndims(im_org) == 3
        R = sum(sum(im_org(:,:,1)));
        G = sum(sum(im_org(:,:,2)));
        B = sum(sum(im_org(:,:,3)));
        Max = max([R, G, B]);
        ratio = [Max / R, Max / G, Max / B];

        satLevel1 = 0.005 * ratio;
        satLevel2 = 0.005 * ratio;
        satLevel1 = min(max(satLevel1, 0), 1);
        satLevel2 = min(max(satLevel2, 0), 1);

        [m, n, p] = size(im_org);
        imRGB_orig = zeros(p, m * n);
        for i = 1 : p
            imRGB_orig(i, :) = reshape(double(im_org(:, :, i)), [1, m * n]);
        end
    else
        satLevel1 = 0.001;
        satLevel2 = 0.005;
        satLevel1 = min(max(satLevel1, 0), 1);
        satLevel2 = min(max(satLevel2, 0), 1);

        [m, n] = size(im_org);
        p = 1;
        imRGB_orig = reshape(double(im_org), [1, m * n]);
    end

    imRGB = zeros(size(imRGB_orig));
    for ch = 1 : p
        q = [satLevel1(ch), 1 - satLevel2(ch)];
        tiles = quantile(imRGB_orig(ch, :), q);
        temp = imRGB_orig(ch, :);
        temp(temp < tiles(1)) = tiles(1);
        temp(temp > tiles(2)) = tiles(2);
        imRGB(ch, :) = temp;
        bottom = min(imRGB(ch, :)); 
        top = max(imRGB(ch, :));
        imRGB(ch, :) = (imRGB(ch, :) - bottom) * num / (top - bottom); 
    end

    if ndims(im_org) == 3
        outval = zeros(size(im_org));
        for i = 1 : p
            outval(:, :, i) = reshape(imRGB(i, :), [m, n]); 
        end
    else
        outval = reshape(imRGB, [m, n]); 
    end
    outval = uint8(outval);
end
  

