function restored_image = fog_work(image_path)
    % 示例调用 fog_restoration 函数

    % 加载图像
    I = im2double(imread(image_path));

    % 设置参数
    patch_size = 7;  % 补丁大小
    t0 = 0.4;  % 透射率下限值

    % 恢复图像
    restored_image = fog_restoration(I, patch_size, t0);

    % 获取估计的大气光
    A = estimate_atmospheric_light(I, patch_size);

    % 显示估计的大气光
    disp('Estimated atmospheric light:');
    disp(A);

    % 显示结果
    figure;
    subplot(1, 2, 1);
    imshow(I);
    title('原始图像');

    subplot(1, 2, 2);
    imshow(restored_image);
    title('恢复后的图像');
end


function restored_image = fog_restoration(I, patch_size, t0)
    % 输入:
    % I - 输入的雾天图像
    % patch_size - 补丁大小，用于计算暗通道
    % t0 - 透射率的最小值，用于防止除以零
    % 输出:
    % restored_image - 恢复后的图像
    
    % 将图像转换为双精度类型
    I = im2double(I);

    % 估计大气光
    A = estimate_atmospheric_light(I, patch_size);

    % 计算透射率
    t = compute_transmission(I, A, patch_size);

    % 确保透射率不小于 t0
    t = max(t, t0);

    % 恢复图像
    restored_image = restore_image(I, A, t);
end

function A = estimate_atmospheric_light(I, patch_size)
    % 估计大气光
    % 输入:
    % I - 输入的图像
    % patch_size - 补丁大小
    % 输出:
    % A - 大气光

    % 计算暗通道
    dark_channel = get_dark_channel(I, patch_size);

    % 选择暗通道中亮度最高的点
    [~, idx] = max(dark_channel(:));
    [i, j] = ind2sub(size(dark_channel), idx);
    
    % 大气光的估计为原始图像中对应位置的亮度值
    A = I(i, j, :);
end

function dark_channel = get_dark_channel(I, patch_size)
    % 计算暗通道
    % 输入: I - 输入图像
    %       patch_size - 补丁大小
    % 输出: dark_channel - 暗通道图像

    % 图像尺寸
    [height, width, ~] = size(I);

    % 初始化暗通道图像
    dark_channel = zeros(height, width);

    % 半补丁大小
    half_patch = floor(patch_size / 2);

    % 遍历图像
    for i = 1:height
        for j = 1:width
            % 计算局部补丁的边界
            i_min = max(1, i - half_patch);
            i_max = min(height, i + half_patch);
            j_min = max(1, j - half_patch);
            j_max = min(width, j + half_patch);

            % 提取局部补丁
            local_patch = I(i_min:i_max, j_min:j_max, :);

            % 计算局部补丁中的最小值
            dark_channel(i, j) = min(local_patch(:));
        end
    end
end

function t = compute_transmission(I, A, patch_size)
    % 计算透射率
    % 输入:
    % I - 输入图像
    % A - 大气光
    % patch_size - 补丁大小
    % 输出:
    % t - 透射率

    % 归一化大气光
    norm_I = bsxfun(@rdivide, I, reshape(A, 1, 1, 3));

    % 计算暗通道
    dark_channel = get_dark_channel(norm_I, patch_size);

    % 计算透射率
    omega = 0.95;  % 雾霾去除因子
    t = 1 - omega * dark_channel;
end

function J = restore_image(I, A, t)
    % 恢复图像
    % 输入:
    % I - 输入图像
    % A - 大气光
    % t - 透射率
    % 输出:
    % J - 恢复后的图像

    % 计算去雾后的图像 J(x)
    J = zeros(size(I));
    for c = 1:3
        J(:, :, c) = (I(:, :, c) - A(c)) ./ t + A(c);
    end
    
    % 确保结果在 [0, 1] 范围内
    J = max(min(J, 1), 0);
end
