function restored_image = dynamic_test(image_path)
    % 读取图像并进行预处理
    I = imread(image_path);
    B = double(I) / 255;  % 归一化到 [0, 1] 范围

    % 逆伽马校正
    gamma = 2.2;
    B_corrected = B .^ (1 / gamma);

    % 计算图像梯度
    [Gx, Gy] = gradient(B_corrected);

    % 模糊核估计参数
    num_scales = 5;
    kernel_size = 15;
    num_iterations = 50;
    K = ones(kernel_size) / kernel_size^2;  % 初始模糊核

    % 开始迭代不同尺度下的模糊核估计
    for scale = num_scales:-1:1
        % 调整图像大小
        scale_factor = (1 / sqrt(2))^(scale - 1);
        B_scaled = imresize(B_corrected, scale_factor, 'bilinear');

        % 计算缩放后图像的梯度
        [Gx_s, Gy_s] = gradient(B_scaled);

        % 模糊核估计
        K = inferKernel(Gx_s, Gy_s, K, num_iterations);

        % 在下一尺度上更新模糊核的大小
        if scale > 1
            K = imresize(K, [kernel_size, kernel_size], 'bilinear');
        end
    end
    % 初始化变分贝叶斯推理的参数
    Lx = Gx;  % 潜在图像的x方向梯度
    Ly = Gy;  % 潜在图像的y方向梯度
    mu_K = K;  % 模糊核的均值
    sigma_K = 0.01 * ones(size(K));  % 模糊核的标准差
    mu_Lx = Lx;  % 图像梯度的均值
    sigma_Lx = 0.01 * ones(size(Lx));  % 图像梯度的标准差
    mu_Ly = Ly;  % 图像梯度的均值
    sigma_Ly = 0.01 * ones(size(Ly));  % 图像梯度的标准差

    % 定义优化参数
    learning_rate = 0.001;
    num_iterations_vb = 25;
    clip_threshold = 1.0;
    lambda_tv = 0.01;  % 总变差正则化权重

    % 变分贝叶斯迭代优化
    for iter = 1:num_iterations_vb
        K_sample = mu_K + sigma_K .* randn(size(K));
        Lx_sample = mu_Lx + sigma_Lx .* randn(size(Lx));
        Ly_sample = mu_Ly + sigma_Ly .* randn(size(Ly));

        % 计算梯度
        [grad_mu_K, grad_sigma_K] = compute_gradients_K(K_sample, B_corrected, Lx_sample, Ly_sample);
        [grad_mu_Lx, grad_sigma_Lx] = compute_gradients_L(Lx_sample, B_corrected, K_sample);
        [grad_mu_Ly, grad_sigma_Ly] = compute_gradients_L(Ly_sample, B_corrected, K_sample);

        % 添加总变差正则化梯度
        grad_tv_Lx = grad_total_variation(Lx_sample);
        grad_tv_Ly = grad_total_variation(Ly_sample);
        grad_mu_Lx = grad_mu_Lx + lambda_tv * grad_tv_Lx;
        grad_mu_Ly = grad_mu_Ly + lambda_tv * grad_tv_Ly;

        % 梯度裁剪
        grad_mu_K = max(min(grad_mu_K, clip_threshold), -clip_threshold);
        grad_sigma_K = max(min(grad_sigma_K, clip_threshold), -clip_threshold);
        grad_mu_Lx = max(min(grad_mu_Lx, clip_threshold), -clip_threshold);
        grad_sigma_Lx = max(min(grad_sigma_Lx, clip_threshold), -clip_threshold);
        grad_mu_Ly = max(min(grad_mu_Ly, clip_threshold), -clip_threshold);
        grad_sigma_Ly = max(min(grad_sigma_Ly, clip_threshold), -clip_threshold);

        % 更新变分分布的参数
        mu_K = mu_K + learning_rate * grad_mu_K;
        sigma_K = sigma_K + learning_rate * grad_sigma_K;
        mu_Lx = mu_Lx + learning_rate * grad_mu_Lx;
        sigma_Lx = sigma_Lx + learning_rate * grad_sigma_Lx;
        mu_Ly = mu_Ly + learning_rate * grad_mu_Ly;
        sigma_Ly = sigma_Ly + learning_rate * grad_sigma_Ly;

        % 计算并输出总变差
        tv_Lx = total_variation(mu_Lx);
        tv_Ly = total_variation(mu_Ly);
        fprintf('Iteration %d: TV_Lx = %f, TV_Ly = %f\n', iter, tv_Lx, tv_Ly);
    end

    % 反卷积使用估计的模糊核
    NSR = 0.1;  % 噪声功率与信号功率的比率

    % 确保模糊核 mu_K 是有限的
    if any(~isfinite(mu_K(:)))
        error('模糊核 mu_K 包含无效值。');
    end

    % 归一化模糊核
    mu_K = mu_K / sum(mu_K(:));
    % RGB通道分别处理
    L_rgb = zeros(size(B_corrected));
    for channel = 1:3
        L_rgb(:, :, channel) = deconvwnr(double(B_corrected(:, :, channel)), mu_K, NSR);
    end

    % 返回复原后的图像
    restored_image = L_rgb;

    % 显示结果（可选）
    figure;
    subplot(1, 2, 1);
    imshow(uint8(L_rgb * 255));
    title('修复的图像');
    
    subplot(1, 2, 2);
    imshow(K);
    title('模糊核');
end


% 模糊核估计函数
function K = inferKernel(Gx_s, Gy_s, K, num_iterations)
    % 模糊核估计的函数
    % Gx_s - 缩放后图像的x方向梯度
    % Gy_s - 缩放后图像的y方向梯度
    % K - 初始模糊核
    % num_iterations - 迭代次数
    
    learning_rate = 0.001;
    
    for iter = 1:num_iterations
        % 如果输入梯度包含颜色通道，分别处理每个通道
        if size(Gx_s, 3) > 1
            B_hat_x = zeros(size(Gx_s));
            B_hat_y = zeros(size(Gy_s));
            for channel = 1:size(Gx_s, 3)
                B_hat_x(:, :, channel) = conv2(Gx_s(:, :, channel), K, 'same');
                B_hat_y(:, :, channel) = conv2(Gy_s(:, :, channel), K, 'same');
            end
        else
            B_hat_x = conv2(Gx_s, K, 'same');
            B_hat_y = conv2(Gy_s, K, 'same');
        end
        
        % 计算误差
        error_x = Gx_s - B_hat_x;
        error_y = Gy_s - B_hat_y;
        
        % 计算梯度
        grad_K_x = zeros(size(K));
        grad_K_y = zeros(size(K));
        if size(Gx_s, 3) > 1
            for channel = 1:size(Gx_s, 3)
                                grad_K_x = grad_K_x + conv2(rot90(error_x(:, :, channel), 2), Gx_s(:, :, channel), 'valid');
                grad_K_y = grad_K_y + conv2(rot90(error_y(:, :, channel), 2), Gy_s(:, :, channel), 'valid');
            end
        else
            grad_K_x = conv2(rot90(error_x, 2), Gx_s, 'valid');
            grad_K_y = conv2(rot90(error_y, 2), Gy_s, 'valid');
        end
        
        % 更新模糊核
        K = K + learning_rate * (grad_K_x + grad_K_y);
        
        % 归一化模糊核
        K = K / sum(K(:));
    end
end

% 计算模糊核梯度的函数
function [grad_mu, grad_sigma] = compute_gradients_K(K_sample, B_corrected, Lx_sample, Ly_sample)
    % 计算模糊核的梯度
    
    if size(B_corrected, 3) > 1
        B_hat_x = zeros(size(B_corrected));
        B_hat_y = zeros(size(B_corrected));
        for channel = 1:size(B_corrected, 3)
            B_hat_x(:, :, channel) = conv2(Lx_sample(:, :, channel), K_sample, 'same');
            B_hat_y(:, :, channel) = conv2(Ly_sample(:, :, channel), K_sample, 'same');
        end
    else
        B_hat_x = conv2(Lx_sample, K_sample, 'same');
        B_hat_y = conv2(Ly_sample, K_sample, 'same');
    end
    
    error = B_corrected - (B_hat_x + B_hat_y);
    
    grad_mu = zeros(size(K_sample));
    if size(B_corrected, 3) > 1
        for channel = 1:size(B_corrected, 3)
            grad_mu = grad_mu + conv2(rot90(Lx_sample(:, :, channel), 2), error(:, :, channel), 'valid');
            grad_mu = grad_mu + conv2(rot90(Ly_sample(:, :, channel), 2), error(:, :, channel), 'valid');
        end
    else
        grad_mu = conv2(rot90(Lx_sample, 2), error, 'valid');
        grad_mu = grad_mu + conv2(rot90(Ly_sample, 2), error, 'valid');
    end
    
    grad_sigma = grad_mu .* K_sample;
end

% 计算图像梯度的梯度函数
function [grad_mu, grad_sigma] = compute_gradients_L(L_sample, B_corrected, K_sample)
    % 计算图像梯度的梯度
    
    if size(B_corrected, 3) > 1
        B_hat = zeros(size(B_corrected));
        for channel = 1:size(B_corrected, 3)
            B_hat(:, :, channel) = conv2(L_sample(:, :, channel), K_sample, 'same');
        end
    else
        B_hat = conv2(L_sample, K_sample, 'same');
    end
    
    % 计算误差
    error = B_corrected - B_hat;
    
    % 计算梯度
    grad_mu = zeros(size(L_sample));
    if size(B_corrected, 3) > 1
        for channel = 1:size(B_corrected, 3)
            grad_mu(:, :, channel) = conv2(error(:, :, channel), rot90(K_sample, 2), 'same');
        end
    else
        grad_mu = conv2(error, rot90(K_sample, 2), 'same');
    end
    
    grad_sigma = grad_mu .* L_sample;
end

% 计算总变差正则化的函数
function tv = total_variation(x)
    % 计算图像的总变差
    [dx, dy] = gradient(x);
    tv = sum(sum(sqrt(dx.^2 + dy.^2)));
end

% 计算总变差正则化梯度的函数
function grad_tv = grad_total_variation(x)
    % 计算总变差的梯度
    if ndims(x) == 3  % 如果输入是三维数据
        grad_tv = zeros(size(x));
        for channel = 1:size(x, 3)
            [dx, dy] = gradient(x(:, :, channel));
            norm_grad = sqrt(dx.^2 + dy.^2 + 1e-8);  % 避免除零
            grad_x = -divergence(dx ./ norm_grad, dy ./ norm_grad);
            grad_tv(:, :, channel) = grad_x;
        end
    else  % 如果输入是二维数据
        [dx, dy] = gradient(x);
        norm_grad = sqrt(dx.^2 + dy.^2 + 1e-8);  % 避免除零
        grad_tv = -divergence(dx ./ norm_grad, dy ./ norm_grad);
    end
end



