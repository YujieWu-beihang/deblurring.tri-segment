function [S, k, opts] = blind_deconv_main(blur_B, k, opts)
% Do single-scale blind deconvolution using the input initializations
% 
% I and k. The cost function being minimized is: min_{I,k}
%  |B - I*k|^2  + \gamma*|k|_2 + opts.lambda_pixel*|I|_0 + opts.lambda_grad*|\nabla I|_0
%
%% Input:
% @blur_B: input blurred image 
% @k: blur kernel
% @opts.lambda_pixel: the weight for the L0 regularization on intensity
% @opts.lambda_grad: the weight for the L0 regularization on gradient
%
% Ouput:
% @k: estimated blur kernel 
% @S: intermediate latent image
%
% The Code is created based on the method described in the following paper 
%        Jinshan Pan, Zhe Hu, Zhixun Su, and Ming-Hsuan Yang,
%        Deblurring Text Images via L0-Regularized Intensity and Gradient
%        Prior, CVPR, 2014. 

%   Author: Jinshan Pan (sdluran@gmail.com)
%   Date  : 05/18/2014
%=====================================
%% Note: 
% v4.0 add the edge-thresholding 
%=====================================


% derivative filters
dx = [-1 1; 0 0];
dy = [-1 0; 1 0];
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 2013-08-11
H = size(blur_B,1);    W = size(blur_B,2);
blur_B_w = wrap_boundary_liu(blur_B, opt_fft_size([H W]+size(k)-1));

blur_B_tmp = blur_B_w(1:H,1:W,:);
Bx = conv2(blur_B_tmp, dx, 'valid');
By = conv2(blur_B_tmp, dy, 'valid');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%

salientEdge_Use = 0;

S = blur_B;
for iter = 1:opts.xk_iter
    

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % latent_img sub-problem
    
    %% Modified on 2013-08-27
    if opts.lambda_pixel~=0
%         S = L0Deblur_whole(blur_B_w, k, opts.lambda_pixel, opts.lambda_grad, 2.0);
        S = L0Deblur_whole_fast(blur_B_w, k, opts.lambda_pixel, opts.lambda_grad, 2.0);
    else
       %% L0 deblurring
       S = L0Restoration(blur_B, k, opts.lambda_grad, 2.0);
    end

    S = S(1:H,1:W,:);
    S(S<0) = 0;
    S(S>1) = 1;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % kernel sub-problem  
    
    if(salientEdge_Use)
        % use salientEdge
        [S_salientEdge, latent_x, latent_y] = SalientEdge(blur_B, S, opts.lambda_texture, opts.edge_thresh, opts.dt);
        opts.dt = max(opts.dt/1.1, 1e-4);
        opts.lambda_texture = max(opts.lambda_texture/1.1, 1e-4);
        opts.edge_thresh = max(opts.edge_thresh/1.1, 1e-4);
    else
%         use original latent image
        latent_x = conv2(S, dx, 'valid');
        latent_y = conv2(S, dy, 'valid');
    end

    k_prev = k;
    %% using FFT method for estimating kernel 
    k = estimate_psf(Bx, By, latent_x, latent_y, 2, size(k_prev));  
%     k = estimate_psf_irls_cg_k0(Bx, By, latent_x, latent_y, opts.k_reg_wt, size(k_prev));

    %%
    fprintf('iter:%d, pruning isolated noise in kernel...\n', iter);
    CC = bwconncomp(k,4);
    for ii=1:CC.NumObjects
        currsum=sum(k(CC.PixelIdxList{ii}));
        if currsum<.1 
        k(CC.PixelIdxList{ii}) = 0;
        end
    end

    k(k<0) = 0;
    k=k/sum(k(:));
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
  %% Parameter updating
    if opts.lambda_pixel~=0;
        opts.lambda_pixel = max(opts.lambda_pixel/1.1, 1e-4);
    else
        opts.lambda_pixel = 0;
    end

    if opts.lambda_grad~=0;
        opts.lambda_grad = max(opts.lambda_grad/1.1, 1e-4);
    else
        opts.lambda_grad = 0;
    end

    S(S<0) = 0;
    S(S>1) = 1;

    subplot(2,2,1); imshow(blur_B,[]); title('Blurred image');
    subplot(2,2,2); imshow(S,[]);title('Interim latent image');
    if(salientEdge_Use)
        subplot(2,2,3); imshow(S_salientEdge,[]);title('Salient Edge');
    end
    subplot(2,2,4); imshow(k,[]);title('Estimated kernel');
    drawnow;
end;
