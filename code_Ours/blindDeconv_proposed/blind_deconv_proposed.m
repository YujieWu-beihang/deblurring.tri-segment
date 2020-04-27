function [S, k, opts] = blind_deconv_proposed(blur_B, k, opts, s)

% derivative filters
dx = [-1 1; 0 0];
dy = [-1 0; 1 0];
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
H = size(blur_B,1);    W = size(blur_B,2);
blur_B_w = wrap_boundary_liu(blur_B, opt_fft_size([H W]+size(k)-1));

blur_B_tmp = blur_B_w(1:H,1:W,:);
Bx = conv2(blur_B_tmp, dx, 'valid');
By = conv2(blur_B_tmp, dy, 'valid');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%

N_TwoColor = 5;

convergence = 0;

for iter = 1:opts.xk_iter
    
    if(opts.with_TCL)
        for iter_stage1 = 1:1
            % latent_img stage1
            [S_stage1, c1, c2] = TwoColorDeblur_stage1(blur_B_w, k, opts.lambda_grad, 2.0);
            % [S_stage1, c1, c2] = TwoColorDeblur_stage1_fast(blur_B_w, k, opts.lambda_grad, 2.0);

            S_stage1 = S_stage1(1:H,1:W,:);
            S_stage1(S_stage1<0) = 0;
            S_stage1(S_stage1>1) = 1;

            % kernel sub-problem   
            [k, opts] = estimate_psf_whole(Bx, By, blur_B, S_stage1, k, opts);
            k_stage1 = k;

            if opts.lambda_grad~=0;
                opts.lambda_grad = max(opts.lambda_grad/1.1, 1e-4);
            else
                opts.lambda_grad = 0;
            end
        end
    else
        c1 = 0; c2 = 0;
    end

    for iter_stage2 = 1:1

        S_stage2 = ThreeDeblur(blur_B_w, k, c1, c2, opts, 2.0);
%         S_stage2 = ThreeDeblur_long(blur_B_w, k, c1, c2, opts, 2.0);

        S_stage2 = S_stage2(1:H,1:W,:);
        S_stage2(S_stage2<0) = 0;
        S_stage2(S_stage2>1) = 1;

        % kernel sub-problem   
        [k, opts] = estimate_psf_whole(Bx, By, blur_B, S_stage2, k, opts);
        k_stage2 = k;

        if opts.lambda_grad~=0;
            opts.lambda_grad = max(opts.lambda_grad/1.1, 1e-4);
        else
            opts.lambda_grad = 0;
        end

        if opts.lambda_pixel~=0;
            opts.lambda_pixel = max(opts.lambda_pixel/1.1, 1e-4);
        else
            opts.lambda_pixel = 0;
        end
    end

    S = S_stage2;

    if(opts.showProcess)
        figure(opts.h_main);
        subplot(1,3,1); imshow(blur_B,[]); title('Blurred image');
        subplot(1,3,2); imshow(S,[]);title('Interim latent image');
        subplot(1,3,3); imshow(k,[]);title('Estimated kernel');

        drawnow;
    end
    
    if(convergence)
        index = (opts.num_scales-s)*opts.xk_iter + iter;
        k_convergence = imresize(k, size(opts.kernelTruth)+20);
        c = normxcorr2(opts.kernelTruth, k_convergence);
        opts.KM(index) = max(c(:));
        
        Ik = conv2(S,k,'same');
        Ik = imresize(Ik, size(opts.imgBlur));
        opts.Energy(index) = sum(sum((Ik-opts.imgBlur).^2));
    end

    fprintf('iter : %d is done \n', iter);
    
end

% save KM_convergency KM;
% save Energy_convergency Energy;

end



function [k, opts] = estimate_psf_whole(Bx, By, blur_B, S, k, opts)

dx = [-1 1; 0 0];
dy = [-1 0; 1 0];

[latent_x, latent_y, opts.threshold]= threshold_pxpy_v1(S,max(size(k)),opts.threshold); 

k_prev = k;
%% using FFT method for estimating kernel 
k = estimate_psf(Bx, By, latent_x, latent_y, 2, size(k_prev));  
% k = estimate_psf_irls_cg_k0(Bx, By, latent_x, latent_y, opts.k_reg_wt, size(k_prev));

CC = bwconncomp(k,4);
for ii=1:CC.NumObjects
    currsum=sum(k(CC.PixelIdxList{ii}));
    if currsum<.1 
    k(CC.PixelIdxList{ii}) = 0;
    end
end
k(k<0) = 0;
k=k/sum(k(:));

end