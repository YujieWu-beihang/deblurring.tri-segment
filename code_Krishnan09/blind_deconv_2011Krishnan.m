function [imgLatent, kernel] = blind_deconv_2011Krishnan(imgBlur,opts)

% Do multi-scale blind deconvolution 
% minimize ||x*k-y||^2 + lambda * ||x||
% Input : blur image & opts
% output : kernel-estimate

% gamma correct
imgBlur = imgBlur.^opts.gammaCorrect;

if(size(imgBlur,3)==3)
    imgGray = rgb2gray(imgBlur);
else
    imgGray = imgBlur;
end

% set pixel value to [0,1]
imgGray = im2double(imgGray);

% set kernel size for coarsest level - must be odd
minsize = max(3, 2*floor(((opts.kernelSize - 1)/16)) + 1);
fprintf('Kernel size at coarsest level is %d\n', minsize);

% derivative filters
dx = [-1 1; 0 0];
dy = [-1 0; 1 0];

l2norm = 6;

resizeStep = sqrt(2);
% determine number of scales
numScale = 1;
tmp = minsize;
while(tmp < opts.kernelSize)
    ksize(numScale) = tmp;
    numScale = numScale + 1;
    tmp = ceil(tmp * resizeStep);
    if(mod(tmp,2)==0)
        tmp = tmp + 1;
    end
end
ksize(numScale) = opts.kernelSize;

for s = 1:numScale
    %% kernel module
    if(s==1)
        % at coarest level, initialize kernel
        ks{s} = init_kernel(ksize(1));
        kLength = ksize(1); % always square kernel assumed
    else
        % upsample kernel from previous level to next finer level
        kLength = ksize(s);
        % resize kernel from previous level
        tmp = ks{s-1};
        tmp(tmp<0) = 0;
        tmp = tmp./sum(tmp(:));
        ks{s} = imresize(tmp, [kLength kLength], 'bilinear');
        % bilinear interpolantion no guaranteed to sum to 1 - so
        % renormalize
        ks{s}(ks{s}<0) = 0;
        ks{s} = ks{s}./sum(ks{s}(:));
    end
    
    %% blur image module
    % resize image at the same level
    if(s~=numScale)
        imgRow = floor(size(imgGray,1) * kLength / opts.kernelSize);
        imgCol = floor(size(imgGray,2) * kLength / opts.kernelSize);
    else
        imgRow = size(imgGray,1);
        imgCol = size(imgGray,2);
    end
    
    fprintf('Processing scale %d/%d; kernel size %dx%d; image size %dx%d\n', ...
            s, numScale, kLength, kLength, imgRow, imgCol);
        
    imgScale = imresize(imgGray, [imgRow imgCol], 'bilinear');
    imgGradX = conv2(imgScale, dx, 'valid');
    imgGradY = conv2(imgScale, dy, 'valid');
    
    imgCol = min(size(imgGradX,2), size(imgGradY,2));
    imgRow = min(size(imgGradX,1), size(imgGradY,1));
    
    y = [imgGradX imgGradY]; % this variable is used as observed blur image in every scale
    
    % normalize to have l2 norm of a certain size
    tmp1 = imgGradX;
    tmp1 = tmp1 * l2norm / norm(tmp1(:));
    tmp2 = imgGradY;
    tmp2 = tmp2 * l2norm / norm(tmp2(:));
    y = [tmp1 tmp2];
    
    
    %% latent(sharp) image module
    if(s==1)
        xs{s} = y;
    else
        % upscale the estimated derivative image from previous level
        coltmp = size(xs{s-1},2) / 2;
        tmp1 = xs{s-1}(:,1:coltmp);
        tmp1_up = imresize(tmp1, [imgRow imgCol], 'bilinear');
        tmp2 = xs{s-1}(:,coltmp+1:end);
        tmp2_up = imresize(tmp2, [imgRow imgCol], 'bilinear');
        xs{s} = [tmp1_up tmp2_up];
    end
    
    % normalize to have l2 norm of a certain size
    tmp1 = xs{s}(:,1:imgCol);
    tmp1 = tmp1 * l2norm / norm(tmp1(:));
    tmp2 = xs{s}(:,imgCol+1:end);
    tmp2 = tmp2 * l2norm / norm(tmp2(:));
    xs{s} = [tmp1 tmp2];
    
    %% x(latent image) update
    [xs{s} ks{s}] = blind_deconv_main(y, xs{s}, ks{s}, opts.lambda, opts.delta, ...
                                opts.x_in_iter, opts.x_out_iter, opts.xk_iter, opts.k_reg_wt);
    
    
    %% center the kernel
    tmp1 = xs{s}(:,1:imgCol);
    tmp2 = xs{s}(:,imgCol+1:end);
    [tmp1_shift tmp2_shift ks{s}] = center_kernel_separate(tmp1,tmp2,ks{s});
    xs{s} = [tmp1_shift tmp2_shift];

    % set elements below threshold to 0
    if (s == numScale)
        kernel = ks{s};
        kernel(kernel(:) < opts.kernelThresh * max(kernel(:))) = 0;
        kernel = kernel / sum(kernel(:));
    end;
        
end

imgLatent = xs{s};

end

function [k] = init_kernel(minsize)
  k = zeros(minsize, minsize);
  k((minsize - 1)/2, (minsize - 1)/2:(minsize - 1)/2+1) = 1/2;
end