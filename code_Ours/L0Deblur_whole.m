function S = L0Deblur_whole(Im, kernel, lambda_pixel, lambda_grad, kappa)
%%
% Image restoration with L0 regularized intensity and gradient prior
% The objective function:
% S = argmin ||I*k - B||^2 + \lambda_pixel |I|_0 + lambda_grad |\nabla I|_0
%% Input:
% @Im: Blurred image
% @kernel: blur kernel
% @lambda_pixel: weight for the L0 intensity prior
% @lambda_grad: weight for the L0 gradient prior
% @kappa: Update ratio in the ADM
%% Output:
% @S: Latent image
%
% The Code is created based on the method described in the following paper 
%   [1] Jinshan Pan, Zhe Hu, Zhixun Su, and Ming-Hsuan Yang,
%        Deblurring Text Images via L0-Regularized Intensity and Gradient
%        Prior, CVPR, 2014. 
%   [2] Li Xu, Cewu Lu, Yi Xu, and Jiaya Jia. Image smoothing via l0 gradient minimization.
%        ACM Trans. Graph., 30(6):174, 2011.
%
%   Author: Jinshan Pan (sdluran@gmail.com)
%   Date  : 05/18/2014


if ~exist('kappa','var')
    kappa = 2.0;
end
%% pad image
% H = size(Im,1);    W = size(Im,2);
% Im = wrap_boundary_liu(Im, opt_fft_size([H W]+size(kernel)-1));
%%
S = Im;
miu_max = 1e5;
fx = [1, -1];
fy = [1; -1];
[N,M,D] = size(Im);
sizeI2D = [N,M];
otfFx = psf2otf(fx,sizeI2D);
otfFy = psf2otf(fy,sizeI2D);
%%
KER = psf2otf(kernel,sizeI2D);
Den_KER = abs(KER).^2;
%%
Denormin2 = abs(otfFx).^2 + abs(otfFy ).^2;
if D>1
    Denormin2 = repmat(Denormin2,[1,1,D]);
    KER = repmat(KER,[1,1,D]);
    Den_KER = repmat(Den_KER,[1,1,D]);
end
Normin1 = conj(KER).*fft2(S);
%% pixel sub-problem
beta = 2*lambda_pixel;
%beta = 0.01;
beta_max = 2e3;
while beta< beta_max
    %% pixel l0-norm i.e., (u-S).^2 + lambda_pixel/beta*\|u\|_0
    u = S;
    if D==1
        t = u.^2<lambda_pixel/beta;
    else
        t = sum(u.^2,3)<lambda_pixel/beta;
        t = repmat(t,[1,1,D]);
    end
    u(t) = 0;
    clear t;
    %% Gradient sub-problem
    miu = 2*lambda_grad;
    %miu = 0.01;
    while miu < miu_max
        Denormin = Den_KER + miu*Denormin2 + beta;
        % h-v subproblem
        h = [diff(S,1,2), S(:,1,:) - S(:,end,:)];
        v = [diff(S,1,1); S(1,:,:) - S(end,:,:)];
        if D==1
            t = (h.^2+v.^2)<lambda_grad/miu;
        else
            t = sum((h.^2+v.^2),3)<lambda_grad/miu;
            t = repmat(t,[1,1,D]);
        end
        h(t)=0; v(t)=0;
        clear t;
        % S subproblem
        Normin2 = [h(:,end,:) - h(:, 1,:), -diff(h,1,2)];
        Normin2 = Normin2 + [v(end,:,:) - v(1, :,:); -diff(v,1,1)];
        %Normin2 = u;%% for pixel
        FS = (Normin1 + miu*fft2(Normin2) + beta*fft2(u))./Denormin;
        S = real(ifft2(FS));
        miu = miu*kappa;
        if lambda_grad==0
            break;
        end
    end
    beta = beta*kappa;
end
end
