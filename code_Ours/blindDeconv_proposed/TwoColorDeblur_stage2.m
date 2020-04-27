function [S, c1, c2] = TwoColorDeblur_stage2(Im, kernel, c1, c2, lambda_grad, kappa)

if ~exist('kappa', 'var')
    kappa = 2.0;
end

S = Im;
fx = [1, -1];
fy = [1; -1];
[H,W,D] = size(Im);
sizeI2D = [H,W];
otfFx = psf2otf(fx,sizeI2D);
otfFy = psf2otf(fy,sizeI2D);
KER = psf2otf(kernel,sizeI2D);
Den_KER = abs(KER).^2;
Denormin2 = abs(otfFx).^2 + abs(otfFy).^2;
Normin1 = conj(KER).*fft2(S);

lambda_i = lambda_grad;
beta_i = 1e-4;
beta_i_max = 1e4;
beta_g_max = 1e3;

length_side = 7;

% if isempty(gcp('nocreate'))==1
%     p = parpool('local',4);
% end

while beta_i<beta_i_max
    % S_block = block_seg(S, length_side);
    S_block{1} = S;
    u_block = S_block(:);
    u_block_tmp = u_block;

    % update u
    for index = 1:length(u_block)
        u = u_block{index};
        c_star = u;
        tmp_abs1 = abs(u-c1{index});
        tmp_abs2 = abs(u-c2{index});
        t = (tmp_abs1 <= tmp_abs2);
        c_star(t) = c1{index};
        t = (tmp_abs1 > tmp_abs2);
        c_star(t) = c2{index};
        % clear tmp_abs1 tmp_abs2 t;
        t_tmp1_1 = (u > c1{index});
        t_tmp1_2 = (u < c2{index});
        t_tmp1 = t_tmp1_1 & t_tmp1_2;
        tmp = beta_i * (c_star-u).^2;
        t_tmp2 = tmp < lambda_i;
        t = t_tmp1 & t_tmp2;
        u(t) = c_star(t);
        % clear t_tmp1_1 t_tmp1_2 tmp t_tmp2 t;
        u_block_tmp{index} = u;
    end
    u_block = u_block_tmp;
    tmp = cell2mat(reshape(u_block, size(S_block)));
    u = tmp(1:H, 1:W, :);

    % updata x
    % ||kx-y||^2 + beta_i*||x-u||^2 + beta_g*||\nabla(x)-g||^2 + lambda_grad*||g||_0
    beta_g = lambda_grad;
    while beta_g<beta_g_max
        Denormin = Den_KER + beta_g*Denormin2 + beta_i;
        h = [diff(S,1,2), S(:,1,:) - S(:,end,:)];
        v = [diff(S,1,1); S(1,:,:) - S(end,:,:)];
        t = (h.^2+v.^2)<lambda_grad/beta_g;
        h(t)=0; v(t)=0;
        % clear t;
        Normin2 = [h(:,end,:) - h(:, 1,:), -diff(h,1,2)];
        Normin2 = Normin2 + [v(end,:,:) - v(1, :,:); -diff(v,1,1)];
        FS = (Normin1 + beta_g*fft2(Normin2) + beta_i*fft2(u))./Denormin;
        S = real(ifft2(FS));
        S(S<0) = 0;
        S(S>1) = 1;
        beta_g = beta_g*kappa;
    end

    beta_i = beta_i * kappa;
end

end


