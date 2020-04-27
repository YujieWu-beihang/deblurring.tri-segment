function [S, c1, c2] = TwoColorDeblur_stage1(Im, kernel, lambda_grad, kappa)

if ~exist('kappa', 'var')
    kappa = 2.0;
end

opts = statset('Display', 'final');

S = Im;
fx = [1, -1];
fy = [1; -1];
[H,W,D] = size(S);
sizeI2D = [H,W];
otfFx = psf2otf(fx,sizeI2D);
otfFy = psf2otf(fy,sizeI2D);
KER = psf2otf(kernel,sizeI2D);
Den_KER = abs(KER).^2;
Denormin2 = abs(otfFx).^2 + abs(otfFy).^2;
Normin1 = conj(KER).*fft2(S);

beta_t = 1e-3; 
beta_t_max = 1e3;
beta_g_max = 1e3;

length_side = 5;

while beta_t<beta_t_max

    vx = S;
    v = S;
    v = v(:);
    [Idx, Ctrs, SumD, D] = kmeans(v, 2, 'Replicates', 10);
    c1 = min(Ctrs);
    c2 = max(Ctrs);
    c_mean = mean(Ctrs);
    for l=1:4
        t_1 = (v <= c1);
        t_1mean_tmp1 = (v <= c_mean);
        t_1mean_tmp2 = (v > c1);
        t_1mean = t_1mean_tmp1 & t_1mean_tmp2;

        t_2 = (v >= c2);
        t_mean2_tmp1 = (v >= c_mean);
        t_mean2_tmp2 = (v < c2);
        t_mean2 = t_mean2_tmp1 & t_mean2_tmp2;

        Normin_c1 = 10*sum(v(t_1)) + sum(v(t_1mean));
        Denormin_c1 = 10*numel(v(t_1)) + numel(v(t_1mean));

        Normin_c2 = 10*sum(v(t_2)) + sum(v(t_mean2));
        Denormin_c2 = 10*numel(v(t_2)) + numel(v(t_mean2));

        c1 = Normin_c1 / Denormin_c1;
        c2 = Normin_c2 / Denormin_c2;
        c_mean = (c1+c2)/2;
        % fprintf('c1 : %f; c2 : %f; c_mean : %f \n', c1{index}, c2{index}, c_mean{index});
    end
    t1 = (vx <= c_mean);
    t2 = (vx > c_mean);
    vx(t1) = c1;
    vx(t2) = c2;

    % update x
    % ||kx-y||^2 + beta_t*||x-vx||^2 + beta_g*||\nabla(x)-g||^2 + lambda_grad*||g||_0
    beta_g = lambda_grad;
    while beta_g<beta_g_max
        Denormin = Den_KER + beta_g*Denormin2 + beta_t;
        h = [diff(S,1,2), S(:,1,:) - S(:,end,:)];
        v = [diff(S,1,1); S(1,:,:) - S(end,:,:)];
        t = (h.^2+v.^2)<lambda_grad/beta_g;
        h(t)=0; v(t)=0;
        clear t;
        Normin2 = [h(:,end,:) - h(:, 1,:), -diff(h,1,2)];
        Normin2 = Normin2 + [v(end,:,:) - v(1, :,:); -diff(v,1,1)];
        FS = (Normin1 + beta_g*fft2(Normin2) + beta_t*fft2(vx))./Denormin;
        S = real(ifft2(FS));
        S(S<0) = 0;
        S(S>1) = 1;
        beta_g = beta_g*kappa;
    end
    beta_t = beta_t * kappa;
    % fprintf('beta_t : %f\n', beta_t);
end

end