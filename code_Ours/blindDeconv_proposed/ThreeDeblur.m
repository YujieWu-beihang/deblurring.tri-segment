function S = ThreeDeblur(Im, kernel, c1, c2, opts, kappa)

if ~exist('kappa', 'var')
    kappa = 2.0;
end

S = Im;
fx = [1, -1];
fy = [1; -1];
[H,W,D] = size(Im);
sizeI2D = [H,W];
otfFx = single(psf2otf(fx,sizeI2D));
otfFy = single(psf2otf(fy,sizeI2D));
KER = single(psf2otf(kernel,sizeI2D));
Den_KER = abs(KER).^2;
Denormin2 = abs(otfFx).^2 + abs(otfFy).^2;

Normin1 = conj(KER).*fft2(S); 

dark_r = 35;

lambda_i = opts.lambda_grad;

beta_pixel_max = 2^3;
beta_i_max = 1e4;
beta_g_max = 1e5;

with_DCP = opts.with_DCP;
with_BCP = opts.with_BCP;
with_TCL = opts.with_TCL;

DCP_Valid = opts.DCP_Valid;

beta_pixel = opts.lambda_pixel/(graythresh((S).^2));
while beta_pixel<beta_pixel_max

    if(with_DCP)
        if(DCP_Valid)
            [J, J_idx] = dark_channel(S, dark_r);
            u_DCP = J;
            t = (u_DCP.^2<opts.lambda_pixel/beta_pixel);
            u_DCP(t) = 0;
            clear t;
            u_DCP = assign_dark_channel_to_pixel(S, u_DCP, J_idx, dark_r);
        else
            u_DCP = S;
            t = (u_DCP.^2<opts.lambda_pixel/beta_pixel);
            u_DCP(t) = 0;
            clear t;
        end

        u_DCPfft = fft2(u_DCP);
    else
        u_DCPfft = 0;
    end

    if(with_BCP)
        if(DCP_Valid)
            BS = 1-S;
            [BJ, BJ_idx] = dark_channel(BS, dark_r);
            u_BCP = BJ;
            t = (u_BCP.^2<opts.lambda_pixel/beta_pixel);
            u_BCP(t) = 0;
            clear t;
            u_BCP = assign_dark_channel_to_pixel(BS, u_BCP, BJ_idx, dark_r);
            u_BCP = 1-u_BCP;
        else
            u_BCP = 1-S;
            t = (u_BCP.^2<opts.lambda_pixel/beta_pixel);
            u_BCP(t) = 0;
            clear t;
            u_BCP = 1-u_BCP;
        end

        u_BCPfft = fft2(u_BCP);
    else
        u_BCPfft = 0;
    end

    if(with_TCL)
        u = S;
        c_star = u;
        tmp_abs1 = abs(u-c1);
        tmp_abs2 = abs(u-c2);
        t = (tmp_abs1 <= tmp_abs2);
        c_star(t) = c1;
        t = (tmp_abs1 > tmp_abs2);
        c_star(t) = c2;
        clear tmp_abs1 tmp_abs2 t;
        t_tmp1_1 = (u > c1);
        t_tmp1_2 = (u < c2);
        t_tmp1 = t_tmp1_1 & t_tmp1_2;
        tmp = beta_pixel * (c_star-u).^2;
        t_tmp2 = tmp < lambda_i;
        t = t_tmp1 & t_tmp2;
        u(t) = c_star(t);
        clear t_tmp1_1 t_tmp1_2 tmp t_tmp2 t;

        ufft = fft2(u);
    else
        ufft = 0;
    end


    if(with_DCP==0 & with_BCP==0 & with_TCL==0)
        beta_pixel = beta_pixel_max + 1;
    end


    % updata x
    beta_g = 2*opts.lambda_grad;
    while beta_g<beta_g_max
        Denormin = Den_KER + beta_g*Denormin2 + beta_pixel*(with_TCL+with_DCP+with_BCP);

        h = [diff(S,1,2), S(:,1,:) - S(:,end,:)];
        v = [diff(S,1,1); S(1,:,:) - S(end,:,:)];
        t = (h.^2+v.^2)<opts.lambda_grad/beta_g;
        h(t)=0; v(t)=0;
        clear t;
        
        Normin2 = [h(:,end,:) - h(:, 1,:), -diff(h,1,2)];
        Normin2 = Normin2 + [v(end,:,:) - v(1, :,:); -diff(v,1,1)];

        Normin2fft = (fft2(Normin2));
        
        Normin = Normin1 + beta_g*Normin2fft + beta_pixel*(ufft+u_DCPfft+u_BCPfft);

        FS = Normin./Denormin;
        S = real(ifft2(FS));
        S(S<0) = 0;
        S(S>1) = 1;

        beta_g = beta_g*kappa;
    end

    beta_pixel = beta_pixel*kappa;
end

end