function psf = estimate_psf_irls_cg_k0(blurred_x, blurred_y, latent_x, latent_y, lambda, psf_size)

% min 1/2\|Xk - Y\|^2 + \lambda \|k\|_0

tol = 1e-5;
max_its = 20;

% |k| = | k^(-1) * k^2 |
% w = k^(-1)

% conjgrad : Ax=b
% A : (X^T)*X + W
% b : (X^T)*Y

latent_xf = fft2(latent_x);
latent_yf = fft2(latent_y);
blurred_xf = fft2(blurred_x);
blurred_yf = fft2(blurred_y);
b_f = conj(latent_xf)  .* blurred_xf ...
    + conj(latent_yf)  .* blurred_yf;
b = real(otf2psf(b_f, psf_size));


p.A = conj(latent_xf)  .* latent_xf ...
    + conj(latent_yf)  .* latent_yf;
p.img_size = size(blurred_xf);
p.psf_size = psf_size;

exp_a = 0;
% outer loop
for iter = 1 : 1
    % compute diagonal weights for IRLS
    psf = ones(psf_size) / prod(psf_size);
    weights_l1 = lambda .* (max(abs(psf), 0.0001) .^ (exp_a - 2));
    p.lambda = weights_l1;
    psf = conjgrad(psf, b, max_its, tol, @compute_Ax, p);
    
    psf(psf < max(psf(:))*0.05) = 0;
    psf = psf / sum(psf(:));
end;

end

function y = compute_Ax(x, p)
    x_f = psf2otf(x, p.img_size);
    y = otf2psf(p.A .* x_f, p.psf_size);
    y = y + p.lambda * x;
end