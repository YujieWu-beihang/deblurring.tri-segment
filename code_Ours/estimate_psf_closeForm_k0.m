function psf = estimate_psf_closeForm_k0(blurred_x, blurred_y, latent_x, latent_y, lambda_k, psf_size)

% The objective function
% argmin ||\nabla(x)*k - \nabla(y)||^2 + lambda_k * ||k||_0
% argmin ||\nabla(x)*k - \nabla(y)||^2 + \theta * ||k - g||^2 + lambda_k * ||g||_0

kappa = 2;
size_Img = size(blurred_x);

latent_xf = fft2(latent_x);
latent_yf = fft2(latent_y);
blurred_xf = fft2(blurred_x);
blurred_yf = fft2(blurred_y);

Normin1 = conj(latent_xf).*blurred_xf + conj(latent_yf).*blurred_yf;
Denormin1 = conj(latent_xf).*latent_xf + conj(latent_yf).*latent_yf;

psf = ones(psf_size) / prod(psf_size);

theta = 2*lambda_k;
theta_max = 2e5;

tic;

while theta<theta_max
    g = psf;
    % Solve for g
    t = g.^2 < lambda_k/theta;
    g(t) = 0;
    clear t;
    % Solve for k
    g_f = psf2otf(g,size_Img);
    psf_F = (Normin1 + theta*g_f)./(Denormin1 + theta);
    % psf = real(ifft2(psf_F));
    psf = otf2psf(psf_F, size(psf));
    theta = theta*kappa;
end

toc;

end






