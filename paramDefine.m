function opts = paramDefine(kernelSize)

% the maximum kernel size
if (kernelSize==0)
    opts.kernelSize = 31;
else
    opts.kernelSize = kernelSize;
end

% threshold on fine scale kernel elements
opts.kernelThresh = 0.1;

% set this to 1 for no gamma correction
opts.gammaCorrect = 1.0;

% This is the weight on the likelihood term - it should be decreased for
% noisier images; decreasing it usually makes the kernel "fatter";
% increasing makes the kernel "thinner".
opts.lambda = 10;

% inner/outer iterations for x estimation
opts.x_in_iter = 2; 
opts.x_out_iter = 2;

% maximum number of x/k alternations per level; this is a trade-off
% between performance and quality.
opts.xk_iter = 3;

opts.delta = 0.001;

opts.k_reg_wt = 100;

% non-blind deconvolution
opts.nb_lambda = 3e3;
opts.nb_alpha = 1;

% blind deconvolution
opts.lambda_pixel = 4e-3;
opts.lambda_grad = 4e-3;

opts.with_DCP = 1;
opts.with_BCP = 1;
opts.DCP_Valid = 0;

opts.with_TCL = 1;

opts.showProcess = 0;

end
