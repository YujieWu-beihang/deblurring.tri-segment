%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Choose example to run
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Set which example to load
%   Included examples: 'notredame', 'istanbul'
%   Examples from Cho et al: 'cho_4_tree_synthetic', 'cho_5_night_light', 
%       'cho_6_parking_lot', 'cho_7_scooters', 'cho_8_snow', 'cho_9_restroom'
example = 'notredame';

% Set path to directory containing Cho et al. code / examples
%   Download from http://cg.postech.ac.kr/research/deconv_outliers/
cho_etal_path = '~/Downloads/deconv_outliers_code';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Load data and run non-blind deblurring
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

code_dir = fileparts(mfilename('fullpath'));
% Compile mex file if it doesn't exist
if exist('apply_blur_kernel_mex') ~= 3
    eval(sprintf('mex -largeArrayDims -outdir %s %s/apply_blur_kernel_mex.cpp',code_dir,code_dir));
end

if strcmp(example(1:4),'cho_')
    % Load an example from Cho et al.
    example_path = fullfile(cho_etal_path,'examples',example(5:end));
    dd = struct('kernel',[],'non_uniform',0,'gamma',1);
    dd.kernel = mean(double(imread(fullfile(example_path,'psf.png'))),3);
    dd.kernel = dd.kernel/sum(dd.kernel(:));
    try
        ImBlurry = double(imread(fullfile(example_path,'blurred.png')))/256;
    catch
        ImBlurry = double(imread(fullfile(example_path,'blurred.tif')))/256/256;
    end
else
    % Load one of the included examples
    example_path = fullfile(code_dir,'..','examples',example);
    dd = load(fullfile(example_path,'kernel.mat'));
    ImBlurry = double(imread(fullfile(example_path,'blurry.jpg')))/256;
end

% Apply gamma correction
ImBlurry = ImBlurry.^dd.gamma;

% Put together arguments for deblurring
if dd.non_uniform
    RLargs = {dd.theta_list,dd.K_internal};
else
    RLargs = {};
end
% Different arguments for different variants
mask_threshold = 0.9;
MaskedRLargs    = cat(2,RLargs,'mask',ImBlurry<mask_threshold);
SaturatedRLargs = cat(2,RLargs,'forward_saturation');
CombinedRLargs  = cat(2,RLargs,'forward_saturation','prevent_ringing');

% Deblur
ImRL          = deconvRL(ImBlurry, dd.kernel, dd.non_uniform, RLargs{:});
% ImMaskedRL    = deconvRL(ImBlurry, dd.kernel, dd.non_uniform, MaskedRLargs{:});
% ImSaturatedRL = deconvRL(ImBlurry, dd.kernel, dd.non_uniform, SaturatedRLargs{:});
% ImCombinedRL  = deconvRL(ImBlurry, dd.kernel, dd.non_uniform, CombinedRLargs{:});
% 
% % Display
% figure;
% subplot(221); imshow(ImRL.^(1/dd.gamma));          title('Standard RL');
% subplot(222); imshow(ImMaskedRL.^(1/dd.gamma));    title(sprintf('Masked RL (threshold %.1f)',mask_threshold));
% subplot(223); imshow(ImSaturatedRL.^(1/dd.gamma)); title('Saturated RL');
% subplot(224); imshow(ImCombinedRL.^(1/dd.gamma));  title('Combined RL');
% 
% % Save images to disk
% imwrite(uint8(256*ImRL.^(1/dd.gamma)),fullfile(example_path,'deblurred_rl.jpg'),'Quality',100);
% imwrite(uint8(256*ImMaskedRL.^(1/dd.gamma)),fullfile(example_path,'deblurred_maskedrl.jpg'),'Quality',100);
% imwrite(uint8(256*ImSaturatedRL.^(1/dd.gamma)),fullfile(example_path,'deblurred_saturatedrl.jpg'),'Quality',100);
% imwrite(uint8(256*ImCombinedRL.^(1/dd.gamma)),fullfile(example_path,'deblurred_combinedrl.jpg'),'Quality',100);
