function imgDeconv = nonBlindDeconv(imgBlur,kernel,method,opts)

% method -
% '2009_K' : Krishnan 2009
% '2014_W' : Whyte 2014

% imgBlur = im2double(imgBlur);
imgBlur = im2single(imgBlur);
imgDeconv = imgBlur;

if(size(kernel,3)~=1)
    kernel = rgb2gray(kernel);
end
kernel = double(kernel);
kernel = kernel / sum(kernel(:));

switch method
    case '2009_K'
        % Based on the NIPS 2009 paper of Krishnan and Fergus "Fast
        % Image Deconvolution using Hyper-Laplacian Priors"
        addpath(genpath('code_Krishnan09/'));
        
        bhs = floor(opts.kernelSize / 2);
        for i=1:size(imgBlur,3)
            imgBlurPad = padarray(imgBlur(:,:,i), [1 1]*bhs, 'replicate', 'both');
            for j=1:4
                imgBlurPad = edgetaper(imgBlurPad,kernel);
            end

            tmp = fast_deconv_bregman(imgBlurPad, kernel, opts.nb_lambda, opts.nb_alpha);   
            imgDeconv(:,:,i) = tmp(bhs+1 : end-bhs, bhs+1 : end-bhs);
        end
        
        rmpath('code_Krishnan09/');
    case '2014_W'
        % Based on the IJCV 2014 paper of Whyte "Deblurring Shaken
        % and Partially Saturated Images"
        addpath(genpath('code_Whyte_Saturation/'));
        non_uniform = 0;
        RLargs = {};
        % Different arguments for different variants
        mask_threshold = 0.9;
        MaskedRLargs    = cat(2,RLargs,'mask',imgBlur<mask_threshold);
        SaturatedRLargs = cat(2,RLargs,'forward_saturation');
        CombinedRLargs  = cat(2,RLargs,'forward_saturation','prevent_ringing');

        % Deblur
        ImRL          = deconvRL(imgBlur, kernel, non_uniform, RLargs{:});
        % ImMaskedRL    = deconvRL(imgBlur, kernel, non_uniform, MaskedRLargs{:});
        % ImSaturatedRL = deconvRL(imgBlur, kernel, non_uniform, SaturatedRLargs{:});
%         ImCombinedRL  = deconvRL(imgBlur, kernel, non_uniform, CombinedRLargs{:});

        imgDeconv = ImRL;
        
        rmpath('code_Whyte_Saturation/');
    case '2011_Z'
        addpath(genpath('code_Zoran2011'));
        noiseSD = 0.01;
        patchSize = 8;
        ks = floor((size(kernel, 1) - 1)/2);
        load GSModel_8x8_200_2M_noDC_zeromean.mat;
        % initialize prior function handle
        excludeList = [];
        prior = @(Z,patchSize,noiseSD,imsize) aprxMAPGMM(Z,patchSize,noiseSD,imsize,GS,excludeList);

        % comment this line if you want the total cost calculated
        LogLFunc = [];
        imgDeconv = double(zeros(size(imgBlur)));
        
        bhs = floor(opts.kernelSize / 2);
        % deblur
        for i=1:size(imgBlur,3)
            imgBlurPad = padarray(imgBlur(:,:,i), [1 1]*bhs, 'replicate', 'both');
            [tmp,cost] = EPLLhalfQuadraticSplitDeblur(imgBlurPad,64/noiseSD^2,kernel,patchSize,50*[1 2 4 8 16 32 64],1,prior,LogLFunc);
            imgDeconv(:,:,i) = tmp(bhs+1 : end-bhs, bhs+1 : end-bhs);
        end
        clear GS;
        rmpath('code_Zoran2011')
    case '2011_L'
        addpath(genpath('code_Levin2011'));
        
        kernel = rot90(kernel,2);
        imgDeconv = deconvSps(imgBlur,kernel,0.0068,70);
        
        rmpath('code_Levin_2011');
    case '2014_P'
        addpath(genpath('code_Pan_nonDeblur'));
        lambda_tv = 0.001;
        lambda_l0 = 5e-4;
        weight_ring = 1;
        imgDeconv = ringing_artifacts_removal(imgBlur, kernel, lambda_tv, lambda_l0, weight_ring);

        rmpath('code_Pan_nonDeblur');
    otherwise
        imgDeconv = imgBlur;
end


end