clear all;
close all;
clc;
warning('off','all');

nameBlur = 'manmade01';
imgBlur = imread('blur-kernel01.png');
imgGroundTruth = imread('groundtruth.png');
imgGroundTruth = im2single(imgGroundTruth);
imgBlur = im2single(imgBlur);
kernelTruth = imread('kernel_01.png');
kernelTruth = kernel_normalize(kernelTruth);
kernelTruth = im2single(kernelTruth);
kernelTruth = kernelTruth / sum(kernelTruth(:));

fprintf('processing %s \n', nameBlur);

if ~exist('result_tmp','dir')
    mkdir('./result_tmp/');
end
cd('./result_tmp/');
mkdir(nameBlur);
cd('../');
writeRoot_str = strcat('./result_tmp/', nameBlur, '/');

if(size(kernelTruth,1)<100 && ~isempty(kernelTruth))
    kernelTruth = rot90(kernelTruth,2);
    opts = paramDefine(31);
    kernelTruth_flag = 1;
else
    opts = paramDefine(75);
    kernelTruth_flag = 0;
end

% 31 51 55 75

fprintf('img size is %d * %d \n', size(imgBlur,1), size(imgBlur,2));
fprintf('kernel size is %d \n', opts.kernelSize);

opts.kernelTruth = kernelTruth;
opts.imgBlur = imgBlur;

doBlindDeconv = 1;
if(doBlindDeconv)
    tic;
    Blind_method = 'Ours';
    [imgLatent, kernel, opts] = blindDeconv(imgBlur,Blind_method,opts);
    t_BlindDeconve = toc;
    
    kernel_write = kernel_write(kernel);
    cd(writeRoot_str);
    kernel_str = strcat(nameBlur, '_kernelEstimate.png');
    imwrite(kernel_write, kernel_str, 'png');
    cd('../../');
    clear kernel_write kernel_str;
    fprintf('blind deconvolution is done, the time is %f\n', t_BlindDeconve);
end


doNonBlindDeconv = 1;

method_NonBlindDeconv = '2014_P';  % 2009_K, 2011_Z, 2014_P, 2014_W
if(length(database)==10)
    if(database == '2009_Levin')
        method_NonBlindDeconv = '2011_L';
    end
end

if(doNonBlindDeconv)
    tic;
    if(doBlindDeconv)  
        kernel_NonBlindDeconv = kernel;
    else
        kernel_NonBlindDeconv = kernelTruth;
    end

    imgDeblur = nonBlindDeconv(imgBlur,kernel,method_NonBlindDeconv,opts);
    imgDeblur_write = im2uint8(imgDeblur);
    cd(writeRoot_str);
    imgDeblur_str = strcat(nameBlur, '_imgDeblur_', method_NonBlindDeconv, '_Estimate.png'); 
    imwrite(imgDeblur_write, imgDeblur_str, 'png');
    cd('../../');
    clear imgDeblur_write imgDeblur_str;
    t_nonBlindDeconve = toc;
    fprintf('non-blind deconvolution is done, the time is %f\n', t_nonBlindDeconve);
end

showResult = 1;
pauseTime = 1;
if showResult
    
    figure,imshow(imgBlur,[]),title('blurOrigin');
    pause(pauseTime);

    figure,imshow(imgDeblur,[]),title(method_NonBlindDeconv);
    pause(pauseTime);

    if (exist('kernel') & ~isempty(kernel))
        figure,imshow(kernel,[]),title('kernelEstimate');
        pause(pauseTime);
    end
end




