function [imgBlur,kernelTruth,imgSharpTruth] = loadDatabase(database)

% there are main four database
% CVPR2009 - Levin
% ECCV2012 - Koehler
% CVPR2016 - Lai
% ICCP2013 - Sun

switch database
    case '2016_Lai'
        addpath(genpath('/media/wyj/HDD/Dataset/Deblur/Lai_CVPR2016_Benchmark/Benchmark/synthetic_dataset/uniform/'));
        addpath(genpath('/media/wyj/HDD/Dataset/Deblur/Lai_CVPR2016_Benchmark/Benchmark/synthetic_dataset/kernel/'));
        addpath(genpath('/media/wyj/HDD/Dataset/Deblur/Lai_CVPR2016_Benchmark/Benchmark/synthetic_dataset/ground_truth/'));
        
        % manmade / natural / people / saturated / text
        img_style = 'natural'; 
        img_num = 5; % 1-5
        kernel_num = 4; % 1-4
        imgBlur_str = strcat(img_style,'_0',num2str(img_num),'_kernel_0',num2str(kernel_num),'.png');
        kernelTruth_str = strcat('kernel_0',num2str(kernel_num),'.png');
        imgSharpTruth_str = strcat(img_style,'_0',num2str(img_num),'.png');
        
        imgBlur = imread(imgBlur_str);
        kernelTruth = imread(kernelTruth_str);
        kernelTruth = kernel_normalize(kernelTruth);
        imgSharpTruth = imread(imgSharpTruth_str);

        rmpath('/media/wyj/HDD/Dataset/Deblur/Lai_CVPR2016_Benchmark/Benchmark/synthetic_dataset/uniform/');
        rmpath('/media/wyj/HDD/Dataset/Deblur/Lai_CVPR2016_Benchmark/Benchmark/synthetic_dataset/kernel/');
        rmpath('/media/wyj/HDD/Dataset/Deblur/Lai_CVPR2016_Benchmark/Benchmark/synthetic_dataset/ground_truth/');
    case '2016_Lai_real'
        addpath(genpath('/media/wyj/HDD/Dataset/Deblur/Lai_CVPR2016_Benchmark/Benchmark/real_dataset/'));
        imgBlur = imread('text2.jpg');
        kernelTruth = [];
        imgSharpTruth = [];
        rmpath('/media/wyj/HDD/Dataset/Deblur/Lai_CVPR2016_Benchmark/Benchmark/real_dataset/');  
    case '2012_Koehler'
        addpath(genpath('/media/wyj/HDD/Dataset/Deblur/Koehler_ECCV2012_Benchmark/Benchmark/BlurryImages/'));
        addpath(genpath('/media/wyj/HDD/Dataset/Deblur/Koehler_ECCV2012_Benchmark/Benchmark/KernelsAsMat/'));
        addpath(genpath('/media/wyj/HDD/Dataset/Deblur/Koehler_ECCV2012_Benchmark/Benchmark/GroundTruthImg/'));
        
        img_num = 1; % 1-4
        kernel_num = 2; % 1-10
        imgS_num = 2; % 1-199
        imgBlur_str = strcat('Blurry',num2str(img_num),'_',num2str(kernel_num),'.png');
        kernelTruth_str = strcat('Non_Stationary_kernel_trajectory_',num2str(kernel_num),'.mat');
        imgSharpTruth_str = strcat('GroundTruth',num2str(img_num),'_',num2str(kernel_num),'.mat');

        imgBlur = imread(imgBlur_str);
        kernelTruth = load(kernelTruth_str);
        kernelTruth = kernelTruth.kernel;
        imgSharpTruth = load(imgSharpTruth_str);
        imgSharpTruth = imgSharpTruth.GroundTruth(:,:,:,imgS_num);
        
        rmpath('/media/wyj/HDD/Dataset/Deblur/Koehler_ECCV2012_Benchmark/Benchmark/BlurryImages/');
        rmpath('/media/wyj/HDD/Dataset/Deblur/Koehler_ECCV2012_Benchmark/Benchmark/KernelsAsMat/');
        rmpath('/media/wyj/HDD/Dataset/Deblur/Koehler_ECCV2012_Benchmark/Benchmark/GroundTruthImg/');
    case '2009_Levin'
        addpath(genpath('/media/wyj/HDD/Dataset/Deblur/Levin_CVPR2009_Benchmark/Levin09blurdata/'));
        
        img_num = 6; % 5-8
        kernel_num = 5; % 1-8
        imgkernel_str = strcat('im0',num2str(img_num),'_flit0',num2str(kernel_num),'.mat');
        
        imgCombine = load(imgkernel_str);
        imgBlur = imgCombine.y;
        kernelTruth = imgCombine.f;
        imgSharpTruth = imgCombine.x;
        
        rmpath('/media/wyj/HDD/Dataset/Deblur/Levin_CVPR2009_Benchmark/Levin09blurdata/');
    case '2013_Sun'
        addpath(genpath('/media/wyj/HDD/Dataset/Deblur/Sun_ICCP2013_Benchmark/input80imgs8kernels/'));
        
        img_num = 76; % 1-80
        kernel_num = 1; % 1-8
        imgBlur_str = strcat(num2str(img_num),'_',num2str(kernel_num),'_blurred.png');
        
        imgBlur = imread(imgBlur_str);
        kernelTruth = [];
        imgSharpTruth = [];
        
        rmpath('/media/wyj/HDD/Dataset/Deblur/Sun_ICCP2013_Benchmark/input80imgs8kernels/');
    case 'code_test'
        addpath(genpath('code_test/'));        
        addpath(genpath('/media/wyj/HDD/Dataset/Deblur/Lai_CVPR2016_Benchmark/Benchmark/synthetic_dataset/kernel/'));
        strKernelRoot = '/media/wyj/HDD/Dataset/Deblur/Lai_CVPR2016_Benchmark/Benchmark/synthetic_dataset/kernel/';
        kernelPath = strcat(strKernelRoot,'kernel_02.png');
        
        imgBlur = load('blur_21_kernel_02.mat');
        imgBlur = imgBlur.imgBlur;
        kernelTruth = imread(kernelPath);       
        imgSharpTruth = load('blur_21_sharp.mat');
        imgSharpTruth = imgSharpTruth.imgSharp;
    otherwise
        addpath(genpath('imgblur_example/imgBlurry/'));
        addpath(genpath('imgblur_example/kernel/'));
        strImgRoot = 'imgblur_example/imgBlurry/';
        strKernelRoot = 'imgblur_example/kernel/';
        imgBlurPath = strcat(strImgRoot,'city.png');
        kernelPath = strcat(strKernelRoot,'kernel_valid.png');
%         kernelPath = [];
        imgSharpPath = [];
        
        imgBlur = imread(imgBlurPath);
        kernelTruth = imread(kernelPath);
        imgSharpTruth = [];
end

if(size(kernelTruth,3)~=1)
    kernelTruth = rgb2gray(kernelTruth);
end
if(~isempty(kernelTruth))
    kernelTruth = im2double(kernelTruth);
    kernelTruth = kernelTruth / sum(kernelTruth(:));
end

imgSharpTruth = im2double(imgSharpTruth);
imgBlur = im2double(imgBlur);

end