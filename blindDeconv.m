function [imgLatent, kernel, opts] = blindDeconv(imgBlur,method,opts)

% method -
% '2009_K' : Krishnan 2009
% '2014_W' : Whyte 2014

% imgBlur = im2double(imgBlur);
imgBlur = im2single(imgBlur);
imgLatent = imgBlur;

switch method
    case '2011_Krishnan'
        rmpath('code_Krishnan09/');
        addpath(genpath('code_Krishnan09/'));
        [imgLatent, kernel, opts] = blind_deconv_2011Krishnan(imgBlur,opts);
        rmpath('code_Krishnan09/');
    case 'Ours'
        rmpath('code_Krishnan09/');
        addpath(genpath('code_Ours/'));
        
        if (size(imgBlur,3)~=1)
            imgBlur = rgb2gray(imgBlur);
        end
        [kernel, imgLatent, opts] = blind_deconv(imgBlur, opts);
        
        rmpath('code_Ours/');
    otherwise

end


end