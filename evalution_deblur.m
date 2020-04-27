function metrics = evalution_deblur(imgGroundTruth, imgDeblur, metrix_indicator)

if(~isa(imgDeblur, 'uint8'))
    imgDeblur_double = double(zeros(size(imgDeblur)));
    for i=1:3
        tmp0 = imgDeblur(:,:,i);
        tmp1 = tmp0(:);
        tmp2 = mapminmax(tmp1, 0, 1);
        tmp3 = reshape(tmp2, size(tmp0));
        imgDeblur_double(:,:,i) = tmp3;
    end
    imgDeblur = im2uint8(imgDeblur_double);
end


if(~isa(imgGroundTruth, 'uint8'))
    imgGroundTruth_double = double(zeros(size(imgGroundTruth)));
    for i=1:3
        tmp0 = imgGroundTruth(:,:,i);
        tmp1 = tmp0(:);
        tmp2 = mapminmax(tmp1, 0, 1);
        tmp3 = reshape(tmp2, size(tmp0));
        imgGroundTruth_double(:,:,i) = tmp3;
    end
    imgGroundTruth = im2uint8(imgGroundTruth_double);
end


addpath(genpath('./evalution_code/'));
x = double(imgGroundTruth);
xf = fft2(mean(x,3));

z = double(imgDeblur);
zs = (sum(vec(x.*z)) / sum(vec(z.*z))) .* z;
zf = fft2(mean(zs,3));

[output Greg] = dftregistration(xf, zf, 1);
shift = output(3:4);

c = ones(size(imgDeblur));
% Apply shift 
cr = imshift(double(c), shift, 'same');
zr = imshift(double(zs), shift, 'same');  
xr = x.*cr; 

metrics = metrix_mux(xr, zr, metrix_indicator);

end