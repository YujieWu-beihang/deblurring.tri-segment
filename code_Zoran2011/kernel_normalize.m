function kernel_norm = kernel_normalize(kernel)

if(size(kernel,3)~=1)
    kernel = rgb2gray(kernel);
end
kernel = double(kernel);
kernel = kernel / sum(kernel(:));

kernel_norm = kernel;


end