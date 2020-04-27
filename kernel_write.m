function kernel = kernel_write(k)

kernel = (k-min(k(:)))/(max(k(:))-min(k(:)));
kernel = im2uint8(kernel);

end