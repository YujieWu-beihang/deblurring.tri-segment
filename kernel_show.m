function kernel = kernel_show(k)

kernel = (k-min(k(:)))/(max(k(:))-min(k(:)));

end