function k_final = kernel_denoise(kernel,thresh)
    
region_num = 2;
region_num_threshold = 1;

if(thresh>1)
    errordlg('thresh should be smaller than 1');
end

i = 0;
while(region_num>region_num_threshold)
    kernel(kernel(:) < max(kernel(:))*thresh) = 0;
    kernel = kernel / sum(kernel(:));

    CC = bwconncomp(kernel,4);
    for ii=1:CC.NumObjects
        currsum=sum(kernel(CC.PixelIdxList{ii}));
        if currsum<.2 
          kernel(CC.PixelIdxList{ii}) = 0;
        end
    end

    region_num = CC.NumObjects;
    i = i+1;
    fprintf('i=%d, region_num=%d \n', i,region_num);
    if(i>10)
        break;
    end
end

region_num = CC.NumObjects;
i = 0;
while(region_num>1)
    radius = i;
    if(radius>size(kernel,1)/2)
        break;
    end
    str_close = strel('disk', radius);
    kernel = imclose(kernel, str_close);

    CC = bwconncomp(kernel,4);
    for ii=1:CC.NumObjects
        currsum=sum(kernel(CC.PixelIdxList{ii}));
        if currsum<.2 
          kernel(CC.PixelIdxList{ii}) = 0;
        end
    end
    fprintf('Now, region_num=%d, close_radius=%d, iter=%d\n',region_num,radius,i);
    i = i + 1;
end

    
k_final = kernel;