function [kernel, interim_latent, opts] = blind_deconv(y, opts)
% 
% Do multi-scale blind deconvolution
%
%% Input:
% @y : input blurred image (grayscale); 
% @opts: see the description in the file "demo_text_deblurring.m"
%% Output:
% @kernel: the estimated blur kernel
% @interim_latent: intermediate latent image


% gamma correct
if opts.gammaCorrect~=1
    y = y.^opts.gammaCorrect;
end

% set kernel size for coarsest level - must be odd
%minsize = max(3, 2*floor(((opts.kernel_size - 1)/16)) + 1);
%fprintf('Kernel size at coarsest level is %d\n', maxitr);
%%
ret = sqrt(0.5);
%%
maxitr=max(floor(log(5/min(opts.kernelSize))/log(ret)),0);
num_scales = maxitr + 1;
fprintf('Maximum iteration level is %d\n', num_scales);
%%
retv=ret.^[0:maxitr];
k1list=ceil(opts.kernelSize*retv);
k1list=k1list+(mod(k1list,2)==0);
k2list=ceil(opts.kernelSize*retv);
k2list=k2list+(mod(k2list,2)==0);

I_X = conv2(y, [-1,1;0,0], 'valid'); %vertical edges
I_Y = conv2(y, [-1,0;1,0], 'valid'); %vertical edges
I_mag = sqrt(I_X.^2+I_Y.^2);
k1 = opts.kernelSize;
k2 = opts.kernelSize;
% used for the salientEdge of image
edge_thresh = determine_truck(I_X, I_Y, I_mag,k1,k2);
clear I_X I_Y I_mag k1 k2;

% used for the structure of image
lambda_texture = 1;
% used for shock filter
dt = 1;

opts.edge_thresh = edge_thresh;
opts.lambda_texture = lambda_texture;
opts.dt = dt;

fprintf('the method used are: \n');
if(opts.DCP_Valid)
    fprintf('DCP------%d\nBCP------%d\nTCL------%d\n', opts.with_DCP, opts.with_BCP, opts.with_TCL);
else
    fprintf('D-Intensity------%d\nB-Intensity------%d\nTCL------%d\n', opts.with_DCP, opts.with_BCP, opts.with_TCL);
end


if(opts.showProcess)
    opts.h_main = figure;
    set(opts.h_main, 'name', 'Process image');
end

opts.num_scales = num_scales;

% blind deconvolution - multiscale processing
for s = num_scales:-1:1
    if (s == num_scales)
       
        % at coarsest level, initialize kernel
        ks = init_kernel(k1list(s));
        k1 = k1list(s);
        k2 = k1; % always square kernel assumed
    else
    % upsample kernel from previous level to next finer level
    k1 = k1list(s);
    k2 = k1; % always square kernel assumed

    % resize kernel from previous level
    ks = resizeKer(ks,1/ret,k1list(s),k2list(s));

    end;

    
  %%%%%%%%%%%%%%%%%%%%%%%%%%%
    cret=retv(s);

    ys=downSmpImC(y,cret);
    % ys = imresize(y,k1/opts.kernelSize);

    if (s == num_scales)
        [~, ~, threshold]= threshold_pxpy_v1(ys,max(size(ks)));
        opts.threshold = threshold;
    end

    fprintf('Processing scale %d/%d; kernel size %dx%d; image size %dx%d\n', ...
            s, num_scales, k1, k2, size(ys,1), size(ys,2));

  %-----------------------------------------------------------%
    % [interim_latent, ks, opts] = blind_deconv_main(ys, ks, opts);

    % [interim_latent, ks, opts] = blind_deconv_main_TwoColor(ys, ks, opts);

    [interim_latent, ks, opts] = blind_deconv_proposed(ys, ks, opts, s);


    %% center the kernel
    ks = adjust_psf_center(ks);

    ks(ks(:)<0) = 0;
    sumk = sum(ks(:));
    ks = ks./sumk;
    %% set elements below threshold to 0
    if (s == 1)
        kernel = ks;
        if opts.kernelThresh>1
            kernel(kernel(:) < max(kernel(:))/opts.kernelThresh) = 0;
        elseif (opts.kernelThresh>0 & opts.kernelThresh<1)
            kernel(kernel(:) < max(kernel(:))*opts.kernelThresh) = 0;
        else
            kernel(kernel(:) < 0) = 0;
        end
        kernel = kernel / sum(kernel(:));
    end;
end;
%% end kernel estimation

end
%% Sub-function
function [k] = init_kernel(minsize)
  k = single(zeros(minsize, minsize));
  k((minsize - 1)/2, (minsize - 1)/2:(minsize - 1)/2+1) = 1/2;
end

%%
function sI=downSmpImC(I,ret)
%% refer to Levin's code
if (ret==1)
    sI=I;
    return
end
%%%%%%%%%%%%%%%%%%%

sig=1/pi*ret;

g0=[-50:50]*2*pi;
sf=exp(-0.5*g0.^2*sig^2);
sf=sf/sum(sf);
csf=cumsum(sf);
csf=min(csf,csf(end:-1:1));
ii=find(csf>0.05);

sf=sf(ii);
sum(sf);

I=conv2(sf,sf',I,'valid');

[gx,gy]=meshgrid([1:1/ret:size(I,2)],[1:1/ret:size(I,1)]);

sI=interp2(I,gx,gy,'bilinear');
end
%%
function k=resizeKer(k,ret,k1,k2)
%%
% levin's code
k=imresize(k,ret);
k=max(k,0);
k=fixsize(k,k1,k2);
if max(k(:))>0
    k=k/sum(k(:));
end

end

function nf=fixsize(f,nk1,nk2)
[k1,k2]=size(f);

while((k1~=nk1)|(k2~=nk2))
    
    if (k1>nk1)
        s=sum(f,2);
        if (s(1)<s(end))
            f=f(2:end,:);
        else
            f=f(1:end-1,:);
        end
    end
    
    if (k1<nk1)
        s=sum(f,2);
        if (s(1)<s(end))
            tf=zeros(k1+1,size(f,2));
            tf(1:k1,:)=f;
            f=tf;
        else
            tf=zeros(k1+1,size(f,2));
            tf(2:k1+1,:)=f;
            f=tf;
        end
    end
    
    if (k2>nk2)
        s=sum(f,1);
        if (s(1)<s(end))
            f=f(:,2:end);
        else
            f=f(:,1:end-1);
        end
    end
    
    if (k2<nk2)
        s=sum(f,1);
        if (s(1)<s(end))
            tf=zeros(size(f,1),k2+1);
            tf(:,1:k2)=f;
            f=tf;
        else
            tf=zeros(size(f,1),k2+1);
            tf(:,2:k2+1)=f;
            f=tf;
        end
    end
    
    
    
    [k1,k2]=size(f);
    
end

nf=f;
end