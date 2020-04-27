function [latentImage kernel] = blind_deconv_main(y, x, k, lambda, delta, x_in_iter, x_out_iter, ...
                                              xk_iter, k_reg_wt)
                                          
%
% Do single-scale blind deconvolution using the input initializations
% x and k. The cost function being minimized is: min_{x,k}
% \lambda/2 |y - x \oplus k|^2 + |x|_1/|x|_2 + k_reg_wt*|k|_1
%

latentImage = zeros(size(y));

khs = floor(size(k, 1) / 2);

[m, n] = size(y);
[k1, k2] = size(k);
m2 = n / 2;

% arrays to hold costs
lcost = [];
pcost = [];
totiter = 1;

% Split y into 2 parts: x and y gradients; handle independently throughout
y1{1} = y(:, 1 : m2);
y1{2} = y(:, m2 + 1 : end);
y2{1} = y1{1}(khs + 1 : end - khs, khs + 1 : end - khs);
y2{2} = y1{2}(khs + 1 : end - khs, khs + 1 : end - khs);

x1{1} = x(:, 1 : m2);
x1{2} = x(:, m2 + 1 : end);

tmp1{1} = conv2(x1{1}, k, 'same') - y1{1};
tmp1{2} = conv2(x1{2}, k, 'same') - y1{2};

lcost(totiter) = (lambda/2)*(norm(tmp1{1}(:))^2 + norm(tmp1{2}(:))^2); 
pcost(totiter) = norm(x1{1}(:), 1)/norm(x1{1}(:)) + norm(x1{2}(:), 1)/norm(x1{2}(:));

normy(1) = norm(y1{1}(:));
normy(2) = norm(y1{2}(:)); 

% mask
mask{1} = ones(size(x1{1}));
mask{2} = ones(size(x1{2}));

lambda_orig = lambda;
delta_orig = delta;
% figure;
for iter = 1:xk_iter
  lambda = lambda_orig; %/(1.15^(xk_iter-iter)); % seems to work better
                        %without this
  
  totiter_before_x = totiter;
  cost_before_x = lcost(totiter) + pcost(totiter);
  x2 = x1;
  
  while (delta > 1e-4)
    for out_iter = 1 : x_out_iter
    
      normx(1) = norm(x1{1}(:));
      normx(2) = norm(x1{2}(:));
    
      beta(1) = lambda * normx(1);
      beta(2) = lambda * normx(2);
      
%       beta(1) = lambda;
%       beta(2) = lambda;
              
      for inn_iter=1 : x_in_iter
      
        totiter = totiter + 1;
        lambdas(totiter) = lambda;
        x1prev = x1;
      
%         v{1} = x1prev{1} + beta(1)*delta*conv2((y1{1} - conv2(x1prev{1}, k, 'same')), flipud(fliplr(k)), 'same');
%         v{2} = x1prev{2} + beta(2)*delta*conv2((y1{2} - conv2(x1prev{2}, k, 'same')), flipud(fliplr(k)), 'same');
        
        v{1} = x1prev{1} + conv2((y1{1} - conv2(x1prev{1}, k, 'same')) .* mask{1}, flipud(fliplr(k)), 'same');
        v{2} = x1prev{2} + conv2((y1{2} - conv2(x1prev{2}, k, 'same')) .* mask{2}, flipud(fliplr(k)), 'same');
      
        x1{1} = max(abs(v{1}) - delta, 0) .* sign(v{1});
        x1{2} = max(abs(v{2}) - delta, 0) .* sign(v{2});
        
        tmp1{1} = conv2(x1{1}, k, 'same') - y1{1};
        tmp1{2} = conv2(x1{2}, k, 'same') - y1{2};
      
        lcost(totiter) = (lambda/2)*(norm(tmp1{1}(:))^2 + norm(tmp1{2}(:))^2);
        pcost(totiter) = norm(x1{1}(:), 1)/norm(x1{1}(:)) + norm(x1{2}(:), 1)/norm(x1{2}(:));
      end; 

    end;
  
    % prevent blow up in costs due to going uphill; since l1/l2 is so
    % nonconvex it is sometimes difficult to prevent going uphill crazily.
    cost_after_x = lcost(totiter) + pcost(totiter);
    if (cost_after_x > 3*cost_before_x)
      totiter = totiter_before_x;
      x1 = x2;
      delta = delta/2;
      %break;
    else
      break;
    end;
    
    
  end;
  
  % set up options for the kernel estimation
  opts.use_fft = 1;
  opts.lambda = k_reg_wt;
  opts.pcg_tol=1e-4;
  opts.pcg_its = 1;
    
  k_prev = k;
  
  %% use cg-irls
  k = pcg_kernel_irls_conv(k_prev, x1, y2, opts); % using conv2's
  
%   Bx = y1{1};
%   By = y1{2};
%   latent_x = x1{1};
%   latent_y = x1{2};
%   k = estimate_psf(Bx, By, latent_x, latent_y, 2, size(k_prev));  
  
  fprintf('iter:%d, pruning isolated noise in kernel...\n', iter);
  CC = bwconncomp(k,8);
  for ii=1:CC.NumObjects
      currsum=sum(k(CC.PixelIdxList{ii}));
      if currsum<.1 
          k(CC.PixelIdxList{ii}) = 0;
      end
  end
  
  k(k < 0) = 0;
  sumk = sum(k(:));
  k = k ./ sumk;

%   imshow(k,[]);
end;

% combine back into output
latentImage(:, 1 : m2) = x1{1};
latentImage(:, m2 + 1 : end) = x1{2};

kernel = k;