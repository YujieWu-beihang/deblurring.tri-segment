function [S, S_x, S_y] = SalientEdge(img_blur, img_latent, lambda_texture, edge_thresh, dt)


h = 1;
window_size = 7;
r = edge_map(img_blur, window_size);
r = exp(-r.^0.8);
structure = structure_adaptive_map(img_latent, lambda_texture*r, 100,0.95);
I_= shock(structure,100,dt,h,'org');
structure = I_;

S_x = conv2(structure, [-1,1;0,0], 'valid');
S_y = conv2(structure, [-1,0;1,0], 'valid');

S_mag = sqrt(S_x.^2 + S_y.^2);
S_x = S_x.*heaviside_function(S_mag,edge_thresh);
S_y = S_y.*heaviside_function(S_mag,edge_thresh);

% S_x(1:2,:) =0;
% S_x(end-1:end,:) = 0;
% S_x(:,1:2) =0;
% S_x(:,end-1:end) = 0;
% 
% S_y(1:2,:) =0;
% S_y(end-1:end,:) = 0;
% S_y(:,1:2) =0;
% S_y(:,end-1:end) = 0;

S = structure;

end