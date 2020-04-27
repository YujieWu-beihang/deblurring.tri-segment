function S_block = block_seg(S, length_side)

% S_block is added 0 to make up
    
if(size(S,3)~=1)
    errordlg('S must be gray image');
end

[h, w] = size(S);
h_block = ceil(h/length_side);
w_block = ceil(w/length_side);
h_new = h_block * length_side;
w_new = w_block * length_side;
tmp = [S zeros(h,w_new-w); zeros(h_new-h,w) zeros(h_new-h,w_new-w)];
clear S;
S = tmp;

h_cell = repmat(length_side, 1, h_block);
w_cell = repmat(length_side, 1, w_block);
S_block = mat2cell(S, h_cell, w_cell);

end