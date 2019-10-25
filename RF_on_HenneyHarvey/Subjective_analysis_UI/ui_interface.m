function la_img = ui_interface(img, gt_b_img, s_img, col_map)
%{
INPUT:  1. RGB image color map
        2. Ground truth boundary image.
        3. Segmented image.
        4. Color map of RGB image.
OUTPUT: Labelled image.

DESCRIPTION:
 1. Left click to select coronal hole. Selected coronal hole
    will be highlighted as yellow.
 2. A popup text box will appear.
 3. Label
     1 => Reject
     2 => Accept
 4. Right click to go to next date.
%}

a_img = zeros(size(s_img)); % The image coronal holes that are accepted
r_img = zeros(size(s_img)); % The image coronal holes that are rejected

% Coronal hole loop
l_img               = bwlabel(s_img);
uniq_lables         = unique(l_img);
uniq_lables         = sort(uniq_lables); % Sorting in ascending order
num_ch              = length(uniq_lables);
goto_next_ch_flag   = true;
i                   = num_ch;
while i > 1 % 1 as we don't wanna lable background
    cur_lab     = uniq_lables(i);
    if cur_lab ~= 0
        cur_ch      = 1*(l_img == cur_lab);
        color_img   = mark_selected_ch(gt_b_img, s_img, col_map, cur_ch);
        figure(1);
        imagesc(color_img);
        title("Left click to accept, Right click to reject");
        [x,y,button]= ginput(1);
        if button       == 1
            a_img              = a_img + 2*(cur_ch);
            i = i - 1;
        elseif button   == 3
            r_img              = r_img + 1*(cur_ch);
            i = i - 1;
        else
            waitfor(msgbox('Only Left and Right clicks are supported', 'Error','error'));
        end
    end
    
end
la_img = zeros([size(s_img), 3]);
la_img(:,:,1) = r_img; % red for rejected cornal holes.
la_img(:,:,2) = a_img; % green for accepted coronal holes
end



function color_img = mark_selected_ch(gt_b_img, s_img, col_map, cur_ch)
%{
    INPUT:  1. Coordinates of points selected
            2. Color image
            3. Selected image.
    OUTPUT: Selected coronal hole with label 3.
%}

% Make the selected coronal hole Yellow in color imgage.
col_map = [col_map;1 0 0];              % Red
gt_b_img = 2*gt_b_img;                    % Making g_b_img to label 2
s_img   = s_img;                        % Making segmented image to label 1

cur_ch_boundary  = create_boundary(cur_ch);

s_img(cur_ch_boundary > 0) = 3;                  % Making coronal hole currently selected as 3
o_img   = max(gt_b_img,s_img);           % Overlap image
color_img = label2rgb(o_img,col_map);
end


function boundary_img = create_boundary(in_img)
% INPUT: Binary image
% OUTPUT: Boundary of binary image
SE = strel('square',4);
dil_img = imdilate(in_img, SE);
boundary_img = dil_img - in_img;
end