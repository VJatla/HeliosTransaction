function total_area = get_area(bin_img,r_sqr)
%INPUT:
%       A binary image whose total area is needed
%OUPUT:
%       Total area of white patch when projected onto a sphere
%DESCRIPTION:
%       Finds area of white pathes when projected onto a sphere of
%       radius 1.

num_rows = size(bin_img,1);
num_col  = size(bin_img,2);

dphi            = (pi)/num_rows;
dtheta          = (2*pi)/num_col;
total_area      = 0;

for y = 1 : num_rows
    phi                 = (0+dphi/2) + y*dphi;
    area_element        = r_sqr*sin(phi)*dphi*dtheta;
    row_pixels          = sum(bin_img(y,:));
    total_area          = total_area + row_pixels*area_element;
end

end