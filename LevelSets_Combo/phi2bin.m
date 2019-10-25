function bin_img = phi2bin(phi,org_img)
%
%INPUTS:
%       phi matrix
%       original image (synoptic gong in this case)
%
%OUTPUT:
%       binary image
%       phi < 0     =>      bin_img = 1;
%       phi > 0     =>      bin_img = 0;
%       no data     =>      bin_img = 0;

bin_img         =       phi <= 0;
org_bin_img     =       org_img > 0;
bin_img         =       bin_img.*org_bin_img; 
end