%  This Matlab code demonstrates an edge-based active contour model as an application of 
%  the Distance Regularized Level Set Evolution (DRLSE) formulation in the following paper:
%
%  C. Li, C. Xu, C. Gui, M. D. Fox, "Distance Regularized Level Set Evolution and Its Application to Image Segmentation", 
%     IEEE Trans. Image Processing, vol. 19 (12), pp. 3243-3254, 2010.
%
% Author: Chunming Li, all rights reserved
% E-mail: lchunming@gmail.com   
%         li_chunming@hotmail.com 
% URL:  http://www.imagecomputing.org/~cmli//
function y = demo_solar (x, syn_path, gt_path, photo_path, henney_path, testing_flag)

DISPLAY_IMAGES = false;
USE_PHOTOMAP = true;
USE_PREVIOUS_ALG = true;

Img= fitsread(syn_path);
Img=double(Img(:,:,1));
Img = imresize(Img, [360 720]);

%% parameter setting
timestep=5;  % time step
mu=0.2/timestep;  % coefficient of the distance regularization term R(phi)
iter_inner=5;
iter_outer=60; %vj
lambda=5; % coefficient of the weighted length term L(phi)
alfa=x(2);  % coefficient of the weighted area term A(phi)
epsilon=1.5; % papramater that specifies the width of the DiracDelta function


sigma=x(1);    % scale parameter in Gaussian kernel, original = 0.8 i gave it 0.2
G=fspecial('gaussian',15,sigma); % Gaussian kernel
Img_smooth=conv2(Img,G,'same');  % smooth image by Gaussiin convolution
[Ix,Iy]=gradient(Img_smooth);
f=Ix.^2+Iy.^2;
g=1./(1+f);  % edge indicator function.

if USE_PHOTOMAP
    %Adding information into g from photomap%
    %   Transition boundaries are identified and these boundaries are assigned
    %   zero. Multiplying this image with g will give us an image which has low
    %   values at photomap boundaries. Hence the evolution of curve will not
    %   cross this boundary
    %
    mag_boundries = photomap_boundary(photo_path);
    
    SE = strel('square',4);
    mag_boundries = imdilate(mag_boundries,SE);
    mag_boundries = imcomplement(mag_boundries);
    g = g.*mag_boundries;
    if DISPLAY_IMAGES
        figure(11)
        imagesc(mag_boundries); axis image; colormap gray;
        title('Boundaries of magnitude images');
    end
end

if DISPLAY_IMAGES
    figure(6);
    imagesc(g);colormap gray; axis image;
    title('g');
end

if USE_PREVIOUS_ALG
    % initialize LSF as binary step function
    c0=2;
    initialLSF = c0*ones(size(Img));
    % img_alg1 = Optimizedalg1(syn_path, photo_path);
    img_henney = double(imread(henney_path));
    img_henney = img_henney/(max(img_henney(:)));
    % img_alg1 = alg1(syn_path, photo_path);
    % img_alg1 = imresize(img_alg1, [180 360]);
    initialLSF = initialLSF - 2*c0*(img_henney);
    phi=initialLSF;
else
    % initialize LSF as binary step function
    c0=2;
    initialLSF = c0*ones(size(Img));
    % generate the initial region R0 as two rectangles
    % initialLSF(25:35,20:25)=-c0; 
    % initialLSF(25:35,40:50)=-c0;
    %vj using erode to initialize contours%
    img_bin = image_threshold(Img,135);
    SE = strel('square',4);
    img_close = imclose(img_bin,SE);
    img_open = imopen(img_close,SE);
    SE = strel('square',4);
    img_erode = imerode(img_open,SE);

    if DISPLAY_IMAGES
        figure(10);
        imagesc(img_erode), axis image, colormap gray;
    end

    initialLSF = initialLSF - 2*c0*(img_erode);
    phi=initialLSF;

    if DISPLAY_IMAGES 
        figure(1);
        mesh(-phi);   % for a better view, the LSF is displayed upside down
        hold on;  contour(phi, [0,0], 'r','LineWidth',2);
        title('Initial level set function');
        view([-80 35]);
    end

    if DISPLAY_IMAGES
        figure(2);
        imagesc(Img,[0, 255]); axis off; axis equal; colormap(gray); hold on;  contour(phi, [0,0], 'r');
        title('Initial zero level contour');
        pause(0.5);
    end

end



potential=2;  
if potential ==1
    potentialFunction = 'single-well';  % use single well potential p1(s)=0.5*(s-1)^2, which is good for region-based model 
elseif potential == 2
    potentialFunction = 'double-well';  % use double-well potential in Eq. (16), which is good for both edge and region based models
else
    potentialFunction = 'double-well';  % default choice of potential function
end  

% start level set evolution
tic;
for n=1:iter_outer
    phi = drlse_edge(phi, g, lambda, mu, alfa, epsilon, timestep, iter_inner, potentialFunction);    
    if mod(n,2)==0
        if DISPLAY_IMAGES
            figure(2);
            imagesc(Img,[0, 255]); axis off; axis equal; colormap(gray); hold on;  contour(phi, [0,0], 'r');
        end
    end
end

% refine the zero level contour by further level set evolution with alfa=0
alfa=0;
iter_refine = 10;
phi = drlse_edge(phi, g, lambda, mu, alfa, epsilon, timestep, iter_inner, potentialFunction);
toc;

finalLSF=phi;

if DISPLAY_IMAGES
    figure(2);
    imagesc(Img,[0, 255]); axis off; axis equal; colormap(gray); hold on;  contour(phi, [0,0], 'r');
    hold on;  contour(phi, [0,0], 'r');
    str=['Final zero level contour, ', num2str(iter_outer*iter_inner+iter_refine), ' iterations'];
    title(str);
end

if DISPLAY_IMAGES
    figure;
    mesh(-finalLSF); % for a better view, the LSF is displayed upside down
    hold on;  contour(phi, [0,0], 'r','LineWidth',2);
    view([-80 35]);
    str=['Final level set function, ', num2str(iter_outer*iter_inner+iter_refine), ' iterations'];
    title(str);
    axis on;
    [nrow, ncol]=size(Img);
    axis([1 ncol 1 nrow -5 5]);
    set(gca,'ZTick',[-3:1:3]);
    set(gca,'FontSize',14)
end

% The following code is added by venkatesh Jatla%
phi_bin = phi2bin(phi,Img); % converts phi into binary image with coronal holes marked as white

if DISPLAY_IMAGES
    figure;
    imagesc(phi_bin); axis image; colormap gray; title('binary thresholding phi');
end

[sens spec unit_dist] = compare_against_gt(phi_bin, gt_path);

% Saving testing image
if testing_flag
    [filepath,name,ext] = fileparts(syn_path);
    imwrite(phi_bin, "segmented_images/"+name+".png");
end

y = unit_dist;
end