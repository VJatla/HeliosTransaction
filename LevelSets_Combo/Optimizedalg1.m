%Usage:
%       Input = x where 
%                       x(1) = Area threshold
%                       x(2) = polarity threshold
%                       .
%                       .
%                       .

function  img_alg1 = Optimizedalg1(synoptic_path,photomap_path)
%Flags
use_original_open_close = true;


a = fitsread(synoptic_path);
largeimage = imresize(a, 2, 'bilinear');



%Part2:
%Gets all the data which is not zero into one array called "image_data"
%Finds the data mean and standard deviation
ind_nonzero = find(largeimage>0);
ind0 = find(largeimage <=0);
if(~isempty(ind_nonzero))
    image_data = largeimage(ind_nonzero);
else
    display('synoptic_GONG_xxxxx.fits has no valid data');
end
mean_image_data = mean(image_data);

stdev_image_data = std(image_data);



%Part3:
%Threshold image based on cosine function, max of 0.8 at equator, min of
%0.7 at the poles
radeg = 180/pi;
deg = 0;
for row = 181:360
    for col = 1:720
        if(largeimage(row,col) < (mean_image_data - stdev_image_data*(0.207324 + 0.466333*cos(deg/radeg))))
            largeimage(row,col) = 255;
        else
            largeimage(row,col) = 0;
        end
    end
    deg = deg + 0.5;
end
deg = 0;
offset = 181;
for row = 1:180
    for col = 1:720
        if(largeimage(offset - row,col) < (mean_image_data - stdev_image_data*(0.27324 + 0.466333*cos(deg/radeg))))
            largeimage(offset - row,col) = 255;
        else
            largeimage(offset - row,col) = 0;
        end
    end
    deg = deg + 0.5;
end
largeimage(ind0) = 0;


%Part4:
%Open-Close filtering
%Close to remove gaps
%Open to remove noise

if use_original_open_close
    SE = strel('disk',3,0);
    closeimage = imclose(largeimage,SE);
    SE = ones(4,4);
    morphThresh = imopen(closeimage,SE);
else
    SE = strel('disk',round(x(3)),0);
    closeimage = imclose(largeimage,SE);

    SE = ones(round(x(3)),round(x(3)));
    morphThresh = imopen(closeimage,SE);
end

%Part5:
%Measuring polarity information
%
%
mag = fitsread(photomap_path);
box_car = fspecial('average',[5 5]);
mag = imfilter(mag,box_car,'replicate');
kernal = 7;
k1 = 3;
k2 = 4;
nyMag = 180;
nxMag = 360;
pol = zeros(size(mag,1),size(mag,2));
for nLat = k1:nyMag-k2
    for nLng = k1:nxMag-k2
        regionBuf = mag(nLat-k1+1:nLat+k1+1,nLng-k1+1:nLng+k1+1);
        nnPos = sum(sum(regionBuf > 0));
        nnNeg = sum(sum(regionBuf < 0));
        polSum = nnPos + nnNeg;
        if nnPos > nnNeg
            polRatio = nnPos/polSum;
        end
        if nnNeg > nnPos
            polRatio = nnNeg/polSum;
        end
        if polRatio > 0.65
            pol(nLat,nLng) = polRatio*100;
        end
    end
end
pol0 = imresize(pol,2,'bilinear');



%Part6:
%Removing coronal holes which have area less than 25 pixels
%
%
%

morphThresh = bwareaopen(morphThresh, 25,8); % x(1) = Area Threshold




%part7:
%removing coronal holes which have mean polarity less than 65
CC = bwconncomp(morphThresh,8);
for jCH = 1:CC.NumObjects
    coord = CC.PixelIdxList{1,jCH};   %coordinates of coronal holes coord = row*(width of matrix) + coloumn
    meanpol(jCH) = mean(pol0(coord)); %ASSUMPTION = coordinate names given by CC will go along coloumn
    if meanpol(jCH) < 65              %removing coronal holes,    x(2) = polarity threshold 
       morphThresh(coord) = 0; 
    end
end

img_alg1 = morphThresh;


end