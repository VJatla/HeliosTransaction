function mag_edge = photomap_boundary(photo_path)

mag = fitsread(photo_path);
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
            polRatio = -1*nnNeg/polSum;
        end
            pol(nLat,nLng) = polRatio*100;
    end
end
%detecting edge%
mag_edge = edge(pol,'sobel');
mag_edge = imresize(mag_edge, [360 720]); % <--- New, Vj

end