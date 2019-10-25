import numpy as np
import cv2
from matplotlib import pyplot as plt
from astropy.io import fits
import pdb
import os
import sys
cur_dir = os.path.dirname(__file__)
sys.path.insert(0,cur_dir+'/blobTools/')
from BlobTools import *
from skimage.transform import resize as skimage_resize


class HeliosData(BlobTools):
    """
    Provides methods that can be applied to both consensus and model data.
    """

    def open_fits(self, path, DEBUG_FLAG):
        """
        Reads fits file using astropy library

        :param path: absolute or relative path to fits file
        :param DEBUG_FLAG: when set to 1 displays consensus images.
        :returns: HDU list object.

        :Example:

        >>> import os
        >>> from preProc import HeliosData
        >>> dir = os.path.dirname(__file__)
        >>> t_obj  = HeliosData()
        >>> fits_data = t_obj.open_fits(dir+'/testCases/photoMap.fits',1)
        """
        fits_data = fits.open(path)
        if(DEBUG_FLAG):
            print('Fits file is successfully opened, and here is the info')
            print(fits_data.info())
        return fits_data

    def save_matched_img(self, m ,no_m):
        """
        m = matched
        no_m = not matched (generated or removed)
        """
        m[m>0]= (m[m>0]*10) + 1
        no_m[no_m>0]= (no_m[no_m>0]*10) + 2
        complete_map = m[:,:,0] + no_m[:,:,0]
        complete_map = m[:,:,1] + no_m[:,:,1]
        pos_ch = ((m[:,:,0] > 0) + (no_m[:,:,0] > 0))
        neg_ch = -1*((m[:,:,1] > 0) + (no_m[:,:,1] > 0))
        pol_map = pos_ch + neg_ch

        img = np.dstack((complete_map, pol_map))
        img = np.swapaxes(img, 0, 2)
        img = np.swapaxes(img, 1, 2)
        print(img.shape)

        hdu = fits.PrimaryHDU(img)
        hdu.writeto('new.fits')
        pdb.set_trace()



    def split(self, bin_img, pm_img):
        """
        Splits input binary image based on polarity calculated from photo map.
        This algorithm removes coronal holes below 65% polarity similar to
        Heney Harvey algorithm to auto detect coronal holes.

        :param bin_img: Binary image having coronal holes
        :param pm_img: Photomap image (smoothed)
        :returns: 3D Binary image having positive and negative
                  coronal holes in different channels.
         """
        pos_bin_img = np.zeros(bin_img.shape, dtype=bin_img.dtype)
        neg_bin_img = np.zeros(bin_img.shape, dtype=bin_img.dtype)

        # If binary image is bigger than pm_img then we rescale pm_img
        # to match the size of binary image.
        if (bin_img.shape[0] > pm_img.shape[0]):
            pdb.set_trace()
            pm_img = cv2.resize(pm_img, (bin_img.shape[1],bin_img.shape[0]))
        k1 = 3
        k2 = 4
        ny_max = bin_img.shape[0]  # number of rows
        nx_max = bin_img.shape[1]  # number of columns
        pol_ratio_map = np.zeros((ny_max, nx_max), dtype=pm_img.dtype)

        # Creating ploar ratio map using 7x7 window.
        # 1. Select 7x7 window. (Border pixels are omitted, made to 0)
        # 2. nn_pos = number of positive pixels
        # 3. nn_neg = number of negative pixels
        # 4. Determine polarity ratio by comparing nn_pos, nn_neg.
        for lat in range(k1, ny_max-k2):
            for lng in range(k1, nx_max-k2):
                reg_buf = pm_img[lat-k1:lat+k1+1, lng-k1:lng+k1+1]
                try:
                    nn_pos = sum(sum(reg_buf > 0))
                    nn_neg = sum(sum(reg_buf < 0))
                except:
                    pdb.set_trace()
                
                if nn_pos > nn_neg:
                    pol_ratio = nn_pos/float(nn_pos+nn_neg)
                if nn_neg > nn_pos:
                    pol_ratio = -1*(nn_neg/float(nn_pos+nn_neg))
                pol_ratio_map[lat, lng] = pol_ratio*100

        # Determining and thresholding coronal holes based on polarity.
        # 1. nn_pos_elem = number of positive pixels in coronal hole
        # 2. nn_neg_elem = number of negative pixels in coronal hole
        # 3. Determine current coronal hole polarity based mojority pixels.
        # 4. Accept coronal hole if mean polairty is greater that 65%
        lab_img = self.con_comp(bin_img)
        for cur_lab in range(1, np.amax(lab_img)+1):
            cur_ch_img = (lab_img == cur_lab)
            cur_pol_img = cur_ch_img*pol_ratio_map
            num_pos_elem = sum(sum(cur_pol_img > 0))
            num_neg_elem = sum(sum(cur_pol_img < 0))
            if (num_pos_elem > num_neg_elem):
                cur_pol_ratio = sum(sum(cur_pol_img))
                if(cur_pol_ratio > 65):
                    pos_bin_img = pos_bin_img + cur_ch_img
            if (num_neg_elem > num_pos_elem):
                cur_pol_ratio = sum(sum(cur_pol_img))
                if(cur_pol_ratio < -65):
                    neg_bin_img = neg_bin_img + cur_ch_img
            if (num_neg_elem == num_pos_elem):
                cur_pol_ratio = sum(sum(cur_pol_img))
                if(cur_pol_ratio > 65):
                    pos_bin_img = pos_bin_img + cur_ch_img
                elif(cur_pol_ratio < -65):
                    neg_bin_img = neg_bin_img + cur_ch_img
                else:
                    print("Cannot determine polarity of coronal hole")
                    
        return np.dstack((pos_bin_img, neg_bin_img))

    def remove_no_data(self,img, no_data):
        for channel_idx in range(0,img.shape[2]):
            img[:,:,channel_idx] = img[:,:,channel_idx]*no_data
        return img
    def disp_img(self, img):
        plt.figure()
        plt.imshow(img, cmap='gray', interpolation='nearest')



class ConData(HeliosData):
    """
    Derived from HeliosData, this class has attributes and methods that help in
    pre processing consensus data.

    :ivar con_ch_img: cornal hole image.
    :ivar con_pm_img: photomap image.
    :ivar con_bin_imgs:
        - Channel 0 = +ve coronal holes after polarity threshold.
        - Channel 1 = -ve coronal holes after polarity threshold.
        - Channel 2 = Coronal holes before polarity threshold.
        - Channel 3 = No data region, 0 => no data region.
   """

    def load_con_data(self, date, path, DEBUG_FLAG):
        """
        Loads consensus data and corresponding photomap into
        ``con_ch_img`` and ``con_pm_img`` respectively.

        :param date: date in YYYYMMDD format
        :param path: path to root directory of consensus data.
        :param DEBUG_FLAG: 1 = displays consensus data

        .. note::
            - Default consensus is assumed to be present in R8
              directory with name R8_1_drawn_euvi_new_YYYMMDD.fits.
            - Default photomap is assumed to be present in
              photomap directory with name photomap_GONG_YYYYMMDD.fits

        :Example:

        >>> import os
        >>> from preProc import ConData
        >>> dir = os.path.dirname(__file__)
        >>> con_obj = ConData()
        >>> con_obj.load_con_data('20100713',dir+'/testCases/',1)

        """

        con_ch_name = 'R8_1_drawn_euvi_new_'+date+'.fits'
        con_pm_name = 'photomap_GONG_' + date + '.fits'
        con_path = path + 'R8/' + con_ch_name
        pm_path = path + 'photomap/' + con_pm_name
        pm_data = self.open_fits(pm_path, 0)
        con_data = self.open_fits(con_path, 0)
        self.con_ch_img = con_data[0].data
        self.con_pm_img = pm_data[0].data

        if(DEBUG_FLAG):
            plt.figure()
            plt.imshow(self.con_ch_img, cmap="gray")
            plt.figure()
            plt.imshow(self.con_pm_img, cmap="gray")
            plt.show()

    def create_bin_imgs(self, DEBUG_FLAG):
        """
        Creates 3D binary image,(``con_bin_imgs``) from consensus image and
        it's corresponding photomap.

        :param DEBUG_FLAG: 1 = displays binary images.

        :Example:
        >>> import os
        >>> from preProc import ConData
        >>> dir = os.path.dirname(__file__)
        >>> con_obj = ConData()
        >>> con_obj.load_con_data('20100713',dir+'/testCases/',0)
        >>> con_obj.create_bin_imgs(1)

        .. warning::
            - astropy loads photomap image as ``>f8`` data type.
              Implementing OpenCV funcitons on this data type
              will give you undesired output. Converting
              data type to ``float`` solves this problem.

              :Example:
              >>> pm_img = self.con_pm_img.astype(float)

        """
        # Extracting binary image from labelled consensus image
        # Added 20 as opencv is not happy with negative pixel values, but
        # in our case no data region is represented by -20.
        ret, bin_img = cv2.threshold(self.con_ch_img+20, 20, 1,
                                     cv2.THRESH_BINARY)

        # Joining nearby corinal holes
        kernel = np.zeros((15, 15), np.uint8)
        kernel[:, 7] = 1
        kernel[7, :] = 1
        bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel)

        # Resizing binary image to match photomap image size
        bin_img = 1*(skimage_resize(bin_img, (180,360), order=1)>0)
        bin_img = bin_img.astype('uint8')

        # Smoothing photo-map image
        kernel = np.ones((5, 5), np.float32)/25
        pm_img = self.con_pm_img.astype(float)
        pm_img = cv2.copyMakeBorder(pm_img, 3,3,3,3, cv2.BORDER_REPLICATE)
        pm_img = cv2.filter2D(pm_img, -1, kernel)[3:183,3:363]
        
        # Split based on polarity
        bin_img_split = self.split(bin_img, pm_img)
        self.con_bin_imgs = bin_img_split[:,:,0] + bin_img_split[:,:,1]
        self.con_bin_imgs = np.dstack((bin_img_split, self.con_bin_imgs))

        # Creating no-data region
        ret, no_data = cv2.threshold(self.con_ch_img+20, 0,
                                     1, cv2.THRESH_BINARY)
        no_data = 1*(skimage_resize(no_data, (180,360), order=1)>0)
        no_data[0:29, :] = 0
        no_data[150:179, :] = 0
        no_data = np.ceil(no_data)
        self.con_bin_imgs = np.dstack((self.con_bin_imgs, no_data))

        # After applying no_data region remove small coronal holes.
        if(DEBUG_FLAG):
            plt.figure()
            plt.imshow(self.con_ch_img, cmap="gray")

            plt.figure()
            plt.imshow(bin_img, cmap="gray")

            plt.figure()
            plt.imshow(self.con_pm_img, cmap="gray")

            plt.figure()
            plt.imshow(no_data, cmap="gray")
            plt.show()

        return self.con_bin_imgs.astype('uint8')


class WSAData(HeliosData):
    """
    Derived from HeliosData class, and has attributes
    and methods for pre processing WSA models.

    :ivar wsa_ch_img: Coronal holes generated by WSA model.
    :ivar wsa_pm_img: Photo map generated by WSA model.
    :ivar wsa_bin_imgs:
        - Channel 0: Positive coronal holes after polarity threshold.
        - Channel 1: Negative coronal holes after polarity threshold.
        - Channel 2: WSA generated coronal holes without polarity threshold.
    """
    def load_wsa_data(self, date, m_num, path, DEBUG_FLAG):
        """
        Loads WSA generated cornal hole prediction data.

        :param date: date in YYYYMMDD format
        :param path: path to model data's root directory.
                     This folder should contain WSA directory.
        :param DEBUG_FLAG: 1 = displays WSA predicted data.

        .. note::
            - Default wsa model is assumed to follow
              wsa_YYYYMMDD2300R001_ans.fits naming convenction.

        :Example:

        >>> import os
        >>> from preProc import WSAData
        >>> dir = os.path.dirname(__file__)
        >>> t_obj = WSAData()
        >>> t_obj.load_wsa_data('20100713', dir+'/testCases/' 1)

        """
        wsa_name = 'wsa_' + date + '2300R0' + '%.2d'%m_num + '_ans.fits'
        wsa_path = path + 'WSA/' + wsa_name
        self.wsa_data = self.open_fits(wsa_path, 0)
        self.wsa_pm_img = self.wsa_data[0].data[4]
        self.wsa_ch_img = self.wsa_data[0].data[6]
        if(DEBUG_FLAG):
            plt.figure()
            plt.imshow(self.wsa_ch_img, cmap="gray")
            plt.figure()
            plt.imshow(self.wsa_pm_img, cmap="gray")
            plt.show()

    def create_bin_imgs(self, DEBUG_FLAG):
        """
        Creates 3D binary images,(``wsa_bin_imgs``) from WSA model image and
        it's corresponding photomap.

        :param DEBUG_FLAG: 1 = displays binary images.

        :Example:
        >>> import os
        >>> from preProc import WSAData
        >>> dir = os.path.dirname(__file__)
        >>> t_obj = WSAData()
        >>> t_obj.load_wsa_data('20100713',dir+'/testCases/',0)
        >>> t_obj.create_bin_imgs(1)

        .. warning::
            - astropy loads photomap image as ``>f8`` data type.
              Implementing OpenCV funcitons on this data type
              will give you undesired output. Converting
              data type to ``float`` solves this problem.

              :Example:
              >>> pm_img = self.wsa_pm_img.astype(float)

        """
        # Extracting binary image from labelled consensus image
        ret, bin_img = cv2.threshold(self.wsa_ch_img, 0, 1, cv2.THRESH_BINARY)
        bin_img = (bin_img > 0).astype('uint8')

        # Joining nearby corinal holes
        kernel = np.zeros((3, 3), np.uint8)
        kernel[:, 1] = 1
        kernel[1, :] = 1
        bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel)
        bin_img = bin_img.astype('uint8')

        # Smoothing photo-map image
        kernel = np.ones((3, 3), np.float32)/9
        pm_img = self.wsa_pm_img.astype(float)
        pm_img = cv2.filter2D(pm_img, -1, kernel)

        # Split based on polarity
        bin_img_split = self.split(bin_img, pm_img)

        # Resize to match consensus size, (180,360)
        pos_bin_img = 1*(skimage_resize(bin_img_split[:,:,0], (180,360), order=1)>0)
        
        neg_bin_img = 1*(skimage_resize(bin_img_split[:,:,1], (180,360), order=1)>0)

        bin_img = pos_bin_img + neg_bin_img
        self.wsa_bin_imgs = np.dstack((pos_bin_img,
                                      neg_bin_img, bin_img))
        if(DEBUG_FLAG):
            plt.figure()
            plt.imshow(pos_bin_img, cmap="gray")

            plt.figure()
            plt.imshow(bin_img, cmap="gray")

            plt.figure()
            plt.imshow(neg_bin_img, cmap="gray")
            plt.show()

        return self.wsa_bin_imgs.astype('uint8')




class LvlSetData(HeliosData):
    """
    """

    def load_ls_data(self, date, path, DEBUG_FLAG):
        """

        """

        ls_ch_name = 'Seg_img_'+date+'.fits'
        ls_pm_name = 'photomap_GONG_' + date + '.fits'
        ls_path = path + 'Segmented_LvlSets/' + ls_ch_name
        pm_path = path + 'photomap/' + ls_pm_name
        pm_data = self.open_fits(pm_path, 0)
        ls_data = self.open_fits(ls_path, 0)
        self.ls_ch_img = ls_data[0].data
        self.ls_pm_img = pm_data[0].data

        if(DEBUG_FLAG):
            plt.figure()
            plt.imshow(self.ls_ch_img, cmap="gray")
            plt.figure()
            plt.imshow(self.ls_pm_img, cmap="gray")
            plt.show()

    def create_bin_imgs(self, DEBUG_FLAG):
        """
        """
        # Extracting binary image from labelled consensus image
        # Added 20 as opencv is not happy with negative pixel values, but
        # in our case no data region is represented by -20.
        bin_img = self.ls_ch_img

        # Joining nearby corinal holes
        kernel = np.zeros((15, 15), np.uint8)
        kernel[:, 7] = 1
        kernel[7, :] = 1
        bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel)

        # Smoothing photo-map image
        kernel = np.ones((5, 5), np.float32)/25
        pm_img = self.ls_pm_img.astype(float)
        pm_img = cv2.copyMakeBorder(pm_img, 3,3,3,3, cv2.BORDER_REPLICATE)
        pm_img = cv2.filter2D(pm_img, -1, kernel)[3:183,3:363]
        
        # Split based on polarity
        bin_img_split = self.split(bin_img, pm_img)
        self.ls_bin_imgs = bin_img_split[:,:,0] + bin_img_split[:,:,1]
        self.ls_bin_imgs = np.dstack((bin_img_split, self.ls_bin_imgs))

        # Creating no-data region
        no_data      = np.ones(self.ls_ch_img.shape)
        no_data[0:29, :] = 0
        no_data[150:179, :] = 0
        no_data = np.ceil(no_data)
        self.ls_bin_imgs = np.dstack((self.ls_bin_imgs, no_data))

        # After applying no_data region remove small coronal holes.
        if(DEBUG_FLAG):
            plt.figure()
            plt.imshow(self.ls_bin_imgs[:,:,0], cmap="gray")

            plt.figure()
            plt.imshow(self.ls_bin_imgs[:,:,1], cmap="gray")

            plt.figure()
            plt.imshow(self.ls_bin_imgs[:,:,2], cmap="gray")

            plt.figure()
            plt.imshow(self.ls_bin_imgs[:,:,3], cmap="gray")
            plt.show()

        return self.ls_bin_imgs.astype('uint8')





class R4Data(HeliosData):
    """
    Derived from HeliosData, this class has attributes and methods that help in
    pre processing consensus data.

    :ivar con_ch_img: cornal hole image.
    :ivar con_pm_img: photomap image.
    :ivar con_bin_imgs:
        - Channel 0 = +ve coronal holes after polarity threshold.
        - Channel 1 = -ve coronal holes after polarity threshold.
        - Channel 2 = Coronal holes before polarity threshold.
        - Channel 3 = No data region, 0 => no data region.
   """

    def load_con_data(self, date, path, DEBUG_FLAG):
        """
        Loads consensus data and corresponding photomap into
        ``con_ch_img`` and ``con_pm_img`` respectively.

        :param date: date in YYYYMMDD format
        :param path: path to root directory of consensus data.
        :param DEBUG_FLAG: 1 = displays consensus data

        .. note::
            - Default consensus is assumed to be present in R8
              directory with name R8_1_drawn_euvi_new_YYYMMDD.fits.
            - Default photomap is assumed to be present in
              photomap directory with name photomap_GONG_YYYYMMDD.fits

        :Example:

        >>> import os
        >>> from preProc import ConData
        >>> dir = os.path.dirname(__file__)
        >>> con_obj = ConData()
        >>> con_obj.load_con_data('20100713',dir+'/testCases/',1)

        """

        con_ch_name = 'R4_1_drawn_euvi_new_'+date+'.fits'
        con_pm_name = 'photomap_GONG_' + date + '.fits'
        con_path = path + 'R4/' + con_ch_name
        pm_path = path + 'photomap/' + con_pm_name
        pm_data = self.open_fits(pm_path, 0)
        con_data = self.open_fits(con_path, 0)
        self.con_ch_img = con_data[0].data
        if(self.con_ch_img.shape[0] > 360 or
           self.con_ch_img.shape[1] > 720):
            nr              = self.con_ch_img.shape[0]
            nc              = self.con_ch_img.shape[1]
            extra_rows      = self.con_ch_img.shape[0] - 360
            extra_col       = self.con_ch_img.shape[1] - 720
            erh             = extra_rows/2
            ech             = extra_col/2
            self.con_ch_img = self.con_ch_img[erh:nr-erh, ech:nc-ech]
        self.con_pm_img = pm_data[0].data

        if(DEBUG_FLAG):
            plt.figure()
            plt.imshow(self.con_ch_img, cmap="gray")
            plt.figure()
            plt.imshow(self.con_pm_img, cmap="gray")
            plt.show()

    def create_bin_imgs(self, DEBUG_FLAG):
        """
        Creates 3D binary image,(``con_bin_imgs``) from consensus image and
        it's corresponding photomap.

        :param DEBUG_FLAG 1 = displays binary images.

        :Example:
        >>> import os
        >>> from preProc import ConData
        >>> dir = os.path.dirname(__file__)
        >>> con_obj = ConData()
        >>> con_obj.load_con_data('20100713',dir+'/testCases/',0)
        >>> con_obj.create_bin_imgs(1)

        .. warning::
            - astropy loads photomap image as ``>f8`` data type.
              Implementing OpenCV funcitons on this data type
              will give you undesired output. Converting
              data type to ``float`` solves this problem.

              :Example:
              >>> pm_img = self.con_pm_img.astype(float)

        """
        # Extracting binary image from labelled consensus image
        # Added 20 as opencv is not happy with negative pixel values, but
        # in our case no data region is represented by -20.
        ret, bin_img = cv2.threshold(self.con_ch_img+20, 20, 1,
                                     cv2.THRESH_BINARY)

        # Joining nearby corinal holes
        kernel = np.zeros((15, 15), np.uint8)
        kernel[:, 7] = 1
        kernel[7, :] = 1
        bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel)

        # Resizing binary image to match photomap image size
        bin_img = 1*(skimage_resize(bin_img, (180,360), order=1)>0)
        bin_img = np.ceil(bin_img)
        bin_img = bin_img.astype('uint8')

        # Smoothing photo-map image
        kernel = np.ones((5, 5), np.float32)/25
        pm_img = self.con_pm_img.astype(float)
        pm_img = cv2.copyMakeBorder(pm_img, 3,3,3,3, cv2.BORDER_REPLICATE)
        pm_img = cv2.filter2D(pm_img, -1, kernel)[3:183,3:363]
        
        # Split based on polarity
        bin_img_split = self.split(bin_img, pm_img)
        self.con_bin_imgs = bin_img_split[:,:,0] + bin_img_split[:,:,1]
        self.con_bin_imgs = np.dstack((bin_img_split, self.con_bin_imgs))

        # Creating no-data region
        ret, no_data = cv2.threshold(self.con_ch_img+20, 0,
                                     1, cv2.THRESH_BINARY)
        no_data = 1*(skimage_resize(no_data, (180,360), order=1)>0)
        no_data[0:29, :] = 0
        no_data[150:179, :] = 0
        no_data = np.ceil(no_data)
        self.con_bin_imgs = np.dstack((self.con_bin_imgs, no_data))

        # After applying no_data region remove small coronal holes.
        if(DEBUG_FLAG):
            plt.figure()
            plt.imshow(self.con_ch_img, cmap="gray")

            plt.figure()
            plt.imshow(bin_img, cmap="gray")

            plt.figure()
            plt.imshow(self.con_pm_img, cmap="gray")

            plt.figure()
            plt.imshow(no_data, cmap="gray")
            plt.show()

        return self.con_bin_imgs.astype('uint8')






class R7Data(HeliosData):
    """
    Derived from HeliosData, this class has attributes and methods that help in
    pre processing consensus data.

    :ivar con_ch_img: cornal hole image.
    :ivar con_pm_img: photomap image.
    :ivar con_bin_imgs:
        - Channel 0 = +ve coronal holes after polarity threshold.
        - Channel 1 = -ve coronal holes after polarity threshold.
        - Channel 2 = Coronal holes before polarity threshold.
        - Channel 3 = No data region, 0 => no data region.
   """

    def load_con_data(self, date, path, DEBUG_FLAG):
        """
        Loads consensus data and corresponding photomap into
        ``con_ch_img`` and ``con_pm_img`` respectively.

        :param date: date in YYYYMMDD format
        :param path: path to root directory of consensus data.
        :param DEBUG_FLAG: 1 = displays consensus data

        .. note::
            - Default consensus is assumed to be present in R8
              directory with name R8_1_drawn_euvi_new_YYYMMDD.fits.
            - Default photomap is assumed to be present in
              photomap directory with name photomap_GONG_YYYYMMDD.fits

        :Example:

        >>> import os
        >>> from preProc import ConData
        >>> dir = os.path.dirname(__file__)
        >>> con_obj = ConData()
        >>> con_obj.load_con_data('20100713',dir+'/testCases/',1)

        """

        con_ch_name = 'R7_1_drawn_euvi_new_'+date+'.fits'
        con_pm_name = 'photomap_GONG_' + date + '.fits'
        con_path = path + 'R7/' + con_ch_name
        pm_path = path + 'photomap/' + con_pm_name
        pm_data = self.open_fits(pm_path, 0)
        con_data = self.open_fits(con_path, 0)
        self.con_ch_img = con_data[0].data
        self.con_pm_img = pm_data[0].data
        if(self.con_ch_img.shape[0] > 360 or
           self.con_ch_img.shape[1] > 720):
            nr              = self.con_ch_img.shape[0]
            nc              = self.con_ch_img.shape[1]
            extra_rows      = self.con_ch_img.shape[0] - 360
            extra_col       = self.con_ch_img.shape[1] - 720
            erh             = extra_rows/2
            ech             = extra_col/2
            self.con_ch_img = self.con_ch_img[erh:nr-erh, ech:nc-ech]

        if(DEBUG_FLAG):
            plt.figure()
            plt.imshow(self.con_ch_img, cmap="gray")
            plt.figure()
            plt.imshow(self.con_pm_img, cmap="gray")
            plt.show()

    def create_bin_imgs(self, DEBUG_FLAG):
        """
        Creates 3D binary image,(``con_bin_imgs``) from consensus image and
        it's corresponding photomap.

        :param DEBUG_FLAG: 1 = displays binary images.

        :Example:
        >>> import os
        >>> from preProc import ConData
        >>> dir = os.path.dirname(__file__)
        >>> con_obj = ConData()
        >>> con_obj.load_con_data('20100713',dir+'/testCases/',0)
        >>> con_obj.create_bin_imgs(1)

        .. warning::
            - astropy loads photomap image as ``>f8`` data type.
              Implementing OpenCV funcitons on this data type
              will give you undesired output. Converting
              data type to ``float`` solves this problem.

              :Example:
              >>> pm_img = self.con_pm_img.astype(float)

        """
        # Extracting binary image from labelled consensus image
        # Added 20 as opencv is not happy with negative pixel values, but
        # in our case no data region is represented by -20.
        ret, bin_img = cv2.threshold(self.con_ch_img+20, 20, 1,
                                     cv2.THRESH_BINARY)

        # Joining nearby corinal holes
        kernel = np.zeros((15, 15), np.uint8)
        kernel[:, 7] = 1
        kernel[7, :] = 1
        bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel)

        # Resizing binary image to match photomap image size
        bin_img = 1*(skimage_resize(bin_img, (180,360), order=1)>0)
        bin_img = np.ceil(bin_img)
        bin_img = bin_img.astype('uint8')

        # Smoothing photo-map image
        kernel = np.ones((5, 5), np.float32)/25
        pm_img = self.con_pm_img.astype(float)
        pm_img = cv2.copyMakeBorder(pm_img, 3,3,3,3, cv2.BORDER_REPLICATE)
        pm_img = cv2.filter2D(pm_img, -1, kernel)[3:183,3:363]
        
        # Split based on polarity
        bin_img_split = self.split(bin_img, pm_img)
        self.con_bin_imgs = bin_img_split[:,:,0] + bin_img_split[:,:,1]
        self.con_bin_imgs = np.dstack((bin_img_split, self.con_bin_imgs))

        # Creating no-data region
        ret, no_data = cv2.threshold(self.con_ch_img+20, 0,
                                     1, cv2.THRESH_BINARY)
        no_data = 1*(skimage_resize(no_data, (180,360), order=1)>0)
        no_data[0:29, :] = 0
        no_data[150:179, :] = 0
        no_data = np.ceil(no_data)
        self.con_bin_imgs = np.dstack((self.con_bin_imgs, no_data))

        # After applying no_data region remove small coronal holes.
        if(DEBUG_FLAG):
            plt.figure()
            plt.imshow(self.con_ch_img, cmap="gray")

            plt.figure()
            plt.imshow(bin_img, cmap="gray")

            plt.figure()
            plt.imshow(self.con_pm_img, cmap="gray")

            plt.figure()
            plt.imshow(no_data, cmap="gray")
            plt.show()

        return self.con_bin_imgs.astype('uint8')



class ComboData(HeliosData):
    """
    """

    def load_data(self, date, segmented_path, magnetic_path, DEBUG_FLAG):
        """

        """

        ls_ch_name = 'synoptic_GONG_'+date+'.png'
        ls_pm_name = 'photomap_GONG_' + date + '.fits'
        ls_path = segmented_path + "/" + ls_ch_name
        pm_path = magnetic_path + 'photomap/' + ls_pm_name
        pm_data = self.open_fits(pm_path, 0)
        self.ls_ch_img = cv2.imread(ls_path, cv2.IMREAD_GRAYSCALE)
        self.ls_ch_img = cv2.resize(self.ls_ch_img,(0,0),fx=0.5,fy=0.5)

        self.ls_pm_img = pm_data[0].data

        if(DEBUG_FLAG):
            plt.figure()
            plt.imshow(self.ls_ch_img, cmap="gray")
            plt.figure()
            plt.imshow(self.ls_pm_img, cmap="gray")
            plt.show()

    def create_bin_imgs(self, DEBUG_FLAG):
        """
        """
        # Extracting binary image from labelled consensus image
        # Added 20 as opencv is not happy with negative pixel values, but
        # in our case no data region is represented by -20.
        bin_img = self.ls_ch_img;

        # Joining nearby corinal holes
        kernel = np.zeros((15, 15), np.uint8)
        kernel[:, 7] = 1
        kernel[7, :] = 1
        bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel)

        # Smoothing photo-map image
        kernel = np.ones((5, 5), np.float32)/25
        pm_img = self.ls_pm_img.astype(float)
        pm_img = cv2.copyMakeBorder(pm_img, 3,3,3,3, cv2.BORDER_REPLICATE)
        pm_img = cv2.filter2D(pm_img, -1, kernel)[3:183,3:363]
        
        # Split based on polarity
        bin_img_split = self.split(bin_img, pm_img)
        self.ls_bin_imgs = bin_img_split[:,:,0] + bin_img_split[:,:,1]
        self.ls_bin_imgs = np.dstack((bin_img_split, self.ls_bin_imgs))

        # Creating no-data region
        no_data      = np.ones(self.ls_ch_img.shape)
        no_data[0:29, :] = 0
        no_data[150:179, :] = 0
        no_data = np.ceil(no_data)
        self.ls_bin_imgs = np.dstack((self.ls_bin_imgs, no_data))

        # After applying no_data region remove small coronal holes.
        if(DEBUG_FLAG):
            plt.figure()
            plt.imshow(self.ls_bin_imgs[:,:,0], cmap="gray")

            plt.figure()
            plt.imshow(self.ls_bin_imgs[:,:,1], cmap="gray")

            plt.figure()
            plt.imshow(self.ls_bin_imgs[:,:,2], cmap="gray")

            plt.figure()
            plt.imshow(self.ls_bin_imgs[:,:,3], cmap="gray")
            plt.show()

        return self.ls_bin_imgs.astype('uint8')
