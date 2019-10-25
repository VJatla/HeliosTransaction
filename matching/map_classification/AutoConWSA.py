import numpy as np
import cv2
from matplotlib import pyplot as plt
from astropy.io import fits
import pdb
import pandas as pd

class LoadExpData:
    """
    Provides methods that loads and extract generated, removed and
    matchable clusters from FITS files created through matching
    frame work.
    """

    def choose_date_model(self, full_path, DEBUG_FLAG):
        """
        A fine control is provided where one can select date and
        model to be considered for feature extraction. This method
        forms a data frame out of excel sheet where

            - Row index is ``date``
            - Column index is ``model``

        For example if you have two dates, 20100713 and 20100714, and two
        models, model1 and model2 then the excel sheet is

        ==================   ============       ================
        Date                 model1             model2
        ==================   ============       ================
        20100713                1               0
        20100714                0               1
        ==================   ============       ================

        **1** in above table implies that we consider corresponding
        date and model for creating features. **0** implies the
        converse.

        :param full_path: full path to excel sheet, including it's name.
        :param DEBUG_FLAG: 1 prints the dataframe created.

        :Example:
        >>> import os
        >>> from GenRem import LoadGenRem
        >>> dir = os.path.dirname(__file__)
        >>> gr_obj = LoadGenRem()
        >>> gr_obj.create_training_table(dir+'/testCases/GenRem.xls',1)
       """
        valid_set = pd.read_excel(full_path)
        valid_set.index = valid_set['Date']
        valid_train_set = valid_set.drop('Date', 1)
        if(DEBUG_FLAG):
            print(valid_train_set)
        return valid_train_set

    def extract_imgs(self, f_path, f_name, flag):
        """
        flag = 0 => Use output from automated matching
        flag = 1 => Use output from manual matching
        """
        fits_data = fits.open(f_path + f_name)
        all_imgs = fits_data[0].data

        if flag == 0:
            con_match = all_imgs[0]
            wsa_match = all_imgs[2]
            con_pol   = all_imgs[1]
            wsa_pol   = all_imgs[3]
        if flag == 1:
            con_match = all_imgs[1]
            wsa_match = all_imgs[4]
            con_pol   = all_imgs[2]
            wsa_pol   = all_imgs[5]
                

        # Extracting separating generated and removed coronal holes
        #   Since these are labelled 10*(label) + 2 when we modulo
        #   con_clus by 10 and then divide by 2 only generated or removed
        #   coronal holes reamin to be 1. Rest of them become fractional or 0.
        gen_bin = np.floor(np.mod(wsa_match, 10)/2).astype('uint8')
        rem_bin = np.floor(np.mod(con_match, 10)/2).astype('uint8')
        
        con_bin = ((con_match > 0)*(rem_bin == 0)).astype('uint8')
        wsa_bin = ((wsa_match > 0)*(gen_bin == 0)).astype('uint8')

        gen_bin_pos = ((gen_bin) * (wsa_pol > 0)).astype('uint8')
        gen_bin_neg = ((gen_bin) * (wsa_pol < 0)).astype('uint8')
        rem_bin_pos = ((rem_bin) * (con_pol > 0)).astype('uint8')
        rem_bin_neg = ((rem_bin) * (con_pol < 0)).astype('uint8')
        con_bin_pos = ((con_bin) * (con_pol > 0)).astype('uint8')
        con_bin_neg = ((con_bin) * (con_pol < 0)).astype('uint8')
        wsa_bin_pos = ((wsa_bin) * (wsa_pol > 0)).astype('uint8')
        wsa_bin_neg = ((wsa_bin) * (wsa_pol < 0)).astype('uint8')


        img = np.dstack((con_bin_pos, con_bin_neg,
                         wsa_bin_pos, wsa_bin_neg,
                         gen_bin_pos, gen_bin_neg,
                         rem_bin_pos, rem_bin_neg
                         ))
        return img







class Features:
    def load_old_features(self, path, ch_type, feat_list):
        """
        Reads in previous features and creates a data frame.

        :param path: path to excel file containing features
        :param feat_type: Type of feature this is under consideration.

        .. Note::
            Excel file name is assumed to be ``<feat_type>.xls``
        """
        self.df = pd.read_excel(path+ch_type+'.xls',feat_list, 
                                     converters={'model1': str, 'model2': str,
                                                 'model3': str, 'model4': str,
                                                 'model4': str, 'model5': str,
                                                 'model6': str, 'model7': str,
                                                 'model8': str, 'model9': str,
                                                 'model10': str,
                                                 'model11': str,
                                                 'model12': str})


    def write_features(self,path, ch_type, feat_list):
        """
        Writes data frames updated with latest features. It creates an
        excel page with *gen, rem and match* sheets that follow the same
        structure discussed in ``class GRFeatures``.

        :param path: Path to where excel file needs to be written.
        :param feature_type: Feature that is being written

        """
        writer = pd.ExcelWriter(path+ch_type+'.xls', engine='openpyxl')
        for idx, feat in enumerate(feat_list):
            cur_df = self.df[feat]
            cur_df.to_excel(writer, sheet_name=feat)
        writer.save()







    def get_feature(self, matched_imgs, ch_type, feat_type):
        sph_area_map = self.create_sph_area_map(matched_imgs.shape[1], matched_imgs.shape[0])
        if ch_type == 'gen':
            gen_bin_pos = ( matched_imgs[:,:,4] > 0).astype('uint8')
            gen_bin_neg = ( matched_imgs[:,:,5] > 0).astype('uint8')
            gen_bin_imgs = gen_bin_pos + gen_bin_neg
            if feat_type == 'sph_area':
                area = self.get_spherical_area(gen_bin_imgs, sph_area_map)
                area = area + self.get_spherical_area(gen_bin_imgs, sph_area_map)
                return area
            if feat_type == 'pix_area':
                pix_area = gen_bin_pos.sum() + gen_bin_neg.sum()
                return pix_area
            if feat_type == 'num':
                # Generated coronal holes are not clustered, hence
                # number of generated coronal hole clusters is equal to
                # number of generated coronal holes
                num_gen = self.num_con_comp(gen_bin_imgs)
                return num_gen
                
        if ch_type == 'rem':
            rem_bin_pos              = ( matched_imgs[:,:,6] > 0).astype('uint8')
            rem_bin_neg              = ( matched_imgs[:,:,7] > 0).astype('uint8')
            rem_bin_imgs             = rem_bin_pos + rem_bin_neg
            if feat_type == 'sph_area':
                area                 = self.get_spherical_area(rem_bin_imgs, sph_area_map)
                area                 = area + self.get_spherical_area(rem_bin_imgs, sph_area_map)
                return area
            if feat_type == 'pix_area':
                pix_area = rem_bin_pos.sum() + rem_bin_neg.sum()
                return pix_area
            if feat_type == 'num':
                # Removed coronal holes are not clustered, hence
                # number of removed coronal hole clusters is equal to
                # number of removed coronal holes
                num_rem              = self.num_con_comp(rem_bin_imgs)
                return num_rem
        if ch_type == 'mat':
            con_mat_pos              = (matched_imgs[:,:,0] > 0).astype('uint8')
            con_mat_neg              = (matched_imgs[:,:,1] > 0).astype('uint8')
            wsa_mat_pos              = (matched_imgs[:,:,2] > 0).astype('uint8')
            wsa_mat_neg              = (matched_imgs[:,:,3] > 0).astype('uint8')
            if feat_type == 'sph_area_overlap':
                pos_over             = wsa_mat_pos*con_mat_pos
                neg_over             = wsa_mat_neg*con_mat_neg
                area                 = self.get_spherical_area(pos_over, sph_area_map)
                area                 = area + self.get_spherical_area(neg_over, sph_area_map)
                return area
            if feat_type == 'sph_area_overestimate':
                sph_area_con_mat_pos = self.get_spherical_area(con_mat_pos, sph_area_map)
                sph_area_con_mat_neg = self.get_spherical_area(con_mat_neg, sph_area_map)
                tot_sph_area_con     = sph_area_con_mat_pos + sph_area_con_mat_neg
                sph_area_wsa_mat_pos = self.get_spherical_area(wsa_mat_pos, sph_area_map)
                sph_area_wsa_mat_neg = self.get_spherical_area(wsa_mat_neg, sph_area_map)
                tot_sph_area_wsa     = sph_area_wsa_mat_pos + sph_area_wsa_mat_neg
                sph_over_est_area = tot_sph_area_wsa - tot_sph_area_con
                return sph_over_est_area
            if feat_type == 'area_overestimate':
                con_area = con_mat_pos.sum() + con_mat_neg.sum()
                wsa_area = wsa_mat_pos.sum() + wsa_mat_neg.sum()
                over_est_area = wsa_area - con_area
                return over_est_area
            if feat_type == 'wsa_pix_area':
                wsa_mat_area = wsa_mat_pos.sum() + wsa_mat_neg.sum()
                return wsa_mat_area
            if feat_type == 'con_pix_area':
                con_mat_area = con_mat_pos.sum() + con_mat_neg.sum()
                return con_mat_area


    def get_spherical_area(self, bin_img, sph_area_map):
        """
        """
        return np.sum(bin_img*sph_area_map)

    def create_sph_area_map(self, width, height):
        """
        Projecting area of pixel onto sphere
        """
        d_phi = np.pi/height
        d_theta = 2*(np.pi)/width
        phi_vec = np.arange(0, np.pi, d_phi)
        area_vec = np.sin(phi_vec)*d_phi*d_theta
        area_map = np.repeat(area_vec, width)
        area_map = area_map.reshape(height,width)
        return area_map

    def num_con_comp(self, bin_img):
        """
        Takes in a binary image (uint8) format and
        returns a labelled image of type uint8
        """
        bin_copy = bin_img.astype('uint8').copy()
        _, contours, hierarchy = cv2.findContours(bin_copy, 
                                               cv2.RETR_EXTERNAL, 
                                               cv2.CHAIN_APPROX_NONE)
        num_con_comp = len(contours)
        return num_con_comp
