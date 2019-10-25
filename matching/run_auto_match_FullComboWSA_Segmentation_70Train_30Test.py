"""
Matches Consensus and WSA and stores images in ./auto_ConWSA/

Output image name convention is auto_ConWSA_<date>_<model number>.fits
"""
import sys
import os

import warnings
warnings.filterwarnings("ignore")

cur_dir = os.path.dirname(os.path.abspath(__file__))
from matplotlib import pyplot as plt
import preProc
import ClusterCH
import Matching
sys.path.insert(0,cur_dir+'/GenRem/')
import GenRem
import pdb
import ViewOutput as disp # can be removed after finishing development
import numpy as np
from astropy.io import fits

def save_matched_img(con_m ,con_not_m, wsa_m, wsa_not_m, date, m_num):
        """
        m = matched
        no_m = not matched (generated or removed)
        """
        # Consensus
        m = con_m
        no_m = con_not_m
        m[m>0]= (m[m>0]*10) + 1
        no_m[no_m>0]= (no_m[no_m>0]*10) + 2
        pos_ch = ((m[:,:,0] > 0) + (no_m[:,:,0] > 0))
        neg_ch = -1*((m[:,:,1] > 0) + (no_m[:,:,1] > 0))
        con_complete_map = m[:,:,0] + no_m[:,:,0]
        con_complete_map = con_complete_map + m[:,:,1] + no_m[:,:,1]
        con_pol_map = pos_ch + neg_ch

        # WSA
        m = wsa_m
        no_m = wsa_not_m
        m[m>0]= (m[m>0]*10) + 1
        no_m[no_m>0]= (no_m[no_m>0]*10) + 2
        pos_ch = ((m[:,:,0] > 0) + (no_m[:,:,0] > 0))
        neg_ch = -1*((m[:,:,1] > 0) + (no_m[:,:,1] > 0))
        wsa_complete_map = m[:,:,0] + no_m[:,:,0]
        wsa_complete_map = wsa_complete_map + m[:,:,1] + no_m[:,:,1]
        wsa_pol_map = pos_ch + neg_ch

        # Combining
        img = np.dstack((con_complete_map, con_pol_map,
                        wsa_complete_map, wsa_pol_map))
        img = np.swapaxes(img, 0, 2)
        img = np.swapaxes(img, 1, 2)
        hdu = fits.PrimaryHDU(img)

        # Creating name of the fits file
        save_path = cur_dir + '/map_classification/auto_match/FullComboWSA_70Train_30Test/'
        fits_name = str(date) + '_' + str(m_num) + '_' + 'ComboWSA_auto.fits'
        hdu.writeto(save_path+fits_name, clobber=True)

def save_clustered_img(ref, wsa, date, m_num):
        # reference image
        ref_copy = np.copy(ref)
        ref_copy = np.swapaxes(ref_copy,0,2)
        ref_copy = np.swapaxes(ref_copy,1,2)
        hdu = fits.PrimaryHDU(ref_copy)
        save_path = cur_dir + '/map_classification/auto_match/ComboWSA/clust/'
        fits_name = str(date) + '_' + str(m_num) + '_' + 'Combo_clustered.fits'
        hdu.writeto(save_path+fits_name, overwrite=True)

        # Wsa model image
        wsa_copy = np.copy(wsa)
        wsa_copy = np.swapaxes(wsa_copy,0,2)
        wsa_copy = np.swapaxes(wsa_copy,1,2)
        hdu = fits.PrimaryHDU(wsa_copy)
        save_path = cur_dir + '/map_classification/auto_match/ComboWSA/clust/'
        fits_name = str(date) + '_' + str(m_num) + '_' + 'WSA_clustered.fits'
        hdu.writeto(save_path+fits_name, overwrite=True)



ls_obj = preProc.ComboData()
wsa_obj = preProc.WSAData()

print(sys.argv[1])
with open(sys.argv[1]) as dates_file:
    data_path_wsa = dates_file.readline().splitlines()[0]
    data_path_magnetic = data_path_wsa
    for line in dates_file:
        date = line.splitlines()[0]
        print(date)
        for m_num in range(1,13):
            print('\tmodel'+str(m_num))
    
            # Loading consensus and wsa data and creating binary
            # images.
            #   Return value is a 3D image with
            #       Channel 0 = Positive binary image
            #       Channel 1 = Negative binary image
            #       Channel 2 = Complete binary image
            #       Channel 3 = Region of interest, data region (only for consensus, model doesnot have no-data region)
            data_path = "../LevelSets_Combo/segmented_images/all"
            ls_obj.load_data(date, data_path, data_path_magnetic, 0)
            ls_imgs = ls_obj.create_bin_imgs(0)
            wsa_obj.load_wsa_data(date, m_num, data_path_wsa, 0)
            wsa_imgs = wsa_obj.create_bin_imgs(0)
            print('\t \t Pre-processed ...')
    

            # Clustering before extracting generated and removed coronal holes
            clus_obj = ClusterCH.ClusterCH()
            ls_imgs = clus_obj.cluster_very_close(ls_imgs)
            wsa_imgs = clus_obj.cluster_very_close(wsa_imgs)
            print('\t \t Clustered ...')




            # Extracting generated and removed coronal holes.
            gr_obj = GenRem.Classify()
            gen, rem = gr_obj.extract_gen_rem(ls_imgs, wsa_imgs)
            print('\t \t Gen Rem extracted ...')
            
            
            # Cluster coronal holes so that matchable maps have same number of coronal
            # holes. 
            clus_obj = ClusterCH.ClusterCH()
            ls_mat, wsa_mat = clus_obj.extract_matchable(gen, rem, wsa_imgs, ls_imgs)
            ls_clustered, wsa_clustered = clus_obj.cluster(ls_mat, wsa_mat)
#            disp.c_clustered(ls_clustered, wsa_clustered,
#                             ls_mat, wsa_mat)
            print('\t \t Clustered ...')
   

            # Aplly LP on matchable coronal holes
            match_obj = Matching.Matching()
    
            ls_matched, wsa_matched = match_obj.match(ls_clustered, wsa_clustered, 'lp')
            print('\t \t Matched ...')

            # Saving fits file
            # This is how images are stored from ground truth frame work
            # Channel 0 = Consensus clustered
            # Chanel 1 = Consensus matched
            # Channel 2 = polarity image, 1 for positive, -1 for negative
            # channel 3 = Model clustered
            # Channel 4 = Model matched
            # Channel 5 = Model polarity
            save_matched_img(ls_matched, rem, wsa_matched, gen, date, m_num)
