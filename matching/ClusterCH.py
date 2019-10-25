import math
import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
import sys
cur_dir = os.path.dirname(__file__)
sys.path.insert(0,cur_dir+'/blobTools/')
from BlobTools import *

class ClusterCH(BlobTools):
    """
    Clusters matchable coronal holes.
    """

    def extract_matchable(self, gen, rem, wsa, con):
        """
        """
        con_mat = con[:, :, 0] * (1*np.logical_not(rem[:, :, 0]))
        con_mat = np.dstack((con_mat, con[:, :, 1] *
                             (1*np.logical_not(rem[:, :, 1]))))
        
        wsa_mat = wsa[:, :, 0] * (1*np.logical_not(gen[:, :, 0]))
        wsa_mat = np.dstack((wsa_mat, wsa[:, :, 1] *
                             (1*np.logical_not(gen[:, :, 1]))))
        return con_mat, wsa_mat

    def cluster(self, con_mat, wsa_mat):
        """
        gen, 2 channels, positive and negative.
        rem, 2 channels, positive and negative.
        wsa, 3 channels, positive, negative and all.
        con, 3 channels, positive, negative and all.
        """
        # Loop through different polarities.
        for pol_idx in range(0, 2):
            cur_con = con_mat[:, :, pol_idx]
            cur_wsa = wsa_mat[:, :, pol_idx]

            # Cluster polar coronal holes
            cur_con_lab = self.clus_poles(cur_con)
            cur_wsa_lab = self.clus_poles(cur_wsa)

            ch_diff = len(np.unique(cur_con_lab)) - len(np.unique(cur_wsa_lab))

            if ch_diff > 0:
                cur_con_lab = self.cluster_closest(cur_con_lab, ch_diff)

            if ch_diff < 0:
                cur_wsa_lab = self.cluster_closest(cur_wsa_lab, abs(ch_diff))

            if pol_idx == 0:
                con_clus = cur_con_lab
                wsa_clus = cur_wsa_lab
            else:
                con_clus = np.dstack((con_clus, cur_con_lab))
                wsa_clus = np.dstack((wsa_clus, cur_wsa_lab))

        return con_clus, wsa_clus
    
    #####################################################################################
    # def clus_very_close(self, lab_img):                                               #
    #     """                                                                           #
    #     Cluster coronal holes that are very close to each other. I am                 #
    #     not yet sure about the threshold, I am setting it to 10 for                   #
    #     now that I came up by analyzing cases which I consider close,                 #
    #     thourgh visual verification. Should visit this problem again.                 #
    #     """                                                                           #
    #     loop_flag = 1                                                                 #
    #     while (loop_flag):                                                            #
    #         max_lab = -1                                                              #
    #         min_lab = -1                                                              #
    #         cluster_pair = (min_lab, max_lab)                                         #
    #         dist_th = 10 # pixels                                                     #
    #         labels = np.unique(lab_img)                                               #
    #         labels = labels[np.nonzero(labels)]                                       #
    #         for lab1 in labels:                                                       #
    #             ch_lab1 = (lab_img == lab1).astype('uint8')                           #
    #             for lab2 in np.delete(labels, np.where(labels==lab1)):                #
    #                 ch_lab2 = (lab_img == lab2).astype('uint8')                       #
    #                 rect_dist = self.calc_rect_sd(np.copy(ch_lab1), np.copy(ch_lab2)) #
    #                 if rect_dist <= dist_th:                                          #
    #                     cluster_pair = (lab1, lab2)                                   #
    #                     dist_th = rect_dist                                           #
    #         max_lab = max(cluster_pair)                                               #
    #         min_lab = min(cluster_pair)                                               #
    #         if (max_lab == -1 and min_lab == -1):                                     #
    #             loop_flag = 0                                                         #
    #         else:                                                                     #
    #             loop_flag = 1                                                         #
    #             lab_img[lab_img == max_lab] = min_lab                                 #
    #     return lab_img                                                                #
    #####################################################################################

    def cluster_very_close(self, img):
        """
        INPUT:
        An image file having 3 or 4 channels.
        - Channel 0 = Positive binary image
        - Channle 1 = Negative binary image
        - Channel 2 = Complete binary image
        - Channel 3 = No data region (only if img is consensus)

        OUTPUT:
        An image file having 3 or 4 channels.
        - Channel 0 = Positive clustered image
        - Channel 1 = Negative clustered image
        - Channel 2 = Complete binary image ???
        - Channel 3 = No data region (only if img is consensus)

        DESCRIPTION:
        This function clusters blobs that are very close to
        each other.
        """

        lab_img = []
        # Polarity loop
        for pol_idx in range(0,2):
            cur_img = img[:,:,pol_idx]
            cur_lab_img = self.con_comp(cur_img)
            loop_flag = 1                                                            
            while (loop_flag):                                                            
                max_lab = -1                                                              
                min_lab = -1                                                              
                cluster_pair = (min_lab, max_lab)                                         
                dist_th_very_close = 6 # pixels                                                     
                labels = np.unique(cur_lab_img)                                               
                labels = labels[np.nonzero(labels)]                                       
                for lab1 in labels:                                                       
                    ch_lab1 = (cur_lab_img == lab1).astype('uint8')                           
                    for lab2 in np.delete(labels, np.where(labels==lab1)):                
                        ch_lab2 = (cur_lab_img == lab2).astype('uint8')                       
                        rect_dist = self.calc_rect_sd(np.copy(ch_lab1), np.copy(ch_lab2))
                        dist_th   = self.get_dist_th(np.copy(ch_lab1), np.copy(ch_lab2))
                        if rect_dist <= dist_th_very_close:
                            cluster_pair = (lab1, lab2)                                   
                            dist_th = rect_dist                                           
                            max_lab = max(cluster_pair)
                            min_lab = min(cluster_pair)
                        if rect_dist <= dist_th:
                            cluster_pair = (lab1, lab2)                                   
                            dist_th = rect_dist                                           
                            max_lab = max(cluster_pair)
                            min_lab = min(cluster_pair)

                if (max_lab == -1 and min_lab == -1):                                     
                    loop_flag = 0                                                         
                else:                                                                     
                    loop_flag = 1                                                         
                    cur_lab_img[cur_lab_img == max_lab] = min_lab
            img[:,:,pol_idx] = cur_lab_img
        return img



    def get_dist_th(self, ch_lab1, ch_lab2):
        """
        ****
        Return distance threshold based on area of cornal hole.
        """
        ar1 = np.sum(ch_lab1)
        ar2 = np.sum(ch_lab2)

        # Very small cornal holes distance threshold
        if ( (ar1 <= 20) and (ar2 <= 20)):
            return 10
        # One small one big distance threshold
        if ( (ar1 <= 20) and (ar2 > 20)):
            return 10
        if ( (ar1 > 20) and (ar2 <= 20)):
            return 10
        # Two big distance threshold
        if ( (ar1 > 20) and (ar2 > 20)):
            return 8
        

    
    def cluster_closest(self, lab_img, ch_diff):
        """
        """
        for diff_idx in range(0,ch_diff):
            nearest_dist = 10000 # ??? max this out
            labels = np.unique(lab_img)
            labels = labels[np.nonzero(labels)]
            for lab1 in labels:
                for lab2 in np.delete(labels,np.where(labels==lab1)):
                    ch_lab1 = (lab_img == lab1).astype('uint8')
                    ch_lab2 = (lab_img == lab2).astype('uint8')
                    sph_dist = self.calc_sph_sd(ch_lab1, ch_lab2)
                    if sph_dist < nearest_dist:
                        nearest_dist = sph_dist
                        nearest_ch_lab = (lab1, lab2)
            # relabel so that nearest blobs of same label
            min_lab = min(nearest_ch_lab)
            max_lab = max(nearest_ch_lab)
            lab_img[lab_img == max_lab] = min_lab
        return lab_img



    def calc_rect_sd(self, ch_lab1, ch_lab2):
        """
        """
        # Building contour for ch_lab1
        _, contours, _ = cv2.findContours(ch_lab1, cv2.RETR_TREE,
                                                        cv2.CHAIN_APPROX_SIMPLE)
        ch1_contour = np.zeros(ch_lab1.shape)
        cv2.drawContours(ch1_contour, contours, -1, 1, 0)
        ch1_contour = ch1_contour.astype('uint8')

        # Building countour for ch_lab2
        _, contours, _ = cv2.findContours(ch_lab2, cv2.RETR_TREE,
                                                        cv2.CHAIN_APPROX_SIMPLE)
        ch2_contour = np.zeros(ch_lab2.shape)
        cv2.drawContours(ch2_contour, contours, -1, 1, 0)
        ch2_contour = ch2_contour.astype('uint8')

        # Finding rectangular distance between these boundaries
        lab1_bound_pts = np.transpose(np.nonzero(ch_lab1))
        lab2_bound_pts = np.transpose(np.nonzero(ch_lab2))

        sd = 1000000 # ??? maximize this
        for lab1_pt in lab1_bound_pts:
            for lab2_pt in lab2_bound_pts:
                cur_dst_man = math.sqrt((lab1_pt[0]-lab2_pt[0])**2 +
                                        (lab1_pt[1]-lab2_pt[1])**2)
                if(cur_dst_man < sd):
                    sd = cur_dst_man
        return sd

    def calc_sph_sd(self, ch_lab1, ch_lab2):
        """
        """
        over_lap = ch_lab1*ch_lab2
        uniq_lab = np.unique(over_lap)
        if len(uniq_lab > 1):
            min_sph_dist = 0
        else:
            ch1_bound_pts, _ = cv2.findContours(ch_lab1, cv2.RETR_TREE,
                                                cv2.CHAIN_APPROX_SIMPLE)
            ch2_bound_pts, _ = cv2.findContours(CH_lab2, cv2.RETR_TREE,
                                                cv2.CHAIN_APPROX_SIMPLE)
            sph_dist_list = list()
            for pt1 in ch1_bound_pts:
                for pt2 in ch2_bound_pts:
                    sph_dist = self.calc_sph_dist_pts(pt1, pt2, coord_map)
                    sph_dist_list.append(sph_dist)
            min_sph_dist = np.min(sph_dist_list)
        return min_sph_dist
