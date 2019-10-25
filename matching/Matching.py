import sys
import numpy as np
import random
from matplotlib import pyplot as plt
import pulp
import pdb # debugger
import os
import sys
cur_dir = os.path.dirname(__file__)
sys.path.insert(0,cur_dir+'/blobTools/')
from BlobTools import *

class Matching(BlobTools):
    """

    """
    def match(self, con_mat_clus, wsa_mat_clus, matching_method):
        wsa_relab = np.zeros(wsa_mat_clus.shape)
        # Loop that goes through polarity
        for pol_idx in range(0,2):
            cur_con = con_mat_clus[:, :, pol_idx]
            cur_wsa = wsa_mat_clus[:, :, pol_idx]

            # Perform lp sove only when there are atleast
            # one corona hole cluster in cur_con and cur_wsa
            cur_con_uniq = np.unique(cur_con[np.nonzero(cur_con)])
            cur_wsa_uniq = np.unique(cur_wsa[ np.nonzero(cur_wsa)])
            tot_uniq_lab = cur_con_uniq + cur_wsa_uniq # should be atleast 2, 1 coronal hole in each map
            # Consensus labels are kept constant, while
            # wsa model is relabelled to reflect matching.
            if tot_uniq_lab.size >= 2:
                wsa_relab[:, :, pol_idx] = self.use_lp(cur_con, cur_wsa)
            else:
                wsa_relab[:,:, pol_idx] = cur_wsa
        return con_mat_clus, wsa_relab


    def use_lp(self, con, wsa):
        method = 'euclidean'
        features = ['rect_dist']    # Other possibilities include,
                                    #   1. sph_dist
                                    #   2. rect_area_diff
                                    #   3. sph_area_diff
        wt_matrix, lab_matrix = self.calc_weight_matrix(con, wsa, method, features)
        nrows = wt_matrix.shape[0]
        ncols = wt_matrix.shape[1]
        if nrows != ncols:
            print >> sys.stderr, "Matchable maps differ in number of coronal hole clusters"
            pdb.set_trace()
            sys.exit(1)

        var = pulp.LpVariable.dicts('x', range(0,nrows*ncols),0,1) # 0<= x_i <= 1
        lp_prob = pulp.LpProblem('MinWts', pulp.LpMinimize)

        # Constraints, ??? how to do this more properly!
        for cur_row in range(0, nrows):
            tm = np.zeros(wt_matrix.shape)
            tm[cur_row, :] = 1
            tm = tm.flatten()
            lp_prob += pulp.lpSum([tm[i]*var[i] for i in var]) == 1
        for cur_col in range(0,ncols):
            tm = np.zeros(wt_matrix.shape)
            tm[:, cur_col] = 1
            tm = tm.flatten()
            lp_prob += pulp.lpSum([tm[i]*var[i] for i in var]) == 1
        # Objective function
        wts = wt_matrix.flatten()
        lp_prob += pulp.lpSum([wts[i]*var[i] for i in var])
        # Sove the problem
        try:
            lp_prob.solve(pulp.GLPK(msg=0))
        except:
            pdb.set_trace()
        # matching matrix
        matching_matrix = np.zeros(wt_matrix.size)
        res = dict((v,k) for k,v in var.items())
        for v in lp_prob.variables():
            res[v] = v.varValue
        for i in range(0, len(var.keys())):
            matching_matrix[i] = res[var[i]]
        matching_matrix = matching_matrix.reshape(wt_matrix.shape)
        # Relabel wsa model to have matching labels
        wsa_relabelled = self.relabel_wsa(matching_matrix, lab_matrix, wsa)

        return wsa_relabelled

    def relabel_wsa(self, mm, lm, lab_img):
        """
        Relabels wsa image.

        :param mm: Matching matrix.
        :param lm: label matrix, has two channels.
                    - Channel 0 gives consensus labels
                    - Channel 1 gives wsa labels
        :param lab_img: wsa image which needs to be relabelled.
        """
        relab_img = np.copy(lab_img)
        con_match_matrix = lm[:, :, 0]*mm
        wsa_match_matrix = lm[:, :, 1]*mm
        for cur_lab in np.unique(lm[:, :, 1]):
            fut_lab = con_match_matrix[wsa_match_matrix == cur_lab]
            relab_img[lab_img == cur_lab] = fut_lab
        return relab_img




    def calc_weight_matrix(self, con, wsa, method, features):
        con_ch_uniq = np.unique(con)
        con_ch_uniq = con_ch_uniq[np.nonzero(con_ch_uniq)]
        wsa_ch_uniq = np.unique(wsa)
        wsa_ch_uniq = wsa_ch_uniq[np.nonzero(wsa_ch_uniq)]

        c_lab_matrix = np.zeros((len(con_ch_uniq),len(wsa_ch_uniq)))
        w_lab_matrix = np.zeros((len(con_ch_uniq),len(wsa_ch_uniq)))

        wt_matrix = np.zeros((len(con_ch_uniq),len(wsa_ch_uniq)))

        for idx_lab_c,lab_c in enumerate(con_ch_uniq):
            for idx_lab_w, lab_w in enumerate(wsa_ch_uniq):
                con_lab_c = (con == lab_c).astype('uint8')
                wsa_lab_w = (wsa == lab_w).astype('uint8')
                cur_wt = self.calc_pair_wt(con_lab_c, wsa_lab_w)
                if np.isnan(cur_wt):
                    pdb.set_trace()
                wt_matrix[idx_lab_c, idx_lab_w] = cur_wt
                c_lab_matrix[idx_lab_c, idx_lab_w] = lab_c
                w_lab_matrix[idx_lab_c, idx_lab_w] = lab_w
        lab_matrix = np.dstack((c_lab_matrix,w_lab_matrix))
        return wt_matrix, lab_matrix

    def calc_pair_wt(self, con_bin, wsa_bin):
        """
        """
        # If they overlap, spherical distance is 0
        overlap_img = con_bin*wsa_bin
        if (len(np.unique(overlap_img[np.nonzero(overlap_img)] > 0))):
            sph_sd = 0
        else:
            # Calculating spherical coordinate map
            sph_coord_map = self.create_sph_coord_map(con_bin.shape[1],
                                                      con_bin.shape[0])
            # Calculate shortest spherical distance
            _, con_pts, _ = cv2.findContours(con_bin, cv2.RETR_TREE,
                                          cv2.CHAIN_APPROX_SIMPLE)
            con_pts = np.squeeze(np.concatenate(np.array(con_pts)))
            if con_pts.size <= 2:
                con_pts = con_pts.reshape(1,2)
            _, wsa_pts, _ = cv2.findContours(wsa_bin, cv2.RETR_TREE,
                                          cv2.CHAIN_APPROX_SIMPLE)
            wsa_pts = np.squeeze(np.concatenate(np.array(wsa_pts)))
            if wsa_pts.size <= 2:
                wsa_pts = wsa_pts.reshape(1,2)

            sph_dist_list = list()
            for con_idx, con_pt in enumerate(con_pts):
                for wsa_idx, wsa_pt in enumerate(wsa_pts):
                    sph_dist = self.calc_sph_dist_pts(con_pt, wsa_pt, sph_coord_map)
                    sph_dist_list.append(sph_dist)
            
            sph_dist_np = np.array(sph_dist_list)
            sph_dist_np = sph_dist_np[~np.isnan(sph_dist_np)]
            sph_sd = np.min(sph_dist_np)
        
        wt = sph_sd # Assuming weight to be only sphereical dist
        if np.isnan(wt):
            pdb.set_trace()
        return wt

