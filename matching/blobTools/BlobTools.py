import numpy as np
import cv2
from matplotlib import pyplot as plt
import pdb

class BlobTools:
    """
    Methods that operate on binary images.
    """

    def con_comp(self, bin_img):
        """
        Takes in a binary image (uint8) format and
        returns a labelled image of type uint8
        """
        bin_copy = bin_img.astype('uint8').copy()
        im2, contours, hierarchy = cv2.findContours(bin_copy, 
                                               cv2.RETR_EXTERNAL, 
                                               cv2.CHAIN_APPROX_NONE)
        num_con_comp = len(contours)
        new_img = np.zeros(bin_img.shape)
        for idx in range(0, num_con_comp):
            cv2.drawContours(new_img, contours, idx, idx+1, -1)

        return new_img.astype('uint8')


    def clus_poles(self, lab_img):
        """
        Cluster polar coronal holes.
        """
        #   North pole
        row_30 = lab_img[30, :]  
        lab_row_30 = np.unique(row_30[np.nonzero(row_30)])
        for cur_lab in lab_row_30:
            lab_img[ lab_img == cur_lab] = np.min(lab_row_30)
        
        
        #   South pole
        row_149 = lab_img[146, :]
        lab_row_149 = np.unique(row_149[np.nonzero(row_149)])
        for cur_lab in lab_row_149:
            lab_img[lab_img == cur_lab] = np.min(lab_row_149)

        
        return lab_img

    def clus_very_close(self, lab_img):
        """
        Cluster coronal holes that are very close to each other. I am
        not yet sure about the threshold.
        """

        
        return lab_img

    def calc_sph_dist_pts(self, pt1, pt2, coord_map):
        """
        Returns great circle distance between points, pt1 and pt2, when
        projected back to a sphere. Distance between points is calculated
        using 
        dist = arccos(sin(phi1).sin(phi2) + cos(phi1).cos(phi2).cos(theta1-theta2)

        ... Note::
            - pt1 and pt2 are of the form (<width>,<height>)
            - coord_map has channel 0 to be azimuthal angle (phi) and channel 1
              to be polar angle (theta).
            - azimuthal angle varies from -pi/2 to pi/2
            - polar angle varies from -pi to pi
        """
        # Subtracting pi/2 and pi so as to map azimuthal and
        # polar angle to (-pi/2, pi/2) and {-pi, pi) respectively.
        # This is required as our coordinate map assumes angles to
        # vary from (0, pi) and (0, 2*pi)
        [phi1, tet1] = [coord_map[pt1[1], pt1[0], 0] - np.pi/2 ,
                        coord_map[pt1[1], pt1[0] ,1] - np.pi]
        [phi2, tet2] = [coord_map[pt2[1], pt2[0], 0] - np.pi/2,
                        coord_map[pt2[1], pt2[0] ,1] - np.pi]
        # Calculating distance
        part1 = np.sin(phi1)*np.sin(phi2)
        part2 = np.cos(phi1)*np.cos(phi2)*np.cos(tet1-tet2)
        dist = np.arccos(part1 + part2)
        return dist

    def create_sph_coord_map(self, width, height):
        """
        Creates a 2 channel map having azimuthal and polar. This
        map is created using Dr. Pattichis summer 2014 project.
        angles at corresponding pixel locations.
            - Channel0 = phi coordinates
            - Channel1 = theta coordinates
        """
        d_phi = np.pi/height
        d_theta = (2*np.pi)/width

        phi_coords = np.arange(0,np.pi,d_phi)
        phi_coords = np.repeat(phi_coords, 360).reshape(180,360)

        theta_coords = np.arange(0, 2*np.pi, d_theta)
        theta_coords = np.repeat(theta_coords, 180).reshape(360,180)
        theta_coords = np.transpose(theta_coords)

        return np.dstack((phi_coords, theta_coords))

