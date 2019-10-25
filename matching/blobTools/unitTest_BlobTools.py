import unittest
import BlobTools
import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
cur_dir = os.path.dirname(__file__)
print(cur_dir)

class BlobToolsTest(unittest.TestCase):

    def test_con_comp_vj(self):
        img = cv2.imread(cur_dir+'/../testCases/BlobTools/con_comp_vj_test1.png')
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_bin = (img_gray > 0).astype('uint8') # making binary image
        
        obj = BlobTools.BlobTools()
        lab_img = obj.con_comp(img_bin)
        num_blobs = len(np.unique(lab_img)) - 1
        self.assertEqual(num_blobs, 3)


    def test_calc_sph_dist_pts_pk(self):
        obj = BlobTools.BlobTools()
        # points are {<width>, <height>)
        # image size = 360x180        
        coord_map = obj.create_sph_coord_map(360,180)
        
        '''
        when points are at pi distance on Equator test 1
        '''
        pt1 = [1,90]
        pt2 = [181,90]
        dist = obj.calc_sph_dist_pts(pt1, pt2, coord_map)
        abs_diff = abs(dist)
        print ('test 1: ', abs_diff)
        self.assertAlmostEqual(abs_diff,3.14, places=2) 
        
        '''
        when points are at north pole,distance should be close to zero. test 2
        '''
        pt1 = [1,1]
        pt2 = [181,1]
        dist = obj.calc_sph_dist_pts(pt1, pt2, coord_map)
        abs_diff = abs(dist)
        print ('test 2: ', abs_diff)
        self.assertAlmostEqual(abs_diff,0, places=1)
        
         
        '''
        when points are at south pole,distance should be close to zero. test 3
        '''
        pt1 = [1,179]
        pt2 = [181,179]
        dist = obj.calc_sph_dist_pts(pt1, pt2, coord_map)
        abs_diff = abs(dist)
        print ('test 3: ', abs_diff)
        self.assertAlmostEqual(abs_diff,0, places=1)
        
           
        '''
        when one point is on equator and other at north pole on same vertical circle. test 4
        '''
        pt1 = [45,1]
        pt2 = [45,90]
        dist = obj.calc_sph_dist_pts(pt1, pt2, coord_map)
        abs_diff = abs(dist)
        print ('test 4: ', abs_diff)
        self.assertAlmostEqual(abs_diff,1.57, places=1)
        
                 
        '''
        when one point is on equator and other at south pole on same vertical circle. test 5
        '''
        pt1 = [79,179]
        pt2 = [79,90]
        dist = obj.calc_sph_dist_pts(pt1, pt2, coord_map)
        abs_diff = abs(dist)
        print ('test 5: ', abs_diff)
        self.assertAlmostEqual(abs_diff,1.57, places=1)   
        
                                
        '''
        when one point is on north pole and other at south pole on same vertical circle. test 6
        '''
        pt1 = [79,179]
        pt2 = [79,0]
        dist = obj.calc_sph_dist_pts(pt1, pt2, coord_map)
        abs_diff = abs(dist)
        print ('test 6: ', abs_diff)
        self.assertAlmostEqual(abs_diff,3.14, places=1)                                                                        
        
        
        ''' 
        When points are on same horizontal circle. test 7
        eg: distance between (15,45) (25,45) should be equal to distance between (355,45)(5,45)
        '''
        pt1 = [15,45]
        pt2 = [25,45]
        dist1 = obj.calc_sph_dist_pts(pt1, pt2, coord_map)
        abs_diff1 = abs(dist1)
        
        pt3 = [355,45]
        pt4 = [5,45]
        dist2 = obj.calc_sph_dist_pts(pt3, pt4, coord_map)
        abs_diff2 = abs(dist2)
        print ('test 7: ', abs_diff1, abs_diff2)       
        self.assertEqual(abs_diff1,abs_diff2)
       
        
        


if __name__ == '__main__':
    unittest.main()
