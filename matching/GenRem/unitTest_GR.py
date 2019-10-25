import unittest
from GenRem import *
import os
import pandas as pd
import pdb
import numpy as np

class ExtractFeaturesTest(unittest.TestCase):
    def test_calc_sph_dist_pts_vj(self):
        obj = ExtractFeatures()
        # points are {<width>, <height>)
        # image size = 360x180
        coord_map = obj.create_sph_coord_map(360,180)
        # Equator test 1
        pt1 = [1,90]
        pt2 = [180,90]
        dist = obj.calc_sph_dist_pts(pt1, pt2, coord_map)
        per_diff = (abs(dist-np.pi)/np.pi)*100
        self.assertLess(per_diff,5) # asserting error less than 5%
        # Equator test 2
        pt1 = [1,90]
        pt2 = [359,90]
        dist = obj.calc_sph_dist_pts(pt1, pt2, coord_map)
        abs_diff = abs(0-dist)*100
        self.assertLess(per_diff,5) # absolute error is less than 5%
        # North pole test
        pt1 = [1,1]
        pt2 = [180,1]
        dist = obj.calc_sph_dist_pts(pt1, pt2, coord_map)
        abs_diff = (0-dist)*100
        self.assertLess(per_diff,5) # absolute error is less than 5%
        # South pole test
        pt1 = [1,179]
        pt2 = [180,179]
        dist = obj.calc_sph_dist_pts(pt1, pt2, coord_map)
        abs_diff = (0 - dist)*100
        self.assertLess(per_diff,5) # absolute error is less than 5%

    def test_get_sph_area(self):
        obj = ExtractFeatures()
        area_map = obj.create_sph_area_map(360,180)
        # Full area test
        full_img = np.ones((180,360))
        area = obj.get_sph_area(full_img, area_map)
        per_diff = abs(area-np.pi*4)/(4*np.pi) * 100
        self.assertLess(per_diff, 5)

        # Half area
        half_img = np.zeros((180,360))
        half_img[0:179, 0:179] = 1
        area = obj.get_sph_area(half_img, area_map)
        per_diff = abs(area-np.pi*2)/(4*np.pi) * 100
        self.assertLess(per_diff, 5)

        # Quater area
        quat_img = np.zeros((180,360))
        quat_img[0:179, 0:89] = 1
        area = obj.get_sph_area(quat_img, area_map)
        per_diff = abs(area-np.pi)/(4*np.pi) * 100
        self.assertLess(per_diff, 5)

if __name__ == '__main__':
    unittest.main()
