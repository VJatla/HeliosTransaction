from matplotlib import pyplot as plt
import numpy as np

def c_gen_rem(con_imgs, wsa_imgs, gen,rem):
    """
    Displays following 2 images
        Generated coronal holes on wsa map
        Removed coronal holes on consensus map

    Generated and removed coronal holes appear to be
    brighter than matchable coronal holes.
    """
    print("Generated and removed coronal holes appear"+
          " brighter than matchable coronal holes.")
    no_data = con_imgs[:,:,3]
    full_con = con_imgs[:,:,2]
    full_wsa = wsa_imgs[:,:,2]
    full_con = full_con*no_data
    full_wsa = full_wsa*no_data
    full_con_rem = full_con + rem[:,:,0] + rem[:,:,1]
    full_wsa_gen = full_wsa + gen[:,:,0] + gen[:,:,1] 

    plt.figure()
    plt.imshow(full_con_rem, cmap='gray')
    plt.figure()
    plt.imshow(full_wsa_gen, cmap='gray')
    plt.show()

def c_all_channels(img, color='gray'):
    """
    Displays gray maps of all channels in a
    3D image.
    """
    num_ch = img.shape[2]
    for cur_ch in range(0,num_ch):
        plt.figure()
        plt.imshow(img[:,:,cur_ch], cmap=color)
    plt.show()

def c_clustered(con_clus, wsa_clus, con_not_clus, wsa_not_clus, color='gray'):
    plt.figure()
    plt.imshow(con_not_clus[:,:,1])
    plt.title('Conensus not clustered')

    plt.figure()
    plt.imshow(wsa_not_clus[:,:,1])
    plt.title('WSA not clustered')

    plt.figure()
    plt.imshow(con_clus[:,:,1])
    plt.title('Consensus clustered')

    plt.figure()
    plt.imshow(wsa_clus[:,:,1])
    plt.title('WSA clustered')
    plt.show()

def c_ref_img(img):
    """
    Channel 0 = Positive binary image
    Channel 1 = Negative binary image
    Channel 2 = Complete binary image
    Channel 3 = No data region (Not available for Level sets)
    """
    pos_img = img[:,:,0]
    neg_img = img[:,:,1]
    plt.figure()
    plt.imshow(pos_img)
    plt.show()

    
