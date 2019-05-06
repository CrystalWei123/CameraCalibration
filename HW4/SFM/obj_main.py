import matplotlib.pyplot as plt

import cv2
import plotly
import plotly.plotly as py
from scipy.spatial import Delaunay
import plotly.figure_factory as FF

from .CheckVisible import CheckVisible


plotly.tools.set_config_file(world_readable=False,
                             sharing='private')


def obj_main(P, p_img2, M, tex_name, im_index):
    img = cv2.imread(tex_name)
    img_size = img.shape

    tri = Delaunay(p_img2)
    ax = plt.axes(projection='3d')
    ax.plot_trisurf(P[:, 0], P[:, 1], P[:, 2], triangles=tri.simplices, cmap='viridis', edgecolor='none')
    plt.show()

    '''fid = open('model' + str(im_index) + '.obj', 'w')
	# cmd_fid = 'fid = fopen(''model' + str(im_index) + '.obj'', ''wt'');'
    print('{} # obj file\n'.format(fid))
    print('{} mtllib model'.format(fid) + str(im_index) + '.mtl\n\n')
	# cmd_print = 'fprintf(fid, ''mtllib model' + str(im_index) + '.mtl\n\n'');'
    print('{} usemtl Texture\n'.format(fid))

    length, dummy = P.shape
    for i in range(length):
        print(fid, P[i, 0], P[i, 1], P[i, 2])
    print(fid)

    for i in range(length):
        print(fid, p_img2[i, 0] / img_size[1], 1 - p_img2[i, 1] / img_size[0])
    print(fid)

    len_tri, dummy = tri.shape
    bVisible = 0
    for i in range(len_tri):
        bVisible = CheckVisible(M, P[tri[i:1],:], P[tri[i:2],:],P[tri[i:3],:])
        if bVisible:
            print(fid, tri[i, 1], tri[i, 1], tri[i, 2], tri[i, 2], tri[i, 3], tri[i, 3])
        else:
            print(fid, tri[i, 2], tri[i, 2], tri[i, 1], tri[i, 1], tri[i, 3], tri[i, 3])
    fid.close()
    mtl = open('model' + str(im_index) + '.mtl', 'w')
    print('{} # MTL file\n'.format(mtl))
    print(mtl, 'newmtl Texture')
    print('{} Ka 1 1 1\nKd 1 1 1\nKs 1 1 1\n'.format(mtl))
    print('{} map_Kd'.format(mtl) + 'tex_name' + '.mtl\n\n')'''
