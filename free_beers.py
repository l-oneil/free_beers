import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib import path

guney = np.array(plt.imread("images/guney.jpg"), dtype=np.float) * (1./255)
bed   = np.array(plt.imread("images/bed.png"), dtype=np.float)

top_left     = [152,120]
top_right    = [305,150]
bottom_right = [241,295]
bottom_left  = [38,255]

print("Bed Image Size: ", bed.shape)
print("Guney Image Size: ", guney.shape)

img_points = [top_left, bottom_left,            bottom_right,                        top_right]
gun_points = [[0,0],    [guney.shape[0]-1, 0], [guney.shape[0]-1, guney.shape[1]-1], [0,guney.shape[1]-1]  ]

# plt.imshow(bed[top_left[0]:bottom_right[0], top_left[1]:bottom_left[1], :])
# plt.show()

def check_pts(p, pts):
    return p.contains_points([pts])

def inpolygon(img_points):
    p = path.Path(img_points) 

    pts = []

    for y in range(bed.shape[1]):
        for x in range(bed.shape[0]):
            if check_pts(p, (x,y)):
                pts.append([x,y])
    return np.array(pts)

def calculate_interior_pts(img_points):
    x_range = range(top_left[0], top_right[0])
    y_range = range(top_left[1], bottom_right[1])

    sample_pts = []

    for y in y_range:
        for x in x_range:
            sample_pts.append(np.array([x,y])) 

    return(np.array(sample_pts))

def est_homography(img_pts, gun_pts):

    A = []

    for i in range(len(img_pts)):
        xv = img_pts[i][0];
        yv = img_pts[i][1];
        xl = gun_pts[i][0];
        yl = gun_pts[i][1];
        ax = [ -xv, -yv, -1, 0, 0, 0, xv*xl, yv*xl, xl]
        ay = [ 0, 0, 0, -xv, -yv, -1, xv*yl, yv*yl, yl]

        A.append(ax)
        A.append(ay)

    A = np.array(A)

    U, S, Vh = np.linalg.svd(A)
    L = Vh[-1,:] / Vh[-1,-1]
    H = L.reshape(3, 3)

    return H

def warp_pts():

    sample_pts = inpolygon(img_points)
    # print(warp_pts.shape)

    H = est_homography(img_points,gun_points)

    warped_pts = []
    for i in range(sample_pts.shape[0]):
        x = sample_pts[i,0]
        y = sample_pts[i,1]

        X = np.array([x,y,1])

        X_bar = np.dot(H,X.T)

        warped_pt = [X_bar[0]/X_bar[2],X_bar[1]/X_bar[2]] 
        warped_pts.append(warped_pt)

    return np.array(warped_pts), sample_pts

def my_sub2ind(pts,img):
    rows = pts[:,1]
    cols = pts[:,0]
    return np.array(rows + (cols-1)*img.shape[0], dtype=int)

def inverse_warping(bed, guney, sample_pts, warped_pts):
    pts_final   = np.array(np.ceil(sample_pts), dtype=int)
    pts_final[pts_final < 0] = 0

    pts_initial = np.array(np.ceil(warped_pts),dtype=int)
    pts_initial[pts_initial < 0] = 0
    
    nPts = sample_pts.shape[0]

    projected_img = bed

    for color in range(3):
        sub_img_final   = bed[:,:,color]
        sub_img_initial = guney[:,:,color]

        for i in range(nPts):
            print(sub_img_final[pts_final[i][1],pts_final[i][0]], sub_img_initial[pts_initial[i][0]-1,pts_initial[i][1]-1])

            sub_img_final[pts_final[i][1],pts_final[i][0]] = sub_img_initial[pts_initial[i][0]-1,pts_initial[i][1]-1]

        projected_img[:,:,color] = sub_img_final

    plt.imshow(projected_img)
    plt.show()


warped_pts,sample_pts = warp_pts()

for pt in warped_pts:
    if pt.any() < 0:
        print(pt)

inverse_warping(bed, guney, sample_pts, warped_pts)

