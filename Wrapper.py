from glob import glob
from scipy import optimize
from pprint import pprint
import numpy as np
import cv2
import os

def world_points():
    x, y = np.meshgrid(range(9), range(6))
    x = x.reshape((54, -1))
    y = y.reshape((54,-1))
    world_points = np.hstack((x,y))*21.5
    return world_points

def get_chessboard_points(img, res):
    ret, corners = cv2.findChessboardCorners(res, (9, 6),flags=cv2.CALIB_CB_ADAPTIVE_THRESH +cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
    if ret:
        fnl = cv2.drawChessboardCorners(img, (9, 6), corners, ret)
        corners = corners.reshape((54,-1))
        return fnl, corners
    else:
        print("No Checkerboard Found")
        return np.zeros(img.shape), 0
def find_homography(M, m):
    homography_matrix, _= cv2.findHomography(M, m)
    return homography_matrix

def calc_v(H,i,j):
    i,j = i-1,j-1
    v_ij = np.array([H[0, i]*H[0, j],
                    H[0, i]*H[1, j] + H[1, i]*H[0, j],
                    H[1, i]*H[1, j],
                    H[2, i]*H[0, j] + H[0, i]*H[2, j],
                    H[2, i]*H[1, j] + H[1, i]*H[2, j],
                    H[2, i]*H[2, j] 
                    ])
    return v_ij

def find_intrinsic_matrix(V):
    # solution for Vb = 0 using SVD
    U, S, Vt = np.linalg.svd(np.array(V))
    # b = Vt[np.argmin(S)]
    b11, b12, b22, b13, b23, b33 = Vt[np.argmin(S)]
    # print('This is b matrix', b)
    v_0 = (b12*b13 - b11*b23)/(b11*b22 - b12**2)
    lambda_ = b33 - (b13**2 + v_0*(b12*b13 - b11*b23))/b11
    alpha = np.sqrt(lambda_/b11)
    # print(b11*b22 - b12**2)
    beta = np.sqrt(lambda_*b11 /(b11*b22 - b12**2))
    gamma = -1*b12*(alpha**2)*beta/lambda_
    u_0 = gamma*v_0/beta -b13*(alpha**2)/lambda_

    A = np.array([[alpha, gamma, u_0],[0, beta, v_0],[0, 0, 1]])
    return A

def find_extrinsic_param(A, h_matrix):

    h1 = h_matrix[:,0]
    h2 = h_matrix[:,1]
    h3 = h_matrix[:,2]
    _lambda = 1/np.linalg.norm(np.linalg.inv(A).dot(h1))
    r1 = _lambda*np.linalg.inv(A).dot(h1)
    r2 = _lambda*np.linalg.inv(A).dot(h2)
    r3 = np.cross(r1,r2)
    
    t = _lambda*np.linalg.inv(A).dot(h3)

    return np.stack((r1,r2,r3,t), axis=1)

def calc_reprojection_error(intrinsic, img_pts, world_pts, extrinsic):
    P = np.dot(intrinsic, extrinsic)
    error = []
    # print(real_pts)
    for x,y in zip(img_pts, world_pts):
        real_pts = np.hstack((y, 0, 1))
        board_pts = np.hstack((x, 1))
		# print(model_pts)
        proj_points = np.dot(P,real_pts)
        # print(proj_points)
        proj_points = proj_points/proj_points[2]
        error.append(np.linalg.norm(board_pts-proj_points, ord=2))

    return np.mean(error)

def loss_function(init, i_pts, w_pts, h_matrices):
    a = np.zeros(shape=(3, 3))
    
    a[0, 0], a[1, 1], a[0, 2], a[1, 2], a[0, 1], a[2,2] = init[0], init[1], init[2], init[3], init[4], 1
    k1, k2 = init[5], init[6]
    u0, v0 = init[2], init[3]
    err = []
    i = 0
    for im_pts, H in zip(i_pts, h_matrices):
        ext_mat = find_extrinsic_param(a, H)
        for pt, w_pt in zip(im_pts, w_pts):
            w_hom = np.array([[w_pt[0]], [w_pt[1]], [0], [1]], dtype = 'float')
            
            proj = np.dot(ext_mat, w_hom)
            proj = proj/proj[2]
            x, y = proj[0], proj[1]

            U = np.dot(a, proj)
            U = U/U[2]
            u, v = U[0], U[1]

            t = x**2 + y**2
            u_cap = u + (u-u0)*(k1*t + k2*(t**2))
            v_cap = v + (v-v0)*(k1*t + k2*(t**2))
            err.append(pt[0]-u_cap)
            err.append(pt[1]-v_cap)
          

    return np.float64(err).flatten()



def optimize_params(intrinsic_mat, h_matrices, i_pts, w_pts):
    alpha = intrinsic_mat[0, 0]
    beta= intrinsic_mat[1, 1]
    gamma = intrinsic_mat[0, 1]
    u_0 = intrinsic_mat[0, 2]
    v_0 = intrinsic_mat[1, 2]
    k1 = 0
    k2 = 0
    init = [alpha, beta, u_0, v_0, gamma, k1, k2]
    final = optimize.least_squares(fun=loss_function, x0=init,method="lm", args=[i_pts, w_pts, h_matrices])

    [alpha, beta, u_0, v_0, gamma, k1, k2] = final.x
    final_A = np.array([[alpha, gamma, u_0],[0, beta,  v_0],[0, 0, 1]])

    return final_A,k1,k2


def estimateReprojectionErrorDistortion(K,extrinsic,imgpoints, objpoints, k1, k2):
    err = []
    reproject_points = []

    u0, v0 = K[0, 2], K[1, 2]


    for impt, objpt in zip(imgpoints, objpoints):
        model = np.array([[objpt[0]], [objpt[1]], [0], [1]])
        proj_point = np.dot(extrinsic, model)
        proj_point = proj_point/proj_point[2]
        x, y = proj_point[0], proj_point[1]

        U = np.dot(K, proj_point)
        U = U/U[2]
        u, v = U[0], U[1]

        t = x**2 + y**2
        u_cap = u + (u-u0)*(k1*t + k2*(t**2))
        v_cap = v + (v-v0)*(k1*t + k2*(t**2))

        reproject_points.append([u_cap, v_cap])

        err.append(np.sqrt((impt[0]-u_cap)**2 + (impt[1]-v_cap)**2))

    return np.mean(err), reproject_points

def rectify(imgpoints,rect_points,images):
    for i,(image, imgpoints, optpoints) in enumerate(zip(images, imgpoints, rect_points)):
        img = cv2.imread(image)
        optpoints = np.array(optpoints)
        optpoints = optpoints.reshape(((54,-1)))
        H,_ = cv2.findHomography(imgpoints,np.array(optpoints))
        img_warp = cv2.warpPerspective(img,H,(img.shape[1],img.shape[0]))
        for point in optpoints:
            point = point.astype(int)
            cv2.circle(img_warp,(point),5,(0,0,255),-1)
        cv2.imwrite("Output/rectify_{}.jpg".format(i), img_warp)

def main():
    # cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
    # cv2.resizeWindow("frame", 700, 700)
    image_files = sorted(glob('Calibration_Imgs/*.jpg'))
    if not os.path.isdir('Output'):
        os.mkdir('Output')
    m = []
    H = []
    extrinsic_matrices = []
    reprojection_err = []
    final_extrinsic_matrices = []
    final_reprojection_err = []
    rectified_points=[]
    img_count = 0
    points_world = world_points()
    print('Calculating Initial parameters')
    for i in image_files:
        
        img1 = cv2.imread(i)
        gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        drawn_corners, corners = get_chessboard_points(img1, gray)
        H_matrix = find_homography(points_world, corners)
        m.append(corners)
        H.append(H_matrix)
        v_11 = calc_v(H_matrix, 1, 1)
        v_12 = calc_v(H_matrix, 1, 2)
        v_22 = calc_v(H_matrix, 2, 2)
        v = np.vstack(((v_12), (v_11 - v_22)))
        if img_count == 0:
            v_total = v
        else:
            v_total = np.vstack((v_total, v))
        
        img_count +=1 
    #     cv2.imshow("frame", drawn_corners)
    #     cv2.waitKey(0)

    # cv2.destroyAllWindows()
    A = find_intrinsic_matrix(v_total)
    print('********Initial Intrinsic matrix**********')
    pprint(A)
    for j in range(len(H)):
        extrinsic_mat = find_extrinsic_param(A, H[j])
        extrinsic_matrices.append(extrinsic_mat)
        e = calc_reprojection_error(A, m[j], points_world, extrinsic_mat)
        reprojection_err.append(e)
    
    print('The error before optimization is ', np.mean(reprojection_err))
    print('**********Optimization started***********')
    optimized_A, k1,k2 = optimize_params(A, H, m, points_world)
    

    print('********Final Intrinsic matrix**********')
    pprint(optimized_A)
    print('k1', k1)
    print('k2', k2)

    for j in range(len(H)):
        final_extrinsic_mat = find_extrinsic_param(optimized_A, H[j])
        final_extrinsic_matrices.append(final_extrinsic_mat)
        e, new_points = estimateReprojectionErrorDistortion(optimized_A,final_extrinsic_mat, m[j], points_world, k1,k2)
        rectified_points.append(new_points)
        final_reprojection_err.append(e)
    
    print('The error after optimization is ', np.mean(final_reprojection_err))
    rectify(m, rectified_points, image_files )

if __name__ == '__main__':
    main()