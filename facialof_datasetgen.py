import numpy as np
import scipy
import scipy.interpolate
import os
import shutil
from scipy.spatial import Delaunay
import cv2
import subprocess
import csv
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

"""
=============
Flow Section
=============
"""
TAG_FLOAT = 202021.25
def read_flow(file):
    assert type(file) is str, "file is not str %r" % str(file)
    assert os.path.isfile(file) is True, "file does not exist %r" % str(file)
    assert file[-4:] == ".flo", "file ending is not .flor %r" % file[-4:]
    f = open(file, 'rb')
    flo_number = np.fromfile(f, np.float32, count = 1)[0]
    assert flo_number == TAG_FLOAT, 'Flow number %r incorrect. Invalid .flo file' % flo_number
    w = np.fromfile(f, np.int32, count = 1)
    h = np.fromfile(f, np.int32, count =1)
    try:
        data = np.fromfile(f, np.float32, count = 2*w*h)
    except:
        data = np.fromfile(f, np.float32, count = 2*w[0]*h[0])
    flow = np.resize(data, (int(h), int(w), 2))
    f.close()
    return flow

TAG_STRING = 'PIEH'
def write_flow(flow, filename):
    assert type(filename) is str, "file is not str %r" % str(filename)
    assert filename[-4:] == '.flo', "file ending is not .flo %r" % file[-4:]
    height, width, nBands = flow.shape
    assert nBands == 2, "number of bands = %r != 2" % nBands
    u = flow[:, :, 0]
    v = flow[:, :, 1]
    assert u.shape == v.shape, "Invalid flow shape"
    height, width = u.shape
    pyvers = sys.version_info[0]
    if pyvers < 3:
        f = open(filename, 'wb')
    elif pyvers >= 3:
        f = open(filename, 'w')
    f.write(TAG_STRING)
    np.array(width).astype(np.int32).tofile(f)
    np.array(height).astype(np.int32).tofile(f)
    tmp = np.zeros((height, width * nBands))
    tmp[:, np.arange(width)*2] = u
    tmp[:, np.arange(width)*2 + 1] = v
    tmp.astype(np.float32).tofile(f)
    f.close()

def makeColorwheel():

	#  color encoding scheme

	#   adapted from the color circle idea described at
	#   http://members.shaw.ca/quadibloc/other/colint.htm

	RY = 15
	YG = 6
	GC = 4
	CB = 11
	BM = 13
	MR = 6

	ncols = RY + YG + GC + CB + BM + MR

	colorwheel = np.zeros([ncols, 3]) # r g b

	col = 0
	#RY
	colorwheel[0:RY, 0] = 255
	colorwheel[0:RY, 1] = np.floor(255*np.arange(0, RY, 1)/RY)
	col += RY

	#YG
	colorwheel[col:YG+col, 0]= 255 - np.floor(255*np.arange(0, YG, 1)/YG)
	colorwheel[col:YG+col, 1] = 255;
	col += YG;

	#GC
	colorwheel[col:GC+col, 1]= 255 
	colorwheel[col:GC+col, 2] = np.floor(255*np.arange(0, GC, 1)/GC)
	col += GC;

	#CB
	colorwheel[col:CB+col, 1]= 255 - np.floor(255*np.arange(0, CB, 1)/CB)
	colorwheel[col:CB+col, 2] = 255
	col += CB;

	#BM
	colorwheel[col:BM+col, 2]= 255 
	colorwheel[col:BM+col, 0] = np.floor(255*np.arange(0, BM, 1)/BM)
	col += BM;

	#MR
	colorwheel[col:MR+col, 2]= 255 - np.floor(255*np.arange(0, MR, 1)/MR)
	colorwheel[col:MR+col, 0] = 255
	return 	colorwheel

def computeColor(u, v):

	colorwheel = makeColorwheel();
	nan_u = np.isnan(u)
	nan_v = np.isnan(v)
	nan_u = np.where(nan_u)
	nan_v = np.where(nan_v) 

	u[nan_u] = 0
	u[nan_v] = 0
	v[nan_u] = 0 
	v[nan_v] = 0

	ncols = colorwheel.shape[0]
	radius = np.sqrt(u**2 + v**2)
	a = np.arctan2(-v, -u) / np.pi
	fk = (a+1) /2 * (ncols-1) # -1~1 maped to 1~ncols
	k0 = fk.astype(np.uint8)	 # 1, 2, ..., ncols
	k1 = k0+1;
	k1[k1 == ncols] = 0
	f = fk - k0

	img = np.empty([k1.shape[0], k1.shape[1],3])
	ncolors = colorwheel.shape[1]
	for i in range(ncolors):
		tmp = colorwheel[:,i]
		col0 = tmp[k0]/255
		col1 = tmp[k1]/255
		col = (1-f)*col0 + f*col1
		idx = radius <= 1
		col[idx] = 1 - radius[idx]*(1-col[idx]) # increase saturation with radius    
		col[~idx] *= 0.75 # out of range
		img[:,:,2-i] = np.floor(255*col).astype(np.uint8)

	return img.astype(np.uint8)


def computeImg(flow):

	eps = sys.float_info.epsilon
	UNKNOWN_FLOW_THRESH = 1e9
	UNKNOWN_FLOW = 1e10

	u = flow[: , : , 0]
	v = flow[: , : , 1]

	maxu = -999
	maxv = -999

	minu = 999
	minv = 999

	maxrad = -1
	#fix unknown flow
	greater_u = np.where(u > UNKNOWN_FLOW_THRESH)
	greater_v = np.where(v > UNKNOWN_FLOW_THRESH)
	u[greater_u] = 0
	u[greater_v] = 0
	v[greater_u] = 0 
	v[greater_v] = 0

	maxu = max([maxu, np.amax(u)])
	minu = min([minu, np.amin(u)])

	maxv = max([maxv, np.amax(v)])
	minv = min([minv, np.amin(v)])
	rad = np.sqrt(np.multiply(u,u)+np.multiply(v,v)) 
	maxrad = max([maxrad, np.amax(rad)])
	print('max flow: %.4f flow range: u = %.3f .. %.3f; v = %.3f .. %.3f\n' % (maxrad, minu, maxu, minv, maxv))

	u = u/(maxrad+eps)
	v = v/(maxrad+eps)
	img = computeColor(u, v)
	return img
def compute_flow(V, Vn):
        flow = np.zeros((V.shape[0], V.shape[0], 2))
        X, Y = V[:, 0], V[:, 1] 
        Xn, Yn = Vn[:, 0], Vn[:, 1]
        flow[:, :, 0] = np.c_[Y, X]
        flow[:, :, 1] = np.c_[Y, X]
        flowu, flowv = Xn - X, Yn - Y
        flow[Y, X, 0] = flowu
        flow[Y, X, 1] = flowv
        u, v = np.zeros((Yn.shape[0], Xn.shape[0], 1)), np.zeros((Yn.shape[0], Xn.shape[0], 1))
        u[:, :, 0], v[:, :, 1] = np.c_[Yn, Xn], np.c_[Yn, Xn]
        u[:] = Xn - X
        v = Yn - Y
        return u, v
"""
=============
End of Flow Section
=============
"""
def get_before_extension(filename):
    new_name = ''
    for i, letters in enumerate(filename):
        if letters == '.':
            break
        new_name = new_name + filename[i]
    return new_name

def read_csv_data(csv_files):
    float_data = []
    string_data = []
    for new_files in csv_files:
        with open(new_files) as csv_file:
            for j, row in enumerate(csv.reader(csv_file)):
                if j == 0:
                    for elements in row:
                            string_data.append(elements)
                else:
                    data = row
                    for elements in data:
                        float_data.append(float(elements))
    return string_data, float_data

def get_csv_landmarks(string_data, float_data, eye_num):
    x_lm = np.zeros(eye_num + 1 + 68)
    y_lm = np.zeros(eye_num + 1 + 68)
    e_x_0 = string_data.index(" eye_lmk_x_0")
    e_x_f = string_data.index(" eye_lmk_x_55")
    x_lm[0:eye_num] = float_data[e_x_0:e_x_0 + eye_num]
    e_y_0 = string_data.index(" eye_lmk_y_0")
    e_y_f = string_data.index(" eye_lmk_y_55")
    y_lm[0:eye_num] = float_data[e_y_0:e_y_0 + eye_num]
    x_0 = string_data.index(" x_0")
    x_f = string_data.index(" x_67")
    y_0 = string_data.index(" y_0")
    y_f = string_data.index(" y_67")
    x_lm[eye_num + 1:] = float_data[int(x_0): int(x_f)+ 1]
    y_lm[eye_num + 1:] = float_data[int(y_0): int(y_f)+1]
    landmarks = np.c_[np.array(y_lm), np.array(x_lm)]
    origin = np.where(landmarks == [0, 0])
    for origins in origin:
        landmarks = np.delete(landmarks, origins, axis = 0)
    return landmarks

def affine_transform_triangles(ypq,ypr,yqr,U, N):
    affine_pq = np.matmul(np.c_[ypq, np.ones((N,1))], U)
    affine_pr = np.matmul(np.c_[ypr, np.ones((N,1))], U)
    affine_qr = np.matmul(np.c_[yqr, np.ones((N,1))], U)
    return affine_pq, affine_pr, affine_qr

def generate_triangle(p,q,r,N):
    t = np.linspace(0,1,N)
    parameter = np.array([1-t,1-t]).T
    ypq = p + parameter * (q-p)
    ypr = p + parameter * (r-p)
    yqr = q + parameter * (r-q)
    return ypq, ypr, yqr

def get_triangle_map(p,q,r,p1,q1,r1):
    u = np.array([np.append(p,1), np.append(q,1), np.append(r,1)])
    X = np.array([np.append(p1,1), np.append(q1,1), np.append(r1,1)])
    try:
        U =  np.matmul(np.linalg.inv(u),X)
    except:
        U = np.eye(3)
    return U

def triangle_interior(p,q,r):
    ly = 0.9 * np.min(np.array([p[1], q[1], r[1]]))
    uy = 1.1 * np.max(np.array([p[1], q[1], r[1]]))
    lx = 0.9 * np.min(np.array([p[0], q[0], r[0]]))
    ux = 1.1 * np.max(np.array([p[0], q[0], r[0]]))
    x_grid = np.arange(lx, ux, 1)
    y_grid = np.arange(ly, uy, 1) 
    x = x_grid
    y = y_grid
    d = q[0] * r[1] - q[1] * r[0] + p[1] * (-q[0] + r[0]) + p[0] * (q[1] - r[1])
    pr = p[1] * r[0] - p[0] * r[1]
    pq = p[0] * q[1] - p[1] * q[0]
    rp1 = float(r[1] - p[1])
    pr0 = float(p[0] - r[0])
    pq = p[0] * q[1] - p[1] * q[0]
    pq1 = float(p[1] - q[1])
    qp0 = float(q[0] - p[0])
    V = []
    X_interior = []
    Y_interior = []
    V_integer = []
    for i in x:
        for j in y:
            s = float(pr + rp1*i + pr0*j)/float(d)
            t = float(pq + pq1*i + qp0*j)/float(d)
            if s >= 0 and t >= 0 and s + t <= 1:
                V.append(np.array([i, j]))
                V_integer.append(np.array([int(i), int(j)]))
                X_interior.append(int(i))
                Y_interior.append(int(j))
    return np.asarray(V), np.asarray(V_integer), np.asarray(X_interior), np.asarray(Y_interior)
    
def cubic_interpolate(yr, xr, image):
    try:
        z1 = cv2.imread(image)[:,:,2]
        z2 = cv2.imread(image)[:,:,1]
        z3 = cv2.imread(image)[:,:,0]
    except:
        try:
            z1 = image[0, :, :, 2]
            z2 = image[0, : ,: , 1]
            z3 = image[0, :, :, 3]
        except:
            z1 = image[:, :, 2]
            z2 = image[:, :, 1]
            z3 = image[:, :, 0]
    try:
        cubic1 = scipy.interpolate.interp2d(yr, xr, z1.T)
        cubic2 = scipy.interpolate.interp2d(yr, xr, z2.T)
        cubic3 = scipy.interpolate.interp2d(yr, xr, z3.T)
    except:
        cubic1 = scipy.interpolate.interp2d(yr, xr, z1)
        cubic2 = scipy.interpolate.interp2d(yr, xr, z2)
        cubic3 = scipy.interpolate.interp2d(yr, xr, z3)
    return cubic1, cubic2, cubic3

def triangulate(landmarks):
        try:
            tri = Delaunay(landmarks[:,0:2])
        except:
            tri = Delaunay(landmarks.T)
        N_triangle = tri.simplices.shape[0]
        tri_count = 0
        tri_indices = np.zeros(N_triangle)
        P = []
        for k_triangle in range(N_triangle):
            try:
                triangle_points = landmarks[tri.simplices[k_triangle,:]]
            except:
                triangle_points = landmarks.T[tri.simplices[k_triangle,:]]
            # get p, q, r vertices of triangle
            pn, qn, rn = triangle_points
            # produce triangle lines ypqn, yprn, yqrn joining vertices pq, pr, and qr
            ypqn, yprn, yqrn = generate_triangle(pn, qn, rn, N_points)
            _, V_integer_n, _, _ = triangle_interior(pn, qn, rn) # return interior of the triangle
            # Add the three vertices to P, to store triangle information to infer affine map
            P.append([int(pnp) for pnp in pn])
            P.append([int(qnp) for qnp in qn])
            P.append([int(rnp) for rnp in rn])
            try: 
                for vints in V_integer_n:
                    new_array = np.zeros(2)
                    for vv,vintsp in enumerate(vints):
                        new_array[vv] = int(vintsp)
                    P.append(new_array)
                N_interior = np.asarray(V_integer_n).shape[0]
                tri_count += N_interior + 3
                tri_indices[k_triangle] = tri_count
            except:
                tri_count += 3
                tri_indices[k_triangle] = tri_count
        P = np.asarray(P).reshape((len(P),2))
        return np.asarray(P).reshape((len(P), 2)), tri, tri_indices, ypqn, yprn, yqrn,pn,qn,rn, N_triangle

def resample_flow(P, flow_field):
    minX = np.min(P[:, 1]); maxX = np.max(P[:, 1]); minY = np.min(P[:, 0]); maxY = np.max(P[:, 0])
    Y, X = np.meshgrid(np.arange(int(minY), int(maxY), 1), np.arange(int(minX), int(maxX), 1))
    tups = np.rec.fromarrays([Y, X], names = 'Y,X')
    arranged_tuples = np.zeros((flow_field.shape[0]*flow_field.shape[1],2))
    count = 0
    for tuples in tups:
        for inner_tuples in tuples:
            arranged_tuples[count, :] = np.asarray(list(inner_tuples))
            count += 1
    resampled_flow_field = scipy.interpolate.griddata(P, flow_field, arranged_tuples, 'linear')
    return resampled_flow_field, arranged_tuples

def tracking_and_csv(current_image, jpg_save_name, openface_dir, current_dir, save_dir):
    try:
        os.chdir(openface_dir)
        cv2.imwrite(openface_dir + jpg_save_name, current_image)
        subprocess.call(["./bin/FaceLandmarkImg", "-f", openface_dir + jpg_save_name])
        os.system('rm ' + jpg_save_name)
        os.chdir("processed")
        csv_files = [get_before_extension(jpg_save_name) + '.csv']
        print(csv_files)
        dest = shutil.copy(csv_files[0], save_dir + '/' + csv_files[0])
        string_data, float_data = read_csv_data(csv_files)
        eye_num = 0
        landmarks = get_csv_landmarks(string_data, float_data, eye_num)
        save_files = [get_before_extension(jpg_save_name) + '.jpg', get_before_extension(jpg_save_name) + '.csv', get_before_extension(jpg_save_name) + '.hog']
        save_files.append(get_before_extension(jpg_save_name) + '_of_details.txt')
        os.chdir(current_dir)
    except:
        os.chdir(openface_dir + "processed")
        save_files = []
        landmarks = ["Continue"]
        os.chdir(current_dir)
    return landmarks, save_files
def Resize_image(images, Resize):
    try:
        Test = images[0, :, :, :]
    except:
        Test = images
    W = Resize
    height, width, depth = Test.shape
    try:
        m = len(images[:,0,0,0])
    except:
        m = 1
    resized_images = np.zeros((m,Resize[0],Resize[1],depth))
    for k in range(m):
        try:
            Test = images[k, :, :, :]
        except:
            Test = images
        Test = cv2.resize(Test, (Resize[1], Resize[0]))
        resized_images[k,:,:]=Test
    return resized_images

def Resize_image_withlandmarks(images, Resize, landmarks):
    try:
        Test = images[0, :, :, :]
    except:
        Test = images
    height, width, depth = Test.shape
    lm_x, lm_y = landmarks[:, 0], landmarks[:, 1]
    new_lm_x, new_lm_y = float(float(Resize[1])/float(width)) * lm_x, float(float(Resize[0])/float(height)) * lm_y
    new_lm = np.c_[new_lm_x, new_lm_y]
    try:
        m = len(images[:, 0, 0, 0])
    except:
        m = 1
    resized_images = np.zeros((m,Resize[0],Resize[1],depth))
    for k in range(m):
        try:
            Test = images[k, :, :, :]
        except:
            Test = images
        Test = cv2.resize(Test, (Resize[1], Resize[0]))
        resized_images[k, :, :] = Test
    return resized_images, new_lm

def Read_Two_Column_File(file_name):
    with open(file_name, 'r') as data:
        x = []
        y = []
        for line in data:
            p = line.split()
            x.append(float(p[0]))
            y.append(float(p[1]))
    return x, y

def delete_files(filepath, this_file):
    files = os.listdir(filepath)
    os.chdir(filepath)
    for process_files in files:
        if os.path.isdir(filepath + process_files) and (process_files in this_file):
            subprocess.call(["rm","-r",process_files])
        elif (process_files in this_file):
            subprocess.call(["rm",process_files])

def mkdir_p(mypath):
    '''Creates a directory. equivalent to using mkdir -p on the command line'''
    from errno import EEXIST
    from os import makedirs, path

    try:
        makedirs(mypath)
    except OSError as exc: # Python >2.5
        if exc.errno == EEXIST and path.isdir(mypath):
            pass
        else: raise

img_formats = ['jpg', 'jpeg', 'png','bmp', 'jp2', 'webp', 'pbm', 'pgm', 'ppm', 'pxm', 'tiff', 'tif', 'hdr', 'pic']
# FOR THE USER TO MODIFY: the directory names for your dataset
openface_dir = "/home/malkaddour/OpenFace/build/" # Location of OpenFace build directory
data_root_path = '/hdd2/malkaddour/datasets/CK/CK+/CK+/' # Root directory of dataset
images_path = data_root_path + "cohn-kanade-images/" # Directory which contains image directories of subjects
save_path = data_root_path + "/CK+_opticalflow" # Root directory to save generated optical flow
generate_new_landmarks = True # Set this to false if you already have the landmarks
if not generate_new_landmarks:
    landmarks_path = data_root_path + "Landmarks/" # Directory which contains landmark directories of subjects

N_points = 100 # Number of points during affine mapping process
subject_string_all = os.listdir(images_path)
# transformations needed later
R_transform_y = np.array([[0, 1], [-1, 0]])
R_transform_x = np.array([[1, 0],[0, -1]])

mkdir_p(save_path)
try:
    os.mkdir(save_path)
except:
    FFFFFFF = 1000000 # Useless 

these_files = []
landmark_exception_indicator = 0

# Loop over all subjects
for S, subjects in enumerate(subject_string_all):
    subject_path = images_path + subjects + '/'
    if not generate_new_landmarks:
        subject_lm_path = landmarks_path + subjects + '/'
    subject_savepath = save_path + '/' + subjects # subject path in new OF directory
    mkdir_p(subject_savepath)
    if os.path.isdir(subject_path):
        sequence_list = os.listdir(subject_path) # Get list of sequences in current subject
    else:
        continue

    # Loop over the sequences in current subject
    for i, sequences in enumerate(sequence_list):
        current_dir = subject_savepath + '/' + sequences # sequence path in new OF directory
        sequence_path = subject_path + sequences + "/"
        if not generate_new_landmarks:
            sequence_lm_path = subject_lm_path + sequences + "/"
        if os.path.isdir(sequence_path):
            # Get list of all images in current subject + sequence, extracts only files with .jpg, .png, etc...
            all_objects = sorted([x for x in os.listdir(sequence_path) if x.split('.')[-1] in img_formats])
            if not generate_new_landmarks:
                # Get list of all landmark .txt files, 1 for each image
                all_landmarks = sorted(os.listdir(sequence_lm_path))
        else:
            continue
        j_count = 0
        file_del_counter = 0
        landmark_exception = 0
        # Make directories inside OF root to save files in
        mkdir_p(current_dir)
        mkdir_p(subject_savepath + '/' + 'flow_viz_' + sequences + '/')
        mkdir_p(subject_savepath + '/' + 'flow_' + sequences + '/')
        for j, objects in enumerate(all_objects):
            object_name_0 = all_objects[max(0, j-1)].split('.')[0] # .flo filename is name of image X0 in images (X0, X1)
            object_name_1 = objects.split('.')[0]
            if objects.startswith('._') or objects.startswith('.'):
                continue
            current_image = sequence_path + objects # current image to be used for processing
            if not generate_new_landmarks:
                current_lm =  sequence_lm_path + object_name_1 + "_landmarks.txt" # Current .txt file with landmarks
            H, W, D = cv2.imread(current_image).shape # dimensions of current image
            
            # preparing file names to save inside OF directory
            save_jpg = objects # current image name
            save_landmarks_path = subject_savepath + '/' + sequences # directory to save generated landmarks in
            save_tri = subject_savepath + '/' + 'flow_viz_' + sequences + '/' + object_name_1+ "_tri" + '.png' # triangulation visualization
            save_im_path = subject_savepath + '/' + 'flow_' + sequences + '/' + object_name_1 + '.jpg' # current image in OF directory
            save_flow_path = subject_savepath + '/' + 'flow_' + sequences + '/' + object_name_0 + '.flo' # .flo file
            save_flow_viz = subject_savepath + '/' + 'flow_viz_' + sequences + '/' + object_name_0 + '.png' # flow visualization
            image = cv2.imread(current_image)

            # GET LANDMARKS HERE
            if generate_new_landmarks: # will obtain landmarks via OpenFace
                image_resized = Resize_image(image, np.array([512, 384])) #FlowNet accepts 512 x 384
                if image_resized.shape[0] == 1:
                    image_resized = image_resized[0, :, :, :]
                # The tracking_and_csv instantiates OpenFace, generates and saves landmarks info, and returns landmarks
                landmarks, saved_file = tracking_and_csv(image_resized, save_jpg, openface_dir, current_dir, save_landmarks_path)
                these_files.append(saved_file) # list of image files saved in openface directory
                if landmarks[0] == "Continue":
                    continue
                # this will delete image files in openface directory after every 100 iterations
                if np.mod(file_del_counter, 100) == 0:
                    n_files = len(these_files)
                    delete_files(openface_dir + "processed/", these_files[0:n_files-10])
                    these_files = []
            else:
                # reads landmarks from .txt file, then resizes image + landmarks while accounting for new image dimensions
                lm_x, lm_y = Read_Two_Column_File(current_lm)
                old_lm = np.c_[lm_x, lm_y]
                image_resized, landmarks = Resize_image_withlandmarks(image, np.array([512, 384]), old_lm)
                lm_x, lm_y = landmarks[:, 0], landmarks[:, 1]
                landmarks = np.c_[lm_y, lm_x]

            try:
                if image_resized.shape[0] == 1:
                    image_resized = image_resized[0, :, :, :]
            except:
                1
            
            # Write resized image to new OF dataset
            cv2.imwrite(save_im_path, image_resized)
            try:
                H, W, D = image_resized.shape[1:4]
            except:
                H, W, D = image_resized.shape
            xr = np.arange(0, W, 1)
            yr = np.arange(0, H, 1)
            cubic1, cubic2, cubic3 = cubic_interpolate(yr, xr, image_resized)

            # Prepare array
            mapped_image = np.zeros((W, H, D), dtype = np.uint8)

            ind = 1
            ind1, ind2 = np.mod(ind, 2), np.mod(ind + 1, 2)
            # For the first image in the sequence, only triangulate (no OF to be computed)
            if j_count == 0:
                j_count += 1
                # Uses landmarks to triangulate and return vertices and indices information
                Pn, tri, tri_indices, affpqn, affprn, affqrn, pn, qn, rn, N_triangle = triangulate(landmarks)
                plt.figure()
                plt.imshow(image_resized)

                # Initialize flow field for next iteration
                flow_field = np.zeros((H, W, 2))
                temp_flow_field = np.zeros((H, W, 2))
                landmarks_new = landmarks
            if j > 0:
                tri_count = 0
                delta_t = 1
                P = Pn # Pn will be updated to be the pixels mapped by the affine map. All of them.
                Pn = []
                landmarks_old = landmarks_new
                landmarks_new = landmarks
                plt.figure()

                # Loop over all triangles produced by the Delaunay triangulation
                for k_triangle in range(N_triangle):
                    try:
                        # Use vertices index information to get landmark vertices
                        triangle_points_old = landmarks_old[tri.simplices[k_triangle,:]]
                        triangle_points_new = landmarks_new[tri.simplices[k_triangle,:]]
                    except:
                        # If landmarks are not properly computed previously, this will raise an exception and will skip this sequence
                        landmark_exception += 1
                        landmark_exception_indicator = 1
                        Pn = P
                        break
                    # Vertices of the current and previous triangle
                    p, q, r = triangle_points_old
                    pn, qn, rn = triangle_points_new
                    # produce triangle lines ypq, ypr, yqr joining vertices pq, pr, and qr                    
                    ypq, ypr, yqr = generate_triangle(p, q, r, N_points)
                    ypqn, yprn, yqrn = generate_triangle(pn, qn, rn, N_points)
                    # Uses old and new vertices to infer affine map U
                    U = get_triangle_map(p, q, r, pn, qn, rn)
                    current_tri_index = tri_indices[k_triangle]
                    # We return the identity matrix if the mapping is degenerate, and thus we continue to the next triangle
                    if np.all(np.equal(U, np.eye(3))):
                        tri_count = current_tri_index
                        Pn.append([pnp for pnp in pn])
                        Pn.append([qnp for qnp in qn])
                        Pn.append([rnp for rnp in rn])
                        continue
                    
                    # Affine map on all triangle lines 
                    affpq, affpr, affqr = affpqn, affprn, affqrn
                    affpqn, affprn, affqrn = affine_transform_triangles(ypq, ypr, yqr, U, N_points)
                    # Vertices in homogenous coordinates
                    Xcheck = np.c_[P[int(tri_count):int(current_tri_index), 0], P[int(tri_count):int(current_tri_index), 1], np.ones((int(current_tri_index - tri_count), 1))]
                    # New vertices in homogenous coordinates
                    mapped_pixels_homo = np.matmul(Xcheck, U)
                    tri_count = current_tri_index
                    # Add new vertices to big Pn matrix
                    for mpixs in mapped_pixels_homo[:, 0:2]:
                        Pn.append(mpixs)
                    # Visualize the triangulation
                    plt.plot(ypq[:,ind1], H-ypq[:,ind2], 'r--', ypr[:,ind1], H-ypr[:,ind2], 'r--', yqr[:,ind1], H-yqr[:,ind2], 'r--')
                # If there has been an exception with the landmarks in this image, continue to the next image in the sequence
                if landmark_exception_indicator:
                    landmark_exception_indicator = 0
                    continue
                # Make sure vertices matrix is an array
                Pn = np.asarray(Pn)
                plt.show()
                image_resized_plot = np.array(image_resized, dtype = 'uint8')
                image_resized_plot = image_resized_plot[:, :, [2, 1, 0]]
                # Overlay image on top of visualized triangulation
                plt.imshow(image_resized_plot, extent = [0, W, 0, H])
                plt.savefig(save_tri)
                plt.close()
                try:
                    assert (Pn.shape == P.shape), "Different number of mapped pixels."
                except:
                    continue
                try:
                    # Flow field computed and resampled on a discrete grid
                    current_flow = (Pn - P) / delta_t
                    temp_flow_field[P[:, 0].astype(int), P[:, 1].astype(int), :] = current_flow
                    resampled_flow, P_integer = resample_flow(P, current_flow)
                    flow_field[P_integer[:, 0].astype(int),P_integer[:, 1].astype(int), :] = resampled_flow
                    vec_positions = P_integer
                except:
                    continue
                # Orient the frame
                flow_field = np.matmul(flow_field, R_transform_y) 
                flow_field = np.matmul(flow_field, R_transform_x.T)
                # Save .flo file
                write_flow(flow_field, save_flow_path)

                # Compute and save flow visualization
                flow = read_flow(save_flow_path)
                img = computeImg(flow)
                cv2.imwrite(save_flow_viz, img)
                file_del_counter += 1 # to later delete files in OpenFace directory