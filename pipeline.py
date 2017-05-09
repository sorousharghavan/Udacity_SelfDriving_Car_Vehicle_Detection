from helpers import *

color_space = 'RGB' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
spatial_size = (32, 32) # Spatial binning dimensions
hist_bins = 32    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off
y_start_stop = [300, None] # Min and max in y to search in slide_window()

def process_frame(image):
    draw_image = np.copy(image)
    heatmap = np.zeros_like(image)
    y_start_stop = [350, 550] 
    window_size = (96, 96)
    overlap = (0.8, 0.8)
    windows = slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop, 
                        xy_window=window_size, xy_overlap=overlap)

    hot_windows = search_windows(image, windows, svc, X_scaler, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)                       

    add_heat(heatmap, hot_windows)

    y_start_stop = [400, 600] 
    window_size = (1.25*96, 1.25*96)
    windows = slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop, 
                        xy_window=window_size, xy_overlap=overlap)

    hot_windows = search_windows(image, windows, svc, X_scaler, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)                       


    add_heat(heatmap, hot_windows)

    y_start_stop = [400, None] 
    window_size = (2*96, 2*96)
    windows = slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop, 
                        xy_window=window_size, xy_overlap=overlap)

    hot_windows = search_windows(image, windows, svc, X_scaler, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)                       

    #window_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)                    

    add_heat(heatmap, hot_windows)
    heatmap = apply_threshold(heatmap, 3)
    labels = label(heatmap)

    window_img = draw_labeled_bboxes(draw_image, labels)

    return window_img

image = mpimg.imread('./test_images/test1.jpg')
fname = "classifier.pkl"
scaler_filename = "scaler.pkl"
if (os.path.isfile(fname) and os.path.isfile(scaler_filename)):
    svc = joblib.load(fname)
    X_scaler = joblib.load(scaler_filename)
else:
    # Divide up into cars and notcars
    images = glob.glob('./non-vehicles/*.png')
    #shuffle(images)
    cars = []
    notcars = []
    for i in images:
        notcars.append(i)
    images = glob.glob('./vehicles/*.png')
    #shuffle(images)
    for i in images:
        cars.append(i)

    # Reduce the sample size because
    # The quiz evaluator times out after 13s of CPU time
    sample_size = 7000
    cars = cars[0:sample_size]
    notcars = notcars[0:sample_size]

    car_features = extract_features(cars, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)
    notcar_features = extract_features(notcars, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)

    X = np.vstack((car_features, notcar_features)).astype(np.float64)                        
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    joblib.dump(X_scaler, scaler_filename) 
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.2, random_state=rand_state)

    print('Using:',orient,'orientations',pix_per_cell,
        'pixels per cell and', cell_per_block,'cells per block')
    print('Feature vector length:', len(X_train[0]))
    # Use a linear SVC 
    svc = LinearSVC()
    # Check the training time for the SVC
    t=time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    joblib.dump(svc, fname)
    print(round(t2-t, 2), 'Seconds to train SVC...')
    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
    # Check the prediction time for a single sample
    t=time.time()

output_file = 'test.mp4'
clip1 = VideoFileClip("project_video.mp4")
output_clip = clip1.fl_image(process_frame)
output_clip.write_videofile(output_file, audio=False)

#image = mpimg.imread('bbox-example-image.jpg')
#draw_image = np.copy(image)

# Uncomment the following line if you extracted training
# data from .png images (scaled 0 to 1 by mpimg) and the
# image you are searching is a .jpg (scaled 0 to 255)
#image = image.astype(np.float32)/255

#windows = slide_window(image, x_start_stop=[600, None], y_start_stop=y_start_stop, 
 #                   xy_window=(96, 96), xy_overlap=(0.8, 0.8))

#hot_windows = search_windows(image, windows, svc, X_scaler, color_space=color_space, 
 #                       spatial_size=spatial_size, hist_bins=hist_bins, 
 #                       orient=orient, pix_per_cell=pix_per_cell, 
 #                       cell_per_block=cell_per_block, 
 #                      hog_channel=hog_channel, spatial_feat=spatial_feat, 
  #                      hist_feat=hist_feat, hog_feat=hog_feat)                       

#window_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)                    

#heatmap = np.zeros_like(image)
#add_heat(heatmap, hot_windows)
#heatmap = apply_threshold(heatmap, 5)
#labels = label(heatmap)

#window_img = draw_labeled_bboxes(draw_image, labels)                    

#plt.imshow(window_img)
#plt.show()
