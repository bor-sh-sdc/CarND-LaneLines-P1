import os
import math
#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
# jupyter specific
#%matplotlib inline

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def sobely(img, ksize=5):
    """
    Applies the sobel transform 
    More info: http://docs.opencv.org/3.1.0/d5/d0f/tutorial_py_gradients.html
    """
    return cv2.Sobel(img, cv2.CV_8U,1,0,ksize)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, thickness=2, color=[255, 0, 0]):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    # y = mx + b
    # works so far when lines are not screwed up by some side lines - like in the switchlane image - need to improve either canny or hough - knowledge gap here
    m_right = np.mean([ ((y2-y1)/(x2-x1)) for line in lines for x1,y1,x2,y2 in line if ((y2-y1)/(x2-x1)) > 0])
    b_right = np.mean([ y2 - ((y2-y1)/(x2-x1))*x2 for line in lines for x1,y1,x2,y2 in line if ((y2-y1)/(x2-x1)) > 0])

    x1 = 960
    x2 = 510

    y1 = int(m_right * x1 + b_right)
    y2 = int(m_right * x2 + b_right)

#    cv2.line(img, (x1, y1), (x2, y2), color, thickness)

    m_left = np.mean([ ((y2-y1)/(x2-x1)) for line in lines for x1,y1,x2,y2 in line if ((y2-y1)/(x2-x1)) < 0])
    b_left = np.mean([ y2 - ((y2-y1)/(x2-x1))*x2 for line in lines for x1,y1,x2,y2 in line if ((y2-y1)/(x2-x1)) < 0])

    x1 = 0
    x2 = 460

    y1 = int(m_left * x1 + b_left)
    y2 = int(m_left * x2 + b_left)

#    cv2.line(img, (x1, y1), (x2, y2), color, thickness)

    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, 2)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns hough lines
    """
    return cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    
    return cv2.addWeighted(initial_img, α, img, β, λ)

def find_lane_lines(path=None, toplot=False):
    """
    big bang line finding approach - improvable but step by step
    at the moment not really flexible (not the goal) and everything hard coded
    """
    if not path:
        return
    print("Processing "+path)
    #reading in an image
    image = mpimg.imread(path)

    #printing out some stats and plotting
    print('This image is:', type(image), 'with dimesions:', image.shape)

    gray = grayscale(np.copy(image))

    # Define a kernel size and apply Gaussian smoothing
    kernel_size = 3
    blur_gray = gaussian_blur(gray, kernel_size)
    ## played around with sobel - lets skip it for the moment
    ## looks good but some lines are aside need to check why - *brainfuck*
    #blur_gray = sobely(gray)

    # Define our parameters for Canny and apply
    low_threshold = 50
    high_threshold = 150
    edges = canny(blur_gray, low_threshold, high_threshold)

    # This time we are defining a four sided polygon to mask
    imshape = image.shape
    upper_thickness=50
    hight=60
    left_bottom=60
    right_bottom=50
    vertices = np.array([[(left_bottom, imshape[0]),(imshape[1]/2 - upper_thickness, imshape[0]/2 + hight), (imshape[1]/2 + upper_thickness, imshape[0]/2 + hight), (imshape[1] - right_bottom ,imshape[0])]], dtype=np.int32)

    masked_edges = region_of_interest(edges, vertices)

    # Define the Hough transform parameters
    # TODO: explanation in jupyter why choosing this parameter?
    rho = 1 # distance resolution in pixels of the Hough grid
    theta = np.pi/180 # angular resolution in radians of the Hough grid
    threshold = 20     # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 20 #minimum number of pixels making up a line
    max_line_gap = 5    # maximum gap in pixels between connectable line segments

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = hough_lines(masked_edges, rho, theta, threshold, min_line_length, max_line_gap)

    lines_edges = np.copy(image)
    draw_lines(lines_edges, lines, 10)
    ## why weighted_img?
    #lines_edges = weighted_img(image, lines_edges)

    if toplot:
       plot_images(image, vertices, lines_edges, edges, gray)

    if path:
        path = path.replace('.jpg','LinesAdded.jpg')
        mpimg.imsave(path, lines_edges)

def plot_images(image, vertices, lines_edges, edges, gray):
    """plots images - with additional vertices and some gray"""
    plt.subplot(321), plt.imshow(image)
    x = [vertices[0,0][0],vertices[0,1][0],vertices[0,2][0],vertices[0,3][0]]
    y = [vertices[0,0][1],vertices[0,1][1],vertices[0,2][1],vertices[0,3][1]]
    plt.plot(x,y, 'b--', lw=4)
    plt.subplot(322), plt.imshow(lines_edges)
    plt.subplot(323), plt.imshow(edges,  cmap='gray')
    plt.subplot(324), plt.imshow(gray, cmap='gray')
    #plt.subplot(325), plt.imshow(img_lines)

    plt.show()

def clean_up_images():
    """cleans generated images"""
    for f in os.listdir("test_images/"):
      if f.endswith("LinesAdded.jpg"):
          os.remove(os.path.join("test_images",f))

# let see what cv2 version is there
print(cv2.__version__)

if cv2.__version__ < "3.1.0":
  print("Oh oh this code was developed with version 3.1.0 of cv installed by conda") 

clean_up_images()

image="whiteCarLaneSwitch.jpg"
#find_lane_lines('test_images/'+image, True)

for image in os.listdir("test_images/"):
  find_lane_lines('test_images/'+image)