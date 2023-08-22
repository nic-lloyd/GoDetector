import cv2
import numpy as np
import math
from sklearn.cluster import KMeans
import os


class GoDetector:
    def __init__(self, image_path, size):
        self.image_path = image_path
        self.size = size    #size of the Go board
    
    def canny_processing(self, image):
        
        image = cv2.imread(image)
        # convert image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # use Canny edge detection
        edges = cv2.Canny(gray,200,300,apertureSize=3,L2gradient=True)
        # define the kernel for morphological transformation
        # kernel is the size of the neighborhood in pixels (5x5)
        kernel = np.ones((5,5),np.uint8)
        # dilate and erode the edges to merge parallel lines
        dilation = cv2.dilate(edges,kernel,iterations=1)
        erosion = cv2.erode(dilation,kernel,iterations=1)
        return erosion

    def hough_lines_corner_detection(self, input_image):
        lines = cv2.HoughLinesP(
            input_image,        # bw image with detected edges
            1,                  # distance resolution in pixels
            np.pi/180,          # angle resolution in radians
            threshold=200,      # min number of votes for a valid line
            minLineLength=100,  # min allowed length of line
            maxLineGap=350      # max allowed gap between line for joining them
        )
        # location of corners of the image
        top_left = (0,0)
        top_right = (0, input_image.shape[0])
        bot_left = (input_image.shape[1], 0)
        bot_right = (input_image.shape[1],input_image.shape[0])
        
        # represents the corners of the go board
        board_top_left= board_top_right= board_bot_left= board_bot_right = None
        
        # distance from the corners of the go board to the corners of the image
        top_left_dist= top_right_dist= bot_left_dist= bot_right_dist = 99999999
        
        # identify the closest corner to each endpoint of each line
        # and storing the minimum distance to represent the corners of the
        # lines on the board
        for points in lines:
            x1,y1,x2,y2=points[0]
            
            if math.dist((x1,y1),top_left) < top_left_dist:
                top_left_dist = math.dist((x1,y1),top_left)
                board_top_left = (x1,y1)
            
            if math.dist((x2,y2),top_left) < top_left_dist:
                top_left_dist = math.dist((x2,y2),top_left)
                board_top_left = (x2,y2)
            
            if math.dist((x1,y1),top_right) < top_right_dist:
                top_right_dist = math.dist((x1,y1),top_right)
                board_top_right = (x1,y1)
            
            if math.dist((x2,y2),top_right) < top_right_dist:
                top_right_dist = math.dist((x2,y2),top_right)
                board_top_right = (x2,y2)

            if math.dist((x1,y1),bot_left) < bot_left_dist:
                bot_left_dist = math.dist((x1,y1),bot_left)
                board_bot_left = (x1,y1)
            
            if math.dist((x2,y2),bot_left) < bot_left_dist:
                bot_left_dist = math.dist((x2,y2),bot_left)
                board_bot_left = (x2,y2)
                
            if math.dist((x1,y1),bot_right) < bot_right_dist:
                bot_right_dist = math.dist((x1,y1),bot_right)
                board_bot_right = (x1,y1)
            
            if math.dist((x2,y2),bot_right) < bot_right_dist:
                bot_right_dist = math.dist((x2,y2),bot_right)
                board_bot_right = (x2,y2)
                
        return (board_top_left,board_bot_left,board_bot_right,board_top_right)
    
    def perspective_transform(self, image_path, corners):
        image = cv2.imread(image_path)
        top_r,top_l,bot_l,bot_r = corners
        
        # determine width of new image which is the max distance between
        # (bot_r and bot_l) or (top_r and top_l)
        width_A = math.dist(bot_l,bot_r)
        width_B = math.dist(top_l,top_r)
        width = max(int(width_A), int(width_B))
        
        # determine height of new image which is the max distance between
        # (top_l and bot_l) or (top_r and bot_r)
        height_A = math.dist(top_l,bot_l)
        height_B = math.dist(top_r,bot_r)
        height = max(int(height_A), int(height_B))
        
        # construct new points to obtain top down view of image
        dimensions = np.array([[0,0],[width,0],[width,height],[0,height]],dtype='float32')
        
        # convert to numpy format
        ordered_corners = np.array(corners, dtype='float32')
        
        # find perpective transform matrix
        matrix = cv2.getPerspectiveTransform(ordered_corners, dimensions)
        
        result = cv2.warpPerspective(image,matrix,(width,height))
        
        #return the transformed image
        cv2.imwrite('transformedperspective.png',result)
        return result

    def detect_color(self, img, x, y):
        '''
            Identify the average color of a neighborhood of pixels
        '''
        kernel = img[x-10:x+10,y-10:y+10]   # dimensions of the area we will check
        avg_color = np.mean(kernel,axis=(0,1))
        return avg_color
    
    def identify_intersections(self, img):
        '''
            calculate the locations of each intersection on the go board and 
            store the color value at that intersection
            
            img - image of a Go board cropped to only the playable surface at a top down view
        '''
        HEIGHT = img.shape[0]
        WIDTH = img.shape[1]
        W_INC = WIDTH / (self.size-1)   # amount of px between each intersection
        H_INC = HEIGHT / (self.size-1)
        
        colors = []
        
        x_pointer = 0
        y_pointer = 0
        
        for y in range(self.size):
            y_pointer_approx = int(y_pointer)
            x_pointer = 0
            for x in range(self.size):
                x_pointer_approx = int(x_pointer)
                if x_pointer_approx == 0:
                    x_pointer_approx += 11
                elif x_pointer_approx >= WIDTH-10:
                    x_pointer_approx -= 11
                
                if y_pointer_approx == 0:
                    y_pointer_approx += 11
                elif y_pointer_approx >= HEIGHT-10:
                    y_pointer_approx -= 11
                color = self.detect_color(img,y_pointer_approx,x_pointer_approx)
                colors.append((x,y,color))
                x_pointer += W_INC
            y_pointer += H_INC
        cv2.imwrite('average_colors.png',img)
        return colors
        
    def color_kmeans(self,colors):
        '''
            Given a list of average color values at each intersection,
            perform the KMeans algorithm to define clusters to sort the intersections
            by color
            
            colors - iterable of BGR values each of which represent the average color
                    at an intersection in a 20x20 range
        '''
        isolated_colors = []    #array of only BGR values
        for i in colors:
            isolated_colors.append(i[2])
        km = KMeans(
            n_clusters=3, init='random',
            n_init=10, max_iter=300,
            tol=1e-04, random_state=0
        )
        y_km = km.fit_predict(isolated_colors)
        cluster1,cluster2,cluster3 = km.cluster_centers_
        # calculate the sum of the BGR values of each cluster center
        cluster1 = np.sum(cluster1)
        cluster2 = np.sum(cluster2)
        cluster3 = np.sum(cluster3)
        # the minimum sum BGR value will represent the color closest to Black (0,0,0)
        min_cluster = np.argmin([cluster1,cluster2,cluster3])
        # the maximum sum BGR valur will represent the color closest to white (255,255,255)
        max_cluster = np.argmax([cluster1,cluster2,cluster3])
        result = []
        # combine colors array and kmeans result into 1 array in a
        # format that can be used by TenukiJS
        for a,b in zip(colors,y_km):
            if b == min_cluster:
                result.append([a[0],a[1],'black'])
            elif b == max_cluster:
                result.append([a[0],a[1],'white'])
        return result
        
    def write_string_to_file(self, file_path, content):
        try:
            with open(file_path, 'w') as file:
                file.write(content)
            print("String written to file successfully.")
        except IOError:
            print("An error occurred while writing to the file.")
        
    def full_analysis(self):
        eroded_image = self.canny_processing(self.image_path)
        hough_transfer = self.hough_lines_corner_detection(eroded_image)
        colors = self.identify_intersections(self.perspective_transform(self.image_path,hough_transfer))
        data = self.color_kmeans(colors)
        return data
    
os.system('node ./index.js')
