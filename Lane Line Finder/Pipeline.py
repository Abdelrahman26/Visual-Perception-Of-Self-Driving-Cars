# gray scale 
img2 = cv2.imread("test_images/solidWhiteRight.jpg")
h , w, c = img2.shape
img_gray2 = grayscale(img2)
# Filtring 
img_filter2 = gaussian_blur(img_gray2, 9)
# canny 
img_edge2 = canny(img_filter2, 50, 200)
#Only keeps the region of the image defined by the polygon 
verticies = np.array([ [140,h] , [450,320] , [510, 320] , [w - 100, h] ] , np.uint)
masked_img2= region_of_interest(img_edge2,verticies)
# hough transform 
lines2 = hough_lines(masked_img2, 1, np.pi/180, 10 , 1 ,50)
rst    = weighted_img(lines2, img2)
# testing in one image
display(rst)
