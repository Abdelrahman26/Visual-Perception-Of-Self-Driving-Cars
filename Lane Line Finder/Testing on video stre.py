def process_image(image):
    # gray scale 
    img2 = image
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
    #
    rst = weighted_img(lines2, img2)
    rst = cv2.cvtColor(rst, cv2.COLOR_BGR2RGB)
    return rst

white_output = 'test_videos_output/solidWhiteRight.mp4'
clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4")
white_clip = clip1.fl_image(process_image)
%time white_clip.write_videofile(white_output, audio=False)
HTML("""
<video width="960" height="540" controls>
  <source src="{0}"  type="video/mp4">
</video>
""".format(white_output))
