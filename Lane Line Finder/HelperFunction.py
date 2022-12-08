# Helper Functions

def grayscale(img):
   return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def canny(img, low_threshold, high_threshold):
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
  mask = np.zeros_like(img)  
  if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
  else:
        ignore_mask_color = 255
  cv2.fillPoly(mask, [vertices], ignore_mask_color)
  masked_image = cv2.bitwise_and(img, mask)
  return masked_image

def draw_lines(img, lines, color=[0, 0, 255], thickness=2):
   cnt = 0
   for line in lines:
        cnt = cnt + 1
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)
            slope = (y1 - y2) / (x1 - x2)
            
def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
   lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
   line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
   draw_lines2(line_img, lines)
   return line_img

def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
   return cv2.addWeighted(initial_img, α, img, β, γ)

def display(img):
  plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
  img2 = cv2.imread("test_images/solidWhiteRight.jpg")
