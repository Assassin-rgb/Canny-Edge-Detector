# load packages
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# load the image
I = Image.open('Image3.jpeg')


# One-D convolution operation
def convolve1d(image, kernel, axis):
    if axis == 0:
        out = np.zeros(image.shape)
        # row-wise convolution
        for r in range(image.shape[0]):
            out[r, :] = np.convolve(image[r, :], kernel, 'same')
        return out
    elif axis == 1:
        out = np.zeros(image.shape)
        # column-wise convolution
        for c in range(image.shape[1]):
            out[:, c] = np.convolve(image[:, c], kernel, 'same')
        return out


# create 1D gaussian filter
def gaussian_filter(k_size, sigma):
    x = np.linspace(-(k_size // 2), k_size // 2, k_size)
    kernel = (1 / (np.sqrt(2 * np.pi) * sigma ** 2)) * (np.exp(-((x ** 2) / (2 * sigma ** 2))))
    return kernel


# non-max suppression
def non_max_suppression(image, i_theta):
    out = np.zeros(image.shape)
    # convert angle from radians to degrees
    i_angle = i_theta * 180 / np.pi
    i_angle[i_angle < 0] += 180
    p_1 = 255
    p_2 = 255

    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):
            # check 0 degrees relative to the angle
            if (0 <= i_angle[i, j] <= 22.5) or (157.5 <= i_angle[i, j] <= 180):
                p_1 = image[i, j + 1]
                p_2 = image[i, j - 1]
            # check 45 degrees relative to the angle
            elif 22.5 <= i_angle[i, j] < 67.5:
                p_1 = image[i + 1, j - 1]
                p_2 = image[i - 1, j + 1]
            # check 90 degrees relative to the angle
            elif 67.5 <= i_angle[i, j] < 112.5:
                p_1 = image[i + 1, j]
                p_2 = image[i - 1, j]
            # check 135 degrees relative to the angle
            elif 112.5 <= i_angle[i, j] < 157.5:
                p_1 = image[i - 1, j - 1]
                p_2 = image[i + 1, j + 1]
            # update non-max
            if (image[i, j] >= p_1) and (image[i, j] >= p_2):
                out[i, j] = image[i, j]
            else:
                out[i, j] = 0
    return out


# hysteresis threshold
def hysteresis_threshold(image, strong, weak, high_threshold, low_threshold):
    out = np.zeros(image.shape)

    # calculate the thresholds
    high = image.max() * high_threshold
    low = high * low_threshold

    # find the locations of weak and strong pixels
    strong_x, strong_y = np.where(image >= high)
    weak_x, weak_y = np.where((image <= high) & (image >= low))

    # update the hysteresis matrix with strong and weak values obtained
    out[strong_x, strong_y] = strong
    out[weak_x, weak_y] = weak

    # update the hysteresis matrix weak values
    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):
            if out[i, j] == weak:
                # check for strong neighbour
                if ((out[i + 1, j - 1] == strong)
                        or (out[i + 1, j] == strong)
                        or (out[i + 1, j + 1] == strong)
                        or (out[i, j - 1] == strong)
                        or (out[i, j + 1] == strong)
                        or (out[i - 1, j - 1] == strong)
                        or (out[i - 1, j] == strong)
                        or (out[i - 1, j + 1] == strong)):
                    out[i, j] = strong
                # turn off the pixel
                else:
                    out[i, j] = 0
    return out


# canny edge detector
def canny_detector(image, k_size, sigma):
    # convert image to numpy array
    data = np.asarray(image)
    plt.imshow(data, cmap='gray')
    plt.title('Original image')
    plt.show()
    # gaussian kernel
    kernel = gaussian_filter(k_size, sigma)
    # 1D derivative masks
    Gx = [-1, 0, 1]
    Gy = Gx
    # convolve along x and y axis
    Ix = convolve1d(data, kernel, 0)
    plt.imshow(Ix, cmap='gray')
    plt.title('Gaussian kernel convoluted across x')
    plt.show()
    Iy = convolve1d(data, kernel, 1)
    plt.imshow(Iy, cmap='gray')
    plt.title('Gaussian kernel convoluted across y')
    plt.show()
    # convolve Ix and Iy with Gx and Gy
    Ix_ = convolve1d(Ix, Gx, 0)
    plt.imshow(Ix_, cmap='gray')
    plt.title('Derivative mask applied across x')
    plt.show()
    Iy_ = convolve1d(Iy, Gy, 1)
    plt.imshow(Iy_, cmap='gray')
    plt.title('Derivative mask applied across y')
    plt.show()
    # calculate magnitude matrix
    I_magnitude = np.hypot(Ix_, Iy_)
    I_magnitude = (I_magnitude / np.max(I_magnitude)) * 255
    plt.imshow(I_magnitude, cmap='gray')
    plt.title('Magnitude image')
    plt.show()
    # calculate angle matrix
    I_theta = np.arctan2(Iy_, Ix_)
    # apply non-max suppression
    I_non_max = non_max_suppression(I_magnitude, I_theta)
    plt.imshow(I_non_max, cmap='gray')
    plt.title('Non Max Suppression')
    plt.show()
    # apply hysteresis threshold
    I_hst = hysteresis_threshold(I_non_max, 255, 50, 0.3, 0.05)
    plt.imshow(I_hst, cmap='gray')
    plt.title('Hysteresis threshold')
    plt.show()


# execute file
if __name__ == '__main__':
    canny_detector(I, 5, 2)
