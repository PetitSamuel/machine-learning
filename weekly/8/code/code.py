from PIL import Image
import numpy as np


def convolutional_layer(image, kernel):
    x_positions = len(image) - len(kernel) + 1
    y_positions = len(image[0]) - len(kernel[0]) + 1
    assert x_positions > 0 and y_positions > 0, "image should not be smaller than kernel"
    result = []
    for i in range(x_positions):
        row = []
        for j in range(y_positions):
            sum = 0
            for k in range(len(kernel)):
                for l in range(len(kernel[0])):
                    sum = sum + (kernel[k][l] * image[i + k][j + l])
            row.append(sum)
        result.append(row)
    return result


im = Image.open('triangle.PNG')
rgb = np.array(im.convert('RGB'))
r = rgb[:, :, 0]

kernel1 = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
kernel2 = np.array([[0, -1, 0], [-1, 8, -1], [0, -1, 0]])

im_k1 = convolutional_layer(r, kernel1)
im_k2 = convolutional_layer(r, kernel2)

Image.fromarray(np.uint8(im_k1)).show()
Image.fromarray(np.uint8(im_k2)).show()
