from PIL import Image
import numpy as np
from scipy.fftpack import dct

frame1 = np.array(Image.open('Picture1.jpg').convert('L'))
frame2 = np.array(Image.open('Picture2.jpg').convert('L'))

block_size = 16

num_areas = 3 # Number of areas to process

for i in range(num_areas):
    # selecting random block from frame2
    rand_x = np.random.randint(0, frame2.shape[1] - block_size)
    rand_y = np.random.randint(0, frame2.shape[0] - block_size)
    selected_block = frame2[rand_y:rand_y+block_size, rand_x:rand_x+block_size]

    # Motion Estimation with sad
    min_sad = float('inf')
    best_match = None
    for y in range(frame1.shape[0] - block_size):
        for x in range(frame1.shape[1] - block_size):
            target_block = frame1[y:y+block_size, x:x+block_size]
            sad = np.sum(np.abs(target_block - selected_block))
            if sad < min_sad:
                min_sad = sad
                best_match = (x, y)

    # calculating difference
    x, y = best_match
    difference = selected_block - frame1[y:y+block_size, x:x+block_size]

    # DCT and Round coefficients for difference
    diff_dct = np.round(dct(difference, type=2, norm='ortho'))

    # DCT and Round coefficients for selected block
    selected_block_dct = np.round(dct(selected_block, type=2, norm='ortho'))

    # Compare coefficients
    coefficients_equal = np.all(diff_dct == selected_block_dct)

    print(f"Area {i+1}: Coefficients equal: {coefficients_equal}")