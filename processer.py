import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import keras
import sudoku

def process(filepath: str):
    img = cv.imread(filepath, cv.IMREAD_GRAYSCALE)
    gray = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 91, 30)
    img = cv.imread(filepath)
       
    contours, _ = cv.findContours(gray, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    m = max(contours, key=cv.contourArea)
    
    marea = cv.contourArea(m)
    start_x, start_y, height, width = cv.boundingRect(m)
    start_y = int(start_y * 0.8)
    start_x = int(start_x*0.8)
    height = int(height*1.3)
    width = int(width*1.3)
    gray = gray[start_y:start_y+height, start_x:start_x+width]
    img = img[start_y:start_y+height, start_x:start_x+width, :]

    gray_with_nums = gray.copy()
    
    contours, _ = cv.findContours(gray, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    digits = []
    for c in contours:
        a = cv.contourArea(c)
        b = cv.boundingRect(c)
        if a < (marea/81) * 0.25:
            if b[2]*b[3] > (marea/81) * 0.1:# - (marea/81)/30:
                digits.append(c)
    
    gray = cv.drawContours(gray, digits, -1, (0,0,0), -1)
    
    vertical_kernel = cv.getStructuringElement(cv.MORPH_RECT, (1,3))
    gray = cv.morphologyEx(gray, cv.MORPH_CLOSE, vertical_kernel, iterations=12)
    horizontal_kernel = cv.getStructuringElement(cv.MORPH_RECT, (3,1))
    gray = cv.morphologyEx(gray, cv.MORPH_CLOSE, horizontal_kernel, iterations=12)
    
    gray = 255 - gray
    
    contours, _ = cv.findContours(gray, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    squares = []
    for c in contours:
        if cv.contourArea(c) < marea/81:
            if cv.contourArea(c) > (marea/81) * 0.5:
                # print('good')
                squares.append(c)
    
    squares.sort(key = lambda x:cv.contourArea(x), reverse=True)
    squares = squares[:81]
    
    def sort_contours(cnts):
        def custom_comparator(a, b):
            if abs(a[1][1] - b[1][1]) <= 20: 
                return a[1][0] - b[1][0]
            return a[1][1] - b[1][1] 
        
        from functools import cmp_to_key
    
        a = zip(cnts, [cv.boundingRect(c) for c in cnts])
        (cnts, bounds) = zip(*sorted(a, key=cmp_to_key(custom_comparator)))
        return cnts, bounds
    
    squares, sq_bounds = sort_contours(squares)
    digits, digit_bounds = sort_contours(digits)

    def contains(big: cv.typing.Rect, small: cv.typing.Rect) -> bool:
        return (
            big[0] <= small[0] and
            big[1] <= small[1] and
            big[0]+big[3] >= small[0]+small[3] and
            big[1]+big[2] >= small[1]+small[2]
        )
    
    gd, gdb = [], []
    for i, b in enumerate(digit_bounds):
        found = False
        for bs in sq_bounds:
            if contains(bs, b):
                found = True
                break
    
        if not found:
            continue
        gdb.append(b)
        gd.append(digits[i])
    
    digits, digit_bounds = gd, gdb # removes digits outside of the squares
    
    game = [[0 for _ in range(9)] for _ in range(9)]
    
    keras.config.disable_interactive_logging()
    model = keras.models.load_model("digitrecognizer.keras")
    
    digit_idx = 0
    game_idx = 0
    for i, b in enumerate(sq_bounds):
        if digit_idx == len(digits):
            break
        if not contains(b, digit_bounds[digit_idx]):
            # print("empty")
            game_idx+=1
            continue
        
        num = gray_with_nums[b[1]:b[1]+b[2], b[0]:b[0]+b[3]]
        num = cv.dilate(num, np.ones((3, 3)), iterations=1)
        num = cv.erode(num, np.ones((3, 3)), iterations=1)
        num = cv.resize(num, (28, 28))
    
        pred = model.predict(np.array([num]), verbose=0)
        n = pred.argmax()
        conf = pred.max()
        digit_idx += 1
        row = game_idx // 9
        col = game_idx - 9 * row
        game[row][col] = int(n)
        game_idx+=1
    sudoku.solve(game)
    return img, game, sq_bounds

def overlay_transparent(base_image, transparent_image, x_offset, y_offset):
    _, _, _, alpha = cv.split(transparent_image)

    alpha = alpha / 255.0

    h, w = transparent_image.shape[:2]
    y1, y2 = y_offset, y_offset + h
    x1, x2 = x_offset, x_offset + w

    if y1 < 0 or y2 > base_image.shape[0] or x1 < 0 or x2 > base_image.shape[1]:
        raise ValueError("The overlay position and size are out of bounds of the base image!")

    roi = base_image[y1:y2, x1:x2]

    for c in range(3):  # Loop over the color channels
        roi[:, :, c] = (alpha * transparent_image[:, :, c] + (1 - alpha) * roi[:, :, c])

    base_image[y1:y2, x1:x2] = roi

    return base_image

def sudoku_to_img(img, board, sq_bounds):
    sq_idx = 0
    for r in board:
        for c in r:
            ov = cv.imread(f"nums/{c}.png", cv.IMREAD_UNCHANGED)
            ov = cv.resize(ov, (sq_bounds[sq_idx][3], sq_bounds[sq_idx][2]))
            img = overlay_transparent(img, ov, sq_bounds[sq_idx][0], sq_bounds[sq_idx][1])
            sq_idx+=1
    return img
    
