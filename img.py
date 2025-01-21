import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import keras
import sys
import sudoku

file = sys.argv[1]
img = cv.imread(file, cv.IMREAD_GRAYSCALE)
img2 = cv.imread(file)
# _, gray = cv.threshold(img,140,255,cv.THRESH_BINARY_INV)
gray = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 91, 30)

plt.imshow(gray, cmap="gray")
plt.show()


contours, _ = cv.findContours(gray, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
m = max(contours, key=cv.contourArea)
i = img2.copy()
cv.drawContours(i, [m], -1, (255, 0, 0), 20)
plt.imshow(i)
plt.show()

marea = cv.contourArea(m)
start_x, start_y, height, width = cv.boundingRect(m)
start_y = int(start_y * 0.8)
start_x = int(start_x*0.8)
height = int(height*1.3)
width = int(width*1.3)
gray = gray[start_y:start_y+height, start_x:start_x+width]
img2 = img2[start_y:start_y+height, start_x:start_x+width]

gray_nums = gray.copy()

contours, _ = cv.findContours(gray, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
digits = []
for c in contours:
    a = cv.contourArea(c)
    b = cv.boundingRect(c)
    # i = img2.copy()
    # cv.drawContours(i, [c], -1, (255, 0,0), -1)
    # plt.imshow(i)
    # plt.show()

    if a < (marea/81) * 0.25:
        if b[2]*b[3] > (marea/81) * 0.1:# - (marea/81)/30:
            digits.append(c)
    #     else:
    #         print("area too small")
    # else:
    #     print("area too big")

# cv.drawContours(img2, digits, -1, (255,0,0), -1)
# plt.imshow(img2)

gray = cv.drawContours(gray, digits, -1, (0,0,0), -1)

vertical_kernel = cv.getStructuringElement(cv.MORPH_RECT, (1,3))
gray = cv.morphologyEx(gray, cv.MORPH_CLOSE, vertical_kernel, iterations=9)
horizontal_kernel = cv.getStructuringElement(cv.MORPH_RECT, (3,1))
gray = cv.morphologyEx(gray, cv.MORPH_CLOSE, horizontal_kernel, iterations=9)

gray = 255 - gray
plt.imshow(gray, cmap="gray")
plt.show()

contours, _ = cv.findContours(gray, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
squares = []
for c in contours:
    # i = img2.copy()
    # cv.drawContours(i, [c], -1, (255,0,0), 10)
    # plt.imshow(i)
    # plt.show()

    if cv.contourArea(c) < marea/81:
        if cv.contourArea(c) > (marea/81) * 0.5:
            # print('good')
            squares.append(c)
    #     else: print("area too small")
    # else:
    #     print("area too big")

i = img2.copy()
cv.drawContours(i, squares, -1, (255, 0, 0), 5)
plt.imshow(i)
plt.show()

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

# cv.drawContours(img2, squares, -1, (255, 0, 0), 5)
# plt.imshow(img2)
# plt.show()

def contains(big: cv.typing.Rect, small: cv.typing.Rect) -> bool:
    return (
        big[0] <= small[0] and
        big[1] <= small[1] and
        big[0]+big[3] >= small[0]+small[3] and
        big[1]+big[2] >= small[1]+small[2]
    )

# i = img2.copy()
# cv.drawContours(i, digits, -1, (255, 0, 0), -1)
# plt.imshow(i)
# plt.show()

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

digits, digit_bounds = gd, gdb

# i = img2.copy()
# cv.drawContours(i, digits, -1, (255, 0, 0), -1)
# cv.drawContours(i, squares, -1, (0, 255, 0), 10)
# plt.imshow(i)
# plt.show()


game = [[0 for _ in range(9)] for _ in range(9)]

keras.config.disable_interactive_logging()
model = keras.models.load_model("digitrecognizer.keras")

digit_idx = 0
game_idx = 0
for i, b in enumerate(sq_bounds):
    # print(f"square: {b[2]*b[3]}")
    # print(f"digit: {digit_bounds[digit_idx][2]*digit_bounds[digit_idx][3]}")
    # c = img2.copy()
    # cv.rectangle(c, (b[0], b[1]), (b[0]+b[3], b[1]+b[2]), (255, 0, 0))
    # cv.rectangle(c, (digit_bounds[digit_idx][0], digit_bounds[digit_idx][1]), (digit_bounds[digit_idx][0]+digit_bounds[digit_idx][3], digit_bounds[digit_idx][1]+digit_bounds[digit_idx][2]), (255, 0, 0))
    # cv.drawContours(c, digits, digit_idx, (0, 255, 0), -1)
    # plt.imshow(c)
    # plt.show()
    if digit_idx == len(digits):
        break
    if not contains(b, digit_bounds[digit_idx]):
        # print("empty")
        game_idx+=1
        continue
    
    num = gray_nums[b[1]:b[1]+b[2], b[0]:b[0]+b[3]]
    num = cv.dilate(num, np.ones((3, 3)), iterations=1)
    num = cv.erode(num, np.ones((3, 3)), iterations=1)
    num = cv.resize(num, (28, 28))
    # num = keras.utils.normalize(num, axis=1)

    pred = model.predict(np.array([num]), verbose=0)
    n = pred.argmax()
    conf = pred.max()
    # print(f"{n}: {conf}")
    digit_idx += 1
    row = game_idx // 9
    col = game_idx - 9 * row
    game[row][col] = int(n)
    game_idx+=1
for r in game:
    print(r)

print()
print("solving")
print()

sudoku.solve(game)
for r in game:
    if 0 in r:
        print('bad sudoku, unsolvable')
        exit()

for r in game:
    print(r)


# six = gray_nums[sq_bounds[6][1]:sq_bounds[6][1]+sq_bounds[6][2], sq_bounds[6][0]:sq_bounds[6][0]+sq_bounds[6][3]]
# six = cv.dilate(six, np.ones((3, 3)), iterations=1)
# six = cv.erode(six, np.ones((3, 3)), iterations=1)
# six = keras.utils.normalize(six, axis=1)
#
# sm = cv.resize(six, (28, 28))
#
# p = model.predict(np.array([sm]))
# print(p)
# print(p.argmax())
# plt.imshow(six, cmap="gray")
# plt.show()
