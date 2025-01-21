def update_line(i, j, val, poss):
    for j2 in range(9):
        if j2 == j:
            continue
        poss[i][j2][val - 1] = 0

def update_col(i, j, val, poss):
    for i2 in range(9):
        if i2 == i:
            continue
        poss[i2][j][val - 1] = 0

def update_square(i, j, val, poss):
    sq_row = (i // 3) * 3
    sq_col = (j // 3) * 3
    for si in range(sq_row, sq_row + 3):
        for sj in range(sq_col, sq_col + 3):
            if i == si and j == sj:
                continue
            poss[si][sj][val - 1] = 0

def check_possible(i, j, poss):
    found = 0
    for k in range(9):
        if poss[i][j][k] == 1:
            if found != 0:
                return 0  # Multiple possibilities
            found = k + 1
    return found

class EarlyBreak(Exception): pass
def check_square_gaps(sq_row, sq_col, poss, board):
    filled = 0
    for i in range(9):
        found_is, found_js = [], []
        try:
            for si in range(sq_row, sq_row + 3):
                for sj in range(sq_col, sq_col + 3):
                    if board[si][sj] == i + 1:
                        # Skip this number if already found in the square
                        raise EarlyBreak
                    if board[si][sj] == 0 and poss[si][sj][i] == 1:
                        # if found_is.__len__() != 0:
                        #     # Multiple possibilities, skip this number
                        #     raise EarlyBreak
                        found_is.append(si)
                        found_js.append(sj)
        except EarlyBreak:
            continue
        if found_is.__len__() != 0:
            if found_is.__len__() == 1:
                found_i = found_is[0]
                found_j = found_js[0]
                board[found_i][found_j] = i + 1
                filled += 1
                update_line(found_i, found_j, i + 1, poss)
                update_col(found_i, found_j, i + 1, poss)
                update_square(found_i, found_j, i + 1, poss)
            if all([ii == found_is[0] for ii in found_is]):
                for j in range(9):
                    if j in found_js:continue
                    poss[found_is[0]][j][i] = 0
            elif all([jj == found_js[0] for jj in found_js]):
                for ii in range(9):
                    if ii in found_is:continue
                    poss[ii][found_js[0]][i] = 0
    return filled

def solve(board):
    to_fill = 81

    poss = [[[1] * 9 for _ in range(9)] for _ in range(9)]
    for i in range(9):
        for j in range(9):
            if board[i][j] != 0:
                to_fill -= 1
                poss[i][j] = [0] * 9
                poss[i][j][board[i][j] - 1] = 1

    # Initial update of rows, columns, and squares
    for i in range(9):
        for j in range(9):
            if board[i][j] != 0:
                update_line(i, j, board[i][j], poss)
                update_col(i, j, board[i][j], poss)
                update_square(i, j, board[i][j], poss)

    count = 0
    while to_fill > 0:
        last_to_fill = to_fill
        # Try to fill in cells where only one possibility exists
        for i in range(9):
            for j in range(9):
                if board[i][j] != 0:
                    continue
                v = check_possible(i, j, poss)
                if v != 0:
                    to_fill -= 1
                    board[i][j] = v
                    update_line(i, j, v, poss)
                    update_col(i, j, v, poss)
                    update_square(i, j, v, poss)

        # Check for gaps in each square and try to fill them
        for sq_row in range(0, 9, 3):
            for sq_col in range(0, 9, 3):
                filled = check_square_gaps(sq_row, sq_col, poss, board)
                to_fill -= filled

        if last_to_fill == to_fill:
            break
        count += 1
