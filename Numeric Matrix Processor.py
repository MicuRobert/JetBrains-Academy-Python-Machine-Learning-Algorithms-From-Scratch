# Write your code here

size_A = tuple([int(x) for x in input().split(' ')])
matrix_A = [list(map(int, input().split(' '))) for row in range(size_A[0])]
size_B = tuple([int(x) for x in input().split(' ')])
matrix_B = [list(map(int, input().split(' '))) for row in range(size_B[0])]

if size_A == size_B:
    sum = [[matrix_A[i][j] + matrix_B[i][j] for j in range(size_A[1])] for i in range(size_A[0])]
    for row in sum:
        row = [str(x) for x in row]
        print(' '.join(row))
else:
    print('ERROR')
