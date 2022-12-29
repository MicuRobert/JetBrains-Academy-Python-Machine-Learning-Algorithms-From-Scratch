# Write your code here

size_A = tuple([int(x) for x in input().split(' ')])
matrix_A = [list(map(int, input().split(' '))) for row in range(size_A[0])]
constant = int(input())
# size_B = tuple([int(x) for x in input().split(' ')])
# matrix_B = [list(map(int, input().split(' '))) for row in range(size_B[0])]


class MatrixProcessor:
    def __init__(self):
        self.print_matrix = self.print_matrix
        pass

    def sum_two_matrix(self, m1, m2, s1, s2):
        if s1 == s2:
            sum = [[m1[i][j] + m2[i][j] for j in range(s1[1])] for i in range(s1[0])]
            self.print_matrix(sum)
        else:
            print('ERROR')

    def multi_by_num(self, m1, c, s):
        multi_by_number = [list(map(lambda x: x * c, m1[i])) for i in range(s[0])]
        self.print_matrix(multi_by_number)

    def print_matrix(self, m):
        for row in m:
            row = [str(x) for x in row]
            print(' '.join(row))


processor = MatrixProcessor()
processor.multi_by_num(matrix_A, constant, size_A)
