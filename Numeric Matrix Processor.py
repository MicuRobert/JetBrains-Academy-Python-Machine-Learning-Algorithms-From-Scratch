# Write your code here

class MatrixProcessor:
    def __init__(self):
        self.print_matrix = self.print_matrix
        pass

    def sum_two_matrix(self, m1, m2, s1, s2):
        if s1 == s2:
            sum = [[m1[i][j] + m2[i][j] for j in range(s1[1])] for i in range(s1[0])]
            self.print_matrix(sum)
        else:
            print('The operation cannot be performed.')

    def multi_by_num(self, m1, c, s):
        multi_by_number = [list(map(lambda x: x * c, m1[i])) for i in range(s[0])]
        self.print_matrix(multi_by_number)

    def multi_two_matrix(self, m1, m2, s1, s2):
        if s1[1] == s2[0]:
            multi_matrix = [[sum(a*b for a,b in zip(X_row,Y_col)) for Y_col in zip(*m2)] for X_row in m1]
            self.print_matrix(multi_matrix)
        else:
            print('The operation cannot be performed.')

    def print_matrix(self, m):
        print('The result is:')
        for row in m:
            row = [str(x) for x in row]
            print(' '.join(row))


processor = MatrixProcessor()

user_choice = 1
while user_choice != 0:
    print('''
    1. Add matrices
    2. Multiply matrix by a constant
    3. Multiply matrices
    0. Exit''')
    user_choice = int(input('Your choice:'))
    if user_choice == 1:
        size_A = tuple([int(x) for x in input('Enter size of first matrix').split(' ')])
        print('Enter first matrix:')
        matrix_A = [list(map(float, input().split(' '))) for row in range(size_A[0])]
        size_B = tuple([int(x) for x in input('Enter size of second matrix').split(' ')])
        print('Enter second matrix:')
        matrix_B = [list(map(float, input().split(' '))) for row in range(size_B[0])]
        processor.sum_two_matrix(matrix_A, matrix_B, size_A, size_B)
    elif user_choice == 2:
        size_A = tuple([int(x) for x in input('Enter size of matrix').split(' ')])
        print('Enter matrix:')
        matrix_A = [list(map(float, input().split(' '))) for row in range(size_A[0])]
        constant = int(input('Enter constant:'))
        processor.multi_by_num(matrix_A, constant, size_A)
    elif user_choice == 3:
        size_A = tuple([int(x) for x in input('Enter size of first matrix').split(' ')])
        print('Enter first matrix:')
        matrix_A = [list(map(float, input().split(' '))) for row in range(size_A[0])]
        size_B = tuple([int(x) for x in input('Enter size of second matrix').split(' ')])
        print('Enter second matrix:')
        matrix_B = [list(map(float, input().split(' '))) for row in range(size_B[0])]
        processor.multi_two_matrix(matrix_A, matrix_B, size_A, size_B)
    elif user_choice == 0:
        exit()
