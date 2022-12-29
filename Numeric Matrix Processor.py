# Write your code here

class MatrixProcessor:
    def __init__(self):
        self.print_matrix = self.print_matrix
        self.matrix_A = ...
        self.matrix_B = ...
        self.size_A = ...
        self.size_B = ...

    def insert_A(self):
        size_A = tuple([int(x) for x in input('Enter size of first matrix').split(' ')])
        print('Enter first matrix:')
        matrix_A = [list(map(lambda x: int(x) if x.isdigit() else float(x), input().split(' '))) for row in
                    range(size_A[0])]
        self.size_A = size_A
        self.matrix_A = matrix_A

    def insert_B(self):
        size_B = tuple([int(x) for x in input('Enter size of second matrix').split(' ')])
        print('Enter second matrix:')
        matrix_B = [list(map(lambda x: int(x) if x.isdigit() else float(x), input().split(' '))) for row in
                    range(size_B[0])]
        self.size_B = size_B
        self.matrix_B = matrix_B

    def sum_two_matrix(self):
        if self.size_A == self.size_B:
            sum = [[self.matrix_A[i][j] + self.matrix_B[i][j] for j in range(self.size_A[1])] for i in range(self.size_A[0])]
            self.print_matrix(sum)
        else:
            print('The operation cannot be performed.')

    def multi_by_num(self, c):
        multi_by_number = [list(map(lambda x: x * c, self.matrix_A[i])) for i in range(self.size_A[0])]
        self.print_matrix(multi_by_number)

    def multi_two_matrix(self):
        if self.size_A[1] == self.size_B[0]:
            multi_matrix = [[sum(a*b for a,b in zip(X_row,Y_col)) for Y_col in zip(*self.matrix_B)] for X_row in self.matrix_A]
            self.print_matrix(multi_matrix)
        else:
            print('The operation cannot be performed.')

    def transpose_square_matrix(self, choice):
        if choice == 1:
            self.print_matrix(map(list, zip(*self.matrix_A)))
        elif choice == 2:
            self.print_matrix(list(map(list, zip(*self.matrix_A[::-1])))[::-1])
        elif choice == 3:
            self.print_matrix([x[::-1] for x in self.matrix_A])
        elif choice == 4:
            self.print_matrix(self.matrix_A[::-1])

    def calc_det(self, m, s):
        if s == (1,1):
            return m[0][0]
        if s == (2, 2):
            return m[0][0] * m[1][1] - m[0][1] * m[1][0]

        det = 0
        for index, x in enumerate(m[0]):
            sub_matrix = [[*arr[0:index],*arr[index+1:]] for arr in m[1:]]
            sub_matrix_shape = (s[0]-1, s[1]-1)
            subdet = self.calc_det(sub_matrix, sub_matrix_shape)
            if index % 2 == 0:
                det += x * subdet
            else:
                det -= x * subdet
        return det

    def print_matrix(self, m):
        print('The result is:')
        for row in m:
            row = [str(x) for x in row]
            print(' '.join(row))


processor = MatrixProcessor()

user_choice = 1
while user_choice != 0:
    print('''1. Add matrices
2. Multiply matrix by a constant
3. Multiply matrices
4. Transpose matrix
5. Calculate a determinant
0. Exit''')
    user_choice = int(input('Your choice:'))
    if user_choice == 1:
        processor.insert_A()
        processor.insert_B()
        processor.sum_two_matrix()
    elif user_choice == 2:
        processor.insert_A()
        constant = int(input('Enter constant:'))
        processor.multi_by_num(constant)
    elif user_choice == 3:
        processor.insert_A()
        processor.insert_B()
        processor.multi_two_matrix()
    elif user_choice == 4:
        print('''1. Main diagonal
2. Side diagonal
3. Vertical line
4. Horizontal line''')
        user_choice_transpose = int(input('Your choice:'))
        processor.insert_A()
        processor.transpose_square_matrix(user_choice_transpose)
    elif user_choice == 5:
        processor.insert_A()
        det = processor.calc_det(processor.matrix_A, processor.size_A)
        print(f'The result is:\n{det}')
    elif user_choice == 0:
        exit()
