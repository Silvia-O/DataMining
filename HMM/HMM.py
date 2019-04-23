import numpy as np

class HMM:
    def __init__(self, A, B, Pi, O):
        # hidden state transition probability matrix
        self.A = np.array(A, np.float)
        # observation probability matrix
        self.B = np.array(B, np.float)
        # initial hidden state probability vector
        self.Pi = np.array(Pi, np.float)
        # observed states sequence
        self.O = np.array(O, np.int)
        # number of hidden states
        self.n = self.B.shape[0]
        # number of observed states
        self.m = self.B.shape[1]

    # input: A, B, Pi, O
    # output: probability of the observed states sequence
    def forward(self):
        print("forward...")

        # length of observed states sequence
        T = len(self.O)
        alpha = np.zeros((T, self.n), np.float)

        # 1. initialize
        for i in range(self.n):
            alpha[0, i] = self.Pi[i] * self.B[i, self.O[0]]

        # 2. recurse
        for t in range(1, T):
            for i in range(self.n):
                sum = 0.0
                for j in range(self.n):
                    sum += alpha[t-1, j] * self.A[j, i]
                alpha[t, i] = sum * self.B[i, self.O[t]]

        # 3. terminal
        sum = 0.0
        for i in range(self.n):
            sum += alpha[T-1, i]
        Polambda = sum

        print("Polambda is", Polambda)
        print("alpha matrix is")
        print(alpha)
        # return alpha, Polambda

    # input: A, B, Pi, O
    # output: hidden state sequence when P(O|Lambda) gets max
    def viterbi(self):
        print("viterbi...")

        # length of observed states sequence
        T = len(self.O)
        # hidden state sequence to output
        I = np.zeros(T, np.int)
        #
        delta = np.zeros((T, self.n), np.float)
        #
        psi = np.zeros((T, self.n), np.float)

        # 1. initialize
        for i in range(self.n):
            delta[0, i] = self.Pi[i] * self.B[i, self.O[0]]
            psi[0, i] = 0

        # 2. recurse
        for t in range(1, T):
            for i in range(self.n):
                n_max = 0
                n_argmax = -1
                for j in range(self.n):
                    tmp = delta[t-1, j] * self.A[j, i]
                    if tmp >= n_max:
                        n_max = tmp
                        n_argmax = j
                delta[t, i] = self.B[i, self.O[t]] * n_max
                psi[t, i] = n_argmax

        # 3. end
        P = delta[T-1, :].max()
        I[T-1] = delta[T-1, :].argmax()

        # 4. trace back
        for t in range(T-2, -1, -1):
            I[t] = psi[t+1, I[t+1]]

        print("I is", I)
        # return I

if __name__=="__main__":
    # hidden states: sunny, cloudy, rainy
    # observed states: hot, cold, damp
    A = [[0.8, 0.1, 0.1],
         [0.3, 0.4, 0.3],
         [0.4, 0.2, 0.4]]
    B = [[0.8, 0.1, 0.1],
         [0.2, 0.5, 0.3],
         [0.1, 0.2, 0.7]]
    Pi = [30.0/47.0, 9.0/47.0, 8.0/47.0]
    O = [0, 0, 1, 2, 1, 2, 1, 0]
    hmm = HMM(A, B, Pi, O)
    hmm.forward()
    hmm.viterbi()





