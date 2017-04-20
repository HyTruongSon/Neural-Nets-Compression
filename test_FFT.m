function [] = test_FFT()
    nRows = 784;
    nCols = 256;
    A = rand(nRows, nCols);
    F_A = fft(reshape(A, nRows * nCols, 1));
    for i = nRows * nCols : -1 : 1
        f_A = F_A;
        f_A(i + 1 : end) = 0.0;
        A_ = reshape(ifft(f_A), nRows, nCols);
        average_error = sum(sum(abs(A - A_))) / (nRows * nCols);
        fprintf('Frequency %d: Average norm-1 error = %.6f\n', i, average_error);
        if average_error > 0.1
            break;
        end
    end
end