#include <torch/extension.h>
#include <cusparse_v2.h>
#include <cublas_v2.h>
#include <vector>

torch::Tensor cusparse_coo_multiply(
    torch::Tensor cooRowInd,
    torch::Tensor cooColInd,
    torch::Tensor cooVal,
    torch::Tensor x,
    int A_num_rows,
    int A_num_cols
   ) {
    AT_ASSERTM(cooVal.is_cuda(), "cooVal must be a CUDA tensor");
    AT_ASSERTM(cooRowInd.is_cuda(), "cooRowInd must be a CUDA tensor");
    AT_ASSERTM(cooColInd.is_cuda(), "cooColInd must be a CUDA tensor");
    AT_ASSERTM(x.is_cuda(), "x must be a CUDA tensor");
    auto options = cooVal.options();
    auto y = torch::zeros({A_num_rows}, options);

    cusparseHandle_t handle = NULL;
    cusparseSpMatDescr_t matA;
    cusparseDnVecDescr_t vecX, vecY;
    double alpha = 1.0, beta = 0.0;
    void* dBuffer = NULL;
    size_t bufferSize = 0;

    cusparseCreate(&handle);

    // Create sparse matrix A in COO format
    cusparseCreateCoo(&matA,
                      A_num_rows, A_num_cols, cooVal.numel(),
                      cooRowInd.data_ptr<int>(), cooColInd.data_ptr<int>(), cooVal.data_ptr<double>(),
                      CUSPARSE_INDEX_32I, // Index base
                      CUSPARSE_INDEX_BASE_ZERO, // Index base
                      CUDA_R_64F); // Data type

    // Create dense vector x and y
    cusparseCreateDnVec(&vecX, x.numel(), x.data_ptr<double>(), CUDA_R_64F);
    cusparseCreateDnVec(&vecY, y.numel(), y.data_ptr<double>(), CUDA_R_64F);

    // Compute buffer size and allocate buffer
    cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                            &alpha, matA, vecX, &beta, vecY, CUDA_R_64F,
                            CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize);
    cudaMalloc(&dBuffer, bufferSize);

    // Perform matrix-vector multiplication
    cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                 &alpha, matA, vecX, &beta, vecY, CUDA_R_64F,
                 CUSPARSE_SPMV_ALG_DEFAULT, dBuffer);

    // Cleanup
    cusparseDestroySpMat(matA);
    cusparseDestroyDnVec(vecX);
    cusparseDestroyDnVec(vecY);
    cudaFree(dBuffer);
    cusparseDestroy(handle);

    return y;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> cusparse_csr_ichol(
    torch::Tensor csrRowInd,
    torch::Tensor csrColInd,
    torch::Tensor csrVal,
    int A_num_rows,
    int A_num_cols,
    int nnz
   ) {
    AT_ASSERTM(csrVal.is_cuda(), "cooVal must be a CUDA tensor");
    AT_ASSERTM(csrRowInd.is_cuda(), "cooRowInd must be a CUDA tensor");
    AT_ASSERTM(csrColInd.is_cuda(), "cooColInd must be a CUDA tensor");
    auto csrVal1 = csrVal.clone();
    auto csrRowInd1 = csrRowInd.clone();
    auto csrColInd1 = csrColInd.clone();

    auto options = csrVal.options();
    int num_offsets = csrRowInd.numel();
    int m = A_num_rows;

    // Create cuSPARSE matrix descriptor for L
    cusparseFillMode_t   fill_lower    = CUSPARSE_FILL_MODE_LOWER;
    cusparseDiagType_t   diag_non_unit = CUSPARSE_DIAG_TYPE_NON_UNIT;

    cusparseIndexBase_t  baseIdx = CUSPARSE_INDEX_BASE_ZERO;
    cusparseSpMatDescr_t matL;
    cusparseCreateCsr(&matL, A_num_rows, A_num_cols, nnz, csrRowInd1.data_ptr<int>(),
                                      csrColInd1.data_ptr<int>(), csrVal1.data_ptr<double>(),
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      baseIdx, CUDA_R_64F);
    cusparseSpMatSetAttribute(matL,
                              CUSPARSE_SPMAT_FILL_MODE,
                              &fill_lower, sizeof(fill_lower));

    cusparseSpMatSetAttribute(matL,
                              CUSPARSE_SPMAT_DIAG_TYPE,
                              &diag_non_unit,
                              sizeof(diag_non_unit));
    cusparseHandle_t cusparseHandle = NULL;
    cusparseCreate(&cusparseHandle);

    // Incomplete Cholesky factorization
    cusparseMatDescr_t descrM;
    csric02Info_t      infoM        = NULL;
    int                bufferSizeIC = 0;
    void*              d_bufferIC;
    cusparseCreateMatDescr(&descrM);
    cusparseSetMatIndexBase(descrM, baseIdx);
    cusparseSetMatType(descrM, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatFillMode(descrM, CUSPARSE_FILL_MODE_LOWER);
    cusparseSetMatDiagType(descrM, CUSPARSE_DIAG_TYPE_NON_UNIT);
    //creates and initializes the solve and analysis structure of incomplete Cholesky to default values
    cusparseCreateCsric02Info(&infoM);
    //returns size of buffer used in computing the incomplete-Cholesky factorization
    cusparseDcsric02_bufferSize(
                        cusparseHandle, m, nnz, descrM, csrVal1.data_ptr<double>(),
                        csrRowInd1.data_ptr<int>(), csrColInd1.data_ptr<int>(), infoM, &bufferSizeIC);
    //performs the analysis phase of the incomplete-Cholesky factorization
    cudaMalloc(&d_bufferIC, bufferSizeIC);
    cusparseDcsric02_analysis(
                        cusparseHandle, m, nnz, descrM, csrVal1.data_ptr<double>(),
                        csrRowInd1.data_ptr<int>(), csrColInd1.data_ptr<int>(), infoM,
                        CUSPARSE_SOLVE_POLICY_NO_LEVEL, d_bufferIC);
    int structural_zero;
    //know where the structural zero is.
    cusparseXcsric02_zeroPivot(cusparseHandle, infoM,
                                               &structural_zero);
    cusparseDcsric02(
                        cusparseHandle, m, nnz, descrM, csrVal1.data_ptr<double>(),
                        csrRowInd1.data_ptr<int>(), csrColInd1.data_ptr<int>(), infoM,
                        CUSPARSE_SOLVE_POLICY_NO_LEVEL, d_bufferIC);
    // Find numerical zero
    int numerical_zero;
    cusparseXcsric02_zeroPivot(cusparseHandle, infoM,
                                               &numerical_zero);

    cusparseDestroyCsric02Info(infoM);
    cusparseDestroyMatDescr(descrM);
    cudaFree(d_bufferIC);

    cusparseDestroy(cusparseHandle);

    auto result = std::make_tuple(csrRowInd1, csrColInd1, csrVal1);
    return result;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("coo_multiply", &cusparse_coo_multiply, "Sparse matrix-vector multiplication (COO format) using cuSparseSpMV");
  m.def("csr_ichol", &cusparse_csr_ichol, "Incomplete Cholesky factorization (CSR format) using cuSparse");
}
