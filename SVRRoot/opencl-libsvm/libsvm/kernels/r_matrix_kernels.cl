

__kernel void print_q(  
                          __global double const * Q
                        , ulong Q_sz1
                        , ulong Q_sz2
                    )
{
    for(ulong i = 0; i < Q_sz1; ++i)
        for(ulong j = 0; j < Q_sz2; ++j)
             printf("Q[%i, %i] = %f\n", i, j, Q[i*Q_sz1 + j]);
}


__kernel void build_qsi(
                          __global double const * Q
                        , ulong Q_sz1
                        , __global long const * sv_indexes
                        , ulong sv_indexes_size
                        , long SampleIndex
                        , __global double * qsi
                    )
{
    uint const gid = get_global_id(0);
    
    if(gid == 0)
        qsi[0] = 1;
    else
        qsi[gid] = Q[SampleIndex * Q_sz1 + sv_indexes[gid-1]];
}


__kernel void mul_mat_vec(
                          __global double const * mat
                        , ulong mat_rows
                        , ulong mat_cols
                        , __global double const * vec
                        , __global double * res_vec
                    )
{
    uint const gid = get_global_id(0);

    double tmp = 0.0;
    ulong offset = mat_cols * gid;
    
    for(ulong i = 0; i < mat_cols; ++i)
    {
//        printf("mul_mat_vec: gid: %i, i: %i, vec: %f, mat: %f\n", gid, i, vec[i], mat[offset + i]);
        tmp += vec[i] * mat[offset + i];
    }
    
    res_vec[gid] = tmp;
}


__kernel void mul_mat_vec_invert(
                          __global double const * mat
                        , ulong mat_rows
                        , ulong mat_cols
                        , __global double const * vec
                        , __global double * res_vec
                    )
{
    uint const gid = get_global_id(0);

    double tmp = 0.0;
    ulong offset = mat_cols * gid;
    
    for(ulong i = 0; i < mat_cols; ++i)
    {
//        printf("mul_mat_vec_invert: gid: %i, i: %i, vec: %f, mat: %f\n", gid, i, vec[i], mat[offset + i]);
        tmp += vec[i] * mat[offset + i];
    }
    
    res_vec[gid] = -tmp;
    
}


__kernel void build_qxi(
                          __global double const * Q
                        , ulong Q_sz1
                        , long SampleIndex
                        , __global double * qxi
                    )
{
    uint const gid = get_global_id(0);
    qxi[gid] = Q[SampleIndex * Q_sz1 + gid];
}


//    Matrix<double>* M = new Matrix<double>();
//    for (int i = 0; i < SampleIndex; i++) {
//        Vector<double>* V2 = new Vector<double>(sv_indexes.size() - 1);
//        for (size_t j = 0; j < sv_indexes.size() - 1; j++) {
//            V2->AddFast(Q.GetValue(i, sv_indexes[j]));
//        }
//        M->AddRowRef(V2);
//    }
//    return M;
__kernel void build_qxs(
                          __global double const * Q
                        , ulong Q_sz1
                        , __global long const * sv_indexes
                        , ulong sv_indexes_size
                        , __global double * qxs
                    )
{
    uint const gid = get_global_id(0);
    
    __global double * start = qxs + sv_indexes_size * gid;
    
    *start++ = 1;
    for(ulong j = 1; j < sv_indexes_size; ++j)
        *start++ = Q[Q_sz1 * gid + sv_indexes[j-1]];
}


__kernel void add_vec_vec(
                          __global double * in_out
                        , __global double const * in
                    )
{
    uint const gid = get_global_id(0);
    in_out[gid] += in[gid];
}


//    vektor<double>* Zeros = vektor<double>::zero_vector(R->get_length_cols());
//    R->add_col_copy(Zeros);
//    Zeros->add(0);
//    R->add_row_ref(Zeros);
__kernel void add_zeros_to_r(
                          __global double const * old_r
                        , __global double       * new_r
                    )
{
    uint const row = get_global_id(0)
        , col = get_global_id(1)
        , rows = get_global_size(0)
        , cols = get_global_size(1)
        , new_cols = cols + 1;
        ;
        
    new_r[new_cols * row + col] = old_r[cols * row + col];
    
    if(col == cols-1)
        new_r[new_cols * row + cols] = 0;
    
    if(row == rows-1)
        new_r[new_cols * (row + 1) + col] = 0;
    
    if(row == rows-1 && col == cols-1)
        new_r[new_cols * (row + 1) + cols] = 0;
}


//        Beta->add(1);  Beta has already got a cell for the 1 to be put into
//        vmatrix<double>* BetaMatrix = vmatrix<double>::product_vector_vector(Beta, Beta);
//        BetaMatrix->divide_scalar(Gamma->get_value(SampleIndex));
//        R->sum_matrix(BetaMatrix);
//        delete BetaMatrix;
__kernel void rebuild_r(
                          __global double const * beta
                        , __global double       * r_matrix
                        , double const GammaSampleIndex
                    )
{
    uint const row = get_global_id(0);
    uint const col = get_global_id(1);
    uint const col_size = get_global_size(1);
    
//    printf("GM[%i,%i]=%f\n", row, col, beta[row] * beta[col]);

    r_matrix[col_size * row + col] += beta[row] * beta[col] / GammaSampleIndex;
    
}

