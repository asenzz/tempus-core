//On AMD GPU
#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#else
#error "Double precision floating point not supported by OpenCL implementation."
#endif

#define blockX(i, j) (X[(startX + (i)) * input_internal_size2+(j)])
#define blockY(i, j) (Y[(startY + (i)) * input_internal_size2+(j)])

#define type_float double
#define w(i_j_plus_1) (1.0 / w_sum_sym * (i_j_plus_1) * (i_j_plus_1))

__kernel void
path_kernel_run(
        __global type_float *X, __global type_float *Y, const long input_internal_size2, const long size1,
        const long nXy, const long output_internal_size2, const long nYy, const int dimvect,
        const type_float sigma, __global type_float *total_result, __global type_float *supermin,
        const long startX,
        const long startY, const long numX, const long numY, const type_float w_sum_sym,
        const type_float lambda)
{
#define DIFF_COEFF 0.25
#define Nx_local 1024
#define TILE_WIDTH 32
//Must be 32x32 == 1024 == Nx_local, careful!
    int len = (int) (nXy / dimvect);

    barrier(CLK_LOCAL_MEM_FENCE);

    __local double power_mult[TILE_WIDTH];
    __local double  ta[TILE_WIDTH][TILE_WIDTH];
    __local double  tam1[TILE_WIDTH][TILE_WIDTH];//for index-1
    __local double  tb[TILE_WIDTH][TILE_WIDTH];
    __local double  tbm1[TILE_WIDTH][TILE_WIDTH];//for index-1
    int kk = get_global_id(0);
    int mm = get_global_id(1);

    int tx=get_local_id(0);
    int ty=get_local_id(1);

    if ((get_group_id(0)*TILE_WIDTH<numX) * (get_group_id(1)*TILE_WIDTH<numY)) { //the other case can only happen by mistake in the call, but still, better check

/*
        int cut_len = 0;
        for (int ii = 0; ii < len; ++ii) {
            if (1. / ((ii + 1.) * (ii + 1.)) < 0.1) {
                cut_len = ii;
                break;
            }
        }
*/
        //Set initial value for searched minimum.
        int kk_internal = 0;
        //int mm_internal = 0;
        double matrix_prod_sum = 0.0;
        for (int jA = 0; jA < dimvect; ++jA) {
            double s_mm = 0;
            /*
                for (int ii = 0; ii < len - 1; ++ii) {
                    s_mm += (blockX(kk, (ii + 1 + jA * len)) - blockX(kk, ii + jA * len)) *
                            (blockX(kk, (ii + 1 + jA * len)) - blockX(kk, ii + jA * len));
                }
                double delta_left = sqrt(s_mm / ((double) len - 1.));
                s_mm = 0.;
                for (int ii = 0; ii < len - 1; ++ii) {
                    s_mm += (blockY(mm, (ii + 1 + jA * len)) - blockY(mm, ii + jA * len)) *
                            (blockY(mm, (ii + 1 + jA * len)) - blockY(mm, ii + jA * len));
                }
                double delta_right = sqrt(s_mm / ((double) len - 1.));
                s_mm = 0.;
            */
            //this will be split in two            for (kk_internal = 0; kk_internal < len; ++kk_internal) {
            for (int kk_internal_big = 0; kk_internal_big < len/TILE_WIDTH + (len%TILE_WIDTH==0?0:1); ++kk_internal_big) {
                if (tx==0){
                    if (ty+kk_internal_big*TILE_WIDTH<len){
                        power_mult[ty]=pow(1./((double)(len-(ty+kk_internal_big*TILE_WIDTH))),2*lambda)*w_sum_sym;
                    }
                }
                if ((kk<numX)*(TILE_WIDTH*kk_internal_big+ty < len)){
                    ta[tx][ty]=blockX(kk, TILE_WIDTH*kk_internal_big+ty + jA * len);
                    if (TILE_WIDTH*kk_internal_big+ty>0){
                        tam1[tx][ty]=ta[tx][ty]-blockX(kk, TILE_WIDTH*kk_internal_big+ty-1 + jA * len);
                    }
                }
                if ((mm<numY)*( TILE_WIDTH*kk_internal_big+tx <len)){
                    tb[ty][tx]=blockY(mm, TILE_WIDTH*kk_internal_big+tx + jA * len);
                    if (TILE_WIDTH*kk_internal_big+tx>0){
                        tbm1[ty][tx]=tb[ty][tx]-blockY(mm, TILE_WIDTH*kk_internal_big+tx-1 + jA * len);
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);

                if ((kk<numX) * (mm<numY)){
                    for (int kk_internal_small = 0; kk_internal_small < TILE_WIDTH; ++kk_internal_small) {
                        kk_internal = kk_internal_small+kk_internal_big*TILE_WIDTH;
                        if (kk_internal<len){

                            //mm_internal = kk_internal;
                            double x_y = ta[tx][kk_internal_small]-tb[ty][kk_internal_small];
                            double t_left = x_y * x_y;
                            double t_right = 0;
                            if (kk_internal>0){
                                double diff_x_y = 0;
                                diff_x_y = tam1[tx][kk_internal_small]-tbm1[ty][kk_internal_small];
                                t_right = diff_x_y * diff_x_y;
                            }
/*
                for (mm_internal = max(kk_internal - cut_len, 0);
                     mm_internal < min(len, kk_internal + cut_len + 1); ++mm_internal) {
                    if (mm_internal == kk_internal) continue;//already computed above
                    double x_y = blockX(kk, (kk_internal + jA * len)) - blockY(mm, (mm_internal + jA * len));
                    x_y = x_y * x_y / w_sum_sym + delta_right * delta_right * 4. * w(kk_internal - mm_internal);
                    t_left = fmin(t_left, x_y);
                    x_y = blockX(kk, (mm_internal + jA * len)) - blockY(mm, (kk_internal + jA * len));
                    x_y = x_y * x_y / w_sum_sym + delta_left * delta_left * 4. * w(kk_internal - mm_internal);
                    t_right = fmin(t_right, x_y);
                }
*/
                            s_mm += (t_left+DIFF_COEFF* t_right) * power_mult[kk_internal_small];
                        }//end if kk_internal
                    }//end for kk_internal_small
                }//end if
                barrier(CLK_LOCAL_MEM_FENCE);//DO NOT REMOVE!
            }//end for kk_internal_big - tiles
            matrix_prod_sum += s_mm;
        }//end for jA
        if ((kk<numX) * (mm<numY)){
            total_result[(startX + kk) * output_internal_size2 + (startY + mm)] = 1. - matrix_prod_sum / (2. * sigma * sigma);
        }
    }//end if check get_global 0 and 1
}