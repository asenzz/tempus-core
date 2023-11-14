//!!!!!STILL NOT FUNCTIONAL DON"T USE IT !!!!!\\

#pragma OPENCL EXTENSION cl_khr_fp64: enable

enum { C_SVC = 0, NU_SVC, ONE_CLASS, EPSILON_SVR, NU_SVR };	/* svm_type */
enum { LINEAR = 0, POLY, RBF, SIGMOID, PRECOMPUTED }; /* kernel_type */
enum { LOWER_BOUND = 0, UPPER_BOUND, FREE }; /*alpha status value*/

#define TAU 1e-12

double my_dot(__global const double* px, __global const double* py, const int xW)
{
    double sum = 0;
    int i = 0;
    while (i < xW - 1){
        int remain = xW - i;
        if (remain >= 8){
            double8 xv = vload8(i / 8, px);
            double8 yv = vload8(i / 8, py);
            double8 d = xv * yv;            
            sum += d.s0 + d.s1 + d.s2 + d.s3 + d.s4 + d.s5 + d.s6 + d.s7;
            i += 8; 
        }
        else if (remain >= 4){
            double4 xv = vload4(i / 4, px);
            double4 yv = vload4(i / 4, py);
            double4 d = xv * yv;                
            sum += d.x + d.y + d.z + d.w;
            i += 4;
        }
        else if (remain >= 2){
            double2 xv = vload2(i / 2, px);
            double2 yv = vload2(i / 2, py);
            double2 d = xv * yv;                
            sum += d.x + d.y;
            i += 2;
        }
        else if (remain >= 1){
            double d = px[i] * py[i];
            sum += d;
            i++;
        }        
    }
    return sum;
}

double kernel_linear(const int i, const int j, __global const double* x, const int xW){
    return my_dot(&x[i * xW], &x[j * xW], xW);
}

double kernel_poly( const int i, const int j, __global const double* x, const int xW, 
                    const double gamma, const double coef0, const int degree){    
    return pow(gamma * my_dot(&x[i * xW], &x[j * xW], xW) + coef0, degree);
}

double kernel_rbf(  const int i, const int j, __global const double* x, const int xW,
                    double gamma, __global double* x_square){   
    return exp(-gamma * (x_square[i] + x_square[j] - 2 * my_dot(&x[i * xW], &x[j * xW], xW)));
}

double kernel_sigmoid(const int i, const int j, __global const double* x, 
                      const int xW, const double gamma, const double coef0){
    return tanh(gamma * my_dot(&x[i * xW], &x[j * xW], xW) + coef0);
}

double kernel_precomputed(const int i, const int j, __global const double* x, const int xW){
    int jIndex = (int)x[j * xW];
    return x[i * xW + jIndex];
}

float svrQgetQ_f(const int len, const int i, const int kernel_type,
                __global const double* x, const int xW, const double gamma,
                const double coef0, const int degree, __global double* x_square, 
                const int index){
    float retVal = 0.f;
    switch (abs(kernel_type)){
    case LINEAR:
        retVal = (float)kernel_linear(i, index, x, xW);
        break;
    case POLY:
        retVal = (float)kernel_poly(i, index, x, xW, gamma, coef0, degree);
        break;
    case RBF:            
        retVal = (float)kernel_rbf(i, index, x, xW, gamma, x_square);
        break;
    case SIGMOID:
        retVal = (float)kernel_sigmoid(i, index, x, xW, gamma, coef0);
        break;
    case PRECOMPUTED:
        retVal = (float)kernel_precomputed(i, index, x, xW);
        break;
    }
    return retVal;
}

float svcQgetQ_f(const int len, const int i, const int kernel_type, 
                __global const char* y, __global const double* x, const int xW,
                double gamma, double coef0, int degree, __global double* x_square,
                const int index){
    float res = y[i] * y[index];
    switch (abs(kernel_type)){
    case LINEAR:
        res *= (float)kernel_linear(i, index, x, xW);
        break;
    case POLY:
        res *= (float)kernel_poly(i, index, x, xW, gamma, coef0, degree);
        break;
    case RBF:
        res *= (float)kernel_rbf(i, index, x, xW, gamma, x_square);
        break;
    case SIGMOID:
        res *= (float)kernel_sigmoid(i, index, x, xW, gamma, coef0);
        break;
    case PRECOMPUTED:
        res *= (float)kernel_precomputed(i, index, x, xW);
        break;
    }
    return res;
}

__kernel void svcQgetQ( __global float* data, const int dataLen, const int start, 
                        const int len, const int i, const int kernel_type, 
                        __global const char* y, __global const double* x, const int xW,
                        double gamma, double coef0, int degree, __global double* x_square)
{  
    
    const int index = get_global_id(0);
    const int realIndex = index + start;    
    if (realIndex < dataLen){        
        data[realIndex] = svcQgetQ_f(len, i, kernel_type, y, x, xW, gamma, 
                                    coef0, degree, x_square, realIndex);
    }
}

__kernel void svrQgetQ (__global float* data, const int dataLen, const int start, 
                        const int len, const int i, const int kernel_type,
                         __global const double* x, const int xW, const double gamma,
                        const double coef0, const int degree, __global double* x_square)
{    
    const int index = get_global_id(0);
    const int realIndex = index + start;    
    if (realIndex < dataLen){
        data[realIndex] = svrQgetQ_f(len, i, kernel_type, x, xW, gamma,
                                    coef0, degree, x_square, realIndex);
    }
}

bool is_lower_bound(int i, __global const char* alpha_status) {
    return alpha_status[i] == LOWER_BOUND;
}

bool is_upper_bound(int i, __global const char* alpha_status) {
    return alpha_status[i] == UPPER_BOUND;
}

__kernel void selectWorkingSet( const int activeSize, const int i, __global const char* y,
                                __global const char* alpha_status, __global double* grad_diff,
                                const double Gmax, __global const double* G,
                                __global const double* QD, __global const float *Q_i,
                                __global double* obj_diff){
    const int index = get_global_id(0);
    if (index < activeSize){
        if (y[index] == 1.) {
            if (!is_lower_bound(index, alpha_status)) {
                grad_diff[index] = Gmax + G[index];                
                if (grad_diff[index] > 0) {                    
                    double quad_coef = QD[i] + QD[index] - 2.0 * y[i] * Q_i[index];
                    if (quad_coef > 0)
                        obj_diff[index] = -(grad_diff[index] * grad_diff[index]) / quad_coef;
                    else
                        obj_diff[index] = -(grad_diff[index] * grad_diff[index]) / TAU;                    
                }
            }
        } else {
            if (!is_upper_bound(index, alpha_status)) {
                grad_diff[index] = Gmax - G[index];
                if (grad_diff[index] > 0) {                    
                    double quad_coef = QD[i] + QD[index] + 2.0 * y[i] * Q_i[index];
                    if (quad_coef > 0)
                        obj_diff[index] = -(grad_diff[index] * grad_diff[index]) / quad_coef;
                    else
                        obj_diff[index] = -(grad_diff[index] * grad_diff[index]) / TAU;
                }
            }
        }
    }
}


