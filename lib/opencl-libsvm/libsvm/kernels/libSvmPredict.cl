#pragma OPENCL EXTENSION cl_khr_fp64: enable

enum { C_SVC = 0, NU_SVC, ONE_CLASS, EPSILON_SVR, NU_SVR };	/* svm_type */
enum { LINEAR = 0, POLY, RBF, SIGMOID, PRECOMPUTED };           /* kernel_type */

//typedef struct _svm_node{
//    int index;
//    double value;
//}svm_node;

//typedef struct _svm_node_float{
//    int index;
//    float value;
//}svm_node_float;

typedef struct _svm_parameter
{
    int svm_type;
    int kernel_type;
    int degree;	/* for poly */
    float gamma;	/* for poly/rbf/sigmoid */
    float coef0;	/* for poly/sigmoid */    
}svm_parameter;

typedef struct _svm_model
{
    svm_parameter* param;	/* parameter */
    int nr_class;		/* number of classes, = 2 in regression/one class svm */
    int svsLength;		/* total #SV */
    int svsWidth;
    //switch to one dimension
    __global const float* SV;
    //switch to one dimension    
    //dimensions are l: nr_class - 1, w: svsLength
    __global const float* sv_coef;
    //dimension is: nr_class * (nr_class - 1) / 2;
    __constant float* rho;
    //dimension is: nr_class * (nr_class - 1) / 2;
//    double* probA;		/* pariwise probability information */
//    //dimension is: nr_class * (nr_class - 1) / 2;
//    double* probB;
//    //passing null
//    int* sv_indices;        /* sv_indices[0,...,nSV-1] are values in [1,...,num_traning_data] to indicate SVs in the training set */

    /* for classification only */
    //dimension is: nr_class
    __constant int* label;	/* label of each class (label[k]) */
    //dimension is: nr_class
    __constant int* nSV;    /* number of SVs for each class (nSV[k]) */
                                /* nSV[0] + nSV[1] + ... + nSV[k-1] = l */
    /* XXX */
    int free_sv;		/* 1 if svm_model is created by svm_load_model*/
                                /* 0 if svm_model is created by svm_train */
}svm_model;

float my_dot(__global const float* px, const size_t xLen, __global const float* py)
{
    float sum = 0;
    int i = 0;
    while (i < xLen - 1){
        int remain = xLen - i;
        if (remain >= 8){
            float8 xv = vload8(i / 8, px);
            float8 yv = vload8(i / 8, py);
            float8 d = xv * yv;            
            sum += d.s0 + d.s1 + d.s2 + d.s3 + d.s4 + d.s5 + d.s6 + d.s7;
            i += 8; 
        }
        else if (remain >= 4){
            float4 xv = vload4(i / 4, px);
            float4 yv = vload4(i / 4, py);
            float4 d = xv * yv;                
            sum += d.x + d.y + d.z + d.w;
            i += 4;
        }
        else if (remain >= 2){
            float2 xv = vload2(i / 2, px);
            float2 yv = vload2(i / 2, py);
            float2 d = xv * yv;                
            sum += d.x + d.y;
            i += 2;
        }
        else if (remain >= 1){
            float d = px[i] * py[i];
            sum += d;
            i++;
        }        
    }
    return sum;
}

float kfunction_rbf(    const __global float* x, const size_t xLen,
                        const __global float* y, const size_t yLen,
                        float gamma){
    float sum = 0;    
    int i = 0;
    while (i < xLen){
        int remain = xLen - i;
        if (remain >= 8){
            float8 xv = vload8(i / 8, x);
            float8 yv = vload8(i / 8, y);
            float8 d = xv - yv;
            d *= d;
            sum += d.s0 + d.s1 + d.s2 + d.s3 + d.s4 + d.s5 + d.s6 + d.s7;
            i += 8; 
        }
        else if (remain >= 4){
            float4 xv = vload4(i / 4, x);
            float4 yv = vload4(i / 4, y);
            float4 d = xv - yv;            
            d *= d;            
            sum += d.x + d.y + d.z + d.w;
            i += 4;
        }
        else if (remain >= 2){
            float2 xv = vload2(i / 2, x);
            float2 yv = vload2(i / 2, y);
            float2 d = xv - yv;
            d *= d;            
            sum += d.x + d.y;
            i += 2;
        }
        else if (remain >= 1){
            float d = x[i]  - y[i];
            sum += d * d;
            i++;
        }        
    }
    return exp(-gamma * sum);
}

float k_function(   const __global float* x, const size_t xLen,
                    const __global float* y, const size_t yLen, 
                    const svm_parameter* param) {
    switch (param->kernel_type) {
        case LINEAR:
            return my_dot(x, xLen, y);
        case POLY:
            return pow(param->gamma * my_dot(x, xLen, y) + param->coef0, param->degree);
        case RBF:
        {
            return kfunction_rbf(x, xLen, y, yLen, param->gamma);
        }
        case SIGMOID:
            return tanh(param->gamma * my_dot(x, xLen, y) + param->coef0);
        case PRECOMPUTED: //x: test (validation), y: SV
            return x[(int)y[0]];
        default:
            return 0; // Unreachable 
    }
}

double svm_predict_values(  const svm_model *model, const __global float* x,
                            const size_t xlen, __local int* start, __local int* vote)
{
    int i;
    if(model->param->svm_type == ONE_CLASS ||
       model->param->svm_type == EPSILON_SVR ||
       model->param->svm_type == NU_SVR)
    {       
        //sv_coef[0] this is the same
        __global const float* sv_coef = model->sv_coef; 
        double sum = 0;
        for(i =0 ; i < model->svsLength; i++){
            float kVal = k_function(x, xlen, &model->SV[i * model->svsWidth], model->svsWidth, model->param);
            sum += sv_coef[i] * kVal;
        }
        sum -= model->rho[0];        

        if(model->param->svm_type == ONE_CLASS)
            return (sum > 0) ? 1 : -1;
        else
            return sum;
    }
    else
    {        
        int nr_class = model->nr_class;
        
        start[0] = 0;
        for(i = 1; i < nr_class; i++)
            start[i] = start[i - 1] + model->nSV[i - 1];
        
        for(i = 0; i < nr_class; i++)
            vote[i] = 0;                    

        int p = 0;
        for(i = 0; i < nr_class; i++)
            for(int j = i + 1; j < nr_class; j++)
            {
                double sum = 0;
                int si = start[i];                
                int sj = start[j];                
                int ci = model->nSV[i];                
                int cj = model->nSV[j];                

                int k;
                __global const float* coef1 = &(model->sv_coef[(j - 1) * model->svsLength]);                
                __global const float* coef2 = &(model->sv_coef[i * model->svsLength]);                                                
                
                for(k = 0; k < ci; k++){
                    float kval = k_function(    x, xlen, 
                                                &model->SV[(si + k) * model->svsWidth], 
                                                model->svsWidth, model->param);                    
                    sum += coef1[si + k] * kval;//kvalue[si + k];                
                }
                for(k = 0; k < cj; k++){
                    float kval = k_function(    x, xlen, 
                                                &model->SV[(sj + k) * model->svsWidth], 
                                                model->svsWidth, model->param);
                    sum += coef2[sj + k] * kval;//kvalue[sj + k];
                }
                sum -= model->rho[p];                

                if(sum > 0.f)
                    ++vote[i];
                else
                    ++vote[j];
                p++;
            }

        int vote_max_idx = 0;
        for(i = 1; i < nr_class; i++)
            if(vote[i] > vote[vote_max_idx])
                vote_max_idx = i;
        
        return model->label[vote_max_idx];
    }
}

//dec_values size = 1 or nr_class * (nr_class - 1) / 2, needs to be just allocated
//kvalue size = svsLength, needs to be just allocated
//start size = nr_class, needs to be just allocated
//vote size = nr_class, needs to be just allocated
double svm_predict( const svm_model *model, const __global float *x, const size_t xLen,
                    __local int* start, __local int* vote){
                        
    double pred_result = svm_predict_values(model, x, xLen, start, vote);    
    return pred_result;    
}

__kernel void predict(  //input args
                        __global const float* x, const uint xLen, const uint xNumOfParameters,
                        //model args
                        const int nr_class, const int svsLength,
                        const int svsWidth, __global const float* SV,
                        __global const float* sv_coef, __constant float* rho,
                        __constant int* label, __constant int* nSV, const int free_sv,
                        //parameter args
                        const int svm_type, const int kernel_type, const int degree,
                        const float gamma, const float coef0,
                        //return args
                        __global uchar* results,
                        //prealocated args, specified for work group                        
                        __local int* startMat, __local int* voteMat){
    const int index = get_global_id(0);    
    if (index < xLen){
        const int localIndex = get_local_id(0);
        svm_model model;
        model.nr_class = nr_class;
        model.svsLength = svsLength;
        model.svsWidth = svsWidth;
        model.SV = SV;
        model.sv_coef = sv_coef;
        model.rho = rho;
        if (label)
            model.label = label;
        if (nSV)
            model.nSV = nSV;
        model.free_sv = free_sv;

        svm_parameter parameter;
        parameter.svm_type = svm_type;
        parameter.kernel_type = kernel_type;
        parameter.degree = degree;
        parameter.gamma = gamma;
        parameter.coef0 = coef0;    

        model.param = &parameter;

        const __global float* currX = x + index * xNumOfParameters;
        
        __local int* start = startMat + (localIndex * nr_class);
        __local int* vote = voteMat + (localIndex * nr_class);
        
        double res = svm_predict(&model, currX, xNumOfParameters, start, vote);
        if (res > 0.500003)
            results[index] = 1;
        else
            results[index] = 0;
    }
}