
__kernel void predict(__global float* pixelParameters, __global float* coefs,
                        float borderValue, uint parameterCount, uint pixelCount, 
                        __global uchar* retResults){

    const int index = get_global_id(0);
    if (index < pixelCount){
        __global float* currParameters = pixelParameters + (index * parameterCount);
        float result = coefs[parameterCount];
        for (int j = 0; j < parameterCount; j++){
            float a = currParameters[j] * coefs[j];
            result += a;
        }
        result = -result;
        result = exp(result);
        result = 1.f + result;
        result = 1.f / result;
        if (result > borderValue)
            retResults[index] = 1U;
        else
            retResults[index] = 0U;
    }
}
