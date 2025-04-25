#include "OpenCLRegressionPredict.h"

#ifdef _OPENCL
namespace core{
    namespace opencl{
        namespace regression{
            
            using namespace core::util;            
            using namespace std;            
            
            OpenCLRegressionPredict::OpenCLRegressionPredict() : Singleton<OpenCLRegressionPredict>(){
                initVars();
            }
            
            OpenCLRegressionPredict::~OpenCLRegressionPredict(){
                cleanUp();
            }
            
            string OpenCLRegressionPredict::getClassName(){
                return "core::opencl::regression::OpenCLRegressionPredict";
            }
            
            void OpenCLRegressionPredict::initWorkVars(){
                pixelCount = 0;
                parameterCount = 0;
                pixelParametersBuff = 0;
                predictedBuff = 0;
                pixelParameters = 0;
            }
            
            void OpenCLRegressionPredict::cleanWorkPart(){
                if (pixelParametersBuff != 0)
                    clReleaseMemObject(pixelParametersBuff);
                if (predictedBuff)
                    clReleaseMemObject(predictedBuff);
            }
            
            void OpenCLRegressionPredict::initVars(){
                OpenClBase::initVars();
                regressionCoefs = 0;
                regressionCoefsBuff = 0;
                initWorkVars();
            }
            
            void OpenCLRegressionPredict::cleanUp(){
                OpenClBase::cleanUp();
                if (regressionCoefs)
                    DeleteArr(regressionCoefs);
                if (regressionCoefsBuff)
                    clReleaseMemObject(regressionCoefsBuff);
                cleanWorkPart();                
            }
            
            float* createRegressionCoefs(vector<float> coefs){
                float* retArr = New float[coefs.size()];
                for (uint i = 0; i < coefs.size(); i++){
                    retArr[i] = coefs[i];
                }
                return retArr;
            }
            
            uchar* OpenCLRegressionPredict::predict(const core::util::Matrix<float>& imagePixelParameters,
                                const int& pixelCount, const int& parameterCount,
                                std::vector<float> coefs, float borderValue){
                this->parameterCount = parameterCount;
                this->pixelCount = pixelCount;
                pixelParameters = &imagePixelParameters;                
                if (regressionCoefs == 0){
                    regressionCoefs = createRegressionCoefs(coefs);
                    regressionCoefsNum = coefs.size();
                }
                this->borderValue = borderValue;
                createBuffers();
                setKernelArgs();
                
                size_t local_ws = workGroupSize[0];                
                size_t global_ws = shrRoundUp(local_ws, pixelCount);
                err = clEnqueueNDRangeKernel(command_queue, kernel[0], 1, NULL, &global_ws, &local_ws, 0, NULL, NULL);
                err_check(err, "OpenCLRegressionPredict::predict clEnqueueNDRangeKernel");
                
                size_t size = pixelCount * sizeof(cl_uchar);
                uchar* retVec = New uchar[pixelCount];
                err = clEnqueueReadBuffer(command_queue, predictedBuff, CL_TRUE, 0, size, retVec, 0, NULL, NULL);
                err_check(err, "OpenclTools::predict clEnqueueReadBuffer");
                clFlush(command_queue);
                clFinish(command_queue);
                return retVec;
            }
            
            void OpenCLRegressionPredict::createBuffers() throw(SDException&){
                cl_device_type type;
                clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof (cl_device_type), &type, 0);
                int flag1, flag2;
                if (type == CL_DEVICE_TYPE_GPU) {
                    flag1 = CL_MEM_WRITE_ONLY;
                    flag2 = CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR;
                } else if (type == CL_DEVICE_TYPE_CPU) {
                    flag1 = CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR;
                    flag2 = CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR;
                } else {
                    SDException exc(SHADOW_NOT_SUPPORTED_DEVICE, "Init buffers, currently not supported device");
                    throw exc;
                }
                size_t size = pixelCount * parameterCount * sizeof(cl_float);
                pixelParametersBuff = clCreateBuffer(context, flag2, size, pixelParameters->getVec(), &err);
                err_check(err, "OpenCLRegressionPredict::createBuffers pixelParametersBuff");                
                size = pixelCount * sizeof(cl_uchar);
                predictedBuff = clCreateBuffer(context, flag1, size, 0, &err);
                err_check(err, "OpenCLRegressionPredict::createBuffers hlsImageBuff");
                if (regressionCoefsBuff == 0){
                    size = regressionCoefsNum * sizeof(cl_float);
                    regressionCoefsBuff = clCreateBuffer(context, flag2, size, regressionCoefs, &err);
                    err_check(err, "OpenCLRegressionPredict::createBuffers regressionCoefsBuff");
                }
            }
            
            void OpenCLRegressionPredict::setKernelArgs() throw(SDException&){
                err = clSetKernelArg(kernel[0], 0, sizeof (cl_mem), &pixelParametersBuff);
                err_check(err, "OpenCLRegressionPredict::setKernelArgs pixelParametersBuff");
                err = clSetKernelArg(kernel[0], 1, sizeof (cl_mem), &regressionCoefsBuff);
                err_check(err, "OpenCLRegressionPredict::setKernelArgs regressionCoefsBuff");
                err = clSetKernelArg(kernel[0], 2, sizeof (cl_float), &borderValue);
                err_check(err, "OpenCLRegressionPredict::setKernelArgs regressionCoefsNum");
                err = clSetKernelArg(kernel[0], 3, sizeof (cl_uint), &parameterCount);
                err_check(err, "OpenCLRegressionPredict::setKernelArgs parameterCount");
                err = clSetKernelArg(kernel[0], 4, sizeof (cl_uint), &pixelCount);
                err_check(err, "OpenCLRegressionPredict::setKernelArgs imageHeight");
                err = clSetKernelArg(kernel[0], 5, sizeof (cl_mem), &predictedBuff);
                err_check(err, "OpenCLRegressionPredict::setKernelArgs predictedBuff");
            }
            
        }
    }
}
#endif 
