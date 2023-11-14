#ifdef _OPENCL
#include "OpenCLToolsPredict.h"
#include "thirdparty/lib_svm/svm.h"
#include "core/util/Matrix.h"

namespace core {
    namespace opencl {
        namespace libsvm {

            using namespace core::util;
            using namespace std;
            
            OpenCLToolsPredict::OpenCLToolsPredict(): Singleton<OpenCLToolsPredict>(){
                initVars();
            }
            
            OpenCLToolsPredict::~OpenCLToolsPredict(){                
            }
            
            void OpenCLToolsPredict::initVars(){
                OpenClBase::initVars();
                
                modelSVs        = 0;
                clModelSVs      = 0;
                clModelRHO      = 0;
                clModelSVCoefs  = 0;
                clModelLabel    = 0;
                svCoefs         = 0;
                clModelNsv      = 0;
                modelRHOs       = 0;
                modelChanged    = true;
                
                initWorkVars();
            }
            
            void OpenCLToolsPredict::initWorkVars(){
                clPixelParameters = 0;
                clPredictResults = 0;
            }
            
            void OpenCLToolsPredict::cleanUp(){
                OpenClBase::cleanUp();
                
                if (modelSVs)
                    Delete(modelSVs);
                if (clModelSVs){
                    err = clReleaseMemObject(clModelSVs);
                    err_check(err, "OpenclTools::cleanUp clModelSVs");
                }
                if (clModelRHO){
                    err = clReleaseMemObject(clModelRHO);
                    err_check(err, "OpenclTools::cleanUp clModelRHO");
                }
                if (clModelSVCoefs){
                    err = clReleaseMemObject(clModelSVCoefs);
                    err_check(err, "OpenclTools::cleanUp clModelSVCoefs");
                }            
                if (clModelLabel){
                    err = clReleaseMemObject(clModelLabel);
                    err_check(err, "OpenclTools::cleanUp clModelLabel");
                }
                if (svCoefs)
                    Delete(svCoefs);
                if (clModelNsv){
                    err = clReleaseMemObject(clModelNsv);
                    err_check(err, "OpenclTools::cleanUp clModelNsv");
                }
                if (modelRHOs)
                    Delete(modelRHOs);
                modelChanged = true;
                
                cleanWorkPart();
                initVars();
            }
            
            void OpenCLToolsPredict::cleanWorkPart(){
                if (clPixelParameters){
                    clReleaseMemObject(clPixelParameters);
                    err_check(err, "OpenclTools::cleanWorkPart clPredictResults");
                }
                if (clPredictResults){
                    err = clReleaseMemObject(clPredictResults);
                    err_check(err, "OpenclTools::cleanWorkPart clPredictResults");
                }
                
                initWorkVars();
            }
            
            size_t getSVsWidth(svm_model* model) {
                size_t maxWidth = 0;
                for (int i = 0; i < model->l; i++) {
                    size_t currWidth = 1;
                    svm_node* node = model->SV[i];
                    while (node->index != -1) {
                        currWidth++;
                        node++;
                    }
                    if (maxWidth < currWidth)
                        maxWidth = currWidth;
                }
                return maxWidth;
            }

            Matrix<float>* convertSVs(svm_model* model) {
                int height = model->l;
                int width = getSVsWidth(model);
                Matrix<float>* matrix = New Matrix<float>(width, height);
                for (int i = 0; i < height; i++) {
                    for (int j = 0; j < width; j++) {
                        //(*matrix)[i][j].index = model->SV[i][j].index;
                        (*matrix)[i][j] = model->SV[i][j].value;
                        if (model->SV[i][j].index == -1)
                            break;
                    }
                }
                return matrix;
            }

            Matrix<cl_float>* convertSVCoefs(svm_model* model) {
                int width = model->l;
                int height = model->nr_class - 1;
                Matrix<cl_float>* matrix = New Matrix<cl_float>(width, height);
                for (int i = 0; i < height; i++) {
                    for (int j = 0; j < width; j++) {
                        (*matrix)[i][j] = model->sv_coef[i][j];
                    }
                }
                return matrix;
            }

            cl_float* convertRHO(svm_model* model) {
                int count = model->nr_class * (model->nr_class - 1) / 2;
                cl_float* retVec = New cl_float[count];
                for (int i = 0; i < count; i++) {
                    retVec[i] = model->rho[i];
                }
                return retVec;
            }

            size_t getDecValuesSize(svm_model* model) {
                if (model->param.svm_type == ONE_CLASS || model->param.svm_type == EPSILON_SVR ||
                        model->param.svm_type == NU_SVR)
                    return 1;
                else {
                    return model->nr_class * (model->nr_class - 1) / 2;
                }
            }

            void OpenCLToolsPredict::createBuffers(const Matrix<float>* parameters,
                    svm_model* model) {
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

                size_t size = parameters->getWidth() * parameters->getHeight() * sizeof (cl_float);
                cl_float* pNodes = parameters->getVec();
                clPixelParameters = clCreateBuffer(context, flag2, size,
                        pNodes, &err);
                err_check(err, "OpenclTools::createBuffersPredict clPixelParameters");

                if (modelChanged) {
                    if (modelSVs != 0)
                        Delete(modelSVs);
                    modelSVs = convertSVs(model);
                    size = modelSVs->getWidth() * model->l * sizeof (cl_float);
                    if (clModelSVs) {
                        err = clReleaseMemObject(clModelSVs);
                        err_check(err, "OpenclTools::createBuffersPredict delete clModelSVs");
                    }
                    clModelSVs = clCreateBuffer(context, flag2, size, modelSVs->getVec(), &err);
                    err_check(err, "OpenclTools::createBuffersPredict clModelSVs");

                    if (svCoefs)
                        Delete(svCoefs);
                    svCoefs = convertSVCoefs(model);
                    size = (model->nr_class - 1) * (model->l) * sizeof (cl_float);
                    if (clModelSVCoefs) {
                        err = clReleaseMemObject(clModelSVCoefs);
                        err_check(err, "OpenclTools::createBuffersPredict delete clModelSVCoefs");
                    }
                    clModelSVCoefs = clCreateBuffer(context, flag2, size, svCoefs->getVec(), &err);
                    err_check(err, "OpenclTools::createBuffersPredict clModelSVCoefs");

                    int count = model->nr_class * (model->nr_class - 1) / 2;
                    size = count * sizeof (cl_float);
                    if (modelRHOs)
                        Delete(modelRHOs);
                    modelRHOs = convertRHO(model);
                    if (clModelRHO) {
                        err = clReleaseMemObject(clModelRHO);
                        err_check(err, "OpenclTools::createBuffersPredict delete clModelRHO");
                    }
                    clModelRHO = clCreateBuffer(context, flag2, size, modelRHOs, &err);
                    err_check(err, "OpenclTools::createBuffersPredict clModelRHO");

                    size = model->nr_class * sizeof (cl_int);
                    if (clModelLabel) {
                        err = clReleaseMemObject(clModelLabel);
                        err_check(err, "OpenclTools::createBuffersPredict delete clModelLabel");
                    }
                    if (model->label) {
                        clModelLabel = clCreateBuffer(context, flag2, size, (cl_int*) model->label, &err);
                        err_check(err, "OpenclTools::createBuffersPredict clModelLabel");
                    } else {
                        clModelLabel = clCreateBuffer(context, flag2, sizeof (cl_int), &dummyInt, &err);
                        err_check(err, "OpenclTools::createBuffersPredict clModelLabel");
                    }

                    size = model->nr_class * sizeof (cl_int);
                    if (clModelNsv) {
                        err = clReleaseMemObject(clModelNsv);
                        err_check(err, "OpenclTools::createBuffersPredict delete clModelNsv");
                    }
                    if (model->nSV) {
                        clModelNsv = clCreateBuffer(context, flag2, size, (cl_int*) model->nSV, &err);
                        err_check(err, "OpenclTools::createBuffersPredict clModelNsv");
                    } else {
                        clModelNsv = clCreateBuffer(context, flag2, sizeof (cl_int), &dummyInt, &err);
                        err_check(err, "OpenclTools::createBuffersPredict clModelLabel");
                    }
                    modelChanged = false;
                }

                size = parameters->getHeight() * sizeof (cl_uchar);
                clPredictResults = clCreateBuffer(context, flag1, size, 0, &err);
                err_check(err, "OpenclTools::createBuffersPredict clPredictResults");
            }

            void OpenCLToolsPredict::setKernelArgs(uint pixelCount, uint paramsPerPixel,
                    svm_model* model) {
                err = clSetKernelArg(kernel[0], 0, sizeof (cl_mem), &clPixelParameters);
                err_check(err, "OpenclTools::setKernelArgsPredict clPixelParameters");
                err = clSetKernelArg(kernel[0], 1, sizeof (cl_uint), &pixelCount);
                err_check(err, "OpenclTools::setKernelArgsPredict pixelCount");
                err = clSetKernelArg(kernel[0], 2, sizeof (cl_uint), &paramsPerPixel);
                err_check(err, "OpenclTools::setKernelArgsPredict paramsPerPixel");
                err = clSetKernelArg(kernel[0], 3, sizeof (cl_int), &model->nr_class);
                err_check(err, "OpenclTools::setKernelArgsPredict nr_class");
                err = clSetKernelArg(kernel[0], 4, sizeof (cl_int), &model->l);
                err_check(err, "OpenclTools::setKernelArgsPredict l");
                int modelSvsWidth = modelSVs->getWidth();
                err = clSetKernelArg(kernel[0], 5, sizeof (cl_int), &modelSvsWidth);
                err_check(err, "OpenclTools::setKernelArgsPredict svsWidth");
                err = clSetKernelArg(kernel[0], 6, sizeof (cl_mem), &clModelSVs);
                err_check(err, "OpenclTools::setKernelArgsPredict clModelSVs");
                err = clSetKernelArg(kernel[0], 7, sizeof (cl_mem), &clModelSVCoefs);
                err_check(err, "OpenclTools::setKernelArgsPredict clModelSVCoefs");
                err = clSetKernelArg(kernel[0], 8, sizeof (cl_mem), &clModelRHO);
                err_check(err, "OpenclTools::setKernelArgsPredict clModelRHO");
                if (clModelLabel) {
                    err = clSetKernelArg(kernel[0], 9, sizeof (cl_mem), &clModelLabel);
                    err_check(err, "OpenclTools::setKernelArgsPredict clModelLabel");
                } else {
                    //                err = clSetKernelArg(kernel[0], 9, 0, 0);
                    //                err_check(err, "OpenclTools::setKernelArgsPredict clModelLabel", -1);
                }
                if (clModelNsv) {
                    err = clSetKernelArg(kernel[0], 10, sizeof (cl_mem), &clModelNsv);
                    err_check(err, "OpenclTools::setKernelArgsPredict clModelNsv");
                } else {
                    //                err = clSetKernelArg(kernel[0], 10, 0, 0);
                    //                err_check(err, "OpenclTools::setKernelArgsPredict clModelNsv", -1);
                }
                err = clSetKernelArg(kernel[0], 11, sizeof (cl_int), &model->free_sv);
                err_check(err, "OpenclTools::setKernelArgsPredict free_sv");
                err = clSetKernelArg(kernel[0], 12, sizeof (cl_int), &model->param.svm_type);
                err_check(err, "OpenclTools::setKernelArgsPredict param.svm_type");
                err = clSetKernelArg(kernel[0], 13, sizeof (cl_int), &model->param.kernel_type);
                err_check(err, "OpenclTools::setKernelArgsPredict param.kernel_type");
                err = clSetKernelArg(kernel[0], 14, sizeof (cl_int), &model->param.degree);
                err_check(err, "OpenclTools::setKernelArgsPredict param.degree");
                cl_float gamma = model->param.gamma;
                err = clSetKernelArg(kernel[0], 15, sizeof (cl_float), &gamma);
                err_check(err, "OpenclTools::setKernelArgsPredict param.gamma");
                cl_float coef0 = model->param.coef0;
                err = clSetKernelArg(kernel[0], 16, sizeof (cl_float), &coef0);
                err_check(err, "OpenclTools::setKernelArgsPredict param.coef0");
                //====
                err = clSetKernelArg(kernel[0], 17, sizeof (cl_mem), &clPredictResults);
                err_check(err, "OpenclTools::setKernelArgsPredict clPredictResults");
                //====
                size_t size = model->nr_class * sizeof (cl_int) * workGroupSize[0];
                err = clSetKernelArg(kernel[0], 18, size, 0);
                err_check(err, "OpenclTools::setKernelArgsPredict start");

                size = model->nr_class * sizeof (cl_int) * workGroupSize[0];
                err = clSetKernelArg(kernel[0], 19, size, 0);
                err_check(err, "OpenclTools::setKernelArgsPredict vote");
            }

            uchar* OpenCLToolsPredict::predict(svm_model* model, const Matrix<float>* parameters) {
                createBuffers(parameters, model);
                setKernelArgs(parameters->getHeight(), parameters->getWidth(), model);

                size_t local_ws = workGroupSize[0];
                int numValues = parameters->getHeight() * parameters->getWidth();
                size_t global_ws = shrRoundUp(local_ws, numValues);
                err = clEnqueueNDRangeKernel(command_queue, kernel[0], 1, NULL, &global_ws, &local_ws, 0, NULL, NULL);
                err_check(err, "OpenclTools::predict clEnqueueNDRangeKernel");
                size_t size = parameters->getHeight() * sizeof(cl_uchar);
                uchar* retVec = New uchar[parameters->getHeight()];
                err = clEnqueueReadBuffer(command_queue, clPredictResults, CL_TRUE, 0, size, retVec, 0, NULL, NULL);
                err_check(err, "OpenclTools::predict clEnqueueReadBuffer");
                clFlush(command_queue);
                clFinish(command_queue);
                return retVec;
            }

            void OpenCLToolsPredict::markModelChanged() {
                modelChanged = true;
            }
            
            string OpenCLToolsPredict::getClassName(){
                return string("core::opencl::libsvm::OpenCLToolsPredict");
            }

        }
    }
}
#endif