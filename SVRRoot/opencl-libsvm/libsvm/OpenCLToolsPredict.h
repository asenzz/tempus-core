#ifndef __OPENCLTOOLS_PREDICT_H__ 
#define __OPENCLTOOLS_PREDICT_H__

#ifdef _OPENCL

#include "core/opencl/OpenClToolsBase.h"
#include "core/util/Singleton.h"

struct svm_model;

namespace core{
    
    namespace util{
        template<typename T> class Matrix;
    }
    
    namespace opencl{
        namespace libsvm{
            /**
             * Class performing parallel SVM prediction
             */
            class OpenCLToolsPredict : public core::opencl::OpenClBase, public core::util::Singleton<OpenCLToolsPredict>{
                friend class core::util::Singleton<OpenCLToolsPredict>;
            private:
                cl_int          dummyInt;
                
                bool            modelChanged;            
                cl_mem          clPixelParameters;
                core::util::Matrix<cl_float>*    modelSVs;
                cl_mem          clModelSVs;
                cl_mem          clModelRHO;
                cl_mem          clModelSVCoefs;
                cl_mem          clModelLabel;
                core::util::Matrix<cl_float>*       svCoefs;
                cl_mem          clModelNsv;
                cl_mem          clPredictResults;
                cl_float*       modelRHOs;
                /**
                 * Creates OpenCL memory structures needs for overall process
                 * @param parameters
                 * Parameters for each pixel of image
                 * @param model
                 * Precalculated SVM prediction model
                 */
                void createBuffers(const core::util::Matrix<float>* parameters, svm_model* model);
                /**
                 * Passes parameters to OpenCL kernel function
                 * @param pixelCount
                 * number of pixels in image
                 * @param paramsPerPixel
                 * number of parameters per pixel
                 * @param model
                 * precalculated libsvm model
                 */
                void setKernelArgs(uint pixelCount, uint paramsPerPixel, svm_model* model);                
            protected:
                /**
                 * constructor, please see base class constructor
                 */
                OpenCLToolsPredict();
                /**
                 * @return 
                 * returns class name, the same as specified in config
                 */
                virtual std::string getClassName();
            public:                
                /**
                 * Please see base class destructor
                 */
                virtual ~OpenCLToolsPredict();
                /**
                 * initialize variables requested for overall process
                 */
                virtual void initVars();
                /**
                 * initialize variables used in per image calculations
                 */
                virtual void initWorkVars();
                /**
                 * deallocate variables used in overall process
                 */
                virtual void cleanUp();
                /**
                 * deallocates variables used in per image calculations
                 */
                virtual void cleanWorkPart();
                /**
                 * 
                 * @param model
                 * precalculated libsvm model
                 * @param parameters
                 * parameters per each pixel
                 * @return 
                 * predicted values for each pixel
                 */
                uchar* predict( svm_model* model, const core::util::Matrix<float>* parameters);
                /**
                 * call when libsvm model is changed
                 */
                void markModelChanged();
            };
            
        }
    }
}

#endif

#endif
