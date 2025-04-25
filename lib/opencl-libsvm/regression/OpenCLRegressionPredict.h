#ifndef __OPENCL_REGRESSION_PREDICT_H__
#define __OPENCL_REGRESSION_PREDICT_H__

#include <vector>
#include "core/opencl/OpenClToolsBase.h"
#include "core/util/Singleton.h"
#include "core/util/Matrix.h"

#ifdef _OPENCL
namespace core{
    namespace opencl{
        namespace regression{
            
            /**
             * opencl implementation of regression based prediction
             */
            class OpenCLRegressionPredict : public OpenClBase, public core::util::Singleton<OpenCLRegressionPredict>{
                friend class core::util::Singleton<OpenCLRegressionPredict>;
            private:
                /**
                 * constructor, please see documents of base class constructor
                 */
                OpenCLRegressionPredict();
                /**
                 * initialize cl memory structures used in calculations
                 */
                void createBuffers() throw(SDException&);
                /**
                 * passing parameters to kernel function
                 */
                void setKernelArgs() throw(SDException&);
                const core::util::Matrix<float>* pixelParameters;
                cl_mem pixelParametersBuff;                
                cl_mem predictedBuff;
                
                uint parameterCount;
                uint pixelCount;                
                
                float* regressionCoefs;
                cl_mem regressionCoefsBuff;
                ulong regressionCoefsNum;
                float borderValue;
            protected:                                
                /**
                 * overriden 
                 * @return
                 * class name later used in config 
                 */
                virtual std::string getClassName();
            public:
                /**
                 * destructor please see documentation of base class destructor
                 */
                virtual ~OpenCLRegressionPredict();
                /**
                 * initialize variables used in per image calculations
                 */
                virtual void initWorkVars();
                /**
                 * frees and deinitialize variables used in per image calculations
                 */
                virtual void cleanWorkPart();
                /**
                 * overriden, calls base class initVars()
                 */
                virtual void initVars();
                /**
                 * overriden, calls base class cleanUp()
                 */
                virtual void cleanUp();
                /**
                 * 
                 * @param imagePixelParameters
                 * pixel parameters for each pixel in image
                 * @param pixelCount
                 * number of pixels in image
                 * @param parameterCount
                 * number of parameters per pixel
                 * @param coefs
                 * coefficients used in prediction, number is number of parameters per pixel + 1
                 * @param borderValue
                 * border value which determines if pixel is 0 or 1
                 * @return
                 * predicted values per each pixel 
                 */
                uchar* predict( const core::util::Matrix<float>& imagePixelParameters,
                                const int& pixelCount, const int& parameterCount,
                                std::vector<float> coefs, float borderValue);
            };

        }
    }
}
#endif

#endif
