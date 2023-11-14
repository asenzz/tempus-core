#ifndef __OPENCLTOOLS_TRAIN_H__ 
#define __OPENCLTOOLS_TRAIN_H__

#ifdef _OPENCL

#include "core/opencl/OpenClToolsBase.h"
#include "core/util/Singleton.h"

struct svm_node;

namespace core{
    
    namespace util{
        template<typename T> class Matrix;
    }
    
    namespace opencl{
        namespace libsvm{
            /**
             * Class for parallelized libsvm calculations
             */
            class OpenCLToolsTrain : public core::opencl::OpenClBase, public core::util::Singleton<OpenCLToolsTrain>{
                friend class core::util::Singleton<OpenCLToolsTrain>;
            private:
                cl_mem clData;
                int clDataLen;
                cl_mem clY;
                cl_mem clX;
                cl_mem clXSquared;            
                bool newTask;
                bool newSelectWorkingSet;
                core::util::Matrix<double>* xMatrix;

                cl_mem clGradDiff;
                cl_mem clObjDiff;
                cl_mem clAlphaStatus;
                cl_mem clYSelectWorkingSet;
                cl_mem clG;
                cl_mem clQD;
                cl_mem clQI;
                
                /**
                 * Creates openCL memory structures used in process
                 * @param data
                 * @param dataLen
                 * @param i
                 * @param y
                 * @param yLen
                 * @param x
                 * @param start
                 * @param steps
                 * @param clDataChanged
                 * @param xSquared
                 */
                void createBuffersSVM(  float* data, int dataLen, int i,
                                    char* y, int yLen,
                                    core::util::Matrix<svm_node>* x,
                                    int start, int steps, bool& clDataChanged,
                                    double* xSquared);
                /**
                 * Passes parameters to OpenCL kernel function, for svc calculations
                 * @param start
                 * @param len
                 * @param i
                 * @param kernel_type
                 * @param xW
                 * @param dataLen
                 * @param gamma
                 * @param coef0
                 * @param degree
                 * @param clDataChanged
                 */
                void setKernelArgsSVC( cl_int start, cl_int len, cl_int i, 
                                        cl_int kernel_type, cl_int xW, cl_int dataLen,
                                        cl_double gamma, cl_double coef0, cl_int degree,
                                        bool clDataChanged);
                /**
                 * passes parameters to OpenCL kernel function for svr calculations
                 * @param start
                 * @param len
                 * @param i
                 * @param kernel_type
                 * @param xW
                 * @param dataLen
                 * @param gamma
                 * @param coef0
                 * @param degree
                 * @param clDataChanged
                 */
                void setKernelArgsSVR(  cl_int start, cl_int len, cl_int i, 
                                        cl_int kernel_type, cl_int xW, cl_int dataLen,
                                        cl_double gamma, cl_double coef0, cl_int degree,
                                        bool clDataChanged); 
                /**
                 * creates openCL memory structures used in working set part parallelization
                 * @param activeSize
                 * @param grad_diff
                 * @param obj_diff
                 * @param alpha_status
                 * @param l
                 * @param y
                 * @param G
                 * @param QD
                 * @param Q_i
                 */
                void createBuffersWorkingSet(const int& activeSize, double* grad_diff,
                                            double* obj_diff, const char* alpha_status, 
                                            const int& l, const char* y, const double* G,
                                            const double* QD, const float* Q_i);
                /**
                 * Passes parameters to OpenCL kernel function
                 * @param activeSize
                 * @param i
                 * @param Gmax
                 */
                void setKernelArgsWorkingSet(const int& activeSize, const int& i,
                                            const double& Gmax);
                
            protected:
                /**
                 * Constructor, calls constructor of base class
                 */
                OpenCLToolsTrain();
                /**
                 * 
                 * @return 
                 * class name used in config
                 */
                virtual std::string getClassName();
            public:
                /**
                 * destructor, please see base class destructor documentation
                 */
                virtual ~OpenCLToolsTrain();
                /**
                 * overrides base class initVars(). Initialize all variables for overrall process
                 */
                virtual void initVars();
                /**
                 * Initializes variables needed for one iteration
                 */
                virtual void initWorkVars();
                /**
                 * Overrides base class cleanUp().
                 */
                virtual void cleanUp();
                /**
                 * Clean variables needed for one iteration
                 */
                virtual void cleanWorkPart();
                /**
                 * parallel get_Q libsvm function
                 * @param data
                 * @param dataLen
                 * @param start
                 * @param len
                 * @param i
                 * @param kernel_type
                 * @param y
                 * @param yLen
                 * @param x
                 * @param classType
                 * @param gamma
                 * @param coef0
                 * @param degree
                 * @param xSquared
                 */
                void get_Q( float* data, int dataLen, int start, int len, int i, int kernel_type, 
                        char* y, int yLen, core::util::Matrix<svm_node>* x, LIBSVM_CLASS_TYPE classType,
                        double gamma, double coef0, int degree, double *xSquared) throw (SDException);
                /**
                 * parallel working set part function
                 * @param activeSize
                 * @param i
                 * @param y
                 * @param alpha_status
                 * @param l
                 * @param grad_diff
                 * @param Gmax
                 * @param G
                 * @param QD
                 * @param Q_i
                 * @param obj_diff
                 */
                void selectWorkingSet(  const int& activeSize, const int& i, const char* y,
                                        const char* alpha_status, const int& l, double* grad_diff,
                                        const double& Gmax, const double* G, const double* QD, 
                                        const float* Q_i, double* obj_diff) throw (SDException);

                int64_t durrData;
                int64_t durrSetSrgs;
                int64_t durrBuff;
                int64_t durrExec;
                int64_t durrReadBuff;
            };
            
        } 
    }
}

#endif
#endif
