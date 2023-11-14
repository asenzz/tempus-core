#ifndef __OPENCL_BASE_H__
#define __OPENCL_BASE_H__

#ifdef _OPENCL

#include <CL/cl.h>
#include "typedefs.h"
#include <vector>

namespace core{
    namespace opencl{
        
        /**
         * helper struct, translation of libsvm structure
         */
        struct cl_svm_node{
            cl_int index;
            cl_double value;
        };
        
        /**
         * helper struct, translation of libsvm structure with float instead of double value
         */
        struct cl_svm_node_float{
            cl_int index;
            cl_float value;
        };
        
        /**
         * Base class for opencl tools. All "OpenCL" classes should be derived from this one, and then specified in configuration file
         */
        class OpenClBase{
        private:
            std::string dirToOpenclprogramFiles;
            
            /**
             * compiles program file
             * @param programFileName
             * input file to compile
             * @return 
             * path to compiled file
             */
            std::string getBinaryFile(const std::string& programFileName);
            /**
             * calculate work group sizes for each kernel
             */
            void createWorkGroupSizes();
            /**
             * create openCL kernels from program
             */
            void createKernels();
            /**
             * get kernel function names used in class from config
             * @return 
             */
            std::vector<std::string> getKernelNamesForClass();
            /**
             * return list of source file for class specified in config
             * @return 
             */
            std::vector<std::string> getProgramFilesForClass();
        protected:
            /**
             * Device used in calculations
             */
            cl_device_id device;
            /**
             * OpenCL error code
             */
            cl_int err;
            /**
             * OpenCL command cue for program
             */
            cl_command_queue command_queue;
            /**
             * OpenCL program
             */
            cl_program program;
            /**
             * OpenCL program context
             */
            cl_context context;
            //kernel connected variables
            /**
             * array of all kernels used in class
             */
            cl_kernel* kernel;
            /**
             * max workgroup thread number, for kernel on device
             */
            size_t* workGroupSize;
            /**
             * number of kernels used in class
             */
            int kernelCount;
            //class variables
            /**
             * if initialized or not
             */
            bool initialized;
            
            /**
             * check for openCL error
             * @param err
             * OpenCL error code
             * @param err_code
             * Text describing error 
             */
            void err_check(int err, std::string err_code) throw (SDException&);
            /**
             * uses for get number of global work size
             * @param localSize
             * OpenCL workgroup size
             * @param allSize
             * number of all threads
             * @return 
             */
            size_t shrRoundUp(size_t localSize, size_t allSize);
            /**
             * global function for load program
             * @param kernelFileName
             * kernel file name 
             */
            void loadProgramFile(const std::string& programFileName);
            /**
             * load OpenCL program from source
             * @param kernelFileName
             * path to kernel source file
             */
            void loadProgramFileFromSource(const std::string& programFileName);
            /**
             * load OpenCL program from precompiled binary
             * @param kernelFileName
             * path to precompiled kernel source
             * @return
             * true if source exists and can be loaded, otherwise false 
             */
            bool loadProgramFromBinary(const std::string& programFileName);
            /**
             * saves compiled OpenCL program binary loaded from source
             * @param programFileName
             * kernel file name
             * @return
             * path where compiled binaries are saved 
             */
            char* saveProgramBinary(const std::string& programFileName);
            /**
             * 
             * @return 
             * class names used in config
             */
            virtual std::string getClassName() = 0;
        public:
            /**
             * constructor. Calls initVars()
             */
            OpenClBase();
            /**
             * destructor call cleanUp()
             */
            virtual ~OpenClBase();                        
            /**
             * init global openCL variables
             */
            virtual void initVars();
            /**
             * init openCL variables necessary for one image processing
             */
            virtual void initWorkVars() = 0;
            /**
             * clean up global variables
             */
            virtual void cleanUp();
            /**
             * clean up variables used for single image processing
             */
            virtual void cleanWorkPart() = 0;
            /**
             * @return 
             * if init() method was called;
             */
            bool hasInitialized();
            /**
             * init variables for OpenclTools class instances
             * @param platformID
             * wanted openCL platformID
             * @param deviceID
             * wanted openCL deviceID
             * @param listOnly
             * if true then method will only list all possible platforms and devices
             */
            void init(uint platformID, uint deviceID, bool listOnly) throw (SDException&);
        };
        
    }
}

#endif

#endif
