#ifdef _OPENCL
#include "OpenClToolsBase.h"
#include <iostream>
#include "core/util/raii/RAIIS.h"
#include "core/util/MemTracker.h"
#include "core/util/Config.h"

#define MAX_DEVICES 100
#define MAX_SRC_SIZE 5242800
#define MAX_PLATFORMS 100

namespace core{
    namespace opencl{
        
        using namespace std;
        using namespace core::util::raii;
        using namespace core::util;
        
        OpenClBase::OpenClBase(){
        }
        
        OpenClBase::~OpenClBase(){
            cleanUp();
        }
        
        void OpenClBase::initVars(){
            initialized     = false;
            kernel          = 0;
            workGroupSize   = 0;
            kernelCount     = 0;
            program         = 0;
            context         = 0;
            command_queue   = 0;
        }
        
        void OpenClBase::cleanUp(){            
            if (program)
                clReleaseProgram(program);
            if (context)
                clReleaseContext(context);
            if (command_queue)
                clReleaseCommandQueue(command_queue);
            
            if (kernel){
                for (int i = 0; i < kernelCount; i++) {
                    if (kernel[i]){
                        clReleaseKernel(kernel[i]);
                        err_check(err, "OpenClBase::cleanUp clReleaseKernel");
                    }
                }
                DeleteArr(kernel);
            }
            if (workGroupSize){
                DeleteArr(workGroupSize);
            }
        }
        
        bool OpenClBase::hasInitialized(){
            return initialized;
        }
        
        void OpenClBase::err_check(int err, string err_code) throw (SDException&) {
            if (err != CL_SUCCESS) {
                cout << "Error: " << err_code << "(" << err << ")" << endl;
                if (err == CL_BUILD_PROGRAM_FAILURE) {
                    // Determine the size of the log
                    size_t log_size;
                    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
                    // Allocate memory for the log
                    char* log = New char[log_size];
                    VectorRaii<char> vraii(log);
                    // Get the log
                    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
                    // Print the log
                    cout << log << endl;
                    
                }
                SDException exc(SHADOW_OTHER, err_code);
                throw exc;
            }
        }
        
        size_t OpenClBase::shrRoundUp(size_t localSize, size_t allSize) {
            if (allSize % localSize == 0) {
                return allSize;
            }
            int coef = allSize / localSize;
            return ((coef + 1) * localSize);
        }
        
        void OpenClBase::loadProgramFile(const string& programFileName){
            string usePrecompiledStr = Config::getInstancePtr()->getPropertyValue("general.openCL.UsePrecompiledKernels");
            bool usePrecompiled = usePrecompiledStr.compare("true") == 0;
            if (usePrecompiled){
                bool succ = loadProgramFromBinary(programFileName);
                if (succ){
                    return;
                }
            }
            loadProgramFileFromSource(programFileName);
            if (usePrecompiled){
                char* fp = saveProgramBinary(programFileName);
                if (fp != 0){
                    remove(fp);
                    Delete(fp);
                }
            }
        }
        
        void OpenClBase::loadProgramFileFromSource(const string& programFileName){
            fstream kernelFile;
            string file = dirToOpenclprogramFiles + "/" + programFileName + ".cl";
            kernelFile.open(file.c_str(), ifstream::in);
            FileRaii fRaii(&kernelFile);
            if (kernelFile.is_open()) {
                char* buffer = 0;
                buffer = New char[MAX_SRC_SIZE];
                if (buffer) {
                    VectorRaii<char> vraiiBuff(buffer);
                    kernelFile.read(buffer, MAX_SRC_SIZE);
                    if (kernelFile.eof()) {
                        size_t readBytes = kernelFile.gcount();
                        program = clCreateProgramWithSource(context, 1, (const char **) &buffer, &readBytes, &err);
                        err_check(err, programFileName + " clCreateProgramWithSource");
                        cout << "Build program: " << programFileName << " started" << endl;
                        err = clBuildProgram(program, 1, &device, OCL_BUILD_OPTIONS, NULL, NULL);
                        err_check(err, programFileName + " clBuildProgram");
                        cout << "Build program: " << programFileName << " finished" << endl;
                    }                    
                } else {                    
                    SDException exc(SHADOW_NO_MEM, "Init Program: " + programFileName);
                    throw exc;
                }                
            } else {
                SDException exc(SHADOW_READ_UNABLE, "Init Program: " + programFileName);
                throw exc;
            }
        }
        
        string OpenClBase::getBinaryFile(const string& programFileName){
            cl_device_type type;
            clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(cl_device_type), &type, 0);
            char deviceName[256];
            err = clGetDeviceInfo(device, CL_DEVICE_NAME, 256, deviceName, 0);            
            err_check(err, programFileName + " Get device name, save binary kernel");            
            string file = deviceName;
            file += "_" + programFileName + ".ptx";
            return file;
        }
        
        bool OpenClBase::loadProgramFromBinary(const string& kernelFileName){
            fstream kernelFile;            
            string file = getBinaryFile(kernelFileName);
            kernelFile.open(file.c_str(), ifstream::in | ifstream::binary);
            FileRaii fRaii(&kernelFile);
            if (kernelFile.is_open()) {
                char* buffer = 0;
                buffer = New char[MAX_SRC_SIZE];
                if (buffer) {
                    VectorRaii<char> vRaiiBuff(buffer);
                    kernelFile.read(buffer, MAX_SRC_SIZE);
                    if (kernelFile.eof()) {
                        size_t readBytes = kernelFile.gcount();
                        program = clCreateProgramWithBinary(context, 1, &device, &readBytes, (const uchar**)&buffer, 0, &err);
                        try{
                            err_check(err, kernelFileName + " clCreateProgramWithBinary");
                        }
                        catch (SDException& e){
                            cout << e.what() << endl;
                            return false;
                        }
                        err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
                        try{
                            err_check(err, kernelFileName + " clBuildProgram");
                        }
                        catch (SDException& e){
                            cout << e.what() << endl;
                            return false;
                        }
                    }                    
                } else {                    
                    return false;
                }                
            } else {
                return false;
            }
            return true;
        }
        
        char* getCStrCopy(string str){
            char* ret = New char[str.size() + 1];
            ret[str.size()] = '\0';
            strcpy(ret, str.c_str());
            return ret;
        }
        
        char* OpenClBase::saveProgramBinary(const string& kernelFileName){            
            fstream kernel;
            string programFile = getBinaryFile(kernelFileName);
            kernel.open(programFile.c_str(), ofstream::out | ofstream::binary);            
            if (kernel.is_open()){
                FileRaii fRaii(&kernel);
                cl_uint nb_devices;
                size_t retBytes;
                err = clGetProgramInfo(program, CL_PROGRAM_NUM_DEVICES, sizeof(cl_uint), &nb_devices, &retBytes);
                try{
                    err_check(err, kernelFileName + " OpenClBase::saveKernelBinary: clGetProgramInfo");
                }
                catch (SDException& e){
                    cout << e.what() << endl;
                    return getCStrCopy(programFile);
                }
                
                size_t* binarySize = 0;
                binarySize = New size_t[nb_devices];
                if (binarySize == 0){
                    SDException exc(SHADOW_NO_MEM, "Get binary sizes");
                    cout << exc.what() << endl;
                    return getCStrCopy(programFile);
                }
                VectorRaii<size_t> bsRaii(binarySize);
                err = clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES, sizeof(size_t) * nb_devices, binarySize, 0);
                try{
                    err_check(err, kernelFileName + " OpenClBase::saveKernelBinary: clGetProgramInfo2");
                }
                catch (SDException& e){
                    cout << e.what() << endl;
                    return getCStrCopy(programFile);
                }
                uchar**  buffer = 0;
                buffer = New uchar*[nb_devices];
                if (buffer != 0){                    
                    for (uint i = 0; i < nb_devices; i++){
                        buffer[i] = New uchar[binarySize[i]]; 
                    }
                    MatrixRaii<uchar> mRaii(buffer, nb_devices);
                    
                    size_t read;
                    err = clGetProgramInfo(program, CL_PROGRAM_BINARIES, sizeof(uchar*)*nb_devices, buffer, &read);
                    try{
                        err_check(err, kernelFileName + " OpenClBase::saveKernelBinary: clGetProgramInfo3");
                    }
                    catch (SDException& e){
                        cout << e.what() << endl;
                        return getCStrCopy(programFile);
                    }
                    //because I know that is on one device
                    kernel.write((const char*)buffer[0], binarySize[0]);
                }
                else{
                    SDException exc(SHADOW_NO_MEM, "Save binary kernel");
                    cout << exc.what() << endl;
                    return getCStrCopy(programFile);
                }                
            }
            else{
                SDException exc(SHADOW_WRITE_UNABLE, "Save binary kernel");
                cout << exc.what() << endl;
                return getCStrCopy(programFile);
            }
            return 0;
        }
        
        void OpenClBase::createWorkGroupSizes() {
            workGroupSize = New size_t[kernelCount];
            for (int i = 0; i < kernelCount; i++) {
                if (kernel[i]){
                    err = clGetKernelWorkGroupInfo(kernel[i], device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(workGroupSize[i]), &(workGroupSize[i]), NULL);
                    err_check(err, "penClBase::createWorkGroupSizes clGetKernelWorkGroupInfo");
                }
            }
        }
        
        vector<string> OpenClBase::getKernelNamesForClass(){
            vector<string> retVec;
            Config* conf = Config::getInstancePtr();
            string className = getClassName();
            string keyStr = "general.classes." + className + ".kernels";
            string k1 = keyStr + ".kernelCount";            
            string kerCountStr = conf->getPropertyValue(k1);
            int count = atoi(kerCountStr.c_str());
            for (int i = 0; i < count; i++){
                char index[4];
                sprintf(index, "%d", i);
                k1 = keyStr + ".kernelNo" + index;
                string kernelName = conf->getPropertyValue(k1);
                retVec.push_back(kernelName);
            }
            return retVec;
        }
        
        vector<string> OpenClBase::getProgramFilesForClass(){
            vector<string> retVec;
            Config* conf = Config::getInstancePtr();
            string className = getClassName();
            string keyStr = "general.classes." + className + ".programs.programFile";
            string programFile = conf->getPropertyValue(keyStr);
            retVec.push_back(programFile);
            keyStr = "general.classes." + className + ".programs.rootDir";
            dirToOpenclprogramFiles = conf->getPropertyValue(keyStr);
            return retVec;
        }
        
        void OpenClBase::createKernels(){
            vector<string> kernelNames = getKernelNamesForClass();
            kernelCount = kernelNames.size();
            kernel = New cl_kernel[kernelCount];
            for (int i = 0; i < kernelCount; i++){
                kernel[i] = clCreateKernel(program, kernelNames[i].c_str(), &err);
                err_check(err, "OpenClBase::createKernels clCreateKernel: " + kernelNames[i]);
            }            
        }
        
        void OpenClBase::init(uint platformID, uint deviceID, bool listOnly) throw (SDException&) {
            char info[256];
            cl_platform_id platform[MAX_PLATFORMS];
            cl_uint num_platforms;                        
            
            err = clGetPlatformIDs(MAX_PLATFORMS, platform, &num_platforms);
            err_check(err, "OpenclTools::init clGetPlatformIDs");
            cout << "Found " << num_platforms << " platforms." << endl;                        
            cout << "=============" << endl;
            for (uint i = 0; i < num_platforms; i++) {
                cl_device_id devices[MAX_DEVICES];
                cl_uint num_devices;
                err = clGetPlatformInfo(platform[i], CL_PLATFORM_NAME, 256, info, 0);
                err_check(err, "OpenclTools::init clGetPlatformInfo");
                cout << "Platform name: " << info << endl;
                try {
#if defined _AMD
                    err = clGetDeviceIDs(platform[i], CL_DEVICE_TYPE_ALL, MAX_DEVICES, devices, &num_devices);
#else
                    err = clGetDeviceIDs(platform[i], CL_DEVICE_TYPE_GPU, MAX_DEVICES, devices, &num_devices);
#endif
                    err_check(err, "OpenclTools::init clGetDeviceIDs");
                    cout << "Found " << num_devices << " devices" << endl;

                    for (uint j = 0; j < num_devices; j++) {
                        err = clGetDeviceInfo(devices[j], CL_DEVICE_NAME, 256, info, 0);
                        err_check(err, "OpenclTools::init clGetDeviceInfo CL_DEVICE_NAME");
                        cl_device_type type;
                        err = clGetDeviceInfo(devices[j], CL_DEVICE_TYPE, sizeof(cl_device_type), &type, 0);
                        err_check(err, "OpenclTools::init clGetDeviceInfo CL_DEVICE_TYPE");
                        string typeStr = "DEVICE_OTHER";
                        if (type == CL_DEVICE_TYPE_CPU)
                            typeStr = "DEVICE_CPU";
                        else if (type == CL_DEVICE_TYPE_GPU)
                            typeStr = "DEVICE_GPU";
                        cl_ulong maxAllocSize;
                        err = clGetDeviceInfo(devices[j],  CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(cl_ulong), &maxAllocSize, 0);
                        err_check(err, "OpenclTools::init clGetDeviceInfo CL_DEVICE_MAX_MEM_ALLOC_SIZE");
                        cout << "Device " << j << " name: " << info << " type: " << typeStr << " max alloc: " << maxAllocSize << endl;
                    }
                }                
                catch (SDException& exception) {
                    cout << "Platform not supported by this build" << endl;
                    cout << exception.what() << endl;
                }
                cout << "=============" << endl;
            }
            
            if (listOnly)
                return;
            
            if (platformID >= num_platforms){
                SDException exc(SHADOW_NO_OPENCL_PLATFORM, "Init platform");
                throw exc;
            }
            
            cl_device_id devices[MAX_DEVICES];
            cl_uint num_devices;
#if defined _AMD
                err = clGetDeviceIDs(platform[platformID], CL_DEVICE_TYPE_ALL, MAX_DEVICES, devices, &num_devices);
#else
                err = clGetDeviceIDs(platform[platformID], CL_DEVICE_TYPE_GPU, MAX_DEVICES, devices, &num_devices);
#endif
            err_check(err, "OpenClBase::init clGetDeviceIDs2");
            if (deviceID >= num_devices){
                SDException exc(SHADOW_NO_OPENCL_DEVICE, "OpenClBase::init Init devices");
                throw exc;
            }
            device = devices[deviceID];
                        
            context = clCreateContext(0, 1, &device, NULL, NULL, &err);
            err_check(err, "OpenClBase::init clCreateContext");
            command_queue = clCreateCommandQueue(context, device, 0, &err);
            err_check(err, "OpenClBase::init clCreateCommandQueue");            

            cl_bool sup;
            size_t rsize;
            clGetDeviceInfo(device, CL_DEVICE_IMAGE_SUPPORT, sizeof (sup), &sup, &rsize);
            if (sup != CL_TRUE) {
                SDException exception(SHADOW_IMAGE_NOT_SUPPORTED_ON_DEVICE, "OpenClBase::init Check for image support");
                throw exception;
            }            
            //image processing section
            vector<string> programFile = getProgramFilesForClass();
            loadProgramFile(programFile[0]);
            createKernels();
            createWorkGroupSizes();
            initialized = true;            
        }
        
    }
}
#endif
