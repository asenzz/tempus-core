#define __CL_ENABLE_EXCEPTIONS

#include "CL/cl.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <tuple>

using FloatType = float;
using MatrixSize = std::pair<std::size_t, std::size_t>;

template <class T>
cl::Buffer copyVectorToDevice(cl::CommandQueue &queue,
                              cl::Context &context,
                              const std::vector<T> &vector)
{
    cl::Buffer dWeights(context, CL_MEM_READ_WRITE, sizeof(T) * vector.size());
    queue.enqueueWriteBuffer(dWeights, CL_TRUE, 0, sizeof(T) * vector.size(), vector.data());
    return dWeights;
}

template <class T>
std::vector<T> copyBufferToHost(cl::CommandQueue &queue,
                                cl::Buffer &buffer,
                                unsigned size)
{
    std::vector<T> vector(size);
    auto err = queue.enqueueReadBuffer(buffer, CL_TRUE, 0, sizeof(T) * size, vector.data());
    return vector;
}

template <class Container>
class SimpleExecute
{
public:
    SimpleExecute(
            cl::Program &program, cl::CommandQueue &queue,
            Container &container, cl::Buffer &real, const MatrixSize &matrixSize):
        _program(program), _queue(queue), _container(container), _real(real), _matrixSize(matrixSize)
    {
        _container.prepareKernel(program);
        _exampleGrad = cl::Kernel(program, "example_grad");
        _exampleGrad.setArg(0, _container.weights);
        _exampleGrad.setArg(1, _real);
        _exampleGrad.setArg(2, _container.grad);
    }

    void calculateFakeGrad()
    {
        _queue.enqueueNDRangeKernel(_exampleGrad, cl::NullRange, cl::NDRange(_matrixSize.first, _matrixSize.second), cl::NullRange);
//        _queue.flush();
    }

    void iteration()
    {
        ++_iterations;
        calculateFakeGrad();
        _container.enqueueKernel(_queue, _matrixSize);
//        _queue.flush();
        if(_iterations % 10 == 0)
        {
            std::vector<FloatType> vec = copyBufferToHost<FloatType>(
                _queue, _container.weights, _matrixSize.first * _matrixSize.second);
            for(unsigned i = 0; i < vec.size(); ++i)
            {
                std::cout << vec[i] << " ";
            }
            std::cout << std::endl;
        }
    }

private:
    cl::Program &_program;
    cl::CommandQueue &_queue;
    Container &_container;
    cl::Buffer &_real;
    const MatrixSize &_matrixSize;
    cl::Kernel _exampleGrad;
    unsigned _iterations = 0;
};

struct SGDContainer
{
    void prepareKernel(cl::Program &program)
    {
        kernel = cl::Kernel(program, "sgd");
        kernel.setArg(0, weights);
        kernel.setArg(1, grad);
        kernel.setArg(2, alpha);
    }

    void enqueueKernel(cl::CommandQueue &queue, const MatrixSize &matrixSize)
    {
        queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(matrixSize.first, matrixSize.second), cl::NullRange);
    }

    cl::Buffer weights;
    cl::Buffer grad;
    FloatType alpha = 0;

    cl::Kernel kernel;
};

struct MomentumContainer
{
    void prepareKernel(cl::Program &program)
    {
        kernel = cl::Kernel(program, "momentum");
        kernel.setArg(0, weights);
        kernel.setArg(1, grad);
        kernel.setArg(2, velocity);
        kernel.setArg(3, alpha);
        kernel.setArg(4, gamma);
    }

    void enqueueKernel(cl::CommandQueue &queue, const MatrixSize &matrixSize)
    {
        queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(matrixSize.first, matrixSize.second), cl::NullRange);
    }

    cl::Buffer weights;
    cl::Buffer grad;
    cl::Buffer velocity;
    FloatType alpha = 0;
    FloatType gamma = 0;

    cl::Kernel kernel;
};

struct AdamContainer
{
    void prepareKernel(cl::Program &program)
    {
        kernel = cl::Kernel(program, "adam");
        kernel.setArg(0, weights);
        kernel.setArg(1, grad);
        kernel.setArg(2, momentum);
        kernel.setArg(3, momentum);
        kernel.setArg(4, alpha);
        kernel.setArg(5, beta1);
        kernel.setArg(6, beta2);
        kernel.setArg(7, iteration);
    }

    void enqueueKernel(cl::CommandQueue &queue, const MatrixSize &matrixSize)
    {
        ++iteration;
        kernel.setArg(7, iteration);
        queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(matrixSize.first, matrixSize.second), cl::NullRange);
    }

    cl::Buffer weights;
    cl::Buffer grad;
    cl::Buffer momentum;
    cl::Buffer rate;
    FloatType alpha = 0;
    FloatType beta1 = 0;
    FloatType beta2 = 0;
    unsigned iteration = 0;

    cl::Kernel kernel;
};


void loadSource(const std::string &fname, std::string &source)
{
    std::ifstream t(fname.c_str());
    std::stringstream ss;
    ss << t.rdbuf();
    source = ss.str();
}

void loadSources(std::vector<std::string> &rawSources)
{
    std::string source;
    loadSource("./gradcl.cl", source);
    rawSources.push_back({source.c_str(), source.size()});
}

int main()
{
    std::vector<cl::Platform> all_platforms;
    cl::Platform::get(&all_platforms);
    if(all_platforms.empty())
    {
        std::cout<< "No platforms" << std::endl;
        exit(1);
    }

    for(auto& platform : all_platforms)
    {
        std::cout << platform.getInfo<CL_PLATFORM_NAME>() << std::endl;
    }

    cl::Platform default_platform=all_platforms[0];

    std::vector<cl::Device> all_devices;
    default_platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);

    if(all_devices.empty())
    {
        std::cout<< "No devices" << std::endl;
        exit(2);
    }

    cl::Device default_device = all_devices[0];
    std::cout<< "Using device: " << default_device.getInfo<CL_DEVICE_NAME>() << std::endl;

    cl::Context context(default_device);
    std::vector<std::string> rawSources;
    loadSources(rawSources);

    cl::Program::Sources sources;
    for(const std::string &rawSource : rawSources)
    {
        sources.push_back({rawSource.c_str(), rawSource.size()});
    }
    cl::Program program(context, sources);

    try
    {
        program.build({default_device});
    }
    catch (const cl::Error &e)
    {
        if (e.err() == CL_BUILD_PROGRAM_FAILURE)
        {
            // Get the build log
            std::string name = default_device.getInfo<CL_DEVICE_NAME>();
            std::string buildlog = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(default_device);
            std::cerr << "Build log for " << name << ":" << std::endl << buildlog << std::endl;
        }
        else
        {
            throw e;
        }
    }

    cl::CommandQueue queue(context, default_device);


    constexpr MatrixSize MATRIX_SIZES(10, 10);
    constexpr unsigned MATRIX_SIZE = MATRIX_SIZES.first * MATRIX_SIZES.second;
    const FloatType alpha = 1e-2;
    const FloatType gamma = 0.9;
    const FloatType beta1 = 0.9;
    const FloatType beta2 = 0.999;
    std::vector<FloatType> zeroMatrix(MATRIX_SIZE, 0.0);

    std::vector<FloatType> hWeights(MATRIX_SIZE, 0.0);
    for(unsigned i = 0; i < MATRIX_SIZE; ++i)
    {
        hWeights[i] = i;
    }
    cl::Buffer real = copyVectorToDevice(queue, context, hWeights);

    SGDContainer sgd;
    sgd.alpha = alpha;
    sgd.weights = copyVectorToDevice(queue, context, zeroMatrix);
    sgd.grad = copyVectorToDevice(queue, context, zeroMatrix);

    MomentumContainer momentum;
    momentum.alpha = alpha;
    momentum.gamma = gamma;
    momentum.velocity = copyVectorToDevice(queue, context, zeroMatrix);
    momentum.weights = copyVectorToDevice(queue, context, zeroMatrix);
    momentum.grad = copyVectorToDevice(queue, context, zeroMatrix);

    AdamContainer adam;
    adam.alpha = alpha;
    adam.beta1 = beta1;
    adam.beta2 = beta2;
    adam.iteration = 0;
    adam.momentum = copyVectorToDevice(queue, context, zeroMatrix);
    adam.rate = copyVectorToDevice(queue, context, zeroMatrix);
    adam.weights = copyVectorToDevice(queue, context, zeroMatrix);
    adam.grad = copyVectorToDevice(queue, context, zeroMatrix);

    SimpleExecute<SGDContainer> sgdExec(program, queue, sgd, real, MATRIX_SIZES);
    for(unsigned i = 0; i < 100; ++i)
    {
        sgdExec.iteration();
    }

    std::cout << "--------------------" << std::endl;

    SimpleExecute<MomentumContainer> momentumExec(program, queue, momentum, real, MATRIX_SIZES);
    for(unsigned i = 0; i < 100; ++i)
    {
        momentumExec.iteration();
    }

    std::cout << "--------------------" << std::endl;

    SimpleExecute<AdamContainer> adamExec(program, queue, adam, real, MATRIX_SIZES);
    for(unsigned i = 0; i < 100; ++i)
    {
        adamExec.iteration();
    }

    std::cout << "--------------------" << std::endl;
}
