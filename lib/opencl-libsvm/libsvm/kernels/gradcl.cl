void kernel multiplyByTwo(
    __global float *a/*, size_t nrows, size_t ncols*/)
{
    size_t row = get_global_id(0);
    size_t nrows = get_global_size(0);
    size_t col = get_global_id(1);
    size_t ncols = get_global_size(1);
    a[row * ncols + col] *= 2;
}

void kernel example_grad(__global float *weights, __global float *real, __global float *grad)
{
    size_t row = get_global_id(0);
    size_t nrows = get_global_size(0);
    size_t col = get_global_id(1);
    size_t ncols = get_global_size(1);

    size_t idx = row * ncols + col;
    grad[idx] = real[idx] - weights[idx];
}

void kernel sgd(
    __global float *weights, __global float *grad, float alpha)
{
    size_t row = get_global_id(0);
    size_t nrows = get_global_size(0);
    size_t col = get_global_id(1);
    size_t ncols = get_global_size(1);

    weights[row * ncols + col] += alpha * grad[row * ncols + col];
}

void kernel momentum(
    __global float *weights, __global float *grad, __global float *velocity,
    float alpha, float gamma)
{
    size_t row = get_global_id(0);
    size_t nrows = get_global_size(0);
    size_t col = get_global_id(1);
    size_t ncols = get_global_size(1);

    float v = velocity[row * ncols + col];
    v *= gamma;
    v += alpha * grad[row * ncols + col];
    velocity[row * ncols + col] = v;
    weights[row * ncols + col] += v;
}

void kernel adam(
    __global float *weights, __global float *grad,
    __global float *momentum, __global float *rate,
    float alpha, float beta1, float beta2,
    unsigned iteration)
{
    size_t row = get_global_id(0);
    size_t nrows = get_global_size(0);
    size_t col = get_global_id(1);
    size_t ncols = get_global_size(1);

    float g = grad[row * ncols + col];

    float m = momentum[row * ncols + col];
    m *= beta1;
    m += (1.0-beta1) * g;
    momentum[row * ncols + col] = m;

    float r = rate[row * ncols + col];
    r *= beta2;
    r += (1.0-beta2) * g;
    rate[row * ncols + col] = r;

    float mkhat = m / (1.0 - pow(beta1, iteration));
    float rkhat = r / (1.0 - pow(beta2, iteration));

    weights[row * ncols + col] += (alpha * mkhat) / (sqrt(rkhat) + 1e-12);
}
