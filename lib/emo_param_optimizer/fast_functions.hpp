#define ALPHA_ONE_PERCENT 0.01
extern int split_oemd(const std::vector<double> & vals, const std::vector<std::vector<double>>& mask, const std::vector<int> & siftings,  std::vector<std::vector<double> > & oemd_levels);
extern int  make_trainings(std::vector< std::vector<double>>&oemd_levels, std::vector<size_t>&test_positions, size_t six_minutes,   size_t ahead , size_t back_skip,size_t back_size, std::vector<std::vector<double>> & xtrain,std::vector<std::vector<double>>&ytrain,
std::vector<double>& trimmed_mean_vals,
std::vector<double>& y_mean_val);
extern int find_range(double alpha, std::vector<double>::iterator begin, std::vector<double>::iterator end , double & left_border, double & right_border,double & mean_val);
extern double msecs();

extern int opti_param(int numtries, const std::vector<std::vector<double>> & xtrain , const std::vector<std::vector<double>> & ytrain,int the_level,int adjacent_left, int adjacent_right,
std::vector<double>parameters);
extern void gpu_blas_mmul(const double *A, const double *B, double *C, const int m, const int k, const int n) ;
extern double score_distance_kernel(size_t sizeX, double*Z_distances, double*Y);
extern int compute_cos_sin(const  std::vector<double> & omega, std::vector<double>& phase_cos, std::vector<double>& phase_sin, double step);
extern int step_decompose_matrix(const std::vector<double>& phase_cos, const std::vector<double>& phase_sin,size_t len,  const double *values, const std::vector<double>& previous, std::vector<double>& decomposition);
