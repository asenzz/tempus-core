#define _LARGEFILE64_SOURCE
#define _FILE_OFFSET_BITS 64

#include <cstdio>
#include <iostream>
#include <cinttypes>
#include <algorithm>
#include <vector>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include <iostream>
#include <algorithm>
#include <cmath>
#include <vector>
#include <cassert>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <unistd.h>
#include <fcntl.h>
#include <sys/time.h>
#include <omp.h>
#include <iostream>     // std::cout, std::fixed
#include <iomanip>
#include <string>       // std::string
#include <iostream>     // std::cout
#include <sstream>
#include <limits>


//#define USE_MPI

#ifdef USE_MPI

#include <mpi.h>

#endif
#define ARMA_NO_DEBUG
#define ARMA_ALLOW_FAKE_GCC

#include <armadillo>
#include <fstream>

//inverse normal cdf

#include <sstream>
#include <string>

//#include "kernel_fast.hpp"

#include "reper_dates.h"

#include "fast_functions.hpp"


double times[10];

double msecs()
{
    struct timespec start;
    long mtime, seconds, useconds;

    //gettimeofday(&start, NULL);
    clock_gettime(CLOCK_MONOTONIC, &start);
    seconds = start.tv_sec;
    useconds = start.tv_nsec;

    double stime = ((seconds) * 1 + (double) useconds / 1000000000.0);


    return stime;
}

int read_data_real(std::vector<double> &vals, std::vector<int> &year, std::vector<int> &month, std::vector<int> &day, std::vector<int> &hour, std::vector<int> &minute, std::vector<int> &second,
                   std::vector<uint64_t> &utimes,
                   std::vector<int> &hour_positions, std::vector<int> &reper_positions)
{
    char file_int[] = {"/users/eatanassov/goalpha/full/simplify_fullv.bin"};
    int input_manip = open(file_int, O_RDONLY);
    if (input_manip < 0) {
        abort();
    }
    struct stat file_stat;
    int rc = fstat(input_manip, &file_stat);
    if (rc != 0) {
        perror("fstat failed or file is not a regular file");
        return EXIT_FAILURE;
    }
    const off_t if_size = file_stat.st_size;
    const size_t multiple = (1 + 1 + 1 + 1 + 1 + 1 + 1);
    size_t data_size = if_size / (sizeof(double) + sizeof(double) * multiple);

    std::cout << data_size << std::endl;
    ssize_t read_bytes;
    vals.resize(data_size);
    read_bytes = read(input_manip, vals.data(), data_size * sizeof(double));
    if (read_bytes != data_size * sizeof(double)) {
        std::cout << " Check file system " << std::endl;
        abort();
    }
    std::cout << vals[0] << " " << vals[vals.size() - 1] << std::endl;
    char file_utimes[] = {"/users/eatanassov/goalpha/full/simplify_utimes.bin"};
    input_manip = open(file_int, O_RDONLY);
    utimes.resize(data_size);
    read_bytes = read(input_manip, utimes.data(), data_size * sizeof(double));
    if (read_bytes != data_size * sizeof(double)) {
        std::cout << " Check file system " << std::endl;
        abort();
    }
    std::cout << utimes[0] << " " << utimes[utimes.size() - 1] << std::endl;
    std::cout << vals[0] << " " << vals[vals.size() - 1] << std::endl;
    return 0;
}


int distance_compute_mat(const arma::mat &X, arma::mat &Z, double power2)
{
    int sizeX = arma::size(X, 1);
    arma::mat A = arma::trans(X);
    arma::mat B = arma::sum(arma::pow(A.t(), 2), 0);
    Z = arma::ones(sizeX, 1) * B;
    Z = Z + Z.t();
    Z = Z - 2 * A * A.t();
    if (power2 == 1.) {
        Z = arma::sqrt(arma::abs(Z));
    } else {
        if (power2 == 2.) {
            Z = arma::abs(Z);
        } else {
            std::cout << "ZLE " << std::endl;
        }
    }
    return 0;
}

#if 0
int distance_compute_mat_gpu(const arma::mat & X, arma::mat & Z,double power2,int counter,double*&d_Z){
    Z.resize(arma::size(X,0),arma::size(X,0));
    do_gpu_distance_compute_mat(arma::size(X,0),arma::size(X,1), X.memptr(), Z.memptr(),counter,d_Z,0.,0.);
    abort();
    return 0;
}

#endif


int distance_compute_mat_old(const arma::mat &X, arma::mat &Z, double beta)
{
    int sizeX = arma::size(X, 1);

    arma::mat exper = arma::exp(-beta * arma::cumsum(arma::ones(arma::size(X, 0), 1)));
    arma::mat multer = exper * arma::ones(1, sizeX);

    arma::mat A = arma::trans(X % multer);
    arma::mat B = arma::sum(arma::pow(A.t(), 2), 0);
    Z = arma::ones(sizeX, 1) * B;
    Z = Z + Z.t();
    Z = Z - 2 * A * A.t();
    return 0;
}

int distance_compute(const arma::mat &X, arma::mat &Z, int fresh_dimensions)
{
    int sizeX = arma::size(X, 1);
#pragma omp parallel for collapse(2)
    for (int i = 0; i < sizeX; i++) {
        for (int j = 0; j < sizeX; j++) {
            for (int k = 0; k < fresh_dimensions; k++) {
                //			Z(i,j)+=pow(X(k,i)-X(k,j),2);
                Z(i, j) += fabs(X(k, i) - X(k, j));
            }
        }
    }
    /*
        arma::mat A = X.t();
        arma::mat B = arma::sum(arma::pow(X,2),0);
        Z = arma::ones(sizeX,1)*B;
        Z = Z + Z.t();
        Z = Z - 2*A*X;
    */
    return 0;
}

double gtau = 0.;
int use_mista_solve = 0;

double evaluate_dimension(std::vector<arma::mat> &X_IN, int total_levels, std::vector<arma::mat> &Z, int dimension)
{
    arma::mat X = X_IN[0];
    const double cutoff = 0.05;
    int sizeX = arma::size(X, 1);
    int maxlag = arma::size(X, 0);
    const int vsize = (int) (cutoff * (double) sizeX);
    //compute_distances
    Z.resize(total_levels);
    for (int i = 0; i < total_levels; i++) {
        Z[i] = arma::zeros(arma::size(X, 1), arma::size(X, 1));
    }
    std::vector<arma::mat> ZM1;
    ZM1.resize(total_levels);
    for (int i = 0; i < total_levels; i++) {
        ZM1[i] = arma::zeros(arma::size(X, 1), arma::size(X, 1));
    }
    double power2 = 2.;
    for (int i = 0; i < total_levels; i++) {
        std::cout << "trying " << i << std::endl;
        distance_compute_mat(X_IN[i].rows(maxlag - 1 - dimension + 1, maxlag - 1), Z[i], power2);
        distance_compute_mat(X_IN[i].rows(maxlag - 1 - dimension + 1, maxlag - 2), ZM1[i], power2);
    }
    for (int level_num = 0; level_num < total_levels; level_num++) {
        X = X_IN[level_num];
        arma::vec zdiff = arma::trans(X.row(maxlag - 1) - X.row(maxlag - 2));

        arma::mat ZP1sum = 0. * Z[0];
        arma::mat ZM1sum = 0. * Z[0];
        for (int adj_levels = 0; (level_num - adj_levels >= 0) || (level_num + adj_levels < total_levels); adj_levels++) {
            for (int cross_level = std::max(0, level_num - adj_levels); cross_level <= std::min(level_num + adj_levels, total_levels - 1); cross_level++) {
                ZP1sum += Z[cross_level];
                ZM1sum += ZM1[cross_level];
            }
            std::cout << "Going " << adj_levels << std::endl;
            arma::mat result(sizeX, 1);
            arma::mat result2(sizeX, 1);
            arma::mat result3(sizeX, 1);
            arma::mat result4(sizeX, 1);
            arma::mat result5(sizeX, 1);
            arma::mat result6(sizeX, 1);
            arma::mat result7(sizeX, 1);
#pragma omp parallel for default(shared)
            for (int i = 0; i < sizeX; i++) {
                arma::vec temp = ZM1sum.col(i);
                std::vector<double> temp_vec(sizeX);
                for (int j = 0; j < sizeX; j++) temp_vec[j] = temp(j);
                std::nth_element(temp_vec.begin(), temp_vec.begin() + vsize, temp_vec.end());
                double imp_value = temp_vec[vsize];
                arma::vec temp_p1 = ZP1sum.col(i);
                std::vector<double> temp_vec_p1(sizeX);
                for (int j = 0; j < sizeX; j++) temp_vec_p1[j] = temp_p1(j);
                std::nth_element(temp_vec_p1.begin(), temp_vec_p1.begin() + vsize, temp_vec_p1.end());
                double imp_value_p1 = temp_vec_p1[vsize];
                int success2 = 0;
                double s = 0.;
                double s2 = 0.;
                int cntr = 0;
                double minv, maxv;
                for (int j = 0; j < sizeX; j++) {
                    if (temp(j) < imp_value) {
                        if ((cntr == 0) || (zdiff(j) < minv)) {
                            minv = zdiff(j);
                        }
                        if ((cntr == 0) || (zdiff(j) > maxv)) {
                            maxv = zdiff(j);
                        }
                        cntr++;
                        s += zdiff(j);
                        s2 += pow(zdiff(j), 2);
                        if (temp_p1(j) < imp_value_p1) {
                            success2++;
                        }
                    }
                }
#ifdef COMPLICATED
                double z = zdiff(i);

                double mu =  s/cntr;
                double sigma = sqrt((double)cntr/((double)cntr-1.)*(s2/(double)cntr - pow((s/(double)cntr),2)));
                double val = (z-mu)/sigma;
                double probab = 0.5*erfc(-val/M_SQRT2);
                double probabmin = 0.5*erfc(-(minv-mu)/sigma/M_SQRT2);
                double probabmax = 0.5*erfc(-(maxv-mu)/sigma/M_SQRT2);
                double probababs = 0.5*erfc(-(std::max(maxv-mu,mu-minv))/sigma/M_SQRT2);
                double predict = fabs(z-mu);

#endif
                result2(i, 0) = (double) success2 / vsize;
#ifdef COMPLICATED
                result3(i,0)= ((probab>1-cutoff)?1:0) + (probab<cutoff?1:0);
                result6(i,0)= (probababs>1-1./10./(double)vsize)?1:0;
                result4(i,0)= (probabmin<1./10./(double)vsize)?1:0;
                result5(i,0)= (probabmax>1-1./10./(double)vsize)?1:0;
                result7(i,0)= predict;
#endif
            }
#ifdef COMPLICATED
            arma::mat some_mean7 =  arma::mean(result7);
            arma::mat some_mean4 =  arma::mean(result4);
            arma::mat some_mean5 =  arma::mean(result5);
            arma::mat some_mean6 =  arma::mean(result6);
            arma::mat some_mean =  arma::mean(result3);
#endif
            arma::mat some_mean2 = arma::mean(result2);
            //std::cout << arma::mean(result) <<std::endl;
            std::cout << "level " << level_num << " adjust " << adj_levels << " dimension " << dimension << " " << some_mean2(0, 0) << std::endl;
        }

    }
    //return some_mean(0,0);
    return 0.;
}

double ggau;

extern int find_omega(int levels, const std::vector<double> &vals, std::vector<double> &omega);


int find_range(double alpha, std::vector<double>::iterator begin, std::vector<double>::iterator end, double &left_border, double &right_border, double &mean_val)
{
    std::vector<double> v;
    for (int i = 0; i < std::distance(begin, end) - 1; i++) {
        v.push_back(*(begin + i + 1) - *(begin + i));
    }

    const std::size_t pos_left = (size_t) (alpha * v.size());
    const std::size_t pos_right = (size_t) ((1. - alpha) * v.size());
    begin = v.begin();
    end = v.end();
    std::vector<double>::iterator med = begin + pos_left;
    std::nth_element(begin, med, end);
    left_border = *med;
    med = begin + pos_right;
    std::nth_element(begin, med, end);
    right_border = *med;
    mean_val = 0.;
    size_t cntr = 0;
    for (int i = 0; i < v.size(); i++) {
        if ((v[i] >= left_border) && (v[i] <= right_border)) {
            mean_val += std::abs(v[i]);
            cntr++;
        }
    }
    mean_val = mean_val / (double) cntr;
    return 0;
}

int get_mask_input(const char *filename, size_t num_sift, std::vector<std::vector<double>> &mask, std::vector<int> &siftings)
{
    std::ifstream myfile(filename);
    std::string line;
    mask.resize(0);
    int level = 0;
    while (myfile.good()) {
        try {
            std::getline(myfile, line);
            if (myfile.good()) {
                mask.resize(mask.size() + 1);

                std::stringstream ss(line);
                while (1) {
                    try {
                        double val;
                        ss >> val;
                        if (ss.good()) {
                            mask[level].push_back(val);
                        } else {
                            level++;
                            break;
                        }
                    } catch (std::exception ex) {
                        level++;
                        break;
                    }
                }
            }
        } catch (std::exception ex) {
            break;
        }
    }
    siftings.resize(mask.size());
    for (int i = 0; i < mask.size(); i++) siftings[i] = num_sift;//fixed number for all levels
    myfile.close();
    std::cout << "Mask ended." << std::endl;
    return 0;
}

int load_positions(std::vector<size_t> &test_positions, char *filename)
{
    std::ifstream myfile(filename);
    test_positions.resize(0);
    size_t pos;
    while (myfile.good()) {
        try {
            myfile >> pos;
            if (myfile.good()) {
                test_positions.push_back(pos);
            }
        } catch (std::exception ex) {
            break;
        }
    }
    myfile.close();
}


#define AHEAD 3600

//36000 ahead predict

#define SKIP_SIZE 10
//every 10 seconds from history are needed
#define LAG_SIZE 720
//dimension of features behind to be used
#define SIX_MINUTES 360
//6  minutes ahead in seconds
//so 720x10 = 2 hours

#define NUM_TRIES 2

int main(int argc, char *argv[])
{
    int the_level = atoi(argv[1]);
    times[0] = msecs();
    std::vector<int> siftings;
    std::vector<std::vector<double> > masks;
    int mask_levels;
    //const char* mask_filename= "lines_15_2_better.txt";
    const char *mask_filename = "lines_15_1.txt";
    //const char* mask_filename= "lines_15_2_0.txt";
    //const char* mask_filename= "maskin.txt";
    //const char* mask_filename= "lines_15_2_60.txt";
    size_t num_sift = 1;
    get_mask_input(mask_filename, num_sift, masks, siftings);
    std::cout << masks.size() << std::endl;
    for (int i = 0; i < masks.size(); i++) {
        std::cout << masks[i].size() << std::endl;
    }

    std::cout << "Stage masks " << msecs() - times[0] << std::endl;
    times[1] = msecs();
    std::vector<double> vals;
    std::vector<int> year;
    std::vector<int> month;
    std::vector<int> day;
    std::vector<int> hour;
    std::vector<int> minute;
    std::vector<int> second;
    std::vector<uint64_t> utimes;
    std::vector<int> hour_positions;
    std::vector<int> reper_positions;

    read_data_real(vals, year, month, day, hour, minute, second, utimes, hour_positions, reper_positions);

    std::cout << "Stage read data " << msecs() - times[1] << std::endl;
    times[2] = msecs();

    int ahead = AHEAD;
    int behind_stable = 7200;
//if 72000 seconds are available in history, then ok, otherwise - no prediction is to be done

    int behind_all = 72000;

//10 hours data needed




    std::cout << "Start vmd" << std::endl;
    const int num_vmd_levels = 16;
    std::vector<double> static_good_omega_from_jar(
            {5.5699e-12, 4.0481e-04, 1.1780e-03, 2.4003e-03, 4.2197e-03, 6.3387e-03, 9.1807e-03, 1.2833e-02, 1.6681e-02, 2.1018e-02, 2.6047e-02, 3.2055e-02, 4.0413e-02, 4.6289e-02, 5.5226e-02,
             6.2776e-02});
    std::vector<double> omega = static_good_omega_from_jar;
    std::vector<double> phase_cos, phase_sin;
    phase_cos.resize(num_vmd_levels);
    phase_sin.resize(num_vmd_levels);
    compute_cos_sin(omega, phase_cos, phase_sin, 1.);
    std::vector<double> previous(num_vmd_levels * 2, 0.);
    std::vector<double> next(num_vmd_levels * 2, 0.);
    previous[0] = vals[0];
    arma::mat decomposition(num_vmd_levels * 2, vals.size());
    std::vector<std::vector<double>> cvmd(2 * num_vmd_levels);
    for (int j = 0; j < 2 * num_vmd_levels; j++) {
        decomposition(j, 0) = previous[j];
        cvmd[j].push_back(j == 0 ? vals[0] : 0.);
    }
#define DO_SAVE_VMD 1
    //undef after first run
#ifdef DO_SAVE_VMD
    for (int i = 1; i < vals.size(); i++) {
        std::vector<double> current({vals[i]});
        step_decompose_matrix(phase_cos, phase_sin, 1, current.data(), previous, next);
        for (int j = 0; j < 2 * num_vmd_levels; j++) {
            decomposition(j, i) = next[j];
            previous[j] = next[j];
            cvmd[j].push_back(next[j]);
        }
        if (i % 100000 == 0) std::cout << i << std::endl;
    }
    for (int j = 0; j < 2 * num_vmd_levels; j++) {
        char buffer[1000];
        sprintf(buffer, "cvmd_Yn_%i.bin", j);
        int out_manip = open(buffer, O_CREAT | O_WRONLY | O_TRUNC, 0666);
        ssize_t written = write(out_manip, cvmd[j].data(), cvmd[j].size() * sizeof(double));
        close(out_manip);
        if (written != cvmd[j].size() * sizeof(double)) {
            abort();
        }
    }
#else
    //load from files
    for(int j=0;j<2*num_vmd_levels;j++){
        char buffer[1000];
        sprintf(buffer,"cvmd_Yn_%i.bin",j);
        int in_manip =  open(buffer, O_RDONLY ) ;
         if (in_manip<0){
                        abort();
            }
            struct stat file_stat;
        int rc = fstat(in_manip, &file_stat);
            if (rc != 0 ) {
                        perror("fstat failed or file is not a regular file");
                        abort();
            }
            const off_t if_size = file_stat.st_size;
            size_t data_size = if_size / sizeof(double);
            ssize_t read_bytes;
            cvmd[j].resize(data_size);
            read_bytes = read(in_manip, cvmd[j].data(),data_size*sizeof(double));
            if (read_bytes != data_size*sizeof(double)){
                        std::cout << " Check file system " << std::endl;
                        abort();
            }
    }
#endif
    std::vector<double> remember_vals = vals;

    vals = cvmd[0];
    std::cout << "Start oemd" << std::endl;
    times[3] = msecs();
    std::vector<std::vector<double> > oemd_levels_pure;
    split_oemd(vals, masks, siftings, oemd_levels_pure);
    std::cout << "Stage oemd " << msecs() - times[3] << std::endl;
    times[4] = msecs();

    double alpha = ALPHA_ONE_PERCENT;//for trimmed mean
    size_t initial_skip = 10000000; //skip the initial 10 million, since they are bigger and there are some artefacts there.

    int full_levels = oemd_levels_pure.size() + cvmd.size() - 1;
    std::vector<std::vector<double>> oemd_levels(full_levels);
    for (int i = 0; i < oemd_levels_pure.size(); i++) {
        oemd_levels[i] = oemd_levels_pure[oemd_levels_pure.size() - 1 - i];
        //most noisy level is highest!
    }
    for (int i = oemd_levels_pure.size(); i < full_levels; i++) {
        oemd_levels[i] = cvmd[i - oemd_levels_pure.size() + 1];
    }

    std::vector<double> borders;
    std::vector<double> trimmed_mean_vals;


    for (int i = 0; i < oemd_levels.size(); i++) {
        std::vector<double>::iterator begin_it, end_it;
        begin_it = oemd_levels[i].begin() + initial_skip;
        end_it = oemd_levels[i].end();
        double left_border, right_border, mean_val;
        find_range(alpha, begin_it, end_it, left_border, right_border, mean_val);
        borders.push_back(right_border - left_border);
        trimmed_mean_vals.push_back(mean_val);
        //trimmed_mean_vals.push_back(1.);
        std::cout << "Level " << i << " border difference " << right_border - left_border << " mean val for scaling " << mean_val << std::endl;

        double ss, ss2;
        ss = 0.;
        ss2 = 0.;
        for (size_t j = 0; j < oemd_levels[i].size() - 1; j++) {
            ss += fabs(oemd_levels[i][j] - oemd_levels[i][j + 1]);
        }
        std::cout << "Mean diff " << ss / ((double) oemd_levels[i].size() - 1) << std::endl;

        // mean val is mean of abs of all values that are between left and right border, thus is always smaller than the simple distance between left and right border
    }

    std::cout << "Stage trimmed mean " << msecs() - times[4] << std::endl;
    times[5] = msecs();

    std::vector<size_t> test_positions;
    load_positions(test_positions, "results_positions.txt");
    std::cout << "Stage positions " << msecs() - times[5] << std::endl;

    times[6] = msecs();

    size_t back_skip = SKIP_SIZE;
    size_t back_size = LAG_SIZE;//minutes
    size_t six_minutes = SIX_MINUTES;

    std::vector<std::vector<double>> xtrain;
    std::vector<std::vector<double>> ytrain;
    std::vector<double> y_mean_val;
    make_trainings(oemd_levels, test_positions, six_minutes, ahead, back_skip, back_size, xtrain, ytrain, trimmed_mean_vals, y_mean_val);
    //ytrain is scaled now
    //xtrain is scaled now

    std::cout << "Stage trainings (features) " << msecs() - times[6] << std::endl;


    int adjacent_left = atoi(argv[2]); // number of levels from left, 0 means only the current level
    int adjacent_right = atoi(argv[3]); // number of levels from right, 0 means only the current level
    const int numtries = NUM_TRIES;
    const int dimension = back_size;
    std::vector<double> parameters;
    opti_param(numtries, xtrain, ytrain, the_level, adjacent_left, adjacent_right, parameters);
    return 0;
}
