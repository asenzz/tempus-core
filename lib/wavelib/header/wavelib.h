#ifndef WAVELIB_H_
#define WAVELIB_H_

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_MSC_VER)
#pragma warning(disable : 4200)
#pragma warning(disable : 4996)
#endif

#ifndef fft_type
#define fft_type double
#endif

#ifndef cplx_type
#define cplx_type double
#endif


#include <stdio.h>
#include <math.h>    

//Fejer-Korovkin filters order
enum en_fk_filter_order
{
	en_fk4 = 4,
	en_fk6 = 6,
	en_fk8 = 8,
	en_fk14 = 14,
	en_fk22 = 22
};

typedef struct cplx_t {
	cplx_type re;
	cplx_type im;
} cplx_data;

typedef struct wave_set* wave_object;
wave_object wave_init(char* wname);
wave_object wave_init1(int filter);

struct wave_set {
    char wname[50];
    int filtlength;// When all filters are of the same length. [Matlab uses zero-padding to make all filters of the same length]
    int lpd_len;// Default filtlength = lpd_len = lpr_len = hpd_len = hpr_len
    int hpd_len;
    int lpr_len;
    int hpr_len;
    double *lpd;
    double *hpd;
    double *lpr;
    double *hpr;
    double params[1];
    double* FKfparams;   //F-K filer coeficients
    
    /*
    void copy(const wave_set* ws)   {
        strncpy(wname, ws->wname, 50);
        filtlength = ws->filtlength;
        lpd_len = ws->lpd_len;
        hpd_len = ws->hpd_len;
        lpr_len = ws->lpr_len;
        hpr_len = ws->hpr_len;
        *lpd = *ws->lpd;
        *hpd = *ws->hpd;
        *lpr = *ws->lpr;
        *hpr = *ws->hpr;
        params[0] = ws->params[0];
    }
    */
};

typedef struct fft_t {
  fft_type re;
  fft_type im;
  /*
  void copy(const fft_t* fs){
      re = fs->re;
      im = fs->im;
  }
  */
} fft_data;

typedef struct fft_set* fft_object;

fft_object fft_init(int N, int sgn);

struct fft_set{
    int N;
    int sgn;
    int factors[64];
    int lf;
    int lt;
    fft_data twiddle[1];
    /*
    void copy(const fft_set* fs){
    N = fs->N;
    sgn = fs->sgn;
    for(int i=0; i<64; ++i)
        factors[i] = fs->factors[i];
    lf = fs->lf;
    lt = fs->lt;
    twiddle[1].copy(fs->twiddle[1]);
    }
     */
};

typedef struct fft_real_set* fft_real_object;

fft_real_object fft_real_init(int N, int sgn);

struct fft_real_set{
    fft_object cobj;
    fft_data twiddle2[1];
    /*
    void copy(const fft_real_set* fs){
        cobj = fs->cobj;
        twiddle2[1].copy(fs->twiddle2[1]);
    }
    */
};

typedef struct conv_set* conv_object;

conv_object conv_init(int N, int L);

struct conv_set{
    fft_real_object fobj;
    fft_real_object iobj;
    int ilen1;
    int ilen2;
    int clen;
    /*
    void copy(const conv_set* cs){
    fobj = cs->fobj;
    iobj = cs->iobj;
    ilen1 = cs->ilen1;
    ilen2 = cs->ilen2;
    clen = cs->clen;
    }
    */
};

typedef struct wt_set* wt_object;

wt_object wt_init(wave_object wave,char* method, int siglength, int J);

wt_object wt_init1(wave_object wave, int siglength, int J);

struct wt_set{
    wave_object wave;
    conv_object cobj;
    char method[10];
    int filterorder;//Order of the Fejer-Korovkin filer
    int siglength;// Length of the original signal.
    int outlength;// Length of the output DWT vector
    int lenlength;// Length of the Output Dimension Vector "length"
    int J; // Number of decomposition Levels
    int MaxIter;// Maximum Iterations J <= MaxIter
    int even;// even = 1 if signal is of even length. even = 0 otherwise
    char ext[10];// Type of Extension used - "per" or "sym"
    char cmethod[10]; // Convolution Method - "direct" or "FFT"
    int N; //
    int cfftset;
    int zpad;
    int length[102];
    double *output;
    double* params;
    
    /*
    void copy(const wt_set* ws){
        wave.copy(ws->wave);
        cobj->copy(ws->cobj);
        strcpy(method, ws->method, 10);
        filterorder = ws->filterorder;
        siglength = ws->siglength;
        outlength = ws->outlength;
        lenlength = ws->lenlength;
        J = ws->J;
        MaxIter = ws->MaxIter;
        even = ws->even;
        strcpy(ext, ws->ext, 10);
        strcpy(cmethod, ws->cmethod, 10);
        N = ws->N; 
        cfftset = ws->cfftset;
        zpad = ws->zpad;
        for(int i=0; i<102; ++i)
            length[i] = ws->length[i];
        *output = *ws->outlength;
        params[0] = ws->params[0];
    }
    */
};

typedef struct wtree_set* wtree_object;

wtree_object wtree_init(wave_object wave, int siglength, int J);

struct wtree_set{
    wave_object wave;
    conv_object cobj;
    char method[10];
    int siglength;// Length of the original signal.
    int outlength;// Length of the output DWT vector
    int lenlength;// Length of the Output Dimension Vector "length"
    int J; // Number of decomposition Levels
    int MaxIter;// Maximum Iterations J <= MaxIter
    int even;// even = 1 if signal is of even length. even = 0 otherwise
    char ext[10];// Type of Extension used - "per" or "sym"
    int N; //
    int nodes;
    int cfftset;
    int zpad;
    int length[102];
    double *output;
    int *nodelength;
    int *coeflength;
    double params[1];
};

typedef struct wpt_set* wpt_object;

wpt_object wpt_init(wave_object wave, int siglength, int J);

struct wpt_set{
    wave_object wave;
    conv_object cobj;
    int siglength;// Length of the original signal.
    int outlength;// Length of the output DWT vector
    int lenlength;// Length of the Output Dimension Vector "length"
    int J; // Number of decomposition Levels
    int MaxIter;// Maximum Iterations J <= MaxIter
    int even;// even = 1 if signal is of even length. even = 0 otherwise
    char ext[10];// Type of Extension used - "per" or "sym"
    char entropy[20];
    double eparam;

    int N; //
    int nodes;
    int length[102];
    double *output;
    double *costvalues;
    double *basisvector;
    int *nodeindex;
    int *numnodeslevel;
    int *coeflength;
    double params[1];
};


typedef struct cwt_set* cwt_object;

cwt_object cwt_init(char* wave, double param, int siglength,double dt, int J);

struct cwt_set{
    char wave[10];// Wavelet - morl/morlet,paul,dog/dgauss
    int siglength;// Length of Input Data
    int J;// Total Number of Scales
    double s0;// Smallest scale. It depends on the sampling rate. s0 <= 2 * dt for most wavelets
    double dt;// Sampling Rate
    double dj;// Separation between scales. eg., scale = s0 * 2 ^ ( [0:N-1] *dj ) or scale = s0 *[0:N-1] * dj
    char type[10];// Scale Type - Power or Linear
    int pow;// Base of Power in case type = pow. Typical value is pow = 2
    int sflag;
    int pflag;
    int npad;
    int mother;
    double m;// Wavelet parameter param
    double smean;// Input Signal mean

    cplx_data *output;
    double *scale;
    double *period;
    double *coi;
    double params[1];
};


void dwt(wt_object wt, double *inp);

void idwt(wt_object wt, double *dwtop);

void wtree(wtree_object wt, double *inp);

void dwpt(wpt_object wt, double *inp);

void idwpt(wpt_object wt, double *dwtop);

void swt(wt_object wt, double *inp);

void iswt(wt_object wt, double *swtop);

void modwt(wt_object wt, double *inp);

void imodwt(wt_object wt, double *dwtop);

// modwt1 is test fuction for F-K case
void modwt1(wt_object wt, double *inp);

void inverse_modwt(wt_object wt, double *dwtop);

void setDWTExtension(wt_object wt, char *extension);

void setWTREEExtension(wtree_object wt, char *extension);

void setDWPTExtension(wpt_object wt, char *extension);

void setDWPTEntropy(wpt_object wt, char *entropy, double eparam);

void setWTConv(wt_object wt, char *cmethod);

int getWTREENodelength(wtree_object wt, int X);

void getWTREECoeffs(wtree_object wt, int X, int Y, double *coeffs, int N);

int getDWPTNodelength(wpt_object wt, int X);

void getDWPTCoeffs(wpt_object wt, int X, int Y, double *coeffs, int N);

void setCWTScales(cwt_object wt, double s0, double dj, char *type, int power);

void setCWTScaleVector(cwt_object wt, double *scale, int J, double s0, double dj);

void setCWTPadding(cwt_object wt, int pad);

void cwt(cwt_object wt, double *inp);

void icwt(cwt_object wt, double *cwtop);

int getCWTScaleLength(int N);

void wave_summary(wave_object obj);

void wt_summary(wt_object wt);

void wtree_summary(wtree_object wt);

void wpt_summary(wpt_object wt);

void cwt_summary(cwt_object wt);

void wave_free(wave_object object);

void wt_free(wt_object object);

void wtree_free(wtree_object object);

void wpt_free(wpt_object object);

void cwt_free(cwt_object object);

//int wmaxiter1(int sig_len, int filt_len) {
//	int lev;
//	double temp;
//
//	temp = log((double)sig_len / ((double)filt_len - 1.0)) / log(2.0);
//	lev = (int)temp;
//
//	return lev;
//}
//
//wt_object wt_init_h(wave_object wave, int siglength, int J){
//    int size,i,MaxIter;
//    wt_object obj = NULL;
//
//    size = wave->filtlength;
//
//    if (J > 100) {  
//        return NULL;
//    }
//
//    MaxIter = wmaxiter1(siglength, size);
//
//    obj = (wt_object)malloc(sizeof(struct wt_set));
//               
//    obj->wave =  wave;                             
//    obj->siglength = siglength;                    
//    obj->params = (double*)malloc((siglength * (J + 1)) * sizeof(double));
//    obj->output = &obj->params[0];
//    obj->outlength = siglength * (J + 1);
//
//    obj->J = J;
//    obj->MaxIter = MaxIter;
//
//    if (siglength % 2 == 0) {
//            obj->even = 1;
//    }
//    else {
//            obj->even = 0;
//    }
//
//    obj->cobj = NULL; //?
//
//    obj->cfftset = 0;
//    obj->lenlength = J + 2; //?
//
//    for(i = 0; i < obj->outlength; ++i){                             
//        obj->params[i] = 0.0;
//    }
//
//    return obj;
//}

#ifdef __cplusplus
}
#endif


#endif /* WAVELIB_H_ */
