//#include <gtest/gtest.h>
//extern "C"{
//#include <wavelib/header/wavelib.h>
//}
//
//#include <sstream>
//#include <util/CompressionUtils.hpp>
//#include <spectral_transform.hpp>
//#include <fstream>
//#include <cstdlib>
//#include <iostream>
//#include <math.h>
//#include <stdlib.h>
//#include <stdio.h>
//#include <string.h>
//
//double absmax(double *array, int N) {
//	double max;
//	int i;
//
//	max = 0.0;
//	for (i = 0; i < N; ++i) {
//                        if (fabs(array[i]) >= max)
//                            max = fabs(array[i]);
//                    }
//
//	return max;
//}
//
//char const * test_data_file_name = "../SVRRoot/OnlineSVR/test/modwt_test_data/inputqueue_long.txt";
//
//int deconlevel[] = {2, 4, 8 }; //{2, 4, 6, 8, 10, 12, 14, 16};
//const int edl = sizeof(deconlevel) / sizeof(deconlevel[0]);
//
//int filterorder[] = {en_fk4, en_fk8}; //{en_fk4, en_fk6, en_fk8, en_fk14, en_fk22}
//const int efo = sizeof(filterorder) / sizeof(filterorder[0]);
//
//
//TEST(modwt, FejKor_overlap){
//    //compute all F-K filters for every level of deconstruction
//for(int dl = 0; dl < edl; dl++){
//    printf("deconstruction levels: %d\n", deconlevel[dl]);
//
//    for(int fo=0; fo< efo; fo++){
//       printf("order of filter: %d,\t", filterorder[fo]);
//
//    wave_object wobj_a;
//    wave_object wobj_b;
//    wt_object wt_a;
//    wt_object wt_b;
//
//    double *inp_a, *inp_b, *recon_out, *diff;
//    int rix=0;
//    FILE *ifp=NULL;
//    const int predict = 16;//numbers of predict values
//    int dlv = deconlevel[dl];// levels of deconstruction
//    int filter_order =filterorder[fo];
//    int wvlen = (int)(filter_order * powf(2,dlv+1));
//    int pp = wvlen / 2;
//
//    double* temp_a = (double*)malloc(sizeof(double)*wvlen);
//    double* temp_b = (double*)malloc(sizeof(double)*pp);
//    double* tail = (double*)malloc(sizeof(double)*predict);
//    //memset(tail, 0, predict);
//
//    wobj_a = wave_init1(filter_order);
//    wobj_b = wave_init1(filter_order);
//
//    ifp = fopen(test_data_file_name, "r");
//
//    ASSERT_TRUE(ifp);
//
//    while ( !feof(ifp) && rix< wvlen ) { //read wvlen symbols
//        auto ret = fscanf(ifp, "%lf \n", &temp_a[rix]);
//        (void)ret;
//        rix++;
//    }
//
//    for(int i=0; i<pp; ++i)
//        temp_b[i] = temp_a[pp + i];
//
//    inp_a = (double*)malloc(sizeof(double)* wvlen);
//    inp_b = (double*)malloc(sizeof(double)* pp);
//    recon_out = (double*)malloc(sizeof(double)* wvlen);
//
//    for (rix = 0; rix < wvlen; ++rix) {
//        inp_a[rix] = temp_a[rix];
//    }
//
//    for (rix = 0; rix < pp; ++rix) {
//        inp_b[rix] = temp_b[rix];
//    }
//
//    wt_a = wt_init1(wobj_a, wvlen, dlv);// Initialize the wavelet transform object
//    wt_b = wt_init1(wobj_b, pp, dlv);// Initialize the wavelet transform object
//
//    modwt1(wt_a, inp_a);    //MODWT output can be accessed using wt->output vector. Use wt_summary to find out how to extract appx and detail coefficients
//    modwt1(wt_b, inp_b);
//
//    int step = predict;
//    //get the last 'pp' numbers from wt_b
//    int q = wt_b->outlength - predict;
//    for(int i=0; i<step; ++i){
//        tail[i] = wt_b->params[q + i];
//    }
//    //copy the tail over last 'pp' lenght of wt_a
//    int k= wt_a->outlength - predict;
//    for(int j=0; j < step; ++j){
//        wt_a->params[k+j] = tail[j];
//    }
//
//    inverse_modwt(wt_a, recon_out);
//
//    diff = (double*)malloc( sizeof(double) * predict );
//    //get the diff of last 'predict' numbers
//    for (int p = wt_a->siglength - predict, i=0; p < wt_a->siglength; ++p, ++i){
//         diff[i] = recon_out[p] - inp_a[p];
//    }
//
//    printf("MaxRes %g \n", absmax(diff, predict));// If Reconstruction succeeded then the output should be a small value.
//
//    wave_free(wobj_a);
//    wave_free(wobj_b);
//
//    free(wt_a->params);
//    wt_free(wt_a);
//
//    free(wt_b->params);
//    wt_free(wt_b);
//
//    free(tail);
//    free (temp_a);
//    free (temp_b);
//    free(inp_a);
//    free(inp_b);
//    free(recon_out);
//    free(diff);
//
//    }   }
//}
//
//TEST(modwt, FejKor_simple){
//    //compute all F-K filters for every level of deconstruction
//    for(int dl = 0; dl < edl; dl++){
//        printf("deconstruction levels: %d\n", deconlevel[dl]);
//
//    for(int fo=0; fo< efo; fo++){
//       printf("order of filter: %d,\t", filterorder[fo]);
//
//    wave_object wobj_a;
//    wt_object wt_a;
//
//    double *inp_a, *recon_out;
//    int rix=0;
//    FILE *ifp=NULL;
//
//    int dlv = deconlevel[dl];// levels of deconstruction
//    int filter_order =filterorder[fo];
//    int wvlen = (int)(filter_order * powf(2,dlv+1));
//
//    double* temp_a = (double*)malloc(sizeof(double)*wvlen);
//
//    wobj_a = wave_init1(filter_order);
//
//    ifp = fopen(test_data_file_name, "r");
//
//    ASSERT_TRUE(ifp);
//
//    while ( !feof(ifp) && rix< wvlen ) { //read wvlen symbols
//        auto ret = fscanf(ifp, "%lf \n", &temp_a[rix]);
//        (void)ret;
//        rix++;
//    }
//
//    inp_a = (double*)malloc(sizeof(double)* wvlen);
//    recon_out = (double*)malloc(sizeof(double)* wvlen);
//
//    for (rix = 0; rix < wvlen; ++rix) {
//        inp_a[rix] = temp_a[rix];
//    }
//
//    wt_a = wt_init1(wobj_a, wvlen, dlv);// Initialize the wavelet transform object
//
//    modwt1(wt_a, inp_a);    //MODWT output can be accessed using wt->output vector. Use wt_summary to find out how to extract appx and detail coefficients
//
//    inverse_modwt(wt_a, recon_out);
//
//    double* diff = (double*)malloc( sizeof(double) * wvlen );
//    //get the diff between input and output
//    for(int i=0; i<wt_a->siglength; ++i){
//         diff[i] = inp_a[i] - recon_out[i];
//    }
//
//    printf("MaxRes %g \n", absmax(diff, wt_a->siglength));// If Reconstruction succeeded then the output should be a small value.
//
//    wave_free(wobj_a);
//
//    free(wt_a->params);
//    wt_free(wt_a);
//
//    free(temp_a);
//    free(inp_a);
//    free(recon_out);
//    free(diff);
//
//    }//end for of filters
//
//    }//end for of decon levels
//}
//
//TEST(modwt, FejKor_custom_inp_len){//test with input len > wave len
//    //compute all F-K filters for every level of deconstruction
//    for(int dl = 0; dl < edl; dl++){
//        printf("deconstruction levels: %d\n", deconlevel[dl]);
//
//    for(int fo=0; fo < efo; fo++){
//       printf("order of filter: %d,\t", filterorder[fo]);
//
//    wave_object wobj_a;
//    wt_object   wt_a;
//
//    double *inp_a, *recon_out;
//    int rix = 0;
//    FILE *ifp = NULL;
//
//    int dlv = deconlevel[dl];   // levels of deconstruction
//    int filter_order = filterorder[fo];
//    int wvlen = (int)(filter_order * powf(2,dlv+1));
//    int custom_len = 10*1024;
//    int predict = 15;
//    wvlen = custom_len + predict;   // custom len > wave length from formula
//
//    double* temp_a = (double*)malloc(sizeof(double)*wvlen);
//
//    wobj_a = wave_init1(filter_order);
//
//    ifp = fopen(test_data_file_name, "r");
//
//    ASSERT_TRUE(ifp);
//
//    while ( !feof(ifp) && rix< wvlen ) { //read wvlen symbols
//        if(fscanf(ifp, "%lf \n", &temp_a[rix]) < 1) printf("Warn, value could not be read.");
//        rix++;
//    }
//
//    inp_a = (double*)malloc(sizeof(double)* wvlen);
//    recon_out = (double*)malloc(sizeof(double)* wvlen);
//
//    for (rix = 0; rix < wvlen; ++rix) {
//        inp_a[rix] = temp_a[rix];
//    }
//
//    wt_a = wt_init1(wobj_a, wvlen, dlv);// Initialize the wavelet transform object
//
//    modwt1(wt_a, inp_a);    //MODWT output can be accessed using wt->output vector. Use wt_summary to find out how to extract appx and detail coefficients
//
//    inverse_modwt(wt_a, recon_out);
//
//    double* diff = (double*)malloc( sizeof(double) * wvlen );
//
//    //get the diff betwwen input and output
//    for(int i=0; i<wt_a->siglength; ++i){
//         diff[i] = inp_a[i] - recon_out[i];
//    }
//
//    printf("MaxRes %g \n", absmax(diff, wt_a->siglength));// If Reconstruction succeeded then the output should be a small value.
//
//    wave_free(wobj_a);
//
//    free(wt_a->params);
//    wt_free(wt_a);
//
//    free(temp_a);
//    free(inp_a);
//    free(recon_out);
//    free(diff);
//
//    }//end for of filters
//
//    }//end for of decon levels
//}
