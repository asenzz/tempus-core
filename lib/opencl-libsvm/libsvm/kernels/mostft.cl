/* ************************************************************************
 * Copyright 2013 Advanced Micro Devices, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ************************************************************************/
#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#else
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif
__constant double2 twiddles[15] = {
    (double2) (1.0000000000000000000000000000000000e+00, -0.0000000000000000000000000000000000e+00),
    (double2) (1.0000000000000000000000000000000000e+00, -0.0000000000000000000000000000000000e+00),
    (double2) (1.0000000000000000000000000000000000e+00, -0.0000000000000000000000000000000000e+00),
    (double2) (1.0000000000000000000000000000000000e+00, -0.0000000000000000000000000000000000e+00),
    (double2) (1.0000000000000000000000000000000000e+00, -0.0000000000000000000000000000000000e+00),
    (double2) (1.0000000000000000000000000000000000e+00, -0.0000000000000000000000000000000000e+00),
    (double2) (9.2387953251128673848313610506011173e-01, -3.8268343236508978177923268049198668e-01),
    (double2) (7.0710678118654757273731092936941423e-01, -7.0710678118654746171500846685376018e-01),
    (double2) (3.8268343236508983729038391174981371e-01, -9.2387953251128673848313610506011173e-01),
    (double2) (7.0710678118654757273731092936941423e-01, -7.0710678118654746171500846685376018e-01),
    (double2) (6.1232339957367660358688201472919830e-17, -1.0000000000000000000000000000000000e+00),
    (double2) (-7.0710678118654746171500846685376018e-01, -7.0710678118654757273731092936941423e-01),
    (double2) (3.8268343236508983729038391174981371e-01, -9.2387953251128673848313610506011173e-01),
    (double2) (-7.0710678118654746171500846685376018e-01, -7.0710678118654757273731092936941423e-01),
    (double2) (-9.2387953251128684950543856757576577e-01, 3.8268343236508967075693021797633264e-01),
};
#define fptype double
#define fvect2 double2
#define C8Q  0.70710678118654752440084436210485

__attribute__ ((always_inline)) void
FwdRad4B1(double2 *R0, double2 *R2, double2 *R1, double2 *R3) {
    double2 T;
    (*R1) = (*R0) - (*R1);
    (*R0) = 2.0f * (*R0) - (*R1);
    (*R3) = (*R2) - (*R3);
    (*R2) = 2.0f * (*R2) - (*R3);

    (*R2) = (*R0) - (*R2);
    (*R0) = 2.0f * (*R0) - (*R2);
    (*R3) = (*R1) + (fvect2) (-(*R3).y, (*R3).x);
    (*R1) = 2.0f * (*R1) - (*R3);

    T = (*R1);
    (*R1) = (*R2);
    (*R2) = T;

}

__attribute__ ((always_inline)) void
InvRad4B1(double2 *R0, double2 *R2, double2 *R1, double2 *R3) {
    double2 T;
    (*R1) = (*R0) - (*R1);
    (*R0) = 2.0f * (*R0) - (*R1);
    (*R3) = (*R2) - (*R3);
    (*R2) = 2.0f * (*R2) - (*R3);

    (*R2) = (*R0) - (*R2);
    (*R0) = 2.0f * (*R0) - (*R2);
    (*R3) = (*R1) + (fvect2) ((*R3).y, -(*R3).x);
    (*R1) = 2.0f * (*R1) - (*R3);

    T = (*R1);
    (*R1) = (*R2);
    (*R2) = T;

}

__attribute__ ((always_inline)) void
FwdPass0(uint rw, uint b, uint me, uint inOffset, uint outOffset, __global double *bufIn, __global double *bufIn2, __local double *bufOutRe, __local double *bufOutIm, double2 *R0, double2 *R1, double2 *R2, double2 *R3) {
    if (rw) {
        (*R0).x = bufIn[inOffset + (0 + me * 1 + 0 + 0)*1];
        (*R1).x = bufIn[inOffset + (0 + me * 1 + 0 + 4)*1];
        (*R2).x = bufIn[inOffset + (0 + me * 1 + 0 + 8)*1];
        (*R3).x = bufIn[inOffset + (0 + me * 1 + 0 + 12)*1];
    }
    if (rw > 1) {
        (*R0).y = bufIn2[inOffset + (0 + me * 1 + 0 + 0)*1];
        (*R1).y = bufIn2[inOffset + (0 + me * 1 + 0 + 4)*1];
        (*R2).y = bufIn2[inOffset + (0 + me * 1 + 0 + 8)*1];
        (*R3).y = bufIn2[inOffset + (0 + me * 1 + 0 + 12)*1];
    } else {
        (*R0).y = 0;
        (*R1).y = 0;
        (*R2).y = 0;
        (*R3).y = 0;
    }
    FwdRad4B1(R0, R1, R2, R3);
    if (rw) {
        bufOutRe[outOffset + (((1 * me + 0) / 1)*4 + (1 * me + 0) % 1 + 0)*1] = (*R0).x;
        bufOutRe[outOffset + (((1 * me + 0) / 1)*4 + (1 * me + 0) % 1 + 1)*1] = (*R1).x;
        bufOutRe[outOffset + (((1 * me + 0) / 1)*4 + (1 * me + 0) % 1 + 2)*1] = (*R2).x;
        bufOutRe[outOffset + (((1 * me + 0) / 1)*4 + (1 * me + 0) % 1 + 3)*1] = (*R3).x;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (rw) {
        (*R0).x = bufOutRe[outOffset + (0 + me * 1 + 0 + 0)*1];
        (*R1).x = bufOutRe[outOffset + (0 + me * 1 + 0 + 4)*1];
        (*R2).x = bufOutRe[outOffset + (0 + me * 1 + 0 + 8)*1];
        (*R3).x = bufOutRe[outOffset + (0 + me * 1 + 0 + 12)*1];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (rw) {
        bufOutIm[outOffset + (((1 * me + 0) / 1)*4 + (1 * me + 0) % 1 + 0)*1] = (*R0).y;
        bufOutIm[outOffset + (((1 * me + 0) / 1)*4 + (1 * me + 0) % 1 + 1)*1] = (*R1).y;
        bufOutIm[outOffset + (((1 * me + 0) / 1)*4 + (1 * me + 0) % 1 + 2)*1] = (*R2).y;
        bufOutIm[outOffset + (((1 * me + 0) / 1)*4 + (1 * me + 0) % 1 + 3)*1] = (*R3).y;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (rw) {
        (*R0).y = bufOutIm[outOffset + (0 + me * 1 + 0 + 0)*1];
        (*R1).y = bufOutIm[outOffset + (0 + me * 1 + 0 + 4)*1];
        (*R2).y = bufOutIm[outOffset + (0 + me * 1 + 0 + 8)*1];
        (*R3).y = bufOutIm[outOffset + (0 + me * 1 + 0 + 12)*1];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
}

__attribute__ ((always_inline)) void
FwdPass1(uint rw, uint b, uint me, uint inOffset, uint outOffset, __local double *bufInRe, __local double *bufInIm, __global double2 *bufOut, __global double2 *bufOut2, double2 *R0, double2 *R1, double2 *R2, double2 *R3) {
    {
        double2 W = twiddles[3 + 3 * ((1 * me + 0) % 4) + 0];
        double TR, TI;
        TR = (W.x * (*R1).x) - (W.y * (*R1).y);
        TI = (W.y * (*R1).x) + (W.x * (*R1).y);
        (*R1).x = TR;
        (*R1).y = TI;
    }
    {
        double2 W = twiddles[3 + 3 * ((1 * me + 0) % 4) + 1];
        double TR, TI;
        TR = (W.x * (*R2).x) - (W.y * (*R2).y);
        TI = (W.y * (*R2).x) + (W.x * (*R2).y);
        (*R2).x = TR;
        (*R2).y = TI;
    }
    {
        double2 W = twiddles[3 + 3 * ((1 * me + 0) % 4) + 2];
        double TR, TI;
        TR = (W.x * (*R3).x) - (W.y * (*R3).y);
        TI = (W.y * (*R3).x) + (W.x * (*R3).y);
        (*R3).x = TR;
        (*R3).y = TI;
    }
    FwdRad4B1(R0, R1, R2, R3);
    bufInRe[inOffset + (1 * me + 0 + 0)*1] = (*R0).x;
    bufInRe[inOffset + (1 * me + 0 + 4)*1] = (*R1).x;
    bufInRe[inOffset + (1 * me + 0 + 8)*1] = (*R2).x;
    bufInRe[inOffset + (1 * me + 0 + 12)*1] = (*R3).x;
    barrier(CLK_LOCAL_MEM_FENCE);
    (*R0).x = bufInRe[inOffset + (me + 1)*1];
    (*R1).x = bufInRe[inOffset + (me + 5)*1];
    (*R2).x = bufInRe[inOffset + (16 - (me + 1))*1];
    (*R3).x = bufInRe[inOffset + (16 - (me + 5))*1];
    if (rw && !me) {
        bufOut[outOffset].x = bufInRe[inOffset];
        bufOut[outOffset].y = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    bufInIm[inOffset + (1 * me + 0 + 0)*1] = (*R0).y;
    bufInIm[inOffset + (1 * me + 0 + 4)*1] = (*R1).y;
    bufInIm[inOffset + (1 * me + 0 + 8)*1] = (*R2).y;
    bufInIm[inOffset + (1 * me + 0 + 12)*1] = (*R3).y;
    barrier(CLK_LOCAL_MEM_FENCE);
    (*R0).y = bufInIm[inOffset + (me + 1)*1];
    (*R1).y = bufInIm[inOffset + (me + 5)*1];
    (*R2).y = bufInIm[inOffset + (16 - (me + 1))*1];
    (*R3).y = bufInIm[inOffset + (16 - (me + 5))*1];
    if ((rw > 1) && !me) {
        bufOut2[outOffset].x = bufInIm[inOffset];
        bufOut2[outOffset].y = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (rw) {
        bufOut[outOffset + (me + 1)*1] = (double2) (((*R0).x + (*R2).x)*0.5, +((*R0).y - (*R2).y)*0.5);
        bufOut[outOffset + (me + 5)*1] = (double2) (((*R1).x + (*R3).x)*0.5, +((*R1).y - (*R3).y)*0.5);
    }
    if (rw > 1) {
        bufOut2[outOffset + (me + 1)*1] = (double2) (((*R0).y + (*R2).y)*0.5, +(-(*R0).x + (*R2).x)*0.5);
        bufOut2[outOffset + (me + 5)*1] = (double2) (((*R1).y + (*R3).y)*0.5, +(-(*R1).x + (*R3).x)*0.5);
    }
}

typedef union {
    uint u;
    int i;
} cb_t;

__kernel __attribute__ ((reqd_work_group_size(64, 1, 1)))
void fft_fwd(__constant cb_t *cb __attribute__ ((max_constant_size(32))), __global double * restrict gbIn, __global double2 * restrict gbOut, ulong in_offs, ulong out_offs) {
    uint me = get_local_id(0);
    uint batch = get_group_id(0);
    __local double lds[256];

    uint iOffset;
    uint oOffset;
    uint iOffset2;
    uint oOffset2;
    __global double *lwbIn2;
    __global double *lwbIn;
    __global double2 *lwbOut2;
    __global double2 *lwbOut;
    double2 R0, R1, R2, R3;
    uint this = (cb[0].u) - batch * 32;
    uint rw = (me < ((this + 1) / 2)*4) ? (this - 2 * (me / 4)) : 0;
    uint b = 0;
    iOffset = (batch * 32 + 0 + 2 * (me / 4))*16;
    oOffset = (batch * 32 + 0 + 2 * (me / 4))*16;
    iOffset2 = (batch * 32 + 1 + 2 * (me / 4))*16;
    oOffset2 = (batch * 32 + 1 + 2 * (me / 4))*16;
    lwbIn2 = gbIn + in_offs + iOffset2;
    lwbIn = gbIn + in_offs + iOffset;
    lwbOut2 = gbOut + out_offs + oOffset2;
    lwbOut = gbOut + out_offs + oOffset;
    FwdPass0(rw, b, me % 4, 0, (me / 4)*16, lwbIn, lwbIn2, lds, lds, &R0, &R1, &R2, &R3);
    FwdPass1(rw, b, me % 4, (me / 4)*16, 0, lds, lds, lwbOut, lwbOut2, &R0, &R1, &R2, &R3);
}

__attribute__ ((always_inline)) void
InvPass0(uint rw, uint b, uint me, uint inOffset, uint outOffset, __global double2 *bufIn, __global double2 *bufIn2, __local double *bufOutRe, __local double *bufOutIm, double2 *R0, double2 *R1, double2 *R2, double2 *R3) {
    if (rw && !me) {
        bufOutRe[outOffset] = bufIn[inOffset].x;
    }
    if (rw) {
        (*R0).x = bufIn[inOffset + (me + 1)*1].x;
        (*R1).x = bufIn[inOffset + (me + 5)*1].x;
    }
    if (rw > 1) {
        (*R2).x = bufIn2[inOffset + (me + 1)*1].y;
        (*R3).x = bufIn2[inOffset + (me + 5)*1].y;
    } else {
        (*R2).x = 0;
        (*R3).x = 0;
    }
    bufOutRe[outOffset + (16 - (me + 1))*1] = ((*R0).x + (*R2).x);
    bufOutRe[outOffset + (16 - (me + 5))*1] = ((*R1).x + (*R3).x);
    bufOutRe[outOffset + (me + 1)*1] = ((*R0).x - (*R2).x);
    bufOutRe[outOffset + (me + 5)*1] = ((*R1).x - (*R3).x);
    barrier(CLK_LOCAL_MEM_FENCE);
    (*R0).x = bufOutRe[outOffset + (0 + me * 1 + 0 + 0)*1];
    (*R1).x = bufOutRe[outOffset + (0 + me * 1 + 0 + 4)*1];
    (*R2).x = bufOutRe[outOffset + (0 + me * 1 + 0 + 8)*1];
    (*R3).x = bufOutRe[outOffset + (0 + me * 1 + 0 + 12)*1];
    barrier(CLK_LOCAL_MEM_FENCE);
    if ((rw > 1) && !me) {
        bufOutIm[outOffset] = bufIn2[inOffset].x;
    }
    if ((rw == 1) && !me) {
        bufOutIm[outOffset] = 0;
    }
    if (rw) {
        (*R0).y = bufIn[inOffset + (me + 1)*1].y;
        (*R1).y = bufIn[inOffset + (me + 5)*1].y;
    }
    if (rw > 1) {
        (*R2).y = bufIn2[inOffset + (me + 1)*1].x;
        (*R3).y = bufIn2[inOffset + (me + 5)*1].x;
    } else {
        (*R2).y = 0;
        (*R3).y = 0;
    }
    bufOutIm[outOffset + (16 - (me + 1))*1] = (-(*R0).y + (*R2).y);
    bufOutIm[outOffset + (16 - (me + 5))*1] = (-(*R1).y + (*R3).y);
    bufOutIm[outOffset + (me + 1)*1] = ((*R0).y + (*R2).y);
    bufOutIm[outOffset + (me + 5)*1] = ((*R1).y + (*R3).y);
    barrier(CLK_LOCAL_MEM_FENCE);
    (*R0).y = bufOutIm[outOffset + (0 + me * 1 + 0 + 0)*1];
    (*R1).y = bufOutIm[outOffset + (0 + me * 1 + 0 + 4)*1];
    (*R2).y = bufOutIm[outOffset + (0 + me * 1 + 0 + 8)*1];
    (*R3).y = bufOutIm[outOffset + (0 + me * 1 + 0 + 12)*1];
    barrier(CLK_LOCAL_MEM_FENCE);
    InvRad4B1(R0, R1, R2, R3);
    if (rw) {
        bufOutRe[outOffset + (((1 * me + 0) / 1)*4 + (1 * me + 0) % 1 + 0)*1] = (*R0).x;
        bufOutRe[outOffset + (((1 * me + 0) / 1)*4 + (1 * me + 0) % 1 + 1)*1] = (*R1).x;
        bufOutRe[outOffset + (((1 * me + 0) / 1)*4 + (1 * me + 0) % 1 + 2)*1] = (*R2).x;
        bufOutRe[outOffset + (((1 * me + 0) / 1)*4 + (1 * me + 0) % 1 + 3)*1] = (*R3).x;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (rw) {
        (*R0).x = bufOutRe[outOffset + (0 + me * 1 + 0 + 0)*1];
        (*R1).x = bufOutRe[outOffset + (0 + me * 1 + 0 + 4)*1];
        (*R2).x = bufOutRe[outOffset + (0 + me * 1 + 0 + 8)*1];
        (*R3).x = bufOutRe[outOffset + (0 + me * 1 + 0 + 12)*1];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (rw) {
        bufOutIm[outOffset + (((1 * me + 0) / 1)*4 + (1 * me + 0) % 1 + 0)*1] = (*R0).y;
        bufOutIm[outOffset + (((1 * me + 0) / 1)*4 + (1 * me + 0) % 1 + 1)*1] = (*R1).y;
        bufOutIm[outOffset + (((1 * me + 0) / 1)*4 + (1 * me + 0) % 1 + 2)*1] = (*R2).y;
        bufOutIm[outOffset + (((1 * me + 0) / 1)*4 + (1 * me + 0) % 1 + 3)*1] = (*R3).y;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (rw) {
        (*R0).y = bufOutIm[outOffset + (0 + me * 1 + 0 + 0)*1];
        (*R1).y = bufOutIm[outOffset + (0 + me * 1 + 0 + 4)*1];
        (*R2).y = bufOutIm[outOffset + (0 + me * 1 + 0 + 8)*1];
        (*R3).y = bufOutIm[outOffset + (0 + me * 1 + 0 + 12)*1];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
}

__attribute__ ((always_inline)) void
InvPass1(uint rw, uint b, uint me, uint inOffset, uint outOffset, __local double *bufInRe, __local double *bufInIm, __global double *bufOut, __global double *bufOut2, double2 *R0, double2 *R1, double2 *R2, double2 *R3) {
    {
        double2 W = twiddles[3 + 3 * ((1 * me + 0) % 4) + 0];
        double TR, TI;
        TR = (W.x * (*R1).x) + (W.y * (*R1).y);
        TI = -(W.y * (*R1).x) + (W.x * (*R1).y);
        (*R1).x = TR;
        (*R1).y = TI;
    }
    {
        double2 W = twiddles[3 + 3 * ((1 * me + 0) % 4) + 1];
        double TR, TI;
        TR = (W.x * (*R2).x) + (W.y * (*R2).y);
        TI = -(W.y * (*R2).x) + (W.x * (*R2).y);
        (*R2).x = TR;
        (*R2).y = TI;
    }
    {
        double2 W = twiddles[3 + 3 * ((1 * me + 0) % 4) + 2];
        double TR, TI;
        TR = (W.x * (*R3).x) + (W.y * (*R3).y);
        TI = -(W.y * (*R3).x) + (W.x * (*R3).y);
        (*R3).x = TR;
        (*R3).y = TI;
    }
    InvRad4B1(R0, R1, R2, R3);
    if (rw) {
        bufOut[outOffset + (1 * me + 0 + 0)*1] += (*R0).x * 6.2500000000000000e-02;
        bufOut[outOffset + (1 * me + 0 + 4)*1] += (*R1).x * 6.2500000000000000e-02;
        bufOut[outOffset + (1 * me + 0 + 8)*1] += (*R2).x * 6.2500000000000000e-02;
        bufOut[outOffset + (1 * me + 0 + 12)*1] += (*R3).x * 6.2500000000000000e-02;
    }
    if (rw > 1) {
        bufOut2[outOffset + (1 * me + 0 + 0)*1] += (*R0).y * 6.2500000000000000e-02;
        bufOut2[outOffset + (1 * me + 0 + 4)*1] += (*R1).y * 6.2500000000000000e-02;
        bufOut2[outOffset + (1 * me + 0 + 8)*1] += (*R2).y * 6.2500000000000000e-02;
        bufOut2[outOffset + (1 * me + 0 + 12)*1] += (*R3).y * 6.2500000000000000e-02;
    }
}

__kernel __attribute__ ((reqd_work_group_size(64, 1, 1)))
void fft_back(__constant cb_t *cb __attribute__ ((max_constant_size(32))), __global double2 * restrict gbIn, __global double * restrict gbOut, ulong in_offs, ulong out_offs) {
    uint me = get_local_id(0);
    uint batch = get_group_id(0);
    __local double lds[256];
    uint iOffset;
    uint oOffset;
    uint iOffset2;
    uint oOffset2;
    __global double2 *lwbIn2;
    __global double2 *lwbIn;
    __global double *lwbOut2;
    __global double *lwbOut;
    double2 R0, R1, R2, R3;
    uint this = (cb[0].u) - batch * 32;
    uint rw = (me < ((this + 1) / 2)*4) ? (this - 2 * (me / 4)) : 0;
    uint b = 0;
    iOffset = (batch * 32 + 0 + 2 * (me / 4))*16;
    oOffset = (batch * 32 + 0 + 2 * (me / 4))*16;
    iOffset2 = (batch * 32 + 1 + 2 * (me / 4))*16;
    oOffset2 = (batch * 32 + 1 + 2 * (me / 4))*16;
    lwbIn2 = gbIn + in_offs +iOffset2;
    lwbIn = gbIn + in_offs  + iOffset;
    lwbOut2 = gbOut + out_offs  + oOffset2;
    lwbOut = gbOut + out_offs  + oOffset;
    InvPass0(rw, b, me % 4, 0, (me / 4)*16, lwbIn, lwbIn2, lds, lds, &R0, &R1, &R2, &R3);
    InvPass1(rw, b, me % 4, (me / 4)*16, 0, lds, lds, lwbOut, lwbOut2, &R0, &R1, &R2, &R3);
}
