#include "cudainit.h"

/*
extern void cuda_iwt_5_3_prerun();
extern void cuda_iwt_desl_9_3_prerun();
extern void cuda_iwt_haar_prerun();
extern void cuda_iwt_13_5_prerun();
extern void cuda_iwt_daub_9_7_prerun();
extern void cuda_iwt_fidelity_prerun();
extern void cuda_iiwt_5_3_prerun();
extern void cuda_iiwt_desl_9_3_prerun();
extern void cuda_iiwt_haar_prerun();
extern void cuda_iiwt_13_5_prerun();
extern void cuda_iiwt_daub_9_7_prerun();
extern void cuda_iiwt_fidelity_prerun();
*/
extern "C" {

void cudawl_init()
{
/*
    /// Prerun all kernels
    cuda_iwt_5_3_prerun();
    cuda_iwt_desl_9_3_prerun();
    cuda_iwt_haar_prerun();
    cuda_iwt_13_5_prerun();
    cuda_iwt_daub_9_7_prerun();
    cuda_iwt_fidelity_prerun();

    cuda_iiwt_5_3_prerun();
    cuda_iiwt_desl_9_3_prerun();
    cuda_iiwt_haar_prerun();
    cuda_iiwt_13_5_prerun();
    cuda_iiwt_daub_9_7_prerun();
    cuda_iiwt_fidelity_prerun();
*/
}

void cudawl_exit()
{
    /// Free temporary storage etc
}

};
