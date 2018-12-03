#ifndef _RKNN_API_H
#define _RKNN_API_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

/*
    Definition of extended flag for rknn_init.
*/
/* set high priority context. */
#define RKNN_FLAG_PRIOR_HIGH                    0x00000000

/* set medium priority context */
#define RKNN_FLAG_PRIOR_MEDIUM                  0x00000001

/* set low priority context. */
#define RKNN_FLAG_PRIOR_LOW                     0x00000002

/* asynchronous mode.
   when enable, rknn_outputs_get will not block for too long because it directly retrieves the result of
   the previous frame which can increase the frame rate on single-threaded mode, but at the cost of
   rknn_outputs_get not retrieves the result of the current frame.
   in multi-threaded mode you do not need to turn this mode on. */
#define RKNN_FLAG_ASYNC_MASK                    0x00000004

/* collect performance mode.
   when enable, you can get detailed performance reports via rknn_query(ctx, RKNN_QUERY_PERF_DETAIL, ...),
   but it will reduce the frame rate. */
#define RKNN_FLAG_COLLECT_PERF_MASK             0x00000008

/*
    Error code returned by the RKNN API.
*/
#define RKNN_SUCC                               0       /* execute succeed. */
#define RKNN_ERR_FAIL                           -1      /* execute failed. */
#define RKNN_ERR_TIMEOUT                        -2      /* execute timeout. */
#define RKNN_ERR_DEVICE_UNAVAILABLE             -3      /* device is unavailable. */
#define RKNN_ERR_MALLOC_FAIL                    -4      /* memory malloc fail. */
#define RKNN_ERR_PARAM_INVALID                  -5      /* parameter is invalid. */
#define RKNN_ERR_MODEL_INVALID                  -6      /* model is invalid. */
#define RKNN_ERR_CTX_INVALID                    -7      /* context is invalid. */
#define RKNN_ERR_INPUT_INVALID                  -8      /* input is invalid. */
#define RKNN_ERR_OUTPUT_INVALID                 -9      /* output is invalid. */
#define RKNN_ERR_DEVICE_UNMATCH                 -10     /* the device is unmatch, please update rknn sdk
                                                           and npu driver/firmware. */

/*
    Definition for tensor
*/
#define RKNN_MAX_DIMS                           16      /* maximum dimension of tensor. */
#define RKNN_MAX_NAME_LEN                       256     /* maximum name lenth of tensor. */

/*
    The query command for rknn_query
*/
enum rknn_query_cmd {
    RKNN_QUERY_IN_OUT_NUM = 0,                          /* query the number of input & output tensor. */
    RKNN_QUERY_INPUT_ATTR,                              /* query the attribute of input tensor. */
    RKNN_QUERY_OUTPUT_ATTR,                             /* query the attribute of output tensor. */
    RKNN_QUERY_PERF_DETAIL,                             /* query the detail performance, need set
                                                           RKNN_FLAG_COLLECT_PERF_MASK when call rknn_init. */
    RKNN_QUERY_PERF_RUN,                                /* query the time of run. */

    RKNN_QUERY_MAX
};

/*
    the information for RKNN_QUERY_IN_OUT_NUM.
*/
struct rknn_input_output_num {
    uint32_t n_input;                                   /* the number of input. */
    uint32_t n_output;                                  /* the number of output. */
};

/*
    the tensor data type.
*/
enum rknn_tensor_type {
    RKNN_TENSOR_FLOAT32 = 0,                            /* data type is float32. */
    RKNN_TENSOR_FLOAT16,                                /* data type is float16. */
    RKNN_TENSOR_INT8,                                   /* data type is int8. */
    RKNN_TENSOR_UINT8,                                  /* data type is uint8. */
    RKNN_TENSOR_INT16                                   /* data type is int16. */
};

/*
    the quantitative type.
*/
enum rknn_tensor_qnt_type {
    RKNN_TENSOR_QNT_NONE = 0,                           /* none. */
    RKNN_TENSOR_QNT_DFP,                                /* dynamic fixed point. */
    RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC                   /* asymmetric affine. */
};

/*
    the information for RKNN_QUERY_INPUT_ATTR / RKNN_QUERY_OUTPUT_ATTR.
*/
struct rknn_tensor_attr {
    uint32_t index;                                     /* input parameter, the index of input/output tensor,
                                                           need set before call rknn_query. */

    uint32_t n_dims;                                    /* the number of dimensions. */
    uint32_t dims[RKNN_MAX_DIMS];                       /* the dimensions array. */
    char name[RKNN_MAX_NAME_LEN];                       /* the name of tensor. */

    uint32_t n_elems;                                   /* the number of elements. */
    uint32_t size;                                      /* the bytes size of tensor. */

    uint8_t type;                                       /* see rknn_tensor_type. */
    uint8_t qnt_type;                                   /* see rknn_tensor_qnt_type. */
    int8_t fl;                                          /* fractional length for RKNN_TENSOR_QNT_DFP. */
    uint32_t zp;                                        /* zero point for RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC. */
    float scale;                                        /* scale for RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC. */
};

/*
    the information for RKNN_QUERY_PERF_DETAIL.
*/
struct rknn_perf_detail {
    char* perf_data;                                    /* the string pointer of perf detail. don't need free it by user. */
    uint64_t data_len;                                  /* the string length. */
};

/*
    the information for RKNN_QUERY_PERF_RUN.
*/
struct rknn_perf_run {
    int64_t run_duration;                               /* real inference time (us) */
};

/*
    set the input data order for rknn_input_set.
    this must match the input settings in rknn-toolkit.
*/
enum rknn_input_order_type {
    RKNN_INPUT_ORDER_NO = 0,                            /* do nothing. */
    RKNN_INPUT_ORDER_012,                               /* reoder the input shape from HWC to CHW when intput is RGB888.
                                                           e.g. [RGBRGBRGBRGB] => [[RRRR] [GGGG] [BBBB]] */
    RKNN_INPUT_ORDER_210,                               /* reoder the input shape from HWC to CHW when intput is RGB888,
                                                           and then swap the R & B.
                                                           e.g. [RGBRGBRGBRGB] => [[BBBB] [GGGG] [RRRR]] */

    RKNN_INPUT_ORDER_MAX
};

/*
    the output information for rknn_outputs_get.
*/
struct rknn_output {
    uint32_t index;                                     /* the output index. */
    void* buf;                                          /* the output buf for index. when rknn_outputs_release call,
                                                           this buf pointer will be free and don't use it anymore. */
    uint32_t size;                                      /* the size of output buf. */
};

/*
    the extend information for rknn_run.
*/
struct rknn_run_extend {
    uint64_t frame_id;                                  /* output parameter, indicate current frame id of run. */
};

/*
    the extend information for rknn_outputs_get.
*/
struct rknn_output_extend {
    uint64_t frame_id;                                  /* output parameter, indicate the frame id of outputs, corresponds to
                                                           struct rknn_run_extend.frame_id.*/
};



/*  rknn_init

    load the rknn model and initial the context.

    input:
        void* model         pointer to the rknn model.
        int len             the size of rknn model.
        uint32_t flag       extend flag, see the define of RKNN_FLAG_XXX_XXX.
    return:
        int                 >= 0    the handle of context.
                            < 0     error code.
*/
int rknn_init(void* model, int len, uint32_t flag);


/*  rknn_destroy

    unload the rknn model and destroy the context.

    input:
        int context         the handle of context.
    return:
        int                 error code.
*/
int rknn_destroy(int context);


/*  rknn_query

    query the information about model or others. see rknn_query_cmd.

    input:
        int context         the handle of context.
        rknn_query_cmd cmd  the command of query.
        void* info          the buffer point of information.
        int info_len        the length of information.
    return:
        int                 error code.
*/
int rknn_query(int context, enum rknn_query_cmd cmd, void* info, int info_len);


/*  rknn_input_set

    set input buffer by input index of rknn model.

    input:
        int context         the handle of context.
        int index           the input index in rknn model.
        void* buf           the input buffer pointer.
        int len             the length of the input buffer.
        uint8_t order       the rgb order of input buffer, see the define of rknn_input_order_type.
    return:
        int                 error code
*/
int rknn_input_set(int context, int index, void* buf, int len, uint8_t order);


/*  rknn_run

    run the model to execute inference.
    this function will not block any time.

    input:
        int context         the handle of context.
        struct rknn_run_extend* extend      the extend information of run.
    return:
        int                 error code.
*/
int rknn_run(int context, struct rknn_run_extend* extend);


/*  rknn_outputs_get

    wait the inference to finish and get the outputs.
    this function will block until inference finish.
    the results will set to outputs[].

    input:
        int context         the handle of context.
        int num             the number of outputs.
        struct rknn_output outputs[]    the arrays of output.
        struct rknn_output_extend*      the extend information of output.
    return:
        int                 >= 0    the handle of outputs.
                            < 0     error code.
*/
int rknn_outputs_get(int context, int num, struct rknn_output outputs[], struct rknn_output_extend* extend);


/*  rknn_outputs_release

    release the outputs handle that get by rknn_outputs_get.
    when handle released, the rknn_output[x].buf get from rknn_outputs_get will also be free.

    input:
        int context         the handle of context.
        int buf_handle      the handle of outputs.
    return:
        int                 error code
*/
int rknn_outputs_release(int context, int buf_handle);


/*  rknn_output_to_float32

    convert the output to float32.

    input:
        int context         the handle of context.
        struct rknn_output &output       the output get from rknn_outputs_get.
        void* dst           the destination buffer of float32.
        int size            the size of destination buffer.
    return:
        int                 error code
*/
int rknn_output_to_float(int context, const struct rknn_output &output, void* dst, int size);

#ifdef __cplusplus
} //extern "C"
#endif

#endif  //_RKNN_API_H
