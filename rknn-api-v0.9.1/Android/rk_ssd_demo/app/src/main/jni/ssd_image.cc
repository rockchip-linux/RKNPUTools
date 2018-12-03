#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

#include <fcntl.h>      // NOLINT(build/include_order)
#include <getopt.h>     // NOLINT(build/include_order)
#include <sys/time.h>   // NOLINT(build/include_order)
#include <sys/types.h>  // NOLINT(build/include_order)
#include <sys/uio.h>    // NOLINT(build/include_order)
#include <unistd.h>     // NOLINT(build/include_order)
#include <string.h>
#include "rknn_api.h"

#include "ssd_image.h"
#include "direct_texture.h"

namespace ssd_image {

int ctx = -1;
bool created = false;

const int input_index = 0;      // node name "Preprocessor/sub"
const int output_index0 = 0;    // node name "concat"
const int output_index1 = 1;    // node name "concat_1"

int img_width = 0;
int img_height = 0;
int img_channels = 0;

int output_elems0 = 0;
int output_size0 = 0;
int output_elems1 = 0;
int output_size1 = 0;

struct rknn_tensor_attr outputs_attr[2];

void create(int inputSize, int channel, int numResult, int numClasses, char *mParamPath)
{
    img_width = inputSize;
    img_height = inputSize;
    img_channels = channel;

    output_elems0 = numResult * 4;
    output_size0 = output_elems0 * sizeof(float);
    output_elems1 = numResult * numClasses;
    output_size1 = output_elems1 * sizeof(float);

    LOGI("try rknn_init!");

    // Load model
    FILE *fp = fopen(mParamPath, "rb");
    if(fp == NULL) {
        LOGE("fopen %s fail!\n", mParamPath);
        return;
    }
    fseek(fp, 0, SEEK_END);
    int model_len = ftell(fp);
    void *model = malloc(model_len);
    fseek(fp, 0, SEEK_SET);
    if(model_len != fread(model, 1, model_len, fp)) {
        LOGE("fread %s fail!\n", mParamPath);
        free(model);
        fclose(fp);
        return;
    }

    fclose(fp);

    // RKNN_FLAG_ASYNC_MASK: enable async mode to use NPU efficiently.
    ctx = rknn_init(model, model_len, RKNN_FLAG_PRIOR_MEDIUM|RKNN_FLAG_ASYNC_MASK);
    free(model);

    if(ctx < 0) {
        LOGE("rknn_init fail! ret=%d\n", ctx);
        return;
    }

    outputs_attr[0].index = output_index0;
    int ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(outputs_attr[0]), sizeof(outputs_attr[0]));
    if(ret < 0) {
        LOGI("rknn_query fail! ret=%d\n", ret);
        return;
    }

    outputs_attr[1].index = output_index1;
    ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(outputs_attr[1]), sizeof(outputs_attr[1]));
    if(ret < 0) {
        LOGI("rknn_query fail! ret=%d\n", ret);
        return;
    }
    created = true;
    LOGI("rknn_init success!");

}

void destroy() {
    LOGI("rknn_destroy!");
    rknn_destroy(ctx);
}

bool run_ssd(char *inData, float *y0, float *y1)
{
    if(!created) {
        LOGE("run_ssd: create hasn't successful!");
        return false;
    }

    int ret = rknn_input_set(ctx, input_index, inData, img_width * img_height * img_channels, RKNN_INPUT_ORDER_012);
    if(ret < 0) {
        LOGE("rknn_input_set fail! ret=%d\n", ret);
        return false;
    }

    ret = rknn_run(ctx, nullptr);
    if(ret < 0) {
        LOGE("rknn_run fail! ret=%d\n", ret);
        return false;
    }

    struct rknn_output outputs[2];
    int h_output = rknn_outputs_get(ctx, 2, outputs, nullptr);
    if(h_output < 0) {
        LOGE("rknn_outputs_get fail! ret=%d\n", ret);
        return false;
    }

    if(outputs[0].size == outputs_attr[0].size && outputs[1].size == outputs_attr[1].size) {
    #if 0
        if(outputs_attr[0].type != RKNN_TENSOR_FLOAT32) {
            rknn_output_to_float(ctx, outputs[0], y0, output_size0);
        } else {
            memcpy(y0, outputs[0].buf, output_size0);
        }

        if(outputs_attr[1].type != RKNN_TENSOR_FLOAT32) {
            rknn_output_to_float(ctx, outputs[1], y1, output_size1);
        } else {
            memcpy(y1, outputs[1].buf, output_size1);
        }
    #else   //dkm: for workround the wrong order issue of output index.
        if(outputs_attr[1].type != RKNN_TENSOR_FLOAT32) {
            rknn_output_to_float(ctx, outputs[1], y0, output_size0);
        } else {
            memcpy(y0, outputs[1].buf, output_size0);
        }

        if(outputs_attr[0].type != RKNN_TENSOR_FLOAT32) {
            rknn_output_to_float(ctx, outputs[0], y1, output_size1);
        } else {
            memcpy(y1, outputs[0].buf, output_size1);
        }
    #endif
    } else {
        LOGI("rknn_outputs_get fail! get outputs_size = [%d, %d], but expect [%d, %d]!\n",
            outputs[0].size, outputs[1].size, outputs_attr[0].size, outputs_attr[1].size);
    }


    rknn_outputs_release(ctx, h_output);
    return true;
}


bool run_ssd(int texId, float *y0, float *y1)
{
    if(!created) {
        LOGE("run_ssd: create hasn't successful!");
        return false;
    }

    char *inData = gDirectTexture.requireBufferByTexId(texId);

    if (inData == nullptr) {
        LOGE("run_ssd: invalid texture, id=%d!", texId);
        return false;
    }

    int ret = rknn_input_set(ctx, input_index, inData, img_width * img_height * img_channels, RKNN_INPUT_ORDER_012);

    gDirectTexture.releaseBufferByTexId(texId);

    if(ret < 0) {
        LOGE("rknn_input_set fail! ret=%d\n", ret);
        return false;
    }

    ret = rknn_run(ctx, nullptr);
    if(ret < 0) {
        LOGE("rknn_run fail! ret=%d\n", ret);
        return false;
    }

    struct rknn_output outputs[2];

    int h_output = rknn_outputs_get(ctx, 2, outputs, nullptr);
    if(h_output < 0) {
        LOGE("rknn_outputs_get fail! ret=%d\n", ret);
        return false;
    }

    if(outputs[0].size == outputs_attr[0].size && outputs[1].size == outputs_attr[1].size) {
    #if 0
        if(outputs_attr[0].type != RKNN_TENSOR_FLOAT32) {
            rknn_output_to_float(ctx, outputs[0], y0, output_size0);
        } else {
            memcpy(y0, outputs[0].buf, output_size0);
        }

        if(outputs_attr[1].type != RKNN_TENSOR_FLOAT32) {
            rknn_output_to_float(ctx, outputs[1], y1, output_size1);
        } else {
            memcpy(y1, outputs[1].buf, output_size1);
        }
    #else   //dkm: for workround the wrong order issue of output index.
        if(outputs_attr[1].type != RKNN_TENSOR_FLOAT32) {
            rknn_output_to_float(ctx, outputs[1], y0, output_size0);
        } else {
            memcpy(y0, outputs[1].buf, output_size0);
        }

        if(outputs_attr[0].type != RKNN_TENSOR_FLOAT32) {
            rknn_output_to_float(ctx, outputs[0], y1, output_size1);
        } else {
            memcpy(y1, outputs[0].buf, output_size1);
        }
    #endif
    } else {
        LOGI("rknn_outputs_get fail! get outputs_size = [%d, %d], but expect [%d, %d]!\n",
            outputs[0].size, outputs[1].size, outputs_attr[0].size, outputs_attr[1].size);
    }

    rknn_outputs_release(ctx, h_output);
    return true;
}

}  // namespace label_image



