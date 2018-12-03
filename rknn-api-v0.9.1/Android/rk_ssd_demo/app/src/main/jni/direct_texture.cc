// Create By randall.zhuo@rock-chips.com
// 2018/10/30

#include <dlfcn.h>
#include <stdio.h>

#include "direct_texture.h"

#include <android/log.h>

#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, "rkssd4j", ##__VA_ARGS__);
#define LOGE(...) __android_log_print(ANDROID_LOG_INFO, "rkssd4j", ##__VA_ARGS__);


DirectTexture  gDirectTexture;

DirectTexture::DirectTexture() {
	char *filename = "libDirectTexture.so";

	mSoHandle = dlopen(filename, RTLD_NOW);

	const char *error = dlerror();

	if (error != NULL) {
		LOGE("dlopen %s fail: %s\n", filename, error);
		return;
	}

	createDirectTexture = (int (*)(int , int , int ))dlsym(mSoHandle, "createDirectTexture");
    deleteDirectTexture = (bool (*)(int ))dlsym(mSoHandle, "deleteDirectTexture");
    requireBufferByTexId = (char* (*)(int ))dlsym(mSoHandle, "requireBufferByTexId");
    releaseBufferByTexId = (bool* (*)(int ))dlsym(mSoHandle, "releaseBufferByTexId");

	error = dlerror();
	if (error != NULL) {
		LOGE("dlsym %s fail: %s\n", filename, error);
		dlclose(mSoHandle);
		mSoHandle = NULL;
		return;
	}	

}

DirectTexture::~DirectTexture() {
	if (mSoHandle != NULL) {
		dlclose(mSoHandle);
		mSoHandle = NULL;
	}
}

