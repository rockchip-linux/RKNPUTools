// Create By randall.zhuo@rock-chips.com
// 2018/10/30

#ifndef MY_DIRECT_TEXTURE_HOOK_H
#define MY_DIRECT_TEXTURE_HOOK_H

class DirectTexture {
public:
	DirectTexture();
	~DirectTexture();

public:
	int (*createDirectTexture)(int texWidth, int texHeight, int format);
	bool (*deleteDirectTexture)(int texId);
	char* (*requireBufferByTexId)(int texId);
	bool* (*releaseBufferByTexId)(int texId);
private:
	void *mSoHandle;
};

extern DirectTexture  gDirectTexture;

#endif
