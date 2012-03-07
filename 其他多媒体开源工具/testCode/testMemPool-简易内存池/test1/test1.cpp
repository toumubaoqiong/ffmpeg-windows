// test1.cpp : 定义控制台应用程序的入口点。
//
#include <assert.h>
#include <stdlib.h>
#include <memory.h>

#define ASSERTTRY(e)					assert(e)
#define ASSERTMUST(e)					assert(e)
#define TRUE	1
#define FALSE	0
typedef int BOOL;

#define MEDIAEXMEMBUF_DEFAULTSIZE 1024*1024*2
struct MediaEx_memBuffer
{
private:
	char* memBuf;
	int memBufSize;
	int start;
	int end;
	int *lastMemLen;
public:
	MediaEx_memBuffer(const int SetMemBufSize = MEDIAEXMEMBUF_DEFAULTSIZE):
	  memBuf(NULL), memBufSize(0), start(0), end(0), lastMemLen(NULL)
	  {
		  if(SetMemBufSize < MEDIAEXMEMBUF_DEFAULTSIZE)
		  {
			  memBufSize = MEDIAEXMEMBUF_DEFAULTSIZE;
		  }
		  else
		  {
			  memBufSize = SetMemBufSize;
		  }
		  memBuf = (char*)malloc(memBufSize);
		  ASSERTTRY(memBuf);
		  ASSERTMUST(memBuf);
	  }
	  ~MediaEx_memBuffer()
	  {
		  ASSERTTRY(start == end);
		  ASSERTMUST(start == end);
		  if(start == end)
		  {
			  free(memBuf);
			  memBuf = NULL;
			  memBufSize = 0;
			  start = end = 0;
		  }
	  }
public:
	void* membuf_malloc(const int mallocSize)
	{
		if(mallocSize <= 0)
		{
			return NULL;
		}
		char *tmpMallocBuf = memBuf+end;
		if(end < start)
		{
			if(end + mallocSize >= start)
			{
				return NULL;
			}
			else
			{
				end += mallocSize;
			}
		}
		else 
		{
			if(end + mallocSize >= memBufSize)
			{
				ASSERTTRY(lastMemLen);
				ASSERTMUST(lastMemLen);
				tmpMallocBuf = memBuf+end;
				if(mallocSize >= start)
				{
					return NULL;
				}
				else
				{
					end = 0;
					*lastMemLen += (memBufSize - end);
					end += mallocSize;
				}
			}
			else
			{
				end += mallocSize;
			}
		}
		lastMemLen = (int*)tmpMallocBuf;
		*lastMemLen = mallocSize;
		return tmpMallocBuf;
	}

	BOOL membuf_free(void** const freeBuf)
	{
		if(!freeBuf || !*freeBuf)
		{
			return FALSE;
		}
		char *buf = (char*)(*freeBuf);
		const int len = *((int*)buf);
		ASSERTTRY(len > 0);
		ASSERTTRY(buf == memBuf+start);
		if(start+len < memBufSize)
		{
			start += len;
		}
		else if(start+len == memBufSize)
		{
			start = 0;
		}
		else
		{
			ASSERTTRY(FALSE);
			ASSERTMUST(FALSE);
		}
		if(start == end)
		{
			start = end = 0;
			lastMemLen = NULL;
		}
		*freeBuf = NULL;
		return TRUE;
	}
};

struct MediaExList_Mem
{
public:
	int dataLen;
	void *mem;
	MediaExList_Mem* next;
};

struct MediaEx_list
{
public:
	MediaEx_list(void):head(NULL), tail(NULL){}
public:
	BOOL pushHead(MediaExList_Mem *const mem)
	{
		if(!mem)
		{
			return FALSE;
		}
		if(!head)
		{
			ASSERTTRY(!tail);
			mem->next = NULL;
			head = mem;
			tail = mem;
		}
		else
		{
			head = mem;
		}
		return TRUE;
	}
	MediaExList_Mem* popBack(void)
	{
		if(!head)
		{
			ASSERTTRY(!tail);
			return NULL;
		}
		MediaExList_Mem *tmpMem = NULL;;
		if(head == tail)
		{
			tmpMem = head;
			head = tail = NULL;
		}
		else 
		{
			tmpMem = head;
			head = head->next;
		}
		return tmpMem;
	}
private:
	MediaExList_Mem *head;
	MediaExList_Mem *tail;
};

struct dataMem1
{
	float memfloat;
	int memInt;
	double memDouble;
};

struct membuf_dataMem1
{
	int dataLen;
	dataMem1 *mem;
};

void testfun1(void)
{
	MediaEx_memBuffer tmpMemBuf;
	int tmpDataLen = sizeof(membuf_dataMem1) + sizeof(dataMem1);
	membuf_dataMem1 *tmpDataMem1 = NULL;
	int i = 0;
	for (i = 0; i < 10000000; i++)
	{
		tmpDataMem1 = (membuf_dataMem1 *)tmpMemBuf.membuf_malloc(tmpDataLen);
		tmpDataMem1->mem = (dataMem1*)(((char*)tmpDataMem1)+sizeof(membuf_dataMem1));
		tmpDataMem1->mem->memfloat = 0.0;
		tmpDataMem1->mem->memInt = 0;
		tmpDataMem1->mem->memDouble = 0.00;
		tmpMemBuf.membuf_free((void**)&tmpDataMem1);	
	}
// 	for (i = 0; i < 10000000; i++)
// 	{
// 		tmpDataMem1 = (membuf_dataMem1 *)tmpMemBuf.membuf_malloc(&tmpDataLen);
// 		if(!tmpDataMem1)
// 		{
// 			continue;
// 		}
// 		memset(tmpDataMem1, 0, tmpDataLen);
// 		tmpDataMem1->mem = (dataMem1*)(((char*)tmpDataMem1)+sizeof(membuf_dataMem1));
// 		tmpDataMem1->mem->memfloat = 0.0;
// 		tmpDataMem1->mem->memInt = 0;
// 		tmpDataMem1->mem->memDouble = 0.00;	
// 	}
}

void testfun1_one(void)
{
	MediaEx_memBuffer tmpMemBuf;
	int tmpDataLen = sizeof(membuf_dataMem1) + sizeof(dataMem1);
	membuf_dataMem1 *tmpDataMem1 = NULL;
	int i = 0;
	for (i = 0; i < 10000000; i++)
	{
		tmpDataMem1 = (membuf_dataMem1 *)tmpMemBuf.membuf_malloc(tmpDataLen);
		tmpMemBuf.membuf_free((void**)&tmpDataMem1);	
	}
}

void testfun2(void)
{
	int tmpDataLen = sizeof(membuf_dataMem1) + sizeof(dataMem1);
	membuf_dataMem1 *tmpDataMem1 = NULL;
	for (int i = 0; i < 10000000; i++)
	{
		tmpDataMem1 = (membuf_dataMem1 *)malloc(tmpDataLen);
		if(!tmpDataLen)
		{
			continue;
		}
		memset(tmpDataMem1, 0, tmpDataLen);
		tmpDataMem1->mem = (dataMem1*)(((char*)tmpDataMem1)+sizeof(membuf_dataMem1));
		tmpDataMem1->mem->memfloat = 0.0;
		tmpDataMem1->mem->memInt = 0;
		tmpDataMem1->mem->memDouble = 0.00;
		free(tmpDataMem1);
	}
}


void testfun2_one(void)
{
	int tmpDataLen = sizeof(membuf_dataMem1) + sizeof(dataMem1);
	membuf_dataMem1 *tmpDataMem1 = NULL;
	for (int i = 0; i < 10000000; i++)
	{
		tmpDataMem1 = (membuf_dataMem1 *)malloc(tmpDataLen);
		if(!tmpDataLen)
		{
			continue;
		}
		memset(tmpDataMem1, 0, tmpDataLen);
		free(tmpDataMem1);
	}
}

void testfun3(void)
{
	MediaEx_memBuffer tmpMemBuf;
	MediaEx_list tmpList;
	int tmpDataLen = sizeof(MediaExList_Mem) + sizeof(dataMem1);
	MediaExList_Mem *tmpListMem = NULL;
	dataMem1* tmpDataMem1;
	int i = 0;
	for (i = 0; i < 10000000; i++)
	{
		tmpListMem = (MediaExList_Mem *)tmpMemBuf.membuf_malloc(tmpDataLen);
		tmpListMem->mem = (dataMem1*)(((char*)tmpListMem)+sizeof(MediaExList_Mem));
		tmpDataMem1 = (dataMem1*)(tmpListMem->mem);
		tmpDataMem1->memfloat = 0.0;
		tmpDataMem1->memInt = 0;
		tmpDataMem1->memDouble = 0.00;
		tmpList.pushHead(tmpListMem);
		tmpListMem = tmpList.popBack();
		tmpMemBuf.membuf_free((void**)&tmpListMem);	
	}
}

int main(void)
{
	testfun1_one();
	return 0;
}

