#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "libavcodec/imgconvert.h"
#include <SDL/SDL.h>
#include <SDL/SDL_thread.h>

#undef main

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include "libavformat/avformat.h"
#include "libswscale/swscale.h"

typedef struct PacketQueue {
    AVPacketList *first_pkt, *last_pkt;
    int nb_packets;
    int size;
    int abort_request;
    SDL_mutex *mutex;
    SDL_cond *cond;
} PacketQueue;
PacketQueue audioq;
static AVPacket flush_pkt;

#define VIDEO_PICTURE_QUEUE_SIZE 1
#define SUBPICTURE_QUEUE_SIZE 4


#define FF_ALLOC_EVENT   (SDL_USEREVENT)
#define FF_REFRESH_EVENT (SDL_USEREVENT + 1)
#define FF_QUIT_EVENT    (SDL_USEREVENT + 2)


typedef struct VideoPicture {
    double pts;                                  ///<presentation time stamp for this picture
    SDL_Overlay *bmp;
    int width, height; /* source height & width */
    int allocated;
} VideoPicture;

typedef struct SubPicture {
    double pts; /* presentation time stamp for this picture */
    AVSubtitle sub;
} SubPicture;

typedef struct VideoState {
    SDL_Thread *parse_tid;
    SDL_Thread *video_tid;
    AVInputFormat *iformat;
    int no_background;
    int abort_request;
    int paused;
    int last_paused;
    int seek_req;
    int seek_flags;
    int64_t seek_pos;
    AVFormatContext *ic;
    int dtg_active_format;

    int audio_stream;

    int av_sync_type;
    double external_clock; /* external clock base */
    int64_t external_clock_time;

    double audio_clock;
    double audio_diff_cum; /* used for AV difference average computation */
    double audio_diff_avg_coef;
    double audio_diff_threshold;
    int audio_diff_avg_count;
    AVStream *audio_st;
    PacketQueue audioq;
    int audio_hw_buf_size;
    /* samples output by the codec. we reserve more space for avsync
       compensation */
    //DECLARE_ALIGNED(16,uint8_t,audio_buf1[(AVCODEC_MAX_AUDIO_FRAME_SIZE * 3) / 2]);
    //DECLARE_ALIGNED(16,uint8_t,audio_buf2[(AVCODEC_MAX_AUDIO_FRAME_SIZE * 3) / 2]);
    uint8_t *audio_buf;
    unsigned int audio_buf_size; /* in bytes */
    int audio_buf_index; /* in bytes */
    AVPacket audio_pkt;
    uint8_t *audio_pkt_data;
    int audio_pkt_size;
    enum SampleFormat audio_src_fmt;
    //AVAudioConvert *reformat_ctx;

    int show_audio; /* if true, display audio samples */
    //int16_t sample_array[SAMPLE_ARRAY_SIZE];
    int sample_array_index;
    int last_i_start;

    SDL_Thread *subtitle_tid;
    int subtitle_stream;
    int subtitle_stream_changed;
    AVStream *subtitle_st;
    PacketQueue subtitleq;
    SubPicture subpq[SUBPICTURE_QUEUE_SIZE];
    int subpq_size, subpq_rindex, subpq_windex;
    SDL_mutex *subpq_mutex;
    SDL_cond *subpq_cond;

    double frame_timer;
    double frame_last_pts;
    double frame_last_delay;
    double video_clock;                          ///<pts of last decoded frame / predicted pts of next decoded frame
    int video_stream;
    AVStream *video_st;
    PacketQueue videoq;
    double video_current_pts;                    ///<current displayed pts (different from video_clock if frame fifos are used)
    int64_t video_current_pts_time;              ///<time (av_gettime) at which we updated video_current_pts - used to have running video pts
    VideoPicture pictq[VIDEO_PICTURE_QUEUE_SIZE];
    int pictq_size, pictq_rindex, pictq_windex;
    SDL_mutex *pictq_mutex;
    SDL_cond *pictq_cond;

    //    QETimer *video_timer;
    char filename[1024];
    int width, height, xleft, ytop;
	int quit;
} VideoState;


SDL_Surface *screen;
//Init packet queue
static void packet_queue_init(PacketQueue *q)
{
    memset(q, 0, sizeof(PacketQueue));
    q->mutex = SDL_CreateMutex();
    q->cond = SDL_CreateCond();
}

static int packet_queue_put(PacketQueue *q, AVPacket *pkt)
{
    AVPacketList *pkt1;

    /* duplicate the packet */
    if (pkt!=&flush_pkt && av_dup_packet(pkt) < 0)
        return -1;

    pkt1 = av_malloc(sizeof(AVPacketList));
    if (!pkt1)
        return -1;
    pkt1->pkt = *pkt;
    pkt1->next = NULL;


    SDL_LockMutex(q->mutex);

    if (!q->last_pkt)

        q->first_pkt = pkt1;
    else
        q->last_pkt->next = pkt1;
    q->last_pkt = pkt1;
    q->nb_packets++;
    q->size += pkt1->pkt.size;
    /* XXX: should duplicate packet data in DV case */
    SDL_CondSignal(q->cond);

    SDL_UnlockMutex(q->mutex);
    return 0;
}
int quit = 0;
static int packet_queue_get(PacketQueue *q, AVPacket *pkt, int block)
{
    AVPacketList *pkt1;
    int ret;

    SDL_LockMutex(q->mutex);

    for(;;) {
        if (quit) {
            ret = -1;
            break;
        }

        pkt1 = q->first_pkt;
        if (pkt1) {
            q->first_pkt = pkt1->next;
            if (!q->first_pkt)
                q->last_pkt = NULL;
            q->nb_packets--;
            q->size -= pkt1->pkt.size;
            *pkt = pkt1->pkt;
            av_free(pkt1);
			pkt1 = NULL;
            ret = 1;
            break;
        } else if (!block) {
            ret = 0;
            break;
        } else {
            SDL_CondWait(q->cond, q->mutex);
        }
    }
    SDL_UnlockMutex(q->mutex);
    return ret;
}
VideoState *global_video_state;
 

int decode_interrupt_cb(void) {
	return (global_video_state && global_video_state->quit);
}

void SaveFrame(AVFrame *pFrame, int width, int height, int iFrame) {
	FILE *pFile;
	char szFilename[32];
	int y;
	// Open file
	sprintf(szFilename, "frame%d.ppm", iFrame);
	pFile=fopen(szFilename, "wb");
	if(pFile==NULL)return;

	// Write header
	fprintf(pFile, "P6\n%d %d\n255\n", width, height);
	// Write pixel data

	for(y=0; y < height; y++)
		fwrite(pFrame->data[0]+y*pFrame->linesize[0], 1, width*3, pFile);

	// Close file
	fclose(pFile);
}
void SaveData(uint8_t* buffer, int len, int iFrame) {
	FILE *pFile;
	char szFilename[32];
	int y;
	// Open file
	sprintf(szFilename, "frame%d.jpg", iFrame);
	pFile=fopen(szFilename, "wb");
	if(pFile==NULL)return;
	fwrite(buffer, 1, len, pFile);
	fclose(pFile);
}

int audio_decode_frame(AVCodecContext *aCodecCtx, uint8_t *audio_buf, int buf_size) {
	static AVPacket pkt;
	static uint8_t *audio_pkt_data = NULL;
	static int audio_pkt_size = 0;
	int len1, data_size;
	for(;;) {
		while(audio_pkt_size > 0) {
			data_size = buf_size;
			len1 = avcodec_decode_audio2(aCodecCtx, (int16_t *)audio_buf, &data_size, audio_pkt_data, audio_pkt_size);
			if(len1 < 0) {
				audio_pkt_size = 0;
				break;
			}
			audio_pkt_data += len1;
			audio_pkt_size -= len1;
			if(data_size <= 0) {
				continue;
			}
			return data_size;
		}
		if(pkt.data)
			av_free_packet(&pkt);

		if(quit) {
			return -1;
		}
		if(packet_queue_get(&audioq, &pkt, 1) < 0)
			return -1;
		audio_pkt_data = pkt.data;
		audio_pkt_size = pkt.size;
	}
}

void audio_callback(void *userdata, Uint8 *stream, int len) {
	AVCodecContext *aCodecCtx = (AVCodecContext *)userdata;
	int len1, audio_size;
	static uint8_t audio_buf[(AVCODEC_MAX_AUDIO_FRAME_SIZE * 3) / 2];
	static unsigned int audio_buf_size = 0;
	static unsigned int audio_buf_index = 0;
	
	while(len > 0) {
		if(audio_buf_index >= audio_buf_size) {
			audio_size = audio_decode_frame(aCodecCtx, audio_buf, sizeof(audio_buf));
			if(audio_buf_size < 0) {
				audio_buf_size = 1024;
				memset(audio_buf, 0, audio_buf_size);
			} else {
				audio_buf_size = audio_size;
			}
			audio_buf_index = 0;
		}
		len1 = audio_buf_size - audio_buf_index;
		if(len1 > len)len1 = len;
		memcpy(stream, (uint8_t *)audio_buf + audio_buf_index, len1);
		len -= len1;
		stream += len1;
		audio_buf_index += len1;
	}
}

int queue_picture(VideoState *is, AVFrame *pFrame)
{
	VideoPicture *vp;
	int dst_pix_fmt;
	AVPicture pict;

	SDL_LockMutex(is->pictq_mutex);
	while(is->pictq_size >= VIDEO_PICTURE_QUEUE_SIZE && !is->quit) {
		SDL_CondWait(is->pictq_cond, is->pictq_mutex);
	}
	SDL_UnlockMutex(is->pictq_mutex);
	if(is->quit)
		return -1;
	// windex is set to 0 initially
	vp = &is->pictq[is->pictq_windex];

	if(!vp->bmp ||
		vp->width != is->video_st->codec->width ||
		vp->height != is->video_st->codec->height) {
		SDL_Event event;

		vp->allocated = 0;

		event.type = FF_ALLOC_EVENT;

		event.user.data1 = is;
		SDL_PushEvent(&event);
		
		SDL_LockMutex(is->pictq_mutex);
		while(!vp->allocated && !is->quit) {
			SDL_CondWait(is->pictq_cond, is->pictq_mutex);
		}
		SDL_UnlockMutex(is->pictq_mutex);
		
		if(is->quit) {
			return -1;
		}
	}
	
	if(vp->bmp) {
		SDL_LockYUVOverlay(vp->bmp);
		dst_pix_fmt = PIX_FMT_YUV420P;

		pict.data[0] = vp->bmp->pixels[0];
		pict.data[1] = vp->bmp->pixels[1];
		pict.data[2] = vp->bmp->pixels[2];

		pict.linesize[0] = vp->bmp->pitches[0];
		pict.linesize[1] = vp->bmp->pitches[1];
		pict.linesize[2] = vp->bmp->pitches[2];

		//Convert the image into YUV format that SDL uses
		img_convert(&pict, dst_pix_fmt,
			(AVPicture *)pFrame, is->video_st->codec->pix_fmt,
			is->video_st->codec->width, is->video_st->codec->height);
	
		SDL_UnlockYUVOverlay(vp->bmp);
		
		if(++is->pictq_windex == VIDEO_PICTURE_QUEUE_SIZE) {
			is->pictq_windex = 0;
		}
		SDL_LockMutex(is->pictq_mutex);
		is->pictq_size++;
		SDL_UnlockMutex(is->pictq_mutex);
	}
	
	return 0;
}


int video_thread(void *arg)
{
	VideoState *is = (VideoState *)arg;
	AVPacket pkt1, *packet = &pkt1;
	int len1, frameFinished;
	AVFrame *pFrame;
	
	pFrame = avcodec_alloc_frame();

	for(; ;) {
		if(packet_queue_get(&is->videoq, packet, 1) < 0) {
			//means we quit getting packets
			break;
		}

		//Decode video frame;
		len1 = avcodec_decode_video(is->video_st->codec, pFrame, &frameFinished, 
			packet->data, packet->size);

		//Did we get a video frame?
		if(frameFinished) {
			if(queue_picture(is, pFrame) < 0) {
				break;
			}
		}
		av_free_packet(packet);
	}
	av_free(pFrame);
	pFrame = NULL;
	return 0;
}

static Uint32 sdl_refresh_timer_cb(Uint32 interval, void *opaque)
{
	SDL_Event event;
	event.type = FF_REFRESH_EVENT;
	event.user.data1 = opaque;
	SDL_PushEvent(&event);
	return 0;
}

static void schedule_refresh(VideoState *is, int delay)
{
	SDL_AddTimer(delay, sdl_refresh_timer_cb, is);
}

_inline long  rint(double x) 
{ 
	if(x >= 0.)
		return (long)(x + 0.5); 
	else 
		return (long)(x - 0.5); 
}
void video_display(VideoState *is)
{
	SDL_Rect rect;
	VideoPicture *vp;
	AVPicture pict;
	float aspect_ratio;
	int w, h, x, y;
	int i;
	
	vp = &is->pictq[is->pictq_rindex];

	if(vp->bmp) {
		if(is->video_st->codec->sample_aspect_ratio.num == 0) {
			aspect_ratio = 0;
		} else {
			aspect_ratio = av_q2d(is->video_st->codec->sample_aspect_ratio) *
				is->video_st->codec->width / is->video_st->codec->height;
		}

		if(aspect_ratio <= 0.0) {
			aspect_ratio = (float)is->video_st->codec->width / 
				(float)is->video_st->codec->height;
		}
		h = screen->h;
		w = ((int)rint(h * aspect_ratio)) & -3;
		if(w > screen->w) {
			w = screen->w;
			h = ((int)rint(w / aspect_ratio)) & -3;
		}
		x = (screen->w - w) / 2;
		y = (screen->h - h) / 2;
		rect.x = x;
		rect.y = y;
		rect.w = w;
		rect.h = h;
	    SDL_DisplayYUVOverlay(vp->bmp, &rect);
	}
}

int main()
{
	
	AVFormatContext *pFormatCtx = NULL;
	AVCodecContext *pCodecCtx, *aCodecCtx;
	int i, videoStream = -1, audiostream = -1;
	unsigned int nb_streams;
	AVCodec *pCodec, *aCodec;
	AVFrame *pFrame, *pFrameRGB;
	uint8_t *buffer;
	int numBytes;
	int frameFinished;
	AVPacket packet;
	AVPicture pict;

	SDL_Overlay *bmp;
	SDL_Rect rect;
	SDL_Event event;
	SDL_AudioSpec wanted_spec, spec; 
	av_register_all();

	pFormatCtx = (AVFormatContext*)malloc(sizeof(AVFormatContext));
	// Open video file

	if(av_open_input_file(&pFormatCtx, "Flying.wmv", NULL, 0, NULL)!=0)
		return -1; // Couldn't open file
	// Retrieve stream information
	
	if(av_find_stream_info(pFormatCtx)<0)
		return -1; // Couldn't find stream information
	
	// Dump information about file onto standard error
	dump_format(pFormatCtx, 0, "Flying.wmv", 0);
	
	//SDL init
	if(SDL_Init(SDL_INIT_VIDEO | SDL_INIT_AUDIO | SDL_INIT_TIMER)) {
		fprintf(stderr, "Could not initialize SDL - %s\n", SDL_GetError());
		exit(1);
	}

	nb_streams = pFormatCtx->nb_streams;
	url_set_interrupt_cb(decode_interrupt_cb);
	// Find the first video stream
	for(i=0; i < nb_streams; i++) {
		if(pFormatCtx->streams[i]->codec->codec_type==CODEC_TYPE_VIDEO && videoStream == -1) {
			videoStream=i;
//			break;
		}
		if(pFormatCtx->streams[i]->codec->codec_type == CODEC_TYPE_AUDIO && audiostream == -1) {
			audiostream = i;
		}
	}

	if(videoStream==-1)return -1; // Didn't find a video stream
	if(audiostream == -1)return -1;
	// Get a pointer to the codec context for the video stream
	pCodecCtx=pFormatCtx->streams[videoStream]->codec;	
	aCodecCtx = pFormatCtx->streams[audiostream]->codec;
	wanted_spec.freq = aCodecCtx->sample_rate;
	wanted_spec.format = AUDIO_S16SYS;
	wanted_spec.channels = aCodecCtx->channels;
	wanted_spec.silence = 0;
	wanted_spec.samples = 1024;//SDL_AUDIO_BUFFER_SIZE;
	wanted_spec.callback = audio_callback;
	wanted_spec.userdata = aCodecCtx;
//	if(SDL_OpenAudio(&wanted_spec, &spec) < 0) {
//		fprintf(stderr, "SDL_OpenAudio: %s\n", SDL_GetError());
//		return -1;
//	}
	// Find the decoder for the video stream
	pCodec=avcodec_find_decoder(pCodecCtx->codec_id);
	if(pCodec==NULL) {
		fprintf(stderr, "Unsupported codec!\n");
		return -1; // Codec not found
	}
	// Open codec
	if(avcodec_open(pCodecCtx, pCodec)<0)
		return -1; // Could not open codec
//	aCodec = avcodec_find_decoder(aCodecCtx->codec_id);
//	if(aCodec == NULL) {
//		fprintf(stderr, "Unsupported codec!\n");
//		return -1;
//	}
//	if(avcodec_open(aCodecCtx, aCodec) < 0)
//		return -1;
	packet_queue_init(&audioq);
//	SDL_PauseAudio(0);
	//SDL create a screen to show
	screen = SDL_SetVideoMode(pCodecCtx->width, pCodecCtx->height, 0, 0);
	if(!screen) {
		fprintf(stderr, "SDL:could not set video mode - exiting\n");
		exit(1);
	}
	bmp = SDL_CreateYUVOverlay(pCodecCtx->width, pCodecCtx->height, SDL_YV12_OVERLAY, screen);

	// Allocate video frame
	pFrame=avcodec_alloc_frame();
	// Allocate an AVFrame structure
	pFrameRGB=avcodec_alloc_frame();
	if(pFrameRGB==NULL)
		return -1;

	// Determine required buffer size and allocate buffer
	numBytes=avpicture_get_size(PIX_FMT_RGB24, pCodecCtx->width, pCodecCtx->height);
	buffer=(uint8_t *)av_malloc(numBytes*sizeof(uint8_t));
	// Assign appropriate parts of buffer to image planes in pFrameRGB
	// Note that pFrameRGB is an AVFrame, but AVFrame is a superset
	// of AVPicture
	avpicture_fill((AVPicture *)pFrameRGB, buffer, PIX_FMT_RGB24,
						pCodecCtx->width, pCodecCtx->height);
	
	while(av_read_frame(pFormatCtx, &packet) >= 0) {
		// Is this a packet from the video stream?
		if(packet.stream_index == videoStream) {
			// Decode video frame
			avcodec_decode_video(pCodecCtx, pFrame, &frameFinished, packet.data, packet.size);
			// Did we get a video frame?
			if(frameFinished) {
				SDL_LockYUVOverlay(bmp);
				pFrameRGB->data[0] = bmp->pixels[0];
				pFrameRGB->data[1] = bmp->pixels[2];
				pFrameRGB->data[2] = bmp->pixels[1];
				pFrameRGB->linesize[0] = bmp->pitches[0];
				pFrameRGB->linesize[1] = bmp->pitches[2];
				pFrameRGB->linesize[2] = bmp->pitches[1];

				// Convert the image from its native format to RGB
				img_convert((AVPicture *)pFrameRGB, PIX_FMT_YUV420P, (AVPicture*)pFrame, pCodecCtx->pix_fmt,
						pCodecCtx->width, pCodecCtx->height);
				SDL_UnlockYUVOverlay(bmp);
				rect.x = 0;
				rect.y = 0;
				rect.w = pCodecCtx->width;
				rect.h = pCodecCtx->height;
				SDL_DisplayYUVOverlay(bmp, &rect);
			}
		} else if(packet.stream_index == audiostream){
			packet_queue_put(&audioq, &packet);
		} else {
			// Free the packet that was allocated by av_read_frame
			av_free_packet(&packet);
		}
		
		SDL_PollEvent(&event);
		switch(event.type) {
		case SDL_QUIT:
			quit = 1;
			SDL_Quit();
			exit(0);
			break;
		default:
			break;
		}
	}
	// Free the RGB image
	av_free(buffer);
	av_free(pFrameRGB);
	// Free the YUV frame
	av_free(pFrame);
	// Close the codec
	avcodec_close(pCodecCtx);
	// Close the video file
	av_close_input_file(pFormatCtx);
	return 0;
}
