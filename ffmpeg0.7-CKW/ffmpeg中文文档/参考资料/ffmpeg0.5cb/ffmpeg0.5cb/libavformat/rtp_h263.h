#ifndef AVFORMAT_RTP_H263_H
#define AVFORMAT_RTP_H263_H

#include "rtpdec.h"
#include "rtpenc.h"

extern RTPDynamicProtocolHandler ff_h263p_dynamic_handler;
extern RTPDynamicProtocolHandler ff_h263pp_dynamic_handler;

int h263_handle_packet(RTPDemuxContext *s, AVPacket *pkt, const uint8_t *buf, int len);

void ff_rtp_send_h263(AVFormatContext *s1, const uint8_t *buf1, int size);
void ff_rtp_send_h263p(AVFormatContext *s1, const uint8_t *buf1, int size);

#endif /* AVFORMAT_RTP_H263_H */
