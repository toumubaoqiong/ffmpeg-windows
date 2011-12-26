#include "rtp_h263.h"

struct PayloadContext {
    uint8_t stock_buf[65536];
    int stock_len;
};

int h263_handle_packet(RTPDemuxContext *s, AVPacket *pkt, const uint8_t *buf, int len)
{
    unsigned char h263_header_len = 4; // default to mode A
    unsigned char sbit = 0;
    unsigned char mark = buf[-11] & 0x80;

    if (len < 1)
        return -1;

    // test F bit for mode A or mode B/C
    // test P bit for mode B or mode C
    if (buf[0] & 0x80)
        h263_header_len = (buf[0] & 0xC0) == 0xC0 ? 12 : 8;

    if (len < h263_header_len)
        return -1;

    sbit = (buf[0] & 0x38) >> 3;

    buf += h263_header_len;
    len -= h263_header_len;

    if (!mark) {
        if (s->stock_len + len <= sizeof(s->stock_buf)) {
            if (sbit > 0 && s->stock_len > 0) {
                unsigned char mask = 0xff >> sbit;
                s->stock_buf[s->stock_len-1] &= ~mask;
                s->stock_buf[s->stock_len-1] |= buf[0] & mask;
                buf++;
                len--;
            }

            memcpy(s->stock_buf+s->stock_len, buf, len);
            s->stock_len += len;
        } else
            av_log(s->ic, AV_LOG_ERROR, "h263 stock buffer overflow\n");
        return -1;
    }

    if (sbit > 0 && s->stock_len > 0) {
        unsigned char mask = 0xff >> sbit;
        s->stock_buf[s->stock_len-1] &= ~mask;
        s->stock_buf[s->stock_len-1] |= buf[0] & mask;
        buf++;
        len--;
    }

    av_new_packet(pkt, s->stock_len+len);

    memcpy(pkt->data, s->stock_buf, s->stock_len);
    memcpy(pkt->data+s->stock_len, buf, len);
    s->stock_len = 0;

/*    int mark;
    int h263_header_len;

    mark = buf[-11] & 0x80;

    if (len < 1)
        return -1;

    // test F bit for mode A or mode B/C
    // test P bit for mode B or mode C
    h263_header_len = (buf[0] & 0x80) ? ((buf[0] & 0xC0) ? 12 : 8) : 4;
    if (len < h263_header_len)
        return -1;

    buf += h263_header_len;
    len -= h263_header_len;

    if (!mark) {
        if (s->stock_len + len <= sizeof(s->stock_buf)) {
            memcpy(s->stock_buf+s->stock_len, buf, len);
            s->stock_len += len;
        } else
            av_log(NULL, AV_LOG_ERROR, "h263 stock buffer overflow\n");
        return -1;
    }

    av_new_packet(pkt, s->stock_len+len);
    memcpy(pkt->data, s->stock_buf, s->stock_len);
    memcpy(pkt->data+s->stock_len, buf, len);
    s->stock_len = 0;
*/
    return 0;
}

static int h263p_handle_packet(AVFormatContext *ctx,
                               PayloadContext *data,
                               AVStream *st,
                               AVPacket *pkt,
                               uint32_t *timestamp,
                               const uint8_t *buf,
                               int len, int flags)
{
    int mark;
    int h264_header_len;
    int P;

    mark = buf[-11] & 0x80;

    if (len < 2)
        return -1;

    P = buf[0] & 0x4;

    h264_header_len = 2;
    // test V bit for extra VRC header
    if (buf[0] & 0x2)
        h264_header_len++;
    // test PLEN for extra picture header
    h264_header_len += ((buf[0] & 0x1) << 5) | ((buf[1] & 0xF8) >> 3);
    if (len < h264_header_len)
        return -1;

    buf += h264_header_len;
    len -= h264_header_len;

    if (!mark) {
        if (P) {
            data->stock_buf[data->stock_len] = data->stock_buf[data->stock_len+1] = 0;
            data->stock_len += 2;
        }
        if (data->stock_len + len <= sizeof(data->stock_buf)) {
            memcpy(data->stock_buf+data->stock_len, buf, len);
            data->stock_len += len;
        } else
            av_log(NULL, AV_LOG_ERROR, "h263p stock buffer overflow\n");
        return -1;
    }

    if (P)
        av_new_packet(pkt, data->stock_len+len+2);
    else
        av_new_packet(pkt, data->stock_len+len);
    memcpy(pkt->data, data->stock_buf, data->stock_len);
    if (P) {
        pkt->data[data->stock_len] = pkt->data[data->stock_len+1] = 0;
        data->stock_len += 2;
    }
    memcpy(pkt->data+data->stock_len, buf, len);
    data->stock_len = 0;

    return 0;
}

static PayloadContext *h263p_new_extradata(void)
{
    PayloadContext *data = av_mallocz(sizeof(PayloadContext));
    return data;
}

static void h263p_free_extradata(PayloadContext *data)
{
    av_free(data);
}

void ff_rtp_send_h263(AVFormatContext *s1, const uint8_t *buf1, int size)
{
    RTPMuxContext *s = s1->priv_data;
    int len;

    // test picture start code
    if (buf1[0] != 0 || buf1[1] != 0 || (buf1[2] & 0xfc) != 0x80) {
        av_log(s1, AV_LOG_ERROR, "bad picture start code (%02x %02x %02x), skip sending frame\n", buf1[0], buf1[1], buf1[2]);
        return;
    }

    // build h263 rtp header (RFC 2190)
    memset(s->buf, 0, 4);
    // P
    s->buf[0] |= (buf1[5] & 0x20) << 5;
    // SRC
    s->buf[1] |= (buf1[4] & 0x1c) << 3;
    // I
    s->buf[1] |= (buf1[4] & 0x02) << 3;
    // U
    s->buf[1] |= (buf1[4] & 0x01) << 3;
    // S
    s->buf[1] |= (buf1[5] & 0x80) >> 5;
    // A
    s->buf[1] |= (buf1[5] & 0x40) >> 5;
    // TR
    s->buf[3] |= (buf1[2] & 0x03) << 6;
    s->buf[3] |= (buf1[3] & 0xfc) >> 2;

    s->timestamp = s->cur_timestamp;

    while (size > 0) {
        len = s->max_payload_size-4;
        if (len > size)
            len = size;

        memcpy(&s->buf[4], buf1, len);
        ff_rtp_send_data(s1, s->buf, len+4, (len == size));

        buf1 += len;
        size -= len;

        if(size > 0)
            usleep(15000);
    }
}

void ff_rtp_send_h263p(AVFormatContext *s1, const uint8_t *buf1, int size)
{
    RTPMuxContext *s = s1->priv_data;
    int len;

    // test picture start code
    if (buf1[0] != 0 || buf1[1] != 0 || (buf1[2] & 0xfc) != 0x80) {
        av_log(s1, AV_LOG_ERROR, "bad picture start code (%02x %02x %02x), skip sending frame\n", buf1[0], buf1[1], buf1[2]);
        return;
    }

    // build h.263+ rtp header (RFC 4629)
    memset(s->buf, 0, 2);
    s->buf[0] = 0x4; // only set P bit
    buf1 += 2;
    size -= 2;

    s->timestamp = s->cur_timestamp;

    while (size > 0) {
        len = s->max_payload_size-2;
        if (len > size)
            len = size;

        memcpy(&s->buf[2], buf1, len);
        ff_rtp_send_data(s1, s->buf, len+2, (len == size));

        s->buf[0] = 0; // clear P bit for segment packet

        buf1 += len;
        size -= len;
    }
}

RTPDynamicProtocolHandler ff_h263p_dynamic_handler = {
    "H263-1998",
    CODEC_TYPE_VIDEO,
    CODEC_ID_H263,
    NULL,
    h263p_new_extradata,
    h263p_free_extradata,
    h263p_handle_packet
};

RTPDynamicProtocolHandler ff_h263pp_dynamic_handler = {
    "H263-2000",
    CODEC_TYPE_VIDEO,
    CODEC_ID_H263,
    NULL,
    h263p_new_extradata,
    h263p_free_extradata,
    h263p_handle_packet
};

