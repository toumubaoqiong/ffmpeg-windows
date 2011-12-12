/*
 * RTP packetization for H.263 video
 * Copyright (c) 2009 Luca Abeni
 * Copyright (c) 2009 Martin Storsjo
 *
 * This file is part of FFmpeg.
 *
 * FFmpeg is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * FFmpeg is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with FFmpeg; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */

#include "avformat.h"
#include "rtpenc.h"

static const uint8_t *find_resync_marker_reverse(const uint8_t *restrict start,
                                                 const uint8_t *restrict end)
{
    const uint8_t *p = end - 1;
    start += 1; /* Make sure we never return the original start. */
    for (; p > start; p -= 2) {
        if (!*p) {
            if      (!p[ 1] && p[2]) return p;
            else if (!p[-1] && p[1]) return p - 1;
        }
    }
    return end;
}

//x-lint version 3.0 build 41150 h263
//Packetize H.263 frames into RTP packets according to RFC 2190
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
    }
}


/*
//Packetize H.263 frames into RTP packets according to RFC 4629
void ff_rtp_send_h263_RFC4629(AVFormatContext *s1, const uint8_t *buf1, int size)
{
    RTPMuxContext *s = s1->priv_data;
    int len, max_packet_size;
    uint8_t *q;

    max_packet_size = s->max_payload_size;

    while (size > 0) {
        q = s->buf;
        if (size >= 2 && (buf1[0] == 0) && (buf1[1] == 0)) {
            *q++ = 0x04;
            buf1 += 2;
            size -= 2;
        } else {
            *q++ = 0;
        }
        *q++ = 0;

        len = FFMIN(max_packet_size - 2, size);

        // Look for a better place to split the frame into packets. 
        if (len < size) {
            const uint8_t *end = find_resync_marker_reverse(buf1, buf1 + len);
            len = end - buf1;
        }

        memcpy(q, buf1, len);
        q += len;

        // 90 KHz time stamp
        s->timestamp = s->cur_timestamp;
        ff_rtp_send_data(s1, s->buf, q - s->buf, (len == size));

        buf1 += len;
        size -= len;
    }
}
*/

