/*
 * MP3 encoder and decoder
 * Copyright (c) 2003 Fabrice Bellard.
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */
#include "avformat.h"

#define ID3_HEADER_SIZE 10
#define ID3_TAG_SIZE 128

#define ID3_GENRE_MAX 125

static const char *id3_genre_str[ID3_GENRE_MAX + 1] = {
    "Blues",
    "Classic Rock",
    "Country",
    "Dance",
    "Disco",
    "Funk",
    "Grunge",
    "Hip-Hop",
    "Jazz",
    "Metal",
    "New Age",
    "Oldies",
    "Other",
    "Pop",
    "R&B",
    "Rap",
    "Reggae",
    "Rock",
    "Techno",
    "Industrial",
    "Alternative",
    "Ska",
    "Death Metal",
    "Pranks",
    "Soundtrack",
    "Euro-Techno",
    "Ambient",
    "Trip-Hop",
    "Vocal",
    "Jazz+Funk",
    "Fusion",
    "Trance",
    "Classical",
    "Instrumental",
    "Acid",
    "House",
    "Game",
    "Sound Clip",
    "Gospel",
    "Noise",
    "AlternRock",
    "Bass",
    "Soul",
    "Punk",
    "Space",
    "Meditative",
    "Instrumental Pop",
    "Instrumental Rock",
    "Ethnic",
    "Gothic",
    "Darkwave",
    "Techno-Industrial",
    "Electronic",
    "Pop-Folk",
    "Eurodance",
    "Dream",
    "Southern Rock",
    "Comedy",
    "Cult",
    "Gangsta",
    "Top 40",
    "Christian Rap",
    "Pop/Funk",
    "Jungle",
    "Native American",
    "Cabaret",
    "New Wave",
    "Psychadelic",
    "Rave",
    "Showtunes",
    "Trailer",
    "Lo-Fi",
    "Tribal",
    "Acid Punk",
    "Acid Jazz",
    "Polka",
    "Retro",
    "Musical",
    "Rock & Roll",
    "Hard Rock",
    "Folk",
    "Folk-Rock",
    "National Folk",
    "Swing",
    "Fast Fusion",
    "Bebob",
    "Latin",
    "Revival",
    "Celtic",
    "Bluegrass",
    "Avantgarde",
    "Gothic Rock",
    "Progressive Rock",
    "Psychedelic Rock",
    "Symphonic Rock",
    "Slow Rock",
    "Big Band",
    "Chorus",
    "Easy Listening",
    "Acoustic",
    "Humour",
    "Speech",
    "Chanson",
    "Opera",
    "Chamber Music",
    "Sonata",
    "Symphony",
    "Booty Bass",
    "Primus",
    "Porn Groove",
    "Satire",
    "Slow Jam",
    "Club",
    "Tango",
    "Samba",
    "Folklore",
    "Ballad",
    "Power Ballad",
    "Rhythmic Soul",
    "Freestyle",
    "Duet",
    "Punk Rock",
    "Drum Solo",
    "A capella",
    "Euro-House",
    "Dance Hall",
};

/* buf must be ID3_HEADER_SIZE byte long */
static int id3_match(const uint8_t *buf)
{
    return (buf[0] == 'I' &&
            buf[1] == 'D' &&
            buf[2] == '3' &&
            buf[3] != 0xff &&
            buf[4] != 0xff &&
            (buf[6] & 0x80) == 0 &&
            (buf[7] & 0x80) == 0 &&
            (buf[8] & 0x80) == 0 &&
            (buf[9] & 0x80) == 0);
}

static void id3_get_string(char *str, int str_size,
                           const uint8_t *buf, int buf_size)
{
    int i, c;
    char *q;

    q = str;
    for(i = 0; i < buf_size; i++) {
        c = buf[i];
        if (c == '\0')
            break;
        if ((q - str) >= str_size - 1)
            break;
        *q++ = c;
    }
    *q = '\0';
}

/* 'buf' must be ID3_TAG_SIZE byte long */
static int id3_parse_tag(AVFormatContext *s, const uint8_t *buf)
{
    char str[5];
    int genre;

    if (!(buf[0] == 'T' &&
          buf[1] == 'A' &&
          buf[2] == 'G'))
        return -1;
    id3_get_string(s->title, sizeof(s->title), buf + 3, 30);
    id3_get_string(s->author, sizeof(s->author), buf + 33, 30);
    id3_get_string(s->album, sizeof(s->album), buf + 63, 30);
    id3_get_string(str, sizeof(str), buf + 93, 4);
    s->year = atoi(str);
    id3_get_string(s->comment, sizeof(s->comment), buf + 97, 30);
    if (buf[125] == 0 && buf[126] != 0)
        s->track = buf[126];
    genre = buf[127];
    if (genre <= ID3_GENRE_MAX)
        pstrcpy(s->genre, sizeof(s->genre), id3_genre_str[genre]);
    return 0;
}

static void id3_create_tag(AVFormatContext *s, uint8_t *buf)
{
    int v, i;

    memset(buf, 0, ID3_TAG_SIZE); /* fail safe */
    buf[0] = 'T';
    buf[1] = 'A';
    buf[2] = 'G';
    strncpy(buf + 3, s->title, 30);
    strncpy(buf + 33, s->author, 30);
    strncpy(buf + 63, s->album, 30);
    v = s->year;
    if (v > 0) {
        for(i = 0;i < 4; i++) {
            buf[96 - i] = '0' + (v % 10);
            v = v / 10;
        }
    }
    strncpy(buf + 97, s->comment, 30);
    if (s->track != 0) {
        buf[125] = 0;
        buf[126] = s->track;
    }
    for(i = 0; i <= ID3_GENRE_MAX; i++) {
        if (!strcasecmp1(s->genre, id3_genre_str[i])) {
            buf[127] = i;
            break;
        }
    }
}

/* mp3 read */
static int mp3_read_header(AVFormatContext *s,
                           AVFormatParameters *ap)
{
    AVStream *st;
    uint8_t buf[ID3_TAG_SIZE];
    int len, ret, filesize;

    st = av_new_stream(s, 0);
    if (!st)
        return AVERROR_NOMEM;

    st->codec->codec_type = CODEC_TYPE_AUDIO;
    st->codec->codec_id = CODEC_ID_MP3;
    st->need_parsing = 1;

    /* try to get the TAG */
    if (!url_is_streamed(&s->pb)) {
        /* XXX: change that */
        filesize = url_fsize(&s->pb);
        if (filesize > 128) {
            url_fseek(&s->pb, filesize - 128, SEEK_SET);
            ret = get_buffer(&s->pb, buf, ID3_TAG_SIZE);
            if (ret == ID3_TAG_SIZE) {
                id3_parse_tag(s, buf);
            }
            url_fseek(&s->pb, 0, SEEK_SET);
        }
    }

    /* if ID3 header found, skip it */
    ret = get_buffer(&s->pb, buf, ID3_HEADER_SIZE);
    if (ret != ID3_HEADER_SIZE)
        return -1;
    if (id3_match(buf)) {
        /* skip ID3 header */
        len = ((buf[6] & 0x7f) << 21) |
            ((buf[7] & 0x7f) << 14) |
            ((buf[8] & 0x7f) << 7) |
            (buf[9] & 0x7f);
        url_fskip(&s->pb, len);
    } else {
        url_fseek(&s->pb, 0, SEEK_SET);
    }

    /* the parameters will be extracted from the compressed bitstream */
    return 0;
}

#define MP3_PACKET_SIZE 1024

static int mp3_read_packet(AVFormatContext *s, AVPacket *pkt)
{
    int ret, size;
    //    AVStream *st = s->streams[0];

    size= MP3_PACKET_SIZE;

    ret= av_get_packet(&s->pb, pkt, size);

    pkt->stream_index = 0;
    if (ret <= 0) {
        return AVERROR_IO;
    }
    /* note: we need to modify the packet size here to handle the last
       packet */
    pkt->size = ret;
    return ret;
}

static int mp3_read_close(AVFormatContext *s)
{
    return 0;
}

#ifdef CONFIG_MUXERS
/* simple formats */
static int mp3_write_header(struct AVFormatContext *s)
{
    return 0;
}

static int mp3_write_packet(struct AVFormatContext *s, AVPacket *pkt)
{
    put_buffer(&s->pb, pkt->data, pkt->size);
    put_flush_packet(&s->pb);
    return 0;
}

static int mp3_write_trailer(struct AVFormatContext *s)
{
    uint8_t buf[ID3_TAG_SIZE];

    /* write the id3 header */
    if (s->title[0] != '\0') {
        id3_create_tag(s, buf);
        put_buffer(&s->pb, buf, ID3_TAG_SIZE);
        put_flush_packet(&s->pb);
    }
    return 0;
}
#endif //CONFIG_MUXERS

AVInputFormat mp3_iformat = {
    "mp3",
    "MPEG audio",
    0,
    NULL,
    mp3_read_header,
    mp3_read_packet,
    mp3_read_close,
	NULL,
	NULL,
	NULL,
    "mp2,mp3,m2a", // extensions /* XXX: use probe */
};

#ifdef CONFIG_MUXERS
AVOutputFormat mp2_oformat = {
    "mp2",
    "MPEG audio layer 2",
    "audio/x-mpeg",
#ifdef CONFIG_MP3LAME
    "mp2,m2a",
#else
    "mp2,mp3,m2a",
#endif
    0,
    CODEC_ID_MP2,
    0,
    mp3_write_header,
    mp3_write_packet,
    mp3_write_trailer,
};

#ifdef CONFIG_MP3LAME
AVOutputFormat mp3_oformat = {
    "mp3",
    "MPEG audio layer 3",
    "audio/x-mpeg",
    "mp3",
    0,
    CODEC_ID_MP3,
    0,
    mp3_write_header,
    mp3_write_packet,
    mp3_write_trailer,
};
#endif
#endif //CONFIG_MUXERS

int mp3_init(void)
{
    av_register_input_format(&mp3_iformat);
#ifdef CONFIG_MUXERS
    av_register_output_format(&mp2_oformat);
#ifdef CONFIG_MP3LAME
    av_register_output_format(&mp3_oformat);
#endif
#endif //CONFIG_MUXERS
    return 0;
}
