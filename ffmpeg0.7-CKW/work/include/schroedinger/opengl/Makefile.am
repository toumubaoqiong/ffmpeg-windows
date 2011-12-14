
noinst_LTLIBRARIES = libopengl.la

libopengl_la_SOURCES = \
	schroopengl.c \
	schroopengl.h \
	schroopenglcanvas.c \
	schroopenglframe.c \
	schroopenglframe.h \
	schroopenglframe_combine.c \
	schroopenglframe_convert.c \
	schroopenglframe_pull.c \
	schroopenglframe_push.c \
	schroopenglmotion.c \
	schroopenglmotion.h \
	schroopenglshader.c \
	schroopengltypes.h \
	schroopenglwavelet.c

libopengl_la_CFLAGS = $(SCHRO_CFLAGS) $(OPENGL_CFLAGS) $(ERROR_CFLAGS) \
	-DSCHRO_ENABLE_UNSTABLE_API -D_GNU_SOURCE -DSCHRO_OPENGL_UNBIND_TEXTURES=1

libopengl_la_LIBADD = $(OPENGL_LIBS)


