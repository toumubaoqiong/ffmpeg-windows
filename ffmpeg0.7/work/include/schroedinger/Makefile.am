
if HAVE_CUDA
SUBDIRS = cuda
endif

if HAVE_OPENGL
SUBDIRS = opengl
endif

ORC_SOURCE=schroorc
include $(top_srcdir)/orc.mak

pkgincludedir = $(includedir)/schroedinger-@SCHRO_MAJORMINOR@/schroedinger

lib_LTLIBRARIES = libschroedinger-@SCHRO_MAJORMINOR@.la

BUILT_SOURCES += \
	schroversion.h

DISTCLEANFILES = schroversion.h

EXTRA_DIST += schroarith-i386.c \
	schroasync-none.c \
	schroasync-pthread.c \
	schroasync-gthread.c \
	schroasync-win32.c

pkginclude_HEADERS = \
	schro.h \
	schrobuffer.h \
	schrobufferlist.h \
	schrodecoder.h \
	schroencoder.h \
	schroengine.h \
	schrofilter.h \
	schroframe.h \
	schro-stdint.h \
	schroarith.h \
	schroanalysis.h \
	schroasync.h \
	schrobitstream.h \
	schrodebug.h \
	schrodomain.h \
	schrofft.h \
	schrohistogram.h \
	schrolimits.h \
	schrolist.h \
	schrometric.h \
	schromotion.h \
	schromotionest.h \
	schropack.h \
	schroparams.h \
	schroparse.h \
	schrophasecorrelation.h \
	schroqueue.h \
	schrossim.h \
	schrotables.h \
	schrounpack.h \
	schroutils.h \
	schrovideoformat.h \
	schrovirtframe.h \
	schrowavelet.h

nodist_pkginclude_HEADERS = \
	schroversion.h

noinst_HEADERS = \
	schrogpuframe.h

if HAVE_CUDA
cuda_sources = \
	schrocuda.c \
	schrocuda.h \
	schrogpuframe.c \
	schrogpumotion.c \
	dummy.cc
cuda_libs = cuda/libdcuda.la
endif

if HAVE_OPENGL
opengl_libs = opengl/libopengl.la
endif

libschroedinger_@SCHRO_MAJORMINOR@_la_SOURCES = \
	schro.c \
	schroarith.c \
	schrodecoder.c \
	schrodebug.c \
	schrodomain.c \
	schroframe.c \
	schrohistogram.c \
	schrobuffer.c \
	schrobufferlist.c \
	schrolowdelay.c \
	schromotion.c \
	schromotionfast.c \
	schroparams.c \
	schroroughmotion.c \
	schroparse.c \
	schroqueue.c \
	schrotables.c \
	schrotablesinit.c \
	schrounpack.c \
	schrounpacktables.c \
	schroutils.c \
	schrovideoformat.c \
	schrovirtframe.c \
	schrowaveletorc.c \
	$(cuda_sources) \
	schroasync-@THREAD_IMPL@.c

nodist_libschroedinger_@SCHRO_MAJORMINOR@_la_SOURCES = $(ORC_NODIST_SOURCES)

if ENABLE_MOTION_REF
libschroedinger_@SCHRO_MAJORMINOR@_la_SOURCES += \
	schromotionref.c
endif

encoder_sources = \
	schroanalysis.c \
	schroencoder.c \
	schroengine.c \
	schrofft.c \
	schrofilter.c \
	schroglobalest.c \
	schrohierbm.c \
	schrolist.c \
	schrometric.c \
	schromotionest.c \
	schropack.c \
	schrophasecorrelation.c \
	schroquantiser.c \
	schrossim.c \
	schrowavelettables.c

if ENABLE_ENCODER
libschroedinger_@SCHRO_MAJORMINOR@_la_SOURCES += \
	$(encoder_sources)
endif


libschroedinger_@SCHRO_MAJORMINOR@_la_CFLAGS = \
	$(SCHRO_CFLAGS) $(ORC_CFLAGS) $(ERROR_CFLAGS) \
	-DSCHRO_ENABLE_UNSTABLE_API \
	$(GTHREAD_CFLAGS) \
	$(GCOV_CFLAGS) \
	$(CUDA_CFLAGS) -I$(srcdir)/cuda -I$(srcdir)/opengl

libschroedinger_@SCHRO_MAJORMINOR@_la_LIBADD = \
	$(cuda_libs) \
	$(CUDA_LIBS) \
	$(opengl_libs) \
	$(OPENGL_LIBS) \
	$(ORC_LIBS) \
	$(GCOV_LIBS) \
	$(PTHREAD_LIBS)
libschroedinger_@SCHRO_MAJORMINOR@_la_LDFLAGS = \
	-no-undefined \
	-version-info $(SCHRO_LIBVERSION) \
	-export-symbols-regex '^(schro|orc)_'

%.c.gcov: .libs/libschroedinger_@SCHRO_MAJORMINOR@_la-%.gcda %.c
	$(GCOV) -b -f -o $^ > $@.out

gcov: $(libschroedinger_1.0_la_SOURCES:=.gcov) $(encoder_sources:=.gcov)

