#ifndef __IPVT_PCB_H__
#define __IPVT_PCB_H__

typedef struct IPVT_CONF_LAYOUT_TAG
{
	int iMember;
} IPVT_CONF_LAYOUT;

typedef struct IPVT_PCB_TAG
{
	// input
	char				szID[10];		// id
	char				szArg[10240];	// ffmpeg arguments
	int					quit;			// quit flag
	int					iMembers;		// conference members
	IPVT_CONF_LAYOUT	confLayout[16];	// conference layout
	int					refresh;		// refresh flag
	// output
} IPVT_PCB;

#endif // __IPVT_PCB_H__
