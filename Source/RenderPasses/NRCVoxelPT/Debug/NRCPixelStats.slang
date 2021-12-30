/** Functionality for collecting runtime stats in the path tracer.

    The host sets the following defines:

    _NRC_PIXEL_STATS_ENABLED       Nonzero if stats should be collected.

*/
__exported import NRCPixelStatsShared;

RWTexture2D<uint> gStatsInferencePathLength;
RWTexture2D<uint> gStatsSuffixPathLength;

#ifdef _NRC_PIXEL_STATS_ENABLED
static uint2 gPixelStatsPixelNRC;
#endif

void logSetPixelNRC(uint2 pixel)
{
#ifdef _NRC_PIXEL_STATS_ENABLED
    gPixelStatsPixelNRC = pixel;
#endif
}

void logInferencePathLength(uint length)
{
#ifdef _NRC_PIXEL_STATS_ENABLED
    gStatsInferencePathLength[gPixelStatsPixelNRC] = length;
#endif
}

void logSuffixPathLength(uint length)
{
#ifdef _NRC_PIXEL_STATS_ENABLED
    gStatsSuffixPathLength[gPixelStatsPixelNRC] = length;
#endif
}
