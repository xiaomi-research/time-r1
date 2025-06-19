import logging
import time
from typing import Tuple

import decord
import torch

from src.utils.vision_process import smart_nframes

logger = logging.getLogger(__name__)


def _read_video_decord_w_timestamp(
    ele: dict,
) -> Tuple[torch.Tensor, float]:
    """read video using decord.VideoReader

    Args:
        ele (dict): a dict contains the configuration of video.
        support keys:
            - video: the path of video. support "file://", "http://", "https://" and local path.
            - video_start: the start time of video.
            - video_end: the end time of video.
    Returns:
        torch.Tensor: the video tensor with shape (T, C, H, W).
    """
    video_path = ele["video"]
    st = time.time()
    vr = decord.VideoReader(video_path)
    total_frames, video_fps = len(vr), vr.get_avg_fps()

    # TODO: support start_pts and end_pts
    video_start = ele.get("video_start", 0.0)
    video_end = ele.get("video_end", total_frames / video_fps)

    start_frame = max(0, int(video_start * video_fps))
    end_frame = min(total_frames, int(video_end * video_fps))
    if end_frame <= start_frame:
        end_frame = start_frame + 1
        if end_frame > total_frames:
            end_frame = total_frames
            start_frame = max(0, end_frame - 1)
    effective_frames = end_frame - start_frame
    logger.info(
        f"decord: {video_path=}, {effective_frames=}, {video_fps=}, time={time.time() - st:.3f}s"
    )
    nframes = smart_nframes(ele, total_frames=effective_frames, video_fps=video_fps)
    if effective_frames == 0:
        idx = [start_frame]
    else:
        idx = (
            torch.linspace(start_frame, end_frame - 1, nframes).round().long().tolist()
        )
    video = vr.get_batch(idx).asnumpy()
    video = torch.tensor(video).permute(0, 3, 1, 2)  # Convert to TCHW format
    sample_fps = nframes / max(effective_frames, 1e-6) * video_fps
    return video, sample_fps


def monkey_patch():
    import src.utils

    src.utils.vision_process.VIDEO_READER_BACKENDS["decord"] = (
        _read_video_decord_w_timestamp  # support start_pts and end_pts
    )
