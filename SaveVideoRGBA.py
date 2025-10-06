from __future__ import annotations
from typing_extensions import override
from typing import Optional
from fractions import Fraction
from enum import Enum
import io
import json
import math
import av
import os
import random
import folder_paths
from comfy_api.latest._input import AudioInput, VideoInput
from comfy_api.latest import ComfyExtension, io, ui
from comfy_api.util import VideoComponents

class SaveVideoRGBA(io.ComfyNode):

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SaveVideoRGBA",
            display_name="Save Video (RGBA)",
            category="image/animation",
            inputs=[
                io.Image.Input("images"),
                io.Float.Input("fps", default=24.0, min=1.0, max=120.0, step=1.0),
                io.String.Input("filename_prefix", default="video/ComfyUI", tooltip="The prefix for the file to save. This may include formatting information such as %date:yyyy-MM-dd% or %Empty Latent Image.width% to include values from nodes."),
                io.Boolean.Input("only_preview", default=False),
                io.Audio.Input("audio", optional=True),
            ],
            outputs=[
                # io.Video.Output("VIDEO")
            ],
            hidden=[io.Hidden.unique_id],
            is_output_node=True
        )

    @classmethod
    def execute(self, images, fps, filename_prefix, only_preview, audio=None, **kwargs):

        B, H, W, C = images.shape
        has_alpha = C == 4

        video = RGBAVideoFromComponents(
            VideoComponents(
                images=images,
                audio=audio,
                frame_rate=Fraction(fps),
            )
        )

        width, height = video.get_dimensions()
        results = list()

        # 预览
        if only_preview or has_alpha:
            output_dir = folder_paths.get_temp_directory()
            prefix_append = "ComfyUI_temp_" + ''.join(random.choice("abcdefghijklmnopqrstupvxyz") for x in range(5))
            full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
                prefix_append,
                output_dir,
                width,
                height
            )
            format = 'webm' if has_alpha else 'auto'
            file = f"{filename}_{counter:05}_.{RGBAVideoContainer.get_extension(format)}"
            video.save_to(
                path=os.path.join(full_output_folder, file),
                format=format,
                codec='auto',
                metadata=None,
            )

            results.append(ui.SavedResult(file, subfolder, io.FolderType.temp))
            counter += 1

        # 保存
        if not only_preview:
            output_dir = folder_paths.get_output_directory()

            full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
                filename_prefix,
                output_dir,
                width,
                height
            )
            format = 'mov' if has_alpha else 'auto'
            codec = 'prores_ks' if has_alpha else 'auto'
            file = f"{filename}_{counter:05}_.{RGBAVideoContainer.get_extension(format)}"
            video.save_to(
                path=os.path.join(full_output_folder, file),
                format=format,
                codec=codec,
                metadata=None,
            )

            if not has_alpha:
                results.append(ui.SavedResult(file, subfolder, io.FolderType.output))
                counter += 1

        return io.NodeOutput(ui=ui.PreviewVideo(results))

class NodeExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            SaveVideoRGBA
        ]

async def comfy_entrypoint() -> NodeExtension:
    return NodeExtension()



class RGBAVideoCodec(str, Enum):
    AUTO = "auto"
    H264 = "h264"  # H.264 codec, no alpha channel
    VP9 = "libvpx-vp9"  # VP9 codec, supports alpha channel
    PRORES = "prores_ks"  # ProRes 4444, supports alpha channel

    @classmethod
    def as_input(cls) -> list[str]:
        """
        Returns a list of codec names that can be used as node input.
        """
        return [member.value for member in cls]

class RGBAVideoContainer(str, Enum):
    AUTO = "auto"
    MP4 = "mp4"  # MP4 container
    WEBM = "webm"  # WebM container, supports VP9 with alpha channel
    MOV = "mov"  # QuickTime MOV, supports ProRes with alpha channel

    @classmethod
    def as_input(cls) -> list[str]:
        """
        Returns a list of codec names that can be used as node input.
        """
        return [member.value for member in cls]

    @classmethod
    def get_extension(cls, value) -> str:
        """
        Returns the file extension for the container.
        """
        if isinstance(value, str):
            value = cls(value)
        if value == RGBAVideoContainer.AUTO:
            return "mp4"
        elif value == RGBAVideoContainer.MP4:
            return "mp4"
        elif value == RGBAVideoContainer.WEBM:
            return "webm"
        elif value == RGBAVideoContainer.MOV:
            return "mov"
        return ""

class RGBAVideoFromComponents(VideoInput):

    def __init__(self, components: VideoComponents):
        self.__components = components

    def get_components(self) -> VideoComponents:
        return VideoComponents(
            images=self.__components.images,
            audio=self.__components.audio,
            frame_rate=self.__components.frame_rate
        )

    def save_to(
            self,
            path: str,
            format: RGBAVideoContainer = RGBAVideoContainer.AUTO,
            codec: RGBAVideoCodec = RGBAVideoCodec.AUTO,
            metadata: Optional[dict] = None
    ):
        # Check if images have alpha channel (4 channels)
        has_alpha = self.__components.images.shape[-1] == 4 if len(self.__components.images.shape) == 4 else False

        # Determine format and codec based on alpha channel
        if has_alpha:
            # For alpha channel support, use webm container with vp9 codec
            if format == RGBAVideoContainer.AUTO:
                format = RGBAVideoContainer.WEBM.value
            if codec == RGBAVideoCodec.AUTO:
                codec = RGBAVideoCodec.VP9.value
            # Ensure format is string for comparison
            format_str = format.value if isinstance(format, RGBAVideoContainer) else format
            codec_str = codec.value if isinstance(codec, RGBAVideoCodec) else codec
            if format_str not in ['webm', 'mov']:
                raise ValueError("Only WEBM and MOV formats support alpha channel")
        else:
            # Ensure format and codec are strings for comparison
            format_str = format.value if isinstance(format, RGBAVideoContainer) else format
            codec_str = codec.value if isinstance(codec, RGBAVideoCodec) else codec
            if format != RGBAVideoContainer.AUTO and format_str not in ['mp4', 'webm', 'mov']:
                raise ValueError("Supported formats: MP4, WEBM, MOV")
            if codec != RGBAVideoCodec.AUTO and codec_str not in ['h264', 'libvpx-vp9', 'prores_ks']:
                raise ValueError("Supported codecs: H264, VP9, ProRes")

        # Set default codec if AUTO
        if codec == RGBAVideoCodec.AUTO:
            codec = RGBAVideoCodec.H264.value
            codec_str = codec

        # Ensure codec_str is defined
        if 'codec_str' not in locals():
            codec_str = codec.value if isinstance(codec, RGBAVideoCodec) else codec

        # Prepare options based on format
        options = {}
        if format_str in ['mp4', 'mov']:
            options['movflags'] = 'use_metadata_tags'

        # Determine the format string for av.open
        output_format = None if format == RGBAVideoContainer.AUTO else format_str

        with av.open(path, mode='w', format=output_format, options=options) as output:
            # Add metadata before writing any streams
            if metadata is not None:
                for key, value in metadata.items():
                    output.metadata[key] = json.dumps(value)

            frame_rate = Fraction(round(self.__components.frame_rate * 1000), 1000)

            # Create a video stream
            video_stream = output.add_stream(codec_str, rate=frame_rate)
            video_stream.width = self.__components.images.shape[2]
            video_stream.height = self.__components.images.shape[1]

            # Set pixel format based on codec and alpha channel
            if has_alpha:
                if codec_str in ['libvpx-vp9']:
                    video_stream.pix_fmt = 'yuva420p'  # VP9 with alpha
                elif codec_str in ['prores_ks']:
                    video_stream.pix_fmt = 'yuva444p10le'  # ProRes 4444
                else:
                    video_stream.pix_fmt = 'yuva420p'  # Default alpha format
            else:
                if codec_str in ['h264']:
                    video_stream.pix_fmt = 'yuv420p'
                elif codec_str in ['libvpx-vp9']:
                    video_stream.pix_fmt = 'yuv420p'
                else:
                    video_stream.pix_fmt = 'yuv420p'

            # Create an audio stream
            audio_sample_rate = 1
            audio_stream: Optional[av.AudioStream] = None
            if self.__components.audio:
                audio_sample_rate = int(self.__components.audio['sample_rate'])
                audio_codec = 'libopus' if format_str == 'webm' else 'aac'
                audio_stream = output.add_stream(audio_codec, rate=audio_sample_rate)

            # Encode video
            for i, frame in enumerate(self.__components.images):
                img = (frame * 255).clamp(0, 255).byte().cpu().numpy()

                if has_alpha:
                    # Create frame with alpha channel (H, W, 4)
                    frame = av.VideoFrame.from_ndarray(img, format='rgba')
                    frame = frame.reformat(format=video_stream.pix_fmt)
                else:
                    # Create frame without alpha channel (H, W, 3)
                    frame = av.VideoFrame.from_ndarray(img, format='rgb24')
                    frame = frame.reformat(format=video_stream.pix_fmt)

                packet = video_stream.encode(frame)
                output.mux(packet)

            # Flush video
            packet = video_stream.encode(None)
            output.mux(packet)

            if audio_stream and self.__components.audio:
                waveform = self.__components.audio['waveform']
                waveform = waveform[:, :,
                           :math.ceil((audio_sample_rate / frame_rate) * self.__components.images.shape[0])]
                frame = av.AudioFrame.from_ndarray(waveform.movedim(2, 1).reshape(1, -1).float().numpy(), format='flt',
                                                   layout='mono' if waveform.shape[1] == 1 else 'stereo')
                frame.sample_rate = audio_sample_rate
                frame.pts = 0
                output.mux(audio_stream.encode(frame))

                # Flush encoder
                output.mux(audio_stream.encode(None))