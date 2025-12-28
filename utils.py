import base64
from pathlib import Path

from gymnasium.wrappers import RecordVideo
from IPython import display as ipythondisplay

#CODE INSPIRED FROM https://github.com/Farama-Foundation/HighwayEnv/blob/master/scripts/utils.py


def record_videos(env, video_folder="videos"):

    env =  RecordVideo(env, video_folder=video_folder, episode_trigger=lambda e: True)
    env.unwrapped.set_record_video_wrapper(env)
    return env


def show_videos(path="videos"):
    html = []
    for mp4 in Path(path).glob("*.mp4"):
        video_b64 = base64.b64encode(mp4.read_bytes())
        html.append(
            """<video alt="{}" autoplay
                      loop controls style="height: 400px;">
                      <source src="data:video/mp4;base64,{}" type="video/mp4" />
                 </video>""".format(
                mp4, video_b64.decode("ascii")
            )
        )
    ipythondisplay.display(ipythondisplay.HTML(data="<br>".join(html)))
