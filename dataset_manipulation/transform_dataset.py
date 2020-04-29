from moviepy.editor import VideoFileClip, vfx
import moviepy
import os
from pathlib import Path
from os import listdir
from os.path import isfile, join
from datetime import datetime
import time

# NOTE: original directory has the following name type:
#            -> sitt_over_open_left
#            -> stnd_undr_clos_rigt
#            -> sitt_over_null_left

original_path_name = "sitt_over_open_left"

def create_dir(path):
    if os.path.exists(path):
        print("Directory %s exists" % path)
        if os.listdir(path) == []:
            return True
        else:
            print("Directory %s is not empty" % path)
            return False

    try:
        os.mkdir(path)
    except OSError:
        print("Creation of the directory %s failed" % path)
        return False
    else:
        print("Successfully created the directory %s " % path)
        return True

def reverse(name):
    if name == "open":
        return "clos"
    elif name == "clos":
        return 'open'
    elif name == "null":
        return "null"
    else:
        return "FAIL"

def mirror(name):
    if name == "left":
        return "rigt"
    elif name == "rigt":
        return "left"
    else:
        return "FAIL"

def main():
    name1 = original_path_name[0:4]
    name2 = original_path_name[5:9]
    name3 = original_path_name[10:14]
    name4 = original_path_name[15:19]

    reverse_path_name         = "MOD_" + name1 + "_" + name2 + "_" + reverse(name3) + "_" + name4
    mirror_path_name          = "MOD_" + name1 + "_" + name2 + "_" + name3 + "_" + mirror(name4)
    mirror_reverse_path_name  = "MOD_" + name1 + "_" + name2 + "_" + reverse(name3) + "_" + mirror(name4)

    original_path       = Path("./" + original_path_name + "/")
    reverse_path        = Path("./" + reverse_path_name + "/")
    mirror_path         = Path("./" + mirror_path_name + "/")
    mirror_reverse_path = Path("./" + mirror_reverse_path_name + "/")

    if not create_dir(reverse_path):
        return 1
    if not create_dir(mirror_path):
        return 1
    if not create_dir(mirror_reverse_path):
        return 1


    original_videos_name = [f for f in listdir(original_path) if isfile(join(original_path, f))]

    ugly_new_name = str(int(time.time())) + "_"
    for i, file_name in enumerate(original_videos_name):
        original_file = original_path / file_name
        clip = VideoFileClip(str(original_file))

        reverse_file = reverse_path / str("reverse_" + ugly_new_name + str(i) + ".mp4")
        reversed_clip = clip.fx(vfx.time_mirror)
        reversed_clip.write_videofile(str(reverse_file))

        mirror_file = mirror_path / str("mirror_" + ugly_new_name + str(i) + ".mp4")
        mirror_clip = clip.fx(vfx.mirror_x)
        mirror_clip.write_videofile(str(mirror_file))

        mirror_reverse_file = mirror_reverse_path / str("reverse_mirror_" + ugly_new_name + str(i) + ".mp4")
        mirror_reverse_clip = reversed_clip.fx(vfx.mirror_x)
        mirror_reverse_clip.write_videofile(str(mirror_reverse_file))


main()
