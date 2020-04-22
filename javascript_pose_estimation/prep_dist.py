import os
import subprocess

if __name__ == "__main__":
    p = subprocess.Popen(["yarn", "build"])
    p.wait()
    with open("dist/index.html", "r+") as fp:
        index_file = fp.read()
        index_file = index_file.replace("socket_client", "dist/socket_client")
        index_file = index_file.replace("camera", "dist/camera")

        print(index_file)
        fp.seek(0)

        fp.write(index_file)

    