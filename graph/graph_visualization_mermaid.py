import tempfile
import subprocess
from PIL import Image
import matplotlib.pyplot as plt
import os


# """
# # Download and install nvm:
# curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.2/install.sh | bash
#
# # in lieu of restarting the shell
# \. "$HOME/.nvm/nvm.sh"
#
# # Download and install Node.js:
# nvm install 22
#
# # Verify the Node.js version:
# node -v # Should print "v22.14.0".
# nvm current # Should print "v22.14.0".
#
# # Verify npm version:
# npm -v # Should print "10.9.2".
#
# # install mmdc
# npm install -g @mermaid-js/mermaid-cli
#
# # install Google Chrome and add sandbox to ~/.bashrc
# export CHROME_DEVEL_SANDBOX=/opt/google/chrome/chrome-sandbox
#
# # generate mermaid graph
# mmdc -i example.mmd -o example.png
# """


def show_mermaid_graph(mermaid_str: str, output_format: str = "png",
                       output_path: str = None, display: bool = True,
                       scale=5):
    assert output_format in {"png", "svg", "pdf"}, "only png, svg and pdf are supported."

    with tempfile.TemporaryDirectory() as tmpdir:
        mmd_path = os.path.join(tmpdir, "graph.mmd")
        if output_path is not None:
            out_path = output_path
        else:
            out_path = os.path.join(tmpdir, f"graph.{output_format}")

        with open(mmd_path, "w", encoding="utf-8") as f:
            f.write(mermaid_str)

        # 获取当前环境变量，并添加 node/mmdc 所在路径
        # witch mmdc
        env = os.environ.copy()
        env["PATH"] = "/home/alokia/.nvm/versions/node/v22.14.0/bin:" + env["PATH"]

        try:
            subprocess.run([
                "mmdc",
                "-i", mmd_path,
                "-o", out_path,
                "-s", str(scale)
            ], check=True, env=env)
        except subprocess.CalledProcessError as e:
            print("Mermaid CLI 执行失败，请确保 mmdc 已正确安装并可用。")
            raise e

        if display:
            if output_format == "png":
                img = Image.open(out_path)
                plt.figure(figsize=(10, 8))
                plt.imshow(img)
                plt.axis("off")
                plt.show()
            else:
                print(f"{output_format} file save in： {out_path}")
