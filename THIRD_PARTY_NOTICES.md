# Third-party sources

SplatAD-Drive uses the official [NeuRAD studio](https://github.com/georghess/neurad-studio)
and [SplatAD gsplat fork](https://github.com/carlinds/splatad), both under Apache-2.0.
The checked out repositories retain their original LICENSE files and copyright notices.
Pinned revisions are recorded in `third_party/versions.json`; this project's Python
reference implementation and adapters are distinct from those upstream sources.

PandaSet was created by Hesai and Scale AI. The official project links to the
[georghess/pandaset dataset mirror](https://huggingface.co/datasets/georghess/pandaset),
which labels the dataset CC-BY-4.0. The downloader retains available license files
and records source revision, member checksums and extraction provenance. Dataset
copyright and license terms are independent of the source-code licenses.

Citation for the algorithm:

Georg Hess, Carl Lindström, Maryam Fatemi, Lennart Petersson, Lennart Svensson.
SplatAD: Real-Time Lidar and Camera Rendering with 3D Gaussian Splatting for
Autonomous Driving. CVPR 2025. https://arxiv.org/abs/2411.16816
