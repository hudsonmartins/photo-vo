#! /bin/bash
if [ ! -d "weights" ]; then
  mkdir weights
fi
wget https://github.com/cvg/LightGlue/releases/download/v0.1_arxiv/superpoint_lightglue.pth -P weights/