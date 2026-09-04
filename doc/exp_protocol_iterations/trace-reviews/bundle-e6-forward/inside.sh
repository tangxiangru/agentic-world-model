#!/bin/bash
set -eu
review_repo=/rmeng_data/robtang/exp-protocol-bundle-work-5iV6EzGB/repo
review_runtime=/tmp/exp-protocol-save-runtime.JEZlHo
review_temp=/tmp/e6-forward-sampling.ntRxXE
exec bwrap --ro-bind / / --unshare-net --dev /dev \
  --bind "$review_temp" "$review_temp" \
  --ro-bind "$review_runtime/rootfs/usr/bin/python3.10" /usr/bin/python3.10 \
  --ro-bind "$review_runtime/rootfs/usr/lib/python3.10" /usr/lib/python3.10 \
  --ro-bind "$review_runtime/rootfs/usr/lib/x86_64-linux-gnu" /usr/lib/x86_64-linux-gnu \
  --ro-bind "$review_runtime/rootfs/usr/local/lib/python3.10" /usr/local/lib/python3.10 \
  --setenv TMPDIR "$review_temp" --setenv CUDA_VISIBLE_DEVICES '' \
  --setenv HF_HUB_OFFLINE 1 --setenv TRANSFORMERS_OFFLINE 1 \
  --setenv PYTHONDONTWRITEBYTECODE 1 --setenv PYTHONNOUSERSITE 1 \
  --setenv PYTHONPATH "$review_repo" --chdir "$review_temp" /usr/bin/python3.10 "$@"
