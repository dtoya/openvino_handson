#!/bin/bash

export HANDSON_ROOT=$PWD

scripts/install-package.sh
scripts/setup-igpu.sh
scripts/install-devtools.sh
scripts/setup-omz-demo.sh
scripts/download-omz-models.sh
scripts/download-media.sh
