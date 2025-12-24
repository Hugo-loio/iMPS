#!/bin/sh

pkg=$(dirname "$0")

pip install --user --break-system-packages -e $pkg || pip install --user -e $pkg
