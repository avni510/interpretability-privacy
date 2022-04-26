#!/usr/bin/env python

import models_runner as models_runner

try:
    models_runner.execute(True)
except Exception as e:
    print(e)
