#!/usr/bin/env python
"""wordcount.py"""

import sys
import time

start = time.time()

counting = {}

for line in sys.stdin:
    line = line.strip()
    words = line.split()
    for word in words:
        if counting.get(word):
            counting[word] += 1
        else:
            counting[word] = 1

for word, count in sorted(counting.items()):
    print '%s\t%s' % (word, count)

print(end - time.time())