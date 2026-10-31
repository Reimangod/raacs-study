"""Pure stdlib native DICE alpha/beta binary reader; no dense Cartesian array.

Layout follows the pinned Qiskit addon Rust reader (Apache-2.0). DICE already
converts interleaved fermionic signs before writing. Qualified against all7
actual tiny outputs; this parser does not modify coefficients or normalize.
"""
import math,struct


def decode(data):
    if len(data)<8:raise ValueError('Truncated native determinant header')
    count,n=struct.unpack_from('<II',data)
    if not 1<=n<=64 or count<1 or len(data)!=8+count*(8+n):raise ValueError('Native output length/header mismatch')
    pairs=[];seen=set()
    for k in range(count):
        start=8+k*(8+n);value=struct.unpack_from('<d',data,start)[0];a=b=0
        if not math.isfinite(value):raise ValueError('Nonfinite native CI amplitude')
        for i,char in enumerate(data[start+8:start+8+n]):
            if char not in b'02ab':raise ValueError('Invalid native occupancy byte')
            if char in b'2a':a|=1<<i
            if char in b'2b':b|=1<<i
        if (a,b) in seen:raise ValueError('Duplicate native determinant pair')
        seen.add((a,b));pairs.append([a,b,value])
    return n,pairs
