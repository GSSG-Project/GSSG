#  Copyright 2021 The PlenOctree Authors.
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#
#  1. Redistributions of source code must retain the above copyright notice,
#  this list of conditions and the following disclaimer.
#
#  2. Redistributions in binary form must reproduce the above copyright notice,
#  this list of conditions and the following disclaimer in the documentation
#  and/or other materials provided with the distribution.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
#  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
#  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
#  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
#  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
#  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
#  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#  POSSIBILITY OF SUCH DAMAGE.


C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396,
]
C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435,
]
C4 = [
    2.5033429417967046,
    -1.7701307697799304,
    0.9461746957575601,
    -0.6690465435572892,
    0.10578554691520431,
    -0.6690465435572892,
    0.47308734787878004,
    -1.7701307697799304,
    0.6258357354491761,
]

SEMANTIC_COLOR_MAP = {
    1: [1.0, 0.0, 0.0],  # Red
    2: [0.0, 1.0, 0.0],  # Green
    3: [0.0, 0.0, 1.0],  # Blue
    4: [1.0, 1.0, 0.0],  # Yellow
    5: [1.0, 0.0, 1.0],  # Magenta
    6: [0.0, 1.0, 1.0],  # Cyan
    7: [1.0, 0.5, 0.0],  # Orange
    8: [0.5, 0.0, 1.0],  # Purple
    9: [0.5, 0.5, 0.5],  # Gray
    10: [0.75, 0.25, 0.25],  # Brownish Red
    11: [0.25, 0.75, 0.25],  # Light Green
    12: [0.25, 0.25, 0.75],  # Light Blue
    13: [1.0, 0.75, 0.25],  # Gold
    14: [0.75, 0.0, 0.75],  # Violet
    15: [0.0, 0.75, 0.75],  # Teal
    16: [0.5, 0.5, 0.0],  # Olive
    17: [0.5, 0.0, 0.5],  # Dark Purple
    18: [0.0, 0.5, 0.5],  # Dark Cyan
    19: [1.0, 0.5, 0.5],  # Pink
    20: [0.5, 1.0, 0.5],  # Mint Green
    21: [0.5, 0.5, 1.0],  # Light Periwinkle
    22: [1.0, 1.0, 0.5],  # Light Yellow
    23: [1.0, 0.5, 1.0],  # Light Magenta
    24: [0.5, 1.0, 1.0],  # Pale Cyan
    25: [0.25, 0.5, 0.0],  # Dark Olive Green
    26: [0.25, 0.0, 0.5],  # Indigo
    27: [0.0, 0.25, 0.5],  # Dark Blue
    28: [0.75, 0.5, 0.25],  # Tan
    29: [0.5, 0.25, 0.75],  # Medium Purple
    30: [0.25, 0.75, 0.5],  # Sea Green
    31: [0.75, 0.25, 0.5],  # Raspberry
    32: [0.5, 0.75, 0.25],  # Lime Green
    33: [0.25, 0.5, 0.75],  # Steel Blue
    34: [0.75, 0.75, 0.25],  # Mustard
    35: [0.75, 0.25, 0.75],  # Orchid
    36: [0.25, 0.75, 0.75],  # Aquamarine
    37: [0.9, 0.6, 0.2],  # Amber
    38: [0.2, 0.6, 0.9],  # Sky Blue
    39: [0.6, 0.2, 0.9],  # Purple Pink
    40: [0.9, 0.2, 0.6],  # Hot Pink
    41: [0.2, 0.9, 0.6],  # Mint
    42: [0.6, 0.9, 0.2],  # Chartreuse
    43: [0.9, 0.9, 0.2],  # Bright Yellow
    44: [0.2, 0.9, 0.9],  # Cyan Light
    45: [0.9, 0.2, 0.2],  # Bright Red
    46: [0.2, 0.2, 0.9],  # Bright Blue
    47: [0.5, 0.3, 0.1],  # Coffee Brown
    48: [0.1, 0.5, 0.3],  # Forest Green
}


def RGB2SH(rgb):
    return (rgb - 0.5) / C0


def SH2RGB(sh):
    return sh * C0 + 0.5
