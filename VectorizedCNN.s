# CNN Forward Pass in RISC-V Assembly using Vector Instructions (Fixed and Vectorized)

.include "conv_filters.asm"
.include "W1.asm"
.include "B1.asm"
.include "W2.asm"
.include "B2.asm"
.include "InputVector.asm"

.data
ConvOut: .space 1152     # 8 filters * 6x6 output * 4 bytes
ReLUOut: .space 1152
Z1:      .space 40       # 10 neurons * 4 bytes
Z2:      .space 40
zero_f:  .float 0.0

.text
.globl _start
_start:
    li x1, 9
    vsetvli x1, x1, e32, m1

    # === Convolution (Vectorized) ===
    la x2, InputVector
    la x3, conv_filters
    la x4, ConvOut

    li x5, 0                 # filter index
conv_filter_loop:
    li x6, 0                 # y
conv_y_loop:
    li x7, 0                 # x
conv_x_loop:
    # Gather patch into v1
    li x8, 0
    la x9, patch_buffer     # Temp patch store (assumed aligned)
patch_gather_loop:
    li x10, 8
    rem x11, x8, 3          # dx
    div x12, x8, 3          # dy
    add x13, x12, x6        # iy = dy + y
    add x14, x11, x7        # ix = dx + x
    li x15, 8               # image width
    mul x16, x13, x15       # iy * width
    add x16, x16, x14
    slli x16, x16, 2
    add x17, x2, x16
    flw f1, 0(x17)
    slli x18, x8, 2
    add x19, x9, x18
    fsw f1, 0(x19)
    addi x8, x8, 1
    li x20, 9
    blt x8, x20, patch_gather_loop

    vlw.v v1, (x9)                  # input patch
    mul x21, x5, 36
    slli x21, x21, 2
    add x22, x3, x21
    vlw.v v2, (x22)                # filter weights

    vfmul.vv v3, v1, v2
    vmv.v.x v0, x0
    vredsum.vs v4, v3, v0
    vmv.x.s x23, v4
    fmv.w.x f10, x23

    mul x24, x5, 36
    mul x25, x6, 6
    add x25, x25, x7
    add x24, x24, x25
    slli x24, x24, 2
    add x26, x4, x24
    fsw f10, 0(x26)

    addi x7, x7, 1
    li x27, 6
    blt x7, x27, conv_x_loop
    addi x6, x6, 1
    li x28, 6
    blt x6, x28, conv_y_loop
    addi x5, x5, 1
    li x29, 8
    blt x5, x29, conv_filter_loop

    # === ReLU ===
    la x2, ConvOut
    la x3, ReLUOut
    la x4, zero_f
    flw f0, 0(x4)
    li x5, 0
relu_loop:
    slli x6, x5, 2
    add x7, x2, x6
    add x8, x3, x6
    flw f1, 0(x7)
    fmax.s f2, f1, f0
    fsw f2, 0(x8)
    addi x5, x5, 1
    li x9, 288              # 8 * 6 * 6
    blt x5, x9, relu_loop

    # === FC1: Z1 = W1 x ReLUOut + B1 ===
    la x2, W1
    la x3, ReLUOut
    la x4, B1
    la x5, Z1

    li x6, 0       # neuron index
fc1_loop:
    vlw.v v1, (x2)
    vlw.v v2, (x3)
    vfmul.vv v3, v1, v2
    vmv.v.x v0, x0
    vredsum.vs v4, v3, v0
    vmv.x.s x7, v4
    fmv.w.x f1, x7

    slli x8, x6, 2
    add x9, x4, x8
    flw f2, 0(x9)
    fadd.s f3, f1, f2
    add x10, x5, x8
    fsw f3, 0(x10)

    addi x2, x2, 1152     # next W1 row
    addi x6, x6, 1
    li x11, 10
    blt x6, x11, fc1_loop

    # === FC2: Z2 = W2 x Z1 + B2 ===
    la x2, W2
    la x3, Z1
    la x4, B2
    la x5, Z2
    li x6, 0
fc2_loop:
    vlw.v v1, (x2)
    vlw.v v2, (x3)
    vfmul.vv v3, v1, v2
    vmv.v.x v0, x0
    vredsum.vs v4, v3, v0
    vmv.x.s x7, v4
    fmv.w.x f1, x7

    slli x8, x6, 2
    add x9, x4, x8
    flw f2, 0(x9)
    fadd.s f3, f1, f2

    add x10, x5, x8
    fsw f3, 0(x10)

    addi x2, x2, 40
    addi x6, x6, 1
    li x11, 10
    blt x6, x11, fc2_loop

    # === Prediction (argmax Z2) ===
    la x1, Z2
    li x2, 10
    li x3, 0
    li x4, 0
    flw f0, 0(x1)
argmax_loop:
    beq x3, x2, end
    flw f1, 0(x1)
    fgt.s x5, f1, f0
    beqz x5, skip
    fmv.s f0, f1
    mv x4, x3
skip:
    addi x1, x1, 4
    addi x3, x3, 1
    j argmax_loop

end:
    # Prediction in x4
    nop

.data
.align 4
patch_buffer: .space 36       # 9 floats for 3x3 input patch
