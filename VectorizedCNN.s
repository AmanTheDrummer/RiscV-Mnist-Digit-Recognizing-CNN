#define STDOUT 0xd0580000

.section .text
.global _start
_start:
## START YOUR CODE HERE

# CNN Forward Pass in RISC-V Assembly using Vector Instructions (Fixed and Vectorized)

.include "./assembly/conv_filters.asm"
.include "./assembly/W1.asm"
.include "./assembly/B1.asm"
.include "./assembly/W2.asm"
.include "./assembly/B2.asm"
.include "./assembly/InputVector.asm"

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
    li x3,3
    div x12,x8,x3
    mul x13,x12,x3 
    sub x11,x8,x13
    li x3,3 
    div x12, x8, x3          # dy
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

    vle32.v v1, (x9)                  # input patch
    li x10, 36
    mul x21, x5, x10 
    slli x21, x21, 2
    add x22, x3, x21
    vle32.v v2, (x22)                # filter weights

    vfmul.vv v3, v1, v2
    vmv.v.x v0, x0
    vredsum.vs v4, v3, v0
    vmv.x.s x23, v4
    fmv.w.x f10, x23

    li x11, 36
    mul x24, x5, x11 
    li x12, 6
    mul x25, x6, x12

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
    vle32.v v1, (x2)
    vle32.v v2, (x3)
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
    vle32.v v1, (x2)
    vle32.v v2, (x3)
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
## END YOU CODE HERE

# Function: print
# Logs values from array in a0 into registers v1 for debugging and output.
# Inputs:
#   - a0: Base address of array
#   - a1: Size of array i.e. number of elements to log
# Clobbers: t0,t1, t2,t3 ft0, ft1.
printToLogVectorized:        
    addi sp, sp, -4
    sw a0, 0(sp)

    li t0, 0x123                 # Pattern for help in python script
    li t0, 0x456                 # Pattern for help in python script
    mv a1, a1                   # moving size to get it from log 
    mul a1, a1, a1              # sqaure matrix has n^2 elements 
	li t0, 0		                # load i = 0
    printloop:
        vsetvli t3, a1, e32           # Set VLEN based on a1
        slli t4, t3, 2                # Compute VLEN * 4 for address increment

        vle32.v v1, (a0)              # Load real[i] into v1
        add a0, a0, t4                # Increment pointer for real[] by VLEN * 4
        add t0, t0, t3                # Increment index

        bge t0, a1, endPrintLoop      # Exit loop if i >= size
        j printloop                   # Jump to start of loop
    endPrintLoop:
    li t0, 0x123                    # Pattern for help in python script
    li t0, 0x456                    # Pattern for help in python script
	
    lw a0, 0(sp)
    addi sp, sp, 4

	jr ra



# Function: _finish
# VeeR Related function which writes to to_host which stops the simulator
_finish:
    li x3, 0xd0580000
    addi x5, x0, 0xff
    sb x5, 0(x3)
    beq x0, x0, _finish

    .rept 100
        nop
    .endr


.data
## ALL DATA IS DEFINED HERE LIKE MATRIX, CONSTANTS ETC


## DATA DEFINE START
.equ MatrixSize, 5
matrix:
    .float -10.0, 13.0, 10.0, -3.0, 2.0
    .float 6.0, 15.0, 4.0, 13.0, 4.0
    .float 18.0, 2.0, 9.0, 8.0, -4.0
    .float 5.0, 4.0, 12.0, 17.0, 6.0
    .float -10.0, 7.0, 13.0, -3.0, 160.0
## DATA DEFINE END
size: .word MatrixSize
